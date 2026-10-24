"""A forest of independent fixtures, merged only for a strict volume saving.

32 persistent samples start from the identical 13 Step3 membranes.  Each
proposal chooses two families uniformly and J/R with balanced scheduling.
No geometry or certificate of another family is touched by a merge.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import trimesh

from whole_task.step1.common import ROOT,DEFAULT_OUTPUT,sha256,write_json
from whole_task.operations.state import Layout,transform
from whole_task.operations.geometry import moved_mesh
from whole_task.step3.run_membranes import load_model
from whole_task.step3 import independent_geometry as G
from whole_task.step3.membranes import positive_components
from .membrane_guidance import MembraneObjective,repair
from .ground_oracle import GroundOracle,clip_core,project_families


def checkpoint(path,value):
    """Atomically publish the population journal after each completed trial."""
    path=Path(path);temporary=path.with_suffix('.pending.json')
    write_json(temporary,value);temporary.replace(path)


def improves_published(group_count,volume,current):
    """A new/resumed population cannot replace a better published fixture."""
    if current.get('method')!='independent_membrane_family_merges':return True
    if not current.get('ground_oracle_certified',False):return True
    count=current.get('group_count',float('inf'))
    return group_count<count or (group_count==count and volume<current['volume_cm3']-1e-5)


def saves_volume(after,left,right,tolerance=1e-5):
    """Compare actual material volumes, with no connecting-baseline subsidy."""
    return bool(np.isfinite([after,left,right]).all() and after>0 and after<left+right-tolerance)


def partition(groups,task_count):
    flat=[i for g in groups for i in g['members']]
    if sorted(flat)!=list(range(task_count)):raise ValueError('Families must partition the original tasks')


def complete_proof(proof,count):
    if not all(proof.get(k,False) for k in ('passed','full_loads','source_face_prisms_checked')):return False
    tasks=proof.get('tasks',[]);paths=proof.get('continuous_paths',[])
    if len(tasks)!=count or len(paths)!=count:return False
    if len({t['id'] for t in tasks})!=count or {t['id'] for t in tasks}!={p['id'] for p in paths}:return False
    if not all(p['passed'] for p in paths):return False
    for task in tasks:
        f=task.get('force',{})
        if (not f.get('passed',False) or f.get('preflight_only',False) or
            f.get('original_sample_count')!=32768 or f.get('passed_original_sample_count')!=32768 or
            f.get('continuous_outer_target_count',0)<=0 or
            f.get('continuous_outer_target_count')!=f.get('passed_continuous_outer_target_count')):return False
    return True


def read_group(group):
    path=Path(group['file'])
    if sha256(path)!=group['sha256']:raise ValueError('Changed group checkpoint')
    with np.load(path) as z:
        if not np.array_equal(z['members'],group['members']):raise ValueError('Group members differ from checkpoint')
        support=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
        if abs(support.volume*1e6-group['volume_cm3'])>1e-7:raise ValueError('Group volume differs from actual mesh')
        return Layout.read(z),support


def save_group(path,members,candidate,operation,parents):
    if not complete_proof(candidate['audit'],len(members)):raise ValueError('Incomplete family certificate')
    if parents and (len(parents)!=2 or not saves_volume(candidate['volume_cm3'],
            parents[0]['volume_cm3'],parents[1]['volume_cm3'])):
        raise ValueError('Merged fixture does not save volume against its two parents')
    np.savez_compressed(path,**candidate['layout'].arrays(),vertices=candidate['support'].vertices,
        faces=candidate['support'].faces,members=np.asarray(members))
    result=dict(file=str(path.resolve()),sha256=sha256(path),members=list(members),
        volume_cm3=candidate['volume_cm3'],operation=operation,parents=parents,
        audit=candidate['audit'])
    if candidate.get('oracle_required',False):
        if not candidate.get('oracle',{}).get('passed',False):raise ValueError('Missing ground-foot existence oracle')
        witness=path.with_name(path.stem+'_witness.npz');mesh=candidate['oracle_witness_mesh']
        np.savez_compressed(witness,vertices=mesh.vertices,faces=mesh.faces)
        result['ground_oracle']=candidate['oracle']
        result['oracle_witness']=dict(file=str(witness.resolve()),sha256=sha256(witness))
    write_json(path.with_suffix('.json'),result)
    group={k:result[k] for k in ('file','sha256','members','volume_cm3')}
    if 'oracle_witness' in result:group['oracle_witness']=result['oracle_witness']
    return group


def joined_layout(left,right,rng,kind,model):
    a,_=read_group(left);b,_=read_group(right)
    na=len(a.poses);ia=int(rng.integers(na));ib=int(rng.integers(len(b.poses)))
    q=Layout(np.concatenate([a.poses,b.poses]),np.concatenate([a.material,b.material]),
        np.concatenate([a.exits,b.exits]))
    qa=a.material[ia]@a.poses[ia];qb=b.material[ib]@b.poses[ib]
    host=transform(model.source.center_mass,qa);guest=transform(model.source.center_mass,qb)
    C=np.eye(4)
    if kind=='rotational_reuse':
        C=qa@np.linalg.inv(qb)
    else:C[:3,3]=host-guest
    q.material[na:]=C@q.material[na:]
    return q,na,ia,na+ib,qa


def seed_layouts(q,na,host,rng,kind,extent,left=None,right=None):
    """Try full-seat reuse and translations around it, including side contact."""
    result=[q]
    if kind=='rotational_reuse':
        for amount in (.008,.018,.035):
            d=rng.normal(size=3);d/=np.linalg.norm(d)
            trial=q.copy();trial.material[na:,:3,3]+=amount*d;result.append(trial)
    else:
        # Offset in the fixed material frame; no world floor restricts Z.
        # Large candidates let shells touch along side walls instead of
        # necessarily placing their entire object seats on top of each other.
        for amount in (.15,.35,.60,.85):
            d=rng.normal(size=3);d/=np.linalg.norm(d)
            trial=q.copy();trial.material[na:,:3,3]+=amount*extent*d;result.append(trial)
    boundary_start=len(result)
    if left is not None:
        qa,sa=read_group(left);qb,sb=read_group(right)
        sa=moved_mesh(sa,q.material[0]@np.linalg.inv(qa.material[0]))
        sb=moved_mesh(sb,q.material[na]@np.linalg.inv(qb.material[0]))
        directions=rng.normal(size=(3,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
        for d in directions:
            # Meet actual peripheral membrane points, without requiring two
            # large already-shared families to occupy the same entire seat.
            a=sa.vertices[np.argmax(sa.vertices@d)]
            b=sb.vertices[np.argmin(sb.vertices@d)]
            for overlap in (.0015,.0035):
                trial=q.copy();trial.material[na:,:3,3]+=a-b-overlap*d
                result.append(trial)
    return result,boundary_start


def parent_union_candidate(model,q,left,right,na):
    """Move the two actual existing fixtures; do not regrow unused skin."""
    qa,sa=read_group(left);qb,sb=read_group(right)
    sa=moved_mesh(sa,q.material[0]@np.linalg.inv(qa.material[0]))
    sb=moved_mesh(sb,q.material[na]@np.linalg.inv(qb.material[0]))
    tasks=model.prepare(q)
    stock=G.union([G.solid(sa),G.solid(sb)])
    # A pose repair may move the current layer relative to the old skin.
    # This candidate only removes material; the rebuilt-layer alternative
    # below can add useful skin if that is needed to restore a certificate.
    stock=stock^G.union([t['stock'] for t in tasks])
    forbidden=[t[k] for t in tasks for k in ('sweep','work')]
    value=stock-G.union(forbidden)
    parts=positive_components(value)
    if not parts:raise ValueError('No shared membrane survives')
    keep=max(parts,key=lambda p:float(p.volume()))
    for part in parts:
        if part is not keep:value=value-part
    support=G.mesh(value.simplify(0.))
    actual_forbidden=[p for t in tasks for p in (G.solid(t['body']),t['work'])]
    for _ in range(3):
        native=G.solid(support)
        if max(G.volume(native^f) for f in actual_forbidden)<=model.tolerance:break
        support=G.mesh((native-G.union(actual_forbidden)).simplify(0.))
    return dict(layout=q.copy(),support=support,tasks=tasks,volume_cm3=float(support.volume*1e6),
        construction=dict(model='union_of_two_actual_parent_fixtures_then_subtract',
            base_plate=False,vertical_backing=False,floor_constraints=False,regrow_unused_skin=False))


def proposal(args):
    source_folder,left,right,kind,seed,repair_rounds,destination=args
    began=time.monotonic();rng=np.random.default_rng(seed)
    base,_,_=load_model(source_folder);members=left['members']+right['members']
    model=base.subset(members);model.proposal_deadline=began+240.
    model.oracle_required=True
    olda,sa=read_group(left);oldb,sb=read_group(right)
    model.guidance_parent_parts=[(0,olda.material[0],sa),(len(olda.poses),oldb.material[0],sb)]
    q,na,ia,ib,host=joined_layout(left,right,rng,kind,model)
    model.oracle_partition=[list(range(na)),list(range(na,len(members)))]
    info=dict(operation=kind,members=members,parent_members=[left['members'],right['members']],
        parent_volumes_cm3=[left['volume_cm3'],right['volume_cm3']],seed=int(seed),trials=[])
    # Guidance cheaply ranks several transformations.  Actual CSG, actual
    # source contacts and every original load still decide acceptance.
    seeds,boundary_start=seed_layouts(q,na,host,rng,kind,float(model.source.extents.max()),left,right)
    projected=[];boundary_flags=[]
    for k,seed_layout in enumerate(seeds):
        fixed=project_families(model,seed_layout,model.oracle_partition)
        if fixed is not None:projected.append(fixed);boundary_flags.append(k>=boundary_start)
    seeds=projected
    if not seeds:return dict(info,status='rejected',seconds=time.monotonic()-began)
    q=seeds[0]
    objective=MembraneObjective(model,q,volume_power=12,envelope_layouts=seeds)
    ranked=[]
    for k,trial in enumerate(seeds):
        row=objective.evaluate(trial)
        if row is not None:
            ranked.append((not row['covered'],row['loss'],row['volume_cm3'],k,trial))
    ranked.sort(key=lambda x:x[:4])
    selected=ranked[:2]
    boundary=[row for row in ranked if boundary_flags[row[3]]]
    if boundary and not any(boundary_flags[row[3]] for row in selected):selected=ranked[:1]+boundary[:1]
    for _,_,_,index,trial in selected:
        try:
            fixed,diagnostic=repair(model,trial,np.arange(na,len(members)),[ia,ib],rounds=repair_rounds)
        except (RuntimeError,ValueError,AssertionError,np.linalg.LinAlgError) as ex:
            info['trials'].append(dict(seed_index=index,status='guidance_unresolved',reason=str(ex)));continue
        choices=[fixed]
        if any(not np.array_equal(fixed.arrays()[key],trial.arrays()[key]) for key in trial.arrays()):choices.append(trial)
        for choice in choices:
            for constructor in ('actual_parent_union','rebuild_thin_layers'):
                try:
                    candidate=(parent_union_candidate(model,choice,left,right,na)
                        if constructor=='actual_parent_union' else model.build(choice))
                    candidate=clip_core(model,candidate)
                    if not saves_volume(candidate['volume_cm3'],left['volume_cm3'],right['volume_cm3']):
                        info['trials'].append(dict(seed_index=index,constructor=constructor,
                            status='volume_rejected',volume_cm3=candidate['volume_cm3']));continue
                    # This preliminary force screen cannot be committed.
                    model.audit(candidate,continuous=False)
                    model.refine_exits(candidate)
                    model.audit(candidate)
                    witness,oracle=GroundOracle(model).query(candidate)
                    candidate.update(oracle_required=True,oracle=oracle,oracle_witness_mesh=witness)
                    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
                    info.update(status='accepted',gradient=diagnostic,volume_cm3=candidate['volume_cm3'],
                        constructor=constructor,boundary_seed=boundary_flags[index])
                    group=save_group(path,members,candidate,kind,[left,right])
                    info['group']=group;info['seconds']=time.monotonic()-began
                    write_json(path.with_name(path.stem+'_proposal.json'),info)
                    return info
                except (RuntimeError,ValueError,AssertionError,np.linalg.LinAlgError) as ex:
                    info['trials'].append(dict(seed_index=index,constructor=constructor,
                        status='physics_rejected',reason=str(ex)))
    info.update(status='rejected',seconds=time.monotonic()-began)
    return info


def initial_groups(source,model,q,workdir):
    groups=[]
    with np.load(source/'initialization.npz') as z:
        validation=json.loads((source/'validation.json').read_text())
        if validation['initialization_sha256']!=sha256(source/'initialization.npz'):raise ValueError('Changed Step3 membranes')
        for i,name in enumerate(model.ids):
            path=workdir/'initial'/f'{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            support=trimesh.Trimesh(z[name+'_vertices'],z[name+'_faces'],process=False)
            qi=Layout(q.poses[i:i+1],q.material[i:i+1],q.exits[i:i+1])
            candidate=dict(layout=qi,support=support,volume_cm3=float(support.volume*1e6),audit=validation['tasks'][i])
            part=model.subset([i]);candidate['tasks']=part.prepare(qi);candidate['construction']={}
            witness,oracle=GroundOracle(part).query(candidate)
            candidate.update(oracle_required=True,oracle=oracle,oracle_witness_mesh=witness)
            groups.append(save_group(path,[i],candidate,'step3_initialization',[]))
            print(name,'INITIAL GROUND ORACLE PASS',flush=True)
    return groups


def validate(source,groups,folder):
    """Fresh disk replay of each final fixture and all of its member tasks."""
    model,initial,report=load_model(source);partition(groups,len(model.ids));proofs=[];total=0.
    for path,digest in report['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Changed Step3 source input: '+path)
    upstream=json.loads((source/'validation.json').read_text())
    if upstream['initialization_sha256']!=sha256(source/'initialization.npz'):raise ValueError('Changed Step3 mesh data')
    history=[];visited=set()
    def ancestry(g):
        if g['sha256'] in visited:return
        q,support=read_group(g);meta=json.loads(Path(g['file']).with_suffix('.json').read_text())
        if meta['sha256']!=g['sha256'] or meta['members']!=g['members']:raise ValueError('Changed group metadata')
        proof=meta['audit'];names={model.ids[k] for k in g['members']}
        if not meta.get('ground_oracle',{}).get('passed',False):raise ValueError('Accepted node lacks ground-foot oracle')
        witness=meta.get('oracle_witness',{})
        if not witness or sha256(witness['file'])!=witness['sha256']:raise ValueError('Changed oracle witness')
        if (not proof['passed'] or not proof['full_loads'] or not proof['source_face_prisms_checked'] or
                {t['id'] for t in proof['tasks']}!=names or
                {t['id'] for t in proof['continuous_paths']}!=names or
                not all(t['passed'] for t in proof['continuous_paths'])):
            raise ValueError('Incomplete accepted-group proof')
        for task in proof['tasks']:
            f=task['force']
            if (not f['passed'] or f.get('preflight_only',False) or
                f['original_sample_count']!=32768 or f['passed_original_sample_count']!=32768 or
                f['continuous_outer_target_count']!=f['passed_continuous_outer_target_count'] or
                f['continuous_outer_target_count']<=0):raise ValueError('Incomplete original paired loads')
        if meta['operation']=='step3_initialization':
            if len(g['members'])!=1:raise ValueError('Initialization must contain singleton families')
            i=g['members'][0]
            for key,value in q.arrays().items():np.testing.assert_array_equal(value,initial.arrays()[key][i:i+1])
            with np.load(source/'initialization.npz') as z:
                np.testing.assert_array_equal(support.vertices,z[model.ids[i]+'_vertices'])
                np.testing.assert_array_equal(support.faces,z[model.ids[i]+'_faces'])
        else:
            parents=meta['parents']
            if len(parents)!=2 or set(parents[0]['members'])&set(parents[1]['members']):
                raise ValueError('A merge must unite two distinct existing families')
            if parents[0]['members']+parents[1]['members']!=g['members']:raise ValueError('Merged members differ')
            if not saves_volume(g['volume_cm3'],parents[0]['volume_cm3'],parents[1]['volume_cm3']):
                raise ValueError('A merge costs more than its separate input fixtures')
            for parent in parents:ancestry(parent)
        visited.add(g['sha256']);history.append(dict(sha256=g['sha256'],members=g['members'],
            operation=meta['operation'],volume_cm3=g['volume_cm3']))
    for group in groups:ancestry(group)
    for g in groups:
        q,support=read_group(g);part=model.subset(g['members'])
        candidate=dict(layout=q,support=support,tasks=part.prepare(q))
        outside=G.volume(G.solid(support)-G.union([t['stock'] for t in candidate['tasks']]))
        if outside>model.tolerance:raise ValueError('Fixture contains material outside its thin layers')
        proof=part.audit(candidate);proofs.append(proof);total+=float(support.volume*1e6)
        from .ground_oracle import pressure_rows,ground_enclosure
        foot=g['oracle_witness']
        with np.load(foot['file']) as z:witness=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
        checks=ground_enclosure(witness,pressure_rows(part,q,candidate['tasks']))
        if not all(r['passed'] for r in checks):raise ValueError('Saved witness fails actual ground pressure enclosure')
        if G.volume(G.solid(support)-G.solid(witness))>part.tolerance:raise ValueError('Saved witness does not contain membrane')
        witness_candidate=dict(layout=q,support=witness,tasks=part.prepare(q))
        witness_audit=part.audit(witness_candidate)
        proof['ground_oracle']=dict(passed=True,ground=checks,witness_audit=witness_audit,
            witness_sha256=foot['sha256'])
        print('DISK REPLAY PASS',[model.ids[k] for k in g['members']],flush=True)
    result=dict(status='passed',group_count=len(groups),total_volume_cm3=total,
        original_sample_count=sum(t['force']['original_sample_count'] for p in proofs for t in p['tasks']),
        continuous_outer_target_count=sum(t['force']['continuous_outer_target_count'] for p in proofs for t in p['tasks']),
        groups=groups,proofs=proofs,accepted_ancestry=history,
        strict_sum_volume_rule_verified_for_every_merge=True,
        ground_oracle_certified=True,
        initial_singletons_equal_saved_step3=True,fresh_disk_replay=True,
        source_initialization_sha256=sha256(source/'initialization.npz'))
    write_json(Path(folder)/'validation.json',result);return model,result


def freeze(workdir,destination,branch=None,*,publish=True,object_name='B',output_root=DEFAULT_OUTPUT):
    """Pin an immutable branch while the other samples continue searching."""
    from collections import Counter
    workdir=Path(workdir);destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    raw=(workdir/'population.json').read_bytes();state=json.loads(raw)
    if branch is None:
        branch=min(range(len(state['branches'])),key=lambda j:(len(state['branches'][j]['groups']),
            sum(g['volume_cm3'] for g in state['branches'][j]['groups'])))
    source_groups=state['branches'][branch]['groups'];copied={}
    def pin(g):
        if g['sha256'] in copied:return copied[g['sha256']]
        read_group(g)
        meta=json.loads(Path(g['file']).with_suffix('.json').read_text())
        parents=[pin(parent) for parent in meta['parents']]
        path=destination/(g['sha256']+'.npz')
        shutil.copy2(g['file'],path)
        result=dict(g,file=str(path))
        meta.update(file=str(path),parents=parents,source_file=g['file'])
        if 'oracle_witness' in meta:
            old=meta['oracle_witness'];witness=destination/(g['sha256']+'_witness.npz')
            shutil.copy2(old['file'],witness)
            meta['oracle_witness']=dict(file=str(witness),sha256=sha256(witness))
            result['oracle_witness']=meta['oracle_witness']
        write_json(path.with_suffix('.json'),meta);copied[g['sha256']]=result;return result
    groups=[pin(g) for g in source_groups]
    (destination/'population_snapshot.json').write_bytes(raw)
    source=ROOT/'whole_task/data'/object_name/'step3_membranes'
    model,validation=validate(source,groups,destination)
    upstream=json.loads((source/'report.json').read_text())
    report=dict(status='passed',method='independent_membrane_family_merges',best_branch=branch,
        ground_oracle_certified=True,
        initial_group_count=len(model.ids),final_group_count=len(groups),
        initial_volume_cm3=upstream['material_volume_cm3'],final_volume_cm3=validation['total_volume_cm3'],
        groups=groups,counts=dict(Counter(t['operation'] for t in state['trials'] if 'operation' in t)),
        initialization_sha256=sha256(source/'initialization.npz'),
        acceptance='V(AB) < V(A)+V(B); every merged member fully certified',
        validation_sha256=sha256(destination/'validation.json'),
        selection=dict(source_workdir=str(workdir.resolve()),branch=branch,
            source_population_snapshot_sha256=sha256(destination/'population_snapshot.json')),
        source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('membrane_guidance.py'),
            Path(__file__).with_name('sampled_guidance.py'),Path(__file__).with_name('policy.json'),
            ROOT/'whole_task/step3/membranes.py',ROOT/'whole_task/step3/run_membranes.py',
            Path(__file__).with_name('ground_oracle.py'),
            Path(__file__).with_name('render_membrane_families.py')]},global_minimum_claimed=False)
    write_json(destination/'report.json',report)
    from .render_membrane_families import render
    image=render(model,groups,report,destination/'overview.png')
    report['overview_sha256']=sha256(image);write_json(destination/'report.json',report)
    if publish:
        out=Path(output_root)/object_name/'step4';out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(image,out/'overview.png')
        write_json(ROOT/'whole_task/data'/object_name/'step4_active.json',dict(workdir=str(destination),
            method=report['method'],volume_cm3=report['final_volume_cm3'],group_count=len(groups),
            ground_oracle_certified=True,
            overview_sha256=sha256(out/'overview.png'),report_sha256=sha256(destination/'report.json')))
    print('FROZEN MEMBRANES PASS',len(groups),report['final_volume_cm3'],flush=True)
    return report


def run(output_root=DEFAULT_OUTPUT,object_name='B',workdir=None,rounds=8,samples=32,workers=4,
        repair_rounds=1,seed=42,resume=False,publish=True,target_max_groups=1):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    source=ROOT/'whole_task/data'/object_name/'step3_membranes'
    model,q,report=load_model(source)
    workdir=Path(workdir or ROOT/'whole_task/data'/object_name/'step4_ground_membrane_families').resolve()
    workdir.mkdir(parents=True,exist_ok=True)
    binding=dict(initialization_sha256=sha256(source/'initialization.npz'),samples=samples,seed=seed,
        ground_oracle_required=True)
    manifest=workdir/'population.json'
    if resume and manifest.exists():
        state=json.loads(manifest.read_text())
        if state['binding']!=binding:raise ValueError('Population initialization differs')
    else:
        if manifest.exists():raise ValueError('Use --resume for this existing membrane population')
        groups=initial_groups(source,model,q,workdir)
        state=dict(binding=binding,round=0,branches=[dict(groups=groups,accepted=0) for _ in range(samples)],
            counts=dict(juxtaposition=0,rotational_reuse=0),trials=[])
        checkpoint(manifest,state)
    rng=np.random.default_rng(seed+state['round']*100003)
    if 'rng_state' in state:rng.bit_generator.state=state['rng_state']
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for round_index in range(state['round'],state['round']+rounds):
            completed={t['branch'] for t in state['trials'] if t['round']==round_index}
            if 'pending_round' not in state:
                operations=['juxtaposition']*((samples+1)//2)+['rotational_reuse']*(samples//2)
                rng.shuffle(operations);entries=[]
                for j,branch in enumerate(state['branches']):
                    if len(branch['groups'])<2 or j in completed:continue
                    a,b=rng.choice(len(branch['groups']),2,replace=False)
                    left,right=branch['groups'][int(a)],branch['groups'][int(b)]
                    kind=operations[j];state['counts'][kind]+=1
                    destination=workdir/f'branch_{j:02}'/f'merge_round_{round_index:03}.npz'
                    args=(str(source),left,right,kind,int(rng.integers(2**31)),repair_rounds,str(destination))
                    entries.append(dict(branch=j,a=int(a),b=int(b),args=args))
                state['pending_round']=dict(round=round_index,entries=entries)
                state['rng_state']=rng.bit_generator.state;checkpoint(manifest,state)
            if state['pending_round']['round']!=round_index:raise ValueError('Inconsistent round journal')
            futures={}
            for entry in state['pending_round']['entries']:
                if entry['branch'] in completed:continue
                futures[pool.submit(proposal,entry['args'])]=(entry['branch'],entry['a'],entry['b'])
            for future in as_completed(futures):
                j,a,b=futures[future]
                try:info=future.result()
                except Exception as ex:info=dict(status='worker_failed',reason=str(ex))
                branch=state['branches'][j]
                if info['status']=='accepted':
                    branch['groups']=[g for k,g in enumerate(branch['groups']) if k not in (a,b)]+[info['group']]
                    branch['accepted']+=1;partition(branch['groups'],len(model.ids))
                state['trials'].append(dict(branch=j,round=round_index,**info))
                checkpoint(manifest,state)
                print('MERGE',round_index,j,info['status'],'groups',len(branch['groups']),
                    'volume',round(sum(g['volume_cm3'] for g in branch['groups']),3),flush=True)
            state['round']=round_index+1;state.pop('pending_round',None);checkpoint(manifest,state)
            if any(len(b['groups'])<=target_max_groups for b in state['branches']):break
    eligible=[j for j,b in enumerate(state['branches']) if len(b['groups'])<=target_max_groups]
    if not eligible:eligible=[j for j,b in enumerate(state['branches']) if len(b['groups'])<=2] or list(range(samples))
    best=min(eligible,key=lambda j:(sum(g['volume_cm3'] for g in state['branches'][j]['groups']),
                                         len(state['branches'][j]['groups'])))
    groups=state['branches'][best]['groups']
    replay_model,validation=validate(source,groups,workdir)
    result=dict(status='passed',method='independent_membrane_family_merges',initial_group_count=len(model.ids),
        ground_oracle_certified=True,
        final_group_count=len(groups),initial_volume_cm3=report['material_volume_cm3'],
        final_volume_cm3=validation['total_volume_cm3'],groups=groups,best_branch=best,
        counts=state['counts'],initialization_sha256=binding['initialization_sha256'],
        selection_rule='minimum volume at requested target group count, with two-group fallback; every merge strictly saves volume',
        acceptance='V(A union B) < V(A) + V(B); all merged members fully certified',
        validation_sha256=sha256(workdir/'validation.json'),global_minimum_claimed=False,
        source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('membrane_guidance.py'),
            Path(__file__).with_name('sampled_guidance.py'),Path(__file__).with_name('policy.json'),
            ROOT/'whole_task/step3/membranes.py',ROOT/'whole_task/step3/run_membranes.py',
            Path(__file__).with_name('ground_oracle.py'),
            Path(__file__).with_name('render_membrane_families.py')]})
    write_json(workdir/'report.json',result)
    active=ROOT/'whole_task/data'/object_name/'step4_active.json'
    if publish and active.exists():
        current=json.loads(active.read_text())
        if Path(current['workdir']).resolve()!=workdir and not improves_published(
                len(groups),result['final_volume_cm3'],current):
            publish=False;result['publication']='retained_better_or_equal_published_result'
            write_json(workdir/'report.json',result)
    if publish:
        from .render_membrane_families import render
        image=render(replay_model,groups,result,workdir/'overview.png')
        out=Path(output_root)/object_name/'step4';out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(image,out/'overview.png')
        result['overview_sha256']=sha256(out/'overview.png');write_json(workdir/'report.json',result)
        write_json(ROOT/'whole_task/data'/object_name/'step4_active.json',dict(workdir=str(workdir),
            method=result['method'],volume_cm3=result['final_volume_cm3'],group_count=len(groups),
            ground_oracle_certified=True,
            overview_sha256=sha256(out/'overview.png'),report_sha256=sha256(workdir/'report.json')))
    print('MEMBRANE STEP4 PASS',json.dumps({k:result[k] for k in ('final_group_count','final_volume_cm3','counts')}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rounds',type=int,default=8)
    p.add_argument('--samples',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--repair-rounds',type=int,default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workdir',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--no-publish',action='store_true')
    p.add_argument('--target-groups',type=int,choices=[1,2],default=1)
    a=p.parse_args();run(rounds=a.rounds,samples=a.samples,workers=a.workers,repair_rounds=a.repair_rounds,
        seed=a.seed,workdir=a.workdir,resume=a.resume,publish=not a.no_publish,target_max_groups=a.target_groups)
