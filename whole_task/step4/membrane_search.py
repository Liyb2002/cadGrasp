"""Certified fixture families, with at least 25% material saving per merge.

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
from . import fast_checks  # noqa: F401  equivalent exact exit checks, installed on import
from .membrane_guidance import MembraneObjective,repair
from .ground_oracle import GroundOracle,clip_core,project_families
from .family_bank import register,best_cover,explore_cover
from .placement_search import PlacementScreen,IncrementalUnion,cached_parent_data,merged_layout,score as placement_score

MERGE_VOLUME_RATIO=.75
GROUPING_POLICY='certified_family_bank_exact_cover_v1'
POLICY=json.loads(Path(__file__).with_name('policy.json').read_text())
PLACEMENT=POLICY.get('placement_search',{})
# Seeds ranked on the actual parent-union construction, including floor-feasible
# SO(3) guest placements, replace the sampled-guidance ranking of v1.
PROPOSAL_ALGORITHM='rigid_placement_merge_v3' if PLACEMENT else 'sampled_guidance_ranking_v1'
TRIAL_TIMEOUT_S=float(POLICY.get('trial_timeout_s',270.))


def checkpoint(path,value):
    """Atomically publish the population journal after each completed trial."""
    path=Path(path);temporary=path.with_suffix('.pending.json')
    write_json(temporary,value);temporary.replace(path)


def improves_published(group_count,volume,current,*,required_merge_ratio=None):
    """A new/resumed population cannot replace a better published fixture."""
    if current.get('method')!='independent_membrane_family_merges':return True
    if not current.get('ground_oracle_certified',False):return True
    if required_merge_ratio is not None and current.get('merge_volume_ratio_limit',1.)>required_merge_ratio:return True
    count=current.get('group_count',float('inf'))
    if group_count<=2 and count<=2:return volume<current['volume_cm3']-1e-5
    if group_count<=2<count:return True
    if count<=2<group_count:return False
    return group_count<count or (group_count==count and volume<current['volume_cm3']-1e-5)


def select_branch(branches):
    """Minimize material among acceptable one/two-set solutions, retaining both."""
    eligible=[j for j,b in enumerate(branches) if len(b['groups'])<=2]
    if eligible:return min(eligible,key=lambda j:(sum(g['volume_cm3'] for g in branches[j]['groups']),len(branches[j]['groups'])))
    return min(range(len(branches)),key=lambda j:(len(branches[j]['groups']),sum(g['volume_cm3'] for g in branches[j]['groups'])))


def saves_volume(after,left,right,tolerance=1e-5,ratio=MERGE_VOLUME_RATIO):
    """Compare actual material volumes, with no connecting-baseline subsidy."""
    if not np.isfinite([after,left,right,ratio]).all() or min(after,left,right,ratio)<=0 or ratio>1:return False
    # Legacy certificates used a strict decrease. New certificates record
    # their own 0.75 limit; no tolerance permits crossing that limit.
    return bool(after<left+right-tolerance if ratio==1. else after<=ratio*(left+right))


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


def complete_ground_proof(proof,count):
    if 'inherited_core_audit' in proof:
        # Search-time oracle: membrane load proof plus every witness exit prism.
        audit=proof['inherited_core_audit'];exits=proof.get('witness_exit_and_interference_audit',{})
        paths=exits.get('continuous_paths',[])
        if not exits.get('passed') or len(paths)!=count or not all(r.get('passed',False) for r in paths):return False
    else:audit=proof.get('full_witness_audit',{})
    if not proof.get('passed',False) or not complete_proof(audit,count):return False
    if POLICY.get('tangent_ground_attempts') and proof.get('tangent_ground_required',True):
        # Step5 grounds the membrane on tangent planes; v2 oracles prove it.
        tangent=proof.get('tangent_installation')
        if tangent is not None and not (tangent.get('passed') and tangent.get('minimum_margin_m',-1.)>=-1e-9):return False
    ground=proof.get('ground',[])
    if len(ground)!=count or {r['id'] for r in ground}!={t['id'] for t in audit['tasks']}:return False
    return all(r.get('passed',False) and r.get('actual_ground_triangle_count',0)>0 and
        r.get('minimum_world_z_m',-1.)>=-1e-9 and
        r.get('pressure_hull_gap_m') is not None and r['pressure_hull_gap_m']<=1e-8 for r in ground)


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
        raise ValueError('Merged fixture exceeds 75% of its two parent volumes')
    np.savez_compressed(path,**candidate['layout'].arrays(),vertices=candidate['support'].vertices,
        faces=candidate['support'].faces,members=np.asarray(members))
    result=dict(file=str(path.resolve()),sha256=sha256(path),members=list(members),
        volume_cm3=candidate['volume_cm3'],operation=operation,parents=parents,
        audit=candidate['audit'],merge_volume_ratio_limit=MERGE_VOLUME_RATIO)
    if candidate.get('oracle_required',False):
        if not complete_ground_proof(candidate.get('oracle',{}),len(members)):
            raise ValueError('Missing full ground-foot existence oracle')
        if POLICY.get('tangent_ground_attempts') and 'tangent_installation' not in candidate['oracle']:
            raise ValueError('Missing tangent-ground installation proof')
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


def seed_layouts(q,na,host,rng,kind,extent,left=None,right=None,center=None):
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
    if left is not None:
        qa,sa=read_group(left);qb,sb=read_group(right)
        # A shared family has several reusable seats. Sampling just one
        # representative can miss a side window that is already present.
        # Rank up to eight different seat alignments in the same proposal.
        center=np.zeros(3) if center is None else center
        pairs=[(a,b) for a in range(len(qa.poses)) for b in range(len(qb.poses))]
        rng.shuffle(pairs)
        if len(pairs)>1:
            for a,b in pairs[:8]:
                host_seat=qa.material[a]@qa.poses[a];guest_seat=qb.material[b]@qb.poses[b]
                C=np.eye(4)
                if kind=='rotational_reuse':C=host_seat@np.linalg.inv(guest_seat)
                else:C[:3,3]=transform(center,host_seat)-transform(center,guest_seat)
                trial=q.copy();trial.material[na:]=C@qb.material;result.append(trial)
        boundary_start=len(result)
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
    else:boundary_start=len(result)
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


def guidance_selection(model,seeds,count):
    """v1 ranking: sampled contact/volume guidance of projected seeds."""
    if not seeds:return []
    objective=MembraneObjective(model,seeds[0],volume_power=12,envelope_layouts=seeds);ranked=[]
    for k,trial in enumerate(seeds):
        row=objective.evaluate(trial)
        if row is not None:ranked.append((not row['covered'],row['loss'],row['volume_cm3'],k,trial))
    ranked.sort(key=lambda x:x[:4])
    return [(k,trial,False) for *_,k,trial in ranked[:(4 if count>=5 else 2)]]


def screened_selection(model,base,left,right,kind,rng,seeds,na,began,info):
    """Rank seeds and new floor-feasible placements on the actual construction.

    The sampled guidance never models the largest-component selection, so a
    covered seed may lose a whole parent. Here every candidate is the actual
    parent union, cutouts and floor clip, with its exact volume and a force
    screen on parent contact rays still inside it. Acceptance is unchanged.
    """
    limit=MERGE_VOLUME_RATIO*(left['volume_cm3']+right['volume_cm3'])
    host=cached_parent_data(base,left,read_group);guest=cached_parent_data(base,right,read_group)
    screen=PlacementScreen(model,host,guest,IncrementalUnion(model,host,guest),limit=limit)
    # Counted samples decide the search; the wall-clock bound is only a safeguard.
    budget=began+.5*(TRIAL_TIMEOUT_S-30.)
    options={k:PLACEMENT[k] for k in ('rotations','directions','centers','refine_starts','iterations','offspring',
        'enough') if k in PLACEMENT}
    rows=screen.propose(rng,kind,seeds,deadline=budget,**options)
    connected=[r for r in rows if r.get('ok')]
    count=PLACEMENT.get('certify_large' if len(model.ids)>=5 else 'certify_small',2)
    chosen=[r for r in connected if not placement_score(r,limit)[1]][:count]
    info['placement_screen']=dict(algorithm=PROPOSAL_ALGORITHM,evaluated=len(rows),connected=len(connected),
        within_gate=sum(1 for r in connected if r['volume_cm3']<=limit),
        screened_force_pass_within_gate=sum(1 for r in connected if r['volume_cm3']<=limit and all(r['proxy'])),
        sources=sorted({r.get('source','') for r in rows}),
        chosen=[dict(source=r.get('source'),volume_cm3=r['volume_cm3'],screened_force_failures=len(r['proxy'])-sum(r['proxy']))
                for r in chosen],limit_cm3=limit,seconds=time.monotonic()-began)
    host=dict(layout=read_group(left)[0]);guest=dict(layout=read_group(right)[0])
    return [(k,merged_layout(host,guest,r['C']),bool(all(r['proxy']))) for k,r in enumerate(chosen)]


def proposal(args):
    source_folder,left,right,kind,seed,repair_rounds,destination=args
    began=time.monotonic();rng=np.random.default_rng(seed)
    base,_,_=load_model(source_folder);members=left['members']+right['members']
    model=base.subset(members);model.proposal_deadline=began+TRIAL_TIMEOUT_S-30.
    model.oracle_required=True
    olda,sa=read_group(left);oldb,sb=read_group(right)
    model.guidance_parent_parts=[(0,olda.material[0],sa),(len(olda.poses),oldb.material[0],sb)]
    q,na,ia,ib,host=joined_layout(left,right,rng,kind,model)
    model.oracle_partition=[list(range(na)),list(range(na,len(members)))]
    info=dict(operation=kind,members=members,parent_members=[left['members'],right['members']],
        parent_volumes_cm3=[left['volume_cm3'],right['volume_cm3']],seed=int(seed),trials=[],
        parent_sha256=[left['sha256'],right['sha256']],proposal_source_sha256=sha256(Path(__file__)),
        merge_volume_ratio_limit=MERGE_VOLUME_RATIO)
    # Guidance cheaply ranks several transformations.  Actual CSG, actual
    # source contacts and every original load still decide acceptance.
    seeds,boundary_start=seed_layouts(q,na,host,rng,kind,float(model.source.extents.max()),left,right,
        model.source.center_mass)
    projected=[];boundary_flags=[];seen_seeds=set()
    for k,seed_layout in enumerate(seeds):
        fixed=project_families(model,seed_layout,model.oracle_partition)
        if fixed is not None:
            key=tuple(np.round(a,12).tobytes() for a in fixed.arrays().values())
            if key in seen_seeds:continue
            seen_seeds.add(key);projected.append(fixed);boundary_flags.append(k>=boundary_start)
    seeds=projected
    info['distinct_projected_seeds']=len(seeds)
    if PLACEMENT:
        selected=screened_selection(model,base,left,right,kind,rng,seeds,na,began,info)
    else:
        selected=guidance_selection(model,seeds,len(members))
    if not selected:return dict(info,status='rejected',seconds=time.monotonic()-began)
    # Peripheral contact remains a proposal, but must not displace a better
    # overlapping seat merely because it connects two separate chunks.
    for index,trial,screened_pass in selected:
        raw=(trial,dict(status='raw_seed',blocks=[]))
        if screened_pass:
            # The actual construction already retains enough parent contact
            # rays; gradient repair is kept for an actual force counterexample.
            choices=[raw]
        else:
            try:
                fixed,diagnostic=repair(model,trial,np.arange(na,len(members)),[ia,ib],rounds=repair_rounds)
            except (RuntimeError,ValueError,AssertionError,np.linalg.LinAlgError) as ex:
                info['trials'].append(dict(seed_index=index,status='guidance_unresolved',reason=str(ex)));continue
            choices=[(fixed,diagnostic)]
            if any(not np.array_equal(fixed.arrays()[key],trial.arrays()[key]) for key in trial.arrays()):
                choices.append(raw)
        feedback_used=False
        for choice,choice_diagnostic in choices:
            force_failure=False
            for constructor in ('actual_parent_union','rebuild_thin_layers'):
                try:
                    candidate=(parent_union_candidate(model,choice,left,right,na)
                        if constructor=='actual_parent_union' else model.build(choice))
                    candidate=clip_core(model,candidate)
                    if not saves_volume(candidate['volume_cm3'],left['volume_cm3'],right['volume_cm3']):
                        info['trials'].append(dict(seed_index=index,constructor=constructor,
                            status='volume_rejected',volume_cm3=candidate['volume_cm3']));continue
                    # Exits are cheap with the exact prism broad phase; the
                    # single full audit still preflights 32 outer demands first.
                    model.refine_exits(candidate)
                    model.audit(candidate)
                    witness,oracle=GroundOracle(model).query(candidate)
                    candidate.update(oracle_required=True,oracle=oracle,oracle_witness_mesh=witness)
                    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
                    info.update(status='accepted',gradient=choice_diagnostic,volume_cm3=candidate['volume_cm3'],
                        constructor=constructor,seed_index=int(index))
                    group=save_group(path,members,candidate,kind,[left,right])
                    info['group']=group;info['seconds']=time.monotonic()-began
                    write_json(path.with_name(path.stem+'_proposal.json'),info)
                    return info
                except (RuntimeError,ValueError,AssertionError,np.linalg.LinAlgError) as ex:
                    force_failure|='paired force/moment certificate failed' in str(ex)
                    info['trials'].append(dict(seed_index=index,constructor=constructor,
                        status='physics_rejected',reason=str(ex)))
            if force_failure and not feedback_used and repair_rounds>0:
                # audit() records the actual failed demand, which the next
                # sampled objective must include. Repair this layout again
                # instead of discarding that concrete counterexample.
                feedback_used=True
                try:
                    retry,record=repair(model,choice,np.arange(na,len(members)),[ia,ib],
                        rounds=max(2,repair_rounds),rebuild=True)
                    if any(not np.array_equal(retry.arrays()[k],choice.arrays()[k]) for k in choice.arrays()):
                        record['after_actual_force_failure']=True;choices.append((retry,record))
                except (RuntimeError,ValueError,AssertionError,np.linalg.LinAlgError) as ex:
                    info['trials'].append(dict(seed_index=index,status='feedback_repair_unresolved',reason=str(ex)))
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


def schedule_entry(state,branch,round_index,source,workdir,repair_rounds,seed,samples):
    """Per-round balanced operations, with independent reproducible branch RNGs."""
    operations=['juxtaposition']*((samples+1)//2)+['rotational_reuse']*(samples//2)
    np.random.default_rng(np.random.SeedSequence([seed,round_index,71483])).shuffle(operations)
    rng=np.random.default_rng(np.random.SeedSequence([seed,round_index,branch,28103]))
    groups=state['branches'][branch]['groups'];a,b=map(int,rng.choice(len(groups),2,replace=False))
    singles=[k for k,g in enumerate(groups) if len(g['members'])==1]
    largest=max(range(len(groups)),key=lambda k:len(groups[k]['members']))
    if state.get('family_bank') and singles and len(groups[largest]['members'])>=3 and rng.random()<.65:
        a=largest;b=int(rng.choice(singles))
    kind=operations[branch]
    args=(str(source),groups[a],groups[b],kind,int(rng.integers(2**31)),repair_rounds,
        str(workdir/f'branch_{branch:02}'/f'merge_round_{round_index:03}.npz'))
    state['counts'][kind]+=1
    return dict(branch=branch,a=a,b=b,round=round_index,args=args)


def advance_population(state,source,workdir,rounds,samples,workers,repair_rounds,seed,target_max_groups):
    """One outstanding job per sample; a slow job never holds a round barrier."""
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    import multiprocessing
    from .membrane_worker import isolated_proposal
    manifest=workdir/'population.json';end_round=state['round']+rounds
    if 'branch_rounds' not in state:
        state['branch_rounds']=[max([t['round']+1 for t in state['trials'] if t['branch']==j]
            or [state['round']]) for j in range(samples)]
    if 'pending_jobs' not in state:
        state['pending_jobs']={}
        if 'pending_round' in state:
            old=state.pop('pending_round');done={t['branch'] for t in state['trials'] if t['round']==old['round']}
            for entry in old['entries']:
                if entry['branch'] not in done:
                    state['pending_jobs'][str(entry['branch'])]=dict(entry,round=old['round'])
    state['scheduling']='asynchronous_sample_progression_with_balanced_per_round_operations'
    task_count=sum(len(g['members']) for g in state['branches'][0]['groups'])
    for branch in state['branches']:
        for group in branch['groups']:register(state,group)
    checkpoint(manifest,state)
    found=(len(state.get('best_cover',range(task_count)))<=target_max_groups or
        any(len(b['groups'])<=target_max_groups for b in state['branches']))
    futures={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        def enqueue(j):
            if len(state['branches'][j]['groups'])<2:return
            key=str(j);entry=state['pending_jobs'].get(key)
            if entry is None:
                r=state['branch_rounds'][j]
                if found or r>=end_round:return
                entry=schedule_entry(state,j,r,source,workdir,repair_rounds,seed,samples)
                state['pending_jobs'][key]=entry
                checkpoint(manifest,state)
            futures[pool.submit(isolated_proposal,entry['args'])]=entry
        for j in range(samples):enqueue(j)
        while futures:
            done,_=wait(futures,return_when=FIRST_COMPLETED)
            for future in done:
                entry=futures.pop(future);j=entry['branch'];r=entry['round'];branch=state['branches'][j]
                try:info=future.result()
                except Exception as error:info=dict(status='worker_failed',reason=str(error),operation=entry['args'][3])
                if info['status']=='accepted':
                    # Parents may only be replaced by this sample's sole
                    # outstanding proposal. No other branch can mutate them.
                    a,b=entry['a'],entry['b']
                    if (branch['groups'][a]['sha256']!=entry['args'][1]['sha256'] or
                        branch['groups'][b]['sha256']!=entry['args'][2]['sha256']):raise ValueError('Outstanding proposal parents changed')
                    branch['groups']=[g for k,g in enumerate(branch['groups']) if k not in (a,b)]+[info['group']]
                    branch['accepted']+=1;partition(branch['groups'],sum(len(g['members']) for g in branch['groups']))
                    register(state,info['group']);branch['stalled']=0
                else:branch['stalled']=branch.get('stalled',0)+1
                state['trials'].append(dict(branch=j,round=r,**info))
                state['pending_jobs'].pop(str(j));state['branch_rounds'][j]=r+1
                state['round']=min(state['branch_rounds'])
                state['best_cover']=best_cover(state,task_count)
                # The completed proposal is committed before regrouping.
                # Outstanding proposals on other samples keep their parents.
                rng=np.random.default_rng(np.random.SeedSequence([seed,r,j,96191]))
                if branch['stalled']>=2 or (info['status']=='accepted' and rng.random()<.35):
                    branch['groups']=explore_cover(state,task_count,rng)
                    partition(branch['groups'],task_count)
                    branch['stalled']=0
                    branch['regroupings']=branch.get('regroupings',0)+1
                checkpoint(manifest,state)
                print('MERGE',r,j,info['status'],'groups',len(branch['groups']),
                    'volume',round(sum(g['volume_cm3'] for g in branch['groups']),3),flush=True)
                found|=len(state['best_cover'])<=target_max_groups
                enqueue(j)
    return state


def validate(source,groups,folder,*,max_merge_ratio=None):
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
        if not complete_ground_proof(meta.get('ground_oracle',{}),len(g['members'])):
            raise ValueError('Accepted node lacks full ground-foot oracle')
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
            ratio=meta.get('merge_volume_ratio_limit',1.)
            if max_merge_ratio is not None and ratio>max_merge_ratio:
                raise ValueError('Ancestry uses an older, weaker merge-volume limit')
            if not saves_volume(g['volume_cm3'],parents[0]['volume_cm3'],parents[1]['volume_cm3'],ratio=ratio):
                raise ValueError('A merge exceeds its recorded parent-volume limit')
            for parent in parents:ancestry(parent)
        visited.add(g['sha256']);history.append(dict(sha256=g['sha256'],members=g['members'],
            operation=meta['operation'],volume_cm3=g['volume_cm3'],
            merge_volume_ratio_limit=meta.get('merge_volume_ratio_limit',1.),
            actual_parent_volume_ratio=(g['volume_cm3']/sum(p['volume_cm3'] for p in meta['parents'])
                if meta['parents'] else None)))
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
        if POLICY.get('tangent_ground_attempts'):
            from .ground_oracle import tangent_installation
            proof['ground_oracle']['tangent_installation']=tangent_installation(part,q,support,
                attempts=POLICY['tangent_ground_attempts'])
        print('DISK REPLAY PASS',[model.ids[k] for k in g['members']],flush=True)
    result=dict(status='passed',group_count=len(groups),total_volume_cm3=total,
        original_sample_count=sum(t['force']['original_sample_count'] for p in proofs for t in p['tasks']),
        continuous_outer_target_count=sum(t['force']['continuous_outer_target_count'] for p in proofs for t in p['tasks']),
        groups=groups,proofs=proofs,accepted_ancestry=history,
        strict_sum_volume_rule_verified_for_every_merge=True,
        maximum_merge_volume_ratio=max((h['actual_parent_volume_ratio'] for h in history
            if h['actual_parent_volume_ratio'] is not None),default=None),
        required_merge_volume_ratio=max_merge_ratio,
        ground_oracle_certified=True,
        initial_singletons_equal_saved_step3=True,fresh_disk_replay=True,
        source_initialization_sha256=sha256(source/'initialization.npz'))
    write_json(Path(folder)/'validation.json',result);return model,result


def freeze(workdir,destination,branch=None,*,publish=True,object_name='B',output_root=DEFAULT_OUTPUT):
    """Pin an immutable branch while the other samples continue searching."""
    from collections import Counter
    workdir=Path(workdir);destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    raw=(workdir/'population.json').read_bytes();state=json.loads(raw)
    use_cover=branch is None and 'best_cover' in state
    if branch is None:
        branch=select_branch(state['branches'])
    source_groups=(state['best_cover'] if use_cover
        else state['branches'][branch]['groups']);copied={}
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
    ratio=state['binding'].get('merge_volume_ratio_limit',1.)
    model,validation=validate(source,groups,destination,max_merge_ratio=ratio)
    upstream=json.loads((source/'report.json').read_text())
    report=dict(status='passed',method='independent_membrane_family_merges',best_branch=branch,
        ground_oracle_certified=True,
        initial_group_count=len(model.ids),final_group_count=len(groups),
        initial_volume_cm3=upstream['material_volume_cm3'],final_volume_cm3=validation['total_volume_cm3'],
        groups=groups,counts=dict(Counter(t['operation'] for t in state['trials'] if 'operation' in t)),
        initialization_sha256=sha256(source/'initialization.npz'),
        acceptance=f'V(AB) <= {ratio} * (V(A)+V(B)); every merged member fully certified',
        merge_volume_ratio_limit=ratio,grouping_policy=state['binding'].get('grouping_policy','independent_branches'),
        validation_sha256=sha256(destination/'validation.json'),
        selection=dict(source_workdir=str(workdir.resolve()),branch=branch,
            source_population_snapshot_sha256=sha256(destination/'population_snapshot.json')),
        source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('membrane_guidance.py'),
            Path(__file__).with_name('sampled_guidance.py'),Path(__file__).with_name('policy.json'),
            Path(__file__).with_name('membrane_worker.py'),
            Path(__file__).with_name('family_bank.py'),
            ROOT/'whole_task/step3/membranes.py',ROOT/'whole_task/step3/run_membranes.py',
            Path(__file__).with_name('ground_oracle.py'),
            Path(__file__).with_name('render_membrane_families.py')]},global_minimum_claimed=False)
    write_json(destination/'report.json',report)
    from .render_membrane_families import render
    image=render(model,groups,report,destination/'overview.png')
    report['overview_sha256']=sha256(image);write_json(destination/'report.json',report)
    active=ROOT/'whole_task/data'/object_name/'step4_active.json'
    if publish and active.exists():
        current=json.loads(active.read_text())
        if Path(current['workdir']).resolve()!=destination and not improves_published(
                len(groups),report['final_volume_cm3'],current,required_merge_ratio=ratio):
            publish=False;report['publication']='retained_better_or_equal_published_result'
            write_json(destination/'report.json',report)
    if publish:
        out=Path(output_root)/object_name/'step4';out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(image,out/'overview.png')
        write_json(ROOT/'whole_task/data'/object_name/'step4_active.json',dict(workdir=str(destination),
            method=report['method'],volume_cm3=report['final_volume_cm3'],group_count=len(groups),
            ground_oracle_certified=True,
            merge_volume_ratio_limit=ratio,
            overview_sha256=sha256(out/'overview.png'),report_sha256=sha256(destination/'report.json')))
    print('FROZEN MEMBRANES PASS',len(groups),report['final_volume_cm3'],flush=True)
    return report


def final_merge(state,workdir,workers,seed,target_max_groups):
    """Endgame: the population stops at a fixed round budget, usually with two sets.

    Search the last pair in parallel (placement_search via endgame.search_pair);
    commit only a fully certified merge within the unchanged 75% gate.
    """
    options=POLICY.get('endgame')
    if not options or target_max_groups!=1:return None
    task_count=sum(len(g['members']) for g in state['branches'][0]['groups'])
    cover=state.get('best_cover') or best_cover(state,task_count)
    if len(cover)!=2:return None
    from .endgame import search_pair
    left,right=sorted(cover,key=lambda g:-len(g['members']))
    folder=Path(workdir)/'endgame'/f'{left["sha256"][:12]}_{right["sha256"][:12]}'
    record=search_pair(folder,left,right,rotations=options['rotations'],starts=options['refine_starts'],
        iterations=options['iterations'],certify_count=options['certify'],workers=workers,
        budget=options['budget_s'],seed=seed,screen_seconds=options.get('screen_seconds',3600.),
        refine_seconds=options.get('refine_seconds',1800.),rounds=options.get('rounds',3))
    entry=dict(parents=[left['sha256'],right['sha256']],search=str(folder/'search.json'),
        accepted=bool(record['accepted_count']),
        best_fully_certified_cm3=(record['best_fully_certified'] or {}).get('volume_cm3'))
    if record['accepted_count']:
        group=record['best_accepted']['group'];register(state,group);state['best_cover']=[group]
        entry['group']=group
    state.setdefault('endgame',[]).append(entry)
    checkpoint(Path(workdir)/'population.json',state)
    return record


def run(output_root=DEFAULT_OUTPUT,object_name='B',workdir=None,rounds=8,samples=32,workers=4,
        repair_rounds=1,seed=42,resume=False,publish=True,target_max_groups=1):
    source=ROOT/'whole_task/data'/object_name/'step3_membranes'
    model,q,report=load_model(source)
    workdir=Path(workdir or ROOT/'whole_task/data'/object_name/'step4_compact_families').resolve()
    workdir.mkdir(parents=True,exist_ok=True)
    binding=dict(initialization_sha256=sha256(source/'initialization.npz'),samples=samples,seed=seed,
        ground_oracle_required=True,merge_volume_ratio_limit=MERGE_VOLUME_RATIO,grouping_policy=GROUPING_POLICY)
    if PLACEMENT:binding['proposal_algorithm']=PROPOSAL_ALGORITHM
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
    advance_population(state,source,workdir,rounds,samples,workers,repair_rounds,seed,target_max_groups)
    final_merge(state,workdir,workers,seed,target_max_groups)
    best=select_branch(state['branches'])
    groups=state.get('best_cover',state['branches'][best]['groups'])
    replay_model,validation=validate(source,groups,workdir,max_merge_ratio=MERGE_VOLUME_RATIO)
    result=dict(status='passed',method='independent_membrane_family_merges',initial_group_count=len(model.ids),
        ground_oracle_certified=True,
        final_group_count=len(groups),initial_volume_cm3=report['material_volume_cm3'],
        final_volume_cm3=validation['total_volume_cm3'],groups=groups,best_branch=best,
        counts=state['counts'],initialization_sha256=binding['initialization_sha256'],
        selection_rule='minimum-volume exact cover from certified subgroups; prefer one/two sets; regroup stale samples',
        acceptance='V(AB) <= 0.75 * (V(A) + V(B)); all merged members fully certified',
        merge_volume_ratio_limit=MERGE_VOLUME_RATIO,grouping_policy=GROUPING_POLICY,
        validation_sha256=sha256(workdir/'validation.json'),global_minimum_claimed=False,
        source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('membrane_guidance.py'),
            Path(__file__).with_name('sampled_guidance.py'),Path(__file__).with_name('policy.json'),
            Path(__file__).with_name('membrane_worker.py'),
            Path(__file__).with_name('family_bank.py'),
            ROOT/'whole_task/step3/membranes.py',ROOT/'whole_task/step3/run_membranes.py',
            Path(__file__).with_name('ground_oracle.py'),Path(__file__).with_name('placement_search.py'),
            Path(__file__).with_name('endgame.py'),Path(__file__).with_name('fast_checks.py'),
            Path(__file__).with_name('render_membrane_families.py')]},
        proposal_algorithm=PROPOSAL_ALGORITHM,endgame=state.get('endgame',[]))
    write_json(workdir/'report.json',result)
    active=ROOT/'whole_task/data'/object_name/'step4_active.json'
    if publish and active.exists():
        current=json.loads(active.read_text())
        if Path(current['workdir']).resolve()!=workdir and not improves_published(
                len(groups),result['final_volume_cm3'],current,required_merge_ratio=MERGE_VOLUME_RATIO):
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
            merge_volume_ratio_limit=MERGE_VOLUME_RATIO,
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
