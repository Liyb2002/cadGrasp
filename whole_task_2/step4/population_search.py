"""Step3-seeded persistent branches with random, explicit family unions."""
import json
import os
import shutil
import time
import sys
import subprocess
import queue
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import numpy as np
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import sha256
from whole_task_2.operations.construction import CompactConstructionModel
from whole_task_2.operations.pose_change import pose_change
from whole_task_2.operations.state import transform
from .optimize import Optimizer, record, legal_improvement, subset
from .group_acceptance import (random_pair, merged_groups, validate_partition,
    freeze_independent_material, place_merged_family)
from .sharing_search import KINDS, projection_model
from .spatial_search import floor_repair
from .aligned_side import common_exit, needle_angles
from .gradient_repair import repair


def clone_checkpoint(source, target):
    """Copy a complete monotone proof chain; immutable meshes may share inodes."""
    source,target = Path(source),Path(target)
    target.mkdir(parents=True,exist_ok=True)
    saved = json.loads((source/'current.json').read_text())
    for index in range(saved['accepted']+1):
        for suffix in ('.npz','.json'):
            name = f'state_{index:03}{suffix}'
            dest = target/name
            if dest.exists() and sha256(dest)==sha256(source/name): continue
            temporary = target/(name+'.copy')
            if temporary.exists(): temporary.unlink()
            try: os.link(source/name,temporary)
            except OSError: shutil.copy2(source/name,temporary)
            temporary.replace(dest)
    for name in ('trace.json','current.json'):
        temporary = target/(name+'.copy')
        shutil.copy2(source/name,temporary); temporary.replace(target/name)
    return saved


def warm_start(source, target):
    source,target = Path(source),Path(target)
    saved = clone_checkpoint(source,target)
    witness_manifest=source/'retention_witnesses.json'
    if witness_manifest.exists():
        info=json.loads(witness_manifest.read_text())
        data=source/'retention_witnesses.npz'
        if (info['parent_state_sha256']==saved['sha256'] and data.exists()
                and sha256(data)==info['supply_sha256']):
            for path in (witness_manifest,data):shutil.copy2(path,target/path.name)
    # The old exact-90-degree demo constraint is provenance, not a restriction
    # on free SO(3) proposals. Every installation still has its physical gate.
    old = source/'requirements.json'
    record(target/'warm_start.json',dict(workdir=str(source.resolve()),
        manifest_sha256=sha256(source/'current.json'),volume_cm3=saved['volume_cm3'],
        accepted=saved['accepted'],legacy_requirements=json.loads(old.read_text())if old.exists()else{}))
    record(target/'requirements.json',dict(all_tasks_fully_certified=True,
        installation_rotation='free_SO3',fixed_needle_world_position_and_axis=True))


def load_retention_witnesses(model,folder):
    folder=Path(folder);manifest=folder/'retention_witnesses.json';data=manifest.with_suffix('.npz')
    if not manifest.exists() or not data.exists():return
    info=json.loads(manifest.read_text());current=json.loads((folder/'current.json').read_text())
    if info['parent_state_sha256']!=current['sha256'] or info['supply_sha256']!=sha256(data):return
    with np.load(data) as z:
        model.retention_witnesses={name:z[name].copy() for name in info['tasks']}


def save_retention_witnesses(model,folder):
    witnesses=getattr(model,'retention_witnesses',{})
    if not witnesses:return
    folder=Path(folder);data=folder/'retention_witnesses.npz';temporary=data.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**witnesses)
    temporary.replace(data)
    current=json.loads((folder/'current.json').read_text())
    record(data.with_suffix('.json'),dict(parent_state_sha256=current['sha256'],
        supply_sha256=sha256(data),tasks=sorted(witnesses),
        scope='union of primal bases from each complete parent load certificate'))


def rotate_family_world_pose(layout, owners, angle, axis, head):
    """Spin legal world poses together, preserving the family's object seats.

    Counter-rotate each installation before the rigid family placement.
    Spinning only its representative would distort existing shared seats.
    """
    q=layout.copy()
    first=owners[0]
    spun=pose_change(layout,first,angle,axis,head)
    world_change=spun.poses[first]@np.linalg.inv(layout.poses[first])
    inverse=np.linalg.inv(world_change)
    for k in owners:
        q.poses[k]=world_change@layout.poses[k]
        q.material[k]=layout.material[k]@inverse
        direction=world_change[:3,:3]@layout.exits[k]
        direction/=np.linalg.norm(direction)
        if direction[2]>=.05+1e-10:q.exits[k]=direction
    return q


def make_proposal(model, branch, pair, kind, variant, rng):
    """Rigidly place one existing family against another; never move outsiders."""
    before = branch.layout
    i,j = pair['owner'],pair['host']
    groups = [list(g) for g in branch.groups]
    own = next(g for g in groups if i in g)
    host_group = next(g for g in groups if j in g)
    if own == host_group:
        raise ValueError('Select representatives from two different families')
    moving = own
    q = before.copy()
    center = model.source.center_mass
    target = transform(center,q.material[j]@q.poses[j])
    current = transform(center,q.material[i]@q.poses[i])
    offset = np.zeros(3) if variant<4 else rng.normal(0.,.006,3)
    axial = 0.
    if kind=='juxtaposition':
        # Sample positions near the host, independent of the old arbitrary
        # separation of the two families in Step3's row.
        extent = float(model.source.extents.max())
        local_offset = np.array(((0.,0.,0.),(.15,0.,0.),(-.15,0.,0.),(0.,.2,0.))[variant%4])*extent
        delta = target-current+q.material[j,:3,:3]@local_offset+offset
        for k in moving: q.material[k,:3,3] += delta
    else:
        angles = [pair['axial_deg'],180.,-90.,90.,pair['axial_deg']+20.,pair['axial_deg']-20.]
        angles += needle_angles(before,model.axis,i,j)
        # Zero spin preserves an existing family's relative installations
        # and gravity exactly while its complete fixture turns into the
        # host's seat. Interleave it with flexible world-pose candidates.
        axial = 0. if variant%2==0 else float(angles[(variant//2)%len(angles)])
        q = rotate_family_world_pose(before,moving,axial,model.axis,model.head)
        desired = q.material[j]@q.poses[j]@np.linalg.inv(q.poses[i])
        if Rotation.from_matrix(desired[:3,:3]@before.material[i,:3,:3].T).magnitude()<np.radians(15.):
            # A pair with already similar world poses must still explore an
            # actual fixture turn, rather than relabeling a translation.
            axial += 180.
            q = rotate_family_world_pose(before,moving,axial,model.axis,model.head)
            desired = q.material[j]@q.poses[j]@np.linalg.inv(q.poses[i])
        desired[:3,3] += offset
        extent = float(model.source.extents.max())
        local_offset = np.array(((0.,0.,0.),(.05,0.,0.),(-.05,0.,0.),(0.,.05,0.))[variant%4])*extent
        desired[:3,3] += q.material[j,:3,:3]@local_offset
        change = desired@np.linalg.inv(q.material[i])
        for k in moving: q.material[k] = change@q.material[k]
        shared = common_exit(q,i,j)
        if shared is not None: q = shared
    projected = projection_model(model,before,q)
    owners = sorted(own+host_group)
    local_model, proposed = subset(projected,q,owners)
    local_groups = [[owners.index(k) for k in g] for g in (own,host_group)]
    relaxed_contacts=getattr(model,'placement_contact_policy','pressure_only')!='preserve_all'
    if relaxed_contacts:
        # Old available contacts are not mandatory after merging. Keeping
        # every old corner above each new floor can pull matched seats apart.
        # Place with necessary pressure constraints; the complete wrench
        # certificate and basic repair select the actual new contacts.
        local_model.current_contacts={}
    local = floor_repair(local_model,proposed,local_groups,owners.index(j))
    member_projection=False
    if local is None:
        # Available contacts are not a fixed required contact set. Allow
        # unused contacts to disappear; the wrench audit decides legality.
        local_model.current_contacts = {}
        local = floor_repair(local_model,proposed,local_groups,owners.index(j))
        relaxed_contacts = True
    if local is None:
        # A world spin can give parallel installation planes different
        # offsets inside an existing family. Allow basic member translations
        # to project them back to feasibility. This does NOT split families;
        # the complete union still receives one full acceptance transaction.
        local_model.current_contacts={}
        local=floor_repair(local_model,proposed,[[k] for k in range(len(owners))],owners.index(j))
        member_projection=True;relaxed_contacts=True
    if local is None:
        raise ValueError('Installation floors incompatible for this family pair')
    for key,array in q.arrays().items():
        array[owners] = local.arrays()[key]
    return q, dict(moving_tasks=[model.ids[k] for k in moving],axial_pose_angle_deg=axial,
        existing_family_seats_rigid_before_floor_repair=True,
        member_floor_projection=member_projection,
        variant=variant,host_local_offset_m=local_offset.tolist(),allow_contact_reselection=relaxed_contacts,
        input_families=[[model.ids[k] for k in g] for g in (own,host_group)],
        merged_family=[model.ids[k] for k in owners])


def certify_choices(optimizer,choices,kind,**detail):
    """Try actual smaller shapes in volume order; probes cannot commit."""
    for q,candidate,guidance in sorted(choices,key=lambda item:item[1]['volume_cm3']):
        if candidate['volume_cm3']>=optimizer.volume-1e-5:continue
        if optimizer.try_operation(q,kind,prepared_candidate=candidate,
                gradient_repair=guidance,**detail):return True
    return False


def worker(config):
    """A process evaluates one persistent branch, never a shared incumbent."""
    cores = config['cores']
    if hasattr(os,'sched_setaffinity') and cores:
        identity = config.get('affinity_slot',(mp.current_process()._identity or (1,))[0]-1)
        chunks = np.array_split(cores,config['workers'])
        os.sched_setaffinity(0,[int(c) for c in chunks[identity%len(chunks)]])
    model = CompactConstructionModel(Path(config['output_root']),config['object_name'])
    branch = Optimizer(model,config['folder'],resume=True,audit_resume=False)
    load_retention_witnesses(model,branch.folder)
    with np.load(config['contacts']) as data:
        model.current_contacts={name:data['points_'+name] for name in model.ids}
        model.contact_profiles={name:data['profile_'+name] for name in model.ids}
    # Contacts of accepted branch states supersede the warm-start signature.
    own_contacts = branch.folder/'contacts.npz'
    if own_contacts.exists():
        with np.load(own_contacts) as data:
            model.current_contacts={name:data['points_'+name] for name in model.ids}
            model.contact_profiles={name:data['profile_'+name] for name in model.ids}
    rng = np.random.default_rng(config['seed'])
    kind = config['kind']; iteration = config['iteration']
    pair = random_pair(model,branch.layout,branch.groups,rng)
    if pair is None:
        result = dict(branch=config['branch'],iteration=iteration+1,kind=kind,tasks=[],
            accepted=False,parent_volume_cm3=branch.volume,volume_cm3=branch.volume,
            family_count=1,reason='All tasks already form one family',folder=str(branch.folder))
        record(branch.folder/'attempt.json',result)
        return result
    tasks = [model.ids[pair[k]] for k in ('owner','host')]
    groups = merged_groups(branch.groups,pair['owner'],pair['host'])
    owners = sorted(pair['owner_group']+pair['host_group'])
    branch.event('random_family_pair_selected',kind,branch=config['branch'],iteration=iteration+1,
        tasks=tasks,input_families=[[model.ids[k] for k in pair[key]]
            for key in ('owner_group','host_group')],parent_volume_cm3=branch.volume)
    success = False; reason = ''; initial = branch.volume
    model.proposal_deadline=time.monotonic()+config.get('trial_timeout_s',900.)
    def finish(accepted, reason=''):
        if accepted:
            np.savez_compressed(own_contacts,**{
                **{'points_'+name:model.current_contacts[name] for name in model.ids},
                **{'profile_'+name:model.contact_profiles[name] for name in model.ids}})
        save_retention_witnesses(model,branch.folder)
        result = dict(branch=config['branch'],iteration=iteration+1,kind=kind,tasks=tasks,
            families=[[model.ids[k] for k in pair[key]] for key in ('owner_group','host_group')],
            checked_tasks=[model.ids[k] for k in owners],family_count=len(branch.groups),
            accepted=accepted,parent_volume_cm3=initial,volume_cm3=branch.volume,
            reason=reason,folder=str(branch.folder))
        record(branch.folder/'attempt.json',result)
        return result
    try:
        policy=json.loads(Path(__file__).with_name('policy.json').read_text())
        mode=policy.get('family_material_mode','local')
        model.family_refit_mode=('free_material' if branch.accepted==0 else 'local') if mode=='thin_seed_then_local' else mode
        model.guidance_geometry=policy.get('guidance_geometry','exact')
        model.guidance_material=policy.get('guidance_material','raw_stock')
        model.placement_contact_policy=policy.get('placement_contact_policy','pressure_only')
        model.simplify_construction_boundaries=policy.get('simplify_construction_boundaries',False)
        model.optimize_connecting_stock=policy.get('optimize_connecting_stock',False)
        freeze_independent_material(model,branch.layout,branch.support,owners)
        profile=policy['candidate_stock']
        model.stock_grid=(profile['pitch_m'],profile['post_width_m'])
        q, detail = make_proposal(model,branch,pair,kind,config['slot']//4+2*iteration,rng)
        q = place_merged_family(model,branch.support,q,owners,clearance=.02)
        detail['installation_change_deg']={model.ids[k]:float(np.degrees(
            Rotation.from_matrix(q.material[k,:3,:3]@branch.layout.material[k,:3,:3].T).magnitude()))
            for k in owners}
        parent=json.loads((branch.folder/'current.json').read_text())
        record(branch.folder/'last_proposal.json',dict(operation=kind,
            parent_state=parent['state'],parent_state_sha256=parent['sha256'],
            owners=[model.ids[k] for k in owners],groups=groups,detail=detail,
            layout={key:value.tolist() for key,value in q.arrays().items()}))
        branch.event('sharing_proposed',kind,tasks=tasks,**detail)
        # A high-level seed is a proposal. Even a legal seed can benefit
        # from the three basic operations before the merge is committed.
        candidate = model.build(q)
        branch.event('sharing_built',kind,tasks=tasks,volume_cm3=candidate['volume_cm3'])
        choices=[(q,candidate,dict(status='unadjusted_high_level_seed'))]
        proposed=q;history=[];feasible=[]
        if policy.get('basic_before_acceptance',False):
            polished,guidance=repair(model,q,[pair['owner'],pair['host']],config['repair_rounds'],branch.event,
                obstruction_context=candidate['tasks'],guidance_owners=owners,feasible_layouts=feasible)
            if polished is not None:
                history.append(guidance)
                if any(not np.array_equal(a,b) for a,b in zip(q.arrays().values(),polished.arrays().values())):
                    try:
                        improved=model.build(polished)
                        choices.append((polished,improved,dict(**guidance,basic_before_acceptance=True)))
                        proposed=polished
                    except (ValueError,RuntimeError,AssertionError,np.linalg.LinAlgError) as error:
                        branch.event('basic_proposal_unbuildable',kind,reason=str(error)[:900])
        if certify_choices(branch,choices,kind,groups=groups,audit_owners=owners,tasks=tasks,
                branch=config['branch'],iteration=iteration+1,high_level_sampling_probability=.5,**detail):
            return finish(True)
        if ('independent family' in getattr(branch,'last_trial_error','')
                and model.guidance_geometry!='sampled'):
            return finish(False,branch.last_trial_error)
        model.check_deadline()
        attempts=int(policy.get('repair_attempts',3)) if model.guidance_geometry=='sampled' else 1
        for refinement in range(attempts):
            repaired,guidance=repair(model,proposed,[pair['owner'],pair['host']],config['repair_rounds'],branch.event,
                obstruction_context=candidate['tasks'],guidance_owners=owners,feasible_layouts=feasible)
            if repaired is None:break
            history.append(guidance)
            same=all(np.array_equal(a,b) for a,b in zip(proposed.arrays().values(),repaired.arrays().values()))
            if same:break
            bundle=dict(status=guidance['status'],guidance_geometry=model.guidance_geometry,
                refinement_attempts=deepcopy(history),active_failed_constraints_used=True)
            success=branch.try_operation(repaired,kind,groups=groups,audit_owners=owners,
                tasks=tasks,branch=config['branch'],iteration=iteration+1,
                high_level_sampling_probability=.5,gradient_repair=bundle,**detail)
            if success:break
            history[-1]['full_validation_rejection']=branch.last_trial_error or branch.last_trial_status
            proposed=repaired
            model.check_deadline()
        # Validate an earlier covered adjustment rather than repeating the
        # already-rejected unchanged high-level seed. Sampling guides only;
        # the original exact acceptance decides each alternative.
        if not success and feasible:
            alternative=feasible[0]
            if not all(np.array_equal(a,b) for a,b in zip(proposed.arrays().values(),alternative.arrays().values())):
                success=branch.try_operation(alternative,kind,groups=groups,audit_owners=owners,
                    tasks=tasks,branch=config['branch'],iteration=iteration+1,
                    gradient_repair=dict(refinement_attempts=deepcopy(history),
                        guidance_geometry=model.guidance_geometry,use_earliest_covered_guidance=True),
                    high_level_sampling_probability=.5,**detail)
        if not success: reason = getattr(branch,'last_trial_error','') or getattr(branch,'last_trial_status','No volume-reducing basic adjustment')
    except (ValueError,RuntimeError,AssertionError,np.linalg.LinAlgError) as error:
        reason = str(error)
        branch.event('sharing_sample_rejected',kind,tasks=tasks,reason=reason[:900])
    return finish(success,reason)


def bounded_worker(config, slots):
    """A slow native Boolean may time out without killing other branches."""
    folder=Path(config['folder']);attempt=folder/'attempt.json'
    if attempt.exists():
        previous=json.loads(attempt.read_text())
        if previous['iteration']==config['iteration']+1 and previous['kind']==config['kind']:
            return previous
    slot=slots.get()
    try:
        config=dict(config,affinity_slot=slot)
        before=json.loads((folder/'current.json').read_text())
        log=folder/f"trial_{config['iteration']+1:03}.log"
        with log.open('w') as stream:
            try:
                subprocess.run([sys.executable,'-m','whole_task_2.step4.population_worker'],
                    input=json.dumps(config),text=True,stdout=stream,stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parents[2],
                    timeout=config.get('trial_timeout_s',900.)+30.,check=True)
                return json.loads(attempt.read_text())
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                after=json.loads((folder/'current.json').read_text())
                accepted=after['volume_cm3']<before['volume_cm3']-1e-5
                # A deadline may arrive after a fully audited atomic commit
                # but before diagnostic contact signatures are serialized.
                if accepted:
                    path=folder/after['state'];metadata=json.loads(path.with_suffix('.json').read_text())
                    proof=dict(volume_cm3=after['volume_cm3'],audit=metadata['audit'],
                        tasks=[dict(id=name) for name in config['task_ids']])
                    if sha256(path)!=after['sha256'] or not legal_improvement(before['volume_cm3'],proof):
                        raise RuntimeError('Cannot recover an unbound or incomplete checkpoint')
                rows=json.loads((folder/'trace.json').read_text())
                selected=next((r for r in reversed(rows) if r['status']=='random_family_pair_selected'
                    and r.get('iteration')==config['iteration']+1),{})
                result=dict(branch=config['branch'],iteration=config['iteration']+1,
                    kind=config['kind'],tasks=selected.get('tasks',[]),
                    accepted=accepted,parent_volume_cm3=before['volume_cm3'],
                    volume_cm3=after['volume_cm3'],reason=(''if accepted else
                        'Evaluation budget exhausted; certified parent retained' if isinstance(error,subprocess.TimeoutExpired)
                        else 'Worker failed; certified parent retained; see '+str(log)),folder=str(folder))
                record(attempt,result)
                return result
    finally:
        slots.put(slot)


def planned_rounds(saved, rng, start, stop, samples):
    """Durable 50/50 assignments, independent of worker completion order."""
    plan={int(k):list(v) for k,v in saved.get('round_assignments',{}).items()}
    old=saved.get('assignments')
    if old:
        plan.setdefault(saved.get('completed_rounds',0),list(old))
    for iteration in range(start,stop):
        if iteration not in plan:
            plan[iteration]=[str(k) for k in rng.permutation(
                [KINDS[0]]*(samples//2)+[KINDS[1]]*(samples//2))]
        if (len(plan[iteration])!=samples or
                any(plan[iteration].count(k)!=samples//2 for k in KINDS)):
            raise ValueError('Every logical round must assign half of each operation')
    return plan


def next_iteration(done, branch, stop):
    iteration=0
    while iteration<stop and (iteration+1,branch) in done:
        iteration+=1
    return iteration


def completed_rounds(done, samples):
    count=0
    while all((count+1,k) in done for k in range(samples)):
        count+=1
    return count


def solve(optimizer, rounds=32, samples=32, repair_rounds=1, seed=42, workers=4):
    if rounds<1 or samples<2 or samples%2 or repair_rounds<1 or workers<1:
        raise ValueError('Positive rounds/workers/repair rounds and an even population >=2 required')
    folder = optimizer.folder; population = folder/'population'; population.mkdir(exist_ok=True)
    state_path = folder/'sharing_search.json'
    saved = json.loads(state_path.read_text()) if state_path.exists() else {}
    if saved and saved.get('method')!='random_family_union_from_step3':
        raise ValueError('Legacy populations cannot resume as Step3 family search')
    if not saved and (optimizer.accepted != 0 or optimizer.groups != [[i] for i in range(len(optimizer.model.ids))]):
        raise ValueError('Start this search at the untouched Step3 initialization')
    if saved and saved.get('population_size')!=samples:
        raise ValueError('A persistent population cannot change its size on resume')
    rng = np.random.default_rng(seed)
    if 'rng_state' in saved: rng.bit_generator.state=saved['rng_state']
    history = saved.get('attempts',[])
    done={(r['iteration'],r['branch']) for r in history}
    completed=completed_rounds(done,samples)
    stop=completed+rounds
    plan=planned_rounds(saved,rng,completed,stop,samples)
    contacts = folder/'population_seed_contacts.npz'
    if not contacts.exists():
        candidate=dict(layout=optimizer.layout,support=optimizer.support,
            tasks=optimizer.model.prepare(optimizer.layout),volume_cm3=optimizer.volume)
        optimizer.model.audit(candidate,full=True,continuous=True)
        from .pose_neighbors import remember_contacts
        remember_contacts(optimizer.model,candidate['tasks'])
        save_retention_witnesses(optimizer.model,folder)
        current=json.loads((folder/'current.json').read_text())
        metadata_path=(folder/current['state']).with_suffix('.json')
        metadata=json.loads(metadata_path.read_text())
        metadata['audit']=candidate['audit']
        record(metadata_path,metadata)
        np.savez_compressed(contacts,**{
            **{'points_'+name:optimizer.model.current_contacts[name] for name in optimizer.model.ids},
            **{'profile_'+name:optimizer.model.contact_profiles[name] for name in optimizer.model.ids}})
    for k in range(samples):
        target=population/f'branch_{k:02}'
        if not (target/'current.json').exists(): warm_start(folder,target)
    def persist():
        complete=completed_rounds(done,samples)
        counts={kind:sum(row['kind']==kind for row in history) for kind in KINDS}
        record(state_path,dict(method='random_family_union_from_step3',
            initialization='accepted Step3 initialization; singleton task families',
            pair_selection='uniform random distinct families, uniform random representative tasks',
            acceptance_scope='union of selected families; certificates retained for unchanged families',
            population_size=samples,workers=workers,completed_rounds=complete,
            scheduling='asynchronous branches, balanced assignments per logical round',
            round_assignments={str(k):v for k,v in plan.items()},round_limit=stop,
            iterations_by_branch=[next_iteration(done,k,stop) for k in range(samples)],
            trial_timeout_s=900.,
            assignments=plan.get(complete,[]),proposal_counts=counts,
            accepted_counts={kind:sum(row['kind']==kind and row['accepted'] for row in history) for kind in KINDS},
            attempts=history,rng_state=rng.bit_generator.state,pso_used=False,
            acceptance_relative_to='each_branch_parent',global_best_volume_cm3=optimizer.volume,
            marginal_operation_probability={kind:.5 for kind in KINDS},
            only_fully_certified_states_committed=True))
    def job(k,iteration):
        assignments=plan[iteration];kind=assignments[k]
        slot=assignments[:k].count(kind)
        return dict(folder=str(population/f'branch_{k:02}'),branch=k,kind=str(kind),slot=slot,
                iteration=iteration,repair_rounds=repair_rounds,seed=seed+1009*iteration+k,
                trial_timeout_s=900.,
                output_root=str(optimizer.model.output_root),object_name=optimizer.model.name,
                contacts=str(contacts),workers=min(workers,samples),
                task_ids=list(optimizer.model.ids),
                cores=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [])
    slots=queue.Queue()
    for slot in range(min(workers,samples)): slots.put(slot)
    persist()
    optimizer.event('asynchronous_population_started','sharing',population_size=samples,
        completed_rounds=completed,round_limit=stop,operation_probability=.5)
    with ThreadPoolExecutor(max_workers=min(workers,samples)) as pool:
        futures={}
        def submit(k):
            iteration=next_iteration(done,k,stop)
            if iteration<stop:
                config=job(k,iteration)
                futures[pool.submit(bounded_worker,config,slots)]=config
        for k in range(samples): submit(k)
        while futures:
            ready,_=wait(futures,return_when=FIRST_COMPLETED)
            for future in ready:
                config=futures.pop(future)
                result=future.result();history.append(result)
                done.add((result['iteration'],result['branch']))
                if result['volume_cm3']<optimizer.volume-1e-5:
                    clone_checkpoint(result['folder'],folder)
                    adopted = Optimizer(optimizer.model,folder,resume=True,audit_resume=False)
                    optimizer.__dict__.update(adopted.__dict__)
                    optimizer.event('population_best_improved',result['kind'],
                        branch=result['branch'],volume_cm3=optimizer.volume,iteration=result['iteration'])
                persist()
                print('POPULATION trial',len(history),'round',result['iteration'],
                    'best_cm3',optimizer.volume,'branch',result['branch'],result['reason'][:160],flush=True)
                submit(config['branch'])
    persist()
    return optimizer.layout,optimizer.support
