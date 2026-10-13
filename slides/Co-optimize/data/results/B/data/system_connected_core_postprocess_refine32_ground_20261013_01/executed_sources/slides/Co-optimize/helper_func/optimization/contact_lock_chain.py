"""Incremental per-pose contact locks; no support Boolean operations in search."""
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector
from co_common import J, save, provenance
from fast_candidate_chain import cheap_candidates


class ContactLocks:
    def __init__(self,mesh,points,normals,length):
        self.points=points;self.normals=normals;self.length=length
        self.epsilon=float(mesh.extents.max())*1e-7
        self.origins=points+self.epsilon*normals
        self.intersector=RayMeshIntersector(mesh)
        self.cache=OrderedDict()
    def locked(self,direction):
        key=tuple(np.asarray(direction,float))
        if key not in self.cache:
            blocked=self.normals@direction>1e-9
            # p lies in the translated object iff p-t*d lies inside the original.
            # Trace that continuous segment, rather than a time-sampled sweep.
            ids=np.flatnonzero(~blocked)
            if len(ids):
                locations,rays,_=self.intersector.intersects_location(self.origins[ids],
                    np.tile(-direction,(len(ids),1)),multiple_hits=False)
                distance=np.linalg.norm(locations-self.origins[ids[rays]],axis=1)
                blocked[ids[rays[distance<=self.length]]]=True
            self.cache[key]=blocked
            if len(self.cache)>128:self.cache.popitem(last=False)
        return self.cache[key]
    def update(self,directions,previous=None):
        if previous is None:locks=np.stack([self.locked(d) for d in directions])
        else:
            locks=previous['locks'].copy()
            changed=np.flatnonzero(np.any(directions!=previous['directions'],axis=1))
            for k in changed:locks[k]=self.locked(directions[k])
        counts=locks.sum(axis=0)
        return locks,counts,counts==0


def contact_state(search,model,directions,previous=None):
    t=time.perf_counter();locks,lock_count,active=model.update(directions,previous)
    lock_seconds=time.perf_counter()-t
    t=time.perf_counter();search.exact_calls+=1
    masks=[];infos=[];supplies=[]
    for k,(task,T) in enumerate(search.states):
        full=np.vstack([search.floors[k],search.rays[k][active]])
        mask,info=J.classify(full,task.targets)
        masks.append(mask);infos.append(info);supplies.append(full)
    return dict(serial=search.exact_calls,directions=directions.copy(),locks=locks,lock_count=lock_count,
                active=active,masks=masks,infos=infos,supplies=supplies,counts=[int(m.sum()) for m in masks],
                contact_lock_seconds=lock_seconds,load_classification_seconds=time.perf_counter()-t,
                killed_contacts=0 if previous is None else int(np.count_nonzero(previous['active']&~active)),
                released_contacts=0 if previous is None else int(np.count_nonzero(~previous['active']&active)))


def contact_chain_search(search,iterations=1):
    began=time.perf_counter();search.candidate_rng=np.random.default_rng(search.chain_seed)
    model=ContactLocks(search.mesh,search.points,search.ray_normals,search.length)
    current=contact_state(search,model,search.normals.copy());initial_counts=current['counts']
    search.remember(current);events=[];timings=[];rounds=0
    np.savez_compressed(search.out/'initial_directions.npz',directions=current['directions'])
    initial_timing=dict(contact_lock_seconds=current['contact_lock_seconds'],load_classification_seconds=current['load_classification_seconds'])
    while search.proposals<search.max_proposals and not all(m.all() for m in current['masks']):
        rounds+=1;count=min(search.candidates_per_round,search.max_proposals-search.proposals)
        rows,timing=cheap_candidates(search,current,count);search.proposals+=count
        accepted=False;checks=[]
        for row in rows[:search.exact_finalists]:
            t=time.perf_counter()
            try:
                proposed=contact_state(search,model,np.array(row['gradient_descent']),current)
                search.choose_loads(proposed);loads=sorted(search.load_bank)
                before,_=search.actual_loss(current,loads);after,_=search.actual_loss(proposed,loads)
                feasible=all(m.all() for m in proposed['masks'])
                protected=all(not a.all() or b.all() for a,b in zip(current['masks'],proposed['masks']))
                accepted=bool(protected and (feasible or after<before-max(1e-12,before*1e-4)))
                row.update(counts=proposed['counts'],accepted=accepted,real_loss_before=before,real_loss_after=after,
                           killed_contacts=proposed['killed_contacts'],released_contacts=proposed['released_contacts'])
                checks.append(dict(candidate=row['candidate'],contact_lock_seconds=proposed['contact_lock_seconds'],
                                   load_classification_seconds=proposed['load_classification_seconds'],total_seconds=time.perf_counter()-t))
                if accepted:current=proposed;search.remember(current)
            except (RuntimeError,ValueError) as error:row.update(error=str(error),accepted=False)
            if accepted:break
        timing.update(round=rounds,candidates=count,contact_checks=checks);timings.append(timing)
        events.append(dict(round=rounds,candidates=rows,accepted=accepted,counts=current['counts'],
                           directions=current['directions'].tolist(),active_contacts=int(current['active'].sum())))
        save(search.out/'chain_trajectory.json',events)
        save(search.out/'candidate_timing.json',dict(initial=initial_timing,rounds=timings))
        print('CONTACT ROUND',rounds,current['counts'],'accepted',accepted,'checks',checks,flush=True)
    np.savez_compressed(search.out/'directions.npz',directions=current['directions'])
    np.savez_compressed(search.out/'contact_locks.npz',directions=current['directions'],locks=current['locks'],
                        lock_count=current['lock_count'],active=current['active'],points=search.points,sources=search.point_sources)
    report=dict(algorithm='one chain, cheap candidates, incremental per-pose contact locks',complete=True,
                contact_model_force_passed=bool(all(m.all() for m in current['masks'])),full_fixture_accepted=False,
                geometry_constructed=False,clearance_certified=False,pose_set=search.group['id'],initial_counts=initial_counts,
                final_counts=current['counts'],load_count_per_pose=[len(t.targets) for t,T in search.states],
                seconds=time.perf_counter()-began,rounds=rounds,candidates=search.proposals,
                contact_count=len(search.points),active_contacts=int(current['active'].sum()),
                contact_model='finite Step3.3 seed contact vertices; local normal compatibility plus continuous reverse-ray segment obstruction; no patch-area/core/clearance certification',
                ray_origin_epsilon_m=model.epsilon,original_loads_reused=True,load_subsampling_for_final_acceptance=False,
                directions=current['directions'].tolist(),classification=current['infos'],
                provenance=provenance([search.seed_path,search.contact_path,search.initial_snapshot]+[p for task,T in search.states for p in task.inputs],
                    [Path(__file__),Path(__file__).with_name('fast_candidate_chain.py'),Path(__file__).with_name('single_pose_chain.py')]+search.additional_code))
    save(search.out/'report.json',report)
    print('CONTACT FINAL',report['final_counts'],'seconds',report['seconds'],flush=True)
    return report
