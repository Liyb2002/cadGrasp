"""Joint head-set/common world exit search on the immutable current dataset.

Directions are searched, never imposed as +Z. No object/fixture registration
variables are optimized. This stage certifies contact programs and their own
finite-depth heads; complete shared-body construction is a subsequent stage.
"""
import argparse,json,time,contextlib
from scipy.optimize import linprog
from pathlib import Path
import numpy as np
from codes.precompute_objects.registry import task_poses
from step3_scheculer import contacts as I,passive_support as U
from step3_scheculer import initialize_cached_v19 as OLD,additive_gpu_v24 as GPU
from step3_scheculer import shared_direction_paths as P
from step3_scheculer.pair_scoring import J,C
from step3_scheculer.joint_prepared import scoring_sources

STAGE='dsl_shared_direction_batch_v33'


@contextlib.contextmanager
def bounded_lps():
    original=C.W.linprog
    def limited(*args,**kwargs):
        kwargs['options']={**kwargs.get('options',{}),'time_limit':1.}
        return original(*args,**kwargs)
    C.W.linprog=limited
    try:yield
    finally:C.W.linprog=original


def unit(v):
    v=np.asarray(v,float)
    if v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-12:
        raise ValueError('Expected a finite nonzero world direction')
    return v/np.linalg.norm(v)


def unit7(v):return v/np.linalg.norm(v)


def direction_key(v):return tuple(np.round(unit(v),10))


def direction_menu(searches):
    """Intersect WORLD object-exit rays, not pose-local IDs or object axes."""
    menus=[]
    for s in searches:
        menus.append({direction_key(-np.asarray(v)):unit(-np.asarray(v))
                      for v in s.catalogue['vectors']})
    common=set(menus[0])
    for m in menus[1:]:common.intersection_update(m)
    return [menus[0][k] for k in sorted(common)]


def local_ray_ids(search,direction):
    return {i for i,v in enumerate(search.catalogue['vectors'])
            if np.linalg.norm(unit(-np.asarray(v))-direction)<1e-8}


def swept_box_proxy(tasks,direction):
    # Ordering only: never label this as constructed fixture volume.
    return sum(float(np.prod(t.domain.mesh.extents+.5*np.abs(direction))) for t in tasks)


class Search(OLD.Search):
    def __init__(self,name,pose,device):
        # Redirect OLD's output variable only during construction; no old files
        # are written. All inputs still come from strict native Step2 caches.
        old_stage=OLD.STAGE
        try:
            OLD.STAGE=STAGE+'_inputs'
            super().__init__(name,pose,device=device)
        finally:OLD.STAGE=old_stage
        self.backend=GPU.GPUClassifier(self.task.targets,device)
        self.force_cache={};self.unresolved={}
        self.pool=[e for p in self.pools.values() for e in p if e['valid']]
        self.by_id={e['contact']['candidate_id']:e for e in self.pool}
        self.seed_path=I.OUTPUTS/name/old_stage/pose/'step3_scheculer/schedule.json'
        self.seed=I.check_report(self.seed_path)
        self.inputs.append(self.seed_path)
        self.analyzer=P.PathAnalyzer(self.task.domain.mesh,self.depth)

    def eligible(self,direction):
        ids=local_ray_ids(self,direction)
        return [e for e in self.pool if ids.intersection(e['directions'][0])]

    def solve_direction(self,direction,max_heads,shortlist):
        pool=self.eligible(direction)
        if not pool:return [],dict(pose=self.pose,passed=False,status='no_cached_compatible_head')
        optimistic=I.merge_columns(self.floor,*[self.columns[e['contact']['candidate_id']] for e in pool])
        for index in np.linspace(0,len(self.task.targets)-1,8,dtype=int):
            target=U.target(self.task.targets[index],7)
            result=linprog(-target,A_ub=optimistic,b_ub=np.zeros(len(optimistic)),
                bounds=[(-1,1)]*7,method='highs',options=dict(time_limit=.25,
                    primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            if result.success and np.linalg.norm(result.x)>1e-12:
                normal=unit7(result.x)
                if np.max(optimistic@normal)<=1e-12 and target@normal>1e-8+1e-12:
                    return [],dict(pose=self.pose,passed=False,status='finite_eligible_cone_cannot_cover',
                        failed_original_load=int(index),separator=normal.tolist())
        # Keep compatible original heads first. A second start permits atomic
        # replacement of a seed whose retained heads restrict roadmap ports.
        allowed={e['contact']['candidate_id'] for e in pool}
        seed=[self.by_id[k] for k in self.seed['result']['selected_ids'] if k in allowed]
        attempts=[]
        for initial in (seed,[]):
            selected=list(initial)
            if len(selected)>max_heads:continue
            if selected:
                _,ports=OLD.compatibility(selected,self.catalogue)
                if not ports:continue
            mask,_=self.classify(selected)
            rounds=[]
            while not mask.all() and len(selected)<max_heads:
                _,ports=OLD.compatibility(selected,self.catalogue) if selected else (set(),None)
                candidates=[e for e in pool if (ports is None or ports.intersection(e['path_components']))
                    and not any(np.linalg.norm(e['contact']['center_m']-v['contact']['center_m'])<self.scale*1e-6 for v in selected)]
                if not candidates:break
                # Locality makes substitutions near initialized contacts likely.
                # The bounded shortlist affects proposals only, never acceptance.
                centers=[e['contact']['center_m'] for e in seed]
                if centers:
                    candidates.sort(key=lambda e:(min(np.linalg.norm(e['contact']['center_m']-c) for c in centers),e['contact']['candidate_id']))
                else:candidates.sort(key=lambda e:e['contact']['candidate_id'])
                best=None
                for e in candidates[:shortlist]:
                    try:new,_=self.classify(selected+[e],mask)
                    except RuntimeError:continue
                    score=int(new.sum())
                    if best is None or score>best[0]:best=(score,e,new)
                if best is None:break
                score,e,new=best
                assert np.all(new[mask])
                selected.append(e);mask=new
                rounds.append(dict(added_id=e['contact']['candidate_id'],covered=score))
            plan=dict(kind='ray',object_translation_waypoints_world_m=[[0.,0.,0.],(.5*direction).tolist()],
                initial_object_exit_world=direction.tolist(),terminal_object_exit_world=direction.tolist(),rotation_allowed=False)
            path=self.analyzer.test(self.analyzer.heads([e['contact'] for e in selected]),plan) if selected else dict(clear=False)
            result=dict(pose=self.pose,heads=len(selected),covered=int(mask.sum()),passed=bool(mask.all() and path['clear']),
                selected_ids=[e['contact']['candidate_id'] for e in selected],rounds=rounds,path=plan,path_check=path)
            attempts.append(result)
            if result['passed']:return selected,result
        return [],dict(pose=self.pose,passed=False,attempts=attempts,status='bounded_head_search_failed')


def solve(name,set_id,config):
    started=time.monotonic()
    manifest=I.ROOT/'objects'/name/'pose_sets.json'
    sets=json.loads(manifest.read_text())['sets']
    group=next((g for g in sets if g['id']==set_id),None)
    if group is None:
        ids=[int(x.removeprefix('pose')) for x in set_id.split('+')]
        if len(ids)<2 or len(set(ids))!=len(ids) or any(i<1 or i>30 for i in ids):raise ValueError('Invalid current pose combination')
        group=dict(id=set_id,poses=[f'pose_{i}' for i in ids])
    out=I.OUTPUTS/name/set_id/'step3_scheculer'/STAGE
    out.mkdir(parents=True,exist_ok=True)
    existing=out/'report.json'
    if existing.is_file():
        d=I.check_report(existing)
        if d['config']!=config:raise ValueError('Existing config differs; preserve recorded result')
        return d
    searches=[Search(name,p,config['device']) for p in group['poses']]
    warm=I.OUTPUTS/name/set_id/'step3_scheculer/dsl_shared_direction_v31/report.json'
    if warm.is_file():
        prior=I.check_report(warm)
        if prior['contact_programs_passed']:
            assert prior['poses']==group['poses']
            for search,row in zip(searches,prior['per_pose']):
                search.seed=dict(result=dict(selected_ids=row['selected_ids']))
                search.inputs.append(warm)
    directions=direction_menu(searches)
    directions.sort(key=lambda d:(swept_box_proxy([s.task for s in searches],d),direction_key(d)))
    if directions:
        diverse=[directions.pop(0)]
        while directions:
            j=min(range(len(directions)),key=lambda j:(max(directions[j]@d for d in diverse),
                swept_box_proxy([s.task for s in searches],directions[j]),direction_key(directions[j])))
            diverse.append(directions.pop(j))
        directions=diverse
    history=[];winner=None
    for direction in directions[:config['directions']]:
        rows=[];contacts=[]
        for s in searches:
            selected,result=s.solve_direction(direction,config['max_heads'],config['shortlist'])
            rows.append(result);contacts.append(selected)
            if not result['passed']:break
        trial=dict(direction_world=direction.tolist(),passed=len(rows)==len(searches) and all(r['passed'] for r in rows),poses=rows)
        history.append(trial)
        I.save(out/'progress.json',dict(complete=False,trials=history))
        print(name,set_id,'world exit',np.round(direction,4),'passed',trial['passed'],flush=True)
        if trial['passed']:
            winner=(direction,contacts,rows);break
    # Alternate direction refinement with actual head-set repair. Trials are
    # atomic; a failed pose leaves every accepted subprogram unchanged.
    if winner:
        best_cost=swept_box_proxy([s.task for s in searches],winner[0])
        nearby=sorted([d for d in directions if np.linalg.norm(d-winner[0])>1e-8],
            key=lambda d:(-float(d@winner[0]),direction_key(d)))[:config['refine_directions']]
        for direction in nearby:
            old_seeds=[s.seed for s in searches]
            for search,selected in zip(searches,winner[1]):
                search.seed=dict(result=dict(selected_ids=[e['contact']['candidate_id'] for e in selected]))
            rows=[];contacts=[]
            for search in searches:
                selected,result=search.solve_direction(direction,config['max_heads'],config['shortlist'])
                rows.append(result);contacts.append(selected)
                if not result['passed']:break
            for search,seed in zip(searches,old_seeds):search.seed=seed
            passed=len(rows)==len(searches) and all(r['passed'] for r in rows)
            cost=swept_box_proxy([s.task for s in searches],direction)
            accepted=passed and cost<best_cost-1e-12
            history.append(dict(direction_world=direction.tolist(),passed=passed,poses=rows,
                operation='direction_and_head_repair',proxy=cost,accepted=accepted))
            if accepted:winner=(direction,contacts,rows);best_cost=cost
            print(name,set_id,'refine',np.round(direction,4),'feasible',passed,'accepted',accepted,flush=True)
    artifacts={}
    if winner:
        direction,contacts,rows=winner
        for s,selected,row in zip(searches,contacts,rows):
            fresh=[]
            for e in selected:
                c=e['contact'];points=c['triangles_m'].reshape(-1,3)
                normals=np.repeat(-s.task.domain.mesh.face_normals[c['source_faces']],3,axis=0)
                np.testing.assert_allclose(np.c_[normals,np.cross(points-s.task.domain.com,normals)],c['wrench_generators'],atol=1e-12,rtol=0)
                fresh.append({k:v for k,v in c.items() if k not in ('wrench_generators','wrench_com_m')})
            mask,info=J.classify(s.task.supply(fresh),s.task.targets)
            assert mask.all() and len(mask)==32768
            row['cpu_original_load_check']=info
            row['all_original_loads_passed']=True
            path=out/f'contacts_{s.pose}.npz';I.save_contacts(path,[e['contact'] for e in selected])
            coverage=out/f'coverage_{s.pose}.npz';np.savez_compressed(coverage,mask=mask)
            artifacts.update({p.name:I.sha256(p) for p in (path,coverage)})
    else:direction=None;rows=[]
    inputs=[manifest]+[p for s in searches for p in s.inputs]
    report=dict(complete=True,schema=STAGE,object=name,poses=group['poses'],config=config,
        contact_programs_passed=winner is not None,passed=winner is not None,
        shared_object_exit_world=direction.tolist() if direction is not None else None,
        object_poses_changed=False,world_xyz_frame=True,exit_direction_prespecified=False,
        force_constraint=U.description(),load_count_per_pose=32768,load_subsampling=False,
        common_menu_size=len(directions),directions_tried=len(history),history=history,per_pose=rows,
        full_support_constructed=False,foreign_heads_and_connections_checked=False,
        passed_scope='Each native-pose head program: all original loads, shared no-uplift equation and its own continuous head exit. NOT a complete shared fixture.',
        seconds=time.monotonic()-started,numerically_unresolved_proposals={s.pose:len(s.unresolved) for s in searches},
        provenance=dict(inputs=I.hashes(inputs),code=I.hashes([Path(__file__),Path(OLD.__file__),Path(GPU.__file__)]+P.sources()+scoring_sources())),artifacts=artifacts)
    I.save(existing,report);I.check_report(existing)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('object');parser.add_argument('--sets',nargs='+',required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--directions',type=int,default=24)
    parser.add_argument('--refine-directions',type=int,default=8)
    parser.add_argument('--max-heads',type=int,default=10);parser.add_argument('--shortlist',type=int,default=64)
    args=parser.parse_args()
    if min(args.directions,args.max_heads,args.shortlist)<1 or args.refine_directions<0:parser.error('Budgets must be positive')
    config={k:getattr(args,k) for k in ('device','directions','max_heads','shortlist','refine_directions')}
    for group in args.sets:
        with bounded_lps():d=solve(args.object,group,config)
        print('DONE',group,d['contact_programs_passed'],d['shared_object_exit_world'],d['seconds'],flush=True)

if __name__=='__main__':main()
