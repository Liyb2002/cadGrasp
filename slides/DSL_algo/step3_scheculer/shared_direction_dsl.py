"""Joint head-set/common world exit search on the immutable current dataset.

Directions are searched, never imposed as +Z. No object/fixture registration
variables are optimized. This stage certifies contact programs and their own
finite-depth heads; complete shared-body construction is a subsequent stage.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from codes.precompute_objects.registry import task_poses
from step3_scheculer import contacts as I,passive_support as U
from step3_scheculer import initialize_cached_v19 as OLD,additive_gpu_v24 as GPU
from step3_scheculer import shared_direction_paths as P
from step3_scheculer.pair_scoring import J
from step3_scheculer.joint_prepared import scoring_sources

STAGE='dsl_shared_direction_v30'


def unit(v):
    v=np.asarray(v,float)
    if v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-12:
        raise ValueError('Expected a finite nonzero world direction')
    return v/np.linalg.norm(v)


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
    if group is None:raise ValueError('Unknown current saved pose set: '+set_id)
    out=I.OUTPUTS/name/set_id/'step3_scheculer'/STAGE
    out.mkdir(parents=True,exist_ok=True)
    existing=out/'report.json'
    if existing.is_file():
        d=I.check_report(existing)
        if d['config']!=config:raise ValueError('Existing config differs; preserve recorded result')
        return d
    searches=[Search(name,p,config['device']) for p in group['poses']]
    directions=direction_menu(searches)
    directions.sort(key=lambda d:(swept_box_proxy([s.task for s in searches],d),direction_key(d)))
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
    parser.add_argument('--max-heads',type=int,default=10);parser.add_argument('--shortlist',type=int,default=64)
    args=parser.parse_args()
    if min(args.directions,args.max_heads,args.shortlist)<1:parser.error('Budgets must be positive')
    config={k:getattr(args,k) for k in ('device','directions','max_heads','shortlist')}
    for group in args.sets:
        d=solve(args.object,group,config)
        print('DONE',group,d['contact_programs_passed'],d['shared_object_exit_world'],d['seconds'],flush=True)

if __name__=='__main__':main()
