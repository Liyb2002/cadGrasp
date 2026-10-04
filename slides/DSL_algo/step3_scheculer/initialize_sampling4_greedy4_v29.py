"""Independent top10 greedy search over immutable Step2 head caches.

No pose sampling, contact fitting, clipping, roadmap generation or collision
rechecks. All original loads and the shared no-uplift constraint remain active.
"""
import argparse,contextlib,json,multiprocessing,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from codes.precompute_objects import head_cache as HC
from codes.precompute_objects.heads.surface import FLOOR_CLEARANCE_M
from codes.precompute_objects.registry import active_objects,task_poses
from step3_scheculer import contacts as I,passive_support as U,floor_support as FLOOR
from step3_scheculer.initialize_current_v16 import GPUClassifier,current_task,top10
from step3_scheculer import initialize_current_v16 as V16
from step3_scheculer.random_search import chain_seed
from step3_scheculer.run_sequential import numerical_recovery as original_recovery
from step3_scheculer import strict_lp_retry
from step3_scheculer.pair_scoring import J,C
from step3_scheculer.joint_prepared import scoring_sources

SCHEMA='sampling4_greedy4_v29'
STAGE='independent_poses_sampling4_greedy4_v29'
AREAS=(.01,.02,.005)


@contextlib.contextmanager
def numerical_recovery(folder,records):
    old_lp,old_nnls=strict_lp_retry.linprog,strict_lp_retry.nnls
    old_wrench_lp=C.W.linprog
    def limited_lp(*args,**kwargs):
        kwargs['options']={**kwargs.get('options',{}),'time_limit':1.}
        return old_lp(*args,**kwargs)
    def limited_nnls(*args,**kwargs):
        kwargs['maxiter']=min(kwargs.get('maxiter',1000),1000)
        return old_nnls(*args,**kwargs)
    strict_lp_retry.linprog,strict_lp_retry.nnls=limited_lp,limited_nnls
    C.W.linprog=limited_lp
    try:
        with original_recovery(folder,records):yield
    finally:
        strict_lp_retry.linprog,strict_lp_retry.nnls=old_lp,old_nnls
        C.W.linprog=old_wrench_lp


def compatibility(entries,catalogue):
    directions=set(range(len(catalogue['vectors'])));components=None
    for e in entries:
        if not e['valid']:return set(),set()
        directions&=set(e['directions'][0]);ports=set(e['path_components'])
        components=ports if components is None else components&ports
    return directions,components or set()


class Search:
    def __init__(self,name,pose,chains=30,device='cuda'):
        self.name=name;self.pose=pose;self.chains=chains;self.out=I.OUTPUTS/name/STAGE/pose/'step3_scheculer'
        self.out.mkdir(parents=True,exist_ok=True);began=time.monotonic()
        self.task=current_task(name,pose);folder=HC.ROOT/'objects'/name/'poses'/pose
        self.cache=HC.load(folder,200,strict=True)
        if self.cache is None:raise ValueError('Missing validated precomputed Step2 heads')
        assert self.cache['report']['object']==name and self.cache['report']['pose']==pose
        self.catalogue=self.cache['catalogue'];self.pools=self.cache['pools'];self.scale=float(self.task.domain.mesh.extents.max())
        self.depth=self.cache['report']['depth_m'];self.backend=GPUClassifier(self.task.targets,device)
        self.floor=U.floor(FLOOR.columns(self.task.floor,self.task.domain.com),self.task.scale)
        self.columns={}
        for pool in self.pools.values():
            for e in pool:
                c=e['contact'];np.testing.assert_array_equal(c['wrench_com_m'],self.task.domain.com)
                self.columns[c['candidate_id']]=U.heads(c['wrench_generators'],self.task.scale)
        self.force_cache={};self.unresolved={};self.records=[];self.inputs=self.task.inputs+[folder/'step2'/p for p in ('report.json','geometry.npz','entries.json.gz')]
        source=I.OUTPUTS/name/'independent_poses_cached_v17'/pose/'step3_scheculer/schedule.json'
        self.original=I.check_report(source)
        self.original_folder=source.parent
        self.inputs.append(source)
        self.reused_prefix_seconds=0.
        self.base=self.classify([])[0];self.setup_seconds=time.monotonic()-began

    def classify(self,entries,known=None):
        key=tuple(sorted(e['contact']['candidate_id'] for e in entries))
        if key in self.unresolved:raise RuntimeError(self.unresolved[key])
        if key not in self.force_cache:
            full=I.merge_columns(self.floor,*[self.columns[k] for k in key])
            try:self.force_cache[key]=self.backend.classify(full,known)
            except (RuntimeError,np.linalg.LinAlgError) as error:
                self.unresolved[key]=str(error);raise RuntimeError(str(error)) from error
        return self.force_cache[key]

    def trajectory(self,index):
        previous=I.OUTPUTS/self.name/'independent_poses_sampling4_greedy4_v28'/self.pose/'step3_scheculer'/f'trajectory_{index:02d}'
        saved=previous/'report.json'
        if saved.is_file():
            result=json.loads(saved.read_text())
            pool=self.pools[result['area_fraction']]
            by_id={e['contact']['candidate_id']:e for e in pool}
            entries=[by_id[k] for k in result['selected_ids']]
            with np.load(previous/'coverage.npz') as data:mask=data['mask'].copy()
            assert len(mask)==len(self.task.targets) and int(mask.sum())==result['covered']
            assert result['heads']==len(entries) and len(entries)<=8
            self.force_cache[tuple(sorted(result['selected_ids']))]=(mask,dict(reused_v28_trajectory=True))
            self.inputs += [saved,previous/'coverage.npz']
            print(self.name,self.pose,'reuse completed v28 trajectory',index,flush=True)
            return entries,result
        area=AREAS[index%3];pool=self.pools[area];rng=np.random.default_rng(chain_seed(20261003,index))
        entries=[];rounds=[];mask=self.base.copy();directions=set(range(len(self.catalogue['vectors'])))
        components=set(self.cache['arrays']['roadmap_labels'].tolist());start=time.monotonic()
        old=next((r for r in self.original['trajectory_results'] if r['trajectory']==index),None)
        if old is not None:
            by_id={e['contact']['candidate_id']:e for e in pool}
            prefix=old['rounds'][:4]
            entries=[by_id[r['selected_id']] for r in prefix]
            rounds=[dict(r,selection_mode='top10_sampling',reused_sampling_prefix=True) for r in prefix]
            for _ in prefix:rng.random()
            if len(old['selected_ids'])<=4:
                source=self.original_folder/f'trajectory_{index:02d}'/'coverage.npz'
                with np.load(source) as data:mask=data['mask'].copy()
                assert len(mask)==len(self.task.targets) and int(mask.sum())==old['covered']
                self.inputs.append(source)
                self.force_cache[tuple(sorted(old['selected_ids']))]=(mask,dict(reused_prefix=True))
            else:mask,_=self.classify(entries)
            for e in entries:
                directions &= set(e['directions'][0]);components &= set(e['path_components'])
            self.reused_prefix_seconds += sum(r['seconds'] for r in prefix)
        for step in range(len(entries),8):
            if entries and mask.all():break
            before=int(mask.sum());rows=[];began=time.monotonic()
            for i,e in enumerate(pool):
                if not e['valid'] or any(np.linalg.norm(e['contact']['center_m']-v['contact']['center_m'])<self.scale*1e-6 for v in entries):continue
                if not (directions&set(e['directions'][0]) and components&set(e['path_components'])):continue
                try:after,_=self.classify(entries+[e],mask)
                except RuntimeError:continue
                rows.append(dict(index=i,id=e['contact']['candidate_id'],covered=int(after.sum())))
            ranked,probabilities=top10(rows,before)
            if not ranked:break
            if step<4:
                draw=float(rng.random());choice=min(int(np.searchsorted(np.cumsum(probabilities),draw,side='right')),len(ranked)-1)
                selection_mode='top10_sampling'
            else:
                draw=None;choice=0;selection_mode='maximum_coverage_greedy'

            e=pool[ranked[choice]['index']];entries.append(e);mask,_=self.classify(entries)
            directions&=set(e['directions'][0]);components&=set(e['path_components'])
            rounds.append(dict(selection_mode=selection_mode,step=step+1,selected_id=e['contact']['candidate_id'],covered=int(mask.sum()),top10=ranked,probabilities=probabilities.tolist(),random_draw=draw,seconds=time.monotonic()-began))
            I.save(self.out/'progress.json',dict(complete=False,trajectory=index,area_fraction=area,heads=len(entries),covered=int(mask.sum())))
            print(self.name,self.pose,'trajectory',index,'area',area,'heads',len(entries),'covered',int(mask.sum()),flush=True)
        passed=bool(entries and mask.all() and directions and components)
        direction=min(directions) if entries and directions else None
        result=dict(trajectory=index,area_fraction=area,heads=len(entries),passed=passed,covered=int(mask.sum()),sample_count=len(mask),
            selected_ids=[e['contact']['candidate_id'] for e in entries],common_direction_ids=sorted(directions),common_path_components=sorted(components),
            exit_direction_id=direction,object_exit_world=(-np.asarray(self.catalogue['vectors'][direction])).tolist() if direction is not None else None,
            local_exit_evidence='Intersection of cached continuously checked individual full-head rays; union has the same rigid translation',
            status='all_loads_and_cached_local_geometry_passed' if passed else 'finite_search_incomplete',rounds=rounds,seconds=time.monotonic()-start)
        folder=self.out/f'trajectory_{index:02d}';folder.mkdir(exist_ok=True)
        I.save_contacts(folder/'contacts.npz',[e['contact'] for e in entries]);np.savez_compressed(folder/'coverage.npz',mask=mask)
        I.save(folder/'report.json',result);return entries,result

    def run(self):
        began=time.monotonic();designs=[]
        for index in range(self.chains):
            entries,result=self.trajectory(index);designs.append(entries);self.records.append(result)
            if result['passed']:break # initialization needs one feasible solution, not every area family
        winner=min(range(len(self.records)),key=lambda i:(not self.records[i]['passed'],-self.records[i]['covered'],i))
        entries=designs[winner];result=self.records[winner];contacts=[e['contact'] for e in entries]
        verify_start=time.monotonic()
        # Reconstruct actual force generators from saved real triangles for the
        # final CPU check; cached moments alone never certify acceptance.
        fresh=[]
        for c in contacts:
            points=c['triangles_m'].reshape(-1,3);normals=np.repeat(-self.task.domain.mesh.face_normals[c['source_faces']],3,axis=0)
            generated=np.c_[normals,np.cross(points-self.task.domain.com,normals)]
            np.testing.assert_allclose(generated,c['wrench_generators'],atol=1e-12,rtol=0)
            fresh.append({k:v for k,v in c.items() if k not in ('wrench_generators','wrench_com_m')})
        mask,info=J.classify(self.task.supply(fresh),self.task.targets)
        np.testing.assert_array_equal(mask,self.classify(entries)[0])
        ds,cs=compatibility(entries,self.catalogue)
        assert ds==set(result['common_direction_ids']) if entries else True
        assert cs==set(result['common_path_components']) if entries else True
        if result['passed']:assert mask.all() and ds and cs
        for e in entries:
            c=e['contact'];assert e['local_clearance']['valid']
            assert not np.intersect1d(c['source_faces'],self.task.domain.work_ids).size
            assert c['triangles_m'][:,:,2].min()>=FLOOR_CLEARANCE_M-self.scale*1e-10
            assert abs(e['area_fraction']/result['area_fraction']-1)<=1e-4+1e-12
        final=self.out/f'final_contacts_{self.pose}.npz';I.save_contacts(final,contacts)
        np.savez_compressed(self.out/'coverage.npz',mask=mask)
        if self.unresolved:I.save(self.out/'unresolved_proposals.json',dict(complete=True,tolerance_relaxed=False,proposals=[dict(ids=list(k),error=v) for k,v in self.unresolved.items()]))
        verify_seconds=time.monotonic()-verify_start
        report=dict(complete=True,schema=SCHEMA,object=self.name,poses=[self.pose],passed=result['passed'],independent=True,
            full_support_constructed=False,step2_recomputed=False,cached_geometry_only=True,max_heads=8,sampling_heads=4,greedy_heads=4,numerical_retry_lp_time_limit_seconds=1.,reused_sampling_prefix_seconds=self.reused_prefix_seconds,top_k=10,area_fractions=list(AREAS),
            chain_budget=self.chains,chains_run=len(self.records),stop_policy='Stop at first passing trajectory; otherwise exhaust the thirty-trajectory budget',
            load_count=len(mask),load_subsampling=False,common_upward_exit_required=False,passive_support_constraint=U.description(),
            result=result,trajectory_results=self.records,backend=dict(self.backend.info,calls=self.backend.calls,lp_count=self.backend.lp_count),
            cpu_final_load_check=info,actual_triangle_force_generators_checked=True,numerically_unresolved_proposals=len(self.unresolved),
            timings=dict(setup_seconds=self.setup_seconds,search_seconds=sum(r['seconds'] for r in self.records),final_load_check_seconds=verify_seconds,
                         total_seconds=self.setup_seconds+time.monotonic()-began),
            provenance=dict(inputs=I.hashes(self.inputs),code=I.hashes([Path(__file__),Path(V16.__file__)]+scoring_sources())),
            artifacts={p.name:I.sha256(p) for p in (final,self.out/'coverage.npz')})
        I.save(self.out/'schedule.json',report);I.check_report(self.out/'schedule.json')
        I.save(self.out/'progress.json',dict(complete=True,passed=result['passed']));return report


def solve(args):
    name,pose,chains,device=args;folder=I.OUTPUTS/name/STAGE/pose;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'pipeline.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        began=time.monotonic()
        try:
            recoveries={}
            with numerical_recovery(folder/'step3_scheculer/numerical_retries',recoveries):
                search=Search(name,pose,chains,device);d=search.run()
            r=d['result'];return dict(object=name,pose=pose,passed=r['passed'],heads=r['heads'],covered=r['covered'],area_fraction=r['area_fraction'],chains_run=d['chains_run'],seconds=time.monotonic()-began,report=str((search.out/'schedule.json').relative_to(I.ROOT)))
        except Exception as error:
            traceback.print_exc();return dict(object=name,pose=pose,passed=False,error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-began)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--objects',nargs='+',default=list(active_objects()))
    parser.add_argument('--poses',nargs='+');parser.add_argument('--jobs',type=int,default=6)
    parser.add_argument('--chains',type=int,default=30);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.chains<3 or args.chains%3:raise ValueError('Three area families require a multiple-of-three trajectory budget')
    batches={};cases=[]
    for name in args.objects:
        poses=args.poses or list(task_poses(name));out=I.OUTPUTS/name/STAGE;out.mkdir(parents=True,exist_ok=True)
        b=dict(complete=False,object=name,poses=poses,config=vars(args),results=[]);batches[name]=(out,b)
        for pose in poses:
            if pose not in task_poses(name):raise ValueError('Unknown current pose')
            path=out/pose/'step3_scheculer/schedule.json'
            if path.is_file():
                try:
                    d=I.check_report(path)
                    if d['schema']!=SCHEMA or d['chain_budget']!=args.chains:raise ValueError('different run')
                    r=d['result'];b['results'].append(dict(object=name,pose=pose,passed=r['passed'],heads=r['heads'],covered=r['covered'],area_fraction=r['area_fraction'],chains_run=d['chains_run'],seconds=d['timings']['total_seconds'],resumed=True));continue
                except (ValueError,RuntimeError,AssertionError):pass
            cases.append((name,pose,args.chains,args.device))
        I.save(out/'batch.json',b)
    cases.sort(key=lambda r:(int(r[1].split('_')[1]),args.objects.index(r[0])))
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(solve,r) for r in cases]):
            row=future.result();out,b=batches[row['object']];b['results'].append(row);I.save(out/'batch.json',b);print('POSE COMPLETE',row,flush=True)
    for name,(out,b) in batches.items():
        b['results'].sort(key=lambda r:int(r['pose'].split('_')[1]));b.update(complete=True,passed_count=sum(r['passed'] for r in b['results']),shared_wall_seconds=time.monotonic()-began)
        I.save(out/'batch.json',b);print('TOTAL',name,b['passed_count'],'/',len(b['results']),flush=True)
    if any('error' in r for _,b in batches.values() for r in b['results']):raise SystemExit(2)

if __name__=='__main__':main()
