"""Independent greedy initialization: six heads, three fixed areas, top10 sampling.

CUDA evaluates reusable LP certificates against ALL stored loads. CPU solves LPs
and continuous geometry. No descent, replacement, force subsampling or +z rule.
"""
import argparse,contextlib,json,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import numpy as np
from scipy.optimize import linprog
from step3_scheculer import contacts as I,passive_support as U
from step3_scheculer.pair_scoring import J,C
from step3_scheculer.run_dsl import saved_task
from step3_scheculer.pair_geometry import PairGeometry,horizontal_catalogue
from step3_scheculer.random_search import chain_seed
from step3_scheculer.joint_prepared import geometry_sources,scoring_sources
from step3_scheculer.run_independent import draw_pose
from step3_scheculer.run_sequential import numerical_recovery
from step2_local_support import withdrawal as W,circles as P

SCHEMA='independent_gpu_initialization_v12'
AREAS=(.01,.02,.005)


def top10(rows,base):
    ranked=sorted(rows,key=lambda r:(-r['covered'],r['id']))[:10]
    gains=np.array([r['covered']-base for r in ranked],float)
    if np.any(gains<0):raise RuntimeError('Added head reduced coverage')
    probability=gains/gains.sum() if gains.sum() else np.ones(len(ranked))/len(ranked) if ranked else np.empty(0)
    return ranked,probability


from codes.precompute_objects.head_directions import catalogue


class GPUClassifier:
    """Same primal/dual certificates as J.classify; GPU only batches target tests."""
    def __init__(self,targets,device='cuda'):
        import torch
        torch.set_num_threads(1)
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        self.torch=torch;self.device=device
        self.targets=U.target(targets,7)
        self.rhs=torch.as_tensor(self.targets,dtype=torch.float64,device=device)
        self.calls=0;self.lp_count=0
        self.info=dict(device=device,dtype='float64',gpu=torch.cuda.get_device_name() if device=='cuda' else None,
            method='Original LP with CUDA batched primal and dual certificates',load_subsampling=False)

    def classify(self,full,known=None):
        t=self.torch;b=self.targets
        accepted=np.zeros(len(b),bool) if known is None else known.copy()
        pending=~accepted;self.calls+=1;lp=0
        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)
        while pending.any():
            remaining=np.flatnonzero(pending);index=int(remaining[0])
            witness=C.W.solve(full,b[index]);lp+=1;pending[index]=False
            ids=t.as_tensor(remaining,device=self.device);rhs=self.rhs[ids]
            if witness is not None:
                accepted[index]=True;basis=full[witness['indices']]
                if not len(basis):continue
                pinv=np.linalg.pinv(basis)
                coefficients=rhs@t.as_tensor(pinv,dtype=t.float64,device=self.device)
                residual=(coefficients@t.as_tensor(basis,dtype=t.float64,device=self.device)-rhs).abs().amax(dim=1)
                # Conservative margin avoids GPU roundoff at zero coefficients.
                certified=((coefficients>=1e-12).all(dim=1)&(residual<=C.FEASIBILITY_TOL-1e-12)).cpu().numpy()
                accepted[remaining[certified]]=True;pending[remaining[certified]]=False
            else:
                result=linprog(-b[index],A_ub=full,b_ub=np.zeros(len(full)),bounds=[(-1,1)]*full.shape[1],method='highs',options=options)
                lp+=1
                if not result.success or np.linalg.norm(result.x)<1e-12:continue
                normal=result.x/np.linalg.norm(result.x)
                if np.max(full@normal)>1e-12:continue
                certified=(rhs@t.as_tensor(normal,dtype=t.float64,device=self.device)>1e-8+1e-12).cpu().numpy()
                pending[remaining[certified]]=False
        self.lp_count+=lp
        return accepted,dict(equilibrium_and_separator_lps=lp,covered=int(accepted.sum()))


class Search:
    def __init__(self,pose,out,count=200,chains=30,device='cuda'):
        self.pose=pose;self.out=Path(out)/pose/'step3_scheculer';self.out.mkdir(parents=True,exist_ok=True)
        from step3_scheculer.pair_tasks import read_task
        self.task=read_task('B',pose)
        self.geometry=PairGeometry([self.task],count=count,initialize_candidates=False)
        if self.geometry.precomputed:
            cached=I.ROOT/'objects'/self.task.domain.data['object']/'poses'/pose/'step2'
            self.task.inputs.extend(cached/f for f in ('report.json','geometry.npz','entries.json.gz'))
            I.save(self.out/'precomputed_heads.json',dict(folder=str(cached.relative_to(I.ROOT)),report_sha256=I.sha256(cached/'report.json'),candidate_geometry_recomputed=False))
        self.catalogue=catalogue(self.task.domain.mesh)
        self.geometry.catalogues=[self.catalogue]
        self.geometry.analyzers=[W.Analyzer(self.task.domain.mesh,self.geometry.depth,self.catalogue)]
        self.backend=GPUClassifier(self.task.targets,device)
        self.count=count;self.chains=chains;self.force_cache={};self.records=[];self.pools={};self.unresolved={}
        self.initial_mask=self.classify([])[0]

    def classify(self,entries,known=None):
        key=tuple(sorted(e['contact']['candidate_id'] for e in entries))
        if key in self.unresolved:raise RuntimeError(self.unresolved[key])
        if key not in self.force_cache:
            try:self.force_cache[key]=self.backend.classify(self.task.supply([e['contact'] for e in entries]),known)
            except RuntimeError as error:
                self.unresolved[key]=str(error)
                I.save(self.out/'unresolved_proposals.json',dict(complete=True,force_constraints_relaxed=False,proposals=[dict(ids=list(k),error=v) for k,v in self.unresolved.items()]))
                raise
        return self.force_cache[key]

    def pool(self,fraction):
        if fraction in self.pools:return self.pools[fraction]
        if self.geometry.precomputed:
            from codes.precompute_objects.head_cache import pool
            rows=pool(self.geometry.precomputed,fraction,self.catalogue)
            self.pools[fraction]=rows
            return rows
        mesh=self.task.domain.mesh;rows=[]
        for index,(center,face) in enumerate(zip(self.geometry.centers,self.geometry.faces)):
            patch,fit=self.geometry.surface.fit_area(center,int(face),fraction*mesh.area)
            entry=self.geometry.make(index,fit['radius_m'],patch,target_area=fraction*mesh.area)
            entry['contact']['candidate_id']=f'{self.pose}_A{fraction*100:g}_C{index+1:03d}'
            entry['area_fit']=fit;rows.append(entry)
            if (index+1)%40==0:print(self.pose,'area',fraction,'candidates',index+1,'legal',sum(e['valid'] for e in rows),flush=True)
        self.pools[fraction]=rows
        I.save_contacts(self.out/f'candidates_{fraction:g}.npz',[e['contact'] for e in rows])
        I.save(self.out/f'candidates_{fraction:g}.json',dict(complete=True,fraction=fraction,records=[dict(id=e['contact']['candidate_id'],valid=e['valid'],reason=e['reason'],area_fraction=e['area_fraction']) for e in rows]))
        return rows

    def trajectory(self,index,fraction):
        rng=np.random.default_rng(chain_seed(20261003,index));entries=[];rounds=[]
        directions=set(range(len(self.catalogue['vectors'])));components=set(self.geometry.paths.labels.tolist())
        mask=self.initial_mask.copy()
        for step in range(6):
            if len(entries)>=1 and mask.all():break
            rows=[];began=time.monotonic()
            for i,e in enumerate(self.pool(fraction)):
                if not e['valid'] or any(e['contact']['candidate_id']==q['contact']['candidate_id'] for q in entries):continue
                if any(np.linalg.norm(e['contact']['center_m']-q['contact']['center_m'])<self.geometry.scale*1e-6 for q in entries):continue
                ds=directions&set(e['directions'][0]);cs=components&set(e['path_components'])
                if not ds or not cs:continue
                try:candidate,_=self.classify(entries+[e],mask)
                except RuntimeError:continue  # no certificate: exclude proposal, not a proof of infeasibility
                rows.append(dict(index=i,id=e['contact']['candidate_id'],covered=int(candidate.sum())))
            ranked,probability=top10(rows,int(mask.sum()))
            if not ranked:break
            draw=float(rng.random());choice=min(int(np.searchsorted(np.cumsum(probability),draw,side='right')),len(ranked)-1)
            picked=ranked[choice];e=self.pool(fraction)[picked['index']];entries.append(e)
            directions&=set(e['directions'][0]);components&=set(e['path_components']);mask,_=self.classify(entries)
            rounds.append(dict(step=step+1,selected_id=picked['id'],covered=int(mask.sum()),top10=ranked,probabilities=probability.tolist(),random_draw=draw,seconds=time.monotonic()-began))
            print(self.pose,'trajectory',index,'area',fraction,'heads',len(entries),'covered',int(mask.sum()),flush=True)
            I.save(self.out/'progress.json',dict(complete=False,trajectory=index,area_fraction=fraction,rounds=rounds))
        path_id=min(directions) if entries and directions else None
        cells=[h for e in entries for h in self.geometry.analyzers[0].heads(e['contact'])]
        exit_check=self.geometry.analyzers[0].test(cells,np.asarray(self.catalogue['vectors'][path_id])) if path_id is not None else dict(clear=False)
        passed=bool(entries and mask.all() and components and exit_check['clear'])
        result=dict(trajectory=index,area_fraction=fraction,heads=len(entries),covered=int(mask.sum()),sample_count=len(mask),passed=passed,
            status='all_loads_and_local_geometry_passed' if passed else 'finite_search_incomplete',rounds=rounds,
            exit_direction_id=path_id,object_exit_direction_world=(-np.asarray(self.catalogue['vectors'][path_id])).tolist() if path_id is not None else None,
            exit_check=exit_check,common_path_components=sorted(components),selected_ids=[e['contact']['candidate_id'] for e in entries])
        folder=self.out/f'trajectory_{index:02d}';folder.mkdir(exist_ok=True)
        I.save_contacts(folder/'contacts.npz',[e['contact'] for e in entries]);np.savez_compressed(folder/'coverage.npz',mask=mask)
        I.save(folder/'report.json',result)
        return entries,result

    def run(self):
        designs=[];done=set();began=time.monotonic()
        for index in range(self.chains):
            fraction=AREAS[index%len(AREAS)]
            if fraction in done:continue
            entries,result=self.trajectory(index,fraction);self.records.append(result);designs.append(entries)
            if result['passed']:done.add(fraction)
        winner=min(range(len(self.records)),key=lambda i:(not self.records[i]['passed'],-self.records[i]['covered'],i))
        entries=designs[winner];result=self.records[winner];contacts=[e['contact'] for e in entries]
        # Original CPU classifier is the independent final load acceptance.
        cpu,info=J.classify(self.task.supply(contacts),self.task.targets)
        np.testing.assert_array_equal(cpu,self.classify(entries)[0])
        for e in entries:
            if abs(e['area_fraction']/result['area_fraction']-1)>P.AREA_REL_TOL+1e-12:raise RuntimeError('Area fit changed')
        final=self.out/f'final_contacts_{self.pose}.npz';I.save_contacts(final,contacts)
        report=dict(schema=SCHEMA,complete=True,object='B',poses=[self.pose],independent=True,shared_heads=False,cross_pose_constraints=False,
            complete_fixture_verified=False,max_heads_per_pose=6,minimum_heads_required=1,area_fractions=list(AREAS),chain_budget=self.chains,
            chains_run=len(self.records),stop_policy='Stop each area family after first passing trajectory; retain ten attempts per family',
            numerically_unresolved_proposal_count=len(self.unresolved),candidate_count_per_area=self.count,top_k=10,load_count=len(cpu),force_subsampling=False,forced_upward_exit=False,
            passive_support_constraint=U.description(),cpu_full_load_replay=info,backend=dict(self.backend.info,calls=self.backend.calls,lp_count=self.backend.lp_count),
            result=result,trajectory_results=self.records,seconds=time.monotonic()-began,
            heads=[dict(id=c['candidate_id'],area_fraction=float(c['triangle_areas_m2'].sum()/self.task.domain.mesh.area),normal_depth_m=self.geometry.depth) for c in contacts],
            provenance=dict(inputs=I.hashes(self.task.inputs),code=I.hashes([Path(__file__),Path(U.__file__),Path(J.__file__),Path(C.__file__),Path(__file__).with_name('run_sequential.py'),Path(__file__).with_name('strict_lp_retry.py')]+geometry_sources()+scoring_sources())),
            artifacts={final.name:I.sha256(final)})
        I.save(self.out/'schedule.json',report);I.check_report(self.out/'schedule.json')
        # Native object/heads gallery, independent of current multi-pose results.
        display=dict(result=dict(covered_counts=[result['covered']],passed=result['passed']))
        draw_pose(self.task,contacts,display,self.out.parent/'heads.png')
        I.save(self.out/'progress.json',dict(complete=True,passed=result['passed']))
        return report


def solve(args):
    pose,out,count,chains,device=args;folder=Path(out)/pose;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'pipeline.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        start=time.monotonic()
        try:
            search=Search(pose,out,count,chains,device)
            recoveries={}
            with numerical_recovery(search.out/'numerical_retries',recoveries):
                report=search.run()
            return dict(pose=pose,passed=report['result']['passed'],heads=report['result']['heads'],covered=report['result']['covered'],area_fraction=report['result']['area_fraction'],chains_run=report['chains_run'],seconds=time.monotonic()-start)
        except Exception as error:
            traceback.print_exc();return dict(pose=pose,passed=False,error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-start)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--poses',nargs='+',default=list(__import__('step1.registry', fromlist=['task_poses']).task_poses('B')))
    parser.add_argument('--candidates',type=int,default=200);parser.add_argument('--chains',type=int,default=30);parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.chains<3 or args.chains%3:raise ValueError('Equal three-area budgets require chains multiple of three')
    out=I.OUTPUTS/'B'/'independent_poses_gpu_v12';out.mkdir(exist_ok=True);batch=dict(complete=False,results=[],config=vars(args))
    I.save(out/'batch.json',batch)
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(solve,(p,out,args.candidates,args.chains,args.device)) for p in args.poses]):
            row=future.result();batch['results'].append(row);I.save(out/'batch.json',batch);print('POSE COMPLETE',row,flush=True)
    batch['results'].sort(key=lambda r:int(r['pose'].split('_')[1]));batch.update(complete=True,passed_count=sum(r['passed'] for r in batch['results']))
    I.save(out/'batch.json',batch)
    lines=['# Independent GPU initialization','', 'Six heads maximum; 0.5%, 1%, 2% fixed-area families; top10 gain-proportional greedy sampling; 30 trajectory budget; all original loads.','',
        '| Pose | Passed | Heads | Area per head | Covered loads | Chains run | Time (s) |','|---|---|---:|---:|---:|---:|---:|']
    for r in batch['results']:lines.append(f"| {r['pose']} | {r['passed']} | {r.get('heads','—')} | {r.get('area_fraction','—')} | {r.get('covered','—')} | {r.get('chains_run','—')} | {r['seconds']:.1f} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n');print('TOTAL',batch['passed_count'],'/',len(args.poses),flush=True)
    if any('error' in r for r in batch['results']):raise SystemExit(2)

if __name__=='__main__':main()
