"""Resume immutable V12 search with byte-identical frozen old task snapshots."""
import argparse,contextlib,json,multiprocessing,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from step3_scheculer import contacts as I,initialize_gpu_v12 as V
from step1.needs import ContinuousNeeds,demand
from step3_scheculer.pair_scoring import TaskProblem,J
from step3_scheculer.run_sequential import numerical_recovery
from step3_scheculer.joint_prepared import geometry_sources,scoring_sources
from step2_local_support import withdrawal as W,circles as P

OUT=I.OUTPUTS/'B'/'independent_poses_gpu_v12'


def frozen_task(group,pose):
    folder=OUT/pose/'input';source=folder/'needs.json';samples=folder/'samples.json';snapshot=folder/'setup.npz'
    domain=ContinuousNeeds.read(source);raw=json.loads(samples.read_text());provenance=domain.data['provenance']
    assert domain.data['pose_id']==pose and domain.data['object']=='B'
    assert I.sha256(source)==raw['provenance']['physical_domain_sha256']
    assert I.sha256(snapshot)==provenance['setup_snapshot_sha256']
    with np.load(snapshot) as z:
        np.testing.assert_array_equal(z['T_world_mesh'],domain.data['frame']['T_world_mesh'])
        np.testing.assert_array_equal(z['com_m'],domain.com)
        floor=z['floor_contact_m'].copy()
    targets=np.asarray(raw['need_wrench']);assert len(targets)==32768
    np.testing.assert_allclose(targets,demand(raw['pt_m'],raw['force_push_mg'],domain.com,domain.gravity),atol=1e-13,rtol=0)
    scale=np.r_[np.ones(3),np.ones(3)/domain.mesh.extents.max()]
    task=TaskProblem(pose,domain,floor,np.ascontiguousarray(targets*scale),scale)
    task.inputs=[source,samples,snapshot];task.random_sample_count=len(targets)
    return task


def migrate(path,task):
    d=json.loads(path.read_text());old=d['provenance']['inputs'];new={}
    for key,h in old.items():
        q=I.ROOT/key
        if not q.is_file() or I.sha256(q)!=h:
            options=[p for p in task.inputs if I.sha256(p)==h]
            if len(options)!=1:raise ValueError('Input migration lacks an identical original: '+key)
            key=str(options[0].resolve().relative_to(I.ROOT))
        new[key]=h
    d['input_path_migration']=dict(geometry_changed=False,source_byte_hashes_unchanged=True,original_input_paths=old)
    d['provenance']['inputs']=new;d['provenance']['code'].update(I.hashes([Path(__file__)]))
    I.save(path,d);I.check_report(path);return d


def finish_interrupted(pose,task):
    out=OUT/pose/'step3_scheculer';final=out/f'final_contacts_{pose}.npz'
    reports=sorted(out.glob('trajectory_*/report.json'))
    if not final.exists() or not reports:return None
    contacts=I.read_contacts(final);records=[json.loads(p.read_text()) for p in reports]
    selected=[r for r,p in zip(records,reports) if I.sha256(p.parent/'contacts.npz')==I.sha256(final)]
    if len(selected)!=1:raise ValueError('Cannot identify interrupted winning trajectory')
    r=selected[0];mask,info=J.classify(task.supply(contacts),task.targets)
    backend=V.GPUClassifier(task.targets);gpu,_=backend.classify(task.supply(contacts));np.testing.assert_array_equal(gpu,mask)
    assert int(mask.sum())==r['covered']
    if r['passed']:assert mask.all()
    depth=float(task.domain.mesh.extents.max())*P.DEPTH_FRACTION
    cat=V.catalogue(task.domain.mesh);analyzer=W.Analyzer(task.domain.mesh,depth,cat)
    ids=r['exit_direction_id'];checks=[]
    for c in contacts:
        assert not np.intersect1d(c['source_faces'],task.domain.work_ids).size
        assert c['triangles_m'][:,:,2].min()>=.002-1e-9
        area=float(c['triangle_areas_m2'].sum()/task.domain.mesh.area)
        assert abs(area/r['area_fraction']-1)<=P.AREA_REL_TOL+1e-12
        checks.append(dict(id=c['candidate_id'],area_fraction=area,normal_depth_m=depth))
    if r['passed']:
        check=analyzer.test([h for c in contacts for h in analyzer.heads(c)],np.asarray(cat['vectors'][ids]))
        assert check['clear'] and r['common_path_components']
    unresolved=out/'unresolved_proposals.json';n=len(json.loads(unresolved.read_text())['proposals']) if unresolved.exists() else 0
    d=dict(schema=V.SCHEMA,complete=True,object='B',poses=[pose],independent=True,shared_heads=False,cross_pose_constraints=False,
        complete_fixture_verified=False,max_heads_per_pose=6,minimum_heads_required=1,area_fractions=list(V.AREAS),chain_budget=30,chains_run=len(records),
        stop_policy='Stop each area family after first passing trajectory; retain ten attempts per family',candidate_count_per_area=200,top_k=10,
        load_count=len(mask),force_subsampling=False,forced_upward_exit=False,passive_support_constraint=V.U.description(),cpu_full_load_replay=info,
        backend=dict(backend.info,calls=backend.calls,lp_count=backend.lp_count,interrupted_search_gpu_calls_not_recorded=True),
        numerically_unresolved_proposal_count=n,result=r,trajectory_results=records,seconds=sum(s['seconds'] for t in records for s in t['rounds']),
        timing_scope='Recovered search-round timings; candidate preparation and original process startup are excluded',
        recovered_interrupted_run=True,heads=checks,
        provenance=dict(inputs=I.hashes(task.inputs),code=I.hashes([Path(__file__),Path(V.__file__)]+geometry_sources()+scoring_sources())),
        artifacts={final.name:I.sha256(final)})
    I.save(out/'schedule.json',d);I.check_report(out/'schedule.json')
    V.draw_pose(task,contacts,dict(result=dict(covered_counts=[r['covered']],passed=r['passed'])),OUT/pose/'heads.png')
    I.save(out/'progress.json',dict(complete=True,passed=r['passed']));return d


def solve(pose):
    folder=OUT/pose;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'resume.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        began=time.monotonic()
        try:
            V.saved_task=frozen_task;task=frozen_task(None,pose);out=folder/'step3_scheculer';path=out/'schedule.json'
            recoveries={}
            with numerical_recovery(out/'numerical_retries',recoveries):
                if path.exists():d=migrate(path,task)
                else:
                    d=finish_interrupted(pose,task)
                    if d is None:
                        d=V.Search(pose,OUT,200,30,'cuda').run()
                        d['provenance']['code'].update(I.hashes([Path(__file__)]));d['frozen_original_inputs']=True
                        I.save(path,d);I.check_report(path)
            r=d['result']
            return dict(pose=pose,passed=r['passed'],heads=r['heads'],covered=r['covered'],area_fraction=r['area_fraction'],chains_run=d['chains_run'],seconds=d['seconds'],recovery_wall_seconds=time.monotonic()-began)
        except Exception as error:
            traceback.print_exc();return dict(pose=pose,passed=False,error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-began)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=6);args=parser.parse_args()
    poses=['pose_20']+[f'pose_{i}' for i in range(1,20)]
    # Fail before launching workers if any frozen input is not the exact original.
    for pose in poses:frozen_task(None,pose)
    batch=dict(complete=False,poses=poses,results=[],search='Immutable V12, input recovery adapter V13');I.save(OUT/'batch_all.json',batch)
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(solve,pose) for pose in poses]):
            row=future.result();batch['results'].append(row);I.save(OUT/'batch_all.json',batch);print('POSE COMPLETE',row,flush=True)
    batch['results'].sort(key=lambda r:int(r['pose'].split('_')[1]));batch.update(complete=True,passed_count=sum(r['passed'] for r in batch['results']),wall_seconds=time.monotonic()-began)
    I.save(OUT/'batch_all.json',batch);print('TOTAL',batch['passed_count'],'/20',flush=True)
    if any('error' in r for r in batch['results']):raise SystemExit(2)

if __name__=='__main__':main()
