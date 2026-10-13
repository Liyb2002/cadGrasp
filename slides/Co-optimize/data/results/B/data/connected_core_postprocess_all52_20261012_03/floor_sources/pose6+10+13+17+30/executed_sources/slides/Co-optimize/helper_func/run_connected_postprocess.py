"""Process each newly certified Step4 case through ground clouds, deletion, legs.

Per-worker selections are immutable even while the source batch is running.
All numerical attempts remain under distinct experiment names.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import argparse,json,os,shutil,subprocess,time
from run_dynamic_postprocess import ROOT,COOPT,BASE,load_stage,save,sha


def worker(payload):
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    case,options,output=payload
    args=SimpleNamespace(**options)
    result=dict(id=case['id'],pose_count=case['pose_count'],experiment=case['experiment'])
    try:
        stage=load_stage('step5.1')
        folder=BASE/case['id']/'step5/step5.1'/args.step51_experiment
        if (folder/'data/report.json').exists():
            record=json.loads((folder/'data/report.json').read_text())
            assert record['complete']and record['selected_step4_experiment']==case['experiment']
        else:
            manifest=Path(output).parent.parent/'floor_sources'/case['id'].replace('/','__')
            manifest.mkdir(parents=True,exist_ok=True)
            sources=stage.freeze_code(manifest)
            execution=manifest/'execution.json';save(execution,dict(case=case,executed_code=sources))
            body=stage.trimesh.load(ROOT/'objects/B/mesh.stl',force='mesh',process=False)
            row=stage.compute_case('B',case,BASE,stage.NativeInputs('B',body),stage.V.Renderer(),0.,
                dict(manifest_path=execution,executed_code=sources),args.step51_experiment)
        result['floor']=dict(passed=True,output=str(folder.relative_to(ROOT)))
        from stage5_batch import safe_worker
        for name,key in [('step5.2','pruning'),('step5.3','construction')]:
            module=load_stage(name)
            destination=BASE/case['id']/'step5'/name/args.experiment
            report=destination/'data/report.json'
            if report.exists():
                r=json.loads(report.read_text())
                assert r['passed']and r['selected_step4_experiment']==case['experiment']
                for filename,digest in r.get('artifacts',{}).items():assert sha(destination/filename)==digest
                row=dict(id=case['id'],passed=True,pose_count=case['pose_count'],output=str(destination.relative_to(ROOT)))
            else:row=safe_worker(module.worker,case,args,BASE,name)
            result[key]=row
            result.update(passed=row['passed'],status='PASS'if row['passed']else key.upper()+'_FAIL')
            save(Path(output),result)
            if not row['passed']:return
    except Exception as error:
        result.update(passed=False,status='RUNTIME_FAIL',error=str(error))
    save(Path(output),result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--source-summary',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=6)
    parser.add_argument('--candidate-budget',type=int,default=64)
    parser.add_argument('--worker-payload',type=Path)
    args=parser.parse_args()
    if args.worker_payload:worker(json.loads(args.worker_payload.read_text()));return
    system=BASE/'data'/('system_'+args.experiment)
    directory=BASE/'data'/args.experiment;directory.mkdir(exist_ok=False)
    system.mkdir(exist_ok=True)
    original=BASE/'data/system_dynamic_postprocess_all52_20261012_01/protected_originals.json'
    shutil.copy2(original,system/'protected_originals.json')
    archive=BASE/'data/system_connected_core_all52_20261012_01/previous_public_images.json'
    shutil.copy2(archive,system/'previous_public_images.json')
    sources=[]
    for name in ['helper_func','vis_func','step4.2','step5.1','step5.2','step5.3','evaluation']:
        for path in (COOPT/name).rglob('*'):
            if path.is_file()and path.suffix in ['.py','.cpp','.h']:
                relative=path.relative_to(ROOT);target=system/'executed_sources'/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                sources.append(dict(path=str(relative),sha256=sha(path),archive=str(target.relative_to(ROOT))))
    save(system/'execution.json',dict(executed_sources=sources,source_summary=str(args.source_summary),
        step4_optimization_run=False,connected_core_policy=True,original_demand_count_per_pose=32768))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    results={};submitted=set();pending={};began=time.monotonic()
    def one(case):
        label=case['id'].replace('/','__')
        selection=system/'selections'/(label+'.json');save(selection,dict(complete=True,results=[case]))
        options=dict(object='B',experiment=args.experiment,step51_experiment=args.experiment,
            selection_summary=str(selection.resolve()),reference_stage='step4',reference_experiment=args.experiment,
            source_contact_material='auto',resume_pruning_experiment=None,clean_object_interiors=False,
            candidate_budget=args.candidate_budget,prune_experiment=args.experiment,leg_profile='flat')
        output=directory/'workers'/(label+'.json');payload=directory/'payloads'/(label+'.json')
        save(payload,[case,options,str(output.resolve())]);log=directory/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
        for stage in ['step5.1','step5.2','step5.3','evaluation']:
            for picture in (COOPT/'output/B'/case['id']/stage).glob('*.png'):picture.unlink()
        with log.open('w')as stream:
            code=subprocess.call([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),
                '--experiment',args.experiment,'--source-summary',str(args.source_summary),
                '--worker-payload',str(payload)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        row=json.loads(output.read_text())if output.exists()else dict(id=case['id'],pose_count=case['pose_count'],
            experiment=case['experiment'],passed=False,status='RUNTIME_FAIL',error='Worker exited '+str(code))
        row['execution_log']=str(log.relative_to(ROOT));return row
    with ThreadPoolExecutor(max_workers=args.jobs)as pool:
        while True:
            batch=json.loads(args.source_summary.read_text());cases=batch['results']
            for case in cases:
                if case['id']in submitted:continue
                if 'pose_count'not in case:
                    case=dict(case,pose_count=len(json.loads((BASE/case['id']/'step4/step4.1/data/report.json').read_text())['poses']))
                submitted.add(case['id'])
                if not case['passed']:
                    results[case['id']]=dict(id=case['id'],pose_count=case['pose_count'],experiment=case['experiment'],
                        passed=False,status='STEP4_FAIL',skipped_reason='No certified connected Step4 contact core')
                else:pending[pool.submit(one,case)]=case['id']
            for future in list(pending):
                if future.done():
                    row=future.result();results[pending.pop(future)]=row
                    print('CONNECTED POSTPROCESS',len(results),'/',batch['group_count'],'PASS',sum(r['passed']for r in results.values()),row['id'],row['status'],flush=True)
            complete=batch['complete']and len(results)==batch['group_count']
            save(directory/'batch.json',dict(complete=complete,group_count=batch['group_count'],results=list(results.values()),
                passed=sum(r['passed']for r in results.values()),eligible_groups=sum(r['passed']for r in cases),
                seconds=time.monotonic()-began,source_summary=str(args.source_summary)))
            if complete:break
            time.sleep(2.)
    protected=json.loads((system/'protected_originals.json').read_text())
    changed=[p for p,h in protected.items()if not(ROOT/p).is_file()or sha(ROOT/p)!=h]
    changed_code=[r['path']for r in sources if sha(ROOT/r['path'])!=r['sha256']]
    save(system/'execution_verification.json',dict(passed=not changed and not changed_code,
        protected_files=len(protected),changed_inputs=changed,changed_sources=changed_code))
    print('CONNECTED POSTPROCESS FINISHED',len(results),'PASS',sum(r['passed']for r in results.values()),flush=True)


if __name__=='__main__':main()
