"""Rerun rounded pruning and fixed legal leg growth on saved connected layouts."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from types import SimpleNamespace
import argparse,json,os,shutil,subprocess,time
from run_dynamic_postprocess import ROOT,COOPT,BASE,load_stage,save,sha


def worker(payload):
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    case,options,previous,output=payload;args=SimpleNamespace(**options)
    row=dict(id=case['id'],pose_count=case['pose_count'],experiment=case['experiment'],
        floor=dict(passed=True,output=previous['step51_source'],reused_saved_stage=True))
    from stage5_batch import safe_worker
    for stage,key in [('step5.2','pruning'),('step5.3','construction')]:
        module=load_stage(stage)
        result=safe_worker(module.worker,case,args,BASE,stage)
        row[key]=result;row.update(passed=result['passed'],status='PASS'if result['passed']else key.upper()+'_FAIL')
        save(Path(output),row)
        if not result['passed']:return


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True);p.add_argument('--source-system',required=True)
    p.add_argument('--sets',nargs='+');p.add_argument('--jobs',type=int,default=6)
    p.add_argument('--rounding-radius',type=float,default=.006)
    p.add_argument('--worker-payload',type=Path)
    args=p.parse_args()
    if args.worker_payload:worker(json.loads(args.worker_payload.read_text()));return
    prepared=BASE/'data'/('system_'+args.source_system)
    old=json.loads((prepared/'previous_evaluation.json').read_text())
    previous={r['id']:r for r in old['cases']}
    selection=prepared/'selection.json';all_cases=json.loads(selection.read_text())['results']
    cases=[c for c in all_cases if args.sets is None or c['id']in args.sets]
    assert cases and all(c['passed']and previous[c['id']]['step53_passed']for c in cases)
    if args.sets:assert len(cases)==len(args.sets)
    system=BASE/'data'/('system_'+args.experiment);system.mkdir(exist_ok=True)
    directory=BASE/'data'/args.experiment;directory.mkdir(exist_ok=False)
    protected=json.loads((prepared/'protected_originals.json').read_text())
    if system!=prepared:shutil.copy2(prepared/'protected_originals.json',system/'protected_originals.json')
    sources=[]
    for name in ['helper_func','vis_func','step4.2','step5.1','step5.2','step5.3','evaluation']:
        for path in (COOPT/name).rglob('*'):
            if path.is_file()and path.suffix in ['.py','.cpp','.h']:
                relative=path.relative_to(ROOT);target=system/'executed_sources'/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                sources.append(dict(path=str(relative),sha256=sha(path),archive=str(target.relative_to(ROOT))))
    save(system/'execution.json',dict(executed_sources=sources,source_system=args.source_system,
        groups=[c['id']for c in cases],step4_rerun=False,step51_rerun=False,rounding_radius_m=args.rounding_radius,
        original_demand_count_per_pose=32768,fixed_leg_reconstruction=True))
    results=[];began=time.monotonic()
    def record():
        save(directory/'batch.json',dict(complete=len(results)==len(cases),group_count=len(cases),results=results,
            passed=sum(r['passed']for r in results),eligible_groups=len(cases),seconds=time.monotonic()-began,
            source_summary=str(selection.relative_to(ROOT)),rounding_radius_m=args.rounding_radius,
            step4_optimization_run=False,step51_rerun=False))
    record()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def one(case):
        prev=previous[case['id']];name=case['id'].replace('/','__');output=directory/'workers'/(name+'.json')
        options=dict(object='B',experiment=args.experiment,selection_summary=str(selection.resolve()),
            step51_experiment=Path(prev['step51_source']).name,reference_stage='step4',reference_experiment=args.experiment,
            source_contact_material='auto',resume_pruning_experiment=None,clean_object_interiors=False,
            candidate_budget=64,prune_experiment=args.experiment,leg_profile='flat',pruning_style='rounded',
            rounding_radius=args.rounding_radius,shape_reference=str((ROOT/prev['step52_source']).resolve()),
            fixed_leg_reference=str((ROOT/prev['step53_source']).resolve()))
        payload=directory/'payloads'/(name+'.json');save(payload,[case,options,prev,str(output.resolve())])
        for stage in ['step5.2','step5.3','evaluation']:
            for image in (COOPT/'output/B'/case['id']/stage).glob('*.png'):image.unlink()
        log=directory/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('w')as stream:
            code=subprocess.call([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),
                '--experiment',args.experiment,'--source-system',args.source_system,'--worker-payload',str(payload)],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        result=json.loads(output.read_text())if output.exists()else dict(id=case['id'],pose_count=case['pose_count'],
            experiment=case['experiment'],passed=False,status='RUNTIME_FAIL',error='Worker exited '+str(code))
        result['execution_log']=str(log.relative_to(ROOT));return result
    with ThreadPoolExecutor(max_workers=args.jobs)as pool:
        for future in as_completed([pool.submit(one,c)for c in cases]):
            result=future.result();results.append(result);record()
            print('ROUNDED POSTPROCESS',len(results),'/',len(cases),'PASS',sum(r['passed']for r in results),result['id'],result['status'],flush=True)
    changed=[path for path,h in protected.items()if not(ROOT/path).is_file()or sha(ROOT/path)!=h]
    changed_sources=[r['path']for r in sources if sha(ROOT/r['path'])!=r['sha256']]
    save(system/'execution_verification.json',dict(passed=not changed and not changed_sources,
        protected_files=len(protected),changed_inputs=changed,changed_sources=changed_sources))
    assert not changed and not changed_sources
    print('ROUNDED COMPLETE',len(cases),'PASS',sum(r['passed']for r in results),flush=True)


if __name__=='__main__':main()
