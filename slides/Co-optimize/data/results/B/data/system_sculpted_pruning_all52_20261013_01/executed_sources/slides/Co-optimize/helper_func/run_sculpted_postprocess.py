"""Run actual compact fillets on saved bodies/feet, keeping all old results."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from types import SimpleNamespace
import argparse,json,os,shutil,subprocess,time
from run_dynamic_postprocess import ROOT,COOPT,BASE,sha,save


def worker(payload):
    import resource,traceback
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    case,options,previous,output=payload;args=SimpleNamespace(**options)
    try:
        from sculpted_postprocess_case import run_case
        result=run_case(case,args,previous)
    except Exception as error:
        result=dict(id=case['id'],pose_count=case['pose_count'],experiment=case['experiment'],passed=False,
            status='SCULPTED_FAIL',error=str(error),traceback=traceback.format_exc())
    save(Path(output),result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True)
    p.add_argument('--source-system',required=True);p.add_argument('--sets',nargs='+');p.add_argument('--jobs',type=int,default=6)
    p.add_argument('--worker-payload',type=Path);args=p.parse_args()
    if args.worker_payload:worker(json.loads(args.worker_payload.read_text()));return
    prepared=BASE/'data'/('system_'+args.source_system);old=json.loads((prepared/'previous_evaluation.json').read_text())
    previous={r['id']:r for r in old['cases']};selection=prepared/'selection.json'
    cases=[c for c in json.loads(selection.read_text())['results']if args.sets is None or c['id']in args.sets]
    assert cases and all(c['passed']and previous[c['id']]['step53_passed']for c in cases)
    if args.sets:assert len(cases)==len(args.sets)
    system=BASE/'data'/('system_'+args.experiment);system.mkdir(exist_ok=True)
    directory=BASE/'data'/args.experiment;directory.mkdir(exist_ok=False)
    for name in ['protected_originals.json','selection.json','previous_evaluation.json']:
        if system!=prepared:shutil.copy2(prepared/name,system/name)
    protected=json.loads((prepared/'protected_originals.json').read_text());sources=[]
    for name in ['helper_func','vis_func','step4.2','step5.1','step5.2','step5.3','evaluation']:
        for path in (COOPT/name).rglob('*'):
            if path.is_file()and path.suffix in ['.py','.cpp','.h']:
                relative=path.relative_to(ROOT);target=system/'executed_sources'/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                sources.append(dict(path=str(relative),sha256=sha(path),archive=str(target.relative_to(ROOT))))
    save(system/'execution.json',dict(executed_sources=sources,source_system=args.source_system,groups=[c['id']for c in cases],
        step4_rerun=False,step51_rerun=False,original_demand_count_per_pose=32768,
        original_material_not_increased=True,real_geometry_fillets=True))
    results=[];began=time.monotonic()
    def record():
        save(directory/'batch.json',dict(complete=len(results)==len(cases),group_count=len(cases),results=results,
            passed=sum(r['passed']for r in results),eligible_groups=len(cases),seconds=time.monotonic()-began,
            source_summary=str(selection.relative_to(ROOT)),step4_optimization_run=False,step51_rerun=False))
    record();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def one(case):
        prev=previous[case['id']];label=case['id'].replace('/','__');output=directory/'workers'/(label+'.json')
        options=dict(experiment=args.experiment,selection_summary=str(selection.resolve()),step51_experiment=Path(prev['step51_source']).name)
        payload=directory/'payloads'/(label+'.json');save(payload,[case,options,prev,str(output.resolve())])
        for stage in ['step5.2','step5.3','evaluation']:
            for picture in (COOPT/'output/B'/case['id']/stage).glob('*.png'):picture.unlink()
        log=directory/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('w')as stream:
            code=subprocess.call([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),
                '--experiment',args.experiment,'--source-system',args.source_system,'--worker-payload',str(payload)],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        result=json.loads(output.read_text())if output.exists()else dict(id=case['id'],pose_count=case['pose_count'],experiment=case['experiment'],passed=False,status='RUNTIME_FAIL',error='exit '+str(code))
        result['execution_log']=str(log.relative_to(ROOT));return result
    with ThreadPoolExecutor(max_workers=args.jobs)as pool:
        for future in as_completed([pool.submit(one,c)for c in cases]):
            result=future.result();results.append(result);record()
            print('SCULPTED',len(results),'/',len(cases),'PASS',sum(r['passed']for r in results),result['id'],result['status'],flush=True)
    changed=[path for path,h in protected.items()if not(ROOT/path).is_file()or sha(ROOT/path)!=h]
    code_changed=[r['path']for r in sources if sha(ROOT/r['path'])!=r['sha256']]
    save(system/'execution_verification.json',dict(passed=not changed and not code_changed,protected_files=len(protected),changed_inputs=changed,changed_sources=code_changed))
    assert not changed and not code_changed
    print('SCULPTED COMPLETE',len(cases),'PASS',sum(r['passed']for r in results),flush=True)


if __name__=='__main__':main()
