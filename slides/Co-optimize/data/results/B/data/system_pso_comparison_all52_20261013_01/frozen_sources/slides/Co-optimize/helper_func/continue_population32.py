"""Continue unfinished cold cases in independent processes after pool failure."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json,os,subprocess,time,hashlib,sys,argparse

ROOT=Path('/home/yli581/Desktop/cadGrasp')
BASE=ROOT/'slides/Co-optimize/data/results/B'
ORIGINAL='population32_all52_20261012_01'
CONTINUATION='population32_all52_20261012_02'
COMBINED='population32_all52_20261012_combined'
SYSTEM=BASE/'data'/('system_'+ORIGINAL)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.writing.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=6)
    parser.add_argument('--continue-existing',action='store_true')
    parser.add_argument('--adopt-running',type=Path)
    args=parser.parse_args()
    source=json.loads((BASE/'data'/ORIGINAL/'batch.json').read_text())
    requested=json.loads((SYSTEM/'requested_sets.json').read_text())
    completed={r['id'] for r in source['results']}
    remaining=[label for label in requested['sets'] if label not in completed]
    directory=BASE/'data'/CONTINUATION;directory.mkdir(exist_ok=args.continue_existing)
    combined=BASE/'data'/COMBINED;combined.mkdir(exist_ok=args.continue_existing)
    batch=dict(complete=False,sets=requested['sets'],group_count=52,results=list(source['results']),
        population_size=32,rounds=8,warm_start=False,algorithm_changed=False,
        sources=[ORIGINAL,CONTINUATION],recovery='independent cold per-case processes',passed=sum(r['passed'] for r in source['results']))
    continuation=dict(complete=False,sets=remaining,group_count=len(remaining),results=[],passed=0)
    if args.continue_existing:
        batch=json.loads((combined/'batch.json').read_text())
        continuation=json.loads((directory/'batch.json').read_text())
        completed={r['id']for r in batch['results']}
        remaining=[label for label in requested['sets']if label not in completed]
    adopted=json.loads(args.adopt_running.read_text())if args.adopt_running else {}
    batch['execution_jobs']=args.jobs
    save(combined/'batch.json',batch);save(directory/'batch.json',continuation)
    frozen=SYSTEM/'frozen_sources/slides/Co-optimize'
    manifest=json.loads((SYSTEM/'frozen_sources.json').read_text())
    assert all(hashlib.sha256((ROOT/r['frozen']).read_bytes()).hexdigest()==r['frozen_sha256']for r in manifest['files'])
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             PYTHONDONTWRITEBYTECODE='1',COOPT_REPO_ROOT=str(ROOT))
    options=dict(output_name=CONTINUATION,iterations=8,frontier_width=32,search_policy='population32',
                 oracle_work_mode='normal',initialization_budget=0,warm_start_experiments=[],
                 warm_checkpoint_experiment=None,initialization_seeds=None,resume=None)
    program=('import sys,json,resource; '
             'resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3)); '
             'resource.setrlimit(resource.RLIMIT_CORE,(0,0)); '
             'sys.path.insert(0,sys.argv[1]); '
             'from run_oracle import worker; '
             'worker((sys.argv[2],json.loads(sys.argv[3])))')
    def one(label):
        clock=time.monotonic();log=directory/'logs'/(label.replace('/','__')+'.log');log.parent.mkdir(exist_ok=True)
        path=BASE/label/'step4/step4.2'/CONTINUATION/'run_result.json'
        if path.exists():
            code=0
        elif label in adopted:
            while not path.exists() and Path('/proc/'+str(adopted[label])).exists():time.sleep(2)
            code=0 if path.exists() else -1
        else:
            with log.open('w')as output:
                p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-c',program,str(frozen/'step4.2'),
                                    label,json.dumps(options)],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                code=p.wait()
        if code==0 and path.exists():result=json.loads(path.read_text())
        else:result=dict(id=label,passed=False,force_passed=False,base_oracle_passed=False,
                         status='runtime_failure',error='Isolated worker exited '+str(code),
                         returncode=code,seconds=time.monotonic()-clock)
        result.update(experiment=CONTINUATION,execution_log=str(log.relative_to(ROOT)),memory_limit_gib=16)
        return result
    began=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs)as pool:
        futures=[pool.submit(one,label)for label in remaining]
        for future in as_completed(futures):
            result=future.result();continuation['results'].append(result);batch['results'].append(result)
            continuation['passed']=sum(r['passed']for r in continuation['results']);batch['passed']=sum(r['passed']for r in batch['results'])
            batch['seconds']=time.monotonic()-began+source.get('seconds',0)
            save(directory/'batch.json',continuation);save(combined/'batch.json',batch)
            print('COMBINED',len(batch['results']),'/52 PASS',batch['passed'],result['id'],result['passed'],flush=True)
    frozen_changed=[r['frozen']for r in manifest['files']if hashlib.sha256((ROOT/r['frozen']).read_bytes()).hexdigest()!=r['frozen_sha256']]
    protected=json.loads((SYSTEM/'protected_originals.json').read_text())
    changed=[p for p,h in protected.items()if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    continuation.update(complete=True,seconds=time.monotonic()-began)
    batch.update(complete=True,protected_unchanged=not changed,frozen_sources_unchanged=not frozen_changed)
    save(directory/'batch.json',continuation);save(combined/'batch.json',batch)
    save(SYSTEM/'combined_verification.json',dict(complete=not changed and not frozen_changed,
        original_files_checked=len(protected),changed_original_files=changed,
        frozen_files_checked=len(manifest['files']),changed_frozen_files=frozen_changed,
        group_count=len(batch['results']),runtime_failure_count=sum(r.get('status')=='runtime_failure'for r in batch['results'])))
    print('COMBINED FINISHED',len(batch['results']),'/52 PASS',batch['passed'],flush=True)


if __name__=='__main__':main()
