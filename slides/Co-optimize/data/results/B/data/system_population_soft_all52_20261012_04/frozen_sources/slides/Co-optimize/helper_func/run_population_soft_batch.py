"""Independent cold workers using an immutable source snapshot."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,json,os,subprocess,time

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'slides/Co-optimize/data/results/B'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix('.writing.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment');parser.add_argument('--jobs',type=int,default=12)
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--source-system');parser.add_argument('--sets',nargs='+')
    parser.add_argument('--allow-oracle-fallback',action='store_true')
    parser.add_argument('--allow-coupled-adjustment',action='store_true')
    parser.add_argument('--allow-group-seating',action='store_true')
    parser.add_argument('--continue-existing',action='store_true')
    args=parser.parse_args();system=BASE/'data'/('system_'+(args.source_system or args.experiment))
    request=json.loads((system/'requested_sets.json').read_text())
    labels=args.sets or request['sets'];directory=BASE/'data'/args.experiment;directory.mkdir(exist_ok=args.continue_existing)
    manifest=json.loads((system/'frozen_sources.json').read_text())['files']
    assert all(sha(ROOT/r['frozen'])==r['frozen_sha256']for r in manifest)
    frozen=system/'frozen_sources/slides/Co-optimize'
    options=dict(output_name=args.experiment,iterations=args.iterations,frontier_width=32,
        search_policy='population_soft',oracle_work_mode='normal',initialization_budget=0,
        warm_start_experiments=[],warm_checkpoint_experiment=None,initialization_seeds=None,
        resume=None,allow_oracle_fallback=args.allow_oracle_fallback,
        allow_coupled_adjustment=args.allow_coupled_adjustment,allow_group_seating=args.allow_group_seating)
    batch=dict(complete=False,sets=labels,group_count=len(labels),results=[],passed=0,
        population_size=32,rounds=args.iterations,warm_start=False,options=options,execution_jobs=args.jobs)
    if args.continue_existing:
        batch=json.loads((directory/'batch.json').read_text())
        previous_options=dict(batch['options']);previous_options.setdefault('allow_coupled_adjustment',False)
        previous_options.setdefault('allow_group_seating',False)
        assert previous_options==options,'Resuming requires identical policy and budget'
        batch['options']=options
        batch.update(complete=False,sets=labels,group_count=len(labels),execution_jobs=args.jobs)
    completed={r['id']for r in batch['results']}
    pending=[label for label in labels if label not in completed]
    save(directory/'batch.json',batch)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',COOPT_REPO_ROOT=str(ROOT))
    program=('import sys,json,resource; '
        'resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3)); '
        'resource.setrlimit(resource.RLIMIT_CORE,(0,0)); '
        'sys.path.insert(0,sys.argv[1]); from run_oracle import worker; '
        'worker((sys.argv[2],json.loads(sys.argv[3])))')
    def one(label):
        clock=time.monotonic();log=directory/'logs'/(label.replace('/','__')+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('w')as output:
            p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-c',program,str(frozen/'step4.2'),
                label,json.dumps(options)],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            code=p.wait()
        path=BASE/label/'step4/step4.2'/args.experiment/'run_result.json'
        if code==0 and path.exists():result=json.loads(path.read_text())
        else:result=dict(id=label,passed=False,force_passed=False,base_oracle_passed=False,
            status='runtime_failure',error='Isolated worker exited '+str(code),returncode=code,
            seconds=time.monotonic()-clock)
        result.update(experiment=args.experiment,execution_log=str(log.relative_to(ROOT)),memory_limit_gib=16)
        return result
    began=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs)as pool:
        futures=[pool.submit(one,label)for label in pending]
        for future in as_completed(futures):
            result=future.result();batch['results'].append(result)
            batch.update(passed=sum(r['passed']for r in batch['results']),seconds=time.monotonic()-began)
            save(directory/'batch.json',batch)
            print('SOFT32 BATCH',len(batch['results']),'/',len(labels),'PASS',batch['passed'],result['id'],result.get('error',''),flush=True)
    protected=json.loads((system/'protected_originals.json').read_text())
    changed=[p for p,h in protected.items()if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    frozen_changed=[r['frozen']for r in manifest if sha(ROOT/r['frozen'])!=r['frozen_sha256']]
    batch.update(complete=True,seconds=time.monotonic()-began,protected_unchanged=not changed,
        frozen_sources_unchanged=not frozen_changed)
    save(directory/'batch.json',batch)
    save(system/(args.experiment+'_verification.json'),dict(changed_originals=changed,changed_frozen=frozen_changed,
        original_files_checked=len(protected),frozen_files_checked=len(manifest)))
    print('SOFT32 FINISHED',len(labels),'PASS',batch['passed'],flush=True)


if __name__=='__main__':main()
