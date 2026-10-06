"""Frozen-algorithm evaluation on the first saved five-pose set of every other object."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'helper_func'))
import _bootstrap
from co_common import I,save


def case(args,row):
    name=row['object'];group=row['group'];out=args.out/name/group['id']
    out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists() or (out/'case.json').exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    began=time.monotonic()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    command=[sys.executable,str(Path(__file__).resolve()),'--case',name,
             '--out',str(args.out),'--iterations',str(args.iterations),
             '--max-proposals',str(args.max_proposals),'--manifest',str(args.manifest)]
    if args.b_gate:command += ['--b-gate',str(args.b_gate)]
    with (out/'run.log').open('w') as log:
        process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    report_path=out/'report.json'
    if report_path.exists():
        I.check_report(report_path);report=json.loads(report_path.read_text())
        result=dict(object=name,pose_set=group['id'],poses=group['poses'],
                    status='complete',passed=bool(report['passed']),
                    force_passed=report['force_passed'],final_counts=report['final_counts'],
                    single_component_passed=report['single_component_passed'],
                    stop_reason=report['stop_reason'],report=str(report_path),
                    returncode=process.returncode,seconds=time.monotonic()-began)
    else:
        failure_path=out/'unresolved.json'
        failure=json.loads(failure_path.read_text()) if failure_path.exists() else {}
        result=dict(object=name,pose_set=group['id'],poses=group['poses'],status=failure.get('status','unresolved'),passed=False,
                    failure=failure,returncode=process.returncode,seconds=time.monotonic()-began)
    save(out/'case.json',result)
    return result


def single(args,row):
    from prepare_five_pose_group_fast import prepare_group_fast,CommonSurfaceInfeasible
    from hybrid_fast import FastHybridSearch
    name=row['object'];group=row['group'];out=args.out/name/group['id']
    stage='prerequisites'
    try:
        path=HERE.parent.parent/'objects'/name/'pose_sets.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['pose_sets_sha256']:
            raise RuntimeError('Saved pose-set manifest changed after selection')
        directions=prepare_group_fast(name,group)
        stage='waiting_for_B_selection'
        if args.b_gate:
            while True:
                selection=json.loads(args.b_gate.read_text())
                if selection['force_exit_passed']>=28:break
                if selection['completed']==selection['total']:
                    raise RuntimeError('B compiled-field hybrid did not reach the predeclared 28/30 success threshold; other-object search not started')
                time.sleep(10)
        stage='step4.2'
        search=FastHybridSearch(name,group,out=out,directions=directions,max_proposals=args.max_proposals)
        search.additional_code += [Path(__file__),HERE/'helper_func/prepare_five_pose_group.py',HERE/'helper_func/prepare_five_pose_group_fast.py']
        search.additional_inputs += [args.manifest]
        report=search.optimize(args.iterations)
        I.check_report(out/'report.json')
        return report
    except Exception as error:
        save(out/'unresolved.json',dict(object=name,pose_set=group['id'],passed=False,
            status='common_surface_force_fail' if isinstance(error,CommonSurfaceInfeasible) else 'unresolved',
            stage=stage,error=f'{type(error).__name__}: {error}'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('generalization5_manifest.json'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--b-gate',type=Path);parser.add_argument('--case');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--max-proposals',type=int,default=1200)
    args=parser.parse_args();args.out=args.out.resolve();args.manifest=args.manifest.resolve()
    rows=json.loads(args.manifest.read_text())['cases']
    if args.case:return single(args,next(r for r in rows if r['object']==args.case))
    args.out.mkdir(parents=True,exist_ok=True)
    ledger=dict(algorithm='concurrent physics-valued sampling + short gradients; compiled guidance field',
                iterations_limit=args.iterations,proposal_budget=args.max_proposals,
                selection='first saved five-pose set per object, fixed before testing',
                total=len(rows),completed=0,passed=0,connectivity_required=False,
                b_selection_gate=str(args.b_gate) if args.b_gate else None,
                b_selection_threshold=28 if args.b_gate else None,
                full_fixture_accepted=False,manifest_sha256=I.sha256(args.manifest),results=[])
    save(args.out/'batch.json',ledger)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(case,args,row) for row in rows]
        for future in as_completed(futures):
            result=future.result();ledger['results'].append(result)
            ledger['completed']=len(ledger['results']);ledger['passed']=sum(r['passed'] for r in ledger['results'])
            save(args.out/'batch.json',ledger)
            print(result['object'],result['status'],result['passed'],round(result['seconds'],1),flush=True)
    print('GENERALIZATION',ledger['passed'],'/',ledger['total'],flush=True)

if __name__=='__main__':main()
