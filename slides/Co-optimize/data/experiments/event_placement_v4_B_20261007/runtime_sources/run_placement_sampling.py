"""Run two-tool layout sampling on the seven original Step4.1 B failures."""
import sys,argparse,json,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import HERE,read_selected_pose_groups,save
from event_placement_sampling import EventPlacementSearch as PlacementSearch
from concurrent.futures import ProcessPoolExecutor,as_completed


def run_case(name,group,out,iterations,candidates,finalists,seed):
    folder=Path(out)/group['id']
    if (folder/'data/report.json').exists():raise RuntimeError('preserve existing result; choose fresh output')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.log').open('w',buffering=1) as log:
        import contextlib
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            report=PlacementSearch(name,group,folder,seed=seed).run(iterations,candidates,finalists)
    return dict(pose_set=group['id'],status='complete',force_exit_passed=report['force_exit_passed'],
        baseline_used=report['baseline_used'],baseline_passed=report['baseline_passed'],final_counts=report['final_counts'],
        maximum_translation_m=report['maximum_translation_m'],seconds=report['seconds'],stop_reason=report['stop_reason'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',default='B');parser.add_argument('--sets',nargs='+')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--candidates',type=int,default=24);parser.add_argument('--finalists',type=int,default=2)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    groups=[]
    for g in read_selected_pose_groups(args.object):
        initial=json.loads((HERE/'output'/args.object/g['id']/'step4/step4.1/data/report.json').read_text())
        if not all(r['force_passed'] for r in initial['state_results']):groups.append(g)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets]
    rows={g['id']:dict(pose_set=g['id'],status='pending') for g in groups}
    def ledger():save(args.out/'batch.json',dict(algorithm='placement-sampling',total=len(rows),completed=sum(r['status']!='pending' for r in rows.values()),
        force_exit_passed=sum(r.get('force_exit_passed',False) for r in rows.values()),
        sampling_solved=sum(r.get('force_exit_passed',False) and not r.get('baseline_used',False) for r in rows.values()),
        baseline_used=sum(r.get('baseline_used',False) for r in rows.values()),results=list(rows.values())))
    ledger()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,args.object,g,args.out,args.iterations,args.candidates,args.finalists,args.seed):g['id'] for g in groups}
        for future in as_completed(futures):
            key=futures[future]
            try:rows[key]=future.result()
            except Exception as error:rows[key]=dict(pose_set=key,status='error',error=str(error))
            ledger();print(json.dumps(rows[key]),flush=True)
    print('BATCH COMPLETE',flush=True)

if __name__=='__main__':main()
