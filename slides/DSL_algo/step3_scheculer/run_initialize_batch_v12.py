"""Run or resume saved independent initializations without overwriting other poses."""
import argparse,json,multiprocessing,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from step3_scheculer import contacts as I
from step3_scheculer.initialize_gpu_v12 import solve,SCHEMA

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--poses',nargs='+',default=list(__import__('step1.registry', fromlist=['task_poses']).task_poses('B')))
    parser.add_argument('--jobs',type=int,default=4);parser.add_argument('--ledger',default='batch_all.json');args=parser.parse_args()
    out=I.OUTPUTS/'B'/'independent_poses_gpu_v12';out.mkdir(exist_ok=True)
    rows=[];pending=[]
    for pose in args.poses:
        p=out/pose/'step3_scheculer/schedule.json'
        if p.is_file():
            try:
                d=I.check_report(p)
                if d['schema']!=SCHEMA:raise ValueError('wrong schema')
                r=d['result'];rows.append(dict(pose=pose,passed=r['passed'],heads=r['heads'],covered=r['covered'],area_fraction=r['area_fraction'],chains_run=d['chains_run'],seconds=d['seconds'],resumed=True));continue
            except (RuntimeError,ValueError,AssertionError):pass
        pending.append(pose)
    began=time.monotonic();batch=dict(complete=False,results=rows,poses=args.poses)
    I.save(out/args.ledger,batch)
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(solve,(pose,out,200,30,'cuda')) for pose in pending]):
            row=future.result();rows.append(row);print('POSE COMPLETE',row,flush=True);I.save(out/args.ledger,batch)
    rows.sort(key=lambda r:int(r['pose'].split('_')[1]));batch.update(complete=True,passed_count=sum(r['passed'] for r in rows),wall_seconds=time.monotonic()-began)
    I.save(out/args.ledger,batch);print('TOTAL',batch['passed_count'],'/',len(rows),flush=True)
    if any('error' in r for r in rows):raise SystemExit(2)

if __name__=='__main__':main()
