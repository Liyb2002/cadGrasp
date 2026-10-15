"""Run an independent queue through the experiment's atomic case claims."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='pso_accelerated_all52_step4_20261014_01')
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--queue-start',type=int,default=0)
    parser.add_argument('--queue-stop',type=int)
    parser.add_argument('--worker-folder',default='supplemental_workers')
    parser.add_argument('--wait-for-result',action='append',default=[])
    args=parser.parse_args()
    if len(args.wait_for_result)>args.jobs:
        parser.error('at most one initial completion gate per worker is allowed')
    if args.jobs<1 or args.queue_start<0 or (args.queue_stop is not None and args.queue_stop<args.queue_start):
        parser.error('jobs must be positive and the queue range must be ordered')
    if Path(args.worker_folder).name!=args.worker_folder or not args.worker_folder.startswith('supplemental'):
        parser.error('worker-folder must be a supplemental folder name')
    base=ROOT/'slides/Co-optimize/data/results/B/data'
    directory=base/args.name;system=base/('system_'+args.name)
    batch=json.loads((directory/'batch.json').read_text())
    if any(label not in batch['sets'] for label in args.wait_for_result):
        parser.error('wait-for-result must name a case in this experiment')
    extra=system/args.worker_folder;extra.mkdir(exist_ok=True)
    def one(index,label):
        if index<len(args.wait_for_result):
            gate=directory/'workers'/(args.wait_for_result[index].replace('/','__')+'.json')
            while not gate.exists():time.sleep(2)
        safe=label.replace('/','__');result=directory/'workers'/(safe+'.json')
        if result.exists():return
        payload=extra/'payloads'/(safe+'.json');log=extra/'logs'/(safe+'.log')
        payload.parent.mkdir(exist_ok=True);log.parent.mkdir(exist_ok=True)
        payload.write_text(json.dumps(dict(id=label,root=str(ROOT),frozen=str(system/'frozen_sources'),
            options=batch['options'],result=str(result),private_output=str(system/'private_output')),indent=2)+'\n')
        began=time.monotonic()
        with log.open('w') as stream:
            code=subprocess.call([str(ROOT/'.venv/bin/python'),str(system/'execution_scripts/step4_worker.py'),str(payload)],
                cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                 PYTHONDONTWRITEBYTECODE='1',COOPT_REPO_ROOT=str(ROOT)),stdout=stream,stderr=subprocess.STDOUT)
        if code and not result.exists():
            result.parent.mkdir(exist_ok=True)
            result.write_text(json.dumps(dict(id=label,passed=False,status='runtime_failure',
                error='Worker exit '+str(code),worker_exit_code=code,seconds=time.monotonic()-began),indent=2)+'\n')
        print('SUPPLEMENT',label,code,round(time.monotonic()-began,1),flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(one,index,label)
                for index,label in enumerate(batch['sets'][args.queue_start:args.queue_stop])]):future.result()


if __name__=='__main__':main()
