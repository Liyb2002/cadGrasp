"""Fill released numerical slots with unclaimed cases in longest-first order."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='pso_accelerated_all52_step4_20261014_01')
    parser.add_argument('--gates',required=True,type=Path)
    args=parser.parse_args();base=ROOT/'slides/Co-optimize/data/results/B/data'
    directory=base/args.name;system=base/('system_'+args.name)
    batch=json.loads((directory/'batch.json').read_text());gates=json.loads(args.gates.read_text())
    work=system/'supplemental_dynamic_workers';work.mkdir(exist_ok=True)
    for sub in ['logs','payloads','reservations']:(work/sub).mkdir(exist_ok=True)
    def worker(gate):
        if gate:
            while not (directory/'workers'/(gate.replace('/','__')+'.json')).exists():time.sleep(1)
        while True:
            selected=None
            with (work/'dispatch.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if all((directory/'workers'/(s.replace('/','__')+'.json')).exists() for s in batch['sets']):return
                for label in batch['sets']:
                    safe=label.replace('/','__');result=directory/'workers'/(safe+'.json')
                    claim=system/'claims'/safe;reservation=work/'reservations'/safe
                    if result.exists() or claim.exists() or reservation.exists():continue
                    reservation.mkdir();selected=label,safe,result,claim,reservation;break
            if selected is None:time.sleep(1);continue
            label,safe,result,claim,reservation=selected
            payload=work/'payloads'/(safe+'.json')
            payload.write_text(json.dumps(dict(id=label,root=str(ROOT),frozen=str(system/'frozen_sources'),
                options=batch['options'],result=str(result),private_output=str(system/'private_output')),indent=2)+'\n')
            began=time.monotonic();owned=False;claim_lost=False
            with (work/'logs'/(safe+'.log')).open('w') as log:
                child=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(system/'execution_scripts/step4_worker.py'),str(payload)],
                    cwd=ROOT,env=dict(os.environ,COOPT_REPO_ROOT=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None and not (claim/'pid').exists():time.sleep(.02)
                if (claim/'pid').exists():
                    owned=int((claim/'pid').read_text())==child.pid
                    claim_lost=not owned
                    if child.poll() is None and claim_lost:
                        # This process is still in the atomic-claim wait loop;
                        # the other owner alone can enter the numerical worker.
                        child.terminate()
                code=child.wait()
            if code and not claim_lost and not result.exists():
                value=dict(id=label,passed=False,status='runtime_failure',worker_exit_code=code,
                    error='Worker exit '+str(code),seconds=time.monotonic()-began,
                    checkpoint_retained=True,physical_infeasibility_established=False)
                temp=result.with_suffix('.dispatch-writing.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(result)
            shutil.rmtree(reservation)
            print('DYNAMIC',label,'owned='+str(owned),code,round(time.monotonic()-began,1),flush=True)
    with ThreadPoolExecutor(max_workers=len(gates)) as pool:
        list(pool.map(worker,gates))


if __name__=='__main__':main()
