"""Audit completed new Step5 cases while the remaining numerical cases run."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys
import time
from wrap_pipeline import ROOT,COOPT,read,save,sha,DEFAULT_EXPERIMENT


def binding(out):
    return {name:sha(out/name) for name in ['data/report.json','support.obj','initial_support.obj','states.json']}


def valid_audit(out):
    p=out/'data/audit_binding.json'
    if not p.exists():return False
    record=read(p)
    return (record['artifacts']==binding(out)
        and record['audit_sha256']==sha(out/'data/independent_audit.json')
        and read(out/'data/independent_audit.json')['passed'])


def invoke(directory,row,cpus):
    out=ROOT/row['output']
    if valid_audit(out):return row['id']
    payload=read(directory/'payloads'/(row['id'].replace('/','__')+'.json'))
    payload.update(action='audit',cpus=cpus)
    path=directory/'payloads'/(row['id'].replace('/','__')+'.audit.json');save(path,payload)
    before=binding(out)
    log=directory/'logs'/(row['id'].replace('/','__')+'.audit.log')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with log.open('w') as handle:
        result=subprocess.run([sys.executable,str(COOPT/'step5/wrap_frozen_worker.py'),str(path)],
            cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:raise ValueError('Independent audit failed: '+row['id'])
    if before!=binding(out):raise ValueError('Case changed during audit: '+row['id'])
    save(out/'data/audit_binding.json',dict(complete=True,passed=True,artifacts=before,
        audit_sha256=sha(out/'data/independent_audit.json')))
    return row['id']


def main():
    experiment=sys.argv[1] if len(sys.argv)>1 else DEFAULT_EXPERIMENT
    directory=COOPT/'data/results/B/data'/('step5_'+experiment)
    save(directory/'audit_scheduler.json',dict(source=__file__,sha256=sha(__file__),started=time.time()))
    seen=set();futures={};errors=[]
    cpus=sorted(os.sched_getaffinity(0));groups=[cpus[:2],cpus[2:4],cpus[4:6]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        while True:
            for path in (directory/'progress').glob('*.json'):
                row=read(path)
                if row['passed'] and row['id'] not in seen:
                    seen.add(row['id']);future=pool.submit(invoke,directory,row,groups[len(seen)%3])
                    futures[future]=row['id']
            for future,label in list(futures.items()):
                if future.done():
                    try:print('AUDIT_PASS',future.result(),flush=True)
                    except Exception as error:errors.append(dict(id=label,error=str(error)));print('AUDIT_FAILED',label,str(error),flush=True)
                    del futures[future]
            if (directory/'summary.json').exists() and read(directory/'execution.json').get('complete') and not futures:
                complete=read(directory/'summary.json')
                if len(seen)>=complete['passed']:break
            time.sleep(3)
    save(directory/'audit_scheduler_complete.json',dict(complete=True,attempted=len(seen),errors=errors))

if __name__=='__main__':main()
