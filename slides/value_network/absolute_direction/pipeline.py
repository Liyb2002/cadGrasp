"""Run the current world-direction workflow, render and collect real Step5 values."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import experiment as E
from step5_evaluate.evaluate import evaluate

def passed(name):
    p=E.OUT/name/'comparison.json'
    if not p.exists() or not E.json.loads(p.read_text()).get('passed'):return False
    try:E.I.check_report(E.OUT/name/'step4/data/growing_support/report.json')
    except (RuntimeError,FileNotFoundError):return False
    return True

def call(script,names):
    if not names:return
    subprocess.run([sys.executable,str(E.HERE/script),'--groups',*names],check=True,
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',default=None);args=p.parse_args()
    names=args.groups or [name for name,_ in E.specs()]
    call('stable_contact_recovery.py',[n for n in names if not passed(n)])
    call('carve_guard_recovery.py',[n for n in names if not passed(n)])
    call('raw_sweep_carving.py',[n for n in names if not passed(n)])
    call('contact_bridge_recovery.py',[n for n in names if not passed(n)])
    rows=[]
    for name in names:
        if passed(name):
            result=evaluate(E.OUT/name);rows.append(dict(group=name,passed=True,volume_cm3=result['metrics']['object_and_support_poses']['box_volume_cm3']))
        else:rows.append(dict(group=name,passed=False,finite_search_failed=True))
    E.save(E.OUT/'progress.json',dict(complete=True,results=rows))
    call('render.py',[n for n in names if passed(n)])
    subprocess.run([sys.executable,str(E.HERE/'review.py')],check=True)
if __name__=='__main__':main()
