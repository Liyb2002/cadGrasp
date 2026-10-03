"""Iterate actual fixed-task rollouts and certified-bank labels for visited states."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from train import save

HERE=Path(__file__).parent

def main(a):
    out=HERE/'experiments';out.mkdir(exist_ok=True);extra=HERE/'on_policy_data';checkpoint=HERE/'fixed_output/best.pt';results=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
    for round_id in range(a.start_round,a.start_round+a.rounds):
        fit=out/f'feedback_{round_id:02d}';rollout=out/f'rollout_{round_id:02d}'
        with (out/f'feedback_{round_id:02d}.log').open('w') as log:
            subprocess.run([sys.executable,str(HERE/'fit_fixed.py'),'--extra',str(extra),'--resume',str(checkpoint),'--output',str(fit),'--epochs',str(a.epochs),'--lr',str(a.lr)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        checkpoint=fit/'best.pt'
        with (out/f'rollout_{round_id:02d}.log').open('w') as log:
            subprocess.run([sys.executable,str(HERE/'rollout_fixed.py'),'--checkpoint',str(checkpoint),'--output',str(rollout),'--collect',str(extra)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        f=json.loads((fit/'report.json').read_text());r=json.loads((rollout/'report.json').read_text())
        maximum=max((item.get('cost_gap',100.) for item in r['results']),default=100.)
        item=dict(round=round_id,passed_groups=r['passed_groups'],max_cost_gap=maximum,fit_metrics=f['metrics'],extra_states=len(list(extra.glob('*.json'))),checkpoint=str(checkpoint),rollout=str(rollout))
        results.append(item);save(out/'feedback_progress.json',dict(rounds=results,complete=False))
        print('ROUND',round_id,'PASS',r['passed_groups'],'/',r['groups'],'MAX GAP',maximum,'FIT',f['metrics']['mae'],'EXTRA',item['extra_states'],flush=True)
        if r['passed_groups']==r['groups'] and maximum<=a.max_gap:
            save(out/'feedback_progress.json',dict(rounds=results,complete=True,checkpoint=str(checkpoint),rollout=str(rollout)));return
    save(out/'feedback_progress.json',dict(rounds=results,complete=False,checkpoint=str(checkpoint),rollout=str(rollout)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=12);p.add_argument('--start-round',type=int,default=1);p.add_argument('--epochs',type=int,default=2000);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--max-gap',type=float,default=.05);main(p.parse_args())
