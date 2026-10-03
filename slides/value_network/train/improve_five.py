"""Fit only five groups; choose checkpoint using training rollouts, then test once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from train import save,digest
from transfer_experiment import ROOT,plan
HERE=Path(__file__).parent

def main(a):
    d=plan();cfg=ROOT/'train_config.json';extra=ROOT/'train_on_policy';results=[];checkpoint=None
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
    for round_id in range(a.rounds):
        fit=ROOT/f'fit_{round_id:02d}';rollout=ROOT/f'train_{round_id:02d}'
        cmd=[sys.executable,str(HERE/'fit_fixed.py'),'--data-config',str(cfg),'--extra',str(extra),
             '--output',str(fit),'--epochs',str(2500 if round_id==0 else 2000),
             '--lr',str(.001 if round_id==0 else .0003),'--seed','20261006']
        if checkpoint is not None:cmd+=['--resume',str(checkpoint)]
        with (ROOT/f'fit_{round_id:02d}.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        checkpoint=fit/'best.pt'
        with (ROOT/f'train_{round_id:02d}.log').open('w') as f:
            subprocess.run([sys.executable,str(HERE/'transfer_experiment.py'),'train-rollout','--checkpoint',str(checkpoint),'--output',str(rollout)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        fr=json.loads((fit/'report.json').read_text());rr=json.loads((rollout/'report.json').read_text())
        item=dict(round=round_id,passed_groups=rr['passed_groups'],max_cost_gap=max(r.get('cost_gap') if r.get('cost_gap') is not None else 100. for r in rr['results']),
                  extra_states=len(list(extra.glob('*.json'))),metrics=fr['metrics'],checkpoint=str(checkpoint),rollout=str(rollout))
        results.append(item);save(ROOT/'training_progress.json',dict(rounds=results,complete=False,selection='training-group rollouts only; no test evaluation'))
        print('FIVE ROUND',round_id,'PASS',item['passed_groups'],'/5','GAP',item['max_cost_gap'],'EXTRA',item['extra_states'],flush=True)
        if item['passed_groups']==5 and item['max_cost_gap']<=.05:break
    converged=item['passed_groups']==5 and item['max_cost_gap']<=.05
    save(ROOT/'training_progress.json',dict(rounds=results,complete=True,training_converged=converged,checkpoint=str(checkpoint),selection='training-group rollouts only; no test evaluation'))
    # Test is run once after checkpoint selection has ended. Never collect its states.
    checkpoint_hash=digest(checkpoint)
    with (ROOT/'heldout.log').open('w') as f:
        subprocess.run([sys.executable,str(HERE/'transfer_experiment.py'),'test','--checkpoint',str(checkpoint),'--output',str(ROOT/'heldout')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (ROOT/'original_ten_test.log').open('w') as f:
        subprocess.run([sys.executable,str(HERE/'transfer_experiment.py'),'test-original','--checkpoint',str(checkpoint),'--output',str(ROOT/'original_ten_test')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert digest(checkpoint)==checkpoint_hash
    save(ROOT/'experiment.json',dict(complete=True,scope=d['scope'],split=d,initialization='random seed 20261006; no original all-ten checkpoint weights used',
         training_converged=converged,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,checkpoint_selection='training groups only',
         frozen_report=str(ROOT/'frozen/report.json'),training_report=str(rollout/'report.json'),test_report=str(ROOT/'heldout/report.json'),original_ten_test_report=str(ROOT/'original_ten_test/report.json')))
    print('HELDOUT',json.loads((ROOT/'heldout/report.json').read_text())['passed_groups'],'/5',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=24);main(p.parse_args())
