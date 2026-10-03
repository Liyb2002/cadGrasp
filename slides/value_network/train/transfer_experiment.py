"""Frozen-checkpoint transfer and strictly train-group-only data collection."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
COPY=HERE.parent/'baseline_algo'
sys.path.insert(0,str(COPY))
from step3_scheculer.value_runtime import RuntimeContext
import rollout_fixed
from bank_teacher import Context
from train import save,digest,DATA

ROOT=HERE/'transfer'
NEW_TRAIN=[[1,7],[2,4,12],[3,6,9,13],[5,8,10,15,17],[1,2,7,12,19]]
NEW_TEST=[[4,6],[8,13],[1,9,15],[2,5,10,17],[3,7,8,12,19]]

def groups(numbers):return [('pose'+'+'.join(map(str,ns)),[f'pose_{n}' for n in ns]) for ns in numbers]

def plan():
    ROOT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((DATA/'pilot20_shared/config.json').read_text());old={tuple(ps) for _,ps in cfg['groups']}
    train,test=groups(NEW_TRAIN),groups(NEW_TEST)
    assert len(set(tuple(ps) for _,ps in train+test))==10
    assert not old & {tuple(ps) for _,ps in train+test}
    assert {p for _,ps in train for p in ps}=={p for _,ps in cfg['groups'] for p in ps}
    d=dict(train=train,test=test,original=cfg['groups'],states_per_group=20,
           state_root=str(HERE.parent/'data/B/transfer10'),
           scope='new combinations of already represented pose/head identities; no unseen pose geometry',
           split_policy='fixed before frozen-model evaluation; no test states/rollouts/labels used for training or checkpoint selection',
           original_checkpoint_sha256=digest(HERE/'current/best.pt'))
    save(ROOT/'plan.json',d);return d

class TrainingContext(RuntimeContext):
    def __init__(self,output):
        super().__init__(output)
        self.banks={p:[tuple(g) for g in json.loads((DATA/'pilot20_shared'/f'bank_{p}.json').read_text())['completions']] for p in self.poses}
        self.allowed={tuple(ps) for _,ps in groups(NEW_TRAIN)}
    def save_label(self,poses,state,folder):
        if tuple(poses) not in self.allowed:raise RuntimeError('Test-group collection is forbidden')
        return Context.save_label(self,poses,state,folder)

def run(checkpoint,output,selected,collect=None):
    rollout_fixed.Context=TrainingContext if collect else RuntimeContext
    args=SimpleNamespace(checkpoint=str(checkpoint),output=str(output),checks=ROOT/'mechanics',
                         collect=collect,groups=selected,state_root=HERE.parent/'data/B/transfer10')
    return rollout_fixed.run(args)

def main(a):
    d=plan()
    if a.mode=='frozen':
        report=run(HERE/'current/best.pt',ROOT/'frozen',d['train']+d['test'])
        report.update(experiment='frozen original ten-group network on ten unseen combinations',split=d,
                      checkpoint_unchanged=digest(HERE/'current/best.pt')==d['original_checkpoint_sha256'])
        save(ROOT/'frozen/report.json',report)
    elif a.mode=='train-rollout':
        assert a.checkpoint
        run(a.checkpoint,Path(a.output),d['train'],ROOT/'train_on_policy')
    elif a.mode in ('test','test-original'):
        assert a.checkpoint
        run(a.checkpoint,Path(a.output),d['test'] if a.mode=='test' else d['original'])
    elif a.mode=='plan':print(json.dumps(d,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['frozen','train-rollout','test','test-original','plan']);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path);main(p.parse_args())
