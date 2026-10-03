"""Sample successful insertion orders in the five training groups only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from transfer_experiment import plan,ROOT,TrainingContext
from train import save

def main():
    d=plan();ctx=TrainingContext(ROOT/'prefix_evidence');out=ROOT/'train_on_policy';out.mkdir(exist_ok=True)
    added=0;rng=np.random.default_rng(20261007)
    for name,poses in d['train']:
        root=json.loads((Path(d['state_root'])/name/'pilot20/state_000.json').read_text())
        best=min((r for r in root['rows'] if r['value'] is not None),key=lambda r:r['value'])
        heads=[(k,i) for k,g in enumerate(best['completion_indices']) for i in g]
        for order in range(24):
            state=[[] for _ in poses]
            for k,i in rng.permutation(heads):
                key=hashlib.sha256(json.dumps([poses,state]).encode()).hexdigest();path=out/f'{key}.json'
                if not path.exists():
                    record=ctx.labels(poses,[list(g) for g in state]);tmp=path.with_suffix('.tmp');save(tmp,record);tmp.replace(path);added+=1
                state[int(k)]=sorted(state[int(k)]+[int(i)])
        print('TRAIN PREFIXES',name,'ADDED SO FAR',added,flush=True)
    save(ROOT/'successful_prefix_sampling.json',dict(complete=True,groups=d['train'],orders_per_group=24,new_states=added,seed=20261007,test_groups_sampled=False))
if __name__=='__main__':main()
