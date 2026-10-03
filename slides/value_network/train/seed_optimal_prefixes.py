"""Label subsets of a certified best-found completion, covering all insertion orders."""
import argparse
import itertools
import json
from pathlib import Path
from bank_teacher import Context
from train import DATA

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',default='pose6+8+10+19');p.add_argument('--output',type=Path,default=Path(__file__).parent/'on_policy_data');a=p.parse_args()
    ctx=Context(Path(__file__).parent/'policy_checks',mechanics=False)
    root=json.loads((DATA/a.group/'pilot20/state_000.json').read_text())
    best=min((r for r in root['rows'] if r['value'] is not None),key=lambda r:r['value'])
    poses=root['poses'];heads=[(k,i) for k,g in enumerate(best['completion_indices']) for i in g]
    for bits in range(2**len(heads)-1):
        state=[[] for _ in poses]
        for bit,(k,i) in enumerate(heads):
            if bits & (1<<bit):state[k].append(i)
        for g in state:g.sort()
        ctx.save_label(poses,state,a.output)
        if bits%32==0:print('PREFIX',a.group,bits,'/',2**len(heads)-1,flush=True)
    print('DONE',a.group,len(heads),'heads',flush=True)
if __name__=='__main__':main()
