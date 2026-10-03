"""Find compact certified seeds by pruning direction/component-compatible supersets."""
import argparse
import json
import numpy as np
from prepare import prepare,ROOT,I
from search import TerminalOracle,prune,install_recorded_recovery

def main():
 p=argparse.ArgumentParser();p.add_argument('--pose',default='pose_2');a=p.parse_args()
 out=ROOT/'slides/value_network/data/B/pilot20_shared';work=out/f'bootstrap_{a.pose}';work.mkdir(exist_ok=True)
 install_recorded_recovery(work)
 problems,pools,_,inputs=prepare('B',[a.pose],out/'inputs');pool=pools[0]
 oracle=TerminalOracle(problems[0],pool,work/'terminal_checks.json')
 groups=set()
 for e in pool:
  if not e['valid']:continue
  for d in e['directions']:
   for component in e['components']:
    groups.add(tuple(i for i,x in enumerate(pool) if x['valid'] and d in x['directions'] and component in x['components']))
 traces=[];found=set();rng=np.random.default_rng(2103)
 for n,g in enumerate(sorted(groups,key=lambda g:(-len(g),g))):
  passed=oracle.check(g)
  if passed:
   for _ in range(8):
    reduced=prune(oracle,g,None,rng)
    if len(reduced)<=6:found.add(reduced)
  traces.append(dict(indices=list(g),passed=passed))
  oracle.save();I.save(work/'result.json',dict(complete=False,completions=[list(g) for g in sorted(found)],maximal_sets=traces,source_inputs=inputs))
  print('SUPERSET',a.pose,n+1,'/',len(groups),'heads',len(g),'pass',passed,'compact seeds',len(found),flush=True)
 I.save(work/'result.json',dict(complete=True,completions=[list(g) for g in sorted(found)],maximal_sets=traces,source_inputs=inputs))
if __name__=='__main__':main()
