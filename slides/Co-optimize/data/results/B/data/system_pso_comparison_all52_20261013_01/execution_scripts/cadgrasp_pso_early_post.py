from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, sys, time
ROOT=Path('/home/yli581/Desktop/cadGrasp')
sys.path.insert(0,str(ROOT/'slides/Co-optimize/helper_func'))
import run_pso_campaign as R
SYSTEM=R.BASE/'data/system_pso_comparison_all52_20261013_01'
def one(policy,exp):
 source=R.read(R.BASE/'data'/exp/'batch.json')
 assert not source['complete']
 cases=[c for c in source['results'] if c['passed']]
 assert len(cases)==1
 case=cases[0];post=exp+'_sculpted';folder=R.BASE/'data'/post
 selection=folder/'early_selection.json'
 R.save(selection,dict(complete=True,group_count=len(cases),results=cases,passed=len(cases),scope='Completed own-campaign cases only'))
 safe=case['id'].replace('/','__');result=folder/'workers'/(safe+'.json');payload=folder/'payloads'/(safe+'.json');log=folder/'logs'/(safe+'.log')
 row=R.execute(SYSTEM/'execution_scripts/cadgrasp_pso_post_worker.py',
  (payload,dict(frozen=str(SYSTEM/'frozen_sources'),case=case,selection=str(selection),initial=exp+'_stock',aesthetic=exp+'_rounded',final=post,floor=exp+'_floor',result=str(result))),result,log)
 row['timing_context']='One sequential early Step5 worker per policy while both Step4 queues run; included separately in comparison'
 R.save(result,row)
 R.save(folder/'batch.json',dict(complete=False,group_count=52,passed=int(row['passed']),results=[row],seconds=row['process_wall_seconds'],eligible_groups=None,policy=policy,postprocess_experiment=post,source_summary=str((R.BASE/'data'/exp/'batch.json').relative_to(ROOT)),early_postprocess=True))
 print('EARLY STEP5',policy,case['id'],row['passed'],row.get('error'),flush=True)
 return row
for policy in ['differential','pso']:
 one(policy,R.POLICIES[policy])
R.verify(SYSTEM)
