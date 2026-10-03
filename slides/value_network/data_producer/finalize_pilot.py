"""Improve every conditional label with the complete merged certified bank."""
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
from prepare import ROOT,I,prepare
from search import object_directions
from collect_states import OUT,joint_value

def main():
 start=time.monotonic();cache=OUT/'pilot20_shared';cfg=json.loads((cache/'config.json').read_text());I.check_hashes(cfg['source_inputs'])
 poses=sorted({p for _,ps in cfg['groups'] for p in ps},key=lambda p:int(p.split('_')[1]))
 problems,pools,catalogues,_=prepare('B',poses,cache/'inputs');index={p:i for i,p in enumerate(poses)}
 banks={};checks={}
 for pose in poses:
  checks[pose]=json.loads((cache/f'terminal_checks_{pose}.json').read_text())
  for worker in (cache/'groups').glob('*'):
   path=worker/f'terminal_checks_{pose}.json'
   if path.exists():
    for key,record in json.loads(path.read_text()).items():
     if record['passed'] is True:checks[pose][key]=record
  I.save(cache/f'terminal_checks_{pose}.json',checks[pose])
  # Keep only positively certified completions; never infer acceptance from group membership alone.
  banks[pose]=[tuple(g) for g in json.loads((cache/f'bank_{pose}.json').read_text())['completions']
               if checks[pose].get(','.join(map(str,g)),{}).get('passed') is True]
 total_improved=total_new=0;summaries=[]
 for name,ps in cfg['groups']:
  gs=time.monotonic();folder=OUT/name/'pilot20';ks=[index[p] for p in ps];local_pools=[pools[k] for k in ks]
  vectors=[object_directions(problems[k],catalogues[k]) for k in ks]
  angles={(a,b):(np.arccos(np.clip(vectors[a]@vectors[b].T,-1,1))/np.pi)**2 for a in range(len(ks)) for b in range(a+1,len(ks))}
  improved=new=0;stats=[]
  for sid in range(cfg['states_per_group']):
   path=folder/f'state_{sid:03d}.json';data=json.loads(path.read_text());state=data['selected_indices'];nselected=sum(map(len,state))
   available=[[g for g in banks[p] if set(s).issubset(g)] for p,s in zip(ps,state)]
   actions=[]
   for k,groups in enumerate(available):
    mapping={i:[] for i in range(len(local_pools[k]))}
    for g in groups:
     for i in g:mapping[i].append(g)
    actions.append(mapping)
   for row in data['rows']:
    if row['status'] in ['step2_rejected','no_path']:continue
    k=ps.index(row['pose']);choices=list(available);choices[k]=actions[k][row['index']]
    best=joint_value(local_pools,choices,angles,nselected,cfg['lambda_direction'])
    if best and (row['value'] is None or best['value']<row['value']-1e-12):
     was_unknown=row['value'] is None
     row.setdefault('value_before_global_bank',row['value']);row.update(best);row['status']='success_found'
     row['completion_ids']=[[local_pools[t][i]['id'] for i in g] for t,g in enumerate(best['completion_indices'])]
     row['object_exit_vectors']=[vectors[t][d].tolist() for t,d in enumerate(best['direction_ids'])]
     improved+=1;new+=was_unknown
   data['completion_certificate_pool']='merged_global'
   counts=Counter(r['status'] for r in data['rows'])
   data['summary'].update(success=counts['success_found'],unknown=counts['budget_unresolved'])
   data['postprocessing']=dict(global_certified_bank=True,code=I.hashes([Path(__file__)]))
   I.save(path,data);stats.append(data['summary'])
  with (folder/'training_records.jsonl').open('w') as f:
   for sid in range(cfg['states_per_group']):
    data=json.loads((folder/f'state_{sid:03d}.json').read_text())
    for row in data['rows']:f.write(json.dumps(dict(group=name,state_id=sid,poses=ps,selected_indices=data['selected_indices'],**row))+'\n')
  summary=json.loads((folder/'summary.json').read_text())
  summary.update(success=sum(s['success'] for s in stats),unknown=sum(s['unknown'] for s in stats),
                 global_bank_improvements=improved,new_success_labels=new,global_bank_refinement_s=time.monotonic()-gs)
  I.save(folder/'summary.json',summary);summaries.append(summary);total_improved+=improved;total_new+=new
  print('REFINED',name,improved,'improvements;',new,'new finite labels',flush=True)
 report=json.loads((cache/'summary.json').read_text());report.update(groups=summaries,total_success=sum(s['success'] for s in summaries),total_unknown=sum(s['unknown'] for s in summaries),
  postprocessing=dict(global_bank_improvements=total_improved,new_success_labels=total_new,elapsed_s=time.monotonic()-start,code=I.hashes([Path(__file__)])))
 I.save(cache/'summary.json',report)
if __name__=='__main__':main()
