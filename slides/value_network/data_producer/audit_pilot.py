"""Validate pilot records against stored full-load certificates and input hashes."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from prepare import ROOT,I,compatible
from collect_states import OUT

def main(group=None):
 cache=OUT/'pilot20_shared';cfg=json.loads((cache/'config.json').read_text());I.check_hashes(cfg['source_inputs']);I.check_hashes(cfg['code'])
 pools={};checks={};vectors={}
 for _,poses in cfg['groups']:
  for pose in poses:
   if pose in pools:continue
   data=json.loads((cache/'inputs'/f'paths_{pose}.json').read_text());pools[pose]=data['candidates']
   checks[pose]=json.loads((cache/f'terminal_checks_{pose}.json').read_text())
   domain=json.loads((ROOT/f'slides/baseline_algo/output/B/independent_poses/{pose}/step_1_needs/needs.json').read_text())
   rotation=np.asarray(domain['frame']['T_world_mesh'])[:3,:3]
   vectors[pose]=np.asarray(data['catalogue']['vectors'])@rotation
 counts=Counter();groups=[]
 for name,poses in cfg['groups']:
  if group and name!=group:continue
  group_checks={pose:json.loads((cache/'groups'/name/f'terminal_checks_{pose}.json').read_text()) for pose in poses}
  folder=OUT/name/'pilot20';states=json.loads((folder/'states.json').read_text())['states'];assert len(states)==cfg['states_per_group']
  assert len({tuple(tuple(g) for g in state) for state in states})==len(states)
  gc=Counter();expected_training={}
  for sid,state in enumerate(states):
   data=json.loads((folder/f'state_{sid:03d}.json').read_text());assert data['complete'] and data['selected_indices']==state
   expected={(pose,i) for pose,g in zip(poses,state) for i in range(len(pools[pose])) if i not in g}
   assert {(r['pose'],r['index']) for r in data['rows']}==expected and len(data['rows'])==len(expected)
   for row in data['rows']:
    gc[row['status']]+=1
    expected_training[(sid,row['pose'],row['index'])]=(row['value'],row['status'])
    if row['value'] is None:assert row['status']!='success_found';continue
    assert row['status']=='success_found'
    gs=row['completion_indices'];ds=row['direction_ids']
    for k,(pose,g,d) in enumerate(zip(poses,gs,ds)):
     assert set(state[k]).issubset(g) and len(g)==len(set(g)) and len(g)<=cfg['max_heads_per_pose']
     assert compatible(pools[pose],g) and all(d in pools[pose][i]['directions'] for i in g)
     certificate_source=checks if data.get('completion_certificate_pool')=='merged_global' else group_checks
     result=certificate_source[pose][','.join(map(str,g))]
     assert result['passed'] and result['sample_count']==32768 and result['covered_count']==32768
     assert result['no_uplift_in_same_reaction_solve']
     assert row['completion_ids'][k]==[pools[pose][i]['id'] for i in g]
     np.testing.assert_allclose(row['object_exit_vectors'][k],vectors[pose][d],atol=1e-12)
    k=poses.index(row['pose']);assert row['index'] in gs[k]
    v=[vectors[pose][d] for pose,d in zip(poses,ds)]
    dispersion=np.mean([(np.arccos(np.clip(np.dot(v[a],v[b]),-1,1))/np.pi)**2 for a in range(len(v)) for b in range(a+1,len(v))])
    remaining=sum(map(len,gs))-sum(map(len,state))-1
    assert remaining==row['remaining_heads'] and remaining>=0 and row['total_heads']==sum(map(len,gs))
    np.testing.assert_allclose([row['dispersion'],row['value']],[dispersion,remaining+cfg['lambda_direction']*dispersion],atol=1e-12)
   gc['states']+=1
  with (folder/'training_records.jsonl').open() as f:
   for line in f:
    record=json.loads(line);sid=record['state_id']
    assert record['group']==name and record['poses']==poses and record['selected_indices']==states[sid]
    assert expected_training.pop((sid,record['pose'],record['index']))==(record['value'],record['status'])
  assert not expected_training
  counts.update(gc);groups.append(dict(group=name,counts=dict(gc)))
 retries=[json.loads(p.read_text()) for p in cache.rglob('numerical_retries/*.json')]
 report=dict(passed=True,groups=groups,counts=dict(counts),source_inputs_unchanged=True,
             full_load_certificates_checked=True,fresh_mechanical_replay=False,
             numerical_unresolved=sum(r['status']=='numerical_unresolved' for r in retries),
             code=I.hashes([Path(__file__)]))
 I.save(cache/('audit.json' if group is None else f'audit_{group}.json'),report)
 if group:
  print(json.dumps(report['counts']));return
 lines=['# Twenty-state pilot','',f"Completed: {len(groups)} groups, {counts['states']} states, {sum(counts[s] for s in ['success_found','budget_unresolved','no_path','step2_rejected'])} state/action records.",'',
        'All finite values refer to stored successful contact completions with all 32,768 original loads accepted per pose. The record audit checks state/action retention, common directions/paths, object-frame direction costs and the head-count formula. No support solid was constructed.','',
        'Null budget-unresolved values are unknown, not proof of infeasibility. Scores are best-found feasible costs. Direction search is exact over the discovered bank for two poses and approximate for larger groups.','',
        '| Group | States | Successful labels | Unknown | No path | Step2 rejected | State compute (s) |','|---|---:|---:|---:|---:|---:|---:|']
 for item in groups:
  name=item['group'];c=item['counts'];s=json.loads((OUT/name/'pilot20/summary.json').read_text())
  lines.append(f"| [{name}](../{name}/pilot20/summary.json) | {c['states']} | {c['success_found']} | {c['budget_unresolved']} | {c['no_path']} | {c['step2_rejected']} | {s['state_compute_s']:.1f} |")
 lines+=['','Each group has `states.json`, twenty `state_NNN.json` files, and `training_records.jsonl` in its `pilot20/` directory.','', '`pose1+3copied` repeats the current pose identities of `pose1+3`, with different sampled states. All inputs come from current independent_poses; historical fixture placements are not used.']
 (cache/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report['counts']))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--group');main(p.parse_args().group)
