import sys,json,hashlib
from pathlib import Path
import numpy as np
root=Path('/home/yli581/Desktop/cadGrasp')
folder=Path(sys.argv[1]);batch=json.loads((folder/'batch.json').read_text())
manifest=json.loads((folder/'runtime_sources/manifest.json').read_text())
errors=[];rows=[]
for item in batch['results']:
 path=folder/item['pose_set']/'data/report.json'
 if not path.exists():continue
 r=json.loads(path.read_text());initial=json.loads((path.parent/'initialization_input.json').read_text())
 original=root/'slides/Co-optimize/output/B'/item['pose_set']/'step4/step4.1/data/report.json'
 assert initial==json.loads(original.read_text())
 layout=np.load(path.parent/'layout.npz');d=layout['directions'];o=layout['offsets_m']
 assert np.allclose(d,r['directions']) and np.allclose(o,r['offsets_m'])
 assert np.allclose(np.linalg.norm(d,axis=1),1.) and np.linalg.norm(o[0])<1e-12
 assert np.linalg.norm(o,axis=1).max()<=r['cumulative_translation_cap_m']+1e-12
 angle=np.rad2deg(np.arccos(np.clip(np.sum(d*np.array([s['direction_fixture'] for s in initial['state_results']]),axis=1),-1,1)))
 assert angle.max()<=r['cumulative_direction_cap_degrees']+1e-9
 assert not r['baseline_used'] and not r['computed_packing_used']
 assert r['force_exit_passed']==all(a==b for a,b in zip(r['final_counts'],r['load_count_per_pose']))
 for k,h in r['provenance']['inputs'].items():
  assert hashlib.sha256((root/k).read_bytes()).hexdigest()==h,k
 for k,h in r['provenance']['code'].items():
  if k.startswith('slides/Co-optimize/'):
   rel=Path(k).relative_to('slides/Co-optimize');snap=folder/'runtime_sources'/rel
  else:snap=root/k
  if not snap.exists():snap=root/k
  assert hashlib.sha256(snap.read_bytes()).hexdigest()==h,k
 for round in r['iterations']:
  winners=[t for t in round['trials'] if t.get('accepted')]
  assert len(winners)==int(round['accepted'])
  for t in winners:
   assert t['protected'] and t['worst_loss_after']<t['worst_loss_before']-max(1e-10,t['worst_loss_before']*1e-5)
   assert t['footprint']['maximum_expansion_fraction']<=r['footprint_expansion_cap']
 rows.append(dict(pose_set=item['pose_set'],passed=r['force_exit_passed'],maximum_direction_change_degrees=float(angle.max()),
  maximum_translation_mm=float(np.linalg.norm(o,axis=1).max()*1000),footprint_expansion_percent=r['footprint']['maximum_expansion_fraction']*100,
  accepted_steps=r['accepted_sampling_steps'],counts=r['final_counts'],exact_evaluations=r['exact_evaluations'],exact_cache_hits=r['exact_cache_hits']))
print(json.dumps(rows,indent=2));print('AUDITED',len(rows),'of',batch['total'])
if len(rows)==batch['total']:
 (folder/'audit.json').write_text(json.dumps(dict(passed=True,checked_groups=len(rows),results=rows),indent=2)+'\n')
