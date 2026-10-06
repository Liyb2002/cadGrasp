import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_timed import *
import sys
name=sys.argv[1];group=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']==name);search=Search('B',group);parts=search.seed.decompose();parts.sort(key=lambda p:-material_volume(p));rows=[]
for k,part in enumerate(parts):
 tri,src=contact_boundary(search.mesh,S.unpack(part),search.allowed);counts=[];proof=None
 for pose,(task,T) in zip(group['poses'],search.states):
  full=supply(task,T,tri,src);mask,info=J.classify(full,task.targets);counts.append(int(mask.sum()))
  if not mask.all() and proof is None:
   bad=int(np.flatnonzero(~mask)[0]);cert=C.W.exact_separator(full,U.target(task.targets[bad]));proof=dict(pose=pose,index=bad,certificate=cert)
 row=dict(component=k,volume_cm3=material_volume(part)*1e6,counts=counts,failure_proof=proof);rows.append(row);print('SEED COMPONENT',name,k,row['volume_cm3'],counts,'exact fail',proof is not None and proof['certificate'] is not None,flush=True)
 if all(count==32768 for count in counts):break
save(search.out/'data/seed_connectivity_bound.json',dict(complete=True,pose_set=name,components=rows,seed_component_count=len(parts),some_original_component_carries_all=any(len(row['counts'])==len(group['poses']) and all(c==32768 for c in row['counts']) for row in rows)))
