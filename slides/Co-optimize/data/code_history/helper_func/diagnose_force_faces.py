"""Identify which withdrawal directions exclude reactions for a failing load."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_timed import *
from fractions import Fraction
g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+4+7+12+21+27');s=Search('B',g);ds=np.load(s.out/'data/connected_run/data/balanced_best_paths.npz')['directions'];keep=np.max(s.face_normals@ds.T,axis=1)<=1e-9;task,T=s.states[1];full=supply(task,T,s.tri[keep],s.src[keep]);mask,info=J.classify(full,task.targets);bad=int(np.flatnonzero(~mask)[0]);target=U.target(task.targets[bad]);dual=linprog(-target,A_ub=full,b_ub=np.zeros(len(full)),bounds=[(-1,1)]*7,method='highs');points=transform_points(s.tri.reshape(-1,3),T);normals=np.repeat(-task.domain.mesh.face_normals[s.src],3,axis=0);raw=np.c_[normals,np.cross(points-task.domain.com,normals)];scores=(U.heads(raw,task.scale)@dual.x).reshape(-1,3).max(1);order=np.argsort(-scores);rows=[]
for k in order[:40]:
 if scores[k]<1e-10:continue
 face=int(s.src[k]);n=s.mesh.face_normals[face];dots=ds@n;owners=np.flatnonzero(dots>1e-9).tolist();row=dict(face=face,score=float(scores[k]),blocking_poses=[g['poses'][j] for j in owners],blocking_indices=owners,dots=dots.tolist(),normal=n.tolist());rows.append(row);print(row,flush=True)
save(s.out/'data/force_face_diagnostic.json',dict(pose='pose_4',failing_original_load=bad,current_counts=int(mask.sum()),directions=ds.tolist(),faces=rows))
