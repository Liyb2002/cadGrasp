"""Jointly release a separating-plane reaction face in every pose."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_tangent_faces import *

def constrained_direction(a,floor,face,lift=.0001):
    A=np.array([-floor,face]);b=np.array([-lift,0.]);candidates=[]
    if np.max(A@a-b)<=1e-12:candidates.append(a)
    for j in range(2):
        p=a-((A[j]@a-b[j])/(A[j]@A[j]))*A[j]
        if np.max(A@p-b)<1e-10:candidates.append(p)
    p=a-A.T@np.linalg.lstsq(A@A.T,A@a-b,rcond=1e-12)[0]
    if np.max(A@p-b)<1e-10:candidates.append(p)
    candidates=[p for p in candidates if np.linalg.norm(p)>1e-12]
    if not candidates:return None
    p=min(candidates,key=lambda p:np.linalg.norm(p-a));return p/np.linalg.norm(p)

def solve_joint(search):
    diagnostic=json.loads((search.base/'step4/step4.2/data/force_face_diagnostic.json').read_text());faces=[]
    for row in diagnostic['faces']:
        if row['face'] not in faces:faces.append(row['face'])
    faces=[f for f in [6,5,98,633]+faces if f in search.allowed][:24];menus=set()
    for f in faces:
        n=search.mesh.face_normals[f]
        for count in [160,512]:
            for a in fibonacci(count):
                ds=[constrained_direction(a,floor,n) for floor in search.normals]
                if any(d is None for d in ds):continue
                ds=np.array(ds);mask=np.max(search.face_normals@ds.T,axis=1)<=1e-9;key=np.packbits(mask).tobytes()
                if key in menus:continue
                menus.add(key);r=search.candidate(ds,label=f'joint reaction release face {f}')
                if r is not None:return r
        print('JOINT FACE FINISHED',f,'menus',len(menus),'proposals',search.evaluations,flush=True)
    return None
if __name__=='__main__':
    cache.install();g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+4+7+12+21+27');s=ConsistentSearch('B',g);s.warm=np.load(s.base/'step4/step4.2/data/connected_run/data/balanced_best_paths.npz')['directions'].copy();np.savez(s.out/'data/warm_paths.npz',directions=s.warm);began=time.monotonic();r=solve_joint(s)
    if r is None:save(s.out/'data/joint_reaction_unresolved.json',dict(complete=False,proposals=s.evaluations,best_counts=None if s.best is None else s.best['counts']));print('JOINT REACTION UNRESOLVED',None if s.best is None else s.best['counts'],flush=True)
    else:
        row=finish_connected(s,r,began);p=s.base/'step4/step4.2/data/report.json';report=json.loads(p.read_text());report['motion_consistent_contacts_required']=True;report['provenance']['code'].update(I.hashes([HERE/f for f in ['step4.2/step42_connected_safe.py','step4.2/step42_connected_local.py','step4.2/step42_connected_independent.py','step4.2/step42_contact_consistency.py','step4.2/step42_topology.py','step4.2/step42_topology_common.py','step4.2/step42_value_guided.py','step4.2/step42_balanced.py','helper_func/step42_geometry_cache.py','step4.2/step42_optimized.py','step4.2/step42_tangent_faces.py','step4.2/step42_joint_reaction.py']]));save(p,report);I.check_report(p);print(row,flush=True)
