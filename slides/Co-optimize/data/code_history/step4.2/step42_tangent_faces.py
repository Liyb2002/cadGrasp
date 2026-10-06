"""Search direction cells that release reactions identified by a failing load."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_optimized import *
from scipy.optimize import linprog

def solve(search):
    ds=search.warm.copy();keep=np.max(search.face_normals@ds.T,axis=1)<=1e-9;faces=[]
    for k,(task,T) in enumerate(search.states):
        full=supply(task,T,search.tri[keep],search.src[keep]);mask,info=J.classify(full,task.targets)
        if mask.all():continue
        bad=int(np.flatnonzero(~mask)[0]);search.indices[k].append(bad);target=U.target(task.targets[bad]);dual=linprog(-target,A_ub=full,b_ub=np.zeros(len(full)),bounds=[(-1,1)]*7,method='highs')
        if not dual.success:continue
        points=transform_points(search.tri.reshape(-1,3),T);normals=np.repeat(-task.domain.mesh.face_normals[search.src],3,axis=0);raw=np.c_[normals,np.cross(points-task.domain.com,normals)];scores=(U.heads(raw,task.scale)@dual.x).reshape(-1,3).max(1)
        seen=set()
        for j in np.argsort(-scores):
            f=int(search.src[j]);owners=np.flatnonzero(ds@search.mesh.face_normals[f]>1e-9)
            if scores[j]>1e-10 and len(owners)==1 and f not in seen:
                faces.append((float(scores[j]),f,int(owners[0])));seen.add(f)
    faces.sort(reverse=True);faces=faces[:12];menus=set()
    print('TARGET REACTION FACES',faces,flush=True)
    for score,face,k in faces:
        n=search.mesh.face_normals[face];axis=np.eye(3)[np.argmin(abs(n))];u=np.cross(n,axis);u/=np.linalg.norm(u);v=np.cross(n,u)
        others=np.delete(ds,k,axis=0);allowed=np.max(search.face_normals@others.T,axis=1)<=1e-9;planes=np.vstack([search.face_normals[allowed],search.normals[k]])
        a=planes@u;b=planes@v;roots=np.mod(np.arctan2(-a,b),2*np.pi);roots=np.unique(np.r_[roots,np.mod(roots+np.pi,2*np.pi),0.]);angles=np.unique(np.r_[roots,(roots+np.roll(roots,-1)+np.r_[np.zeros(len(roots)-1),2*np.pi])/2])
        # Prefer the closest legal cells, but include all normal-menu cells on
        # this great circle, including boundary directions.
        angles=sorted(angles,key=lambda t:-(np.cos(t)*u+np.sin(t)*v)@ds[k])
        for theta in angles:
            d=np.cos(theta)*u+np.sin(theta)*v
            if d@search.normals[k]<1e-8:continue
            dirs=ds.copy();dirs[k]=d;mask=np.max(search.face_normals@dirs.T,axis=1)<=1e-9;key=np.packbits(mask).tobytes()
            if key in menus:continue
            menus.add(key);r=search.candidate(dirs,label=f'dual reaction face {face}, blocking pose {k}',proxy=True)
            if r is not None:return r
        print('REACTION FACE FINISHED',face,'menus',len(menus),'proposals',search.evaluations,flush=True)
    return None
if __name__=='__main__':
    cache.install();g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose1+4+7+12+21+27');s=ConsistentSearch('B',g);s.warm=np.load(s.base/'step4/step4.2/data/connected_run/data/balanced_best_paths.npz')['directions'].copy();np.savez(s.out/'data/warm_paths.npz',directions=s.warm);began=time.monotonic();r=solve(s)
    if r is None:save(s.out/'data/tangent_unresolved.json',dict(complete=False,proposals=s.evaluations,best_counts=None if s.best is None else s.best['counts']));print('REACTION SEARCH UNRESOLVED',None if s.best is None else s.best['counts'],flush=True)
    else:
        row=finish_connected(s,r,began);p=s.base/'step4/step4.2/data/report.json';report=json.loads(p.read_text());report['motion_consistent_contacts_required']=True;report['provenance']['code'].update(I.hashes([HERE/f for f in ['step4.2/step42_connected_safe.py','step4.2/step42_connected_local.py','step4.2/step42_connected_independent.py','step4.2/step42_contact_consistency.py','step4.2/step42_topology.py','step4.2/step42_topology_common.py','step4.2/step42_value_guided.py','step4.2/step42_balanced.py','helper_func/step42_geometry_cache.py','step4.2/step42_optimized.py','step4.2/step42_tangent_faces.py']]));save(p,report);I.check_report(p);print(row,flush=True)
