from co_common import *
from scipy.optimize import linprog
import time

def fibonacci(n):
    k=np.arange(n);z=1-2*(k+.5)/n;theta=k*np.pi*(3-np.sqrt(5));return np.c_[np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z]
def projected(a,normals,lift=.01):
    dirs=np.array([a-min(a@n,0)*n+lift*n for n in normals]);return dirs/np.linalg.norm(dirs,axis=1)[:,None]
def load(g):
    base=HERE/'output/B'/g['id'];states=[state('B',p)[:2] for p in g['poses']];mesh=state('B',g['poses'][0])[2]
    z=np.load(base/'step3/step3.2/data/contacts.npz');tri=z['triangles_mesh_m'];src=z['source_faces'];normals=np.array([T[:3,:3].T@np.array([0,0,1]) for task,T in states]);fail=[]
    for p,(task,T) in zip(g['poses'],states):
        mask=np.load(base/f'step4/step4.1/data/{p}.npz')['force_mask'];bad=np.flatnonzero(~mask);fail.append(int(bad[0]) if len(bad) else 0)
    return states,mesh,tri,src,normals,fail
if __name__=='__main__':
 import sys
 g=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']==sys.argv[1]);states,mesh,tri,src,normals,fail=load(g);began=time.monotonic();best=-1
 for i,a in enumerate(fibonacci(160)):
    dirs=projected(a,normals);keep=np.max(mesh.face_normals[src]@dirs.T,axis=1)<-1e-8;t,s=tri[keep],src[keep]
    if len(t)<5:continue
    supplies=[supply(task,T,t,s) for task,T in states];score=0
    for (task,T),full,idx in zip(states,supplies,fail):
        if C.W.solve(full,U.target(task.targets[idx])) is not None:score+=1
    if score>best:best=score;print('BEST',i,score,len(t),a,flush=True)
    if score!=len(states):continue
    masks=[J.classify(full,task.targets)[0] for (task,T),full in zip(states,supplies)];counts=[int(m.sum()) for m in masks];print('FULL',i,counts,flush=True)
    if all(m.all() for m in masks):
        np.savez(HERE/'data'/f'explore_{g["id"]}.npz',directions=dirs,common=a);print('FOUND',time.monotonic()-began,flush=True);break
 else:print('NOT FOUND',best,time.monotonic()-began,flush=True)
