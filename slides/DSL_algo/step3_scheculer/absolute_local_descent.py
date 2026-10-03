"""Local placement descent against exact foreign sweeps, not global separators.

Only proposals: complete support construction and Step5 select the result.
"""
from dataclasses import replace
import numpy as np
from step3_scheculer import operation_dsl as F,absolute_metrics as M
from step0_pose_selection.floor_points import pressure_centers


def compact(tasks,state,rounds=3):
    # Common floor permits horizontal placement changes without changing local
    # force cones, paths or contact-floor clearances.
    if not np.allclose(state.bases[:,2,:],[0,0,1],atol=1e-8):return state,[]
    seeds,owners,_=F.roots(tasks,state)
    if state.shared_count:return state,[]
    roots=[];sweeps=[];clouds=[]
    for k,(t,row,b) in enumerate(zip(tasks,state.groups,state.bases)):
        points=np.vstack([v@b for cells in seeds[k] for v in cells])
        roots.append(F.F.union([F.S.solid(F.G.hull_mesh(v@b)) for cells in seeds[k] for v in cells]))
        sweeps.append(F.F.transform(F.S.solid(F.E.sweep_mesh(t.domain.mesh,state.paths[k])),b,np.zeros(3)))
        demand=pressure_centers(t.targets/t.scale,t.domain.com)[0]
        # Demand clouds guide volume; they are not prescribed foot locations.
        clouds.append(np.vstack([points,np.c_[demand,np.zeros(len(demand))]@b]))
    offsets=state.offsets.copy();history=[];cache={}
    def loss(o):
        installed=np.vstack([v+x for v,x in zip(clouds,o)])
        supports=[(installed-x)@b.T for x,b in zip(o,state.bases)]
        cloud=np.vstack([t.domain.mesh.vertices for t in tasks]+supports)
        return float(np.prod(np.ptp(cloud,axis=0))*1e6)
    def legal(o,k):
        # Exact pairwise solid intersections; AABB pruning is implicit in
        # Manifold. Cache horizontal pair translations through the line search.
        for j in range(len(tasks)):
            if j==k:continue
            for i,h in ((k,j),(j,k)):
                delta=o[h]-o[i];key=(i,h,*np.round(delta,10))
                if key not in cache:
                    cache[key]=abs(float((sweeps[i]^roots[h].translate(delta/F.S.SCALE)).volume()))*F.S.SCALE**3<=8e-14
                if not cache[key]:return False
        return True
    best=loss(offsets)
    for scale in (.015,.008,.004):
        for _ in range(rounds):
            moved=False
            for k in range(len(tasks)):
                toward=offsets[:,:2].mean(axis=0)-offsets[k,:2]
                vectors=[np.array([x,y],float) for x,y in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))]
                if np.linalg.norm(toward)>1e-10:vectors.insert(0,toward)
                proposals=[]
                for v in vectors:
                    o=offsets.copy();o[k,:2]+=scale*v/np.linalg.norm(v)
                    value=loss(o)
                    if value<best-1e-6:proposals.append((value,o))
                for value,o in sorted(proposals,key=lambda p:p[0]):
                    if legal(o,k):
                        history.append(dict(pose=tasks[k].pose,step_mm=scale*1000,proxy_before_cm3=best,proxy_after_cm3=value))
                        offsets=o;best=value;moved=True;break
            if not moved:break
    return replace(state,offsets=offsets),history
