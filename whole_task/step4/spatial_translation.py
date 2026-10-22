"""Compact placement proposals with all installation floor planes retained."""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from whole_task.step3 import independent_geometry as G
from whole_task.operations.state import transform


def target(model,layout,groups):
    ng=len(groups);count=3*ng
    lookup={k:g for g,group in enumerate(groups)for k in group}
    pressures=[];clouds=[]
    for i,(record,pose) in enumerate(zip(model.records,layout.poses)):
        data=model.world_data(record,pose)
        points=G.pressure_points(data['endpoints']['endpoint_need_wrench'],data['domain'].com)
        points=points[ConvexHull(points).vertices]
        pressures.append(transform(np.c_[points,np.zeros(len(points))],layout.material[i]))
        clouds.append(transform(data['stock_world'].vertices,layout.material[i]))
    A,b=[],[]
    for i,p in enumerate(pressures):
        for j,M in enumerate(layout.material):
            gi,gj=lookup[i],lookup[j]
            if gi==gj:
                continue
            n=M[:3,2]
            distance=float(((p-M[:3,3])@n).min())
            row=np.zeros(count+6)
            row[3*gi:3*gi+3]-=n;row[3*gj:3*gj+3]+=n
            A.append(row);b.append(distance)
    for g,group in enumerate(groups):
        points=np.vstack([clouds[k]for k in group])
        low,high=points.min(0),points.max(0)
        for axis in range(3):
            row=np.zeros(count+6);row[3*g+axis]=-1;row[count+axis]=1
            A.append(row);b.append(low[axis])
            row=np.zeros(count+6);row[3*g+axis]=1;row[count+3+axis]=-1
            A.append(row);b.append(-high[axis])
    objective=np.r_[np.zeros(count),-np.ones(3),np.ones(3)]
    bounds=[(None,None)]*(count+6)
    for k in range(3):bounds[k]=(0.,0.)
    result=linprog(objective,A_ub=np.asarray(A),b_ub=np.asarray(b),bounds=bounds,method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:
        return None
    q=layout.copy()
    for g,group in enumerate(groups):
        for k in group:q.material[k,:3,3]+=result.x[3*g:3*g+3]
    return q


def sweep(optimizer,rounds=5):
    changed=False
    for iteration in range(rounds):
        proposed=target(optimizer.model,optimizer.layout,optimizer.groups)
        if proposed is None:
            break
        start=optimizer.layout.copy()
        shifts=proposed.material[:,:3,3]-start.material[:,:3,3]
        if np.linalg.norm(shifts)<1e-6:
            break
        accepted=False
        for fraction in (1.,.75,.5,.25,.125,.0625):
            q=start.copy();q.material[:,:3,3]+=fraction*shifts
            if optimizer.try_operation(q,'translation',layout='floor_constrained_3D_compaction',
                    fraction=fraction,iteration=iteration):
                changed=accepted=True
                break
        if not accepted:
            break
    return changed
