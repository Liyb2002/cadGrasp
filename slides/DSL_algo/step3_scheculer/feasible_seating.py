"""Restore independent placement feasibility without changing local heads/paths."""
from dataclasses import replace
import numpy as np
from scipy.optimize import linprog
from step3_scheculer import exit_options as E
from step4_connect_support import boxed_support as B,build_coupled_saddle as S
from step0_pose_selection.floor_points import pressure_centers


def restore(tasks,state,root_builder,attempt=0,iterations=60,ground_parts=None):
    """LP seating with exact collision tests and disjunctive separation proposals.

    Hyperplanes are proposal restrictions, not a claim that their rejection proves
    the original geometry infeasible. Local contact geometry and paths stay fixed.
    """
    n=len(tasks);matrix=[];rhs=[]
    seeds,owners,_=root_builder(tasks,state)
    clouds=[];root_solids=[];root_points=[];sweeps=[];sweep_points=[]
    for k,(task,row,b,o) in enumerate(zip(tasks,state.groups,state.bases,state.offsets)):
        demand=pressure_centers(task.targets/task.scale,task.domain.com)[0]
        points=np.vstack([v@b for cells in seeds[k] for v in cells])
        foot=None if ground_parts is None else B.transform(ground_parts[k],b,np.zeros(3))
        if foot is not None:points=np.vstack([points,S.unpack(foot).vertices])
        clouds.append(np.vstack([points,np.c_[demand,np.zeros(len(demand))]@b]))
        root_points.append(points)
        root_solids.append(B.union([S.solid(B.G.hull_mesh(v@b)) for cells in seeds[k] for v in cells]+([] if foot is None else [foot])))
        sweep=B.transform(S.solid(E.sweep_mesh(task.domain.mesh,state.paths[k])),b,np.zeros(3))
        sweeps.append(sweep);sweep_points.append(S.unpack(sweep).vertices)
    for k,points in enumerate(clouds):
        for j,b in enumerate(state.bases):
            if j==k:continue
            normal=b[2];row=np.zeros(6*n);row[3*k:3*k+3]=-normal;row[3*j:3*j+3]=normal
            matrix.append(row);rhs.append(float(np.min(points@normal))-(0. if state.bases[k][2]@normal>1-1e-10 else .006))
    for i,x in enumerate(state.offsets.ravel()):
        for sign in (-1,1):
            row=np.zeros(6*n);row[i]=sign;row[3*n+i]=-1
            matrix.append(row);rhs.append(sign*x)
    def solve(extra=None):
        a=matrix if extra is None else matrix+[extra[0]]
        b=rhs if extra is None else rhs+[extra[1]]
        return linprog(np.r_[np.zeros(3*n),np.ones(3*n)],A_ub=np.asarray(a),b_ub=b,
            bounds=[(None,None)]*(3*n)+[(0,None)]*(3*n),method='highs')
    initial=solve()
    if not initial.success:return None
    offsets=initial.x[:3*n].reshape(n,3)
    normals=np.vstack([np.eye(3),np.vstack([b for b in state.bases]),
        [np.asarray(p['initial_object_exit_world'])@b for p,b in zip(state.paths,state.bases)]])
    normals=np.vstack([normals,-normals]);normals=np.unique(np.round(normals,10),axis=0)
    pairs=[(i,j) for i in range(n) for j in range(n) if i!=j]
    if attempt%2:pairs=pairs[::-1]
    records=[]
    for iteration in range(iterations):
        collisions=[]
        for i,j in pairs:
            volume=abs(float((sweeps[i].translate(offsets[i]/S.SCALE)^root_solids[j].translate(offsets[j]/S.SCALE)).volume()))*S.SCALE**3
            if volume>8e-14:collisions.append((volume,i,j))
        if not collisions:
            return replace(state,offsets=offsets),records
        _,i,j=max(collisions) if attempt<2 else min(collisions)
        choices=[]
        for normal in normals:
            row=np.zeros(6*n);row[3*i:3*i+3]=normal;row[3*j:3*j+3]=-normal
            bound=float(np.min(root_points[j]@normal)-np.max(sweep_points[i]@normal)-.01)
            result=solve((row,bound))
            if result.success:choices.append((float(result.fun),tuple(normal),row,bound,result))
        if not choices:return None
        choices.sort(key=lambda v:(v[0],v[1]));choice=choices[min(attempt//4,len(choices)-1)]
        _,normal,row,bound,result=choice;matrix.append(row);rhs.append(bound)
        offsets=result.x[:3*n].reshape(n,3)
        records.append(dict(pose=tasks[i].pose,foreign_heads=tasks[j].pose,normal=list(normal),iteration=iteration))
    return None


def saved_ground_parts(reference,bases,offsets):
    """Real source landing material serves only as a seating feasibility witness."""
    import trimesh
    mesh=trimesh.load(reference,force='mesh',process=False);parts=[]
    for b,o in zip(bases,offsets):
        native=mesh.copy();native.vertices=(mesh.vertices-o)@b.T
        lo,hi=native.bounds;lo=lo.copy();hi=hi.copy();lo[2]=-1e-6;hi[2]=.006
        slab=B.md.Manifold.cube(((hi-lo)/S.SCALE).tolist()).translate((lo/S.SCALE).tolist())
        parts.append(S.solid(native)^slab)
    return parts
