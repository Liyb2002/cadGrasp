"""Deterministic close-direction initialization under native-floor hemispheres.

The dispersion objective is sum(1 - d_i dot reference). Fixed-start alternating
maximization supplies a local solution, not a global or force-feasibility claim.
"""
import numpy as np
from scipy.optimize import linprog


def _project(reference, normals):
    directions=np.repeat(reference[None,:],len(normals),axis=0)
    dots=normals@reference
    for k in np.flatnonzero(dots<0):
        tangent=reference-dots[k]*normals[k]
        if np.linalg.norm(tangent)<1e-12:
            axis=np.eye(3)[np.argmin(abs(normals[k]))]
            tangent=axis-(axis@normals[k])*normals[k]
        directions[k]=tangent/np.linalg.norm(tangent)
    return directions


def initialize_close_directions(normals, fixed_starts=48, max_iterations=100):
    normals=np.asarray(normals,float)
    if normals.ndim!=2 or normals.shape[1]!=3 or not len(normals) or not np.isfinite(normals).all():
        raise ValueError('finite nonempty Nx3 floor normals required')
    lengths=np.linalg.norm(normals,axis=1)
    if np.any(lengths<1e-12):raise ValueError('nonzero floor normals required')
    normals=normals/lengths[:,None]
    options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    lp=linprog([0,0,0,-1],A_ub=np.c_[-normals,np.ones(len(normals))],b_ub=np.zeros(len(normals)),bounds=[(-1,1)]*3+[(None,None)],options=options,method='highs')
    if not lp.success:raise RuntimeError(lp.message)
    common=None;status='no_nonzero_common_direction'
    if lp.x[3]>1e-9:
        common=lp.x[:3]/np.linalg.norm(lp.x[:3]);status='strict_common_direction'
    else:
        for axis in range(3):
            for sign in [-1.,1.]:
                bounds=[(-1.,1.)]*3;bounds[axis]=(sign,sign)
                edge=linprog(np.zeros(3),A_ub=-normals,b_ub=np.zeros(len(normals)),bounds=bounds,options=options,method='highs')
                if edge.success:
                    common=edge.x/np.linalg.norm(edge.x);status='boundary_only_common_direction';break
                if edge.status!=2:raise RuntimeError(edge.message)
            if common is not None:break
    if common is not None:
        directions=np.repeat(common[None,:],len(normals),axis=0);reference=common;iterations=0;starts=0
    else:
        k=np.arange(fixed_starts);z=1-2*(k+.5)/fixed_starts;phase=k*np.pi*(3-np.sqrt(5))
        sphere=np.c_[np.sqrt(1-z*z)*np.cos(phase),np.sqrt(1-z*z)*np.sin(phase),z]
        seeds=np.vstack([np.eye(3),-np.eye(3),normals,-normals,sphere]);best=None
        for seed in seeds:
            reference=seed.copy()
            for iteration in range(max_iterations):
                directions=_project(reference,normals);total=directions.sum(axis=0)
                if np.linalg.norm(total)<1e-12:break
                next_reference=total/np.linalg.norm(total)
                if np.linalg.norm(next_reference-reference)<1e-11:
                    reference=next_reference;break
                reference=next_reference
            directions=_project(reference,normals);cost=float(np.sum(1-directions@reference))
            if best is None or cost<best[0]-1e-12:best=(cost,directions.copy(),reference.copy(),iteration+1)
        _,directions,reference,iterations=best;starts=len(seeds)
    if np.min(np.einsum('ij,ij->i',normals,directions))<-1e-8:raise RuntimeError('illegal initialization')
    return directions,dict(method='common_floor_direction' if common is not None else 'fixed_multistart_hemisphere_projection',common_direction_status=status,reference_direction=reference.tolist(),dispersion_objective=float(np.sum(1-directions@reference)),minimum_upward_dot=float(np.min(np.einsum('ij,ij->i',normals,directions))),fixed_start_count=starts,iterations=iterations,global_optimum_claimed=False,force_feasibility_claimed=False)
