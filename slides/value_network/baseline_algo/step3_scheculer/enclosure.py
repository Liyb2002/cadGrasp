"""Tangent outer polytopes and residual-to-coefficient bounds for primal certificates.

Adapted from obj_supp/area/continuous.py, using the CURRENT cone angle and
including the zero-force endpoint. Search samples are not used in the proof.
"""
import numpy as np
from scipy.optimize import nnls
from step3_scheculer.stage_imports import load_stage
C = load_stage('score', 'contribution')
from step1.needs import frame

ENTRY_GUARD=1e-10
ROUND_GUARD=128*np.finfo(float).eps
CAP_PADDING=1e-12


def basis_membership(B,targets):
    dimension=targets.shape[1]
    if B.shape!=(dimension,dimension):
        return np.zeros(len(targets),bool),np.full(len(targets),-np.inf)
    try:
        X=np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return np.zeros(len(targets),bool),np.full(len(targets),-np.inf)
    nx=np.linalg.norm(X,ord=np.inf)*(1+ROUND_GUARD)
    error=ROUND_GUARD*(np.abs(B)@np.abs(X)+np.eye(dimension))
    eta=np.max(np.sum(np.abs(np.eye(dimension)-B@X)+error,axis=1))+dimension*ENTRY_GUARD*nx
    if not np.isfinite(eta) or eta>=1:
        return np.zeros(len(targets),bool),np.full(len(targets),-np.inf)
    inverse_bound=nx/(1-eta)*(1+ROUND_GUARD)
    coefficients=targets@X.T
    residual=np.max(np.abs(coefficients@B.T-targets)+
        ROUND_GUARD*(np.abs(coefficients)@np.abs(B).T+np.abs(targets)),axis=1)
    residual+=ENTRY_GUARD*(1+np.abs(coefficients).sum(axis=1))*(1+ROUND_GUARD)
    margin=coefficients.min(axis=1)-inverse_bound*residual
    return margin>0,margin


def cap_vertices(n,half_angle,sides=8,bands=1):
    """Circumscribe the meridian arc by tangent intersections, then each ring.

    Adjacent tangents at lo,hi meet at (rho,z) =
    (sin(mid),cos(mid))/cos((hi-lo)/2). Their polygon above z=cos(alpha)
    contains the entire meridian arc. Revolving its rings and replacing each
    circle by a circumscribed regular polygon preserves containment. Both
    axial padding signs enclose floating-point error at every ideal ring.
    """
    if sides<3 or bands<1 or not 0<half_angle<np.pi/2:
        raise ValueError('Expected at least 3 sides, 1 band and a cap smaller than a hemisphere')
    n=np.asarray(n,float)
    e1,e2=frame(n)
    angles=np.arange(sides)*2*np.pi/sides
    transverse=np.cos(angles)[None,:,None]*e1[:,None,:]+np.sin(angles)[None,:,None]*e2[:,None,:]
    direction_sets=[]
    edges=np.linspace(0,half_angle,bands+1)
    rings=[]
    for low,high in zip(edges[:-1],edges[1:]):
        mid=(low+high)/2;divisor=np.cos((high-low)/2)
        rings.append((np.sin(mid)/divisor,np.cos(mid)/divisor))
    rings.append((np.sin(half_angle),np.cos(half_angle)))
    for radial,height in rings:
        radius=radial/np.cos(np.pi/sides)+CAP_PADDING
        for axial in [height-CAP_PADDING,height+CAP_PADDING]:
            direction_sets.append(axial*n[:,None,:]+radius*transverse)
    return np.concatenate(direction_sets,axis=1)


def targets(problem,sides=8,bands=1):
    d=problem.domain
    points=d.mesh.triangles[d.work_ids].reshape(-1,3)
    n=np.repeat(d.normals,3,axis=0)
    n=n/np.linalg.norm(n,axis=1)[:,None]
    directions=cap_vertices(n,d.half_angle,sides,bands)
    forces=d.k*directions
    moments=-np.cross(points[:,None,:]-d.com,forces)
    wrenches=np.concatenate([-d.gravity-forces,moments],axis=2).reshape(-1,6)*problem.scale
    return C.U.target(np.vstack([np.r_[-d.gravity,[0,0,0]]*problem.scale,wrenches]))


def contain(problem,full,sides=8,bands=1):
    batch=targets(problem,sides,bands)
    assignment=np.full(len(batch),-1,np.int32)
    bases=[]
    minimum=np.inf
    while np.any(assignment<0):
        pending=np.flatnonzero(assignment<0)
        target=batch[pending[0]]
        witness=C.W.solve(full,target)
        if witness is None:
            return dict(status='not_certified',reason='outer_vertex_infeasible',sides=sides,bands=bands),None
        ids=np.array(witness['indices'],int)
        good,margin=basis_membership(full[ids].T,batch[pending])
        if not good[0]:
            # Seek a different reaction basis, without changing the generators.
            try:
                weights=nnls(full.T,target,maxiter=max(1000,3*len(full)))[0]
                ids=np.flatnonzero(weights>1e-12)
                good,margin=basis_membership(full[ids].T,batch[pending])
            except RuntimeError:
                pass
        if not good[0]:
            return dict(status='not_certified',reason='coefficient_error_bound_inconclusive',sides=sides,bands=bands),None
        assignment[pending[good]]=len(bases)
        bases.append(ids)
        minimum=min(minimum,float(margin[good].min()))
    return dict(status='verified',method='triangle_vertices_times_tangent_cap_outer_polytopes_with_coefficient_error_bounds',
        sides=sides,bands=bands,outer_vertex_count=len(batch),basis_count=len(bases),
        minimum_positive_coefficient_margin=minimum,scaled_entry_guard=ENTRY_GUARD,
        cap_vertex_outward_padding=CAP_PADDING,
        cone_half_degrees=float(np.rad2deg(problem.domain.half_angle)),
        includes_zero_force=True,includes_occluded_directions=True),dict(bases=np.array(bases),assignment=assignment)
