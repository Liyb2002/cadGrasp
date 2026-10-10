"""Exact polynomial lift of the original continuous R6 working-load domain.

Compilation uses geometry, gravity and cone constants only. No saved wrench
samples are read. Visibility and the original parameter bounds are retained
in r6_domain.json; a polynomial lift does not convexify that domain.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np

from r6_demand import R6Demand, ROOT, cross_matrix, digest, add_json

SCHEMA = 'cadgrasp_r6_quadratic_lift_v1'


def features(u, v, force):
    """z=(1,P,uP,vP), with P the processing force, not support reaction."""
    p = np.asarray(force, float)
    if p.ndim < 1 or p.shape[-1] != 3 or not np.isfinite(p).all():
        raise ValueError('Finite processing force [...,3] required')
    shape = np.broadcast_shapes(np.shape(u), np.shape(v), p.shape[:-1])
    p = np.broadcast_to(p, shape+(3,))
    u, v = np.broadcast_to(u, shape), np.broadcast_to(v, shape)
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError('Finite triangle coordinates required')
    return np.concatenate((np.ones(shape+(1,)), p, u[...,None]*p, v[...,None]*p), axis=-1)


def feature_jacobian(u, v, force):
    """Exact dz/d(u,v,Px,Py,Pz), for chain rules in a fixed projection cell."""
    p = np.asarray(force, float)
    z = features(u, v, p); shape = z.shape[:-1]
    p = np.broadcast_to(p, shape+(3,))
    j = np.zeros(shape+(10,5))
    j[...,1:4,2:] = np.eye(3)
    j[...,4:7,0] = p; j[...,7:10,1] = p
    j[...,4:7,2:] = np.broadcast_to(u,shape)[...,None,None]*np.eye(3)
    j[...,7:10,2:] = np.broadcast_to(v,shape)[...,None,None]*np.eye(3)
    return j


class QuadraticR6:
    def __init__(self, domain):
        self.domain = domain
        self.matrices = np.zeros((len(domain.a),6,10))
        self.matrices[:,:3,0] = -domain.gravity
        self.matrices[:,:3,1:4] = -np.eye(3)
        for start, arm in [(1,domain.a),(4,domain.e),(7,domain.f)]:
            self.matrices[:,3:,start:start+3] = -cross_matrix(arm)

    @classmethod
    def read(cls, folder):
        folder = Path(folder)
        metadata = json.loads((folder/'r6_quadratic.json').read_text())
        if metadata['schema'] != SCHEMA or digest(folder/'r6_domain.json') != metadata['source_sha256']:
            raise ValueError('Stale compiled R6 source')
        obj = cls(R6Demand.read(folder/'r6_domain.json'))
        with np.load(folder/'r6_quadratic.npz') as saved:
            np.testing.assert_array_equal(obj.matrices,saved['B'])
        return obj

    def cartesian(self, face, u, v, force):
        z = features(u,v,force)
        raw = np.broadcast_to(np.asarray(face),z.shape[:-1])
        if np.any(raw != raw.astype(int)) or np.any(raw<0) or np.any(raw>=len(self.matrices)):
            raise ValueError('Working-face index out of bounds')
        return np.einsum('...ij,...j->...i',self.matrices[raw.astype(int)],z)

    def spherical(self, face, parameters, check_visibility=True):
        index, r, d, _, _, _, magnitude = self.domain._parameters(face,parameters)
        p = np.asarray(parameters)
        wrench = self.cartesian(index,p[...,0],p[...,1],magnitude[...,None]*d)
        reachable = self.domain.visible(self.domain.com+r,d,self.domain.n[index]) if check_visibility else None
        return dict(need_wrench=wrench,reachable=reachable)

    def cell_quadratic(self, face, residual_projector, scale=None):
        """H=B^T S^T Q S B; loss=.5 z^T H z, on ONE valid NNLS cell.

        Q is an orthogonal residual projector. For the actual 7D cone, pass
        its 7x7 Q: the extra target coordinate is zero, never dropped.
        """
        q = np.asarray(residual_projector,float)
        if q.shape not in [(6,6),(7,7)]:
            raise ValueError('Expected a 6D or 7D residual projector')
        b = self.matrices[face]*(np.ones(6) if scale is None else np.asarray(scale))[:,None]
        if len(q)==7: b = np.vstack((b,np.zeros((1,10))))
        return b.T @ q @ b


def moment_loss(h, second_moment):
    """Analytic integral on a valid projection cell, M=integral z z^T dmu."""
    return .5*np.einsum('...ij,...ji->...',h,second_moment)


def compile_all():
    began=time.monotonic(); rows=[]; max_error=0.; max_jacobian_error=0.
    paths=sorted(ROOT.glob('*/poses/pose_*/angle_*/r6_domain.json'))
    assert len(paths)==1890
    for number,path in enumerate(paths,1):
        domain=R6Demand.read(path); compiled=QuadraticR6(domain)
        # Independent geometric check, without reading any sampled demands.
        ids=np.arange(len(domain.a)); u=.23; v=.31
        p=.217*(np.cos(domain.alpha/2)*domain.n+np.sin(domain.alpha/2)*domain.t1)
        r=domain.a+u*domain.e+v*domain.f
        expected=np.c_[-domain.gravity-p,-np.cross(r,p)]
        error=float(np.max(np.abs(compiled.cartesian(ids,u,v,p)-expected),initial=0.))
        jac=np.einsum('fij,fjk->fik',compiled.matrices,feature_jacobian(u,v,p))
        je=float(np.max(np.abs(jac-domain.jacobian_cartesian(ids,u,v,p)),initial=0.))
        max_error=max(max_error,error); max_jacobian_error=max(max_jacobian_error,je)
        if error>1e-13 or je>1e-13: raise RuntimeError(f'Compilation mismatch: {path}')
        target=path.parent/'r6_quadratic.npz'
        if target.exists():
            with np.load(target) as saved: np.testing.assert_array_equal(saved['B'],compiled.matrices)
        else:
            with target.open('xb') as stream: np.savez_compressed(stream,B=compiled.matrices)
        metadata=dict(schema=SCHEMA,source='r6_domain.json',source_sha256=digest(path),
            array='r6_quadratic.npz',array_sha256=digest(target),features=['1','P_x','P_y','P_z','uP_x','uP_y','uP_z','vP_x','vP_y','vP_z'],
            formula='b=B_f z; H_f=B_f^T S^T Q S B_f; L_cell=0.5 trace(H_f M_f)',
            exact_forward=True,sample_fitting=False,sample_rows_read=0,
            constraints_inherited_from='r6_domain.json',
            cell_validity='nonnegative active coefficients and all-ray NNLS KKT',
            continuous_integration='sum over valid projection cells; numerical quadrature locates cells',
            geometry_derivative='contact availability events require resolved geometric secants',
            ground_state_independent=True)
        add_json(path.parent/'r6_quadratic.json',metadata)
        rows.append(dict(expression=str(path.parent.relative_to(ROOT)),source_sha256=digest(path),
                         array_sha256=digest(target),work_faces=len(domain.a)))
        if number%180==0: print('COMPILE',number,len(paths),flush=True)
    index=dict(schema='cadgrasp_r6_quadratic_index_v1',complete=True,expressions=len(rows),
               sample_rows_read=0,sample_fitting=False,max_forward_error=max_error,
               max_analytic_jacobian_error=max_jacobian_error,entries=rows,
               generator_sha256=digest(__file__))
    add_json(ROOT/'r6_quadratic_index.json',index)
    print('COMPILED',len(rows),'seconds',round(time.monotonic()-began,2),'errors',max_error,max_jacobian_error,flush=True)


if __name__=='__main__': compile_all()
