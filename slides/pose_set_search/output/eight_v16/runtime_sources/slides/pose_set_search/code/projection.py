"""Distance to the same unbounded7D cone via a small polar QP.

Cut generation keeps seven optimization variables. Nonnegative primal replay
and every original-ray reduced cost validate the result; no wide TRF fallback.
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, nnls, lsq_linear


def cone_projection(generators,target,scale=None):
    rays=np.asarray(generators,float).reshape(-1,7)
    target=np.asarray(target,float)
    metric=np.ones(7) if scale is None else np.asarray(scale,float)
    if target.shape!=(7,) or metric.shape!=(7,) or np.any(metric<=0) or not np.isfinite(rays).all():
        raise ValueError('Finite7D cone, target and positive metric required')
    coefficients=np.zeros(len(rays))
    weighted=rays*metric
    lengths=np.linalg.norm(weighted,axis=1)
    valid=np.flatnonzero(lengths>0)
    solver='empty'
    if len(valid):
        normalized,first=np.unique(weighted[valid]/lengths[valid,None],axis=0,return_index=True)
        original=valid[first]
        b=target*metric
        selected=[int(np.argmax(normalized @ b))]
        dual=np.zeros(7)
        tolerance=1e-10*max(1.,np.linalg.norm(b))
        for iteration in range(64):
            active=normalized[selected]
            fit=minimize(lambda r:(.5*float((r+b) @ (r+b)),r+b),dual,jac=True,
                         method='SLSQP',constraints=[LinearConstraint(active,0.,np.inf)],
                         options=dict(ftol=1e-13,maxiter=120))
            dual=fit.x
            reduced=normalized @ dual
            if reduced.min()>=-tolerance:
                break
            additions=[int(i) for i in np.argsort(reduced)[:8] if reduced[i]<-tolerance and int(i) not in selected]
            if not additions:
                raise RuntimeError('Polar cone constraint generation stalled')
            selected.extend(additions)
        else:
            raise RuntimeError('Polar cone cut budget exhausted')
        # Polish against the original target, not the approximate dual point.
        # This gives a valid cone point and exact active reduced costs even
        # when the polar QP's stopping error is amplified by floor rays.
        for polish in range(64):
            active=normalized[selected]
            try:
                weights,_=nnls(active.T,b,maxiter=2000)
            except (RuntimeError,np.linalg.LinAlgError):
                fit=lsq_linear(active.T,b,bounds=(0,np.inf),method='bvls',
                               lsq_solver='exact',tol=1e-13,max_iter=300)
                weights=fit.x
            residual_weighted=weights @ active-b
            reduced=normalized @ residual_weighted
            if reduced.min()>=-tolerance:
                break
            additions=[int(i) for i in np.argsort(reduced)[:8] if reduced[i]<-tolerance and int(i) not in selected]
            if not additions:
                raise RuntimeError('Small primal projection polish stalled')
            selected.extend(additions)
        else:
            raise RuntimeError('Small primal projection polish budget exhausted')
        coefficients[original[selected]]=weights/lengths[original[selected]]
        solver='seven_variable_polar_QP_with_primal_replay'
    residual=coefficients @ rays-target
    dual=metric**2*residual
    reduced=rays @ dual
    negative=max(0.,float(-reduced.min())) if len(reduced) else 0.
    active=coefficients>1e-10
    complementarity=float(np.max(np.abs(reduced[active]))) if active.any() else 0.
    return dict(loss=.5*float((metric*residual) @ (metric*residual)),residual=residual,dual=dual,
                coefficients=coefficients,kkt_max_violation=max(negative,complementarity),solver=solver)
