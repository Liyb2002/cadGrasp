"""Same nonnegative cone projection, with iterative bounded least squares.

Avoid the dense augmented QR of the exact TRF fallback for many generators.
Input rays and targets, the metric, and the KKT acceptance rule are unchanged.
"""
import numpy as np
from scipy.optimize import lsq_linear
from physics_guided_cone import _rows


def cone_projection(generators,target,metric=None):
    rays=_rows(generators);target=np.asarray(target,float)
    metric=np.ones(7) if metric is None else np.asarray(metric,float)
    if target.shape!=(7,) or metric.shape!=(7,) or not np.isfinite(metric).all() or np.any(metric<=0):
        raise ValueError('invalid target or projection metric')
    norms=np.linalg.norm(rays*metric,axis=1);keep=norms>1e-14
    coefficients=np.zeros(len(rays))
    if keep.any():
        matrix=(rays[keep]*metric/norms[keep,None]).T
        solution=lsq_linear(matrix,target*metric,bounds=(0,np.inf),lsq_solver='lsmr',
            tol=1e-12,lsmr_tol=1e-14,lsmr_maxiter=100,max_iter=1000)
        if not solution.success:raise RuntimeError('iterative contact cone projection failed: '+solution.message)
        coefficients[keep]=solution.x/norms[keep]
    residual=coefficients@rays-target;dual=metric**2*residual;reduced=rays@dual
    violation=max(0.,float(-reduced.min())) if len(rays) else 0.
    active=coefficients>1e-10
    if active.any():violation=max(violation,float(np.max(np.abs(reduced[active]))))
    return dict(loss=float(.5*np.dot(metric*residual,metric*residual)),residual=residual,dual=dual,
        coefficients=coefficients,kkt_max_violation=violation,solver='iterative_bounded_least_squares')
