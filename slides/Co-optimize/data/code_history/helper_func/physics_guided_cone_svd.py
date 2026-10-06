"""Seven-dimensional cone projection with SVD active-set corrections.

Original unbounded rays and metric; no force caps, regularization, or ray
approximation. Every returned solution is checked against all original rays.
"""
import numpy as np


def cone_projection(generators,target,scale=None):
    rays=np.asarray(generators,float)
    if not rays.size:rays=np.empty((0,7))
    target=np.asarray(target,float);metric=np.ones(7) if scale is None else np.asarray(scale,float)
    if rays.ndim!=2 or rays.shape[1]!=7 or not np.isfinite(rays).all():raise ValueError('finite (N,7) rays required')
    if target.shape!=(7,) or not np.isfinite(target).all():raise ValueError('finite seven-vector target required')
    if metric.shape!=(7,) or not np.isfinite(metric).all() or np.any(metric<=0):raise ValueError('positive finite metric required')
    weighted=rays*metric;norms=np.linalg.norm(weighted,axis=1);valid=np.flatnonzero(norms>0)
    coefficients=np.zeros(len(rays));target_norm=max(float(np.linalg.norm(target*metric)),1e-300)
    if len(valid) and np.linalg.norm(target*metric)>0:
        matrix=(weighted[valid]/norms[valid,None]).T;rhs=target*metric/target_norm
        values=np.zeros(len(valid));active=[]
        for iteration in range(2048):
            residual=matrix@values-rhs;reduced=matrix.T@residual
            # Tiny active-set stationarity errors must not reenter an active ray.
            reduced[active]=np.inf;worst=int(np.argmin(reduced))
            if reduced[worst]>=-1e-11:break
            active.append(worst)
            for correction in range(2048):
                z=np.linalg.lstsq(matrix[:,active],rhs,rcond=1e-14)[0]
                if np.all(z>0):
                    values[:]=0.;values[active]=z;break
                current=values[active];bad=z<=0
                denominator=current[bad]-z[bad]
                ratio=np.divide(current[bad],denominator,out=np.ones_like(denominator),where=denominator>0)
                alpha=min(1.,float(np.min(ratio)))
                updated=current+alpha*(z-current)
                values[:]=0.;values[active]=np.maximum(updated,0.)
                old=active;active=[i for i in old if values[i]>1e-14]
                values[[i for i in old if i not in active]]=0.
                if not active:break
            else:raise RuntimeError('SVD cone active correction limit')
        else:raise RuntimeError('SVD cone active-set iteration limit')
        coefficients[valid]=values*target_norm/norms[valid]
    residual=coefficients@rays-target;dual=metric**2*residual;reduced=rays@dual
    active=coefficients>1e-10
    violation=max(max(0.,float(-reduced.min())) if len(reduced) else 0.,
                  float(np.max(abs(reduced[active]))) if active.any() else 0.)
    if violation>1e-7*max(1.,np.linalg.norm(target)):
        raise RuntimeError(f'SVD cone projection KKT unresolved: {violation}')
    return dict(loss=float(.5*np.dot(metric*residual,metric*residual)),residual=residual,
                dual=dual,coefficients=coefficients,kkt_max_violation=violation,solver='svd_active_set')
