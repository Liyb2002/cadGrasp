"""All-target cone distances using verified active-set regions, with exact fallbacks."""
import numpy as np
from physics_guided_cone_svd import cone_projection


def all_projection_losses(rays,targets,*,failed=None,region_limit=256,chunk=128):
    rays=np.asarray(rays,float);targets=np.asarray(targets,float)
    if rays.size==0:rays=np.empty((0,7))
    if rays.ndim!=2 or rays.shape[1]!=7 or targets.ndim!=2 or targets.shape[1]!=7:
        raise ValueError('seven-coordinate rays and targets required')
    if not np.isfinite(rays).all() or not np.isfinite(targets).all():raise ValueError('finite inputs required')
    if failed is None:failed=np.ones(len(targets),bool)
    failed=np.asarray(failed,bool)
    if failed.shape!=(len(targets),):raise ValueError('failed mask shape mismatch')
    remaining=np.flatnonzero(failed);losses=np.zeros(len(targets));regions=0;fallbacks=0
    ray_norm=np.linalg.norm(rays,axis=1)
    while len(remaining) and regions<region_limit:
        index=int(remaining[np.argmax(np.linalg.norm(targets[remaining],axis=1))])
        seed=cone_projection(rays,targets[index]);active=np.flatnonzero(seed['coefficients']>0)
        losses[index]=seed['loss'];remaining=remaining[remaining!=index];regions+=1
        if not len(remaining):break
        if len(active):
            basis=(rays[active]/ray_norm[active,None]).T
            inverse=np.linalg.lstsq(basis,np.eye(7),rcond=1e-14)[0]
        else:
            basis=np.empty((7,0));inverse=np.empty((0,7))
        not_covered=[]
        for start in range(0,len(remaining),chunk):
            ids=remaining[start:start+chunk];rhs=targets[ids]
            coefficients=rhs@inverse.T
            nonnegative=np.all(coefficients>=-1e-12,axis=1)
            coefficients=np.maximum(coefficients,0.)
            residual=coefficients@basis.T-rhs
            reduced=residual@rays.T
            tolerance=1e-7*np.maximum(1.,np.linalg.norm(rhs,axis=1))
            dual_valid=np.min(reduced,axis=1)>=-tolerance if len(rays) else np.ones(len(ids),bool)
            active_valid=np.max(abs(reduced[:,active]),axis=1)<=tolerance if len(active) else np.ones(len(ids),bool)
            valid=nonnegative & dual_valid & active_valid
            losses[ids[valid]]=.5*np.einsum('ij,ij->i',residual[valid],residual[valid])
            not_covered.extend(ids[~valid])
        remaining=np.asarray(not_covered,int)
    # A region limit is never a permission to omit a load or approximate its gap.
    for index in remaining:
        losses[index]=cone_projection(rays,targets[index])['loss'];fallbacks+=1
    return losses,dict(original_loads=len(targets),failed_loads=int(failed.sum()),
                       active_regions=regions,individual_fallbacks=fallbacks,
                       all_failed_loads_projected=True,original_ray_kkt_checked=True)
