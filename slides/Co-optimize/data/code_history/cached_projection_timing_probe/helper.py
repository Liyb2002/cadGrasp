"""Cheap primal certificates before the unchanged all-ray NNLS projection.

Guidance only: a nonnegative reaction basis is reused only if ALL its original
ray IDs remain available and all seven equations replay within2e-12. Targets
without such a certificate receive the original full-cone NNLS projection.
"""
import numpy as np
from whole_search.classify import replay_coefficients
from .projection import project_demands


def cached_project(model,owner,flags,targets):
    rays,column_ids=model.supply_at_points(owner,flags)
    targets=np.asarray(targets,float);pending=np.ones(len(targets),bool)
    positions={int(key):j for j,key in enumerate(column_ids)}
    certified=0;maximum_bound=0.
    for ids in model.basis_caches[owner]:
        if not pending.any():break
        if not all(i in positions for i in ids):continue
        indices=np.flatnonzero(pending);basis=rays[[positions[i] for i in ids]]
        coefficients=replay_coefficients(basis,targets[indices]);extended=np.asarray(basis,np.longdouble)
        residual=np.max(np.abs(coefficients@extended-np.asarray(targets[indices],np.longdouble)),axis=1)
        roundoff=32*np.finfo(np.longdouble).eps*np.max(coefficients@np.abs(extended),axis=1)
        bound=residual+roundoff;keep=bound<=2e-12
        if keep.any():maximum_bound=max(maximum_bound,float(bound[keep].max()))
        pending[indices[keep]]=False;certified+=int(keep.sum())
    residuals=np.zeros_like(targets);losses=np.zeros(len(targets));kkt=0.
    if pending.any():
        projected=project_demands(rays,targets[pending])
        residuals[pending]=projected['residuals'];losses[pending]=projected['losses']
        kkt=projected['maximum_kkt_violation']
    return dict(losses=losses,residuals=residuals,maximum_kkt_violation=kkt,
        primal_certified_nodes=certified,nnls_nodes=int(pending.sum()),
        maximum_primal_replay_bound=maximum_bound,
        numerical_zero_distance_upper_bound=7*maximum_bound**2,
        all_original_equations_replayed=True,final_acceptance_changed=False)
