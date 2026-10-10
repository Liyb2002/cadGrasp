"""Original LPs with fast, explicitly nonnegative batch primal replays.

Unlike a raw pseudoinverse sign test, a replay clips negative coefficients to
zero and then verifies the resulting actual equilibrium. No negative force
is accepted and the original 2e-9 conditioned equilibrium tolerance remains.
"""
from .common import C, np
from scipy.optimize import linprog
from scipy.linalg import qr


def replay_coefficients(basis,targets):
    """Solve small independent bases in extended precision, then replay all7."""
    count=len(basis)
    if count>basis.shape[1]:
        return np.maximum(targets @ np.linalg.pinv(basis),0.).astype(np.longdouble)
    coordinates=qr(basis,pivoting=True,mode='economic')[2][:count]
    matrix=np.asarray(basis[:,coordinates].T,np.longdouble)
    augmented=np.column_stack([matrix,np.eye(count,dtype=np.longdouble)])
    for column in range(count):
        pivot=column+int(np.argmax(np.abs(augmented[column:,column])))
        augmented[[column,pivot]]=augmented[[pivot,column]]
        divisor=augmented[column,column]
        if divisor==0:
            return np.maximum(targets @ np.linalg.pinv(basis),0.).astype(np.longdouble)
        augmented[column]/=divisor
        for row in range(count):
            if row!=column:
                augmented[row]-=augmented[row,column]*augmented[column]
    inverse=augmented[:,count:]
    return np.maximum(np.asarray(targets[:,coordinates],np.longdouble) @ inverse.T,0.)


def certified_basis(basis, target):
    """Verify nonnegative reactions in every original equilibrium equation."""
    coefficients=replay_coefficients(basis,np.asarray(target)[None])[0]
    extended=np.asarray(basis,np.longdouble)
    residual=np.max(np.abs(coefficients @ extended-np.asarray(target,np.longdouble)))
    roundoff=32*np.finfo(np.longdouble).eps*np.max(coefficients @ np.abs(extended))
    return bool(residual+roundoff<=2e-9),float(residual+roundoff)


def strict_witness(full,target):
    """Retry ill-conditioned LPs without changing rays, loads or tolerances.

    HiGHS may return a basis with inaccurate double-precision coefficients.
    Re-solve its small basis in long double, and try positive column scaling
    and a minimum-reaction objective if the original solve is unresolved.
    Only a verified primal or exact separating plane resolves the fallback.
    """
    original_error=None
    try:
        witness=C.C.W.solve(full,target)
    except RuntimeError as error:
        original_error=error
        witness=None
    if original_error is None:
        if witness is None:return None,False
        basis=full[witness['indices']]
        if len(basis):
            valid,residual=certified_basis(basis,target)
        else:
            valid,residual=bool(np.max(np.abs(target))<=2e-9),float(np.max(np.abs(target)))
        if valid:return dict(witness,replayed_original_residual=residual),False
    scales=np.maximum(np.linalg.norm(full,axis=1),np.finfo(float).tiny)
    equations=(full/scales[:,None]).T
    options=dict(presolve=False,primal_feasibility_tolerance=1e-9,
                 dual_feasibility_tolerance=1e-9)
    for method,objective in [('highs-ds',np.ones(len(full))),
                             ('highs-ipm',np.ones(len(full))),
                             ('highs-ds',np.zeros(len(full)))]:
        result=linprog(objective,A_eq=equations,b_eq=target,bounds=(0,None),
                       method=method,options=options)
        if not result.success:continue
        ids=np.flatnonzero(result.x>0)
        if not len(ids):continue
        valid,residual=certified_basis(full[ids],target)
        if valid:
            return dict(indices=ids.tolist(),replayed_original_residual=residual,
                        numerical_fallback=method),True
    if C.C.W.exact_separator(full,target) is not None:return None,True
    raise RuntimeError('Original equilibrium unresolved after strictly verified LP retries') from original_error


def classify(full, targets, *, basis_cache=None, column_ids=None):
    targets=C.U.target(targets,full.shape[1])
    accepted=np.zeros(len(targets),bool);pending=np.ones(len(targets),bool)
    info=dict(method='original_equilibrium_LP_with_nonnegative_longdouble_batch_replay',
              equilibrium_lps=0,separation_lps=0,primal_batch_accepted=0,dual_batch_rejected=0,
              maximum_primal_replay_residual=0.,equilibrium_tolerance=2e-9)
    info['numerical_lp_retries']=0
    full_extended=np.asarray(full,np.longdouble)
    def replay_basis(basis, remaining):
        coefficients=replay_coefficients(basis,targets[remaining])
        extended_basis=np.asarray(basis,np.longdouble)
        replay=coefficients @ extended_basis
        residual=np.max(np.abs(replay-np.asarray(targets[remaining],np.longdouble)),axis=1)
        roundoff=32*np.finfo(np.longdouble).eps*np.max(coefficients @ np.abs(extended_basis),axis=1)
        certified=residual+roundoff<=2e-9
        accepted[remaining[certified]]=True;pending[remaining[certified]]=False
        info['primal_batch_accepted']+=int(certified.sum())
        if certified.any():
            info['maximum_primal_replay_residual']=max(info['maximum_primal_replay_residual'],float((residual+roundoff)[certified].max()))
        return int(certified.sum())

    # Stable column IDs are original floor rays / surface quadrature points.
    # A basis is reused ONLY when every one of its actual rays still exists.
    # Changed targets are always substituted into all original equations.
    info['reused_basis_accepted'] = 0
    if basis_cache is not None:
        if column_ids is None or len(column_ids) != len(full):
            raise ValueError('Stable IDs required for reusable reaction bases')
        positions = {int(key):j for j,key in enumerate(column_ids)}
        for ids in list(basis_cache):
            if all(key in positions for key in ids) and pending.any():
                basis = full[[positions[key] for key in ids]]
                info['reused_basis_accepted'] += replay_basis(basis,np.flatnonzero(pending))
    while pending.any():
        remaining=np.flatnonzero(pending);index=int(remaining[0]);pending[index]=False
        witness,retried=strict_witness(full,targets[index]);info['equilibrium_lps']+=1
        info['numerical_lp_retries']+=int(retried)
        if witness is not None:
            accepted[index]=True
            basis=full[witness['indices']]
            if not len(basis):
                continue
            if basis_cache is not None:
                ids=tuple(int(column_ids[j]) for j in witness['indices'])
                if ids not in basis_cache:
                    basis_cache.insert(0,ids)
                    del basis_cache[24:]
            replay_basis(basis,remaining)
        else:
            result=linprog(-targets[index],A_ub=full,b_ub=np.zeros(len(full)),
                           bounds=[(-1,1)]*full.shape[1],method='highs',
                           options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            info['separation_lps']+=1
            if not result.success or np.linalg.norm(result.x)<1e-12:
                continue
            normal=result.x/np.linalg.norm(result.x)
            if np.max(full_extended @ np.asarray(normal,np.longdouble))>1e-12:
                continue
            rejected=targets[remaining] @ normal>1e-8
            pending[remaining[rejected]]=False;info['dual_batch_rejected']+=int(rejected.sum())
    info['covered_count']=int(accepted.sum())
    return accepted,info
