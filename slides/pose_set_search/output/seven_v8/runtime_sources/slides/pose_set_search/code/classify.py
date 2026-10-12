"""Original LPs with fast, explicitly nonnegative batch primal replays.

Unlike a raw pseudoinverse sign test, a replay clips negative coefficients to
zero and then verifies the resulting actual equilibrium. No negative force
is accepted and the original 2e-9 conditioned equilibrium tolerance remains.
"""
from common import C, np
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


def classify(full, targets):
    targets=C.U.target(targets,full.shape[1])
    accepted=np.zeros(len(targets),bool);pending=np.ones(len(targets),bool)
    info=dict(method='original_equilibrium_LP_with_nonnegative_longdouble_batch_replay',
              equilibrium_lps=0,separation_lps=0,primal_batch_accepted=0,dual_batch_rejected=0,
              maximum_primal_replay_residual=0.,equilibrium_tolerance=2e-9)
    full_extended=np.asarray(full,np.longdouble)
    while pending.any():
        remaining=np.flatnonzero(pending);index=int(remaining[0]);pending[index]=False
        witness=C.C.W.solve(full,targets[index]);info['equilibrium_lps']+=1
        if witness is not None:
            accepted[index]=True
            basis=full[witness['indices']]
            if not len(basis):
                continue
            coefficients=replay_coefficients(basis,targets[remaining])
            # This is a constructive nonnegative witness in the original
            # equations. Long-double substitution avoids cancellation.
            extended_basis=np.asarray(basis,np.longdouble)
            replay=coefficients @ extended_basis
            residual=np.max(np.abs(replay-np.asarray(targets[remaining],np.longdouble)),axis=1)
            roundoff=32*np.finfo(np.longdouble).eps*np.max(coefficients @ np.abs(extended_basis),axis=1)
            certified=residual+roundoff<=2e-9
            accepted[remaining[certified]]=True;pending[remaining[certified]]=False
            info['primal_batch_accepted']+=int(certified.sum())
            if certified.any():
                info['maximum_primal_replay_residual']=max(info['maximum_primal_replay_residual'],float((residual+roundoff)[certified].max()))
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
