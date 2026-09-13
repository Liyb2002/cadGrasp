"""Additional, recorded LP formulations for an unresolved Step3 solve.

Only invoked on solver errors. All successful reactions are checked in the
original equations; unknown status is never a negative feasibility verdict.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import hashlib
import runpy
import sys
import numpy as np
from scipy.optimize import linprog, nnls
from step3_scheculer import continuous_retry
HERE=Path(__file__).resolve().parent.parent


def recover(full,target):
    attempts=[]
    # Positive row/column rescaling preserves the equations and x >= 0.
    rows=1/np.maximum(np.max(np.abs(full),axis=0),1e-15)
    for row_scale in [np.ones(full.shape[1]),rows]:
        columns=np.maximum(np.linalg.norm(full*row_scale,axis=1),1e-30)
        matrix=(full*row_scale/columns[:,None]).T
        rhs=target*row_scale
        for objective in [np.zeros(len(full)),np.ones(len(full))]:
            for presolve in [True,False]:
                result=linprog(objective,A_eq=matrix,b_eq=rhs,bounds=(0,None),method='highs-ds',
                    options=dict(presolve=presolve,primal_feasibility_tolerance=1e-10,
                                 dual_feasibility_tolerance=1e-10,simplex_dual_edge_weight_strategy='steepest'))
                entry=dict(method='equilibrated_highs_ds',presolve=presolve,status=int(result.status),
                           row_scaling=row_scale.tolist(),objective='zero' if not objective.any() else 'sum')
                attempts.append(entry)
                if not result.success:continue
                x=result.x/columns
                residual=float(np.max(np.abs(full.T@x-target)))
                entry['original_residual']=residual
                if np.min(x)>=0 and residual<=2e-9:
                    ids=np.flatnonzero(x>0)
                    return dict(indices=ids.tolist(),coefficients=x[ids].tolist(),residual=residual,precise=None),dict(status='feasible',attempts=attempts)
    # NNLS provides another primal proposal, accepted only at the same original
    # equilibrium tolerance. Its failure is not an infeasibility certificate.
    try:
        y,_=nnls(matrix,rhs,maxiter=max(1000,10*len(full)))
        x=y/columns;residual=float(np.max(np.abs(full.T@x-target)))
        attempts.append(dict(method='nnls_proposal',original_residual=residual))
        if np.min(x)>=0 and residual<=2e-9:
            ids=np.flatnonzero(x>0)
            return dict(indices=ids.tolist(),coefficients=x[ids].tolist(),residual=residual,precise=None),dict(status='feasible',attempts=attempts)
    except RuntimeError as error:attempts.append(dict(method='nnls_proposal',error=str(error)))
    # Match the base solver's explicit status-2 verdict, requiring agreement
    # between at least two formulations here. Unknown statuses do not count;
    # any reported primal success without a verified witness stays unresolved.
    statuses=[a['status'] for a in attempts if 'status' in a]
    if statuses.count(2)>=2 and 0 not in statuses:
        return None,dict(status='infeasible_numeric',attempts=attempts,
                         agreeing_infeasible_formulations=statuses.count(2))
    raise RuntimeError('Additional LP formulations remain unresolved: '+str(attempts))


def install(objects):
    continuous_retry.install(objects)
    from step3_scheculer.stage_imports import load_stage
    from step3_scheculer import contacts as I
    from step1.cases import pose_name
    C=load_stage('score','contribution');original=C.W.solve;hashes=C.code_hashes
    C.code_hashes=lambda:{**hashes(),**I.hashes([Path(__file__)])}
    def solve(full,target):
        try:return original(full,target)
        except RuntimeError as error:
            digest=hashlib.sha256(np.ascontiguousarray(full).tobytes()+np.ascontiguousarray(target).tobytes()).hexdigest()
            out=I.OUTPUTS/objects[0]/pose_name()/'step3_scheculer/strict_lp_retries';out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/(digest+'.npz'),full=full,target=target)
            I.save(out/(digest+'.json'),dict(complete=False,original_error=str(error)))
            witness,evidence=recover(full,target)
            I.save(out/(digest+'.json'),dict(complete=True,original_error=str(error),same_equations=True,
                **evidence,provenance=dict(inputs={},code=C.code_hashes()),
                artifacts={digest+'.npz':I.sha256(out/(digest+'.npz'))}))
            print('Additional original-equation LP recovery:',evidence['status'],flush=True)
            return witness
    C.W.solve=solve
    return C


if __name__=='__main__':
    stage=Path(sys.argv[1]).resolve()
    if HERE not in stage.parents:raise ValueError('Stage must be inside baseline_algo')
    objects=[s for s in sys.argv[2:] if s in ('A1-f','B','C5')]
    if len(objects)!=1:raise ValueError('One object/pose per recovery process')
    install(objects);sys.argv=[str(stage)]+sys.argv[2:]
    if stage==HERE/'step3_scheculer/verification.py':
        from step3_scheculer import verification
        verification.run(objects[0])
    else:runpy.run_path(str(stage),run_name='__main__')
