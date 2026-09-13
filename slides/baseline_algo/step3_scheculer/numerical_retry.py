"""Run a stage with one additional original-equation LP retry on solver errors.

Existing successful calls are unchanged. This separate entry records its own
source hash, allowing already completed stages to retain truthful provenance.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import hashlib
import json
import runpy
import sys
import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent.parent


def retry_original_equations(full, target):
    result = linprog(np.ones(len(full)), A_eq=full.T, b_eq=target, bounds=(0, None),
        method='highs-ipm', options=dict(presolve=False, primal_feasibility_tolerance=1e-10,
                                        dual_feasibility_tolerance=1e-10, ipm_optimality_tolerance=1e-12))
    if result.status == 2:
        return None, dict(status='infeasible_numeric', highs_status=2,
                          method='original_equations_highs_ipm_no_presolve')
    if not result.success:
        raise RuntimeError('Original-equation IPM retry is unresolved: '+result.message)
    x = result.x
    residual = float(np.max(np.abs(full.T@x-target)))
    if x.min() < 0 or residual > 2e-9:
        raise RuntimeError(f'Retry reaction certificate did not pass: {residual}')
    ids = np.flatnonzero(x > 0)
    return dict(indices=ids.tolist(), coefficients=x[ids].tolist(), residual=residual, precise=None), dict(
        status='feasible', highs_status=0, method='original_equations_highs_ipm_no_presolve',
        original_equilibrium_residual=residual, minimum_coefficient=float(x.min()))


def install(objects):
    from step3_scheculer.stage_imports import load_stage
    from step3_scheculer import contacts as I
    from step1.cases import pose_name
    C = load_stage('score', 'contribution')
    original = C.W.solve
    original_hashes = C.code_hashes
    def code_hashes():
        return {**original_hashes(), **I.hashes([Path(__file__)])}
    C.code_hashes = code_hashes
    def solve(full, target):
        try:
            return original(full, target)
        except RuntimeError as error:
            witness, evidence = retry_original_equations(full, target)
            digest = hashlib.sha256(np.ascontiguousarray(full).tobytes()+np.ascontiguousarray(target).tobytes()).hexdigest()
            for name in objects:
                out = I.OUTPUTS/name/pose_name()/'step3_scheculer/numerical_retries'
                out.mkdir(parents=True, exist_ok=True)
                data = out/(digest+'.npz'); record = out/(digest+'.json')
                if not data.exists():
                    np.savez_compressed(data, full=full, target=target)
                    I.save(record, dict(original_error=str(error), same_equations=True, **evidence,
                        scope='Additional numerical LP method; no force model, scoring or tolerance relaxation.',
                        provenance=dict(inputs={}, code=code_hashes()), artifacts={data.name: I.sha256(data)}))
            print('Recovered unresolved LP using original equations:', evidence['status'], flush=True)
            return witness
    C.W.solve = solve


if __name__ == '__main__':
    stage = Path(sys.argv[1]).resolve()
    if HERE not in stage.parents: raise ValueError('Stage must be inside baseline_algo')
    objects = [value for value in sys.argv[2:] if value in ('A1-f', 'B', 'C5')]
    if not objects: objects = ['A1-f', 'B', 'C5']
    install(objects)
    sys.argv = [str(stage)]+sys.argv[2:]
    runpy.run_path(str(stage), run_name='__main__')
