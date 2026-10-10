"""Classify a finite demand batch with reusable primal and dual LP evidence.

No high-dimensional hull or perturbed generators are needed. A feasible LP
basis certifies other targets with nonnegative coefficients. A separating
plane certifies other infeasible targets. Remaining targets get a fresh LP
with ALL old and new reaction columns available.
"""
import numpy as np
from scipy.optimize import linprog

from step3_scheculer.stage_imports import load_stage

C = load_stage('score', 'contribution')
def classify(full, targets, known_covered=None):
    targets = C.U.target(targets, full.shape[1])
    accepted = np.zeros(len(targets), bool) if known_covered is None else np.array(known_covered, bool, copy=True)
    assert accepted.shape == (len(targets),)
    pending = ~accepted
    info = dict(method='joint_lp_with_reusable_primal_dual_certificates',
                known_feasible_samples=int(accepted.sum()), equilibrium_lps=0, separation_lps=0,
                feasible_bases=0, separating_planes=0, batch_accepted=0, batch_rejected=0)
    options = dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)
    while pending.any():
        remaining = np.flatnonzero(pending)
        # Stored IID sample order is fixed, so the search is reproducible.
        index = int(remaining[0])
        witness = C.W.solve(full, targets[index])
        info['equilibrium_lps'] += 1
        pending[index] = False
        if witness is not None:
            accepted[index] = True
            basis = full[witness['indices']]
            if not len(basis):
                continue
            # Pseudoinverse handles lower-dimensional bases too. Check actual
            # reconstruction in the original conditioned six coordinates.
            coefficients = targets[remaining]@np.linalg.pinv(basis)
            reconstructed = coefficients@basis
            certified = (coefficients >= 0).all(axis=1) & (
                np.max(np.abs(reconstructed-targets[remaining]), axis=1) <= C.FEASIBILITY_TOL)
            accepted[remaining[certified]] = True
            pending[remaining[certified]] = False
            info['feasible_bases'] += 1
            info['batch_accepted'] += int(certified.sum())
        else:
            # h.g <= 0 for every generator and h.target > 0 separates the
            # target from the complete sum cone. This is not a residual
            # calculated with the old contact's reaction coefficients frozen.
            result = linprog(-targets[index], A_ub=full, b_ub=np.zeros(len(full)),
                             bounds=[(-1, 1)]*full.shape[1], method='highs', options=options)
            info['separation_lps'] += 1
            if not result.success or np.linalg.norm(result.x) < 1e-12:
                continue  # the independent equilibrium LP decided this sample
            normal = result.x/np.linalg.norm(result.x)
            if np.max(full@normal) > 1e-12:
                continue  # never batch-reject from an invalid separating plane
            certified = targets[remaining]@normal > 1e-8
            pending[remaining[certified]] = False
            info['separating_planes'] += 1
            info['batch_rejected'] += int(certified.sum())
    info['covered_count'] = int(accepted.sum())
    return accepted, info
