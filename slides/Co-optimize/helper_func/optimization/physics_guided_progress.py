"""Accept physics-informed geometric progress without declaring feasibility.

Compare geometric costs with the SAME frozen nonnegative contact sensitivities.
Record true-loss excursions; retain the best constructed state in the solver.
"""
import numpy as np


def progress_decision(before, after, best, geometry_before, geometry_after):
    values = np.asarray([before, after, best, geometry_before, geometry_after])
    if not np.isfinite(values).all() or np.min(values) < -1e-12:
        raise ValueError('finite nonnegative progress measures required')
    force_improved = after < before - max(1e-12, before * 1e-4)
    geometry_improved = geometry_after < geometry_before - max(1e-7, geometry_before * 1e-4)
    # A contact can disappear discontinuously even under an arbitrarily small
    # rotation. Do not veto a necessary unlock by an invented force threshold.
    # Geometry steps are never a declaration of final physical feasibility.
    reason = 'force' if force_improved else ('geometry' if geometry_improved else 'reject')
    return dict(accepted=reason != 'reject', acceptance_reason=reason,
                force_improved=bool(force_improved), geometry_improved=bool(geometry_improved),
                force_loss_excess_above_best=float(max(0.,after-best)))
