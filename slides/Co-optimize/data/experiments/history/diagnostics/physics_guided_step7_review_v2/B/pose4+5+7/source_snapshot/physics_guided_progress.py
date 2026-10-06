"""Accept physics-informed geometric progress without declaring feasibility.

Compare geometric costs with the SAME frozen nonnegative contact sensitivities.
Bound excursions in true cone loss relative to the best constructed state.
"""
import numpy as np


def progress_decision(before, after, best, geometry_before, geometry_after):
    values = np.asarray([before, after, best, geometry_before, geometry_after])
    if not np.isfinite(values).all() or np.min(values) < -1e-12:
        raise ValueError('finite nonnegative progress measures required')
    force_improved = after < before - max(1e-12, before * 1e-4)
    geometry_improved = geometry_after < geometry_before - max(1e-7, geometry_before * 1e-4)
    # This is a bounded intermediate step, not relaxed final acceptance.
    ceiling = best + max(1e-4, .10 * best)
    bounded = after <= ceiling + 1e-12
    reason = 'force' if force_improved else ('geometry' if geometry_improved and bounded else 'reject')
    return dict(accepted=reason != 'reject', acceptance_reason=reason,
                force_improved=bool(force_improved), geometry_improved=bool(geometry_improved),
                bounded_force_excursion=bool(bounded), force_loss_ceiling=float(ceiling))
