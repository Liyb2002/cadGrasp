"""Accept a new best physical gap or monotone progress toward one frozen goal."""
import numpy as np


def unlock_decision(before,after,best,goal_best,geometry_best,geometry_after):
    values=np.array([before,after,best,goal_best,geometry_best,geometry_after])
    if not np.isfinite(values).all() or np.min(values)<-1e-12:raise ValueError('finite nonnegative progress values required')
    threshold=min(best,goal_best)
    force=after<threshold-max(1e-12,threshold*1e-4)
    geometry=geometry_after<geometry_best-max(1e-7,geometry_best*1e-4)
    reason='force' if force else 'geometry' if geometry else 'reject'
    return dict(accepted=reason!='reject',acceptance_reason=reason,new_best_force=bool(force),
        force_improved=bool(after<before-max(1e-12,before*1e-4)),geometry_improved=bool(geometry),
        force_loss_excess_above_best=float(max(0.,after-best)))
