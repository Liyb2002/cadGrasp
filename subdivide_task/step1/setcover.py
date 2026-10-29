"""Set covers over candidate masks: the fewest columns covering a share of the target's area.

Greedy by newly covered area first; then an integer program (scipy's HiGHS) within a time limit, kept only
when it needs fewer columns and its cover checks out face by face.  Areas enter as fractions of the target.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity, vstack


def covered(masks, chosen, area, target):
    """Share of the target's area the chosen columns cover."""
    total = float(area[target].sum())
    if not len(chosen) or total <= 0: return 0.
    return float(area[target & masks[np.asarray(chosen, int)].any(0)].sum()/total)


def greedy(masks, area, target, share=1.):
    left = target.copy(); need = share*float(area[target].sum()); done = 0.; chosen = []
    while done < need*(1-1e-12):
        gains = masks@(area*left); k = int(np.argmax(gains))
        if gains[k] <= 0: break
        chosen.append(k); done += float(gains[k]); left &= ~masks[k]
    return np.asarray(chosen, int)


def integer(masks, area, target, share, fewer_than, seconds):
    """Fewer than `fewer_than` columns covering `share` of the target area, or None."""
    faces = np.flatnonzero(target & masks.any(0)); cols = np.flatnonzero(masks[:, faces].any(1))
    weight = area[faces]/float(area[target].sum()); n, m = len(cols), len(faces)
    M = csr_matrix(masks[np.ix_(cols, faces)].T.astype(float))
    # x: columns (binary), y: faces (0..1);  y_f <= sum of the columns over f;  sum w_f y_f >= share
    rows = vstack([hstack([-M, identity(m, format='csr')]),
                   hstack([csr_matrix((1, n)), csr_matrix(weight[None, :])]),
                   hstack([csr_matrix(np.ones((1, n))), csr_matrix((1, m))])]).tocsc()
    lower = np.r_[np.full(m, -np.inf), share-1e-9, -np.inf]; upper = np.r_[np.zeros(m), np.inf, fewer_than-1]
    r = milp(np.r_[np.ones(n), np.zeros(m)], integrality=np.r_[np.ones(n), np.zeros(m)], bounds=Bounds(0, 1),
             constraints=LinearConstraint(rows, lower, upper), options=dict(time_limit=float(seconds), disp=False))
    return None if r.x is None else cols[r.x[:n] > .5]


def cover(masks, area, target, share=1., seconds=30.):
    """Few columns covering `share` of the target area; target faces no column covers are left out of it."""
    masks = np.asarray(masks, bool); area = np.asarray(area, float); target = np.asarray(target, bool)&masks.any(0)
    best = greedy(masks, area, target, share)
    if len(best) > 1 and seconds > 0:
        reached = covered(masks, best, area, target)
        better = integer(masks, area, target, min(share, reached), len(best), seconds)
        if better is not None and covered(masks, better, area, target) >= min(share, reached)-1e-9: best = better
    return np.asarray(best, int)
