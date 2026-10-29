"""Strict face-wise set cover: the last small triangle is mandatory too."""
import numpy as np
import time
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack


class IncompleteCoverage(ValueError):
    def __init__(self, face_ids):
        self.face_ids = np.asarray(face_ids, dtype=int)
        super().__init__('Uncovered source faces: '+str(self.face_ids.tolist()))


def greedy(masks, areas, target=None):
    masks = np.asarray(masks, bool)
    areas = np.asarray(areas, float)
    target = np.ones(len(areas), bool) if target is None else np.asarray(target, bool)
    if masks.ndim != 2 or masks.shape[1] != len(areas):
        raise ValueError('Expected candidate-by-face masks')
    missing = target & ~masks.any(axis=0)
    if missing.any():
        raise IncompleteCoverage(np.flatnonzero(missing))
    remaining = target.copy()
    selected = []
    while remaining.any():
        gains = masks @ (areas*remaining)
        k = int(np.argmax(gains))
        if not np.any(masks[k] & remaining):
            raise RuntimeError('No positive coverage gain')
        selected.append(k)
        remaining &= ~masks[k]
    return np.asarray(selected, dtype=int)


def solve(masks, areas, seconds=30., target=None):
    """Integer minimum set cover in this supplied, explicitly identified pool."""
    masks = np.asarray(masks, bool)
    areas = np.asarray(areas, float)
    target = np.ones(len(areas), bool) if target is None else np.asarray(target, bool)
    initial = greedy(masks, areas, target)
    if not target.any():
        return initial, dict(status='optimal', minimum_count=0, lower_bound=0)
    eligible = np.flatnonzero(masks[:, target].any(axis=1))
    selected = initial
    target_ids = np.flatnonzero(target)
    incidence = masks.sum(axis=0)
    # A small set of required faces starts the integer problem. Missing faces
    # are added as constraints until the solution covers ALL original faces.
    # This avoids a large dense presolve for thousands of directional columns.
    rows = np.unique(np.r_[target_ids[np.argsort(incidence[target_ids], kind='stable')[:48]],
                           target_ids[np.linspace(0, len(target_ids)-1, min(48,len(target_ids))).astype(int)]])
    began = time.monotonic()
    lower = 0.
    optimal = False
    result = None
    while time.monotonic()-began < seconds:
        budget = seconds-(time.monotonic()-began)
        A = csc_matrix(masks[eligible][:, rows].T.astype(float))
        A = vstack([A, csc_matrix(np.ones((1, len(eligible))))], format='csc')
        result = milp(np.ones(len(eligible)), integrality=np.ones(len(eligible)),
                      bounds=Bounds(np.zeros(len(eligible)), np.ones(len(eligible))),
                      constraints=LinearConstraint(A, np.r_[np.ones(len(rows)), -np.inf],
                                                   np.r_[np.full(len(rows), np.inf), len(selected)]),
                      options=dict(time_limit=float(min(8.,budget)), mip_rel_gap=0.,
                                   presolve=False, disp=False))
        bound = getattr(result, 'mip_dual_bound', None)
        if bound is not None and np.isfinite(bound):
            lower = max(lower, float(bound))
        if result.x is None:
            continue
        trial = eligible[result.x > .5]
        missing = target & ~masks[trial].any(axis=0)
        if not missing.any():
            if len(trial) <= len(selected):
                selected = trial
            if result.status == 0:
                optimal = True
                break
            # A feasible full cover may improve on another short solve.
            continue
        missing_ids = np.flatnonzero(missing)
        add = (missing_ids if len(missing_ids) <= 128 else
               np.r_[missing_ids[np.argsort(incidence[missing_ids],kind='stable')[:48]],
                     missing_ids[np.linspace(0,len(missing_ids)-1,48).astype(int)]])
        rows = np.unique(np.r_[rows,add])
    if not masks[selected][:, target].any(axis=0).all():
        raise RuntimeError('Solver returned an incomplete cover')
    gap = max(0., (len(selected)-lower)/max(1,len(selected)))
    record = dict(status='optimal' if optimal else 'feasible_time_limited',
                  solver_status=0 if optimal else 1,
                  message='Face-constraint generation; every returned cover checked against all target faces',
                  minimum_count=len(selected) if optimal else None,
                  feasible_count=len(selected), greedy_count=len(initial),
                  lower_bound=float(lower), mip_gap=float(gap), constraint_face_count=len(rows),
                  optimality_scope='only the supplied candidate pool, not continuous SE(3)')
    return np.asarray(selected, int), record


def improve(masks, initial, seconds=30., seed=17):
    """Exchange poses with a smaller cover; accept only strict full coverage."""
    masks = np.asarray(masks,bool)
    best = np.asarray(initial,int).copy()
    if not masks[best].any(0).all():
        raise IncompleteCoverage(np.flatnonzero(~masks[best].any(0)))
    rng = np.random.default_rng(seed)
    count = masks.sum(0).astype(float)
    began = time.monotonic()
    attempts = 0
    while len(best)>1 and time.monotonic()-began<seconds:
        selected = np.delete(best,attempts%len(best)).copy()
        weights = (count.min()/count)**rng.uniform(0.,1.5)
        attempts += 1
        for sweep in range(20):
            changed = False
            for slot in rng.permutation(len(selected)):
                other = np.delete(selected,slot)
                missing = ~masks[other].any(0) if len(other) else np.ones(masks.shape[1],bool)
                gain = masks[:,missing]@weights[missing]
                highest = gain.max()
                choices = np.flatnonzero(gain >= highest-1e-10)
                replacement = int(rng.choice(choices))
                changed |= replacement != selected[slot]
                selected[slot] = replacement
                if masks[selected].any(0).all():
                    best = np.unique(selected)
                    print('SMALLER FULL COVER',len(best),best.tolist(),flush=True)
                    break
                if time.monotonic()-began>=seconds:
                    break
            if len(best) <= len(selected) or time.monotonic()-began>=seconds:
                break
            if not changed or sweep%3 == 2:
                slot = int(rng.integers(len(selected)))
                other = np.delete(selected,slot)
                missing = ~masks[other].any(0)
                gain = masks[:,missing]@weights[missing]
                top = np.argpartition(gain,-min(40,len(gain)))[-min(40,len(gain)):]
                selected[slot] = int(rng.choice(top))
    return best
