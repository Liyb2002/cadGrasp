"""Balanced high-level sampling, then gradient repair of the selected pair."""
import json
from copy import copy
import numpy as np
from scipy.spatial.transform import Rotation

from subdivide_task.operations.state import transform
from subdivide_task.operations.pose_change import pose_change
from .optimize import record
from .gradient_repair import repair
from .spatial_search import floor_repair

KINDS = ('juxtaposition', 'rotational_reuse')


def projection_model(model, before, after):
    """Preserve source-surface contacts in the proposed object's WORLD pose."""
    local = copy(model)
    local.current_contacts = {}
    cached = getattr(model, 'current_contacts', {})
    for i, name in enumerate(model.ids):
        if name in cached:
            local.current_contacts[name] = transform(cached[name],
                after.poses[i]@np.linalg.inv(before.poses[i]))
    return local


class BalancedSchedule:
    """Random order within pairs: marginal P=1/2, count difference <= 1."""
    def __init__(self, rng, pending=()):
        self.rng, self.pending = rng, list(pending)

    def next(self):
        if not self.pending:
            self.pending = list(self.rng.permutation(KINDS))
        return str(self.pending.pop())


def propose(model, layout, kind, owner, host, rng, sample):
    q = layout.copy()
    center = model.source.center_mass
    target = transform(center, q.material[host]@q.poses[host])
    offset = np.zeros(3) if sample == 0 else rng.normal(0., .012, 3)
    if kind == 'juxtaposition':
        # Only seat position changes. Partial overlap is a sampled target,
        # not an independently accepted intermediate state.
        own = transform(center, q.material[owner]@q.poses[owner])
        fraction = 1. if sample == 0 else float(rng.uniform(.55, 1.))
        q.material[owner, :3, 3] += fraction*(target-own)+offset
    elif kind == 'rotational_reuse':
        # A free legal world pose determines a full SO(3) installation that
        # shares the host's object seat. No fixed-pose or yaw-only constraint.
        angle = float(rng.uniform(-180., 180.))
        q = pose_change(q, owner, angle, model.axis, model.head)
        desired = q.material[host]@q.poses[host]@np.linalg.inv(q.poses[owner])
        desired[:3, 3] += offset
        q.material[owner] = desired
    else:
        raise ValueError('Unknown sharing operation: '+kind)
    return q


def solve(optimizer, rounds=8, samples=3, repair_rounds=2, seed=42):
    if rounds < 1 or samples < 2 or repair_rounds < 1:
        raise ValueError('Use positive rounds, at least two samples and positive repair rounds')
    model = optimizer.model
    path = optimizer.folder/'sharing_search.json'
    saved = json.loads(path.read_text()) if path.exists() else {}
    rng = np.random.default_rng(seed)
    if 'rng_state' in saved:
        rng.bit_generator.state = saved['rng_state']
    schedule = BalancedSchedule(rng, saved.get('pending', ()))
    counts = saved.get('proposal_counts', {kind: 0 for kind in KINDS})
    successes = saved.get('accepted_counts', {kind: 0 for kind in KINDS})
    first_round = saved.get('completed_rounds', 0)
    for iteration in range(first_round, first_round+rounds):
        kind = schedule.next()
        counts[kind] += 1
        optimizer.event('sharing_round_started', kind, iteration=iteration+1,
            selection_probability=.5, samples=samples, proposal_counts=dict(counts))
        pairs = [(i, j) for i in range(len(model.ids)) for j in range(len(model.ids)) if i != j]
        centers = np.array([transform(model.source.center_mass, M@P)
            for M,P in zip(optimizer.layout.material, optimizer.layout.poses)])
        weights = np.array([1./(.08+np.linalg.norm(centers[i]-centers[j])) for i,j in pairs])
        weights /= weights.sum()
        selected = rng.choice(len(pairs), min(samples, len(pairs)), replace=False, p=weights)
        accepted = False
        for sample, index in enumerate(selected):
            owner, host = pairs[index]
            tasks = [model.ids[owner], model.ids[host]]
            try:
                q = propose(model, optimizer.layout, kind, owner, host, rng, sample)
                # Project physical floor constraints before differentiating;
                # each task remains independently movable during repair.
                projected = projection_model(model, optimizer.layout, q)
                q = floor_repair(projected, q, [[k] for k in range(len(model.ids))], host)
                if q is None:
                    raise ValueError('Sample has incompatible installation floors')
                q, detail = repair(model, q, [owner, host], repair_rounds, optimizer.event)
                if q is None:
                    raise ValueError('Pair geometry cannot be built for gradient repair')
                # The full gate also checks tasks outside the repaired pair.
                if optimizer.try_operation(q, kind, tasks=tasks, iteration=iteration+1,
                        sample=sample+1, gradient_repair=detail,
                        high_level_sampling_probability=.5):
                    successes[kind] += 1
                    accepted = True
            except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as error:
                optimizer.event('sharing_sample_rejected', kind, tasks=tasks,
                    iteration=iteration+1, sample=sample+1, reason=str(error)[:900])
        record(path, dict(method='balanced_sharing_then_pair_gradient_repair',
            completed_rounds=iteration+1, proposal_counts=counts, accepted_counts=successes,
            samples_per_operation=samples, repair_rounds=repair_rounds,
            marginal_operation_probability={kind: .5 for kind in KINDS},
            pending=schedule.pending, rng_state=rng.bit_generator.state,
            pso_used=False, only_fully_certified_states_committed=True))
        optimizer.event('sharing_round_finished', kind, iteration=iteration+1,
            accepted_any_sample=accepted, current_volume_cm3=optimizer.volume)
    return optimizer.layout, optimizer.support
