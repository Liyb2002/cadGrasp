"""Floor legality of object support contacts in every installed fixture state.

No geometric safety margin is required. All heights use shared fixture
coordinates and final states only; numerical tolerances remain.
"""
import numpy as np

from fixture_state import installed_states

CONTACT_CLEARANCE_M = 0.
HEIGHT_TOL_M = 1e-10


def floor_states(model, layout):
    hosts = sorted(set(int(layout.hosts[k]) for k in layout.active))
    return installed_states(model.native, layout)[hosts, 2, :]


def contact_heights(model, layout, owner):
    Q = layout.placements[owner]
    common = model.points@Q[:3, :3].T+Q[:3, 3]
    floors = floor_states(model, layout)
    return common@floors[:, :3].T+floors[:, 3]


def allowed_contacts(model, layout, owner):
    margin = model.support_floor_clearance_m
    return np.all(contact_heights(model, layout, owner) >= margin-HEIGHT_TOL_M, axis=1)


def contact_event_step(model, layout, owners, axis, sign, minimum):
    """Resolve the nearest floor-contact event along any world XYZ axis.

    A small difference can remain inside a flat contact cell. Enlarge it only
    to the first entry/exit of the exact all-floor contact region, bounded by
    two contact margins. Evaluate both gains and losses with the same objective.
    """
    margin = model.support_floor_clearance_m
    limit = max(minimum, 2*margin)
    nearest = np.inf
    floors = floor_states(model, layout)
    for owner in owners:
        heights = contact_heights(model, layout, owner)
        world_axis = np.eye(3)[axis]*sign
        vector = model.native[layout.hosts[owner], :3, :3].T@world_axis
        rates = floors[:, :3]@vector
        positive, negative, static = rates > 1e-12, rates < -1e-12, np.abs(rates) <= 1e-12
        low = np.full(len(heights), -np.inf)
        high = np.full(len(heights), np.inf)
        if positive.any():
            low = np.max((margin-heights[:, positive])/rates[positive], axis=1)
        if negative.any():
            high = np.min((margin-heights[:, negative])/rates[negative], axis=1)
        feasible = np.all(heights[:, static] >= margin-HEIGHT_TOL_M, axis=1) if static.any() else np.ones(len(heights), bool)
        feasible &= low <= high
        # Entry into the region and exit from it both change the reaction cone.
        entry = feasible & (low > 0.) & (low <= limit)
        exit_ = feasible & (low <= 0.) & (high > 0.) & (high <= limit)
        if entry.any(): nearest = min(nearest, float(low[entry].min()))
        if exit_.any(): nearest = min(nearest, float(high[exit_].min()))
    if not np.isfinite(nearest):
        return minimum
    return max(minimum, min(limit, nearest+max(1e-7, nearest*1e-4)))
