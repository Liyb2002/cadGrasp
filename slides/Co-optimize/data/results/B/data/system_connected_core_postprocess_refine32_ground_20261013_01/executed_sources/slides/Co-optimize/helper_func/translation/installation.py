"""Move a whole final fixture state and all its tasks in world XYZ."""
import numpy as np
from direction_choice.r6_gradient import gradient_loss


def shift_installation(model, layout, hosts, vector):
    trial = layout.copy()
    hosts = tuple(sorted(set(map(int, hosts))))
    members = [k for k in layout.active if int(layout.hosts[k]) in hosts]
    vector = np.asarray(vector, dtype=float).copy()
    # Project only the nonpenetration boundary. No preferred XYZ axis.
    if members:
        height = max(0., min(model.workpiece_height(layout, k) for k in members))
        vector[2] = max(vector[2], -height)
    trial.state_offsets[list(hosts)] += vector
    return trial


def installation_gradient(model, layout, hosts, delta, objective=None):
    objective = objective or (lambda q: gradient_loss(model, q))
    gradient = np.zeros(3)
    for axis in range(3):
        vector = delta*np.eye(3)[axis]
        minus = shift_installation(model, layout, hosts, -vector)
        plus = shift_installation(model, layout, hosts, vector)
        distance = plus.state_offsets[hosts[0], axis]-minus.state_offsets[hosts[0], axis]
        if distance > 1e-15:
            gradient[axis] = (objective(plus)-objective(minus))/distance
    if min(model.workpiece_height(layout, k) for k in layout.active
           if int(layout.hosts[k]) in hosts) <= 1e-9:
        gradient[2] = min(gradient[2], 0.)
    return gradient


def installation_proposals(model, layout, *, fine=False, objective=None):
    hosts = sorted(set(int(layout.hosts[k]) for k in layout.active))
    groups = [(h,) for h in hosts]
    if len(hosts) > 1:
        groups.append(tuple(hosts))
    delta = model.extent*(1/512 if fine else 1/100)
    rows = []; seen = {layout.key()}
    for group in groups:
        gradient = installation_gradient(model, layout, group, delta, objective)
        vectors = [(sign*axis, 'sample') for axis in np.eye(3) for sign in [-1, 1]]
        norm = np.linalg.norm(gradient)
        if norm > 1e-28:
            vectors.insert(0, (-gradient/norm, 'gradient'))
        fractions = ([1/1024, 1/512, 1/256] if fine else [1/128, 1/32, 1/8])
        if not fine and group == tuple(hosts):
            fractions += [1/4, 1/2, 1.]
        for fraction in fractions:
            for vector, kind in vectors:
                trial = shift_installation(model, layout, group, model.extent*fraction*vector)
                if trial.key() in seen:
                    continue
                seen.add(trial.key())
                rows.append(('fixture-translation-'+kind, trial,
                    dict(hosts=[model.poses[h] for h in group], step_m=model.extent*fraction,
                         gradient=gradient.tolist(), finite_difference_m=delta,
                         coordinate='world_xyz', all_poses_in_objective=True,
                         objects_move_with_installed_fixture=True)))
    return rows
