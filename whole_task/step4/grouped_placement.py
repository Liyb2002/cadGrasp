"""Fit group installation translations to shared floor and occupancy constraints.

No task is rehosted here. Whole groups move in material coordinates while all
world object poses remain unchanged. A linear program aligns each group's
floor with the continuous pressure hulls and minimizes an execution-space box.
This is a placement proposal only; actual geometry and all forces certify it.
"""
import numpy as np
from scipy.optimize import linprog

from .grouped_search import memberships, move_group_material


def fit_groups(model, layout, margin=.002, preserve=False):
    model.activate(layout)
    groups = memberships(layout); hosts = list(groups); count = len(hosts)
    index = {h: i for i, h in enumerate(hosts)}
    rows = model.base_oracle.rows(layout)
    # Variables: one material displacement per installation, common executed
    # AABB low/high, and absolute displacement used only as a small tie-break.
    variables = 3*count+6+3*count
    low = 3*count; high = low+3; absolute = high+3
    inequalities, limits = [], []
    for row in rows:
        g = index[int(layout.hosts[row['index']])]
        inverse = np.linalg.inv(row['T'])
        hull = np.c_[row['hull'], np.zeros(len(row['hull']))]
        points = hull @ inverse[:3, :3].T+inverse[:3, 3]
        for h in hosts:
            j = index[h]
            if g == j:
                continue
            other = next(r for r in rows if int(layout.hosts[r['index']]) == h)
            normal = other['T'][2, :3]
            heights = points @ normal+other['T'][2, 3]
            # Strong sufficient floor compatibility: pressure hull itself
            # lies in every other halfspace, without borrowing object contact.
            a = np.zeros(variables)
            a[3*j:3*j+3] += normal
            a[3*g:3*g+3] -= normal
            inequalities.append(a); limits.append(float(heights.min())-margin)
    execution = {h: np.asarray(model.execution(layout, groups[h][0])['world_from_support']) for h in hosts}
    clouds = {h: np.vstack([model.mesh.vertices @ layout.placements[k, :3, :3].T+layout.placements[k, :3, 3]
                           for k in groups[h]]) for h in hosts}
    for h in hosts:
        i = index[h]; t = execution[h]
        for guest in hosts:
            j = index[guest]
            cloud = clouds[guest] @ t[:3, :3].T+t[:3, 3]
            for axis in range(3):
                coefficients = np.zeros(variables)
                coefficients[3*j:3*j+3] += t[axis, :3]
                coefficients[3*i:3*i+3] -= t[axis, :3]
                upper = coefficients.copy(); upper[high+axis] -= 1.
                lower = -coefficients; lower[low+axis] += 1.
                inequalities.extend([upper, lower])
                limits.extend([-float(cloud[:, axis].max()), float(cloud[:, axis].min())])
    for k in range(3*count):
        positive = np.zeros(variables); positive[k] = 1.; positive[absolute+k] = -1.
        negative = np.zeros(variables); negative[k] = -1.; negative[absolute+k] = -1.
        inequalities.extend([positive, negative]); limits.extend([0., 0.])
    objective = np.zeros(variables)
    # First-order box-volume objective, balanced by the current body extents.
    widths = np.maximum(np.ptp(model.mesh.vertices, axis=0), .05)
    objective[low:low+3] = -1./widths
    objective[high:high+3] = 1./widths
    objective[absolute:] = 100. if preserve else .01
    bounds = [(None, None)]*variables
    bounds[:3] = [(0., 0.)]*3  # Material-origin gauge only; not an installation lock.
    bounds[absolute:] = [(0., None)]*(3*count)
    answer = linprog(objective, A_ub=np.array(inequalities), b_ub=np.array(limits),
                     bounds=bounds, method='highs')
    if not answer.success:
        raise ValueError('Group floor/box placement LP: '+answer.message)
    q = layout.copy()
    for h in hosts:
        j = index[h]
        q = move_group_material(model, q, h, answer.x[3*j:3*j+3])
    return q, dict(method='joint_group_floor_and_execution_box_linear_program',
        margin_m=margin, prefer_minimum_translation=preserve, pressure_hull_floor_constraints=len(rows)*(count-1),
        objective=float(answer.fun), surrogate_dimensions_m=(answer.x[high:high+3]-answer.x[low:low+3]).tolist(),
        material_group_displacements_m={model.poses[h]: answer.x[3*index[h]:3*index[h]+3].tolist() for h in hosts},
        final_physical_certificate_required=True)
