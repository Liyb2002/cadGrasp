"""Test discontinuous ground contact without changing a saved fixture state.

No solid rim is required.  Tests use every floor-compatible contact location,
with no bounding box, thickness, connection, work, or withdrawal restriction.
This deliberately generous model can reject a layout, but cannot certify a
complete physical base.  Object-floor contacts are included independently.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

TOL = 1e-9


def floor_inequalities(states, index):
    """In task i ground coordinates: A @ xy + b >= 0 for every task floor."""
    inverse = np.linalg.inv(states[index])
    transforms = np.array([state @ inverse for state in states])
    return transforms[:, 2, :2], transforms[:, 2, 3]


def clip_boundary(hull, a, b):
    """Keep legal subsegments of the demand boundary; never add rim material."""
    hull = np.asarray(hull)[:, :2]
    segments, records = [], []
    for index, (first, last) in enumerate(zip(hull, np.roll(hull, -1, axis=0))):
        direction = last - first
        low, high = 0., 1.
        for value, slope in zip(a @ first + b, a @ direction):
            if abs(slope) <= 1e-15:
                if value < -TOL:
                    high = -1.
                    break
                continue
            bound = -float(value) / float(slope)
            if slope > 0:
                low = max(low, bound)
            else:
                high = min(high, bound)
        if low <= high and high >= 0 and low <= 1:
            low, high = max(0., low), min(1., high)
            segment = np.array([first + low * direction, first + high * direction])
            segments.append(segment)
            records.append(dict(edge_index=index, retained_parameter_interval=[low, high]))
    return np.asarray(segments).reshape(-1, 2, 2), records


def convex_vertices(points):
    points = np.unique(np.asarray(points).reshape(-1, 2), axis=0)
    if len(points) <= 1:
        return points
    centered = points - points.mean(axis=0)
    _, values, axes = np.linalg.svd(centered, full_matrices=False)
    if len(values) < 2 or values[1] <= TOL:
        coordinate = centered @ axes[0]
        return points[[np.argmin(coordinate), np.argmax(coordinate)]]
    return points[ConvexHull(points).vertices]


def covered(points, contacts, tolerance=TOL):
    """Vectorized containment in the convex hull of sparse contacts."""
    points = np.asarray(points).reshape(-1, 2)
    boundary = convex_vertices(contacts)
    if not len(boundary):
        return np.zeros(len(points), dtype=bool)
    if len(boundary) == 1:
        return np.linalg.norm(points - boundary[0], axis=1) <= tolerance
    if len(boundary) == 2:
        vector = boundary[1] - boundary[0]
        length2 = float(vector @ vector)
        if length2 <= tolerance**2:
            return np.linalg.norm(points - boundary[0], axis=1) <= tolerance
        fraction = np.clip((points - boundary[0]) @ vector / length2, 0., 1.)
        return np.linalg.norm(points - boundary[0] - fraction[:, None] * vector, axis=1) <= tolerance
    equations = ConvexHull(boundary).equations
    return np.all(points @ equations[:, :2].T + equations[:, 2] <= tolerance, axis=1)


def unlimited_contact_test(point, a, b, own_ground):
    """Independent LP over the unbounded legal floor region and object contact.

    Let lambda be total fixture-ground normal-reaction weight and t the
    weighted fixture contact xy.  Since the legal floor region is convex,
    all fixture feet can be represented by one mean q=t/lambda:

       t + sum(w_k o_k) = point,  lambda + sum(w_k) = 1,
       A t + b lambda >= 0,  lambda,w_k >= 0.

    Maximize lambda.  Infeasibility rejects ANY number or extent of feet.
    At lambda=0 this formulation admits recession limits: those are reported
    separately, never claimed as a finite realizable contact arrangement.
    An optimal positive lambda gives an explicit finite convex combination.
    Coordinates are scaled to millimetres for numerical conditioning.
    """
    point = np.asarray(point)[:2] * 1000
    own = np.unique(np.asarray(own_ground).reshape(-1, 2), axis=0) * 1000
    a, b = np.asarray(a), np.asarray(b) * 1000
    count = 3 + len(own)
    objective = np.zeros(count)
    objective[2] = -1.
    equality = np.zeros((3, count))
    equality[0, 0] = equality[1, 1] = equality[2, 2] = 1.
    equality[:2, 3:] = own.T
    equality[2, 3:] = 1.
    inequality = np.zeros((len(b), count))
    inequality[:, :2] = -a
    inequality[:, 2] = -b
    result = linprog(objective, A_ub=inequality, b_ub=np.zeros(len(b)),
                     A_eq=equality, b_eq=np.r_[point, 1.],
                     bounds=[(None, None), (None, None)] + [(0., 1.)] * (count - 2),
                     method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                              'dual_feasibility_tolerance': 1e-9})
    if result.status == 2:
        return dict(unlimited_relaxation_feasible=False, finite_contact_witness=False,
                    status='infeasible_even_with_unlimited_legal_feet')
    if not result.success:
        raise RuntimeError(f'Unresolved contact LP: {result.message}')
    error = max(float(np.max(np.abs(equality @ result.x - np.r_[point, 1.]))),
                float(np.max(inequality @ result.x, initial=0.)))
    if error > 1e-6:
        raise RuntimeError(f'Contact LP residual exceeds tolerance: {error} mm')
    weight = float(result.x[2])
    answer = dict(unlimited_relaxation_feasible=True, fixture_ground_weight=weight,
                  object_ground_weights=result.x[3:].tolist(), object_ground_xy_m=(own/1000).tolist(),
                  equation_and_floor_residual_mm=error)
    if weight > 1e-9:
        foot = result.x[:2] / weight / 1000
        reconstructed = foot * weight + (own/1000).T @ result.x[3:]
        if np.max(np.abs(reconstructed-point/1000)) > TOL:
            raise RuntimeError('Finite ground witness reconstruction failed')
        answer.update(finite_contact_witness=True, status='finite_ground_convex_combination',
                      mean_fixture_contact_xy_m=foot.tolist())
    elif bool(covered([point/1000], own/1000)[0]):
        answer.update(finite_contact_witness=True, status='object_ground_hull_already_covers_demand')
    else:
        answer.update(finite_contact_witness=False, status='infinite_recession_limit_only')
    return answer


def separating_floor(hull, a, b, own_ground):
    """Human-readable separating line, including all object-floor contribution."""
    hull = np.asarray(hull)[:, :2]
    own = np.asarray(own_ground).reshape(-1, 2)
    best = None
    for index, (normal, offset) in enumerate(zip(a, b)):
        lower = min(0., float(np.min(own @ normal + offset))) if len(own) else 0.
        values = hull @ normal + offset
        vertex = int(np.argmin(values))
        gap = lower - float(values[vertex])
        if gap > TOL and (best is None or gap > best['gap_m']):
            best = dict(other_task_index=index, demand_vertex_index=vertex,
                        demand_xy_m=hull[vertex].tolist(), normal_xy=normal.tolist(),
                        offset_m=float(offset), attainable_lower_bound_m=lower,
                        gap_m=gap)
    return best


def analyze(row, states, index):
    a, b = floor_inequalities(states, index)
    hull = row['hull'][:, :2]
    own = row['object_ground']
    segments, records = clip_boundary(hull, a, b)
    contacts = np.vstack([segments.reshape(-1, 2), own])
    boundary_length = float(np.linalg.norm(hull-np.roll(hull, -1, axis=0), axis=1).sum())
    retained_length = float(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1).sum())
    mask = covered(row['cloud'][:, :2], contacts)
    hull_mask = covered(hull, contacts)
    tests = [unlimited_contact_test(point, a, b, own) for point in hull]
    infeasible = [i for i, test in enumerate(tests) if not test['unlimited_relaxation_feasible']]
    return dict(pose=row['pose'], demand_count=len(mask), hull_vertex_count=len(hull),
                floor_normal_coefficients_xy=a.tolist(), floor_offsets_m=b.tolist(),
                boundary_length_m=boundary_length, retained_boundary_length_m=retained_length,
                retained_boundary_fraction=retained_length/boundary_length,
                retained_segments_xy_m=segments.tolist(), retained_segments_source=records,
                retained_contact_convex_hull_xy_m=convex_vertices(contacts).tolist(),
                retained_parts_covered_demand_count=int(mask.sum()),
                retained_parts_uncovered_demand_count=int((~mask).sum()),
                retained_parts_cover_entire_hull=bool(hull_mask.all()),
                unlimited_legal_ground_lp_by_hull_vertex=tests,
                unlimited_ground_rejected_hull_vertex_indices=infeasible,
                unlimited_ground_necessary_coverage_passed=not infeasible,
                unlimited_ground_finite_witness_for_every_vertex=all(t['finite_contact_witness'] for t in tests),
                strongest_separating_floor=separating_floor(hull, a, b, own),
                bounding_box_limit_applied=False, continuous_material_rim_required=False,
                connection_thickness_work_and_exit_exclusions_checked=False,
                stage4_result_or_stage51_demands_changed=False)
