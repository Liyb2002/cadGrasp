"""Monotone interval bounds for joint coverage / total contact area."""
import math
import heapq
import numpy as np
from step2_local_support import circles as P
from step3_scheculer import floor_support as F
from step3_scheculer import passive_support as U

EFFICIENCY_REL_TOL = 1e-3
RADIUS_REL_TOL = 1e-5  # relative to the original radius, not a manufacturing limit


def efficiency(row):
    """Counts / m² has the same maximizer as coverage fraction / m²."""
    if row['area_m2'] <= 0 or not math.isfinite(row['area_m2']) or row['covered_count'] < 0:
        raise ValueError('Efficiency requires positive finite area and nonnegative coverage')
    return row['covered_count']/row.get('total_area_m2', row['area_m2'])


def maximize_efficiency(evaluate, radii, initial_radius, radius_tolerance,
                        relative_tolerance=EFFICIENCY_REL_TOL, max_evaluations=512):
    """Bound a non-unimodal ratio using monotone numerator and denominator.

    For a <= r <= b, count(r)/area(r) <= count(b)/area(a).
    Subdivide the interval with the highest upper bound. Never require the
    ratio itself to be monotone, differentiable, or unimodal. Retain the
    original radius unless an evaluated radius strictly improves efficiency.
    """
    radii = sorted(set(map(float, radii)) | {float(initial_radius)})
    if len(radii) < 2 or radii[0] <= 0 or radius_tolerance <= 0:
        raise ValueError('Expected a positive radius interval and tolerance')
    if not 0 < relative_tolerance < 1 or max_evaluations < len(radii):
        raise ValueError('Invalid efficiency tolerance or evaluation budget')
    rows = {}
    best_radius = float(initial_radius)

    def value(radius):
        if radius not in rows:
            rows[radius] = evaluate(radius)
        return efficiency(rows[radius])

    best = value(best_radius)
    original_value = best

    def visit(radius):
        nonlocal best, best_radius
        candidate = value(radius)
        if candidate > best*(1+1e-12):
            best, best_radius = candidate, radius

    for radius in radii:
        visit(radius)
    queue = []
    terminal = []

    def interval(a, b):
        low, high = rows[a], rows[b]
        if low['covered_count'] > high['covered_count'] or low['area_m2'] > high['area_m2']*(1+1e-10):
            raise RuntimeError('Interval bound requires nested coverage and area')
        return high['covered_count']/low.get('total_area_m2', low['area_m2'])

    def push(a, b):
        heapq.heappush(queue, (-interval(a, b), a, b))

    for a, b in zip(radii[:-1], radii[1:]):
        push(a, b)
    subdivisions = 0
    while queue:
        negative_bound, a, b = heapq.heappop(queue)
        bound = -negative_bound
        if bound <= best*(1+relative_tolerance):
            terminal.append((a, b, bound, 'efficiency_bound'))
        elif b-a <= radius_tolerance:
            terminal.append((a, b, bound, 'radius_resolution'))
        elif len(rows) >= max_evaluations:
            heapq.heappush(queue, (negative_bound, a, b))
            break
        else:
            middle = (a+b)/2
            visit(middle)
            push(a, middle)
            push(middle, b)
            subdivisions += 1
    terminal.extend((a, b, -bound, 'evaluation_budget') for bound, a, b in queue)
    upper = max([best]+[row[2] for row in terminal])
    gap = (upper-best)/best if best > 0 else 0.
    report = dict(method='maximize_coverage_divided_by_actual_contact_area',
        original_value_count_per_m2=original_value, best_value_count_per_m2=best,
        upper_bound_count_per_m2=upper, relative_gap=gap,
        relative_tolerance=relative_tolerance, converged=gap <= relative_tolerance+1e-12,
        evaluated_sizes=len(rows), subdivisions=subdivisions, max_evaluations=max_evaluations,
        improved=best > original_value*(1+1e-12), radius_tolerance_m=radius_tolerance,
        bounds=[dict(lower_radius_m=a, upper_radius_m=b, upper_count_per_m2=bound, reason=reason)
                for a, b, bound, reason in sorted(terminal)])
    assert best >= original_value
    return best_radius, report


def patch_columns(domain, patch, floor, scale):
    rows = []
    for face, points in patch.items():
        normal = -domain.mesh.face_normals[face]
        rows.extend(np.c_[np.tile(normal, (len(points), 1)),
                          np.cross(points-domain.com, normal)])
    full = np.vstack([U.heads(rows, scale), U.floor(F.columns(floor, domain.com), scale)])
    ids = np.unique(np.round(full, 13), axis=0, return_index=True)[1]
    return np.ascontiguousarray(full[np.sort(ids)])


def feasible_radius_floor(lower, initial, tolerance, feasible):
    """Keep a feasible upper endpoint while shrinking nested contact regions."""
    if not feasible(initial):
        raise ValueError('Selected initial contact fails the hard rest-equilibrium constraint')
    original=lower
    if feasible(lower):
        return lower, dict(enforced=True,minimum_radius_m=lower,iterations=0)
    iterations=0
    while initial-lower>tolerance:
        mid=(lower+initial)/2
        if feasible(mid): initial=mid
        else: lower=mid
        iterations+=1
    return initial,dict(enforced=True,minimum_radius_m=initial,iterations=iterations,
                        original_area_radius_floor_m=original,last_infeasible_radius_m=lower)


def triangulate(patch):
    triangles, faces = [], []
    for face, polygon in patch.items():
        pieces = P.fan(polygon)
        triangles.extend(pieces)
        faces.extend([face]*len(pieces))
    return np.asarray(triangles).reshape(-1, 3, 3), np.asarray(faces, int)
