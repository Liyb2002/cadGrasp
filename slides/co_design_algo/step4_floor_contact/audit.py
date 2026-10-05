"""Audit current whole-assembly floor demand; retain legacy force-replay helpers."""
import argparse
from fractions import Fraction as FQ
from pathlib import Path
import sys
import numpy as np
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step4_floor_contact import floor_contact as F


def moments(points, forces, origin):
    r = points-origin
    return np.stack([r[..., 1]*forces[..., 2]-r[..., 2]*forces[..., 1],
                     r[..., 2]*forces[..., 0]-r[..., 0]*forces[..., 2],
                     r[..., 0]*forces[..., 1]-r[..., 1]*forces[..., 0]], axis=-1)


def replay(arrays, prefix, loads, count, friction):
    assignment = arrays[prefix+'_assignment']; bases = arrays[prefix+'_basis_indices']
    coefficients = arrays[prefix+'_coefficients_mg']
    assert len(assignment) == len(loads) and np.all(assignment >= 0)
    assert np.min(coefficients) >= -2e-9
    p, n, owner = arrays['contact_points_m'], arrays['contact_normals'], arrays['contact_owners']
    ground_p, ground_f, ground_owner = arrays['ground_points_m'], arrays['ground_forces'], arrays['ground_owners']
    origin, scale = arrays['moment_origin_m'], arrays['scale']
    # Reconstruct every column by force and moment accumulation, independently
    # of the block-matrix assembly used by the solver.
    columns = np.zeros((len(p)+len(ground_p), count+1, 6))
    head = np.c_[n, moments(p, n, origin)]
    columns[:len(p), 0] = head
    for i in range(len(p)):
        if owner[i] >= 0: columns[i, owner[i]+1] = -head[i]
    ground = np.c_[ground_f, moments(ground_p, ground_f, origin)]
    for i in range(len(ground_p)): columns[len(p)+i, ground_owner[i]+1] = ground[i]
    assert np.all(ground_f[:, 2] > 0)
    assert np.all(np.linalg.norm(COORD.floor(ground_f), axis=1) <= friction*ground_f[:, 2]+1e-12)
    if 'equilibrium_matrix' in arrays:
        np.testing.assert_allclose((columns*scale).reshape(len(columns), -1).T,
                                   arrays['equilibrium_matrix'], atol=1e-13, rtol=1e-13)
    maximum = 0.
    for k, basis in enumerate(bases):
        ids = np.flatnonzero(assignment == k)
        if not len(ids): continue
        residual = np.einsum('ni,ibc->nbc', coefficients[ids], columns[basis])
        residual[:, 0] -= loads[ids]
        maximum = max(maximum, float(np.max(np.abs(residual*scale))))
    assert maximum <= 5e-8, maximum
    return dict(load_count=len(loads), maximum_body_equilibrium_residual_conditioned=maximum,
                minimum_reaction_coefficient_mg=float(coefficients.min()),
                every_workpiece_and_support_checked=True)


def replay_separator(arrays, proof):
    dual = list(map(FQ, proof['dual_exact']))
    origin = list(map(lambda v: FQ(float(v)), arrays['moment_origin_m']))
    scores = []
    for p, n, owner in zip(arrays['contact_points_m'], arrays['contact_normals'], arrays['contact_owners']):
        n = [FQ(float(x)) for x in n]
        r = [FQ(float(x))-c for x, c in zip(p, origin)]
        g = n+[r[1]*n[2]-r[2]*n[1], r[2]*n[0]-r[0]*n[2], r[0]*n[1]-r[1]*n[0]]
        value = sum(a*b for a, b in zip(dual[:6], g))
        if owner >= 0: value += dual[6+owner]*n[2]
        scores.append(value)
    scores += [-v for v in dual[6:]]
    load = arrays['load_wrenches'][int(arrays['failing_load_index'])]
    debt = sum(h*FQ(float(b)) for h, b in zip(dual[:6], load))
    assert min(scores) >= 0 and debt < 0
    assert str(min(scores)) == proof['min_generator_dot_exact']
    assert str(debt) == proof['target_dot_exact']
    return dict(exact_separation_replayed=True, generators_checked=len(scores), target_dot_exact=str(debt))


# Current one-body entry; shared geometry and regression helpers stay available.
from step4_floor_contact.whole_assembly import audit as run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('objects', nargs='*')
    for name in parser.parse_args().objects or F.OBJECTS: run(name)
