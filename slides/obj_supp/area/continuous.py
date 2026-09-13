"""Continuous paired-demand containment, independent of the search samples.

Each work triangle is crossed with a circumscribed prism around its 15-degree
direction cap. Its vertex demands enclose ALL continuous demands by bilinearity.
Positive contact-basis solutions certify this outer set. A failed outer vertex
is never called a physical counterexample: that requires an allowed, reachable
load and a separator valid over the entire continuous contact patches.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from scipy.optimize import linprog, nnls
import area as A
from cone_model import frame

VERSION = 1
SIDES = 8
THETA = np.deg2rad(15.)
PAD = 1e-10
# Absolute uncertainty in every scaled wrench entry, in addition to roundoff.
# This also covers face-normal / transform arithmetic in the mesh model.
ENTRY_GUARD = 1e-10
ROUND_GUARD = 128*np.finfo(float).eps


def cap_enclosure(normals, sides=SIDES):
    """A circumscribed polygon × axial interval; vertices need not be unit."""
    n = normals/np.linalg.norm(normals, axis=1)[:, None]
    e1, e2 = frame(n)
    angles = np.arange(sides)*2*np.pi/sides
    radius = np.sin(THETA)/np.cos(np.pi/sides)+PAD
    transverse = radius*(np.cos(angles)[None, :, None]*e1[:, None, :]
                         + np.sin(angles)[None, :, None]*e2[:, None, :])
    return np.concatenate([u*n[:, None, :]+transverse
                           for u in (np.cos(THETA)-PAD, 1+PAD)], axis=1)


def wrench(S, q, d):
    q, d = np.broadcast_arrays(q, d)
    return np.concatenate((A.P.UP-.5*d, -.5*np.cross(q-S.com, d)), axis=-1)*S.scale


def demand_enclosure(S):
    faces = np.flatnonzero(S.demand['work_faces'])
    triangles = S.mesh.triangles[faces]@S.R.T+S.t
    q = triangles.reshape(-1, 3)
    n = np.repeat(-(S.mesh.face_normals[faces]@S.R.T), 3, axis=0)
    n /= np.linalg.norm(n, axis=1)[:, None]
    directions = cap_enclosure(n)
    targets = wrench(S, q[:, None, :], directions).reshape(-1, 6)
    return targets, q, n, faces


def validate_model(S, arrays):
    assert float(S.demand['K']) == .5 and float(S.demand['cone_half_deg']) == 15.
    assert np.max(np.abs(S.R.T@S.R-np.eye(3))) < 1e-12
    assert np.array_equal(arrays['scale'], S.scale)
    assert np.array_equal(arrays['T_world_mesh'], S.T)
    assert np.array_equal(arrays['com_m'], S.com)
    # The saved face-centre columns must be actual points in the shown patches.
    # Check their normals against the original flat faces, including the tiny
    # difference caused by subdivision arithmetic, within the proof's allowance.
    for k in range(1, 5):
        ids = np.flatnonzero(arrays[f'row_masks_{k}'].any(axis=0))
        assert np.array_equal(ids, arrays[f'face_ids_{k}'])
        n = -(S.mesh.face_normals[S.skin.src[ids]]@S.R.T)
        expected = np.vstack([np.c_[n, np.cross(S.skin.cs[ids]-S.com, n)], S.floor6])*S.scale
        assert np.max(np.abs(expected-arrays[f'columns_{k}']*S.scale)) < ENTRY_GUARD/4


def basis_membership(B, targets):
    """Validate nonnegative solutions, including an explicit error allowance.

    For X≈B^-1, ||I-BX||<1 bounds the exact inverse by a Neumann series.
    The residual then bounds the correction to every approximate coefficient.
    ENTRY_GUARD allows entrywise perturbations of both B and each target.
    A strictly positive remainder proves membership without accepting a residual
    as an equilibrium solution. Singular/ambiguous bases prove nothing.
    """
    if B.shape != (6, 6):
        return np.zeros(len(targets), bool), np.full(len(targets), -np.inf)
    try:
        X = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return np.zeros(len(targets), bool), np.full(len(targets), -np.inf)
    nx = np.linalg.norm(X, ord=np.inf)*(1+ROUND_GUARD)
    product_error = ROUND_GUARD*(np.abs(B)@np.abs(X)+np.eye(6))
    eta = np.max(np.sum(np.abs(np.eye(6)-B@X)+product_error, axis=1)) + 6*ENTRY_GUARD*nx
    if not np.isfinite(eta) or eta >= 1:
        return np.zeros(len(targets), bool), np.full(len(targets), -np.inf)
    inverse_bound = nx/(1-eta)*(1+ROUND_GUARD)
    coefficients = targets@X.T
    residual = np.max(np.abs(coefficients@B.T-targets)
                      + ROUND_GUARD*(np.abs(coefficients)@np.abs(B).T+np.abs(targets)), axis=1)
    residual += ENTRY_GUARD*(1+np.abs(coefficients).sum(axis=1))*(1+ROUND_GUARD)
    margin = coefficients.min(axis=1)-inverse_bound*residual
    return margin > 0, margin


def contain(columns, targets):
    assignment = np.full(len(targets), -1, dtype=np.int32)
    bases, margin_min = [], np.inf
    phase = np.c_[columns.T, np.eye(6), -np.eye(6)]
    objective = np.r_[np.zeros(len(columns)), np.ones(12)]
    while np.any(assignment < 0):
        pending = np.flatnonzero(assignment < 0)
        b = targets[pending[0]]
        try:
            x = nnls(columns.T, b, maxiter=max(1000, 3*len(columns)))[0]
        except RuntimeError:
            x = np.zeros(len(columns))
        if np.max(np.abs(columns.T@x-b)) > 1e-9:
            result = linprog(objective, A_eq=phase, b_eq=b, bounds=(0, None),
                             method='highs', options=A.OPTIONS)
            if not result.success:
                return None, None, None, 'solver_unresolved'
            if result.fun > 1e-8:
                return None, None, result.eqlin.marginals, 'outer_vertex_infeasible'
            x = result.x[:len(columns)]
        active = np.flatnonzero(x > 1e-12)
        ok, margin = basis_membership(columns[active].T, targets[pending])
        if not ok[0]:
            return None, None, None, 'no_strict_membership_certificate'
        assignment[pending[ok]] = len(bases)
        bases.append(active)
        margin_min = min(margin_min, float(margin[ok].min()))
    return np.array(bases), assignment, margin_min, 'certified'


def cap_maximizer(v, n):
    """Exact spherical-cap support function, including axial degeneracies."""
    n = n/np.linalg.norm(n, axis=1)[:, None]
    length = np.linalg.norm(v, axis=1)
    dot = np.einsum('ij,ij->i', v, n)
    tangent = v-dot[:, None]*n
    tn = np.linalg.norm(tangent, axis=1)
    unit_tangent = frame(n)[0]
    nonzero = tn > 1e-14*np.maximum(length, 1e-30)
    unit_tangent[nonzero] = tangent[nonzero]/tn[nonzero, None]
    d = np.cos(THETA)*n+np.sin(THETA)*unit_tangent
    interior = (length > 0) & (dot >= np.cos(THETA)*length)
    d[interior] = v[interior]/length[interior, None]
    d[length == 0] = n[length == 0]
    return d


def patch_vertex_columns(S, masks):
    """ALL triangle vertices, with constant source-face normal: full patches.

    Any continuous normal-pressure distribution gives a conic combination of
    these wrenches. No face-centre approximation or convex-hull pruning here.
    """
    ids = np.flatnonzero(masks.any(axis=0))
    q = (S.skin.sub.triangles[ids]@S.R.T+S.t).reshape(-1, 3)
    n = np.repeat(-(S.mesh.face_normals[S.skin.src[ids]]@S.R.T), 3, axis=0)
    return np.vstack((np.c_[n, np.cross(q-S.com, n)], S.floor6))*S.scale


def find_counterexample(S, masks, y, q, n, faces):
    v = -.5*(y[:3]+np.cross(y[3:], (q-S.com)*S.scale[3:]))
    d = cap_maximizer(v, n)
    values = wrench(S, q, d)@y
    full_columns = patch_vertex_columns(S, masks)
    # Start with the analytical worst load, then alternate faces if occluded.
    order = np.argsort(values)[::-1]
    tried = set()
    for idx in order:
        face = int(faces[idx//3])
        if face in tried or values[idx] <= 1e-7:
            continue
        tried.add(face)
        if len(tried) > 48:
            break
        bary = np.full(3, 1e-5/3)
        bary[idx % 3] += 1-1e-5
        point = bary@S.mesh.triangles[face]@S.R.T+S.t
        normal = -(S.mesh.face_normals[face]@S.R.T)
        normal /= np.linalg.norm(normal)
        vector = -.5*(y[:3]+np.cross(y[3:], (point-S.com)*S.scale[3:]))
        direction = cap_maximizer(vector[None], normal[None])[0]
        # Keep the witness strictly inside the allowed cap, away from rounding
        # ambiguity on its rim. This does not affect the sizable separation gap.
        direction = (1-1e-7)*direction+1e-7*normal
        direction /= np.linalg.norm(direction)
        origin = (point-S.t)@S.R+1e-5*S.mesh.face_normals[face]
        if S.mesh.ray.intersects_any(origin[None], (-direction@S.R)[None])[0]:
            continue
        b = wrench(S, point, direction)
        # Strict separation over the ENTIRE patches, not only chosen centres.
        result = linprog(-b, A_ub=full_columns, b_ub=np.full(len(full_columns), -1e-8),
                         bounds=[(-1, 1)]*6, method='highs', options=A.OPTIONS)
        if not result.success:
            continue
        dual = result.x
        dots = full_columns.astype(np.longdouble)@dual.astype(np.longdouble)
        upper = float(dots.max())+ENTRY_GUARD*np.abs(dual).sum()
        gap = float(b.astype(np.longdouble)@dual.astype(np.longdouble))-ENTRY_GUARD*np.abs(dual).sum()
        if upper < 0 and gap > 1e-7:
            return dict(face_index=face, barycentric=bary.tolist(), position_m=point.tolist(),
                        inward_direction=direction.tolist(), force_mg=b[:3].tolist(),
                        moment_mg_m=(b[3:]/S.scale[3:]).tolist(), dual=dual.tolist(),
                        dual_patch_upper_bound=upper, dual_demand_lower_bound=gap,
                        tool_ray_clear=True, ray_offset_m=1e-5)
    return None


def audit(S, report, arrays):
    validate_model(S, arrays)
    targets, q, n, faces = demand_enclosure(S)
    certificates = {}
    for k, row in enumerate(report['rows'], start=1):
        basis, assignment, evidence, reason = contain(arrays[f'columns_{k}']*S.scale, targets)
        verdict = dict(status='not_verified', reason=reason)
        if reason == 'certified':
            certificates[f'bases_{k}'] = basis
            certificates[f'assignment_{k}'] = assignment
            verdict.update(status='certified', basis_count=len(basis),
                           minimum_positive_coefficient_margin=evidence)
        elif evidence is not None:
            witness = find_counterexample(S, arrays[f'row_masks_{k}'], evidence, q, n, faces)
            if witness is not None:
                verdict.update(status='counterexample', counterexample=witness)
        row['continuous_coverage'] = verdict
        print(S.name, 'CONTINUOUS', k, verdict['status'], flush=True)
    report['continuous_verification'] = dict(
        version=VERSION, method='work-triangle vertices × circumscribed spherical-cap prism',
        K=.5, cone_half_angle_deg=15., direction_polygon_sides=SIDES,
        direction_enclosure_pad=PAD, scaled_wrench_entry_guard=ENTRY_GUARD,
        work_triangle_count=len(faces), outer_vertex_count=len(targets),
        domain='all actual work-triangle positions and all unit inward-cap directions; full load only',
        reachability='coverage encloses even occluded directions; counterexamples must be ray-reachable',
        support_model='nonnegative unbounded inward-normal reactions plus original floor ray',
        limits='current triangle mesh and target pose; no certificate for the unmeshed smooth CAD surface',
        geometry_sha256=S.geometry_sha256,
        source_results_sha256=A.P.I.sha256(A.HERE/f'area_{S.name}_results.npz'),
        source_demand_sha256=A.P.I.sha256(A.HERE/f'demand_{S.name}_tip1.npz'))
    np.savez_compressed(A.HERE/f'area_{S.name}_continuous.npz', **certificates)
    (A.HERE/f'area_{S.name}_results.json').write_text(json.dumps(report, indent=2)+'\n')


def check(S, report, arrays):
    """Replay saved bases and physical witnesses without a feasibility solver."""
    validate_model(S, arrays)
    meta = report['continuous_verification']
    assert meta['version'] == VERSION and meta['geometry_sha256'] == S.geometry_sha256
    assert meta['source_results_sha256'] == A.P.I.sha256(A.HERE/f'area_{S.name}_results.npz')
    assert meta['source_demand_sha256'] == A.P.I.sha256(A.HERE/f'demand_{S.name}_tip1.npz')
    targets, _, _, _ = demand_enclosure(S)
    with np.load(A.HERE/f'area_{S.name}_continuous.npz') as z:
        for k, row in enumerate(report['rows'], start=1):
            verdict = row['continuous_coverage']
            if verdict['status'] == 'certified':
                bases, assignment = z[f'bases_{k}'], z[f'assignment_{k}']
                assert assignment.shape == (len(targets),)
                assert assignment.min() >= 0 and assignment.max() < len(bases)
                cols = arrays[f'columns_{k}']*S.scale
                for i, basis in enumerate(bases):
                    assert basis_membership(cols[basis].T, targets[assignment == i])[0].all()
            elif verdict['status'] == 'counterexample':
                w = verdict['counterexample']
                face, bary = w['face_index'], np.array(w['barycentric'])
                assert S.demand['work_faces'][face] and bary.min() > 0 and abs(bary.sum()-1) < 1e-12
                q = bary@S.mesh.triangles[face]@S.R.T+S.t
                assert np.max(np.abs(q-w['position_m'])) < 1e-12
                d, y = np.array(w['inward_direction']), np.array(w['dual'])
                n = -(S.mesh.face_normals[face]@S.R.T)
                assert abs(np.linalg.norm(d)-1) < 1e-12 and d@n > np.cos(THETA)+1e-10
                b = wrench(S, q, d)
                assert np.allclose(b[:3], w['force_mg'], atol=1e-12, rtol=0)
                assert np.allclose(b[3:]/S.scale[3:], w['moment_mg_m'], atol=1e-12, rtol=0)
                columns = patch_vertex_columns(S, arrays[f'row_masks_{k}'])
                upper = np.max(columns.astype(np.longdouble)@y.astype(np.longdouble))+ENTRY_GUARD*abs(y).sum()
                lower = b.astype(np.longdouble)@y.astype(np.longdouble)-ENTRY_GUARD*abs(y).sum()
                assert upper < 0 and lower > 1e-7
                origin = (q-S.t)@S.R+w['ray_offset_m']*S.mesh.face_normals[face]
                assert not S.mesh.ray.intersects_any(origin[None], (-d@S.R)[None])[0]
            else:
                assert verdict['status'] == 'not_verified'
            print(S.name, 'CHECKED CONTINUOUS', k, verdict['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--check', action='store_true', help='Replay saved certificates without solving')
    args = parser.parse_args()
    if any(name not in A.P.OBJECTS for name in args.objects):
        parser.error('objects must be A1-f, B or C5')
    for name in args.objects or A.P.OBJECTS:
        S = A.load(name)
        report = json.loads((A.HERE/f'area_{name}_results.json').read_text())
        with np.load(A.HERE/f'area_{name}_results.npz') as z:
            arrays = {k: z[k] for k in z.files}
        assert report['geometry_sha256'] == S.geometry_sha256
        assert report['source_demand_sha256'] == A.P.I.sha256(A.HERE/f'demand_{name}_tip1.npz')
        (check if args.check else audit)(S, report, arrays)


if __name__ == '__main__':
    main()
