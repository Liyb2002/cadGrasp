"""Deterministic integration of the continuously feasible position/direction set.

Convert continuous contact patches to a polyhedral wrench cone. For each force
direction, clip the working triangles by the joint wrench inequalities and
subtract exact projected occluder polygons. Integrate those polygon AREAS over
the direction cap using adaptive deterministic quadrature. No random draws or
statistical confidence intervals enter the calculation.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
from scipy.integrate import cubature
from scipy.linalg import null_space, qr
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import area as A
import continuous as C

VERSION = 2
RAY_OFFSET = 1e-5
COARSE_RTOL, FINE_RTOL = 2e-3, 5e-4
ATOL = 2e-7
SOURCE = A.HERE/'polygon_integral.cpp'


def native():
    digest = A.P.I.sha256(SOURCE)
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    path = Path(tempfile.gettempdir())/f'cadgrasp-polygon-integral-{digest[:16]}{suffix}'
    if not path.exists():
        subprocess.run(['c++', '-O3', '-shared', '-fPIC', str(SOURCE), '-o', str(path)], check=True)
    lib = ctypes.CDLL(str(path))
    floats = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    longs = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    lib.cap_bounds.argtypes = [ctypes.c_int, ctypes.c_int, floats, floats, floats]
    lib.evaluate.argtypes = [ctypes.c_int, floats, ctypes.c_int, ctypes.c_int, ints, longs,
                             floats, ints, floats, ints, ints, floats, floats, floats, floats]
    lib.clipped_area.argtypes = [ctypes.c_int, floats]
    lib.clipped_area.restype = ctypes.c_double
    return lib


def contact_halfspaces(S, masks):
    """Enumerate the whole cone, including continuous positions on each patch.

    A strictly positive cross-section reduces the six-dimensional cone to a
    bounded five-dimensional convex hull. Start from deterministic extreme rays,
    add every missing facet's most violating real ray, and stop only when every
    original patch vertex satisfies all facets within numerical tolerance.
    """
    full = C.patch_vertex_columns(S, masks)
    result = linprog(np.r_[np.zeros(6), -1], A_ub=np.c_[-full, np.ones(len(full))],
                     b_ub=np.zeros(len(full)), bounds=[(-1, 1)]*6+[(0, None)],
                     method='highs', options=A.OPTIONS)
    assert result.success and result.x[-1] > 1e-8
    axis = result.x[:6]/np.linalg.norm(result.x[:6])
    U = null_space(axis[None])
    section = (full/(full@axis)[:, None])@U
    pivots = qr(full.T, mode='economic', pivoting=True)[2][:6]
    keep = np.unique(np.r_[pivots, section.argmin(axis=0), section.argmax(axis=0)])
    for iteration in range(40):
        hull = ConvexHull(section[keep])
        H = hull.equations[:, :-1]@U.T+hull.equations[:, -1, None]*axis
        H /= np.linalg.norm(H, axis=1)[:, None]
        # Deduplicate coplanar facets but retain their unrounded coefficients.
        unique = np.unique(np.round(H, 10), axis=0, return_index=True)[1]
        H = H[np.sort(unique)]
        maxima = np.full(len(H), -np.inf)
        argmax = np.zeros(len(H), int)
        for start in range(0, len(full), 1024):
            scores = H@full[start:start+1024].T
            ids = scores.argmax(axis=1)
            values = scores[np.arange(len(H)), ids]
            update = values > maxima
            maxima[update], argmax[update] = values[update], start+ids[update]
        missing = np.setdiff1d(np.unique(argmax[maxima > 1e-10]), keep)
        if not len(missing):
            assert maxima.max() < 1e-9
            H = np.vstack([H, -axis])
            return np.ascontiguousarray(H), keep, axis, float(maxima.max())
        keep = np.union1d(keep, missing)
    raise RuntimeError('Contact-cone facet enumeration did not finish')


def coefficients(S, ids, H):
    q = S.mesh.triangles[ids]@S.R.T+S.t
    n = -(S.mesh.face_normals[ids]@S.R.T)
    e1, e2 = C.frame(n)
    frame = np.stack([n, e1, e2], axis=1)
    r = (q-S.com)*S.scale[3:]
    v = -.5*(H[None, :, None, :3]+np.cross(H[None, :, None, 3:], r[:, None]))
    out = np.empty((len(ids), len(H), 3, 4))
    out[..., 0] = H[None, :, None, 2]
    out[..., 1:] = np.einsum('nhvc,nkc->nhvk', v, frame)
    return np.ascontiguousarray(out)


def visibility_geometry(S, faces):
    """Prepare potential occluders and their exact directional projections.

    A conservative finite sweep of the whole position triangle and direction
    cap excludes irrelevant mesh triangles. Retained blockers are clipped to
    the forward half-space. The native kernel projects and subtracts their
    polygons separately at each quadrature direction, handling overlaps once.
    """
    tri, normals = S.mesh.triangles, S.mesh.face_normals
    extent = S.mesh.extents.max()
    angles = np.arange(8)*2*np.pi/8
    face_offsets, vertex_offsets, vertices, projection = [0], [0], [], []
    for face in faces:
        n = normals[face]
        q = tri[face]+RAY_OFFSET*n
        ids = np.flatnonzero(np.max((tri-q[0])@n, axis=1) > 1e-11*extent)
        if len(ids):
            a, b = C.frame(n[None])
            directions = n+(np.tan(C.THETA)/np.cos(np.pi/8)+1e-10)*(
                np.cos(angles)[:, None]*a+np.sin(angles)[:, None]*b)
            cloud = np.concatenate((q, (q[:, None]+4*extent*directions).reshape(-1, 3)))
            planes = ConvexHull(cloud).equations
            distance = np.einsum('tvc,fc->tvf', tri[ids], planes[:, :3])+planes[:, 3]
            ids = ids[~(distance.min(axis=1) > 1e-10*extent).any(axis=1)]
        qw, nw = tri[face]@S.R.T+S.t, n@S.R.T
        dual = np.linalg.pinv((qw[1:]-qw[0]).T)
        e1, e2 = C.frame(-nw[None])
        projection.append((dual@np.stack([e1[0], e2[0]], axis=1)).reshape(-1))
        for other in ids:
            polygon, cut = list(tri[other]), []
            last = polygon[-1]
            last_height = (last-q[0])@n
            for current in polygon:
                height = (current-q[0])@n
                if (height >= 0) != (last_height >= 0):
                    cut.append(last+last_height/(last_height-height)*(current-last))
                if height >= 0:
                    cut.append(current)
                last, last_height = current, height
            if len(cut) < 3:
                continue
            points = np.array(cut)@S.R.T+S.t
            heights = (points-qw[0])@nw-RAY_OFFSET
            base = (points-(heights+RAY_OFFSET)[:, None]*nw-qw[0])@dual.T
            vertices.extend(np.c_[base, heights])
            vertex_offsets.append(len(vertices))
        face_offsets.append(len(vertex_offsets)-1)
    return dict(face_occluder_offsets=np.array(face_offsets, np.int32),
                occluder_vertex_offsets=np.array(vertex_offsets, np.int32),
                projected_vertices=np.array(vertices, float).reshape(-1, 3),
                tangent_projection=np.array(projection, float))


class Integrand:
    def __init__(self, S, faces, halfspaces, geometry):
        self.S, self.lib, self.rows = S, native(), len(halfspaces)
        modes = np.zeros((len(faces), self.rows), np.int32)
        limits = np.array([np.cos(C.THETA), 1., 0., 2*np.pi])
        for j, H in enumerate(halfspaces):
            for start in range(0, len(faces), 256):
                ids = faces[start:start+256]
                out = np.empty((len(ids), 2))
                self.lib.cap_bounds(len(ids), len(H), coefficients(S, ids, H), limits, out)
                modes[start:start+256, j] = np.where(out[:, 1] < 1e-14, -1,
                                                      np.where(out[:, 0] > 1-1e-12, 1, 0))
        fo, vo = geometry['face_occluder_offsets'], geometry['occluder_vertex_offsets']
        active = (modes == 0).any(axis=1) | (np.diff(fo) > 0)
        weights = S.mesh.area_faces[faces]/S.mesh.area_faces[faces].sum()
        self.constant = np.r_[(weights[~active, None]*(modes[~active] > 0)).sum(axis=0), weights[~active].sum()]
        newfo, newvo, vertices = [0], [0], []
        for i in np.flatnonzero(active):
            for blocker in range(fo[i], fo[i+1]):
                vertices.extend(geometry['projected_vertices'][vo[blocker]:vo[blocker+1]])
                newvo.append(len(vertices))
            newfo.append(len(newvo)-1)
        self.fo, self.vo = np.array(newfo, np.int32), np.array(newvo, np.int32)
        self.vertices = np.array(vertices, float).reshape(-1, 3)
        self.projection = np.ascontiguousarray(geometry['tangent_projection'][active])
        self.modes, self.weights = np.ascontiguousarray(modes[active]), np.ascontiguousarray(weights[active])
        self.active_faces, self.all_modes = faces[active], modes
        offsets, parts = [0], []
        for H in halfspaces:
            part = coefficients(S, self.active_faces, H).ravel()
            parts.append(part); offsets.append(offsets[-1]+len(part))
        self.coefficients = np.concatenate(parts)
        self.offsets = np.array(offsets[:-1], np.int64)
        self.counts = np.array([len(h) for h in halfspaces], np.int32)
        self.points, self.started, self.last = 0, time.monotonic(), time.monotonic()
        self.cache = {}

    def __call__(self, x):
        x = np.ascontiguousarray(x, dtype=float)
        keys = [point.tobytes() for point in x]
        missing = {key: point for key, point in zip(keys, x) if key not in self.cache}
        if missing:
            angles = np.array(list(missing.values()))
            values = np.zeros((len(angles), self.rows+1))
            self.lib.evaluate(len(angles), angles, len(self.active_faces), self.rows,
                              self.counts, self.offsets, self.coefficients, self.modes, self.weights,
                              self.fo, self.vo, self.vertices, self.projection, self.constant, values)
            self.cache.update(zip(missing, values))
            self.points += len(angles)
        if time.monotonic()-self.last > 20:
            print(self.S.name, 'quadrature nodes', self.points,
                  'seconds', round(time.monotonic()-self.started, 1), flush=True)
            self.last = time.monotonic()
        # A fresh array allows the quadrature wrapper to rescale in place.
        return np.array([self.cache[key] for key in keys])


def integrate(f, rule, rtol, complements=None, scale=None):
    # Fixed initial sectors prevent a small nonzero region being hidden by a
    # single initial rule. Subsequent refinement is entirely deterministic.
    points = [[C.THETA/2, phi] for phi in np.linspace(0, 2*np.pi, 9)[1:-1]]
    def values(x):
        out = f(x)
        if complements is not None:
            # Near full coverage, integrate the missed area so relative error
            # controls the small missing region instead of the almost-unit total.
            out[:, np.flatnonzero(complements)] = out[:, -1, None]-out[:, np.flatnonzero(complements)]
        return out*scale
    if scale is None:
        # Conditioning only: the final answer comes from adaptive cubature.
        # Equalize component magnitudes so refinement resolves small covered
        # regions as well as large ones. Preserve the original error tolerances.
        nodes, weights = np.polynomial.legendre.leggauss(5)
        pilot = np.array([[(a+1)*C.THETA/2, (b+1)*np.pi] for a in nodes for b in nodes])
        estimate = (f(pilot)*np.outer(weights, weights).ravel()[:, None]).sum(axis=0)*C.THETA*np.pi/2
        if complements is not None:
            estimate[:-1] = np.where(complements, estimate[-1]-estimate[:-1], estimate[:-1])
        scale = 1/np.maximum(np.abs(estimate), ATOL/rtol)
    result = cubature(values, [0, 0], [C.THETA, 2*np.pi], rule=rule, rtol=rtol,
                      atol=ATOL*scale, max_subdivisions=2000, points=points)
    if result.status != 'converged':
        raise RuntimeError(f'{f.S.name}: deterministic quadrature did not converge')
    result.estimate /= scale
    result.error /= scale
    result.atol = ATOL
    for region in result.regions:
        region.estimate /= scale
        region.error /= scale
    return result


def provenance(S):
    return dict(version=VERSION, geometry_sha256=S.geometry_sha256,
                source_results_sha256=A.P.I.sha256(A.HERE/f'area_{S.name}_results.npz'),
                source_demand_sha256=A.P.I.sha256(A.HERE/f'demand_{S.name}_tip1.npz'),
                native_source_sha256=A.P.I.sha256(SOURCE), python_source_sha256=A.P.I.sha256(Path(__file__)),
                coarse_rtol=COARSE_RTOL, fine_rtol=FINE_RTOL, atol=ATOL, ray_offset_m=RAY_OFFSET)


def run(S, report, arrays):
    C.validate_model(S, arrays)
    rows = [k for k, row in enumerate(report['rows'], 1) if row['continuous_coverage']['status'] != 'certified']
    halfspaces, evidence = [], {}
    for k in rows:
        H, indices, axis, residual = contact_halfspaces(S, arrays[f'row_masks_{k}'])
        halfspaces.append(H)
        evidence.update({f'halfspaces_{k}': H, f'contact_vertex_indices_{k}': indices,
                         f'cone_axis_{k}': axis, f'cone_residual_{k}': np.array(residual)})
        print(S.name, 'CONE', k, len(indices), 'extreme rays;', len(H), 'halfspaces', flush=True)
    faces = np.flatnonzero(S.demand['work_faces'])
    geometry = visibility_geometry(S, faces)
    evidence.update(geometry)
    f = Integrand(S, faces, halfspaces, geometry)
    coarse = integrate(f, 'gk15', COARSE_RTOL)
    previous = coarse.estimate[:-1]/coarse.estimate[-1]
    complements = previous > .5
    magnitude = coarse.estimate.copy()
    magnitude[:-1] = np.where(complements, magnitude[-1]-magnitude[:-1], magnitude[:-1])
    fine = integrate(f, 'gk21', FINE_RTOL, complements,
                     1/np.maximum(np.abs(magnitude), ATOL/FINE_RTOL))
    raw_ratios = fine.estimate[:-1]/fine.estimate[-1]
    ratios = np.where(complements, 1-raw_ratios, raw_ratios)
    errors = (fine.error[:-1]+raw_ratios*fine.error[-1])/fine.estimate[-1]
    changes = np.abs(ratios-previous)
    for i, k in enumerate(rows):
        assert 0 < ratios[i] < 1, 'A partial row must resolve its nonzero covered and missed regions'
        # Refinement must agree at the precision used on the figure.
        allowed = max(2e-6, 2e-4*min(ratios[i], 1-ratios[i]))
        if changes[i] > max(allowed, 5*errors[i]):
            raise RuntimeError(f'{S.name} row {k}: quadrature refinement changed the result materially')
        report['rows'][k-1]['coverage_fraction'] = dict(
            method='deterministic_polygon_cubature', estimate=float(ratios[i]),
            numerical_error_estimate=float(errors[i]), refinement_change=float(changes[i]))
    for k, row in enumerate(report['rows'], 1):
        if k not in rows:
            row['coverage_fraction'] = dict(method='continuous_domain_certificate', estimate=1.)
        print(S.name, 'COVERED', k, 100*row['coverage_fraction']['estimate'], flush=True)
    audit = np.array([[C.THETA*a, 2*np.pi*b] for a in (.17, .43, .79) for b in (.07, .31, .57, .89)])
    evidence.update(work_faces=faces, partial_rows=np.array(rows), row_modes=f.all_modes,
                    coarse_estimate=coarse.estimate, coarse_error=coarse.error,
                    fine_estimate=fine.estimate, fine_error=fine.error, integrated_complements=complements,
                    region_a=np.array([r.a for r in fine.regions]), region_b=np.array([r.b for r in fine.regions]),
                    region_estimate=np.array([r.estimate for r in fine.regions]),
                    region_error=np.array([r.error for r in fine.regions]),
                    audit_angles=audit, audit_values=f(audit))
    path = A.HERE/f'area_{S.name}_coverage.npz'
    np.savez_compressed(path, **evidence)
    meta = provenance(S)
    meta.update(method='exact polygon clipping in position; adaptive Gauss-Kronrod integration in direction',
                measure='work-surface area × solid angle, conditioned on tool reachability',
                coarse_rule='gk15', fine_rule='gk21', quadrature_nodes=f.points,
                coarse_regions=len(coarse.regions), fine_regions=len(fine.regions),
                visible_measure=float(fine.estimate[-1]), evidence_sha256=A.P.I.sha256(path),
                error_kind='deterministic quadrature error estimate and refinement difference; not a statistical interval')
    report['coverage_integration'] = meta
    save(S, report)


def save(S, report):
    (A.HERE/f'area_{S.name}_results.json').write_text(json.dumps(report, indent=2)+'\n')


def ensure(S, report, arrays):
    meta = report.get('coverage_integration', {})
    path = A.HERE/f'area_{S.name}_coverage.npz'
    valid = all(meta.get(key) == value for key, value in provenance(S).items())
    valid &= path.exists() and meta.get('evidence_sha256') == A.P.I.sha256(path)
    valid &= all('coverage_fraction' in row for row in report['rows'])
    if not valid:
        run(S, report, arrays)


def check(S, report, arrays):
    C.validate_model(S, arrays)
    meta, path = report['coverage_integration'], A.HERE/f'area_{S.name}_coverage.npz'
    assert all(meta[key] == value for key, value in provenance(S).items())
    assert A.P.I.sha256(path) == meta['evidence_sha256']
    with np.load(path) as z:
        rows, halfspaces = z['partial_rows'], []
        for k in rows:
            H, indices, axis, _ = contact_halfspaces(S, arrays[f'row_masks_{k}'])
            np.testing.assert_allclose(H, z[f'halfspaces_{k}'], atol=1e-12, rtol=0)
            assert np.array_equal(indices, z[f'contact_vertex_indices_{k}'])
            halfspaces.append(H)
        faces = np.flatnonzero(S.demand['work_faces'])
        assert np.array_equal(faces, z['work_faces'])
        geometry = {key: z[key] for key in ('face_occluder_offsets', 'occluder_vertex_offsets',
                                           'projected_vertices', 'tangent_projection')}
        f = Integrand(S, faces, halfspaces, geometry)
        np.testing.assert_allclose(f(z['audit_angles']), z['audit_values'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(z['region_estimate'].sum(axis=0), z['fine_estimate'], atol=1e-12, rtol=0)
        assert np.all(z['fine_error'] <= ATOL+FINE_RTOL*np.abs(z['fine_estimate']))
        for i, k in enumerate(rows):
            value = z['fine_estimate'][i]/z['fine_estimate'][-1]
            if z['integrated_complements'][i]:
                value = 1-value
            assert abs(value-report['rows'][k-1]['coverage_fraction']['estimate']) < 1e-14
        print(S.name, 'DETERMINISTIC INTEGRAL CHECKED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if any(name not in A.P.OBJECTS for name in args.objects):
        parser.error('objects must be A1-f, B or C5')
    for name in args.objects or A.P.OBJECTS:
        S = A.load(name)
        report = json.loads((A.HERE/f'area_{name}_results.json').read_text())
        with np.load(A.HERE/f'area_{name}_results.npz') as z:
            arrays = {key: z[key] for key in z.files}
        (check if args.check else run)(S, report, arrays)


if __name__ == '__main__':
    main()
