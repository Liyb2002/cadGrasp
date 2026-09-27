"""Finite static audit of the exported fixture, independent of contact selection.
Use the repository's frictionless heads, massless fixture and sufficient-floor-
friction assumption. This does not certify a material, a continuum or stiffness.
"""
from pathlib import Path
import json
import hashlib
import sys
import warnings
import numpy as np
import trimesh
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT.parent/'baseline_algo'))
from step3_scheculer.floor_support import COEFFICIENT, rays
D = json.loads((OUT/'data.js').read_text().split('=', 1)[1].rstrip(';\n'))
fixture_path = OUT/'fixture.obj'
fixture = trimesh.load(fixture_path, force='mesh', process=False)
assert fixture.is_watertight, 'Indexed fixture export must retain closed topology'
rows = []
for k, pose in enumerate(D['poses']):
    r = np.asarray(pose['fixtureR']); t = np.asarray(pose['fixtureT'])
    obj = trimesh.Trimesh(np.array(pose['object']['v']).reshape(-1, 3),
                         np.array(pose['object']['f']).reshape(-1, 3), process=False)
    local = trimesh.Trimesh((obj.vertices-t)@r, obj.faces, process=False)
    candidates = np.setdiff1d(np.flatnonzero(local.face_normals[:, 0] > .025), pose['work'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        _, gaps, _ = trimesh.proximity.closest_point(fixture, local.triangles_center[candidates])
    contact_ids = candidates[np.isfinite(gaps) & (gaps < .00003)]
    points = obj.triangles_center[contact_ids]
    normals = -obj.face_normals[contact_ids]
    com = obj.center_mass
    fv = fixture.vertices@r.T+t
    floor = fv[fv[:, 2] < 1e-6]
    floor = floor[ConvexHull(floor[:, :2]).vertices]
    columns = []
    # Equal-and-opposite contact force, shared by the two free-body equations.
    # Object moments use its COM; fixture moments use the world origin.
    for q, n in zip(points, normals):
        columns.append(np.r_[n, np.cross(q-com, n), -n, -np.cross(q, n)])
    ground = obj.vertices[obj.vertices[:, 2].argmin()]
    for n in rays():
        columns.append(np.r_[n, np.cross(ground-com, n), np.zeros(6)])
    for q in floor:
        for n in rays():
            columns.append(np.r_[np.zeros(6), n, np.cross(q, n)])
    A = np.array(columns).T
    scale = np.tile(np.r_[np.ones(3), np.ones(3)*10], 2)
    A *= scale[:, None]
    demands = [dict(face=None, point=com, force=np.zeros(3))]
    work_ids = np.asarray(pose['work'])
    for i in work_ids[np.linspace(0, len(work_ids)-1, 24).astype(int)]:
        n = -obj.face_normals[i]
        u = np.cross(n, [1., 0, 0]); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        for phi in np.arange(8)*np.pi/4:
            force = .5*(np.cos(np.pi/6)*n+np.sin(np.pi/6)*(np.cos(phi)*u+np.sin(phi)*v))
            demands.append(dict(face=int(i), point=obj.triangles_center[i], force=force))
    passes = []; failures = []; max_residual = 0.
    for index, test in enumerate(demands):
        q, f = test['point'], test['force']
        b = np.r_[-f+[0, 0, 1], -np.cross(q-com, f), np.zeros(6)]*scale
        fit = linprog(np.ones(A.shape[1]), A_eq=A, b_eq=b, bounds=(0, None), method='highs')
        residual = float(np.max(np.abs(A@fit.x-b))) if fit.success else None
        ok = fit.success and residual < 1e-7 and fit.x.min() >= -1e-9
        passes.append(bool(ok))
        if ok:
            max_residual = max(max_residual, residual)
        else:
            failures.append(dict(index=index, face=test['face'], force_mg=f.tolist(),
                                 point_m=q.tolist(), solver_status=fit.status, residual=residual))
    row = dict(pose=f'pose_{k+2}', nominal_contact_face_centers=len(contact_ids),
               gravity_equilibrium=passes[0], passed=sum(passes), tested=len(passes),
               maximum_scaled_equilibrium_residual=max_residual, failures=failures)
    rows.append(row)
    print(f"{row['pose']}: gravity={row['gravity_equilibrium']}, finite load cases {row['passed']}/{row['tested']}", flush=True)
report = dict(fixture_file='fixture.obj', fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
              model='two coupled 6D free bodies; frictionless unilateral heads; massless fixture',
              floor_friction=COEFFICIENT, floor_model='repository sufficient-friction four-ray inner pyramid; not a measured coefficient',
              nominal_contact_gap_tolerance_mm=.03, force_limit_mg=.5, cone_half_angle_deg=30,
              sampling='gravity plus 24 evenly indexed work-face centroids, 8 cone-rim directions each, at K',
              results=rows,
              scope='Finite nominal-contact numerical audit. Positive contact gaps are treated as closed; physical fit and manufacturing tolerance remain unresolved. Does not prove continuum coverage, stiffness, strength, full tool-cone access or robot collision freedom.')
(OUT/'load_check.json').write_text(json.dumps(report, indent=2)+'\n')
assert all(r['gravity_equilibrium'] for r in rows), 'A pose lacks nominal gravity equilibrium'
