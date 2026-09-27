"""Build one shared, open, contoured rib fixture for B's three saved poses.
Exact mesh booleans subtract all three horizontal withdrawal sweeps. This is
geometry synthesis for the concept slide, not a task-load or robot certificate.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import trimesh
import manifold3d as md
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
COLORS = ['#d97865', '#d9ad53', '#48a494', '#598fbb']
HALF = .084
REAR = .178
RAIL = .074
SOURCES = [ROOT/'objects/B/mesh.stl'] + [ROOT/f'objects/B/tasks/pose_{i}/setup.npz' for i in (2, 3, 4)]
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
layout = json.loads((OUT/'layout.json').read_text())
raw = trimesh.load(SOURCES[0], force='mesh')
mesh = raw.subdivide()

def solid(m):
    return md.Manifold(md.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)))

def unpack(s):
    m = s.to_mesh64()
    result = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    result.merge_vertices(digits_vertex=12)
    return result

def pack(m):
    return dict(v=np.round(m.vertices, 9).ravel().tolist(), f=m.faces.ravel().tolist())

def union(parts):
    return md.Manifold.batch_boolean(parts, md.OpType.Add)

def bead(p, scale):
    return md.Manifold.sphere(1., 32).scale(scale).translate(p)

def rib(a, b, ra, rb=None):
    return md.Manifold.batch_hull([bead(a, ra), bead(b, ra if rb is None else rb)])

cases = []
for k, source in enumerate(SOURCES[1:]):
    with np.load(source) as z:
        obj_r = z['T_world_mesh'][:3, :3]
        v = mesh.vertices @ obj_r.T + z['T_world_mesh'][:3, 3]
        offset = np.r_[np.asarray(layout['poses'][k]['xy'])-v[:, :2].mean(0), 0.]
        v += offset
        r = Rotation.from_euler('z', layout['poses'][k]['yaw']).as_matrix() @ Rotation.from_euler('x', [0, -90, 180][k], degrees=True).as_matrix()
        t = np.array([0., 0., HALF])
        local = trimesh.Trimesh((v-t) @ r, mesh.faces, process=False)
        world = trimesh.Trimesh(v, mesh.faces, process=False)
        cases.append(dict(local=local, world=world, r=r, t=t, obj_r=obj_r,
                          work=np.flatnonzero(z['work_faces']), com=z['com_m']+offset,
                          floor=z['floor_contact_m']+offset, K=float(z['K']), cone=float(z['cone_half_deg'])))

# The same swept exclusion is applied to the whole fixture, including all heads
# that are inactive in a particular pose. A 10-micron transverse numerical band
# makes subtraction robust without pretending to provide manufacturing tolerance.
sweeps = []
for c in cases:
    sweeps.append(solid(c['local']).minkowski_sum(
        md.Manifold.cube([.32, .00002, .00002]).translate([-.32, -.00001, -.00001])))
forbidden = union(sweeps)
print('Three complete object withdrawal sweeps constructed.', flush=True)

# Keep the original working patches accessible along their face normals.
# This clears continuous triangular prisms, not merely a few sampled rays.
access_parts = []
for c in cases:
    # Keep the face-wise prisms separate before union: a convex hull over the
    # entire work patch would incorrectly block empty, usable space.
    for i in c['work']:
        tri = c['local'].triangles[i]
        normal = c['local'].face_normals[i]
        center = tri.mean(0)
        tri = center + (tri-center)*1.01
        access_parts.append(md.Manifold.hull_points(np.vstack([tri-normal*.00005, tri+normal*.3])))
access = union(access_parts)
access = access.minkowski_sum(md.Manifold.cube([.004, .004, .004]).translate([-.002, -.002, -.002]))
access_mesh = unpack(access)
print('Working-face normal access prisms constructed.', flush=True)

# Find contact regions on the rear-visible envelope shared by all placements.
# The insertion direction is +local X, so material must stay behind this envelope.
selected = []
for k, c in enumerate(cases):
    m = c['local']; p = m.triangles_center; n = m.face_normals
    world_n = n @ c['r'].T
    good = (n[:, 0] > .03) & (np.abs(p[:, 1]) < .076) & (np.abs(p[:, 2]) < .076)
    good &= (c['world'].triangles_center[:, 2] > .01)
    good[c['work']] = False
    for other in cases:
        good &= ~other['local'].ray.intersects_any(p+n*.00004, np.tile([1., 0., 0.], (len(p), 1)))
    good &= ~access_mesh.contains(p+n*.0002)
    ids = np.flatnonzero(good)
    if len(ids) < 4:
        raise RuntimeError(f'Pose {k}: insufficient compatible contact regions')
    # Use the current baseline's frictionless head and sufficient-floor-friction
    # model to prioritize contacts that actually oppose the prescribed loads.
    # Seven equations couple force, torque and the massless no-uplift condition.
    q = c['world'].triangles_center[ids]; reaction = -world_n[ids]
    floor_rays = np.array([[64., 0, 1], [-64., 0, 1], [0, 64., 1], [0, -64., 1]])
    A = np.vstack([np.c_[reaction, np.cross(q-c['com'], reaction), reaction[:, 2]],
                   np.c_[floor_rays, np.cross(c['floor']-c['com'], floor_rays), np.zeros(4)],
                   [0, 0, 0, 0, 0, 0, -1]]).T
    A[3:6] *= 10
    demands = [np.array([0., 0, 1, 0, 0, 0, 0])]
    probe_faces = np.unique(np.r_[c['work'][::16], c['work'][np.linspace(0, len(c['work'])-1, 24).astype(int)]])
    for i in probe_faces:
        normal = -c['world'].face_normals[i]
        u = np.cross(normal, [1., 0, 0]); u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        for phi in np.arange(8)*np.pi/4:
            force = c['K']*(np.cos(np.deg2rad(c['cone']))*normal +
                            np.sin(np.deg2rad(c['cone']))*(np.cos(phi)*u+np.sin(phi)*v))
            demands.append(np.r_[-force+[0, 0, 1],
                                 -10*np.cross(c['world'].triangles_center[i]-c['com'], force), 0])
    use = np.zeros(len(ids)); passed = 0
    for demand in demands:
        fit = linprog(np.ones(A.shape[1]), A_eq=A, b_eq=demand, bounds=(0, None), method='highs')
        if fit.success:
            passed += 1; use += fit.x[:len(ids)]
    print(f'Pose {k+2}: available-domain load probes {passed}/{len(demands)}', flush=True)
    if passed != len(demands):
        raise RuntimeError('Shared contact envelope cannot support the load probes')
    chosen = []
    for j in np.argsort(-use):
        if use[j] <= 0: break
        if not chosen or np.min(np.linalg.norm(p[ids[j]]-p[chosen], axis=1)) > .014:
            chosen.append(int(ids[j]))
    c['load_probes'] = len(demands)
    c['selected'] = chosen
    selected.extend((k, i, p[i], n[i]) for i in chosen)
    print(f'Pose {k+2}: {len(ids)} compatible faces, {len(chosen)} load-guided contact lobes.', flush=True)

# Four long rounded buttresses provide alternate flat landing faces. The rear
# perimeter web joins them into a single material body and offers a robot grasp.
parts = []
for sy in (-1, 1):
    for sz in (-1, 1):
        parts.append(rib([-.025, sy*RAIL, sz*RAIL], [REAR, sy*RAIL, sz*RAIL],
                         [.014, .014, .014], [.015, .014, .014]))
for axis in (1, 2):
    for sign in (-1, 1):
        a = np.array([REAR, -RAIL, -RAIL]); b = a.copy()
        a[axis] = -RAIL; b[axis] = RAIL
        a[3-axis] = b[3-axis] = sign*RAIL
        parts.append(rib(a, b, [.015, .014, .014]))
for k, i, p, n in selected:
    sy, sz = np.where(p[1:] >= 0, 1., -1.)
    # A flared, short load path between a contoured contact lobe and a corner rib.
    joint = np.array([min(REAR-.007, p[0]+.055), sy*RAIL, sz*RAIL])
    parts.append(rib(p+[.005, 0, 0], joint, [.015, .015, .015], [.017, .013, .013]))
blank = union(parts)
clip = md.Manifold.cube([.26, 2*HALF, 2*HALF]).translate([-.05, -HALF, -HALF])
fixture = (blank ^ clip) - (forbidden + access)
components = fixture.decompose()
removed_volume = 0.
if len(components) != 1:
    # Boolean carving can leave tiny detached chips at the access boundaries.
    # Keep only the load-bearing connected body, then recheck every contact.
    components.sort(key=lambda v: v.volume(), reverse=True)
    removed_volume = sum(v.volume() for v in components[1:])
    print(f'Removing {removed_volume*1e6:.2f} cm3 of detached carving remnants.', flush=True)
    fixture = components[0]
    components = fixture.decompose()
fm = unpack(fixture)
assert fm.is_watertight and fm.volume > 0
# Indexed, full-precision OBJ retains tiny boolean edges that binary STL rounds
# away. The mesh is a geometry prototype, not a manufacturing-ready solid.
fm.export(OUT/'fixture.obj', file_type='obj', digits=17, include_normals=False)
if '--mesh-only' in sys.argv:
    print('Full-precision connected fixture exported.', flush=True)
    raise SystemExit(0)

# Independent surface distance, solid overlap, landing polygon and tool-ray
# checks. These checks do not imply force balance or a collision-free robot.
checks = []
for k, c in enumerate(cases):
    m = c['local']; inds = c['selected']; p = m.triangles_center[inds]
    near, gaps, _ = trimesh.proximity.closest_point(fm, p)
    rejected_seeds = [int(i) for i, gap in zip(inds, gaps) if gap >= .0002]
    c['selected'] = [int(i) for i, gap in zip(inds, gaps) if gap < .0002]
    gaps = gaps[gaps < .0002]
    assert len(c['selected']) >= 3, 'Too few contact lobes survived connected-body extraction'
    overlap = (fixture ^ solid(m)).volume()
    swept_overlap = (fixture ^ sweeps[k]).volume()
    fv = fm.vertices @ c['r'].T + c['t']
    floor = fv[fv[:, 2] < 1e-6, :2]
    hull = ConvexHull(floor)
    fc = c['r'] @ fm.center_mass + c['t']
    margin = -np.max(hull.equations[:, :2] @ fc[:2]+hull.equations[:, 2])
    # Contact patches can differ from the seed; report exact closest distances.
    origins = c['local'].triangles_center[c['work']]
    normals = c['local'].face_normals[c['work']]
    blocked = fm.ray.intersects_any(origins+normals*.0001, normals)
    checks.append(dict(pose=f'pose_{k+2}',contact_seed_gap_mm=(gaps*1000).tolist(),
                       detached_contact_seeds_removed=rejected_seeds,
                       object_overlap_mm3=overlap*1e9, withdrawal_sweep_overlap_mm3=swept_overlap*1e9,
                       fixture_min_z_mm=float(fv[:, 2].min()*1000), object_min_z_mm=float(c['world'].vertices[:, 2].min()*1000),
                       fixture_COM_floor_polygon_margin_mm=float(margin*1000),
                       work_normal_rays_tested=len(blocked),work_normal_rays_blocked=int(blocked.sum())))
    assert abs(overlap) < 1e-10 and abs(swept_overlap) < 1e-10
    assert fv[:, 2].min() > -1e-6 and margin > 0
    assert gaps.max() < .0002, gaps
    assert not blocked.any(), 'A working-face normal is blocked'

poses = []
for k, c in enumerate(cases):
    poses.append(dict(name=f'Pose {k+1}', source=f'B / pose_{k+2}',
                      fixtureR=c['r'].tolist(), fixtureT=c['t'].tolist(),
                      object=pack(c['world']), work=c['work'].tolist(), objectR=c['obj_r'].tolist(),
                      contactPoints=c['local'].triangles_center[c['selected']].tolist(),
                      contactNormals=c['local'].face_normals[c['selected']].tolist()))
scope = ('One unchanged connected solid; three saved B orientations. Numerical mesh checks cover final overlap, '
         '320 mm horizontal withdrawal sweeps and fixture-only gravity footprint. Not a task-force, structural, '
         'grasp, machining-access-cone or robot collision certificate.')
visual = fixture.calculate_normals(0, 35).to_mesh64()
visual_mesh = dict(v=np.round(visual.vert_properties[:, :3], 9).ravel().tolist(),
                   n=np.round(visual.vert_properties[:, 3:6], 7).ravel().tolist(),
                   f=visual.tri_verts.ravel().tolist())
data = dict(colors=COLORS, fixture=visual_mesh, poses=poses, scope=scope, duration=28,
            fixtureGraspPoints=[[REAR, -RAIL, .015], [REAR, -RAIL, -.015]])
# Keep the previous motion available only for preview until robot.py replaces it.
old = json.loads((OUT/'data.js').read_text().split('=', 1)[1].rstrip(';\n'))
for key in ('robot', 'kinematics', 'motion', 'duration', 'storyTimes'):
    if key in old: data[key] = old[key]
(OUT/'data.js').write_text('window.REUSE_DATA='+json.dumps(data, separators=(',', ':'))+';\n')
report = dict(sources=hashes, geometry='Shared open four-rib buttress with broad envelope-carved contact lobes and rear perimeter web',
              same_fixture_in_all_poses=True, connected_components=len(components), watertight=fm.is_watertight,
              dimensions_mm=(fm.extents*1000).tolist(), volume_cm3=float(fm.volume*1e6),
              withdrawal_length_mm=320, transverse_boolean_band_mm=.01,
              work_normal_access_expansion_mm=2, detached_volume_removed_cm3=removed_volume*1e6,
              candidate_load_probe_counts=[c['load_probes'] for c in cases],
              checks=checks, scope=scope)
(OUT/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
print(json.dumps(report, indent=2), flush=True)
