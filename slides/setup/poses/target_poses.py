"""Export named target poses and working surfaces, without running support search.

The legacy pose_1 snapshots are preserved. New targets specify gravity in
mesh coordinates and place the lowest vertex on z=0. These are held target
poses, not a claim of passive stability or a verified tipping trajectory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE = ROOT / 'slides/baseline_algo'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASELINE))
from step1.needs import COORD
import big_tip as G
from step2_local_support import render as R
from step1.needs import CONE_HALF_DEG, K, COORD

DEFINITIONS = {
    'B': (
    ('pose_2', 'Side tilt', [1., -.6, .5], 'side / foot', None),
    ('pose_3', 'Inverted / ear A', [-.3, 1., -.8], 'ear A tip', 8),
    ('pose_4', 'Inverted / ear B', [.35, 1., -.1], 'ear B tip', 70),
    ),
    'A1-f': (
        ('pose_2', 'Upright / tilted end', [.25, -1., -.35], 'lower end corner', None),
        ('pose_3', 'Inverted / opposite end', [.45, 1., -.85], 'upper end corner', None),
        ('pose_4', 'Sideways / lateral corner', [-1., .15, -.35], 'lateral corner', None),
    ),
    'C5': (
        ('pose_2', 'Opposite end down', [-.35, 1., .25], 'opposite end vertex', None),
        ('pose_3', 'Back corner down', [-.3, -.25, -1.], 'back corner vertex', None),
        ('pose_4', 'Side corner down', [1., .3, .55], 'side corner vertex', None),
    ),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def target_transform(raw, gravity_local):
    g = np.asarray(gravity_local, float)
    g /= np.linalg.norm(g)
    rotation = trimesh.geometry.align_vectors(g, [0., 0., -1.])[:3, :3]
    vertex = int(np.argmax(raw.vertices @ g))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ raw.vertices[vertex]
    return transform, vertex


def choose_region(mesh, transform, key, previous):
    world = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
    choices = []
    for candidate in G.regions(mesh, transform, key):
        mask = candidate['take']
        if not candidate['band'] or world[mesh.faces[mask], 2].min() <= G.CONTACT_EPS:
            continue
        overlap = max((mesh.area_faces[mask & old].sum() /
                       mesh.area_faces[mask | old].sum() for old in previous), default=0.)
        # Prefer a different surface patch, then an area close to 11 percent.
        choices.append((overlap, abs(candidate['frac'] - .11), candidate['seed'], candidate))
    if not choices:
        raise RuntimeError(f'{key}: no connected, reachable region within the area band')
    return min(choices, key=lambda item: item[:3])[-1]


def inspect(mesh, transform, mask, raw):
    rotation = transform[:3, :3]
    world = mesh.vertices @ rotation.T + transform[:3, 3]
    raw_world = raw.vertices @ rotation.T + transform[:3, 3]
    bottom = float(world[:, 2].min())
    contact = np.flatnonzero(np.abs(raw_world[:, 2]) < 1e-9)
    fraction = float(mesh.area_faces[mask].sum() / mesh.area)
    connectivity = G.components(mesh, mask)
    admissible = G.admissible(mesh, transform)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-12)
    assert bottom >= -1e-10 and abs(bottom) < 1e-9
    assert len(contact) == 1, f'Expected one point on the floor, got {contact}'
    assert connectivity == 1 and .08 <= fraction <= .15
    assert admissible[mask].all()
    return dict(ground_min_z_m=bottom, floor_contact_raw_vertex_ids=contact.tolist(),
                work_area_fraction=fraction, work_face_count=int(mask.sum()),
                work_components=connectivity, work_normal_rays_clear=True,
                minimum_work_outward_normal_z=float((mesh.face_normals @ rotation.T)[mask, 2].min()),
                minimum_work_vertex_z_m=float(world[mesh.faces[mask], 2].min()),
                gravity_in_mesh_frame=(rotation.T @ [0., 0., -1.]).tolist(),
                com_world_m=(rotation @ mesh.center_mass + transform[:3, 3]).tolist())


def panel(mesh, transform, mask, view, contact, size=490, closeup=False):
    world = trimesh.Trimesh(mesh.vertices @ transform[:3, :3].T + transform[:3, 3],
                            mesh.faces, process=False)
    low, high = world.bounds
    margin = .10 * world.extents.max()
    x0, y0 = COORD.floor(low) - margin
    x1, y1 = COORD.floor(high) + margin
    ground = np.array([[x0, y0, 0.], [x1, y0, 0.], [x1, y1, 0.], [x0, y1, 0.]])
    floor = ground[[[0, 1, 2], [0, 2, 3]]]
    triangles = np.concatenate([floor, world.triangles])
    colors = np.tile(R.GREY, (len(triangles), 1))
    colors[:2] = R.FLOOR
    colors[2:][mask] = R.GREEN
    basis = R.axes(view)
    points = np.concatenate([ground, world.vertices]) @ basis.T
    lo, hi = points.min(axis=0), points.max(axis=0)
    focus = ((lo + hi) / 2) @ basis
    width = 1.12 * max(hi[:2] - lo[:2])
    if closeup:
        width = .42 * world.extents.max()
        focus = contact + np.array([0., 0., .10 * world.extents.max()])
    picture, _ = R.raster(triangles, colors, focus, basis, width, size, unlit=(0, 1))
    draw = ImageDraw.Draw(picture)
    x, y = R.project(np.asarray(contact), focus, basis, width, size)[:2]
    draw.ellipse((x-6, y-6, x+6, y+6), outline=tuple(R.ORANGE.astype(int)), width=3)
    if closeup:
        draw.text((18, size-38), 'Orange ring: actual floor contact', font=R.font(17), fill=R.INK)
    return picture


def draw_pose(mesh, transform, mask, metadata):
    rotation = transform[:3, :3]
    profile = rotation @ [-1., .1, .4]
    profile[2] = 0.
    profile /= np.linalg.norm(profile)
    profile[2] = .38
    work_view = np.average((mesh.face_normals @ rotation.T)[mask],
                           axis=0, weights=mesh.area_faces[mask])
    work_view[2] = max(work_view[2], .45)
    contact = np.asarray(metadata['floor_contact_m'])
    width, height = 1510, 604
    page = Image.new('RGB', (width, height), R.PAPER)
    draw = ImageDraw.Draw(page)
    title = f"{metadata['pose_id']}  |  {metadata['label']}"
    fraction = metadata['checks']['work_area_fraction']
    draw.text((24, 14), title, font=R.font(26), fill=R.INK)
    draw.text((24, 51), f'Green: working surface {100*fraction:.2f}%    |    Ground contact: {metadata["contact_anatomy"]}',
              font=R.font(18), fill=R.INK)
    for column, (view, title) in enumerate(((profile, 'Pose'), (work_view, 'Working surface'),
                                           (profile, 'Floor contact / enlarged'))):
        x = 10 + column * 500
        page.paste(panel(mesh, transform, mask, view, contact, closeup=column == 2), (x, 105))
        draw.text((x+14, 82), title, font=R.font(17), fill=R.INK)
    return page


def build(name='B'):
    # These copies are geometry sources only; preserve legacy bytes/provenance.
    source = ROOT / 'slides/obj_supp/area' / f'demand_{name}_tip1.npz'
    target = HERE / name / 'pose_1/setup.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    elif digest(target) != digest(source):
        raise ValueError(f'{name}: pose_1 differs from the preserved legacy setup')

    raw = trimesh.load(ROOT/'objects'/name/'mesh.stl', force='mesh')
    mesh, rounds = G.refine(raw)
    with np.load(HERE/name/'pose_1/setup.npz') as z:
        legacy = {key: z[key].copy() for key in z.files}
    previous = [legacy['work_faces']]
    specs = [('pose_1', 'Original tilt', legacy['T_world_mesh'],
              legacy['work_faces'], 'original floor point', None, None)]
    for pose, label, gravity, anatomy, expected_vertex in DEFINITIONS[name]:
        transform, vertex = target_transform(raw, gravity)
        if expected_vertex is not None:
            assert vertex == expected_vertex, f'{pose}: the intended ear is no longer lowest'
        region = choose_region(mesh, transform, name + '_' + pose, previous)
        previous.append(region['take'])
        specs.append((pose, label, transform, region['take'], anatomy, vertex, region['seed']))

    reports, images = [], []
    for pose, label, transform, mask, anatomy, vertex, seed in specs:
        folder = HERE/name/pose
        folder.mkdir(parents=True, exist_ok=True)
        checks = inspect(mesh, transform, mask, raw)
        vertex = checks['floor_contact_raw_vertex_ids'][0]
        contact = transform[:3, :3] @ raw.vertices[vertex] + transform[:3, 3]
        contact[2] = 0.
        if pose != 'pose_1':
            np.savez_compressed(folder/'setup.npz', object=name, pose_id=pose,
                T_world_mesh=transform, com_m=checks['com_world_m'], work_faces=mask,
                floor_contact_m=contact, mesh_sha256=digest(ROOT/'objects'/name/'mesh.stl'),
                poses_sha256=digest(ROOT/'objects'/name/'poses.json'), K=K,
                cone_half_deg=CONE_HALF_DEG, tip=-1)
        metadata = dict(schema_version=1, object=name, pose_id=pose, label=label,
            target_pose_only=True, placement_trajectory_verified=False,
            support_search_run=False, contact_anatomy=anatomy,
            T_world_mesh=transform.tolist(), floor_contact_m=contact.tolist(),
            work_face_ids=np.flatnonzero(mask).tolist(), work_seed=seed,
            uniform_subdivision_rounds=rounds, checks=checks,
            source_snapshot='setup.npz', source_snapshot_sha256=digest(folder/'setup.npz'),
            generator=str(Path(__file__).relative_to(ROOT)), generator_sha256=digest(__file__),
            work_region_method='Connected geodesic region; outward normal z > 0.35; '
                               'face-center outward rays clear; 8-15% total mesh area',
            reachability_scope='Normal rays only; Step 1 later checks each sampled cone direction')
        image = draw_pose(mesh, transform, mask, metadata)
        image.save(folder/'pose.png')
        save(folder/'setup.json', metadata)
        reports.append(metadata)
        images.append(image)
        print(f'{name}/{pose}: vertex {vertex} ({anatomy}), work {checks["work_area_fraction"]:.2%}', flush=True)
    pairs = []
    for i, a in enumerate(reports):
        for j in range(i+1, len(reports)):
            b = reports[j]
            angle = np.degrees(np.arccos(np.clip(np.dot(a['checks']['gravity_in_mesh_frame'],
                                                       b['checks']['gravity_in_mesh_frame']), -1, 1)))
            intersection = mesh.area_faces[previous[i] & previous[j]].sum()
            union = mesh.area_faces[previous[i] | previous[j]].sum()
            assert angle > (20. if name == 'B' else 60.), 'Targets must differ in tilt, not just world yaw'
            assert intersection / union < .8, 'Work surfaces must be different'
            pairs.append(dict(poses=[a['pose_id'], b['pose_id']], gravity_angle_deg=float(angle),
                              work_area_jaccard=float(intersection / union)))
    save(HERE/name/'poses.json', dict(object=name, status='setup_ready',
        poses=[dict(pose_id=r['pose_id'], label=r['label'], setup=f'{r["pose_id"]}/setup.json',
                    preview=f'{r["pose_id"]}/pose.png') for r in reports], pairwise_checks=pairs))
    gallery = Image.new('RGB', (images[0].width, sum(im.height for im in images)+80), R.PAPER)
    draw = ImageDraw.Draw(gallery)
    draw.text((24, 12), f'{name} / target poses and working surfaces', font=R.font(30), fill=R.INK)
    draw.text((24, 52), 'Setup geometry only  |  Each pose has its own working surface', font=R.font(18), fill=R.INK)
    y = 80
    for picture in images:
        gallery.paste(picture, (0, y))
        y += picture.height
    gallery.save(HERE/name/'poses.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*', help='A1-f, B or C5 (default: B)')
    args = parser.parse_args()
    if any(name not in DEFINITIONS for name in args.objects):
        parser.error('objects must be A1-f, B or C5')
    for name in args.objects or ['B']:
        build(name)
