"""Replay the original B/pose1+3 construction illustration with white bodies.

Run with the cadgrasp Python environment. All outputs stay beside this script;
baseline inputs are read-only. Rendering is CPU-only (no browser required).
This is a historical illustration, not a new shared-head feasibility result.
Use --diagnostic to render the registration check into its own subdirectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import manifold3d as md
import numpy as np
from PIL import Image, ImageColor, ImageDraw
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial import ConvexHull
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'slides/baseline_algo'))
from step2_local_support import geometry as G, render as R
from step4_connect_support import build_coupled_saddle as S
from step4_connect_support.run_fast_local_bodies import recipe
from step4_connect_support import head_registration as H

SOURCE = ROOT / 'slides/baseline_algo/output/B/pose1+3/step4'
ARCHIVE = SOURCE / 'data/history/before_fast_construction'
HISTORICAL_REPORT = ARCHIVE / '028c5be2286a_report.json'
BODY = '#dce2e2'
OBJECT = '#aebabe'
OBJECT_ALPHA = .24
OBJECT_POSE = 'pose_1'
INK = '#263d49'
MUTED = '#667c87'
FLOOR_COLORS = ['#d98a3b', '#438cba']
VIEW = [-1., -1., .82]


def union(values):
    return md.Manifold.batch_boolean(list(values), md.OpType.Add)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_stages():
    case, placement, _, _ = recipe(SOURCE)
    data = SOURCE / 'data'
    report = json.loads(HISTORICAL_REPORT.read_text())

    def archived(name):
        """Resolve the exact archived artifact recorded by the old renderer inputs."""
        sha = report['artifacts'][name]
        basename = Path(name).name
        for path in (ARCHIVE / f'{sha[:12]}_{basename}', ARCHIVE / basename):
            if path.exists() and digest(path) == sha:
                return path
        raise FileNotFoundError(f'Missing archived illustration input: {name} ({sha})')

    design_path = archived('design.json')
    design = json.loads(design_path.read_text())
    if any(c.get('fallback') for c in design['connections']):
        raise ValueError('This illustration expects the ordinary construction.')
    for viewer_path in sorted(ARCHIVE.glob('*index.html')):
        viewer, _ = json.JSONDecoder().raw_decode(
            viewer_path.read_text().split('const DATA=', 1)[1])
        if viewer['report'].get('artifacts') == report['artifacts']:
            break
    else:
        raise FileNotFoundError('No viewer matches the archived construction artifacts.')
    colors = {p['id']: p['color'] for p in viewer['parts']}
    heads, head_colors = [], []
    # Replay the historical six patches exactly, without certifying shared identity.
    for contacts, cells, basis, offset in zip(case.groups, case.heads,
                                             placement['bases'], placement['offsets']):
        for contact, pieces in zip(contacts, cells):
            heads.append(union(S.solid(G.hull_mesh(v @ basis + offset)) for v in pieces))
            head_colors.append(colors[contact['candidate_id']])
    initial, bodies, paths = [], [], list(case.paths)
    for index, body in enumerate(design['local_bodies']):
        seed = data / f'cache/initial_body_{index}.npz'
        path = archived(body['file'])
        with np.load(seed) as saved:
            floor = json.loads(str(saved['initial_floor_json']))
            assert floor['target_floor_index'] == body['initial_floor']['target_floor_index']
            initial.append(S.solid(trimesh.Trimesh(saved['v'], saved['f'], process=False)))
        bodies.append(S.solid(trimesh.load(path, force='mesh', process=False)))
        paths.extend([seed, path])
    final_path = archived('../shape.obj')
    final_mesh = trimesh.load(final_path, force='mesh', process=False)
    stages = [union(heads), union(initial), union(bodies), S.solid(final_mesh)]
    missing = [abs((a - b).volume()) * S.SCALE**3 for a, b in zip(stages, stages[1:])]
    if max(missing) > 8e-14:
        raise ValueError(f'Saved stages are not a nested construction: {missing}')
    if len(stages[-1].decompose()) != 1 or not final_mesh.is_watertight:
        raise ValueError('The final saved fixture must be one closed solid.')
    np.testing.assert_allclose(final_mesh.volume * 1e6, report['volume_cm3'], atol=1e-5)
    # Use the saved final OBJ directly, preserving its final exterior exactly.
    meshes = [S.unpack(s) for s in stages[:-1]] + [final_mesh]
    additions = [None] + [S.unpack(b - a) for a, b in zip(stages, stages[1:])]
    paths.extend([final_path, HISTORICAL_REPORT, design_path,
                  viewer_path, data / 'history/reference_report.json'])
    metadata = dict(
        status='historical_construction_illustration',
        shared_head_identity_valid=False, geometry_validation_performed=False,
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))},
        stage_volume_cm3=[float(s.volume() * S.SCALE**3 * 1e6) for s in stages],
        stage_component_count=[len(s.decompose()) for s in stages],
        missing_volume_between_stages_m3=missing,
        poses=case.poses, floor_samples_per_pose=[len(d) for d in case.demands],
        physical_head_patches=len(heads), head_ids=len(colors),
        final_mesh_is_saved_obj=True, geometry_rebuilt=False, final_audit_rerun=False,
        fallback_shown=False,
        display=dict(view=VIEW, object_shown=True, object_pose=OBJECT_POSE,
                     object_color=OBJECT, object_opacity=OBJECT_ALPHA,
                     body_color=BODY, new_material_highlight=False,
                     head_collar_m=.004,
                     all_floor_points_rendered=True,
                     both_floor_clouds_in_every_step=True,
                     coordinate_frame='saved common fixture frame',
                     floor_bases=placement['bases'].tolist(),
                     floor_offsets_m=placement['offsets'].tolist()))
    return case, placement, meshes, additions, list(zip(heads, head_colors)), viewer['head_regions'], colors, metadata


def clip_box(triangles, low, high):
    """Clip display colors to exterior triangles; never add physical material."""
    candidates = triangles[((triangles.max(1) >= low) & (triangles.min(1) <= high)).all(1)]
    result = []
    for tri in candidates:
        polygon = list(tri)
        for axis in range(3):
            for bound, sign in ((low[axis], 1), (high[axis], -1)):
                output = []
                for p, q in zip(polygon, polygon[1:] + polygon[:1]):
                    dp, dq = sign * (p[axis] - bound), sign * (q[axis] - bound)
                    if dp >= 0:
                        output.append(p)
                    if (dp >= 0) != (dq >= 0):
                        output.append(p + dp / (dp - dq) * (q - p))
                polygon = output
                if not polygon:
                    break
            if not polygon:
                break
        for j in range(1, len(polygon) - 1):
            result.append([polygon[0], polygon[j], polygon[j + 1]])
    return np.asarray(result).reshape(-1, 3, 3)


def raster(triangles, colors, camera, size, bias=None, unlit=()):
    """Small z-buffer renderer, adapted from step2_local_support/render.py.

    Returns depth as well as color so the ghost workpiece can be composited
    only where it is in front of the actual support geometry.
    """
    focus, basis, width = camera
    pixels = np.full((size, size, 3), 255, np.uint8)
    depth = np.full((size, size), -np.inf)
    ids = np.full((size, size), -1, np.int32)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    light = .7 * basis[2] + .3 * basis[1] - .25 * basis[0]
    light /= np.linalg.norm(light)
    shade = .62 + .38 * np.maximum(0, normals @ light)
    lit = np.clip(colors * shade[:, None], 0, 255).astype(np.uint8)
    lit[list(unlit)] = colors[list(unlit)]
    points = R.project(triangles, focus, basis, width, size)
    if bias is not None:
        points[:, :, 2] += np.asarray(bias)[:, None] * width * 1e-5
    low = np.maximum(0, np.floor(points[:, :, :2].min(1)).astype(int))
    high = np.minimum(size - 1, np.ceil(points[:, :, :2].max(1)).astype(int))
    for k in np.flatnonzero((high >= low).all(1)):
        xmin, ymin = low[k]
        xmax, ymax = high[k]
        a, b, c = points[k]
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
        xx, yy = xx + .5, yy + .5
        u = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / det
        v = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / det
        w = 1 - u - v
        z = u * a[2] + v * b[2] + w * c[2]
        old = depth[ymin:ymax+1, xmin:xmax+1]
        hit = (u >= -1e-10) & (v >= -1e-10) & (w >= -1e-10) & (z > old)
        old[hit] = z[hit]
        pixels[ymin:ymax+1, xmin:xmax+1][hit] = lit[k]
        ids[ymin:ymax+1, xmin:xmax+1][hit] = k
    return pixels, depth, ids


def rgb(color):
    return np.asarray(ImageColor.getrgb(color), float)


def object_layer(mesh, camera, size):
    """Faint workpiece surface, depth-sorted with the heads and floor guides."""
    pixels, depth, ids = raster(mesh.triangles,
                                np.tile(rgb(OBJECT), (len(mesh.faces), 1)), camera, size)
    return pixels, depth, np.where(ids >= 0, OBJECT_ALPHA, 0.)


def floor_layer(floor, color, camera, size):
    """Translucent floor guide with all saved points and their dashed hull."""
    triangles = floor['corners'][[[0, 1, 2], [0, 2, 3]]]
    tint = rgb('#e5e8e9')
    pixels, depth, ids = raster(triangles, np.tile(tint, (2, 1)), camera, size, unlit=(0, 1))
    alpha = np.where(ids >= 0, .46, 0.)
    p = R.project(floor['cloud'], *camera, size)
    ix, iy = np.rint(p[:, 0]).astype(int), np.rint(p[:, 1]).astype(int)
    assert ((ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)).all()
    counts = np.zeros((size, size))
    np.add.at(counts, (iy, ix), 1.)
    cloud_alpha = 1 - np.exp(-1.7 * gaussian_filter(counts, sigma=.65))
    guide = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(guide)
    h = R.project(floor['hull'], *camera, size)
    for a, b in zip(h, np.roll(h, -1, axis=0)):
        count = max(2, int(np.linalg.norm(a[:2] - b[:2])))
        for i in range(0, count, 13):
            start = a[:2] + (b[:2] - a[:2]) * i / count
            end = a[:2] + (b[:2] - a[:2]) * min(i+8, count) / count
            draw.line([tuple(start), tuple(end)], fill=220, width=2)
    mark_alpha = np.maximum(cloud_alpha, np.asarray(guide) / 255.)
    mark_alpha[ids < 0] = 0
    total = mark_alpha + alpha * (1 - mark_alpha)
    numerator = rgb(color) * mark_alpha[..., None] + pixels * (alpha * (1 - mark_alpha))[..., None]
    return numerator / np.maximum(total[..., None], 1e-20), depth, total


def composite(layers):
    """Depth-sort transparent layers per pixel, including intersecting planes."""
    depths = np.stack([layer[1] for layer in layers])
    order = np.argsort(depths, axis=0)
    pixels = np.stack([layer[0] for layer in layers])
    alpha = np.stack([layer[2] for layer in layers])
    height, width = depths.shape[1:]
    canvas = np.full((height, width, 3), 255.)
    yy, xx = np.indices((height, width))
    for rank in order:
        a = alpha[rank, yy, xx, None]
        canvas = canvas * (1-a) + pixels[rank, yy, xx] * a
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def scene(mesh, addition, heads, regions, colors, stage, camera, context_layers, size):
    triangles, palette, biases = [], [], []

    def add(tris, color, bias=0):
        if len(tris):
            triangles.append(tris)
            palette.append(np.tile(rgb(color), (len(tris), 1)))
            biases.append(np.full(len(tris), bias))

    # Keep the complete body grey-white throughout growth; only heads retain color.
    add(mesh.triangles, BODY)
    if stage == 0:
        for head, color in heads:
            add(S.unpack(head).triangles, color, 2)
    else:
        for region in regions:
            add(clip_box(mesh.triangles, np.asarray(region['low']), np.asarray(region['high'])), colors[region['id']], 2)
    pixels, depth, ids = raster(np.concatenate(triangles), np.concatenate(palette),
                                camera, size, np.concatenate(biases))
    support = ids >= 0
    edge = support & ~binary_erosion(support)
    pixels[edge] = (pixels[edge] * .80).astype(np.uint8)
    return composite(context_layers + [(pixels, depth, support.astype(float))])


def label(draw, xy, text, size, color=INK, anchor='mm'):
    draw.text(xy, text, font=R.font(size), fill=color, anchor=anchor)


def run_construction(panel_size=720):
    case, placement, meshes, additions, heads, regions, colors, metadata = read_stages()
    size = panel_size
    basis = R.axes(VIEW)
    object_index = case.poses.index(OBJECT_POSE)
    original = case.tasks[object_index].domain.mesh
    object_basis = placement['bases'][object_index]
    object_offset = placement['offsets'][object_index]
    workpiece = trimesh.Trimesh(original.vertices @ object_basis + object_offset,
                               original.faces, process=False)
    np.testing.assert_allclose((workpiece.vertices-object_offset) @ object_basis.T,
                               original.vertices, atol=1e-12)
    metadata['display'].update(object_basis=object_basis.tolist(),
                               object_offset_m=object_offset.tolist())
    floors = []
    for demands, b, o in zip(case.demands, placement['bases'], placement['offsets']):
        lo, hi = demands.min(0) - .014, demands.max(0) + .014
        corners = np.array([[lo[0], lo[1], 0.], [hi[0], lo[1], 0.],
                            [hi[0], hi[1], 0.], [lo[0], hi[1], 0.]])
        hull = demands[ConvexHull(demands).vertices]
        cloud = np.c_[demands, np.zeros(len(demands))] @ b + o
        np.testing.assert_allclose((cloud-o) @ b.T,
                                   np.c_[demands, np.zeros(len(demands))], atol=1e-12)
        floors.append(dict(corners=corners @ b + o, cloud=cloud,
                           hull=np.c_[hull, np.zeros(len(hull))] @ b + o))
    points = np.concatenate([meshes[-1].vertices, workpiece.vertices]
                            + [f['corners'] for f in floors])
    view = points @ basis.T
    low, high = view.min(0), view.max(0)
    width = 1.13 * max(high[:2] - low[:2])
    camera = [((low+high)/2) @ basis, basis, width]
    context_layers = [floor_layer(f, color, camera, size) for f, color in zip(floors, FLOOR_COLORS)]
    context_layers.append(object_layer(workpiece, camera, size))
    metadata['display']['shared_camera_width_m'] = float(width)

    margin, gap, top = 88, 16, 242
    panel_height = size
    page = Image.new('RGB', (2 * margin + 4 * size + 3 * gap, top + panel_height + 180), 'white')
    draw = ImageDraw.Draw(page)
    label(draw, (margin, 48), 'From contact heads to one shared fixture', 45, anchor='lm')
    label(draw, (margin, 102), 'B / Pose 1 + Pose 3   ·   Both floor demands in every step', 25, MUTED, 'lm')
    titles = ['Given heads', 'Grow bodies', 'Add floor contacts', 'Connect']
    descriptions = ['Keep the contact geometry', 'Each head → nearest legal floor', 'Extend to dispersed foot regions', 'Join nearby bodies with short bridges']
    details = ['Heads + both floor-demand clouds', 'Local bodies reach their chosen floors', 'Foot contacts span both required regions', 'One connected fixture']
    assets = HERE / 'steps_data'
    assets.mkdir(exist_ok=True)
    for column in range(4):
        x = margin + column * (size + gap)
        label(draw, (x + 22, 164), f'{column + 1:02d}', 27, '#438499', 'lm')
        label(draw, (x + 76, 164), titles[column], 30, INK, 'lm')
        label(draw, (x + size/2, 206), descriptions[column], 20, MUTED)
        if column < 3:
            ax = x + size - 19
            draw.line((ax-16, 164, ax+16, 164), fill='#91a5ad', width=2)
            draw.line((ax+8, 157, ax+16, 164, ax+8, 171), fill='#91a5ad', width=2)
        print(f'Rendering both floors / {titles[column]}', flush=True)
        panel = scene(meshes[column], additions[column], heads, regions, colors, column,
                      camera, context_layers, size)
        panel.save(assets / f'step_{column+1:02d}.png')
        page.paste(panel, (x, top))
        label(draw, (x + size/2, top + panel_height + 22), details[column], 21, MUTED)
    y = page.height - 92
    labels = [(BODY, 'Support body'),
              (FLOOR_COLORS[0], 'Pose 1 floor demands'), (FLOOR_COLORS[1], 'Pose 3 floor demands')]
    x = margin
    for color, text in labels:
        draw.rounded_rectangle((x, y-10, x+30, y+10), radius=4, fill=color)
        label(draw, (x+42, y), text, 23, MUTED, 'lm')
        x += 450
    for i, color in enumerate(colors.values()):
        draw.ellipse((x+i*17, y-8, x+i*17+18, y+10), fill=color)
    label(draw, (x+108, y), 'Retained heads', 23, MUTED, 'lm')
    x += 450
    ghost_color = tuple(np.rint(rgb(OBJECT)*OBJECT_ALPHA + 255*(1-OBJECT_ALPHA)).astype(int))
    draw.rounded_rectangle((x, y-10, x+30, y+10), radius=4, fill=ghost_color)
    label(draw, (x+42, y), 'B / Pose 1 (ghost)', 23, MUTED, 'lm')
    label(draw, (margin, page.height-40), 'One fixture frame and one fixed view. 32,768 floor points per pose; dashed hulls are guides, not solid bases.', 23, MUTED, 'lm')
    output = HERE / 'steps.png'
    page.save(output, dpi=(200, 200))
    for path, sha in metadata['source_sha256'].items():
        assert digest(ROOT / path) == sha, f'Input changed: {path}'
    metadata['figure_sha256'] = digest(output)
    (assets / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # Retire only the eight panels generated by the old two-row presentation.
    for pose in case.poses:
        for stage in range(1, 5):
            (assets / f'{pose}_{stage:02d}.png').unlink(missing_ok=True)
    print(output, flush=True)
    print(json.dumps({k: metadata[k] for k in ['stage_volume_cm3', 'stage_component_count', 'missing_volume_between_stages_m3']}), flush=True)


def run(panel_size=720, diagnostic=False):
    """Keep the requested construction illustration separate from diagnostics."""
    if not diagnostic:
        run_construction(panel_size)
        return
    case, placement, _, reference = recipe(SOURCE)
    try:
        H.require_saved_registration(case, placement)
    except H.SharedHeadRegistrationError as error:
        render_registration_failure(case, reference, error.registration_checks, panel_size)
        return
    print('Saved layout passes registration; no failure diagnostic written.', flush=True)


def render_registration_failure(case, reference, checks, panel_size):
    exact = checks['exact_registration']
    bases, offsets = np.asarray(exact['bases']), np.asarray(exact['offsets'])
    heads, registration = H.register(case.groups, case.heads, bases, offsets)
    shared_id = case.schedule['shared_head']['selected_id']
    colors = {shared_id: '#dc9d47'}
    colors.update(zip([i for i in case.schedule['selected_ids'] if i != shared_id],
                      ['#ac7098', '#7196c0', '#50a59b', '#77a76a']))
    surfaces = [(union([S.solid(G.hull_mesh(v)) for v in h.cells]), colors[h.ident])
                for h in heads]
    head_mesh = S.unpack(union([s for s, _ in surfaces]))
    workpiece = case.tasks[0].domain.mesh.copy()
    workpiece.vertices = workpiece.vertices @ bases[0] + offsets[0]
    floors = []
    for demands, basis, offset in zip(case.demands, bases, offsets):
        lo, hi = demands.min(0)-.014, demands.max(0)+.014
        corners = np.array([[lo[0],lo[1],0], [hi[0],lo[1],0],
                            [hi[0],hi[1],0], [lo[0],hi[1],0]])
        hull = demands[ConvexHull(demands).vertices]
        floors.append(dict(corners=corners@basis+offset,
            cloud=np.c_[demands, np.zeros(len(demands))]@basis+offset,
            hull=np.c_[hull, np.zeros(len(hull))]@basis+offset))
    size = max(900, panel_size)
    basis = R.axes(VIEW)
    points = np.concatenate([head_mesh.vertices, workpiece.vertices]+[f['corners'] for f in floors])
    view = points@basis.T
    low, high = view.min(0), view.max(0)
    width = 1.13*max(high[:2]-low[:2])
    camera = [((low+high)/2)@basis, basis, width]
    layers = [floor_layer(f, c, camera, size) for f,c in zip(floors,FLOOR_COLORS)]
    layers.append(object_layer(workpiece, camera, size))
    panel = scene(head_mesh, None, surfaces, [], colors, 0, camera, layers, size)
    page = Image.new('RGB', (2*size+160, size+250), 'white')
    page.paste(panel, (35, 180))
    draw = ImageDraw.Draw(page)
    label(draw, (60, 48), 'B / Pose 1 + Pose 3: shared-head correction', 39, anchor='lm')
    label(draw, (60, 106), 'One yellow head shared by both poses. Construction stopped before body growth.',
          25, MUTED, 'lm')
    x = size+75
    label(draw, (x, 240), '5 physical heads; 1 shared yellow head', 29, anchor='lm')
    label(draw, (x, 298), 'Both copies coincide in the fixture frame.', 24, MUTED, 'lm')
    label(draw, (x, 342), 'The object and both floor planes use this registration.', 24, MUTED, 'lm')
    floor_failed = not exact['floor_compatibility']['passed']
    status = 'Fixed contact correspondence fails floor feasibility' if floor_failed else 'Body construction must be rerun'
    label(draw, (x, 440), status, 28, '#a63b2e', 'lm')
    y = 500
    for row in exact['floor_compatibility']['per_pose']:
        if not row['passed']:
            name = row['pose'].replace('_', ' ').title()
            label(draw, (x, y), f"{name}: {row['violating_sample_count']:,} / {row['original_sample_count']:,} floor demands", 25, '#a63b2e', 'lm')
            label(draw, (x, y+44), 'fall outside the region allowed by the other floor.', 24, MUTED, 'lm')
            y += 100
    label(draw, (x, y+35), 'The previous six-patch construction was invalid.', 24, MUTED, 'lm')
    label(draw, (x, y+79), 'Moving the yellow copy alone cannot repair its bodies.', 24, MUTED, 'lm')
    shared = next(h for h in heads if h.ident == shared_id)
    point = R.project(shared.contact_points.mean(0)[None], *camera, size)[0]
    px, py = float(point[0]+35), float(point[1]+180)
    label(draw, (x, 840), 'One shared yellow head', 24, '#ac732b', 'lm')
    draw.line([(x-15,840), (x-60,840), (px,py)], fill='#ac732b', width=2)
    draw.ellipse((px-5,py-5,px+5,py+5), outline='#ac732b', width=2)
    label(draw, (60, page.height-43), 'Registered heads only | Faint B / Pose 1 object | Both original floor-demand clouds | No fixture exported',
          23, MUTED, 'lm')
    assets = HERE/'steps_data/registration_diagnostic'
    assets.mkdir(parents=True, exist_ok=True)
    panel.save(assets/'registered_heads.png')
    output = assets/'steps.png'
    page.save(output, dpi=(200,200))
    paths = set(case.paths) | {SOURCE/'data/history/reference_report.json'}
    metadata = dict(status='shared_head_registration_failed', fixture_constructed=False,
        source_schedule=reference['source_schedule'], registration_checks=checks,
        physical_head_count=registration['physical_head_count'],
        display=dict(object_shown=True, object_pose=OBJECT_POSE, object_opacity=OBJECT_ALPHA,
                     both_floor_clouds_in_every_step=True, coordinate_frame='exact shared-head frame'),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)},
        figure_sha256=digest(output))
    (assets/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output, flush=True)
    print('Rejected six-patch construction. Showing exact registration and floor failure.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-size', type=int, default=720)
    parser.add_argument('--diagnostic', action='store_true',
                        help='Write registration diagnostics separately from steps.png.')
    args = parser.parse_args()
    run(args.panel_size, args.diagnostic)
