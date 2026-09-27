"""Three spatial force-line constructions on the CURRENT objects/B/pose_2.

X intersects gravity's line through c and the push line through q. p intersects
the resultant through X with z=0. General 3-D force lines can be skew: these
examples deliberately use concurrent loads within the saved work patch/cone.
Full force/moment equivalence, including yaw, is checked for every example.
Only the three individual PNGs and row3.png are written; inputs are read-only.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.dont_write_bytecode = True
import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'tools'))
import slide_scene as S
# Several slide folders contain presentation.py; load this neighbor explicitly.
_spec = importlib.util.spec_from_file_location('floor_row3_landings', HERE / 'presentation.py')
_floor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_floor)
landings = _floor.landings

OUT = HERE / 'row3.png'
VIEW = np.array([-1., -.6, .75])
VIEW /= np.linalg.norm(VIEW)
WEIGHT = np.array([0., 0., -1.])  # body weights; positions in metres
BLACK = (40, 45, 50)
FULL_PUSH, LIGHT_PUSH = .5, .1
FORCE_PIXELS_PER_MG = 240.
ARROW_GAP_FROM_X = 64.
DIAGRAM_TOP_PADDING = 200


def load_case():
    folder = ROOT / 'objects/B'
    task = folder / 'tasks/pose_2'
    setup = json.loads((task / 'setup.json').read_text())
    record = json.loads((folder / 'poses.json').read_text())
    mesh = trimesh.load(folder / 'mesh.stl', force='mesh')
    for _ in range(setup['uniform_subdivision_rounds']):
        mesh = mesh.subdivide()
    with np.load(task / 'setup.npz') as saved:
        T = saved['T_world_mesh'].copy()
        work = saved['work_faces'].copy()
        com = saved['com_m'].copy()
        K = float(saved['K'])
        cone = float(saved['cone_half_deg'])
        assert str(saved['poses_sha256']) == hashlib.sha256((folder / 'poses.json').read_bytes()).hexdigest()
    np.testing.assert_array_equal(T, record['poses'][1]['T_world_mesh'])
    np.testing.assert_array_equal(np.flatnonzero(work), setup['work_face_ids'])
    np.testing.assert_allclose(com, T[:3, :3] @ mesh.center_mass + T[:3, 3], atol=1e-12)
    mesh.apply_transform(T)
    assert abs(mesh.vertices[:, 2].min()) < 1e-10
    return SimpleNamespace(mesh=mesh, work_ids=np.flatnonzero(work), com=com,
        K=K, cone_half_deg=cone, data=dict(object='B', pose_id='pose_2'),
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (folder / 'mesh.stl', folder / 'poses.json',
                           task / 'setup.json', task / 'setup.npz')})


def crossing(com, q, force):
    """Intersect q+t*force with the vertical through com; reject skew lines."""
    com, q, force = (np.asarray(v, dtype=float) for v in (com, q, force))
    square = float(force[:2] @ force[:2])
    if square < 1e-20:
        raise ValueError('Parallel vertical force lines do not define a unique X')
    t = float((com[:2] - q[:2]) @ force[:2] / square)
    X = q + t * force
    if np.linalg.norm(X[:2] - com[:2]) > 1e-10:
        raise ValueError('The force lines are skew; there is no intersection X')
    return X


def construction(com, q, force):
    """Check the line construction independently against moment balance."""
    force = np.asarray(force, dtype=float)
    X = crossing(com, q, force)
    W = WEIGHT + force
    if W[2] >= 0:
        raise ValueError('A downward resultant is required')
    p = X - (X[2] / W[2]) * W
    from_moments, _ = landings(q, force, com)
    np.testing.assert_allclose(p, from_moments, atol=1e-12, rtol=0)
    residual = np.cross(com, WEIGHT) + np.cross(q, force) - np.cross(p, W)
    np.testing.assert_allclose(residual, 0., atol=1e-12, rtol=0)
    return dict(q=np.asarray(q), force=force, X=X, W=W, p=p,
                moment_residual=residual)


def examples(domain):
    """Two different 0.5mg loads; then repeat load 2 at exactly 0.1mg."""
    if domain.K < FULL_PUSH:
        raise ValueError('The saved load bound does not permit the requested 0.5mg examples')
    mesh = domain.mesh
    extent = float(mesh.extents.max())
    pool = []
    top = float(mesh.bounds[1, 2])
    for face in domain.work_ids:
        q, normal = mesh.triangles_center[face], mesh.face_normals[face]
        for z in np.linspace(top + .10 * extent, top + .55 * extent, 64):
            X = np.r_[domain.com[:2], z]
            direction = q - X
            length = np.linalg.norm(direction)
            if length < .35 * extent:
                continue
            direction /= length
            cosine = float(-normal @ direction)
            if cosine < np.cos(np.radians(domain.cone_half_deg)) + 1e-10:
                continue
            force = FULL_PUSH * direction
            W = WEIGHT + force
            p = X - X[2] / W[2] * W
            pool.append(dict(face=int(face), q=q, force=force, X=X, W=W, p=p,
                cone_angle_deg=float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))))
    if not pool:
        raise ValueError('No admissible concurrent loads on the current work patch')
    q = np.array([item['q'] for item in pool])
    p = np.array([item['p'] for item in pool])
    directions = np.array([item['force'] / FULL_PUSH for item in pool])
    rays = np.tile(VIEW, (len(pool), 1))
    keep = ~mesh.ray.intersects_any(q - 1e-5 * directions, -directions)
    keep &= ~mesh.ray.intersects_any(q + 1e-5 * VIEW, rays)
    keep &= ~mesh.ray.intersects_any(p + 1e-5 * VIEW, rays)
    X = np.array([item['X'] for item in pool])
    light_W = WEIGHT + LIGHT_PUSH * directions
    light_p = X - (X[:, 2] / light_W[:, 2])[:, None] * light_W
    keep &= ~mesh.ray.intersects_any(light_p + 1e-5 * VIEW, rays)
    right = S.R.axes(VIEW)[0]
    keep &= np.abs((q - domain.com) @ right) > .04 * extent
    candidates = np.flatnonzero(keep)
    if len(candidates) < 2:
        raise ValueError('Too few visible, separated concurrent loads')
    second = int(candidates[np.argmax(np.linalg.norm(p[candidates, :2] - domain.com[:2], axis=1))])
    distance = np.linalg.norm(q[candidates] - q[second], axis=1)
    angle = np.degrees(np.arccos(np.clip(directions[candidates] @ directions[second], -1., 1.)))
    valid = (distance > .15 * extent) & (angle > 25.)
    if not valid.any():
        raise ValueError('Cannot show sufficiently different positions and directions')
    score = distance / extent + angle / 90.
    first = int(candidates[np.argmax(np.where(valid, score, -np.inf))])
    chosen = [dict(pool[first]), dict(pool[second])]
    for item in chosen:
        item.update(construction(domain.com, item['q'], item['force']))
        np.testing.assert_allclose(np.linalg.norm(item['force']), FULL_PUSH, atol=1e-12)
    third = dict(chosen[1])
    third.update(construction(domain.com, chosen[1]['q'],
                              chosen[1]['force'] * (LIGHT_PUSH / FULL_PUSH)))
    chosen.append(third)
    return chosen


def dashed(draw, start, end, color, width=3, dash=10, gap=9):
    a, b = np.asarray(start), np.asarray(end)
    length = float(np.linalg.norm(b - a))
    if length < 1e-8:
        return
    direction = (b - a) / length
    for t in np.arange(0., length, dash + gap):
        draw.line([tuple(a + t * direction), tuple(a + min(t + dash, length) * direction)],
                  fill=color, width=width)


def force_arrow(camera, point, force, *, tip_at_point=False):
    """Projected direction, but a common screen-length scale for force magnitude.

    These are force glyphs, not displacements: avoiding foreshortening makes
    equal loads equally long and the 0.1mg arrow exactly 1/5 of the 0.5mg one.
    """
    force = np.asarray(force, dtype=float)
    point = np.asarray(point, dtype=float)
    anchor = camera.project(point)[:2]
    projected = camera.project(point + force)[:2] - anchor
    length = float(np.linalg.norm(projected))
    if length < 1e-10:
        raise ValueError('Force direction cannot be shown in this camera')
    vector = projected / length * np.linalg.norm(force) * FORCE_PIXELS_PER_MG * camera.size / 1100
    return (anchor - vector, anchor) if tip_at_point else (anchor, anchor + vector)


def separated_arrow(camera, X, force, *, upstream):
    """Slide a force glyph along its line, leaving only dashed lines at X."""
    start, end = force_arrow(camera, X, force, tip_at_point=upstream)
    direction = (end - start) / np.linalg.norm(end - start)
    shift = (-1 if upstream else 1) * ARROW_GAP_FROM_X * camera.size / 1100 * direction
    return start + shift, end + shift


def panel(domain, item, camera, all_floor_points, number):
    scene, _, _ = S.render(domain, cam=camera, ground_points=all_floor_points)
    picture = Image.new('RGB', (scene.width, scene.height + DIAGRAM_TOP_PADDING), 'white')
    picture.paste(scene, (0, DIAGRAM_TOP_PADDING))
    draw = ImageDraw.Draw(picture)
    offset = np.array([0., DIAGRAM_TOP_PADDING])
    extent = float(domain.mesh.extents.max())
    q, F, X, W, p = (item[key] for key in ('q', 'force', 'X', 'W', 'p'))
    project = lambda xyz: camera.project(xyz)[:2] + offset
    c2, q2, x2, p2 = map(project, (domain.com, q, X, p))
    # Place the push downstream of X: its backward extension meets gravity
    # at X, while the surface application point q remains unchanged.
    gravity_top = np.r_[domain.com[:2], X[2] + .055 * extent]
    gravity_start, gravity_end = force_arrow(camera, domain.com, WEIGHT)
    gravity_start, gravity_end = gravity_start + offset, gravity_end + offset
    dashed(draw, project(gravity_top), gravity_end, S.BLUE, width=3)
    force_start, force_end = separated_arrow(camera, X, F, upstream=False)
    force_start, force_end = force_start + offset, force_end + offset
    # Keep the surface marker outside the solid glyph, including the short
    # 0.1mg arrow, so both its shaft and head remain visible.
    force_direction = (force_end - force_start) / np.linalg.norm(force_end - force_start)
    clearance = max(0., np.dot(q2 - force_start, force_direction) + 32.)
    force_start += clearance * force_direction
    force_end += clearance * force_direction
    dashed(draw, x2, force_start, S.RED, width=3)
    dashed(draw, x2, p2, S.ORANGE, width=4, dash=13, gap=10)
    # All force glyphs share one screen-space magnitude scale; construction
    # lines and the positions of q, c, X and p retain their exact 3-D projection.
    resultant_start, resultant_end = separated_arrow(camera, X, W, upstream=False)
    resultant_start, resultant_end = resultant_start + offset, resultant_end + offset
    S.arrow(draw, resultant_start, resultant_end, S.ORANGE, width=6, head=23)
    force_pixels = float(np.linalg.norm(force_end - force_start))
    S.arrow(draw, force_start, force_end, S.RED,
            width=max(3, min(7, int(.15 * force_pixels))), head=min(23, .50 * force_pixels))
    S.arrow(draw, gravity_start, gravity_end, S.BLUE, width=7, head=23)
    # q is a separate surface mark, well away from the displaced solid arrow.
    for xy, color, radius in ((c2, S.BLUE, 5), (q2, S.RED, 4), (x2, BLACK, 7), (p2, S.ORANGE, 9)):
        draw.ellipse((*tuple(xy - radius), *tuple(xy + radius)), fill=color,
                     outline='white', width=2)
    side = 1 if q2[0] >= x2[0] else -1
    S.text(draw, x2 + [-side * 20, -15], 'X', 33, BLACK, 'rm' if side > 0 else 'lm')
    S.text(draw, .45 * resultant_start + .55 * resultant_end + [-side * 22, 0], 'W', 29,
           S.ORANGE, 'rm' if side > 0 else 'lm')
    S.text(draw, p2 + [19, 10], 'p', 36, S.ORANGE, 'lm')
    S.text(draw, (gravity_start + gravity_end) / 2 + [-22, 0], 'mg', 29, S.BLUE, 'rm')
    S.text(draw, force_start + [side * 20, -12], 'F_push', 29, S.RED,
           'lm' if side > 0 else 'rm')
    S.text(draw, (55, 45), f'{number:02d}    |F_push| = {np.linalg.norm(F):g} mg',
           33, S.MUTED, 'lm')
    if number == 3:
        S.text(draw, (55, 90), 'Same point and direction as 02', 25, S.MUTED, 'lm')
    return picture


def main():
    domain = load_case()
    loads = examples(domain)
    floor_points = np.array([item['p'] for item in loads])
    extent = float(domain.mesh.extents.max())
    extra = np.array([item['X'] + [0, 0, .10 * extent] for item in loads])
    camera = S.camera(domain, size=1100, extra=extra, ground_points=floor_points, view=VIEW)
    pages = []
    for i, item in enumerate(loads, 1):
        image = panel(domain, item, camera, floor_points, i)
        image.save(HERE / f'row3_force_{i}.png')
        pages.append(image)
        print(json.dumps(dict(example=i, face=item['face'], cone_angle_deg=item['cone_angle_deg'],
            **{key: item[key].tolist() for key in ('q', 'force', 'X', 'W', 'p', 'moment_residual')})), flush=True)
    sheet = Image.new('RGB', (3300, pages[0].height + 130), 'white')
    draw = ImageDraw.Draw(sheet)
    S.text(draw, (55, 60), 'B / pose_2', 43, BLACK, 'lm')
    for i, image in enumerate(pages):
        sheet.paste(image, (i * 1100, 115))
    sheet.save(OUT)
    print(json.dumps(dict(sources=domain.sources, examples=3,
        claim='Three selected concurrent load pairs; complete moment equivalence checked. '
              'Not a claim that general 3-D load lines intersect or that a fixture is stable.')))
    print(OUT, flush=True)


if __name__ == '__main__':
    main()
