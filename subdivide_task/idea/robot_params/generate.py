"""Seven robot-parameter illustrations at the saved first Step1 object pose.

Run from the repository root with .venv/bin/python -B followed by this path.
Only this directory is written. No task partition or downstream stage is run.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robots_config.constraints import load_config
from subdivide_task.step1.common import ROOT, load_shape, sha256, write_json
from subdivide_task.step1 import plan_reach as P, reach as R, arm as A
from subdivide_task.step1.find_tasks_figure import (
    VIEW, GOLD, Opacity, cone, dashed, tube, tubes, tabletop, task_outline,
    WORKSPACE_COLOR,
)
from subdivide_task.step1.reach_video import Scene, tool_frame
from subdivide_task.step1.workspace import box_edges, points_inside_box

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'subdivide_task/data/B/step1_reach'
ROBOT = ROOT / 'robots_config/04_kuka_lbr_med14_r820.json'
CASES = [
    ('standard', 25., 45., 100., None),
    ('angle_0', 0., 45., 100., 'angle'),
    ('angle_60', 60., 45., 100., 'angle'),
    ('translation_10', 25., 45., 10., 'translation'),
    ('translation_infinite', 25., 45., None, 'translation'),
    ('direction_20', 25., 20., 100., 'direction'),
    ('direction_10', 25., 10., 100., 'direction'),
]
WORKERS = 8
PICTURE = (1120, 1160)
LABEL_HEIGHT = 174
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def unit_direction(degrees):
    theta = np.radians(degrees)
    return np.array([0., -np.cos(theta), -np.sin(theta)])


def fixed_pose(report):
    record = report['areas'][0]
    if record['id'] != '+X':
        raise ValueError('Expected the saved first Step1 pose, +X')
    W = np.eye(4)
    W[:3, :3] = record['placement']['rotation_world_object']
    W[:3, 3] = record['placement']['translation_world_m']
    return W


def certify_one_direction(mesh, W, setup, axes, screen, down, beta):
    """The Step1 needle/head and arm checks, at one fixed pose and without a box."""
    nominal = W[:3, :3].T @ unit_direction(down)
    cosine = axes @ nominal
    in_cone = cosine >= np.cos(np.radians(beta))-1e-12
    jobs = []
    for f in range(len(mesh.faces)):
        ids = np.flatnonzero(in_cone & screen[:, f])
        ids = ids[np.argsort(-cosine[ids], kind='stable')]
        if len(ids):
            jobs.append((f, [(0, int(k), axes[k]) for k in ids],
                         setup['needle_length_m'], setup['head_radius_m'],
                         setup['needle_radius_m'], .005*setup['needle_length_m'],
                         setup['max_incidence_deg']))
    witness = np.zeros((len(mesh.faces), 3))
    gaps = np.zeros((len(mesh.faces), 2))
    tilt = np.full(len(mesh.faces), np.inf)
    joints = np.zeros((len(mesh.faces), 4, 7))
    code = np.full(len(mesh.faces), -1, dtype=np.int8)
    print('CERTIFY', down, 'deg, cone', beta, 'deg:', len(jobs), 'screened faces', flush=True)
    # Spatial filtering is applied below to each requested cube. Removing this
    # field here permits reusing the same exact witnesses for nested windows.
    local_setup = deepcopy(setup)
    local_setup.pop('surface_workspace', None)
    local_setup.update(working_direction_world=unit_direction(down).tolist(), max_tilt_deg=beta)
    with ProcessPoolExecutor(WORKERS, initializer=P._init_mesh,
                             initargs=(str(ROOT/'objects/B/mesh.stl'), local_setup)) as pool:
        for f, _, k, needle, head, _ in pool.map(P._certify, jobs, chunksize=12):
            if k is not None:
                witness[f] = axes[k]
                tilt[f] = np.degrees(np.arccos(np.clip(cosine[k], -1., 1.)))
                gaps[f] = needle, head
        faces = np.flatnonzero(np.isfinite(tilt))
        faces = faces[A.order(mesh.triangles_center[faces] @ W[:3, :3].T)]
        chunks = [(0, W, [(int(f), witness[f]) for f in faces[k:k+60]])
                  for k in range(0, len(faces), 60)]
        for rows in pool.map(P._arm_pairs, chunks):
            for f, _, result, Q in rows:
                code[f] = result
                joints[f] = Q
    tilt[code != A.OK] = np.inf
    print('CERTIFIED', down, 'deg:', int(np.isfinite(tilt).sum()), 'faces', flush=True)
    return dict(tilt_deg=tilt, axis_object=witness, gaps=gaps, joints=joints, arm_code=code)


def prepare(recompute=False):
    report = json.loads((SOURCE/'plan.json').read_text())
    robot = load_config(ROBOT)
    mesh = load_shape(ROOT/'objects/B/mesh.stl')
    W = fixed_pose(report)
    setup = deepcopy(report['setup'])
    assert robot['task_constraint']['cone_half_angle_deg'] == 25
    np.testing.assert_allclose(unit_direction(45.), robot['task_constraint']['working_direction_world'])
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in [
        SOURCE/'plan.json', SOURCE/'plan.npz', SOURCE/'screen.npz', ROBOT,
        ROOT/'objects/B/mesh.stl', ROOT/'subdivide_task/step1/reach.py',
        ROOT/'subdivide_task/step1/arm.py', ROOT/'subdivide_task/step1/plan_reach.py']}
    cache = HERE/'certified.npz'
    meta = HERE/'certification.json'
    cached = {}
    directions = ((45., 60.), (20., 25.), (10., 25.))
    evidence_keys = ('tilt_deg', 'axis_object', 'gaps', 'joints', 'arm_code')
    if not recompute and cache.exists() and meta.exists():
        old = json.loads(meta.read_text())
        if old['input_sha256'] == source_hashes and sha256(cache) == old['arrays_sha256']:
            with np.load(cache) as z:
                cached = {k: z[k] for k in z.files}
            if all(f'd{int(d)}_{key}' in cached for d, _ in directions for key in evidence_keys):
                return mesh, W, setup, cached, old
    with np.load(SOURCE/'screen.npz') as z:
        axes, screened = z['axes'].copy(), z['reach'].copy()
        screen_key = json.loads(str(z['key']))
    assert screen_key['shape'] == sha256(ROOT/'objects/B/mesh.stl')
    assert screen_key['reach'] == sha256(ROOT/'subdivide_task/step1/reach.py')
    assert screen_key['length'] == setup['needle_length_m']
    assert screen_key['radius'] == setup['head_radius_m']
    assert screen_key['needle_radius'] == setup['needle_radius_m']
    # The exact nominal vectors must be present, particularly for a 0° cone.
    policy = json.loads((ROOT/'subdivide_task/step1/reach_policy.json').read_text())
    pixel = policy['screen_pixel_fraction_of_diagonal'] * np.linalg.norm(mesh.extents)
    count = int(policy['screen_samples_per_pixel']*mesh.area/pixel**2)
    screening = R.Screen(mesh, setup['needle_length_m'], setup['head_radius_m'],
                         setup['needle_radius_m'], pixel, count, setup['max_incidence_deg'])
    extra = np.array([W[:3, :3].T @ unit_direction(d) for d, _ in directions])
    screened = np.concatenate([screened, np.array([screening(a) for a in extra])])
    axes = np.concatenate([axes, extra])
    arrays = {}
    for down, beta in directions:
        if all(f'd{int(down)}_{key}' in cached for key in evidence_keys):
            evidence = {key: cached[f'd{int(down)}_{key}'] for key in evidence_keys}
            print('REUSE', down, 'deg', flush=True)
        else:
            evidence = certify_one_direction(mesh, W, setup, axes, screened, down, beta)
        arrays.update({f'd{int(down)}_{key}': value for key, value in evidence.items()})
    nominal = np.isfinite(arrays['d45_tilt_deg']) & (arrays['d45_tilt_deg'] < 1e-6)
    centers = mesh.triangles_center @ W[:3, :3].T+W[:3, 3]
    extent_from_center = np.abs(mesh.triangles-mesh.triangles_center[:, None]).max(axis=(1, 2))
    candidates = np.flatnonzero(nominal & (extent_from_center < .0045))
    old_center = np.array(setup['surface_workspace']['workspace_center_world_m'])
    anchor_face = int(candidates[np.argmin(np.linalg.norm(centers[candidates]-old_center, axis=1))])
    anchor = centers[anchor_face]
    arrays['anchor_world_m'] = anchor
    arrays['anchor_face_id'] = np.array(anchor_face)
    np.savez_compressed(cache, **arrays)
    metadata = dict(input_sha256=source_hashes, arrays_sha256=sha256(cache),
                    pose_index=0, pose_id='+X', world_from_object=W.tolist(),
                    workspace_center_world_m=anchor.tolist(), anchor_face_id=anchor_face,
                    workspace_center_rule='one certified nominal surface contact, fixed for all seven setups',
                    translation_definition='cube edge length; equal XYZ spans, centered at the fixed surface contact',
                    direction_definition='degrees below the horizontal toward world -Y',
                    verification='Step1 exact needle/head, four-point IK and collision checks; no downstream stages',
                    unbounded_direction_checks=[int(d) for d, _ in directions])
    write_json(meta, metadata)
    return mesh, W, setup, arrays, metadata


def case_data(mesh, W, setup, arrays):
    Pworld = R.test_points(mesh) @ W[:3, :3].T+W[:3, 3]
    vertices_world = mesh.vertices @ W[:3, :3].T+W[:3, 3]
    object_bounds = np.array([vertices_world.min(axis=0), vertices_world.max(axis=0)])
    center = arrays['anchor_world_m']
    outputs = []
    for name, beta, down, edge_mm, changed in CASES:
        bounds = None if edge_mm is None else np.array([center-edge_mm/2000., center+edge_mm/2000.])
        key = f'd{int(down)}_'
        mask = np.isfinite(arrays[key+'tilt_deg']) & (arrays[key+'tilt_deg'] <= beta+1e-6)
        if bounds is not None:
            mask &= points_inside_box(Pworld, bounds).all(axis=1)
            # Check the actual FK points, as in the Step1 disk replay.
            from subdivide_task.step1.reach_video import kuka
            arm = kuka(setup)
            for face in np.flatnonzero(mask):
                actual = np.array([arm.forward(q)[0] for q in arrays[key+'joints'][face]])
                if not points_inside_box(actual, bounds).all():
                    mask[face] = False
        if not mask.any():
            raise ValueError(name+': no certified surface in the selected window')
        ids = np.flatnonzero(mask)
        near_nominal = ids[arrays[key+'tilt_deg'][ids] < 1e-6]
        choose = near_nominal if len(near_nominal) else ids
        centers = Pworld[choose, 3]
        face = int(choose[np.argmin(np.linalg.norm(centers-center, axis=1))])
        axis = W[:3, :3] @ arrays[key+'axis_object'][face]
        tip = Pworld[face, 3]
        config = deepcopy(setup)
        config.update(max_tilt_deg=beta, working_direction_world=unit_direction(down).tolist())
        config.pop('surface_workspace', None)
        if bounds is not None:
            config['surface_workspace'] = dict(bounds_world_m=bounds.tolist())
        checker = A.Check(mesh, config)
        if not checker.replay(face, arrays[key+'axis_object'][face], W, arrays[key+'joints'][face]):
            raise ValueError(name+': illustrated tool contact failed replay')
        outputs.append(dict(name=name, permitted_angle_deg=beta, direction_down_deg=down,
                            permitted_translation_mm=edge_mm, changed_parameter=changed,
                            cube_bounds_world_m=None if bounds is None else bounds.tolist(),
                            display_bounds_world_m=(object_bounds if bounds is None else bounds).tolist(),
                            display_box_type='object_AABB' if bounds is None else 'translation_cube',
                            mask=mask, tip=tip, axis=axis, contact_face_id=face,
                            certified_face_count=int(mask.sum()),
                            certified_area_fraction=float(mesh.area_faces[mask].sum()/mesh.area),
                            setup=config))
    masks = {c['name']: c['mask'] for c in outputs}
    for smaller, larger in [('angle_0','standard'), ('standard','angle_60'),
                            ('translation_10','standard'), ('standard','translation_infinite')]:
        assert not np.any(masks[smaller] & ~masks[larger]), (smaller, larger)
    return outputs


def arrow(start, axis, length):
    u, v = R.frame(axis)
    end = start+length*axis
    return [(start, end), (end, end-.008*axis+.004*u), (end, end-.008*axis-.004*u)]


def label(image, case):
    canvas = Image.new('RGB', (PICTURE[0], PICTURE[1]+LABEL_HEIGHT), 'white')
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 35)
    number = lambda v: f'{v:g}'
    extent = '∞' if case['permitted_translation_mm'] is None else number(case['permitted_translation_mm'])+' mm'
    texts = [('angle', 'Permitted angle = '+number(case['permitted_angle_deg'])+'°'),
             ('direction', 'Direction = '+number(case['direction_down_deg'])+'° downward'),
             ('translation', 'Permitted translation = '+extent)]
    for k, (parameter, text) in enumerate(texts):
        color = '#b62c73' if parameter == case['changed_parameter'] else '#252c33'
        x = (canvas.width-draw.textbbox((0, 0), text, font=font)[2])/2
        draw.text((x, PICTURE[1]+12+49*k), text, font=font, fill=color)
    return canvas


def render(mesh, W, setup, arrays, metadata, cases):
    shift = np.array(setup['work_point_m'])
    local_W = W.copy(); local_W[:3, 3] -= shift
    fit = [mesh.vertices @ local_W[:3, :3].T+local_W[:3, 3]]
    for case in cases:
        tip = case['tip']-shift
        root = tip-setup['needle_length_m']*case['axis']
        direction = unit_direction(case['direction_down_deg'])
        wire = cone(root, direction, setup['needle_length_m'], case['permitted_angle_deg'])
        fit.extend([np.array(wire).reshape(-1, 3), np.array([tip, tip-(setup['needle_length_m']+.07)*case['axis']])])
        head_vertices, _ = tube(root, tip-(setup['needle_length_m']+.07)*case['axis'], setup['head_radius_m'])
        fit.append(head_vertices)
        fit.append(np.array(box_edges(case['display_bounds_world_m'], shift)).reshape(-1, 3))
    fit = np.vstack(fit)
    panels = {}
    for case in cases:
        scene = Scene(mesh, case['setup'], *PICTURE, arm=False)
        scene.view = VIEW.copy()
        scene.fit(fit, 1.08)
        scene.ctx.line_width = 2.4
        r = scene.r
        colors = np.tile(r.GRAY, (len(mesh.faces), 1)); colors[case['mask']] = r.PINK
        scene.workspace_wireframe = (np.array(box_edges(case['display_bounds_world_m'], shift)).reshape(-1, 3), WORKSPACE_COLOR)
        tip = case['tip']-shift
        root = tip-setup['needle_length_m']*case['axis']
        nominal = unit_direction(case['direction_down_deg'])
        def shape(geometry, color):
            return r.Shape(scene.ctx, scene.shader, *geometry, color, metallic=.1, roughness=.5)
        wire = cone(root, nominal, setup['needle_length_m'], case['permitted_angle_deg'])
        wire = [(a, b) for a, b in wire if np.linalg.norm(b-a) > 1e-9]
        if case['permitted_angle_deg'] == 0:
            wire = [(root, root+setup['needle_length_m']*nominal)]
        cones = [(shape(tubes(dashed(wire), .0004), GOLD), np.eye(4))]
        cones += [(shape(tubes(arrow(root, nominal, .075), .0003), (.18, .36, .42)), np.eye(4))]
        tools = [(part, tool_frame(tip, case['axis'])) for part in scene.tools[False]]
        compositor = Opacity(scene)
        try:
            data = compositor.draw(local_W, colors, tabletop(scene, setup, shift)+tools,
                                   None, [], .3, cones, .62,
                                   segments=task_outline(mesh, case['mask'], local_W))
            picture = Image.frombytes('RGB', PICTURE, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        finally:
            compositor.close(); scene.close()
        panel = label(picture, case)
        panel.save(HERE/(case['name']+'.png'))
        panels[case['name']] = panel
        print('RENDERED', case['name'], case['certified_face_count'], 'faces', flush=True)
    # A centered reference at left; paired one-parameter changes in three columns.
    width, height = next(iter(panels.values())).size
    gap = 32
    composite = Image.new('RGB', (4*width+5*gap, 2*height+3*gap), 'white')
    composite.paste(panels['standard'], (gap, (composite.height-height)//2))
    columns = [('angle_0','angle_60'), ('translation_10','translation_infinite'), ('direction_20','direction_10')]
    for col, pair in enumerate(columns, 1):
        for row, name in enumerate(pair):
            composite.paste(panels[name], (gap+col*(width+gap), gap+row*(height+gap)))
    composite.save(HERE/'robot_params.png')
    records = [{k: v for k, v in c.items() if k not in ['mask','tip','axis','setup']} for c in cases]
    for record in records:
        record['image'] = record['name']+'.png'
        record['image_sha256'] = sha256(HERE/record['image'])
    write_json(HERE/'setups.json', dict(reference_config=str(ROBOT.relative_to(ROOT)),
               fixed_world_from_object=W.tolist(), fixed_workspace_center_world_m=arrays['anchor_world_m'].tolist(),
               labels_only=['Permitted angle','Direction','Permitted translation'],
               translation_definition=metadata['translation_definition'],
               direction_definition=metadata['direction_definition'],
               same_camera_and_scale=True, cases=records,
               composite='robot_params.png', composite_size=list(composite.size),
               composite_sha256=sha256(HERE/'robot_params.png'), source_sha256=sha256(__file__)))
    print('DONE', HERE/'robot_params.png', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    began = time.monotonic()
    mesh, W, setup, arrays, metadata = prepare(args.recompute)
    cases = case_data(mesh, W, setup, arrays)
    render(mesh, W, setup, arrays, metadata, cases)
    print('SECONDS', round(time.monotonic()-began, 1), flush=True)


if __name__ == '__main__':
    main()
