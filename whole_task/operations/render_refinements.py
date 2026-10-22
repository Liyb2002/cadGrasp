"""Replay exit and translation demonstrations with the pose-rotation object and studio.

The original design timelines are retained. The 5,000-face object and its fitted
outer layer replace the old 1,000-face approximation. These remain geometry
demonstrations; no new force or robot-motion certificate is asserted.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import manifold3d as md
import numpy as np
import trimesh
from PIL import Image, ImageDraw

from whole_task.step1.common import ROOT, sha256, write_json
from whole_task.step3 import independent_geometry as G
from whole_task.step3.exit_geometry import whole_object_sweep
from whole_task.step3.render_independent import BLUE, Renderer
from whole_task.step5.video import raster
from .geometry import moved_mesh


HERE = Path(__file__).parent
DATA = ROOT / 'whole_task/data/B/operation_videos'
REFERENCE = ROOT / 'slides/Co-optimize/operation_demo'
POSES = ['pose_1', 'pose_2', 'pose_4', 'pose_6']
VIEW = np.array([1.1, 1.4, .95])
VIEW /= np.linalg.norm(VIEW)
SIZE = 720
GREEN = (.13, .78, .46)
RED = (.94, .29, .22)
TOLERANCE = 1e-10
SWEEP_REFERENCE = ROOT / 'slides/Co-optimize/vis_func/animate_exit_directions.py'


def load_mesh(path, vertices='vertices', faces='faces'):
    with np.load(path) as saved:
        return trimesh.Trimesh(saved[vertices], saved[faces], process=False)


def work_band(body, mask):
    """Open the fitted layer at the inherited working faces, without a tool head."""
    cells = []
    # Shared vertex offsets join neighbouring face cells along the same quad.
    # Independent face-normal extrusions leave thin triangular webs at edges.
    normals = body.vertex_normals[body.faces[mask]]
    for triangle, normal in zip(body.triangles[mask], normals):
        points = np.vstack([triangle - .001 * normal, triangle + .010 * normal])
        cells.append(md.Manifold.hull_points(points / G.SCALE))
    return G.union(cells)


def component_volumes(value):
    return sorted([float(part.volume()) * G.SCALE**3 * 1e6
                   for part in value.decompose() if float(part.volume()) * G.SCALE**3 > 1e-15],
                  reverse=True)


def connected_shell(value):
    pieces = [part for part in value.decompose() if float(part.volume()) * G.SCALE**3 > 1e-15]
    if not pieces:
        raise ValueError('No fixture material remains')
    return value ^ max(pieces, key=lambda part: float(part.volume()))


def condition_geometry(value, forbidden):
    """Recut unchanged exclusions on the serialized mesh at the same tolerance."""
    for attempt in range(5):
        result = G.mesh(value)
        actual = G.solid(result)
        overlaps = {name: G.volume(actual ^ solid) for name, solid in forbidden.items()}
        if max(overlaps.values()) <= TOLERANCE:
            return result, overlaps
        if attempt == 4:
            raise ValueError(f'Serialized fixture has unresolved exclusions: {overlaps}')
        value = actual
        for name, solid in forbidden.items():
            if overlaps[name] > TOLERANCE:
                cut = solid
                if attempt >= 2:
                    # Remove a local 10-micron Boolean remnant rather than
                    # raising the overlap threshold or offsetting the object.
                    intrusion = actual ^ solid
                    relief = intrusion.minkowski_sum(G.box([-1e-5] * 3, [1e-5] * 3))
                    cut = G.union([solid, relief])
                value = value - cut


def single_exit_schedule(record):
    """Replay the original first task's spherical arc across the single view.

    Preserve all source frame boundaries, pauses, and the 11-second duration;
    the other three tasks no longer silently cut the displayed fixture.
    """
    arc = [np.asarray(record['states'][0]['directions_fixture'][0])]
    arc.extend(np.asarray(row['directions_fixture'][0]) for row in record['states']
               if row['active_pose'] == POSES[0])
    count = len(record['states'])
    for index, row in enumerate(record['states']):
        position = index * (len(arc) - 1) / (count - 1)
        left = min(int(position), len(arc) - 2)
        fraction = position - left
        a, b = arc[left:left + 2]
        angle = np.arccos(np.clip(a @ b, -1., 1.))
        if angle < 1e-10:
            direction = a.copy()
        else:
            direction = (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(angle)
        yield dict(state=index, active_pose=POSES[0], progress=index / (count - 1),
                   directions_fixture=[(direction / np.linalg.norm(direction)).tolist()],
                   first_frame=row['first_frame'], repetitions=row['repetitions'])


def projection(points, bounds, view):
    view = np.asarray(view, float) / np.linalg.norm(view)
    right = np.cross([0., 0., 1.], view)
    right /= np.linalg.norm(right)
    basis = np.array([right, np.cross(view, right), view])
    projected = np.asarray(bounds) @ basis.T
    low, high = projected.min(0), projected.max(0)
    scale = max(high[0] - low[0], high[1] - low[1]) * 1.16
    p = np.asarray(points) @ basis.T
    center = (low + high) / 2
    return np.c_[SIZE / 2 + (p[:, 0] - center[0]) / scale * SIZE,
                 SIZE / 2 - (p[:, 1] - center[1]) / scale * SIZE]


def arrow(image, body, direction, bounds, view):
    center = body.center_mass
    pixels = projection([center, center + .08 * direction], bounds, view)
    delta = pixels[1] - pixels[0]
    norm = np.linalg.norm(delta)
    if norm < 1e-6:
        raise ValueError('Exit arrow is parallel to the camera')
    unit = delta / norm
    normal = np.array([-unit[1], unit[0]])
    middle = np.clip(pixels.mean(0), 105., SIZE - 105.)
    start, end = middle - 67 * unit, middle + 67 * unit
    neck = end - 28 * unit
    polygon = [start + 6 * normal, neck + 6 * normal, neck + 17 * normal,
               end, neck - 17 * normal, neck - 6 * normal, start - 6 * normal]
    ImageDraw.Draw(image).polygon([tuple(p) for p in polygon], fill='#FFD83D', outline='#A47E14', width=2)


class SweepOverlay:
    """Use the original sweep renderer over the pose-rotation studio image."""
    def __init__(self, body, support, length, bounds, view):
        spec = importlib.util.spec_from_file_location('original_operation_exit_sweep', SWEEP_REFERENCE)
        self.reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.reference
        spec.loader.exec_module(self.reference)
        self.wheel = self.reference.Renderer(body, support, [np.eye(4)], [POSES[0]], length,
                                              columns=1, labels=False)
        wheel = self.wheel
        wheel.fig.set_size_inches(SIZE / 100, SIZE / 100, forward=True)
        wheel.fig.patch.set_alpha(0.)
        ax = wheel.axes[0]
        ax.patch.set_alpha(0.)
        for line in list(ax.lines):
            line.remove()
        for border in wheel.borders:
            border.remove()
        # The original frame method still creates its original sweep artist.
        # The studio supplies the opaque body/fixture and the existing arrow.
        collection, arrow_patch = wheel.collection, wheel.arrow_patch

        def make_collection(*args, **kwargs):
            artist = collection(*args, **kwargs)
            if 'facecolors' in kwargs:
                artist.set_visible(False)
            return artist

        def make_arrow(*args, **kwargs):
            artist = arrow_patch(*args, **kwargs)
            artist.set_visible(False)
            return artist

        wheel.collection, wheel.arrow_patch = make_collection, make_arrow
        wheel.fig.canvas.draw()
        inverse = ax.transData.inverted()
        origin = inverse.transform([SIZE / 2, SIZE / 2])
        unit = inverse.transform([SIZE / 2 + 1, SIZE / 2 + 1]) - origin
        view = np.asarray(view, float) / np.linalg.norm(view)
        right = np.cross([0., 0., 1.], view)
        right /= np.linalg.norm(right)
        basis = np.array([right, np.cross(view, right), view])
        projected = np.asarray(bounds) @ basis.T
        low, high = projected.min(0), projected.max(0)
        center = (low + high) / 2
        density = SIZE / (max(high[0] - low[0], high[1] - low[1]) * 1.16)
        matrix = np.zeros((4, 4))
        for i in (0, 1):
            matrix[i, :3] = basis[i] * density * unit[i] / 1000
            matrix[i, 3] = origin[i] - center[i] * density * unit[i]
        matrix[2, :3], matrix[2, 3] = -view / 1000, center[2] - 1
        matrix[3, 3] = 1
        ax.get_proj = lambda: matrix
        # Check that both renderers project the same physical vertices.
        vertices = np.asarray(bounds)[:20]
        q = np.c_[vertices * 1000, np.ones(len(vertices))] @ matrix.T
        pixels = ax.transData.transform(q[:, :2] / q[:, 3:4])
        pixels[:, 1] = SIZE - pixels[:, 1]
        np.testing.assert_allclose(pixels, projection(vertices, bounds, view), atol=1e-7, rtol=0)

    def draw(self, image, support, sweep, direction):
        self.wheel.frame(support, np.asarray([direction]), -1, [self.reference.S.solid(sweep)])
        artist = self.wheel.sweep_artists[0]
        assert artist.get_alpha() == .084
        assert len(artist.get_edgecolor()) == 0
        overlay = Image.fromarray(np.asarray(self.wheel.fig.canvas.buffer_rgba()).copy())
        return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')

    def close(self):
        self.wheel.plt.close(self.wheel.fig)


class Scene:
    def __init__(self):
        archive = ROOT / 'whole_task/data/B/step3_connected_output_legacy/step3.2'
        self.source_path = archive / 'initialization.npz'
        self.source = load_mesh(self.source_path, 'source_vertices_object', 'source_faces')
        self.outer_path = ROOT / 'whole_task/data/B/step3_mold_plate/fitted_outer.npz'
        self.outer = load_mesh(self.outer_path)
        self.record_path = REFERENCE / 'direction/vis/source_animation_record.json'
        self.record = json.loads(self.record_path.read_text())
        self.length = self.record['full_exit_length_m']
        coarse_path = ROOT / 'slides/Co-optimize/data/results/B/pose1+2+4+6/step3/step3.1/registered_object.obj'
        coarse = trimesh.load(coarse_path, force='mesh', process=False)
        _, distances, parents = trimesh.proximity.closest_point(coarse, self.source.triangles_center)
        self.masks, self.transforms, self.inputs = [], [], [self.source_path, self.outer_path, self.record_path, coarse_path]
        for pose in POSES:
            path = ROOT / 'objects/B/poses' / pose / 'needs.json'
            needs = json.loads(path.read_text())
            np.testing.assert_array_equal(needs['geometry']['faces'], coarse.faces)
            mask = np.zeros(len(coarse.faces), bool)
            mask[needs['geometry']['work_face_ids']] = True
            self.masks.append(mask[parents])
            self.transforms.append(np.asarray(needs['frame']['T_world_mesh']))
            self.inputs.append(path)
        self.mask_mapping = dict(method='nearest original triangle to each full-resolution triangle centroid',
                                 original_face_count=len(coarse.faces), replacement_face_count=len(self.source.faces),
                                 maximum_projection_distance_m=float(distances.max()),
                                 source_working_face_counts=[int(mask.sum()) for mask in self.masks])
        self.body = G.solid(self.source)
        self.layer = G.solid(self.outer) - self.body
        self.bands = [work_band(self.source, mask) for mask in self.masks]
        self.stock = self.layer - G.union(self.bands)
        self.initial_directions = np.asarray(self.record['initial_directions_fixture'])
        self.initial_sweeps = [whole_object_sweep(self.source, d, self.length)[0]
                               for d in self.initial_directions]
        self.initial = self.stock - G.union(self.initial_sweeps)

    def bindings(self):
        return {str(path.relative_to(ROOT)): sha256(path) for path in self.inputs}


def encode(filename, frames, fps, folder):
    pending = folder / filename
    process = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{SIZE}x{SIZE}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(pending)],
        stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError('Video encoding failed')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(pending), '-f', 'null', '-'], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate:format=duration',
        '-of', 'json', str(pending)]))
    stream = probe['streams'][0]
    assert int(stream['nb_read_frames']) == len(frames)
    assert abs(float(probe['format']['duration']) - len(frames) / fps) < 1e-6
    pending.replace(HERE / filename)
    # Review the actual encoded frames, including the changing geometry and final view.
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(HERE / filename),
        '-vf', f'fps={8 / (len(frames) / fps)},scale=360:360,tile=4x2',
        '-frames:v', '1', str(folder / 'encoded_review.jpg')], check=True)
    return dict(video_sha256=sha256(HERE / filename), width=SIZE, height=SIZE,
                frames=len(frames), fps=fps, duration_seconds=len(frames) / fps,
                full_decode_passed=True, decoded_frame_count=int(stream['nb_read_frames']))


def publish(filename, details, folder):
    details.update(source_sha256=sha256(Path(__file__)), working_area_color='pink',
                   studio_renderer='same Renderer, body material, and blue fixture as pose_rotation',
                   tool_head_visible=False, text_overlays=False, panel_count=1,
                   no_base_plate=True, no_vertical_backing=True,
                   demo_only=True, new_force_certification_claimed=False,
                   continuous_robot_motion_claimed=False)
    write_json(folder / 'video.json', details)
    manifest = DATA / 'videos.json'
    videos = json.loads(manifest.read_text())
    videos[filename] = details
    write_json(manifest, videos)


def exit_redirection(scene, preview=False):
    folder = DATA / 'exit_redirection'
    folder.mkdir(parents=True, exist_ok=True)
    pose = scene.transforms[0]
    body = moved_mesh(scene.source, pose)
    schedule = list(single_exit_schedule(scene.record))
    # The original video frames the object closely and clips the distant sweep.
    outer_world = moved_mesh(scene.outer, pose)
    bounds = np.vstack([body.vertices, outer_world.vertices])
    bindings = scene.bindings()
    record_path = folder / 'states.json'
    saved = json.loads(record_path.read_text()) if record_path.exists() else {}
    cached = {row['state']: row for row in saved.get('states', [])} if saved.get('inputs_sha256') == bindings else {}
    reused = 0
    stock = scene.layer - scene.bands[0]
    states, frames = [], []
    renderer = Renderer(SIZE, SIZE)
    overlay = SweepOverlay(body, outer_world, scene.length, bounds, VIEW)
    try:
        for index, row in enumerate(schedule):
            path = folder / f'{index:03}.npz'
            directions = np.asarray(row['directions_fixture'])
            sweep = whole_object_sweep(scene.source, directions[0], scene.length)[0]
            old = cached.get(index, {})
            use_cache = (path.exists() and old.get('mesh_sha256') == sha256(path)
                         and all(old.get(key) == value for key, value in row.items()))
            if use_cache:
                support = load_mesh(path)
                reused += 1
            else:
                value = connected_shell(stock - sweep)
                support, overlaps = condition_geometry(value, dict(body=scene.body, exit=sweep))
                actual = G.solid(support)
            support_world = moved_mesh(support, pose)
            image = renderer.draw([(support_world, BLUE, None), (body, raster.GRAY, scene.masks[0])], bounds, VIEW)
            sweep_world = moved_mesh(G.mesh(sweep), pose)
            direction = pose[:3, :3] @ directions[0]
            image = overlay.draw(image, support_world, sweep_world, direction)
            arrow(image, body, direction, bounds, VIEW)
            if preview:
                image.save(folder / 'preview.png')
                return
            if use_cache:
                states.append(old)
            else:
                np.savez_compressed(path, vertices=support.vertices, faces=support.faces,
                                    directions_fixture=directions)
                states.append({**row, 'remaining_volume_cm3': float(support.volume * 1e6),
                               'body_overlap_m3': overlaps['body'], 'full_sweep_overlap_m3': overlaps['exit'],
                               'positive_component_volumes_cm3': component_volumes(actual),
                               'mesh_sha256': sha256(path)})
            assert len(frames) == row['first_frame']
            frames.extend([image] * row['repetitions'])
            if index % 8 == 0:
                print('EXIT STATE', index, '/', len(scene.record['states']) - 1, flush=True)
    finally:
        overlay.close()
        renderer.close()
    assert len(frames) == scene.record['frame_count']
    original_timeline = [{key: row[key] for key in ('state', 'first_frame', 'repetitions')}
                         for row in scene.record['states']]
    new_timeline = [{key: row[key] for key in original_timeline[0]} for row in states]
    assert new_timeline == original_timeline
    frames[0].save(folder / 'first.png')
    frames[-1].save(folder / 'last.png')
    if reused != len(states):
        write_json(record_path, dict(states=states, inputs_sha256=bindings,
            working_face_mapping=scene.mask_mapping, original_frame_timing_preserved=True,
            original_first_task_direction_arc_preserved=True,
            displayed_pose=POSES[0], single_task_fixture_rebuilt_from_current_sweep=True))
    details = encode('exit_redirection.mp4', frames, scene.record['fps'], folder)
    publish('exit_redirection.mp4', dict(**details, original_frame_timing_preserved=True,
        original_first_task_direction_arc_preserved=True,
        source_video=str(REFERENCE / 'direction/vis/pose1+2+4+6_exit_direction_changes.mp4'),
        original_source_record_sha256=sha256(scene.record_path),
        original_object_replaced=True, object_face_count=len(scene.source.faces),
        displayed_pose=POSES[0], fixture_recomputed_for_each_direction=True,
        original_direction_state_count=len(states), inputs_sha256=bindings,
        geometry_states_reused=reused, swept_volume_visible=True, swept_volume_opacity=.084,
        sweep_full_length_m=scene.length, sweep_color='#6ebad8', sweep_edgecolors='none',
        discrete_object_ghosts=False, original_sweep_renderer_reused=True,
        sweep_style_source=str(SWEEP_REFERENCE.relative_to(ROOT)), sweep_style_source_sha256=sha256(SWEEP_REFERENCE),
        sweep_overlays_object=True, object_centered_closeup=True), folder)
    print('EXIT SAVED', HERE / 'exit_redirection.mp4', flush=True)


def camera(elevation, azimuth):
    e, a = np.radians([elevation, azimuth])
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def frame_bounds(center, half_span, view):
    """Give the studio a fixed square framing without overbounding with a cube."""
    right = np.cross([0., 0., 1.], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    return np.array([center + half_span * (sx * right + sy * up)
                     for sx in (-1, 1) for sy in (-1, 1)])


def translation(scene, preview=False):
    folder = DATA / 'translation'
    folder.mkdir(parents=True, exist_ok=True)
    record_path = REFERENCE / 'translation/vis/animation.json'
    geometry_path = REFERENCE / 'combined/vis/data/geometry.json'
    record = json.loads(record_path.read_text())
    geometry = json.loads(geometry_path.read_text())
    host_pose, guest_pose = scene.transforms[:2]
    host = moved_mesh(scene.source, host_pose)
    guest_initial = moved_mesh(scene.source, guest_pose)
    baseline_mesh = moved_mesh(G.mesh(scene.initial), host_pose)
    baseline = G.solid(baseline_mesh)
    host_body = G.solid(host)
    guest_body = G.solid(guest_initial)
    guest_wrap = G.solid(moved_mesh(G.mesh(scene.layer - scene.bands[1]), guest_pose))
    host_work = G.solid(moved_mesh(G.mesh(G.union([scene.bands[i] for i in (0, 2, 3)])), host_pose))
    guest_work = G.solid(moved_mesh(G.mesh(scene.bands[1]), guest_pose))
    fixed_exits = [G.solid(moved_mesh(G.mesh(scene.initial_sweeps[i]), host_pose)) for i in (0, 2, 3)]
    direction = np.asarray(geometry['direction_initial_world'])
    guest_exit = whole_object_sweep(guest_initial, direction, scene.length)[0]
    fixed = G.union([host_body, host_work, *fixed_exits])
    states = {}
    previous = None
    selected = ['translation_0'] if preview else [f'translation_{i}' for i in range(49)]
    rows = []
    for name in selected:
        original = geometry['states'][name]
        offset = np.asarray(original['translation_world_m'])
        shift = offset / G.SCALE
        body = guest_body.translate(shift)
        sweep = guest_exit.translate(shift)
        work = guest_work.translate(shift)
        value = connected_shell(baseline + guest_wrap.translate(shift) - G.union([fixed, body, sweep, work]))
        support, overlaps = condition_geometry(value, dict(body=body, host_body=host_body,
            exit=sweep, work=work, host_work=host_work, **{f'fixed_exit_{i}': solid for i, solid in enumerate(fixed_exits)}))
        actual = G.solid(support)
        value = actual
        current = moved_mesh(guest_initial, np.block([[np.eye(3), offset[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
        kept, added = G.mesh(value ^ baseline), G.mesh(value - baseline)
        fresh = G.mesh(((baseline if previous is None else previous) ^ baseline) - value)
        states[name] = dict(final=support, kept=kept, added=added, fresh=fresh, guest=current)
        path = folder / f'{name}.npz'
        if not preview:
            np.savez_compressed(path, **{f'{kind}_{key}': getattr(mesh, key)
                for kind, mesh in states[name].items() for key in ('vertices', 'faces')})
            rows.append(dict(state=name, translation_world_m=offset.tolist(),
                overlap_m3=overlaps, positive_component_volumes_cm3=component_volumes(actual), mesh_sha256=sha256(path)))
        previous = value
        if int(name.split('_')[-1]) % 8 == 0:
            print('TRANSLATION STATE', name, flush=True)
    initial_view = camera(24., -55.)
    points = np.vstack([states[name][kind].vertices for name in states for kind in ('final', 'guest')] + [host.vertices])
    renderer = Renderer(SIZE, SIZE)
    frames = []
    last = states[selected[-1]]
    final_points = last['final'].vertices
    center = last['final'].bounds.mean(0)
    spans = []
    for angle in np.linspace(-45., 315., 73):
        view = camera(35.264389682754654, angle)
        right = np.cross([0., 0., 1.], view)
        right /= np.linalg.norm(right)
        up = np.cross(view, right)
        spans.append(np.abs((final_points - center) @ np.array([right, up]).T).max())
    orbit_span = max(spans)
    initial_center = (points.min(0) + points.max(0)) / 2
    right = np.cross([0., 0., 1.], initial_view)
    right /= np.linalg.norm(right)
    initial_span = np.abs((points - initial_center) @ np.array([right, np.cross(initial_view, right)]).T).max()
    withdrawal_checks = []
    try:
        timeline = record['timeline'][:1] if preview else record['timeline']
        for index, item in enumerate(timeline):
            name = item['state'] or selected[-1]
            state = states[name]
            phase = item['phase']
            view = initial_view
            bounds = points
            if phase == 'isometric_transition':
                alpha = (index - next(i for i, r in enumerate(timeline) if r['phase'] == phase) + 1) / 6
                alpha = alpha * alpha * (3 - 2 * alpha)
                elevation = 24 * (1 - alpha) + 35.264389682754654 * alpha
                view = camera(elevation, item['camera_azimuth_degrees'])
                bounds = frame_bounds(initial_center * (1 - alpha) + center * alpha,
                                      initial_span * (1 - alpha) + orbit_span * alpha, view)
            elif phase == 'support_orbit':
                view = camera(35.264389682754654, item['camera_azimuth_degrees'])
                bounds = frame_bounds(center, orbit_span, view)
            if phase == 'translation':
                materials = [(state['kept'], BLUE, None), (state['added'], GREEN, None), (state['fresh'], RED, None)]
            elif phase == 'translated_wrap':
                materials = [(state['kept'], BLUE, None), (state['added'], GREEN, None)]
            else:
                materials = [(state['final'], BLUE, None)]
            if item['guest_visible']:
                body = state['guest'].copy()
                body.apply_translation(item['guest_exit_offset_m'] * direction)
                materials.append((body, raster.GRAY, scene.masks[1]))
                if phase == 'object_exit':
                    overlap = G.volume(G.solid(state['final']) ^ G.solid(body))
                    if overlap > TOLERANCE:
                        raise ValueError(f'Playback withdrawal intersects fixture: {overlap}')
                    withdrawal_checks.append(dict(first_frame=item['first_frame'],
                        offset_m=item['guest_exit_offset_m'], body_overlap_m3=overlap))
            image = renderer.draw(materials, bounds, view)
            if preview:
                image.save(folder / 'preview.png')
                return
            assert len(frames) == item['first_frame']
            frames.extend([image] * item['repetitions'])
    finally:
        renderer.close()
    assert len(frames) == record['frame_count']
    frames[0].save(folder / 'first.png')
    frames[-1].save(folder / 'last.png')
    write_json(folder / 'states.json', dict(states=rows, timeline=record['timeline'],
        inputs_sha256={**scene.bindings(), str(record_path.relative_to(ROOT)): sha256(record_path),
                       str(geometry_path.relative_to(ROOT)): sha256(geometry_path)},
        original_translation_and_withdrawal_timeline_preserved=True,
        working_face_mapping=scene.mask_mapping, object_withdrawal_checks=withdrawal_checks))
    details = encode('translation.mp4', frames, record['fps'], folder)
    publish('translation.mp4', dict(**details, original_motion_and_material_sequence_preserved=True,
        source_video=str(REFERENCE / 'translation/vis/pose1+2+4+6_translation.mp4'),
        original_timeline_sha256=sha256(record_path), original_object_replaced=True,
        object_face_count=len(scene.source.faces), translation_distance_m=geometry['translation_distance_m'],
        object_removed_before_camera_orbit=True, camera_orbit_degrees=360.), folder)
    print('TRANSLATION SAVED', HERE / 'translation.mp4', flush=True)


def main(only=None, preview=False):
    scene = Scene()
    if only in (None, 'exit_redirection'):
        exit_redirection(scene, preview)
    if only in (None, 'translation'):
        translation(scene, preview)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['exit_redirection', 'translation'])
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    main(args.only, args.preview)
