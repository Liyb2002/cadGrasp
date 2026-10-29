"""Two uses of one unchanged shell around a stationary working needle.

The fixture uses object coordinates, so its installation and the seated
object pose coincide. The second installation is selected jointly with the
task's axial pose; the needle position and direction remain fixed throughout.
Both windows already exist in the first frame. Only rigid transforms change;
there is no cutting, growth, replacement or deformation of the fixture.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from subdivide_task.step1.common import ROOT, sha256, write_json
from subdivide_task.step1.needle_render import sector_lines
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step3.render_independent import Renderer, BLUE, head_context
from subdivide_task.step5.video import raster
from .geometry import OperationModel, moved_mesh
from .merge_pose import build_fixed_needle_merge
from .state import Layout, transform


DATA = ROOT / 'subdivide_task/data/B/operation_videos/merge'
TASKS = ['W06', 'W08']


def mesh_digest(mesh):
    digest = hashlib.sha256()
    for array in (mesh.vertices, mesh.faces):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def specimen():
    model = OperationModel(task_ids=TASKS)
    path = DATA / 'fixture.npz'
    proof_path = DATA / 'validation.json'
    binding = dict(input_step32_report_sha256=sha256(model.folder / 'report.json'),
        fitted_outer_sha256=sha256(model.inputs[-1]), task_ids=TASKS,
        method='immutable_two_window_shell_with_stationary_needle_v3')
    if path.exists() and proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if all(proof.get(key) == value for key, value in binding.items()) and sha256(path) == proof['fixture_sha256']:
            with np.load(path) as saved:
                support = trimesh.Trimesh(saved['vertices'], saved['faces'], process=False)
                layout = Layout.read(saved)
            return model, layout, support, proof
    layout, support, selected = build_fixed_needle_merge(model)
    assert support.is_watertight and support.is_winding_consistent
    face_sets = [set(record['working_face_ids']) for record in model.records]
    assert not face_sets[0] & face_sets[1], 'The two working windows must use separate source regions'
    np.savez_compressed(path, vertices=support.vertices, faces=support.faces, **layout.arrays())
    proof = dict(**binding, fixture_sha256=sha256(path), passed=True,
        support_representation='wrapped_shell', no_base_plate=True, no_vertical_backing=True,
        disjoint_original_working_regions=True, window_count=2,
        **selected, single_object_seat=True, support_geometry_constant_entire_video=True,
        support_modified_during_video=False, fixture_mesh_sha256=mesh_digest(support),
        support_volume_cm3=float(support.volume * 1e6))
    write_json(proof_path, proof)
    print('MERGE TWO WINDOWS PASS', TASKS, round(support.volume * 1e6, 3), 'cm3', flush=True)
    return model, layout, support, proof


def ease(value):
    value = float(np.clip(value, 0., 1.))
    return value * value * (3. - 2. * value)


def main():
    from .render_videos import video_size
    DATA.mkdir(parents=True, exist_ok=True)
    model, layout, support, proof = specimen()
    fixture_digest = mesh_digest(support)
    width, height = video_size(Path(__file__).parent / 'translation.mp4')
    poses = layout.poses
    rotations = Slerp([0., 1.], Rotation.from_matrix(poses[:, :3, :3]))
    center = model.source.center_mass
    centers = [transform(center, pose) for pose in poses]
    contexts = head_context(model.workspace)
    stationary_tool = G.union([G.solid(mesh) for mesh, _, _ in contexts])
    # If necessary, take the empty fixture slightly away from the stationary
    # needle while turning it. The needle itself never moves.
    for clearance in (0., .025, .05, .075, .1, .15, .2):
        turns = []
        for fraction in np.linspace(0., 1., 49):
            matrix = np.eye(4)
            matrix[:3, :3] = rotations(fraction).as_matrix()
            matrix[:3, 3] = ((1. - fraction) * centers[0] + fraction * centers[1]
                - matrix[:3, :3] @ center + clearance * np.sin(np.pi * fraction) * model.axis)
            turns.append(matrix)
        overlaps = [G.volume(stationary_tool ^ G.solid(moved_mesh(support, matrix))) for matrix in turns]
        if max(overlaps) <= model.tolerance:
            break
    else:
        raise RuntimeError('The empty fixture turn intersects the stationary tool')
    proof['empty_turn_with_stationary_tool'] = dict(checked_states=len(turns),
        maximum_tool_overlap_m3=max(overlaps), fixture_clearance_translation_m=clearance)
    write_json(DATA / 'validation.json', proof)
    lines = sector_lines(model.workspace)
    points = np.vstack([*[transform(support.convex_hull.vertices, matrix) for matrix in turns],
        *[transform(model.source.vertices, pose) for pose in poses],
        lines, *[mesh.vertices for mesh, _, _ in contexts]])
    mask = np.zeros(len(model.source.faces), bool)
    for record in model.records:
        mask[record['working_face_ids']] = True
    renderer = Renderer(width, height)
    camera = [1.1, 1.4, .95]
    video = Path(__file__).parent / 'rotational_reuse.mp4'
    pending = DATA / 'merge.pending.mp4'
    # Preserve every approved animation frame and play the sequence twice as fast.
    fps, count = 48, 288
    encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(pending)], stdin=subprocess.PIPE)
    samples, timeline = [], []
    try:
        for frame in range(count):
            body_matrix = poses[0]
            body_offset = 0.
            matrix = poses[0]
            visible = True
            phase = 'use_first_window'
            if frame < 84:
                pass
            elif frame < 114:
                phase = 'remove_object'
                body_offset = .35 * ease((frame - 84) / 29)
            elif frame < 120:
                phase = 'empty_before_turn'
                visible = False
            elif frame < 168:
                phase = 'turn_empty_fixture'
                visible = False
                matrix = turns[round(ease((frame - 120) / 47) * (len(turns) - 1))]
            elif frame < 174:
                phase = 'empty_after_turn'
                visible = False
                matrix = poses[1]
            elif frame < 204:
                phase = 'insert_object_second_pose'
                matrix = body_matrix = poses[1]
                body_offset = .35 * (1. - ease((frame - 174) / 29))
            elif frame < 216:
                phase = 'settle_second_pose'
                matrix = body_matrix = poses[1]
            else:
                phase = 'use_second_window'
                matrix = body_matrix = poses[1]
            items = [(moved_mesh(support, matrix), BLUE, None)]
            if visible:
                owner = 0 if frame < 174 else 1
                actual_body_matrix = body_matrix.copy()
                actual_body_matrix[:3, 3] += body_offset * layout.exits[owner]
                body = moved_mesh(model.source, actual_body_matrix)
                items.append((body, raster.GRAY, mask))
            items.extend(contexts)
            picture = renderer.draw(items, points, camera, lines)
            encoder.stdin.write(picture.tobytes())
            timeline.append(dict(frame=frame, time_s=frame / fps, phase=phase, object_visible=visible,
                fixture_mesh_sha256=fixture_digest, working_window_count=2,
                fixture_world_from_object=matrix.tolist(), tool_retraction_m=0.,
                object_world_from_object=actual_body_matrix.tolist() if visible else None,
                needle_root_world_m=model.head.tolist(), needle_axis_world=model.axis.tolist()))
            if frame in (0, 35, 60, 112, 120, 144, 168, 203, 240, 287):
                samples.append(picture.copy())
            if frame % 48 == 0:
                print('merge', frame + 1, '/', count, phase, flush=True)
    finally:
        renderer.close()
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Merge encoding failed')
    pending.replace(video)
    assert all(not row['object_visible'] for row in timeline if row['phase'] == 'turn_empty_fixture')
    assert all(row['tool_retraction_m'] == 0. for row in timeline)
    assert mesh_digest(support) == fixture_digest
    sheet = Image.new('RGB', (width * 5, height * 2), 'white')
    for index, picture in enumerate(samples):
        sheet.paste(picture, ((index % 5) * width, (index // 5) * height))
    sheet.save(DATA / 'review.jpg')
    write_json(DATA / 'timeline.json', timeline)
    metadata = dict(path=str(video), sha256=sha256(video), width=width, height=height, fps=fps, frames=count,
        duration_seconds=count / fps,
        text_overlays=False, support_representation='wrapped_shell', no_base_plate=True, no_vertical_backing=True,
        window_count=2, task_ids=TASKS, fixed_tool_working_direction=True,
        fixed_tool_position_entire_video=True, task_pose_optimized=True,
        task_pose_adjustments_deg=proof['task_pose_adjustments_deg'],
        fixture_turn_deg=proof['fixture_turn_deg'],
        same_fixture_mesh_in_both_uses=True, object_removed_before_fixture_turn=True,
        support_geometry_constant_entire_video=True, support_modified_during_video=False,
        fixture_mesh_sha256=fixture_digest,
        object_absent_during_entire_turn=True, object_reinserted_after_turn=True,
        working_area_color='pink', validation='subdivide_task/data/B/operation_videos/merge/validation.json',
        continuous_robot_motion_claimed=False)
    write_json(DATA / 'video.json', metadata)
    index = DATA.parent / 'videos.json'
    videos = json.loads(index.read_text())
    videos['rotational_reuse.mp4'] = metadata
    write_json(index, videos)
    print('MERGE VIDEO PASS', video, flush=True)
    return metadata


if __name__ == '__main__':
    main()
