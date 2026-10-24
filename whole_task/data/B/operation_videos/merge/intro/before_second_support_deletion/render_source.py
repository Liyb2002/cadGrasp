"""Two independent tasks, a shared seat, then the second working window.

The existing six-second reuse segment is appended without re-encoding. The
construction prefix ends at its byte-identical accepted two-window fixture.
"""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import trimesh
import manifold3d as md
from scipy.spatial.transform import Rotation, Slerp

from whole_task.step1.common import ROOT, sha256, write_json
from whole_task.step1.find_tasks_figure import cone
from whole_task.step1.reach_video import slerp
from whole_task.step3 import independent_geometry as G
from whole_task.step3.exit_geometry import whole_object_sweep
from whole_task.step3.render_independent import Renderer, BLUE
from whole_task.step3.run_independent import make_domain
from whole_task.step3.independent_physics import demands, certificate
from whole_task.step5.video import raster
from .geometry import moved_mesh, contact_surface
from .render_merge import mesh_digest, ease
from .render_pose_rotation import VIEW
from .render_refinements import RED, TOLERANCE
from . import render_rotational_reuse as reuse


DATA = reuse.DATA / 'intro'
VIDEO = ROOT / 'whole_task/operations/rotational_reuse.mp4'
FPS = 48
PREFIX_GEOMETRY_FRAMES = 144
PREFIX_FRAMES = 288


def material_only(value):
    """Discard zero-volume Boolean boundary sheets before drawing material."""
    parts = [part for part in value.decompose() if abs(float(part.volume())) * G.SCALE**3 > 1e-15]
    return md.Manifold.compose(parts) if parts else md.Manifold()


def tail():
    """Keep the already checked working sequence and its exact encoded frames."""
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / 'reuse_tail.mp4'
    record = DATA / 'reuse_tail.json'
    if record.exists() and path.exists():
        metadata = json.loads(record.read_text())
        if (sha256(path) == metadata['video_sha256']
                and metadata['source_sha256'] == sha256(Path(reuse.__file__))
                and metadata['fixture_mesh_sha256'] == mesh_digest(load_fixture())):
            return path, metadata
    current = json.loads((reuse.DATA / 'video.json').read_text())
    if (current['frames'] != 288 or current['source_sha256'] != sha256(Path(reuse.__file__))
            or current['video_sha256'] != sha256(VIDEO)):
        reuse.main()
        current = json.loads((reuse.DATA / 'video.json').read_text())
    assert current['frames'] == 288 and current['fps'] == FPS and current['full_decode_passed']
    shutil.copy2(VIDEO, path)
    write_json(record, current)
    return path, current


def load_fixture():
    with np.load(reuse.DATA / 'fixture.npz') as saved:
        return trimesh.Trimesh(saved['vertices'], saved['faces'], process=False)


def geometry(source, fixture, proof):
    """Restore only the other task's window to form each independent shell."""
    archive = json.loads((reuse.ARCHIVE / 'report.json').read_text())
    assert sha256(reuse.ARCHIVE / 'report.json') == proof['input_step32_report_sha256']
    outer_path = ROOT / 'whole_task/data/B/step3_mold_plate/fitted_outer.npz'
    assert sha256(outer_path) == proof['fitted_outer_sha256']
    with np.load(outer_path) as saved:
        outer = trimesh.Trimesh(saved['vertices'], saved['faces'], process=False)
    workspace = dict(archive['source_workspace'],
                     reference_head_front_world_m=proof['fixed_needle_position_world_m'])
    works = []
    for matrix in proof['selected_task_poses_world']:
        matrix = np.asarray(matrix)
        inverse = np.linalg.inv(matrix)
        local = dict(workspace,
            reference_head_front_world_m=(inverse[:3, :3] @ np.asarray(workspace['reference_head_front_world_m'])
                                          + inverse[:3, 3]).tolist(),
            nominal_axis_world=(matrix[:3, :3].T @ np.asarray(workspace['nominal_axis_world'])).tolist())
        works.append(G.work_sector(local, archive['policy'])[0])
    sweep = whole_object_sweep(source, np.asarray(proof['local_extraction_direction']), .7)[0]
    body = G.solid(source)
    base = G.solid(outer) - G.union([body, sweep])
    final = G.solid(fixture)
    # The independent host keeps only window 1 open; the guest keeps only
    # window 2 open. Certify the actual serialized independent shells below.
    caps = [material_only((base - works[0]) ^ works[1]), material_only((base - works[1]) ^ works[0])]
    singles = [G.mesh(final + cap) for cap in caps]
    checks = []
    for index, mesh in enumerate(singles):
        actual = G.solid(mesh)
        values = dict(body_overlap_m3=G.volume(actual ^ body),
            own_work_overlap_m3=G.volume(actual ^ works[index]),
            continuous_sweep_overlap_m3=G.volume(actual ^ sweep),
            missing_accepted_final_material_m3=G.volume(final - actual),
            added_material_outside_other_work_region_m3=G.volume(caps[index] - works[1 - index]))
        assert max(value for name, value in values.items()
                   if name != 'missing_accepted_final_material_m3') <= TOLERANCE, values
        record = next(row for row in archive['tasks'] if row['id'] == ['W06', 'W08'][index])
        pose = np.asarray(proof['selected_task_poses_world'][index])
        domain = make_domain(source, pose, record['working_face_ids'], record['id'], workspace,
                             archive['physical_properties'], archive['policy'])
        endpoints, samples = demands(domain, archive['policy'])
        body_world = moved_mesh(source, pose)
        force, _ = certificate(body_world, contact_surface(body_world, moved_mesh(mesh, pose)),
                                domain, endpoints, samples, search=True)
        assert force['passed'], force
        values['actual_contact_force_certificate'] = force
        checks.append(values)
    cap = G.mesh(caps[0])
    assert cap.volume * 1e6 > 1., 'The second working window must initially be blocked'
    np.savez_compressed(DATA / 'geometry.npz',
        host_vertices=singles[0].vertices, host_faces=singles[0].faces,
        guest_vertices=singles[1].vertices, guest_faces=singles[1].faces,
        second_window_fill_vertices=cap.vertices, second_window_fill_faces=cap.faces,
        final_vertices=fixture.vertices, final_faces=fixture.faces)
    binding = dict(status='passed', task_ids=['W06', 'W08'], initial_task_count=2,
        initial_working_window_count_per_fixture=[1, 1], separate_fixtures=True,
        initial_fixture_volumes_cm3=[float(mesh.volume * 1e6) for mesh in singles],
        second_window_removed_material_cm3=float(cap.volume * 1e6),
        second_window_cut_is_subset_of_original_task2_work_exclusion=True,
        initial_geometry_checks=checks, support_support_overlap_allowed=True,
        merge_preserves_first_task_working_window=True, final_fixture_unchanged=True,
        final_fixture_sha256=sha256(reuse.DATA / 'fixture.npz'),
        final_fixture_mesh_sha256=mesh_digest(fixture),
        original_final_force_proof_preserved=True,
        geometry_sha256=sha256(DATA / 'geometry.npz'), source_sha256=sha256(Path(__file__)),
        inputs_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
            [reuse.DATA / 'fixture.npz', reuse.DATA / 'validation.json',
             reuse.ARCHIVE / 'report.json', reuse.ARCHIVE / 'initialization.npz', outer_path]})
    write_json(DATA / 'geometry_validation.json', binding)
    return singles, caps, works, binding


def framing(start, finish, view, fraction):
    """Blend framing while the camera turns toward the second work area."""
    view = np.asarray(view, float) / np.linalg.norm(view)
    right = np.cross([0., 0., 1.], view)
    right /= np.linalg.norm(right)
    basis = np.array([right, np.cross(view, right), view])
    a, b = np.asarray(start) @ basis.T, np.asarray(finish) @ basis.T
    low = (1 - fraction) * a.min(0) + fraction * b.min(0)
    high = (1 - fraction) * a.max(0) + fraction * b.max(0)
    corners = np.array(np.meshgrid(*zip(low, high), indexing='ij')).reshape(3, -1).T
    return corners @ basis


def render_prefix(source, fixture, timeline, reference, singles, caps, works):
    start_pose = np.asarray(timeline[0]['fixture_world_from_object'])
    second_pose = np.asarray(timeline[-1]['fixture_world_from_object'])
    center = source.center_mass
    target = center @ start_pose[:3, :3].T + start_pose[:3, 3]
    right = np.cross([0., 0., 1.], VIEW)
    right /= np.linalg.norm(right)
    separation = .135 * right
    rotations = Slerp([0., 1.], Rotation.from_matrix([second_pose[:3, :3], start_pose[:3, :3]]))

    def poses(fraction):
        host = start_pose.copy()
        host[:3, 3] -= (1 - fraction) * separation
        guest = np.eye(4)
        guest[:3, :3] = rotations(fraction).as_matrix()
        guest_center = target + (1 - fraction) * separation
        guest[:3, 3] = guest_center - guest[:3, :3] @ center
        return host, guest

    report = json.loads((reuse.ARCHIVE / 'report.json').read_text())
    masks = []
    for name in ('W06', 'W08'):
        mask = np.zeros(len(source.faces), bool)
        mask[next(row for row in report['tasks'] if row['id'] == name)['working_face_ids']] = True
        masks.append(mask)
    combined = masks[0] | masks[1]
    host, guest = poses(0.)
    initial_items = [(moved_mesh(mesh, matrix), BLUE, None) for mesh, matrix in zip(singles, (host, guest))]
    initial_bodies = [moved_mesh(source, matrix) for matrix in (host, guest)]
    # The existing working sequence keeps exactly its current close-up camera.
    tip = np.asarray(reference['tool_tip_world_m'])
    axis = np.asarray(reference['direction_axis_world'])
    contexts = reuse.tool(tip, axis, reference)
    lines = np.asarray(cone(tip - reference['needle_length_m'] * axis, axis,
                            reference['needle_length_m'], 25., 8)).reshape(-1, 3)
    tail_clouds = [lines, *[mesh.vertices for mesh, _, _ in contexts]]
    tail_clouds.extend(moved_mesh(fixture, np.asarray(row['fixture_world_from_object'])).vertices for row in timeline)
    tail_clouds.extend(moved_mesh(source, np.asarray(row['object_world_from_object'])).vertices
                       for row in (timeline[0], timeline[-1]))
    tail_bounds = np.vstack(tail_clouds)
    wide_bounds = np.vstack([*[mesh.vertices for mesh, _, _ in initial_items],
                             *[body.vertices for body in initial_bodies]])
    close_bounds = np.vstack([moved_mesh(singles[0], start_pose).vertices,
                               moved_mesh(source, start_pose).vertices])
    second_faces = np.flatnonzero(masks[1])
    cut_view = start_pose[:3, :3] @ np.average(source.face_normals[second_faces],
                                             weights=source.area_faces[second_faces], axis=0)
    cut_view /= np.linalg.norm(cut_view)
    merge_states = []
    merge_checks = []
    for index in range(49):
        fraction = index / 48
        host_pose, guest_pose = poses(fraction)
        host_mesh = moved_mesh(singles[0], host_pose)
        guest_mesh = moved_mesh(singles[1], guest_pose)
        host_value, guest_value = G.solid(host_mesh), G.solid(guest_mesh)
        # Reusing the original host preserves its first working opening.
        # Guest material may overlap the host and becomes shared material.
        host_work = G.solid(moved_mesh(G.mesh(works[0]), host_pose))
        added = G.mesh(material_only(guest_value - host_value - host_work))
        merged = host_value + guest_value - host_work
        merge_states.append((host_mesh, added))
        merge_checks.append(dict(state=index, support_support_overlap_cm3=G.volume(host_value ^ guest_value) * 1e6,
            first_work_overlap_m3=G.volume(merged ^ host_work)))
    merged_host = moved_mesh(singles[0], start_pose)
    final_world = moved_mesh(fixture, start_pose)
    body_world = moved_mesh(source, start_pose)
    cap_mesh = G.mesh(caps[0])
    low, high = cap_mesh.bounds
    cuts, cut_checks = [], []
    previous = caps[0]
    for index in range(25):
        if index == 0:
            remaining = caps[0]
        elif index == 24:
            remaining = md.Manifold()
        else:
            threshold = low[0] + (high[0] - low[0]) * index / 24
            remaining = material_only(caps[0] - G.box(low - .01, [threshold, high[1] + .01, high[2] + .01]))
        outside = G.volume(remaining - caps[0])
        restored = G.volume(remaining - previous)
        assert max(outside, restored) <= TOLERANCE
        mesh = G.mesh(remaining)
        cuts.append(moved_mesh(mesh, start_pose))
        cut_checks.append(dict(state=index, remaining_second_window_fill_cm3=G.volume(remaining) * 1e6,
                               removed_outside_task2_window_m3=outside, restored_material_m3=restored))
        previous = remaining
    assert cut_checks[-1]['remaining_second_window_fill_cm3'] == 0.
    assert max(row['first_work_overlap_m3'] for row in merge_checks) <= TOLERANCE
    renderer = Renderer(720, 720)
    pending = DATA / 'intro.pending.mp4'
    encoder = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '720x720', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(pending)], stdin=subprocess.PIPE)
    prefix = []
    try:
        for frame in range(PREFIX_GEOMETRY_FRAMES):
            camera = VIEW
            zoom = 0.
            items = []
            current_lines = None
            owner = None
            if frame < 24:
                phase = 'two_separate_single_task_fixtures'
                items = [*initial_items, *[(body, raster.GRAY, mask) for body, mask in zip(initial_bodies, masks)]]
            elif frame < 36:
                phase = 'remove_objects_before_fixture_alignment'
                offset = .32 * ease((frame - 24) / 11)
                local_exit = np.asarray(json.loads((reuse.DATA / 'validation.json').read_text())['local_extraction_direction'])
                items = list(initial_items)
                for matrix, mask in zip((host, guest), masks):
                    actual = matrix.copy()
                    actual[:3, 3] += offset * matrix[:3, :3] @ local_exit
                    items.append((moved_mesh(source, actual), raster.GRAY, mask))
            elif frame < 84:
                phase = 'rotate_and_overlap_two_task_fixtures_into_one_seat'
                fraction = ease((frame - 36) / 47)
                index = round(fraction * 48)
                first, added = merge_states[index]
                items = [(first, BLUE, None), (added, (.36, .70, .85), None)]
                camera = slerp(VIEW, cut_view, fraction)
                zoom = fraction
            elif frame < 96:
                phase = 'second_task_working_area_still_blocked'
                camera, zoom, owner = cut_view, 1., 1
                items = [(final_world, BLUE, None), (cuts[0], RED, None)]
            elif frame < 120:
                phase = 'carve_second_task_working_area'
                index = min(24, round((frame - 96) / 23 * 24))
                camera, zoom, owner = cut_view, 1., 1
                items = [(final_world, BLUE, None), (cuts[index], RED, None)]
            elif frame < 132:
                phase = 'completed_shared_fixture_with_two_working_windows'
                fraction = ease((frame - 120) / 11)
                camera, zoom = slerp(cut_view, VIEW, fraction), 1.
                insertion = start_pose.copy()
                insertion[:3, 3] += .32 * (1 - fraction) * (
                    start_pose[:3, :3] @ np.asarray(json.loads((reuse.DATA / 'validation.json').read_text())['local_extraction_direction']))
                items = [(final_world, BLUE, None), (moved_mesh(source, insertion), raster.GRAY, combined)]
            else:
                phase = 'approach_with_pose_rotation_needle'
                zoom = 1.
                retraction = .1 * (1 - ease((frame - 132) / 11))
                moving_tip = tip - retraction * axis
                items = [(final_world, BLUE, None), (body_world, raster.GRAY, combined),
                         *reuse.tool(moving_tip, axis, reference)]
                current_lines = lines - retraction * axis
            items = [(mesh, color, mask) for mesh, color, mask in items if len(mesh.faces)]
            if frame < 120:
                bounds = framing(wide_bounds, close_bounds, camera, zoom)
            elif frame < 132:
                bounds = framing(close_bounds, tail_bounds, camera, ease((frame - 120) / 11))
            else:
                bounds = tail_bounds
            picture = renderer.draw(items, bounds, camera, current_lines)
            for duplicate in range(2):
                encoder.stdin.write(picture.tobytes())
                prefix.append(dict(frame=2 * frame + duplicate, time_s=(2 * frame + duplicate) / FPS,
                    phase=phase, stage='construction', task_ids=['W06', 'W08'], active_task=owner,
                    support_support_overlap_allowed=True,
                    final_fixture_mesh_sha256=mesh_digest(fixture) if frame >= 120 else None))
            if frame % 24 == 0:
                print('ROTATIONAL INTRO FRAME', frame, '/', PREFIX_GEOMETRY_FRAMES, phase, flush=True)
    finally:
        renderer.close()
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError('Intro encoding failed')
    pending.replace(DATA / 'intro.mp4')
    write_json(DATA / 'prefix_timeline.json', prefix)
    write_json(DATA / 'construction_checks.json', dict(status='passed', merge_states=merge_checks,
        carve_states=cut_checks, final_fixture_mesh_sha256=mesh_digest(fixture),
        initial_second_window_blocking_material_cm3=float(cap_mesh.volume * 1e6),
        cutting_changes_only_second_task_working_exclusion=True))
    return prefix


def frame_hashes(path):
    output = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'framemd5', '-'], text=True)
    return [line.rsplit(',', 1)[-1].strip() for line in output.splitlines() if line and not line.startswith('#')]


def main():
    tail_path, tail_metadata = tail()
    source, fixture, _, timeline, reference, proof = reuse.load()
    singles, caps, works, geometry_proof = geometry(source, fixture, proof)
    prefix = render_prefix(source, fixture, timeline, reference, singles, caps, works)
    concat_file = DATA / 'concat.txt'
    concat_file.write_text(''.join("file '" + str(path) + "'\n" for path in [DATA / 'intro.mp4', tail_path]))
    pending = DATA / 'complete.pending.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat_file),
                    '-c', 'copy', '-movflags', '+faststart', str(pending)], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(pending), '-f', 'null', '-'], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate:format=duration',
        '-of', 'json', str(pending)]))
    assert int(probe['streams'][0]['nb_read_frames']) == 576
    assert abs(float(probe['format']['duration']) - 12.) < 1e-6
    old_hashes, complete_hashes = frame_hashes(tail_path), frame_hashes(pending)
    assert len(old_hashes) == 288 and complete_hashes[PREFIX_FRAMES:] == old_hashes
    pending.replace(VIDEO)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(VIDEO), '-vf',
                    'fps=1,scale=360:360,tile=4x3', '-frames:v', '1', str(reuse.DATA / 'encoded_review.jpg')], check=True)
    combined = list(prefix)
    for row in json.loads((reuse.DATA / 'timeline_tool.json').read_text()):
        current = dict(row, stage='reuse', source_reuse_frame=row['frame'],
                       frame=row['frame'] + PREFIX_FRAMES, time_s=row['time_s'] + 6.)
        combined.append(current)
    write_json(DATA / 'timeline.json', combined)
    metadata = dict(tail_metadata, video_sha256=sha256(VIDEO), sha256=sha256(VIDEO), frames=576,
        duration_seconds=12., decoded_frame_count=576, full_decode_passed=True,
        construction_intro_added=True, construction_intro_seconds=6., construction_intro_frames=PREFIX_FRAMES,
        initial_task_count=2, initial_fixtures_separate=True, initial_window_count_per_fixture=[1, 1],
        task_fixtures_rotated_and_merged_before_second_window_cut=True,
        second_working_area_explicitly_carved=True,
        second_window_removed_material_cm3=geometry_proof['second_window_removed_material_cm3'],
        support_geometry_constant_entire_video=False, support_modified_during_video=True,
        support_geometry_constant_during_reuse=True, support_modified_during_reuse=False,
        original_phase_and_frame_sequence_preserved=False, original_reuse_segment_preserved=True,
        original_reuse_segment_frames=288, original_reuse_segment_offset_frames=PREFIX_FRAMES,
        reuse_segment_encoded_pixels_unchanged=True, reuse_tail_sha256=sha256(tail_path),
        reuse_source_sha256=tail_metadata['source_sha256'],
        intro_geometry_validation=str((DATA / 'geometry_validation.json').relative_to(ROOT)),
        intro_geometry_sha256=geometry_proof['geometry_sha256'],
        timeline=str((DATA / 'timeline.json').relative_to(ROOT)), timeline_sha256=sha256(DATA / 'timeline.json'),
        source_sha256=sha256(Path(__file__)), source_snapshot=str((DATA / 'render_source.py').relative_to(ROOT)))
    (DATA / 'render_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(reuse.DATA / 'video.json', metadata)
    manifest = reuse.DATA.parent / 'videos.json'
    videos = json.loads(manifest.read_text())
    videos['rotational_reuse.mp4'] = metadata
    write_json(manifest, videos)
    write_json(DATA / 'media_validation.json', dict(status='passed', full_decode_passed=True,
        frames=576, fps=FPS, duration_seconds=12., reuse_tail_decoded_frames_identical=True,
        unchanged_tail_frame_count=288, video_sha256=sha256(VIDEO),
        final_fixture_mesh_sha256=mesh_digest(fixture),
        original_final_validation_sha256=sha256(reuse.DATA / 'validation.json'),
        original_tool_validation_sha256=sha256(reuse.DATA / 'tool_validation.json')))
    print('ROTATIONAL REUSE INTRO SAVED', VIDEO, '12s; original 288 frames unchanged', flush=True)
    return metadata


if __name__ == '__main__':
    main()
