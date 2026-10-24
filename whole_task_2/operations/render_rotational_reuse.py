"""Replay the immutable two-window fixture with the pose_rotation tool model."""
import json
from pathlib import Path

import numpy as np
import trimesh

from whole_task_2.step1.common import ROOT, sha256, write_json
from whole_task_2.step1.find_tasks_figure import cone
from whole_task_2.step1.reach_video import tool_frame, HEAD, STEEL
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.render_independent import Renderer, BLUE
from whole_task_2.step5.video import raster
from .geometry import moved_mesh
from .render_merge import mesh_digest, ease
from .render_pose_rotation import first_hit, VIEW
from .render_refinements import encode, TOLERANCE


DATA = ROOT / 'whole_task_2/data/B/operation_videos/merge'
ARCHIVE = ROOT / 'whole_task_2/data/B/step3_connected_output_legacy/step3.2'
REFERENCE = DATA.parent / 'pose_rotation_render.json'


def tool(tip, axis, reference):
    matrix = tool_frame(tip, axis)
    length = reference['needle_length_m']
    needle = reference['needle_radius_m']
    head = reference['head_radius_m']
    items = []
    for v, f, color in [(*raster.tool_mesh(0, length, needle, needle), STEEL),
                        (*raster.tool_mesh(length, length + reference['head_length_m'], head, head), HEAD)]:
        items.append((trimesh.Trimesh(v @ matrix[:3, :3].T + matrix[:3, 3], f, process=False), color, None))
    return items


def load():
    reference = json.loads(REFERENCE.read_text())
    proof = json.loads((DATA / 'validation.json').read_text())
    assert sha256(DATA / 'fixture.npz') == proof['fixture_sha256']
    with np.load(DATA / 'fixture.npz') as saved:
        fixture = trimesh.Trimesh(saved['vertices'], saved['faces'], process=False)
    assert mesh_digest(fixture) == proof['fixture_mesh_sha256']
    with np.load(ARCHIVE / 'initialization.npz') as saved:
        source = trimesh.Trimesh(saved['source_vertices_object'], saved['source_faces'], process=False)
    report = json.loads((ARCHIVE / 'report.json').read_text())
    mask = np.zeros(len(source.faces), bool)
    for row in report['tasks']:
        if row['id'] in ['W06', 'W08']:
            mask[row['working_face_ids']] = True
    old = json.loads((DATA / 'timeline.json').read_text())
    axis = np.asarray(reference['direction_axis_world'])
    tip = np.asarray(reference['tool_tip_world_m'])
    origin = np.asarray(old[0]['needle_root_world_m'])
    shifts = [tip - first_hit(moved_mesh(source, np.asarray(row['object_world_from_object'])), origin, axis)
              for row in (old[0], old[-1])]
    timeline = []
    for row in old:
        frame = row['frame']
        u = ease((frame - 120) / 47)
        shift = shifts[0] * (1 - u) + shifts[1] * u
        current = dict(row)
        for key in ('fixture_world_from_object', 'object_world_from_object'):
            if row[key] is not None:
                matrix = np.asarray(row[key]).copy()
                matrix[:3, 3] += shift
                current[key] = matrix.tolist()
        current['contact_alignment_translation_m'] = shift.tolist()
        timeline.append(current)
    return source, fixture, mask, timeline, reference, proof


def select_tool(source, fixture, timeline, reference):
    """Retract the real needle before using the original object exit paths."""
    tip = np.asarray(reference['tool_tip_world_m'])
    axis = np.asarray(reference['direction_axis_world'])
    states = []
    for row in timeline:
        solids = []
        for kind, mesh, key in [('body', source, 'object_world_from_object'),
                                ('fixture', fixture, 'fixture_world_from_object')]:
            if row[key] is not None:
                moved = moved_mesh(mesh, np.asarray(row[key]))
                solids.append((kind, moved.bounds, G.solid(moved)))
        states.append(solids)
    trace = []
    for stroke in [.10, .15, .20, .25, .30, .40]:
        checks, seen = [], {}
        for row, solids in zip(timeline, states):
            frame = row['frame']
            if frame < 72:
                fraction = 0.
            elif frame < 84:
                fraction = ease((frame - 72) / 11)
            elif frame < 204:
                fraction = 1.
            elif frame < 216:
                fraction = 1. - ease((frame - 204) / 11)
            else:
                fraction = 0.
            retraction = stroke * fraction
            tool_tip = tip - retraction * axis
            items = tool(tool_tip, axis, reference)
            value = G.union([G.solid(mesh) for mesh, _, _ in items])
            vertices = np.vstack([mesh.vertices for mesh, _, _ in items])
            bounds = np.array([vertices.min(0), vertices.max(0)])
            failed = False
            for kind, other, solid in solids:
                matrix = row['object_world_from_object' if kind == 'body' else 'fixture_world_from_object']
                key = (kind, np.asarray(matrix).tobytes(), retraction)
                if key not in seen:
                    seen[key] = (0. if np.any(bounds[1] < other[0]) or np.any(other[1] < bounds[0])
                                 else G.volume(value ^ solid))
                overlap = seen[key]
                checks.append(dict(frame=frame, kind=kind, overlap_m3=overlap))
                if overlap > TOLERANCE:
                    failed = True
                    break
            if failed:
                trace.append(dict(stroke_m=stroke, failed_check=checks[-1]))
                print('REUSE TOOL RETRACTION', stroke, 'failed at', frame, flush=True)
                break
            row['tool_retraction_m'] = retraction
            row['tool_tip_world_m'] = tool_tip.tolist()
        else:
            result = dict(status='passed', tool_fixed_during_both_working_holds=True,
                fixed_tool_direction_entire_video=True, tool_axis_world=axis.tolist(),
                nominal_tilt_deg=0., working_tip_world_m=tip.tolist(), axial_retraction_m=stroke,
                needle_length_m=reference['needle_length_m'], needle_radius_m=reference['needle_radius_m'],
                head_radius_m=reference['head_radius_m'], head_length_m=reference['head_length_m'],
                unique_checked_states=len(seen), checked_frames=len(timeline),
                maximum_body_overlap_m3=max(row['overlap_m3'] for row in checks if row['kind'] == 'body'),
                maximum_fixture_overlap_m3=max(row['overlap_m3'] for row in checks if row['kind'] == 'fixture'),
                tolerance_m3=TOLERANCE, checks=checks, rejected_candidates=trace,
                reference_sha256=sha256(REFERENCE), fixture_sha256=sha256(DATA / 'fixture.npz'),
                original_timeline_sha256=sha256(DATA / 'timeline.json'), source_sha256=sha256(Path(__file__)))
            write_json(DATA / 'tool_validation.json', result)
            print('REUSE TOOL PASS', stroke, 'm axial retraction', len(timeline), 'frames', flush=True)
            return tool(tip, axis, reference), axis, result
    write_json(DATA / 'tool_search.json', dict(status='failed', candidates=trace))
    raise RuntimeError('Needle retraction does not clear the original object motion')


def main():
    source, fixture, mask, timeline, reference, proof = load()
    contexts, axis, validation = select_tool(source, fixture, timeline, reference)
    tip = np.asarray(reference['tool_tip_world_m'])
    lines = np.asarray(cone(tip - reference['needle_length_m'] * axis, axis,
                            reference['needle_length_m'], 25., 8)).reshape(-1, 3)
    clouds = [lines, *[mesh.vertices for mesh, _, _ in contexts]]
    for row in timeline:
        clouds.append(moved_mesh(fixture, np.asarray(row['fixture_world_from_object'])).vertices)
    for row in (timeline[0], timeline[-1]):
        clouds.append(moved_mesh(source, np.asarray(row['object_world_from_object'])).vertices)
    bounds = np.vstack(clouds)
    renderer = Renderer(720, 720)
    frames = []
    try:
        for row in timeline:
            items = [(moved_mesh(fixture, np.asarray(row['fixture_world_from_object'])), BLUE, None)]
            if row['object_visible']:
                items.append((moved_mesh(source, np.asarray(row['object_world_from_object'])), raster.GRAY, mask))
            moving_tip = np.asarray(row['tool_tip_world_m'])
            moving_contexts = tool(moving_tip, axis, reference)
            moving_lines = lines - row['tool_retraction_m'] * axis
            frames.append(renderer.draw([*items, *moving_contexts], bounds, VIEW, moving_lines))
            row.update(needle_axis_world=axis.tolist(),
                       needle_root_world_m=(moving_tip - reference['needle_length_m'] * axis).tolist())
            if row['frame'] % 48 == 0:
                print('ROTATIONAL REUSE FRAME', row['frame'], '/', len(timeline), flush=True)
    finally:
        renderer.close()
    write_json(DATA / 'timeline_tool.json', timeline)
    media = encode('rotational_reuse.mp4', frames, 48, DATA)
    metadata = dict(json.loads((DATA / 'video.json').read_text()), **media)
    metadata.update(sha256=media['video_sha256'], fixed_tool_position_entire_video=False,
        tool_fixed_during_both_working_holds=True, tool_axial_retraction_m=validation['axial_retraction_m'], working_tool='same steel needle and black cylindrical head as pose_rotation',
        needle_length_m=reference['needle_length_m'], needle_radius_m=reference['needle_radius_m'],
        head_radius_m=reference['head_radius_m'], head_length_m=reference['head_length_m'],
        tool_tip_world_m=tip.tolist(), direction_axis_world=axis.tolist(),
        nominal_tool_tilt_deg=validation['nominal_tilt_deg'],
        tool_validation=str((DATA / 'tool_validation.json').relative_to(ROOT)),
        timeline=str((DATA / 'timeline_tool.json').relative_to(ROOT)),
        timeline_sha256=sha256(DATA / 'timeline_tool.json'), source_sha256=sha256(Path(__file__)),
        tool_model_reference_sha256=sha256(REFERENCE),
        fixture_mesh_sha256=mesh_digest(fixture), original_phase_and_frame_sequence_preserved=True,
        tool_contact_on_both_seated_surfaces=True, source_snapshot=str((DATA / 'render_tool_source.py').relative_to(ROOT)))
    (DATA / 'render_tool_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(DATA / 'video.json', metadata)
    manifest = DATA.parent / 'videos.json'
    videos = json.loads(manifest.read_text())
    videos['rotational_reuse.mp4'] = metadata
    write_json(manifest, videos)
    print('ROTATIONAL REUSE SAVED', media['video_sha256'], flush=True)


if __name__ == '__main__':
    main()
