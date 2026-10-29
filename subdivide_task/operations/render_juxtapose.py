"""Join the actual adjacent +X and +Y Step3 shells, then use both seats."""
import json
from pathlib import Path

import numpy as np
import trimesh

from subdivide_task.step1.common import ROOT, sha256, write_json
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step3.render_independent import Renderer, BLUE
from subdivide_task.step5.video import raster
from .geometry import moved_mesh
from .render_merge import ease, mesh_digest
from .render_pose_rotation import VIEW
from .render_refinements import encode, GREEN, RED
from .step3_juxtapose import build, DATA, TASKS, shifted


def main():
    pair, proof = build()
    with np.load(DATA / 'geometry.npz') as saved:
        final = trimesh.Trimesh(saved['final_vertices'], saved['final_faces'], process=False)
        distance = float(saved['translation_m'])
    final_digest = mesh_digest(final)
    states = []
    state_folder = DATA / 'overlap_states'
    state_folder.mkdir(exist_ok=True)
    for index, current_distance in enumerate(np.linspace(0., distance, 49)):
        path = state_folder / f'{index:03}.npz'
        binding = path.with_suffix('.json')
        cache = json.loads(binding.read_text()) if binding.exists() else {}
        if (cache.get('geometry_source_sha256') == proof['geometry_sha256']
                and path.exists() and cache.get('mesh_sha256') == sha256(path)):
            with np.load(path) as saved:
                parts = {key: trimesh.Trimesh(saved[key + '_vertices'], saved[key + '_faces'], process=False)
                         for key in ('host_only', 'guest_only', 'shared', 'union')}
        else:
            guest = pair.solids[1].translate([-current_distance / G.SCALE, 0., 0.])
            host = pair.solids[0]
            # Keep the complete support/support intersection. The shells are
            # design material, so they pass into each other and become a union.
            parts = dict(host_only=G.mesh(host - guest), guest_only=G.mesh(guest - host),
                         shared=G.mesh(host ^ guest), union=G.mesh(host + guest))
            arrays = {}
            for name, mesh in parts.items():
                arrays[name + '_vertices'] = mesh.vertices
                arrays[name + '_faces'] = mesh.faces
            np.savez_compressed(path, **arrays)
            write_json(binding, dict(state=index, guest_translation_m=[-float(current_distance), 0., 0.],
                geometry_source_sha256=proof['geometry_sha256'], mesh_sha256=sha256(path),
                support_support_overlap_cm3=float(parts['shared'].volume * 1e6) if len(parts['shared'].faces) else 0.,
                support_support_overlap_allowed=True, shared_material_subtracted=False,
                source_shells_translated_without_collision_avoidance=True))
        states.append(parts)
        if index % 8 == 0:
            print('JUXTAPOSITION OVERLAP STATE', index, '/', 49, flush=True)
    removed = G.mesh(G.solid(states[-1]['union']) - G.solid(final))
    source = pair.model.source
    masks = []
    bodies = []
    for index, record in enumerate(pair.model.records):
        mask = np.zeros(len(source.faces), bool)
        mask[record['working_face_ids']] = True
        masks.append(mask)
        bodies.append(moved_mesh(source, pair.layout.material[index] @ pair.layout.poses[index]))
    bounds = np.vstack([*[mesh.vertices for mesh in pair.shells],
                        *[mesh.vertices for mesh in bodies], final.vertices])
    renderer = Renderer(720, 720)
    frames, timeline = [], []
    try:
        for frame in range(240):
            current_distance = 0.
            state = None
            owner = None
            body_offset = 0.
            if frame < 24:
                phase = 'adjacent_step3_tasks'
                items = [(mesh, BLUE, None) for mesh in pair.shells]
                items.extend((body, raster.GRAY, mask) for body, mask in zip(bodies, masks))
            elif frame < 42:
                phase = 'withdraw_both_objects_before_overlapping_supports'
                body_offset = .32 * ease((frame - 24) / 17)
                items = [(mesh, BLUE, None) for mesh in pair.shells]
                items += [(shifted(body, body_offset * pair.layout.exits[i]), raster.GRAY, masks[i])
                          for i, body in enumerate(bodies)]
            elif frame < 132:
                phase = 'translate_supports_into_each_other' if frame < 114 else 'retain_shared_support_material'
                index = round(ease((frame - 42) / 71) * 48)
                state = index
                current_distance = distance * index / 48
                parts = states[index]
                items = [(parts['host_only'], BLUE, None),
                         (parts['guest_only'], (.36, .70, .85), None),
                         (parts['shared'], GREEN, None)]
            elif frame < 144:
                phase = 'cut_only_task_body_tool_and_exit_interference'
                current_distance = distance
                items = [(final, BLUE, None), (removed, RED, None)]
            else:
                current_distance = distance
                items = [(final, BLUE, None)]
                if frame < 156:
                    phase = 'insert_guest_object_into_merged_fixture'
                    owner = 1
                    body_offset = .32 * (1 - ease((frame - 144) / 11))
                elif frame < 180:
                    phase = 'use_guest_task_on_merged_fixture'
                    owner = 1
                elif frame < 192:
                    phase = 'withdraw_guest_object'
                    owner = 1
                    body_offset = .32 * ease((frame - 180) / 11)
                elif frame < 204:
                    phase = 'insert_host_object'
                    owner = 0
                    body_offset = .32 * (1 - ease((frame - 192) / 11))
                else:
                    phase = 'use_host_task_on_same_merged_fixture'
                    owner = 0
                if owner is not None:
                    delta = body_offset * pair.layout.exits[owner]
                    if owner == 1:
                        delta += [-distance, 0., 0.]
                    items.append((shifted(bodies[owner], delta), raster.GRAY, masks[owner]))
            items = [(mesh, color, mask) for mesh, color, mask in items if len(mesh.faces)]
            frames.append(renderer.draw(items, bounds, VIEW))
            timeline.append(dict(frame=frame, time_s=frame / 24, phase=phase, design_state=state,
                guest_translation_m=[-current_distance, 0., 0.], active_task=TASKS[owner] if owner is not None else None,
                body_exit_offset_m=body_offset,
                support_support_overlap_allowed=True, shared_support_material_subtracted=False,
                final_fixture_mesh_sha256=final_digest if frame >= 144 else None))
            if frame % 24 == 0:
                print('JUXTAPOSITION FRAME', frame, '/', 240, phase, flush=True)
    finally:
        renderer.close()
    assert all(row['final_fixture_mesh_sha256'] == final_digest for row in timeline[144:])
    assert mesh_digest(final) == final_digest
    write_json(DATA / 'timeline.json', timeline)
    media = encode('juxtaposition.mp4', frames, 24, DATA)
    source_snapshot = DATA / 'render_source.py'
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    metadata = dict(**media, sha256=media['video_sha256'], path=str(Path(__file__).with_name('juxtaposition.mp4')),
        task_ids=TASKS, adjacent_step3_task_indices=[0, 1], source_step3='subdivide_task/data/B/step3_tool_membranes',
        actual_saved_step3_shells_used=True, final_connected=True, final_volume_cm3=proof['final_volume_cm3'],
        guest_translation_m=proof['guest_translation_m'], final_full_audit_passed=proof['audit']['passed'],
        validation=str((DATA / 'validation.json').relative_to(ROOT)), validation_sha256=sha256(DATA / 'validation.json'),
        inputs_sha256=proof['inputs_sha256'], design_states=49,
        support_support_overlap_allowed=True, support_support_intersection_subtracted=False,
        shell_overlap_cm3=proof['shell_overlap_cm3'], retained_shared_material_cm3=proof['retained_shared_material_cm3'],
        both_objects_absent_while_supports_overlap=True,
        shared_support_material_color='green',
        source_sha256=sha256(Path(__file__)), source_snapshot=str(source_snapshot.relative_to(ROOT)),
        final_fixture_mesh_sha256=final_digest, final_fixture_geometry_constant_during_both_uses=True,
        source_geometry_sha256=proof['geometry_sha256'], timeline_sha256=sha256(DATA / 'timeline.json'),
        working_area_color='pink', studio_renderer='same Renderer, body material, and blue fixture as pose_rotation',
        tool_head_visible=False, text_overlays=False, panel_count=1,
        support_representation='wrapped_shell', no_base_plate=True, no_vertical_backing=True,
        continuous_robot_motion_claimed=False)
    write_json(DATA / 'video.json', metadata)
    index = DATA.parent / 'videos.json'
    videos = json.loads(index.read_text())
    videos['juxtaposition.mp4'] = metadata
    write_json(index, videos)
    print('JUXTAPOSITION SAVED', media['video_sha256'], flush=True)
    return metadata


if __name__ == '__main__':
    main()
