"""Join the actual adjacent +X and +Y Step3 shells, then use both seats."""
import json
from pathlib import Path

import numpy as np
import trimesh

from whole_task.step1.common import ROOT, sha256, write_json
from whole_task.step3 import independent_geometry as G
from whole_task.step3.render_independent import Renderer, BLUE
from whole_task.step5.video import raster
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
    for index, current_distance in enumerate(np.linspace(0., distance, 33)):
        path = DATA / f'state_{index:03}.npz'
        binding = path.with_suffix('.json')
        cache = json.loads(binding.read_text()) if binding.exists() else {}
        if (cache.get('geometry_source_sha256') == proof['geometry_sha256']
                and path.exists() and cache.get('mesh_sha256') == sha256(path)):
            with np.load(path) as saved:
                parts = {key: trimesh.Trimesh(saved[key + '_vertices'], saved[key + '_faces'], process=False)
                         for key in ('support', 'kept', 'added', 'removed')}
        else:
            candidate = pair.construct(float(current_distance))
            support = final if index == 32 else candidate['support']
            value = G.solid(support)
            parts = dict(support=support, kept=G.mesh(value ^ pair.solids[0]),
                         added=G.mesh(value - pair.solids[0]), removed=G.mesh(candidate['seed'] - value))
            arrays = {}
            for name, mesh in parts.items():
                arrays[name + '_vertices'] = mesh.vertices
                arrays[name + '_faces'] = mesh.faces
            np.savez_compressed(path, **arrays)
            write_json(binding, dict(state=index, guest_translation_m=[-float(current_distance), 0., 0.],
                geometry_source_sha256=proof['geometry_sha256'], mesh_sha256=sha256(path),
                exclusions=candidate['forbidden_overlaps_m3'],
                construction=candidate['construction'], final_geometry=(index == 32)))
        states.append(parts)
        if index % 8 == 0:
            print('JUXTAPOSITION STATE', index, '/', 33, flush=True)
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
        for frame in range(192):
            current_distance = 0.
            state = None
            owner = None
            body_offset = 0.
            if frame < 24:
                phase = 'adjacent_step3_tasks'
                items = [(mesh, BLUE, None) for mesh in pair.shells]
                items.extend((body, raster.GRAY, mask) for body, mask in zip(bodies, masks))
            elif frame < 36:
                phase = 'withdraw_host_object'
                body_offset = .32 * ease((frame - 24) / 11)
                items = [(mesh, BLUE, None) for mesh in pair.shells]
                items += [(shifted(bodies[0], body_offset * pair.layout.exits[0]), raster.GRAY, masks[0]),
                          (bodies[1], raster.GRAY, masks[1])]
            elif frame < 108:
                phase = 'translate_guest_and_merge_shells'
                index = round(ease((frame - 36) / 71) * 32)
                state = index
                current_distance = distance * index / 32
                parts = states[index]
                items = [(parts['kept'], BLUE, None), (parts['added'], GREEN, None),
                         (parts['removed'], RED, None),
                         (shifted(bodies[1], [-current_distance, 0., 0.]), raster.GRAY, masks[1])]
            else:
                current_distance = distance
                items = [(final, BLUE, None)]
                if frame < 132:
                    phase = 'use_guest_task_on_merged_fixture'
                    owner = 1
                elif frame < 144:
                    phase = 'withdraw_guest_object'
                    owner = 1
                    body_offset = .32 * ease((frame - 132) / 11)
                elif frame < 150:
                    phase = 'unchanged_merged_fixture'
                elif frame < 162:
                    phase = 'insert_host_object'
                    owner = 0
                    body_offset = .32 * (1 - ease((frame - 150) / 11))
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
                final_fixture_mesh_sha256=final_digest if frame >= 108 else None))
            if frame % 24 == 0:
                print('JUXTAPOSITION FRAME', frame, '/', 192, phase, flush=True)
    finally:
        renderer.close()
    assert all(row['final_fixture_mesh_sha256'] == final_digest for row in timeline[108:])
    assert mesh_digest(final) == final_digest
    write_json(DATA / 'timeline.json', timeline)
    media = encode('juxtaposition.mp4', frames, 24, DATA)
    source_snapshot = DATA / 'render_source.py'
    source_snapshot.write_bytes(Path(__file__).read_bytes())
    metadata = dict(**media, sha256=media['video_sha256'], path=str(Path(__file__).with_name('juxtaposition.mp4')),
        task_ids=TASKS, adjacent_step3_task_indices=[0, 1], source_step3='whole_task/data/B/step3_tool_membranes',
        actual_saved_step3_shells_used=True, final_connected=True, final_volume_cm3=proof['final_volume_cm3'],
        guest_translation_m=proof['guest_translation_m'], final_full_audit_passed=proof['audit']['passed'],
        validation=str((DATA / 'validation.json').relative_to(ROOT)), validation_sha256=sha256(DATA / 'validation.json'),
        inputs_sha256=proof['inputs_sha256'], design_states=33,
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
