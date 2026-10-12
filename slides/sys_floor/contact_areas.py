"""Two three-panel figures, with the union shown in Pose 1 or Pose 3.

The combined panel aligns the OBJECTS, using the saved object transforms.
It does not use Step5's independent fixture-placement transforms.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

import steps as D

HERE = Path(__file__).resolve().parent
FLOOR_COLORS = ['#d98a3b', '#438cba']
VIEW = [-1., -1., .82]
SIZE = 1080


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def read_inputs():
    case, _, _, _ = D.recipe(D.SOURCE)
    frames = [np.asarray(t.domain.data['frame']['T_world_mesh']) for t in case.tasks]
    align = frames[0] @ np.linalg.inv(frames[1])
    error = float(np.max(np.abs(transform(case.tasks[1].domain.mesh.vertices, align)
                                - case.tasks[0].domain.mesh.vertices)))
    assert error < 1e-12, 'Objects must coincide in the combined view.'
    # Contact-only figures remain independent of the rejected Step5 viewer.
    shared = case.schedule['shared_head']['selected_id']
    colors = {shared: '#dc9d47'}
    colors.update(zip([i for i in case.schedule['selected_ids'] if i != shared],
                      ['#ac7098', '#7196c0', '#50a59b', '#77a76a']))
    paths = set(case.paths) | {D.SOURCE / 'data/history/reference_report.json'}
    hashes = {str(p.relative_to(D.ROOT)): D.digest(p) for p in sorted(paths)}
    return case, align, colors, hashes, error


def floor_geometry(xy, matrix):
    lo, hi = xy.min(0) - .014, xy.max(0) + .014
    corners = np.array([[lo[0], lo[1], 0.], [hi[0], lo[1], 0.],
                        [hi[0], hi[1], 0.], [lo[0], hi[1], 0.]])
    hull = xy[ConvexHull(xy).vertices]
    return dict(corners=transform(corners, matrix),
                cloud=transform(np.c_[xy, np.zeros(len(xy))], matrix),
                hull=transform(np.c_[hull, np.zeros(len(hull))], matrix))


def project(points, camera):
    return D.R.project(points, *camera, SIZE)


def floor_layer(floor, color, camera):
    return D.floor_layer(floor, color, camera, SIZE)


def composite(layers):
    return D.composite(layers)


def render_panel(mesh, contacts, floors, camera, colors):
    layers = [floor_layer(floor, FLOOR_COLORS[k], camera) for k, floor in floors]
    obj, depth, ids = D.raster(mesh.triangles, np.tile(D.rgb('#aebabe'), (len(mesh.faces), 1)), camera, SIZE)
    layers.append((obj, depth, np.where(ids >= 0, .40, 0.)))
    triangles = np.concatenate([tri for _, tri in contacts])
    palette = np.concatenate([np.tile(D.rgb(colors[ident]), (len(tri), 1)) for ident, tri in contacts])
    pixels, depth, ids = D.raster(triangles, palette, camera, SIZE,
                                 bias=np.full(len(triangles), 2.))
    layers.append((pixels, depth, np.where(ids >= 0, 1., 0.)))
    return composite(layers)


def run():
    case, align, colors, hashes, _ = read_inputs()
    contacts = [[(c['candidate_id'], c['triangles_m']) for c in group] for group in case.groups]
    original_floors = [floor_geometry(demand, np.eye(4)) for demand in case.demands]
    scene_floors = [[(0, original_floors[0])], [(1, original_floors[1])]]
    scene_meshes = [task.domain.mesh for task in case.tasks]
    scene_contacts = list(contacts)
    combined_records = []
    for target, matrices in enumerate(([np.eye(4), align], [np.linalg.inv(align), np.eye(4)])):
        combined = {}
        object_error, shared_error = 0., 0.
        for source, matrix in enumerate(matrices):
            object_error = max(object_error, float(np.max(np.abs(
                transform(case.tasks[source].domain.mesh.vertices, matrix)
                - case.tasks[target].domain.mesh.vertices))))
            for ident, triangles in contacts[source]:
                aligned = transform(triangles, matrix)
                if ident in combined:
                    shared_error = max(shared_error, float(np.max(np.abs(aligned - combined[ident]))))
                else:
                    combined[ident] = aligned
        assert max(object_error, shared_error) < 1e-12
        assert len(combined) == 5
        scene_contacts.append(list(combined.items()))
        scene_meshes.append(case.tasks[target].domain.mesh)
        scene_floors.append([(i, floor_geometry(demand, matrix))
                             for i, (demand, matrix) in enumerate(zip(case.demands, matrices))])
        combined_records.append(dict(
            target_pose=case.poses[target], source_to_target_transforms=[m.tolist() for m in matrices],
            object_alignment_max_error_m=object_error, shared_patch_max_error_m=shared_error,
            unique_contact_count=len(combined)))
    # Both combined views use the original poses and the same camera direction
    # as the individual panels. The previous extra 90-degree orbit is retired.
    cameras = []
    for mesh, floors in zip(scene_meshes, scene_floors):
        axes = D.R.axes(VIEW)
        points = np.concatenate([mesh.vertices] + [f['corners'] for _, f in floors])
        view = points @ axes.T
        low, high = view.min(0), view.max(0)
        cameras.append([((low+high)/2) @ axes, axes, 1.12 * max(high[:2]-low[:2])])
    width = max(c[2] for c in cameras)
    for camera in cameras:
        camera[2] = width

    assets = HERE / 'contact_areas_data'
    assets.mkdir(exist_ok=True)
    titles = ['Pose 1', 'Pose 3', 'Both poses · Pose 1', 'Both poses · Pose 3']
    subtitles = ['3 contact areas + floor demands', '3 contact areas + floor demands',
                 'All 5 contact areas + both floor demands', 'All 5 contact areas + both floor demands']
    names = ['pose_1', 'pose_3', 'combined_pose1', 'combined_pose3']
    panel_width, panel_height = SIZE, SIZE + 108
    panels = []
    for k, (mesh, patches, floors, camera) in enumerate(zip(scene_meshes, scene_contacts, scene_floors, cameras)):
        print(f'Rendering {names[k]}', flush=True)
        image = render_panel(mesh, patches, floors, camera, colors)
        panel = Image.new('RGB', (panel_width, panel_height), 'white')
        panel.paste(image, (0, 108))
        ink = ImageDraw.Draw(panel)
        D.label(ink, (SIZE/2, 36), titles[k], 35)
        D.label(ink, (SIZE/2, 83), subtitles[k], 23, D.MUTED)
        panel.save(assets / f'{names[k]}.png', dpi=(200, 200))
        panels.append(panel)
    figures = []
    for target, filename in enumerate(['contact_areas.png', 'contact_areas_pose3.png']):
        pose_label = case.poses[target].replace('_', ' ').title()
        page = Image.new('RGB', (3 * panel_width + 96, panel_height + 262), 'white')
        draw = ImageDraw.Draw(page)
        D.label(draw, (48, 52), 'Contact areas and floor demands', 47, anchor='lm')
        D.label(draw, (48, 108), 'B / Pose 1 + Pose 3', 28, D.MUTED, 'lm')
        for k, panel in enumerate([panels[0], panels[1], panels[2+target]]):
            page.paste(panel, (48+k*panel_width, 156))
        y = page.height - 83
        for k, title in enumerate(['Pose 1 floor demands', 'Pose 3 floor demands']):
            x = 72 + k * 540
            draw.ellipse((x, y-9, x+20, y+11), fill=FLOOR_COLORS[k])
            D.label(draw, (x+35, y), title, 25, D.MUTED, 'lm')
        D.label(draw, (1260, y), 'Colored surfaces: object contacts', 25, D.MUTED, 'lm')
        D.label(draw, (72, page.height-37),
                f'Combined view: object aligned to {pose_label}; both floor planes move with their pose. 32,768 floor points per pose.',
                25, D.MUTED, 'lm')
        output = HERE / filename
        page.save(output, dpi=(200, 200))
        figures.append(dict(file=filename, combined_pose=case.poses[target], sha256=D.digest(output)))
        print(output, flush=True)
    for path, sha in hashes.items():
        assert D.digest(D.ROOT / path) == sha, f'Source changed: {path}'
    metadata = dict(source_sha256=hashes, figure_sha256=figures[0]['sha256'], figures=figures,
                    object='B', poses=case.poses, combined_views=combined_records,
                    contact_counts=[3, 3, 5], shared_patch_drawn_once=True,
                    floor_point_counts=[len(d) for d in case.demands], all_floor_points_rendered=True,
                    contact_geometry='exact saved Step3 surface triangles, no head collar or body',
                    floors='saved Step4 demands, rigidly transformed; never flattened onto one plane',
                    camera_views=[VIEW] * 4, camera_width_m=width,
                    combined_display_rotation_ccw_degrees=0,
                    floor_color='#e5e8e9', fixture_geometry_rebuilt=False)
    (assets / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (assets / 'combined.png').unlink(missing_ok=True)
    print(json.dumps(combined_records), flush=True)


if __name__ == '__main__':
    run()
