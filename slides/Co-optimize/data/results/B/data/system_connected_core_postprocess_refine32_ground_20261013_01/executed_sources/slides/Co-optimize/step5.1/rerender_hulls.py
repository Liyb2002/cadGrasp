"""Redraw saved Step5.1 results with convex hulls; preserve all demand arrays."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import time

import run as S

np, F, V, H = S.np, S.F, S.V, S.H


def redraw(case, base, mesh, renderer, batch):
    started = time.perf_counter()
    out = base/case['id']/'step5/step5.1'
    report_path = out/'data/report.json'
    record = json.loads(report_path.read_text())
    if not record['complete'] or record['selected_step4_experiment'] != case['experiment']:
        raise ValueError(f'Saved Step5.1 does not use the current selected layout: {out}')
    if not case['passed'] or record['pose_count'] != case['pose_count']:
        raise ValueError('Expected every task in the selected passed layout')
    protected = dict(record['provenance']['inputs'])
    for path, expected in protected.items():
        if S.digest(S.ROOT/path) != expected:
            raise ValueError(f'Current inputs differ from the saved result: {path}')
    history = out/'data/history'/batch['name']
    history.mkdir(parents=True, exist_ok=False)
    for path in [*out.glob('*.png'), out/'index.html', report_path]:
        target = history/path.relative_to(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    fixture_path = S.ROOT/record['selected_step4_source']/'support.obj'
    fixture = S.trimesh.load(fixture_path, force='mesh', process=False)
    common_transform = np.asarray(record['common_view_transform'])
    tiles, clouds, hulls = [], [], []
    for row in record['per_pose']:
        array_path = out/row['arrays']
        protected[S.relative(array_path)] = S.digest(array_path)
        with np.load(array_path, allow_pickle=False) as saved:
            cloud = saved['floor_demands_world_m']
            local_cloud = saved['floor_demands_fixture_m']
            indices = saved['hull_indices']
            object_transform = saved['T_world_object']
            fixture_transform = saved['T_world_fixture']
            equations = saved['hull_equations_xy']
        if len(cloud) != row['sample_count'] or len(cloud) != 32768:
            raise ValueError('Every original saved point must be retained')
        actual_hull = S.ConvexHull(cloud[:, :2])
        if not np.array_equal(indices, actual_hull.vertices):
            raise ValueError(f'Saved polygon is not the minimum convex hull: {array_path}')
        residual = float(np.max(cloud[:, :2]@equations[:, :2].T+equations[:, 2]))
        if residual > 1e-12:
            raise ValueError(f'Saved hull does not enclose all points: {array_path}')
        tile, camera = H.render_pose(renderer, S.moved_mesh(mesh, object_transform),
            S.moved_mesh(fixture, fixture_transform), cloud, row['color'], out/row['image'], hull=cloud[indices])
        tiles.append(tile)
        clouds.append(F.transform_points(local_cloud, common_transform))
        hulls.append(F.transform_points(local_cloud[indices], common_transform))
        row.update(camera=S.camera_record(camera), convex_hull_boundary_drawn=True,
                   convex_hull_vertex_count=len(indices), maximum_hull_halfplane_residual_m=residual)
        print('STEP5.1 HULL',case['id'],row['pose'],len(indices),'vertices',flush=True)
    V.sheet(tiles, out/'floor_demands.png')
    camera = H.render_common(renderer, S.moved_mesh(fixture, common_transform), clouds,
        [r['color'] for r in record['per_pose']], out/'common_fixture_demands.png', hulls=hulls)
    S.write_case_index(out, case['id'], record['per_pose'])
    record.update(convex_hull_boundaries_drawn=True,
        convex_hull_definition='minimum convex polygon of all 32768 saved points in each task floor plane',
        common_view_camera=S.camera_record(camera),
        rerender_seconds=time.perf_counter()-started,
        rerender_provenance=dict(batch=S.relative(batch['manifest_path']), executed_code=batch['executed_code'],
                                previous_images_and_report=S.relative(history), demand_arrays_unchanged=True))
    record['artifacts'] = {str(p.relative_to(out)): S.digest(p)
        for p in [*out.glob('*.png'), out/'index.html', *out.glob('data/*.npz')]}
    S.save(report_path, record)
    for path, expected in protected.items():
        if S.digest(S.ROOT/path) != expected:
            raise ValueError(f'Protected input changed: {path}')
    return dict(id=case['id'], poses=record['poses'], pose_count=record['pose_count'],
        sample_count=record['sample_count'], image_count=len(tiles)+2,
        common_image=str((out/'common_fixture_demands.png').relative_to(base)),
        per_pose_image=str((out/'floor_demands.png').relative_to(base)),
        maximum_hull_halfplane_residual_m=max(r['maximum_hull_halfplane_residual_m'] for r in record['per_pose']),
        seconds=record['rerender_seconds'], provenance=protected, old_images=S.relative(history))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--selection-summary', type=Path)
    parser.add_argument('--sets', nargs='+')
    parser.add_argument('--batch-name', default='step51_convex_hull_v1')
    args = parser.parse_args()
    base = S.COOPT/'output'/args.object
    selection_path = (args.selection_summary or base/'data/stable_gradient_xyz_force_v3/summary.json').resolve()
    cases = json.loads(selection_path.read_text())['cases']
    if args.sets:
        missing = set(args.sets)-{r['id'] for r in cases}
        if missing: parser.error(f'Sets missing from current selection: {sorted(missing)}')
        cases = [r for r in cases if r['id'] in args.sets]
    if not cases: parser.error('No selected sets')
    for case in cases:
        out = base/case['id']/'step5/step5.1'
        if not (out/'data/report.json').is_file(): parser.error(f'No saved Step5.1 result: {out}')
        if (out/'data/history'/args.batch_name).exists(): parser.error(f'Existing history is protected: {out}')
    directory = base/'data'/args.batch_name
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    code = S.freeze_code(directory)
    manifest = directory/'execution.json'
    batch = dict(name=args.batch_name, manifest_path=manifest, executed_code=code)
    S.save(manifest, dict(schema='coopt_step51_convex_hull_rerender_v1',
        started_utc=datetime.now(timezone.utc).isoformat(), object=args.object,
        selection=S.source(selection_path), groups=[r['id'] for r in cases], executed_code=code,
        saved_demand_arrays_reused=True, projection_recomputed=False,
        optimization_run=False, force_solver_run=False, geometry_boolean_run=False))
    gallery = base/'step5.1_index.html'
    if gallery.exists(): shutil.copy2(gallery, directory/'previous_step5.1_index.html')
    mesh = S.trimesh.load(S.ROOT/'objects'/args.object/'mesh.stl', force='mesh', process=False)
    renderer = V.Renderer()
    rows = [redraw(case, base, mesh, renderer, batch) for case in cases]
    protected = {S.relative(selection_path): S.digest(selection_path)}
    for row in rows: protected.update(row['provenance'])
    for path, expected in protected.items():
        if S.digest(S.ROOT/path) != expected: raise ValueError(f'Upstream input changed: {path}')
    for path, expected in code.items():
        if S.digest(S.ROOT/path) != expected or S.digest(directory/'executed_sources'/path) != expected:
            raise ValueError(f'Executed source or archive changed: {path}')
    # Keep unselected sets in the gallery when only part of the batch is redrawn.
    display = json.loads(selection_path.read_text())['cases']
    cards = ''.join(f'<section><h2>{html.escape(r["id"])}</h2>'
        f'<p><a href="{r["id"]}/step5/step5.1/index.html">Each pose</a></p>'
        f'<img src="{r["id"]}/step5/step5.1/common_fixture_demands.png">'
        f'<img src="{r["id"]}/step5/step5.1/floor_demands.png"></section>' for r in display)
    gallery.write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 20px;color:#294553}'
        'img{width:100%;height:auto}section{margin:35px 0}</style>'
        '<h1>Step5.1 · Ground demands and convex hulls</h1>'
        '<p>Colored straight boundaries enclose all saved demand points in each pose floor plane. '
        'Current final layouts and demand arrays are retained.</p>'+cards+'\n')
    summary = dict(complete=True, object=args.object, cases=rows, group_count=len(rows),
        pose_instance_count=sum(r['pose_count'] for r in rows), image_count=sum(r['image_count'] for r in rows),
        original_point_count=sum(r['sample_count'] for r in rows),
        all_saved_points_inside_convex_hulls=True, demand_arrays_unchanged=True,
        protected_inputs_unchanged=True, protected_input_count=len(protected),
        executed_sources_unchanged=True, executed_source_count=len(code),
        maximum_hull_halfplane_residual_m=max(r['maximum_hull_halfplane_residual_m'] for r in rows),
        optimization_run=False, force_solver_run=False, geometry_boolean_run=False,
        total_seconds=time.perf_counter()-started)
    S.save(directory/'summary.json', summary)
    S.save(directory/'verification.json', dict(complete=True, protected_inputs=protected,
        executed_code=code, source_and_snapshot_hashes_match=True, all_upstream_hashes_match=True,
        demand_arrays_unchanged=True, all_saved_points_inside_convex_hulls=True))
    print('STEP5.1 HULLS COMPLETE',len(rows),'sets',summary['pose_instance_count'],'poses',
          summary['image_count'],'images',round(summary['total_seconds'],2),'seconds',flush=True)
    print('GALLERY',S.relative(gallery),flush=True)


if __name__ == '__main__':
    main()
