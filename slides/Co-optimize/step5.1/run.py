"""Compute/render deterministic ground-demand clouds from selected Step4 layouts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cadgrasp-step51-matplotlib')

COOPT = Path(__file__).resolve().parents[1]
ROOT = COOPT.parents[1]
for folder in [ROOT, COOPT / 'helper_func', COOPT / 'vis_func']:
    sys.path.insert(0, str(folder))

import numpy as np
from scipy.spatial import ConvexHull
import trimesh

import system_floor as F
import step51_render as V


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def source(path):
    return dict(path=relative(path), sha256=digest(path))


class NativeInputs:
    """Read original angle arrays once per unique pose, never resample them."""
    def __init__(self, name, mesh):
        self.name = name
        self.mesh = mesh
        self.rows = {}
        self.mesh_hash = digest(ROOT / 'objects' / name / 'mesh.stl')
        self.poses_hash = digest(ROOT / 'objects' / name / 'poses.json')

    def get(self, pose):
        if pose not in self.rows:
            folder = ROOT / 'objects' / self.name / 'poses' / pose
            setup_path = folder / 'setup.npz'
            with np.load(setup_path) as z:
                setup = {k: z[k].copy() for k in z.files}
            if str(setup['object']) != self.name or str(setup['pose_id']) != pose:
                raise ValueError('Native snapshot does not match requested task')
            if str(setup['mesh_sha256']) != self.mesh_hash or str(setup['poses_sha256']) != self.poses_hash:
                raise ValueError('Native task geometry fingerprint differs from current object')
            angle = int(float(setup['cone_half_deg']))
            angle_folder = folder / f'angle_{angle}'
            sample_path = angle_folder / 'samples.npz'
            metadata_path = angle_folder / 'sample_metadata.json'
            variant_path = angle_folder / 'variant.json'
            metadata = json.loads(metadata_path.read_text())
            variant = json.loads(variant_path.read_text())
            if metadata['array_file_sha256'] != digest(sample_path):
                raise ValueError('Saved sample array fingerprint mismatch')
            with np.load(sample_path) as z:
                loads = np.asarray(z['need_wrench'], dtype=float).copy()
                origin = z['moment_origin_m'].copy()
                sample_weight = float(z['weight_per_sample'])
            if loads.shape != (32768, 6) or int(metadata['count']) != len(loads):
                raise ValueError('Expected every original 32768 paired demand')
            np.testing.assert_allclose(origin, setup['com_m'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(F.transform_points(self.mesh.center_mass, setup['T_world_mesh']),
                                       origin, atol=1e-12, rtol=0)
            # The 30-degree NPZ is the same original native demand sequence used
            # by Step4, not a freshly generated angle variant.
            wrench_hash = hashlib.sha256(np.ascontiguousarray(loads).tobytes()).hexdigest()
            if wrench_hash != variant['sample_array_fingerprints']['need_wrench']:
                raise ValueError('COM demand bytes differ from the saved angle input')
            original_sample_path = folder / 'samples.json'
            if digest(original_sample_path) != variant['original_inputs']['samples.json']:
                raise ValueError('Original native demand source changed after angle-array archival')
            self.rows[pose] = dict(setup=setup, loads=loads, sample_weight=sample_weight,
                sources=[source(p) for p in [setup_path, sample_path, metadata_path, variant_path, original_sample_path]])
        return self.rows[pose]


def moved_mesh(mesh, transform):
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def camera_record(camera):
    focus, basis, span = camera
    return dict(focus_m=focus.tolist(), basis=basis.tolist(), span_m=float(span))


def write_case_index(out, group, rows):
    cards = ''.join(
        f'<section id="{r["pose"]}"><h2 style="color:{r["color"]}">{r["pose"]}</h2>'
        f'<p>{r["sample_count"]:,} points · object height {r["workpiece_height_m"]*1000:.4f} mm'
        f' · host {r["host_pose"]}</p><img src="{r["image"]}"></section>' for r in rows)
    text = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{font:16px system-ui;max-width:1300px;margin:24px auto;padding:0 20px;color:#294553}'
        'img{width:100%;height:auto}section{margin:24px 0}p{line-height:1.6}</style>'
        f'<h1>{html.escape(group)} · Step5.1</h1>'
        '<p>Blue: unchanged final fixture. Grey: object. Colored points: system-floor pressure centers. '
        'All 32,768 original samples per task; no resampling. Default fixture self-weight is zero, '
        'as in the existing objects model. This is a ground-demand visualization, not a base acceptance.</p>'
        '<h2>All clouds around the same fixture</h2><p>Each cloud is mapped using its actual fixture state; '
        'the different floor planes remain different planes. Pose colors match the panels below.</p>'
        '<img src="common_fixture_demands.png"><h2>Each task in its own final world placement</h2>'
        '<img src="floor_demands.png">' + cards)
    (out / 'index.html').write_text(text + '\n')


def compute_case(name, case, base, reader, renderer, weight_ratio, batch_record):
    began = time.perf_counter()
    selected = base / case['id'] / 'step4/step4.2' / case['experiment']
    out = base / case['id'] / 'step5/step5.1'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Preserve existing Step5.1 output; choose/archive explicitly: {out}')
    (out / 'data').mkdir(parents=True, exist_ok=True)
    report_path = selected / 'data/report.json'
    layout_path = selected / 'layout.npz'
    fixture_path = selected / 'support.obj'
    selected_source_hashes = {relative(p): digest(p) for p in [report_path, layout_path, fixture_path]}
    report = json.loads(report_path.read_text())
    poses = report['poses']
    if not case['passed'] or len(poses) != case['pose_count']:
        raise ValueError('Selected source must be a completed passed Step4 layout')
    with np.load(layout_path) as z:
        native = z['native_world'].copy()
        placements = z['placements'].copy()
        hosts = z['hosts'].copy()
        active = z['active'].copy()
    if not np.array_equal(np.sort(active), np.arange(len(poses))):
        raise ValueError('Step5.1 must include every task in the final set')
    fixture = trimesh.load(fixture_path, force='mesh', process=False)
    if not np.isclose(abs(fixture.volume)*1e6, case['volume_cm3'], rtol=0, atol=1e-6):
        raise ValueError('Support mesh is not the selected final material result')
    local_fixture_com = fixture.center_mass
    rows, tiles, clouds, all_sources = [], [], [], []
    calculation_seconds = 0.
    for k, pose in enumerate(poses):
        task_started = time.perf_counter()
        task = reader.get(pose)
        setup = task['setup']
        np.testing.assert_allclose(native[k], setup['T_world_mesh'], atol=1e-12, rtol=0)
        fixture_T, object_T = F.final_task_frame(native, placements, hosts, k)
        translation = object_T[:3, 3] - native[k, :3, 3]
        final_com = setup['com_m'] + translation
        final_fixture_com = F.transform_points(local_fixture_com, fixture_T)
        result = F.system_floor_demands(task['loads'], final_com,
            fixture_weight_ratio=weight_ratio, fixture_com_world_m=final_fixture_com)
        reference = F.system_floor_demands(task['loads'], setup['com_m'])
        cloud = result['floor_demands_world_m']
        local_cloud = F.transform_points(cloud, np.linalg.inv(fixture_T))
        np.testing.assert_allclose(F.transform_points(local_cloud, fixture_T), cloud, atol=1e-12, rtol=0)
        formula_residual = None
        if weight_ratio == 0:
            expected_shift = translation[:2] - translation[2] * task['loads'][:, :2] / task['loads'][:, 2, None]
            formula_residual = float(np.max(np.abs(result['floor_demands_xy_m'] - reference['floor_demands_xy_m'] - expected_shift)))
            if formula_residual > 1e-12:
                raise ValueError('Final-position pressure-center conversion disagrees with translation law')
        posed_body = moved_mesh(reader.mesh, object_T)
        posed_fixture = moved_mesh(fixture, fixture_T)
        height = float(posed_body.vertices[:, 2].min())
        if height < -1e-9:
            raise ValueError('Final object penetrates its task floor')
        hull = ConvexHull(result['floor_demands_xy_m'])
        array_path = out / 'data' / f'{pose}.npz'
        np.savez_compressed(array_path,
            **result, floor_demands_fixture_m=local_cloud,
            native_floor_demands_xy_m=reference['floor_demands_xy_m'],
            hull_indices=hull.vertices, hull_equations_xy=hull.equations,
            hull_area_m2=np.asarray(hull.volume),
            object_com_world_m=final_com, fixture_com_world_m=final_fixture_com,
            workpiece_height_m=np.asarray(height), translation_world_m=translation,
            T_world_fixture=fixture_T, T_world_object=object_T,
            sample_weight=np.asarray(task['sample_weight']),
            fixture_weight_ratio=np.asarray(weight_ratio))
        calculation_seconds += time.perf_counter() - task_started
        color = V.COLORS[k % len(V.COLORS)]
        image_path = out / f'{pose}.png'
        tile, camera = V.render_pose(renderer, posed_body, posed_fixture, cloud, color, image_path)
        tiles.append(tile)
        clouds.append(F.transform_points(local_cloud, native[0]))
        all_sources.extend(task['sources'])
        row = dict(pose=pose, host_pose=poses[int(hosts[k])], sample_count=len(cloud),
            rendered_point_count=len(cloud), subsampled=False, color=color,
            image=image_path.name, arrays=str(array_path.relative_to(out)),
            workpiece_height_m=height, airborne=bool(height>1e-9),
            translation_world_m=translation.tolist(), object_com_world_m=final_com.tolist(),
            fixture_com_world_m=final_fixture_com.tolist(),
            object_world_transform=object_T.tolist(), fixture_world_transform=fixture_T.tolist(),
            minimum_floor_normal_mg=float(result['total_floor_normal_mg'].min()),
            maximum_floor_normal_mg=float(result['total_floor_normal_mg'].max()),
            point_bounds_world_m=[cloud.min(axis=0).tolist(), cloud.max(axis=0).tolist()],
            convex_hull_area_cm2=float(hull.volume*1e4),
            maximum_shift_from_native_mm=float(np.linalg.norm(result['floor_demands_xy_m'] - reference['floor_demands_xy_m'],axis=1).max()*1000),
            translation_formula_residual_m=formula_residual, camera=camera_record(camera),
            source_samples=task['sources'])
        rows.append(row)
        print('STEP5.1 POSE',case['id'],pose,len(cloud),'height_mm',round(height*1000,6),flush=True)
    V.sheet(tiles, out / 'floor_demands.png')
    common_camera = V.render_common(renderer, moved_mesh(fixture, native[0]), clouds,
        [r['color'] for r in rows], out / 'common_fixture_demands.png')
    write_case_index(out, case['id'], rows)
    protected_sources = {r['path']: r['sha256'] for r in all_sources}
    protected_sources.update(selected_source_hashes)
    artifacts = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}
    record = dict(complete=True, schema='coopt_step51_final_position_floor_v1', object=name,
        id=case['id'], poses=poses, pose_count=len(poses), sample_count=sum(r['sample_count'] for r in rows),
        selected_step4_experiment=case['experiment'], selected_step4_source=relative(selected),
        fixture_weight_ratio=weight_ratio, fixture_self_weight_included=bool(weight_ratio),
        demand_definition='whole-system required floor wrench from unchanged COM demands at actual final world position',
        pressure_center_definition='x=-tau_O_y/F_z, y=tau_O_x/F_z, z=0',
        grounded_object_floor_reactions_included_in_total_ground_demand=True,
        includes_object_gravity=True, airborne_demand_arrays_generated=False,
        original_demand_samples_reused=True, force_equilibrium_resolved=False,
        step4_pass_rechecked=False, geometry_changed=False, base_constructed=False,
        base_acceptance_run=False, xy_points_do_not_certify_friction_or_yaw=True,
        original_floor_points_shifted_rigidly=False,
        all_points_in_actual_task_world_z_zero=True,
        common_view_coordinates='one fixed fixture, displayed in native first-reference world orientation',
        common_view_transform=native[0].tolist(),
        common_view_camera=camera_record(common_camera),
        common_cloud_transform='native_world[0] @ inverse(T_world_fixture) @ [x_floor,y_floor,0,1]',
        common_floors_merged_or_projected=False, text_in_images=False,
        work_forbidden_overlay=False, exit_sweep_overlay=False,
        numerical_and_io_seconds=calculation_seconds,
        seconds=time.perf_counter()-began, per_pose=rows,
        provenance=dict(inputs=protected_sources, executed_code=batch_record['executed_code'],
            batch=relative(batch_record['manifest_path'])), artifacts=artifacts)
    save(out / 'data/report.json', record)
    for p, expected in protected_sources.items():
        if digest(ROOT/p) != expected:
            raise ValueError(f'Upstream input changed during Step5.1: {p}')
    print('STEP5.1 SET COMPLETE',case['id'],'seconds',round(record['seconds'],2),flush=True)
    return dict(id=case['id'], pose_count=len(poses), sample_count=record['sample_count'],
        source_experiment=case['experiment'], seconds=record['seconds'],
        numerical_and_io_seconds=calculation_seconds,
        airborne_poses=[r['pose'] for r in rows if r['airborne']],
        output=relative(out), common_image=str((out/'common_fixture_demands.png').relative_to(base)),
        per_pose_image=str((out/'floor_demands.png').relative_to(base)),
        report=relative(out/'data/report.json'), provenance=protected_sources)


def freeze_code(directory):
    paths = {Path(__file__).resolve(), Path(F.__file__).resolve(), Path(V.__file__).resolve(),
             BASE_CPP, V.CPU_PATH.resolve()}
    # Archive loaded repository modules without running any previous stage.
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            path = Path(file).resolve()
            if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / '.venv') and path.is_file():
                paths.add(path)
    sources = {}
    for path in sorted(paths):
        target = directory / 'executed_sources' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[relative(path)] = digest(path)
        if digest(target) != sources[relative(path)]:
            raise ValueError('Execution snapshot copy mismatch')
    return sources


BASE_CPP = ROOT / 'slides/baseline_algo/step4_connect_support/translucent_raster.cpp'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--selection-summary', type=Path)
    parser.add_argument('--sets', nargs='+')
    parser.add_argument('--batch-name', default='step51_xyz_floor_v1')
    parser.add_argument('--fixture-weight-ratio', type=float, default=0.)
    args = parser.parse_args()
    if not np.isfinite(args.fixture_weight_ratio) or args.fixture_weight_ratio < 0:
        parser.error('fixture weight ratio must be finite and nonnegative')
    base = COOPT / 'output' / args.object
    selection_path = (args.selection_summary or base / 'data/stable_gradient_xyz_force_v3/summary.json').resolve()
    selection = json.loads(selection_path.read_text())
    cases = selection['cases']
    if args.sets:
        missing = set(args.sets) - {r['id'] for r in cases}
        if missing:
            parser.error(f'Sets missing from selected results: {sorted(missing)}')
        cases = [r for r in cases if r['id'] in args.sets]
    if not cases:
        parser.error('No selected sets')
    for case in cases:
        out = base / case['id'] / 'step5/step5.1'
        if out.exists() and any(out.iterdir()):
            parser.error(f'Existing Step5.1 result is protected: {out}')
    directory = base / 'data' / args.batch_name
    directory.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    executed_code = freeze_code(directory)
    manifest_path = directory / 'execution.json'
    batch = dict(manifest_path=manifest_path, executed_code=executed_code)
    manifest = dict(schema='coopt_step51_execution_v1', started_utc=datetime.now(timezone.utc).isoformat(),
        selection=source(selection_path), groups=[r['id'] for r in cases], object=args.object,
        fixture_weight_ratio=args.fixture_weight_ratio, executed_code=executed_code,
        step4_optimization_run=False, demand_solver_run=False, geometry_boolean_run=False)
    save(manifest_path, manifest)
    mesh_path = ROOT / 'objects' / args.object / 'mesh.stl'
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    reader = NativeInputs(args.object, mesh)
    renderer = V.Renderer()
    rows = []
    protected = {relative(selection_path): digest(selection_path), relative(mesh_path): digest(mesh_path),
        relative(ROOT/'objects'/args.object/'poses.json'): reader.poses_hash}
    for case in cases:
        row = compute_case(args.object, case, base, reader, renderer, args.fixture_weight_ratio, batch)
        rows.append(row)
        protected.update(row['provenance'])
        save(directory / 'progress.json', dict(complete=False, completed=len(rows), cases=rows))
    for path, expected in protected.items():
        if digest(ROOT/path) != expected:
            raise ValueError(f'Protected upstream input changed: {path}')
    for path, expected in executed_code.items():
        if digest(ROOT/path) != expected:
            raise ValueError(f'Executed source changed during batch: {path}')
        if digest(directory/'executed_sources'/path) != expected:
            raise ValueError(f'Execution archive differs from source: {path}')
    result = dict(complete=True, object=args.object, cases=rows, group_count=len(rows),
        pose_instance_count=sum(r['pose_count'] for r in rows),
        original_point_count=sum(r['sample_count'] for r in rows),
        unique_pose_count=len(reader.rows), airborne_pose_instances=sum(len(r['airborne_poses']) for r in rows),
        fixture_weight_ratio=args.fixture_weight_ratio, protected_inputs_unchanged=True,
        protected_input_count=len(protected), executed_sources_unchanged=True,
        executed_source_count=len(executed_code),
        optimization_run=False, force_solver_run=False, geometry_boolean_run=False,
        base_constructed=False, base_acceptance_run=False,
        total_seconds=time.perf_counter()-began, numerical_and_io_seconds=sum(r['numerical_and_io_seconds'] for r in rows))
    save(directory / 'summary.json', result)
    save(directory / 'verification.json', dict(complete=True, protected_inputs=protected,
        executed_code=executed_code, source_and_snapshot_hashes_match=True, all_upstream_hashes_match=True))
    (directory/'progress.json').unlink()
    cards=''.join(f'<section><h2>{html.escape(r["id"])}</h2><p>{r["pose_count"]} poses, '
        f'{r["sample_count"]:,} original points · <a href="{r["id"]}/step5/step5.1/index.html">Each pose</a></p>'
        f'<img src="{r["common_image"]}"><img src="{r["per_pose_image"]}"></section>' for r in rows)
    gallery=base/'step5.1_index.html'
    gallery.write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 20px;color:#294553}'
        'img{width:100%;height:auto}section{margin:35px 0}</style>'
        '<h1>Step5.1 · Final-position ground demands</h1>'
        '<p>Actual final XYZ placement, unchanged original demands, no resampling or force replay. '
        'Blue fixture, grey object, colored system-floor pressure centers. '
        'Fixture self-weight ratio: '+str(args.fixture_weight_ratio)+'.</p>'+cards+'\n')
    print('STEP5.1 ALL COMPLETE',len(rows),'sets',result['pose_instance_count'],'poses',
          result['original_point_count'],'points','seconds',round(result['total_seconds'],2),flush=True)
    print('GALLERY',relative(gallery),flush=True)


if __name__ == '__main__':
    main()
