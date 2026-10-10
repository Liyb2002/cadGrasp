"""Precompute 15/30/60-degree, grounded/airborne inputs for saved poses.

Angles are inward force-cone HALF angles. Airborne means that the workpiece
has no direct floor contact; gravity and the grounded fixture remain present.
Old root inputs retain their exact bytes, with aliases into angle_30_grounded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from codes.precompute_objects import loads as L
from codes.precompute_objects.cases import normalize_pose
from codes.precompute_objects.registry import active_objects, task_poses
from codes.precompute_objects.work_regions import digest, write

ANGLES = (15, 30, 60)
STATES = (True, False)
DEFAULT_CLEARANCE_M = .01
ARRAY_KEYS = ('work_face_index', 'parameters', 'pt_m', 'force_push_mg', 'need_wrench')
LEGACY_FILES = ('needs.json', 'samples.json', 'floor_contact.npz')


def variant_name(half_angle_deg, grounded):
    if isinstance(half_angle_deg, (bool, np.bool_)) or half_angle_deg not in ANGLES:
        raise ValueError('Cone half angle must be 15, 30 or 60 degrees')
    if not isinstance(grounded, (bool, np.bool_)):
        raise ValueError('grounded must be an explicit boolean')
    return f'angle_{int(half_angle_deg)}_{"grounded" if grounded else "airborne"}'


def variant_folder(name, pose, half_angle_deg, grounded):
    # Validate names and task membership using the established registry.
    pose = normalize_pose(pose)
    if pose not in task_poses(name):
        raise ValueError(f'{name}/{pose}: unknown saved pose')
    return ROOT / 'objects' / name / 'poses' / pose / variant_name(half_angle_deg, grounded)


def shifted_domain(original, half_angle_deg, grounded, clearance_m, setup_path, source_hash):
    """Translate the complete pose and domain, not just the gravity origin."""
    if not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError('Airborne clearance must be finite and strictly positive')
    variant_name(half_angle_deg, grounded)
    data = copy.deepcopy(original)
    vertices = np.asarray(data['geometry']['vertices_m'], float)
    original_min = float(vertices[:, 2].min())
    if abs(original_min) > 1e-9:
        raise ValueError('The native pose must be grounded before creating variants')
    shift = np.array([0., 0., 0. if grounded else clearance_m-original_min])
    vertices += shift
    data['geometry']['vertices_m'] = vertices.tolist()
    com = np.asarray(data['frame']['moment_origin_m']) + shift
    transform = np.asarray(data['frame']['T_world_mesh'])
    transform[:3, 3] += shift
    data['frame'].update(T_world_mesh=transform.tolist(), moment_origin_m=com.tolist())
    data['load'].update(cone_half_deg=float(half_angle_deg), gravity_application_point_m=com.tolist())
    data['parameters']['theta_rad'] = [0., float(np.deg2rad(half_angle_deg))]
    data['contact_model'] = dict(workpiece_grounded=bool(grounded),
        workpiece_floor_contact_allowed=bool(grounded),
        required_reaction='fixture + workpiece-floor' if grounded else 'fixture only',
        fixture_may_contact_floor=True, gravity_retained=True,
        airborne_clearance_m=0. if grounded else float(clearance_m),
        native_to_variant_translation_m=shift.tolist(),
        force_moment_about='workpiece center of mass',
        floor_reactions_are_unknowns=True,
        ground_contact_is_not_subtracted_from_external_demand=True)
    data['provenance'].update(setup_snapshot=str(setup_path.relative_to(ROOT)),
        setup_snapshot_sha256=source_hash,
        setup_snapshot_cone_half_deg=float(half_angle_deg),
        variant_generator='codes/precompute_objects/load_variants.py',
        variant_generator_sha256=digest(__file__),
        native_setup_snapshot=original['provenance']['setup_snapshot'],
        native_setup_snapshot_cone_half_deg=original['provenance']['setup_snapshot_cone_half_deg'],
        native_setup_snapshot_sha256=original['provenance']['setup_snapshot_sha256'])
    return data, shift


def as_arrays(samples):
    return {key: np.asarray(samples[key], dtype=np.int64 if key == 'work_face_index' else np.float64)
            for key in ARRAY_KEYS}


def wrench_at_world_origin(wrench_com, com):
    result = np.asarray(wrench_com).copy()
    result[:, 3:] += np.cross(com, result[:, :3])
    return result


def check_arrays(domain, arrays, grounded, clearance_m, check_visibility=True):
    """Independent substitution, barycentric/angle bounds and object-ray audit."""
    count = L.DEFAULT_SAMPLE_COUNT
    for key, width in [('parameters', 5), ('pt_m', 3), ('force_push_mg', 3), ('need_wrench', 6)]:
        if arrays[key].shape != (count, width) or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid {key} array')
    indices = arrays['work_face_index']
    if indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Invalid face indices')
    u, v, theta, phi, magnitude = arrays['parameters'].T
    half = np.deg2rad(domain['load']['cone_half_deg'])
    if (np.any(u < 0) or np.any(v < 0) or np.any(u+v > 1+1e-14)
            or np.any(theta < 0) or np.any(theta > half+1e-14)
            or np.any(phi < 0) or np.any(phi >= 2*np.pi)
            or np.any(magnitude < 0) or np.any(magnitude > .5)):
        raise ValueError('Sample outside physical domain')
    geometry = domain['geometry']
    ids = np.asarray(geometry['work_face_ids'])
    if np.any(indices < 0) or np.any(indices >= len(ids)):
        raise ValueError('Sample references a nonworking face')
    vertices, faces = np.asarray(geometry['vertices_m']), np.asarray(geometry['faces'])
    triangles = vertices[faces[ids[indices]]]
    points = triangles[:, 0] + u[:, None]*(triangles[:, 1]-triangles[:, 0])
    points += v[:, None]*(triangles[:, 2]-triangles[:, 0])
    normals = np.asarray(geometry['inward_normals'])[indices]
    e1, e2 = np.asarray(geometry['tangent1'])[indices], np.asarray(geometry['tangent2'])[indices]
    directions = np.cos(theta)[:, None]*normals + np.sin(theta)[:, None]*(
        np.cos(phi)[:, None]*e1 + np.sin(phi)[:, None]*e2)
    np.testing.assert_allclose(points, arrays['pt_m'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(directions*magnitude[:, None], arrays['force_push_mg'], atol=1e-13, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-12, rtol=0)
    com = np.asarray(domain['frame']['moment_origin_m'])
    force = arrays['force_push_mg']
    expected = np.c_[np.array([0., 0., 1.])-force, -np.cross(arrays['pt_m']-com, force)]
    residual = float(np.max(np.abs(expected-arrays['need_wrench'])))
    np.testing.assert_allclose(expected, arrays['need_wrench'], atol=1e-12, rtol=0)
    world = wrench_at_world_origin(arrays['need_wrench'], com)
    # Separate world-origin external balance includes gravity's world moment.
    external_moment = np.cross(com, [0., 0., -1.]) + np.cross(arrays['pt_m'], force)
    np.testing.assert_allclose(world[:, 3:], -external_moment, atol=1e-12, rtol=0)
    wanted_height = 0. if grounded else clearance_m
    if abs(float(vertices[:, 2].min())-wanted_height) > 1e-9:
        raise ValueError('Workpiece height disagrees with grounded state')
    if check_visibility:
        mesh = L.ContinuousNeeds(domain).mesh
        for start in range(0, count, L.SAMPLE_BATCH_SIZE):
            section = slice(start, start+L.SAMPLE_BATCH_SIZE)
            origins = arrays['pt_m'][section]-domain['reachability']['ray_offset_m']*normals[section]
            if mesh.ray.intersects_any(origins, -directions[section]).any():
                raise ValueError('Saved sample is self-occluded')
    return dict(count=count, maximum_balance_residual=residual,
                workpiece_minimum_z_m=float(vertices[:, 2].min()),
                tool_visibility_checked=bool(check_visibility),
                minimum_required_vertical_force_mg=float(arrays['need_wrench'][:, 2].min()))


def atomic_alias(path, target):
    temporary = path.with_name('.'+path.name+'.alias')
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target)
    temporary.replace(path)


def generate_pose(task):
    name, pose, clearance_m, resume = task
    began = time.monotonic()
    folder = ROOT/'objects'/name/'poses'/pose
    manifest_path = folder/'load_variants.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume:
            raise FileExistsError(f'{folder}: variants already exist; use --resume to validate/reuse')
        if manifest['airborne_clearance_m'] != clearance_m:
            raise ValueError('Existing variants use a different airborne height')
        for row in manifest['variants']:
            read_variant(name, pose, row['cone_half_deg'], row['grounded'])
        return dict(object=name, pose=pose, variants=6, sampled_loads=6*L.DEFAULT_SAMPLE_COUNT,
                    passed=True, reused=True, seconds=time.monotonic()-began)
    originals = {file: digest(folder/file) for file in ('setup.npz', 'setup.json', *LEGACY_FILES)}
    original_domain = json.loads((folder/'needs.json').read_text())
    original_samples = json.loads((folder/'samples.json').read_text())
    if original_samples['count'] != L.DEFAULT_SAMPLE_COUNT or original_samples['seed'] != L.DEFAULT_SAMPLE_SEED:
        raise ValueError('Original fixed sample count/seed does not match dataset')
    if original_samples['provenance']['physical_domain_sha256'] != originals['needs.json']:
        raise ValueError('Original samples have a stale domain')
    native_provenance = original_domain['provenance']
    for file, expected in [('setup.npz', native_provenance['setup_snapshot_sha256']),
            ('../..//mesh.stl', native_provenance['mesh_sha256']),
            ('../..//poses.json', native_provenance['poses_sha256'])]:
        if digest(folder/file) != expected:
            raise ValueError(f'{name}/{pose}: stale original {file}')
    if original_domain['load']['cone_half_deg'] != 30.:
        raise ValueError('Original default domain must be the saved 30-degree case')
    with np.load(folder/'setup.npz', allow_pickle=False) as data:
        original_setup = {key: data[key].copy() for key in data.files}
    pivot = original_setup['floor_contact_m'].reshape(1, 3)
    rows = []
    with tempfile.TemporaryDirectory(prefix='.load_variants_', dir=folder) as temporary:
        stage = Path(temporary)
        for angle in ANGLES:
            grounded_dir = stage/variant_name(angle, True)
            grounded_dir.mkdir()
            final_dir = folder/grounded_dir.name
            if angle == 30:
                for file in LEGACY_FILES:
                    os.link(folder/file, grounded_dir/file)
                shutil.copy2(folder/'setup.npz', grounded_dir/'setup.npz')
                ground_domain = original_domain
                samples = original_samples
            else:
                setup = {key: value.copy() for key, value in original_setup.items()}
                setup.update(cone_half_deg=float(angle), grounded=True,
                             airborne_clearance_m=0., workpiece_floor_contact_allowed=True)
                np.savez_compressed(grounded_dir/'setup.npz', **setup)
                ground_domain, _ = shifted_domain(original_domain, angle, True, clearance_m,
                    final_dir/'setup.npz', digest(grounded_dir/'setup.npz'))
                L.save_json(grounded_dir/'needs.json', ground_domain)
                samples = L.sample_needs(L.ContinuousNeeds(ground_domain))
            ground_arrays = as_arrays(samples)
            ground_audit = check_arrays(ground_domain, ground_arrays, True, clearance_m)
            for grounded in STATES:
                destination = stage/variant_name(angle, grounded)
                final_destination = folder/destination.name
                if grounded:
                    domain = ground_domain
                    arrays = ground_arrays
                    shift = np.zeros(3)
                    audit = ground_audit
                else:
                    destination.mkdir()
                    setup = {key: value.copy() for key, value in original_setup.items()}
                    native_min = float(np.min(original_domain['geometry']['vertices_m'], axis=0)[2])
                    shift = np.array([0., 0., clearance_m-native_min])
                    setup['T_world_mesh'][:3, 3] += shift
                    setup['com_m'] += shift
                    setup.update(cone_half_deg=float(angle), grounded=False,
                        floor_contact_m=np.empty((0, 3)), airborne_clearance_m=clearance_m,
                        workpiece_floor_contact_allowed=False)
                    np.savez_compressed(destination/'setup.npz', **setup)
                    domain, shift = shifted_domain(original_domain, angle, False, clearance_m,
                        final_destination/'setup.npz', digest(destination/'setup.npz'))
                    L.save_json(destination/'needs.json', domain)
                    arrays = {key: value.copy() for key, value in ground_arrays.items()}
                    arrays['pt_m'] += shift
                    # Rigid translation preserves moment arms and self-visibility.
                    # Retain identical COM wrench rows, including original 30 deg.
                    audit = check_arrays(domain, arrays, False, clearance_m, check_visibility=False)
                    audit['tool_visibility_basis'] = 'identical rigidly translated grounded point/direction rays'
                    np.testing.assert_array_equal(arrays['need_wrench'], ground_arrays['need_wrench'])
                com = np.asarray(domain['frame']['moment_origin_m'])
                world_wrench = wrench_at_world_origin(arrays['need_wrench'], com)
                np.savez_compressed(destination/'samples.npz', **arrays,
                    need_wrench_world_origin=world_wrench, moment_origin_m=com,
                    gravity_force_mg=np.array([0., 0., -1.]),
                    workpiece_floor_contact_points_m=pivot if grounded else np.empty((0, 3)),
                    grounded=np.asarray(grounded), cone_half_deg=np.asarray(float(angle)),
                    weight_per_sample=np.asarray(1./L.DEFAULT_SAMPLE_COUNT),
                    sample_count=np.asarray(L.DEFAULT_SAMPLE_COUNT), sample_seed=np.asarray(L.DEFAULT_SAMPLE_SEED))
                if not (angle == 30 and grounded):
                    # This is the REQUIRED assembly floor wrench, not an object
                    # floor reaction or a certificate for any fixture footprint.
                    normal = world_wrench[:, 2]
                    xy = np.c_[-world_wrench[:, 4], world_wrench[:, 3]]/normal[:, None]
                    floor = dict(load_wrenches=arrays['need_wrench'],
                        floor_demands_xy_m=xy, total_floor_normal_mg=normal,
                        moment_origin_m=com, need_wrench_world_origin=world_wrench,
                        workpiece_floor_contact_points_m=pivot if grounded else np.empty((0, 3)),
                        workpiece_floor_contact_allowed=np.asarray(grounded))
                    if grounded:
                        floor['original_pivot_m'] = pivot[0]
                    np.savez_compressed(destination/'floor_contact.npz', **floor)
                meta = {key: value for key, value in samples.items() if key not in ARRAY_KEYS and key != 'provenance'}
                meta.update(schema='cadgrasp_load_variant_samples_v1', pose_id=pose,
                    array_file='samples.npz', array_file_sha256=digest(destination/'samples.npz'),
                    moment_origin_m=com.tolist(), cone_half_deg=angle, grounded=grounded,
                    airborne_clearance_m=0. if grounded else clearance_m,
                    physical_domain_file='needs.json', physical_domain_sha256=digest(destination/'needs.json'),
                    paired_states_have_identical_com_wrenches=True,
                    state_changes_reaction_availability_not_external_load=True)
                write(destination/'sample_metadata.json', meta)
                variants = dict(schema='cadgrasp_load_variant_v1', object=name, pose_id=pose,
                    folder=destination.name, cone_half_deg=angle, cone_opening_deg=2*angle,
                    grounded=grounded, airborne_clearance_m=0. if grounded else clearance_m,
                    native_to_variant_translation_m=shift.tolist(),
                    workpiece_floor_contact_allowed=grounded,
                    workpiece_floor_contact_points_m=pivot.tolist() if grounded else [],
                    required_reaction='fixture + workpiece-floor' if grounded else 'fixture only',
                    gravity_retained=True, count=L.DEFAULT_SAMPLE_COUNT, seed=L.DEFAULT_SAMPLE_SEED,
                    preserved_default_bytes=bool(angle == 30 and grounded),
                    sample_arrays='samples.npz', sample_metadata='sample_metadata.json',
                    whole_fixture_verified=False, ground_footprint_verified=False,
                    original_inputs=originals, audit=audit,
                    provenance=dict(generator='codes/precompute_objects/load_variants.py',
                        code_sha256=digest(__file__), load_sampler_sha256=digest(L.__file__),
                        mesh_sha256=native_provenance['mesh_sha256'], poses_sha256=native_provenance['poses_sha256']),
                    artifacts={file: digest(destination/file) for file in
                        ('needs.json', 'setup.npz', 'samples.npz', 'sample_metadata.json', 'floor_contact.npz')})
                write(destination/'variant.json', variants)
                rows.append(dict(folder=destination.name, cone_half_deg=angle, grounded=grounded,
                    count=L.DEFAULT_SAMPLE_COUNT, variant_sha256=digest(destination/'variant.json')))
        # Complete all six cases before publishing. Restore interrupted partial
        # publications only if they match this deterministic generation exactly.
        for row in rows:
            source, target = stage/row['folder'], folder/row['folder']
            if target.exists():
                if digest(target/'variant.json') != row['variant_sha256']:
                    raise FileExistsError(f'Conflicting partial variant: {target}')
                saved = json.loads((target/'variant.json').read_text())
                for file, expected in saved['artifacts'].items():
                    if digest(target/file) != expected:
                        raise ValueError(f'Corrupt partial variant: {target/file}')
            else:
                source.rename(target)
        for file in LEGACY_FILES:
            if digest(folder/file) != originals[file] or digest(folder/'angle_30_grounded'/file) != originals[file]:
                raise ValueError('Default input changed during publication')
            atomic_alias(folder/file, 'angle_30_grounded/'+file)
        write(manifest_path, dict(schema='cadgrasp_pose_load_variants_v1', object=name, pose_id=pose,
            complete=True, cone_half_angles_deg=list(ANGLES), states=['grounded', 'airborne'],
            airborne_clearance_m=clearance_m, variant_count=6,
            count_per_variant=L.DEFAULT_SAMPLE_COUNT, total_sampled_loads=6*L.DEFAULT_SAMPLE_COUNT,
            legacy_default='angle_30_grounded', legacy_aliases=list(LEGACY_FILES),
            original_inputs=originals, variants=rows))
    return dict(object=name, pose=pose, variants=6, sampled_loads=6*L.DEFAULT_SAMPLE_COUNT,
                passed=True, reused=False, seconds=time.monotonic()-began)


def read_variant(name, pose, half_angle_deg, grounded):
    """Return explicit domain/arrays/contact availability; never infer a floor.

    Existing algorithms keep using the byte-identical 30-grounded root aliases.
    Variant-aware solvers must use workpiece_floor_contact_allowed and must
    not append their old unconditional workpiece floor ray in airborne cases.
    """
    folder = variant_folder(name, pose, half_angle_deg, grounded)
    pose_manifest = json.loads((folder.parent/'load_variants.json').read_text())
    row = next(row for row in pose_manifest['variants'] if row['folder'] == folder.name)
    if not pose_manifest['complete'] or digest(folder/'variant.json') != row['variant_sha256']:
        raise ValueError(f'Incomplete or stale variant: {folder}')
    meta = json.loads((folder/'variant.json').read_text())
    if (meta['object'] != name or meta['pose_id'] != normalize_pose(pose)
            or meta['cone_half_deg'] != half_angle_deg or meta['grounded'] is not bool(grounded)):
        raise ValueError('Variant identity mismatch')
    for file, expected in meta['artifacts'].items():
        if digest(folder/file) != expected:
            raise ValueError(f'Stale variant artifact: {folder/file}')
    for file, expected in meta['original_inputs'].items():
        if digest(folder.parent/file) != expected:
            raise ValueError(f'Stale native pose: {folder.parent/file}')
    for file, key in [('mesh.stl', 'mesh_sha256'), ('poses.json', 'poses_sha256')]:
        if digest(folder.parents[2]/file) != meta['provenance'][key]:
            raise ValueError(f'Stale object: {file}')
    domain = json.loads((folder/'needs.json').read_text())
    with np.load(folder/'samples.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if (domain['load']['cone_half_deg'] != half_angle_deg
            or domain['provenance']['setup_snapshot_cone_half_deg'] != half_angle_deg
            or bool(arrays['grounded']) != grounded
            or float(arrays['cone_half_deg']) != half_angle_deg
            or int(arrays['sample_count']) != L.DEFAULT_SAMPLE_COUNT
            or arrays['workpiece_floor_contact_points_m'].shape != ((1, 3) if grounded else (0, 3))):
        raise ValueError('Variant array metadata mismatch')
    return dict(folder=folder, metadata=meta, domain=domain, arrays=arrays,
                workpiece_floor_contact_allowed=bool(grounded),
                workpiece_floor_contact_points_m=arrays['workpiece_floor_contact_points_m'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects', nargs='+')
    parser.add_argument('--poses', nargs='+')
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--airborne-clearance-m', type=float, default=DEFAULT_CLEARANCE_M)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-name', default='load_variants_v1')
    args = parser.parse_args()
    if args.jobs < 1 or not np.isfinite(args.airborne_clearance_m) or args.airborne_clearance_m <= 0:
        parser.error('Positive jobs and finite positive airborne clearance required')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be a single directory name')
    names = args.objects or list(active_objects())
    if len(set(names)) != len(names) or any(name not in active_objects() for name in names):
        parser.error('Choose distinct active objects')
    requested = [normalize_pose(pose) for pose in args.poses] if args.poses else None
    tasks = [(name, pose, args.airborne_clearance_m, args.resume)
             for name in names for pose in task_poses(name) if requested is None or pose in requested]
    if requested and any(pose not in task_poses(name) for name in names for pose in requested):
        parser.error('Unknown requested pose')
    directory = ROOT/'codes/precompute_objects/data'/args.run_name
    directory.mkdir(parents=True, exist_ok=args.resume)
    source_files = [Path(__file__), Path(L.__file__)]
    sources = {str(file.relative_to(ROOT)): digest(file) for file in source_files}
    for file in source_files:
        target = directory/'executed_sources'/file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(file):
            raise ValueError('Cannot mix executed versions in one run')
        if not target.exists():
            shutil.copy2(file, target)
    protected_path = directory/'protected.json'
    if protected_path.exists():
        protected = json.loads(protected_path.read_text())
    else:
        originals = [ROOT/'objects/cases.json', ROOT/'objects/index.json']
        for name in names:
            base = ROOT/'objects'/name
            originals += [file for file in base.iterdir() if file.is_file()]
            for _, pose, _, _ in [task for task in tasks if task[0] == name]:
                originals += [base/'poses'/pose/file for file in ('setup.npz', 'setup.json', *LEGACY_FILES)]
        protected = {str(file.relative_to(ROOT)): digest(file) for file in originals}
        write(protected_path, protected)
    began = time.monotonic()
    batch = dict(schema='cadgrasp_load_variants_batch_v1', complete=False, passed=False,
        objects=names, poses=len(tasks), angles=list(ANGLES), states=['grounded', 'airborne'],
        airborne_clearance_m=args.airborne_clearance_m, options=vars(args), sources=sources, results=[])
    write(directory/'batch.json', batch)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(generate_pose, task): task for task in tasks}
        for future in as_completed(pending):
            task = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(object=task[0], pose=task[1], passed=False, error=repr(error))
            batch['results'].append(result)
            batch['seconds'] = time.monotonic()-began
            write(directory/'batch.json', batch)
            print(f'{len(batch["results"]):3}/{len(tasks)} {task[0]}/{task[1]} '
                  f'{"PASS" if result["passed"] else result["error"]}', flush=True)
    for relative, expected in {**protected, **sources}.items():
        if digest(ROOT/relative) != expected:
            raise ValueError(f'Protected original changed: {relative}')
    passed = all(row['passed'] for row in batch['results'])
    batch.update(complete=True, passed=passed, original_inputs_unchanged=True,
        executed_sources_unchanged=True, variant_count=6*sum(row['passed'] for row in batch['results']),
        sampled_loads=6*L.DEFAULT_SAMPLE_COUNT*sum(row['passed'] for row in batch['results']),
        seconds=time.monotonic()-began)
    write(directory/'batch.json', batch)
    if passed and requested is None and set(names) == set(active_objects()):
        write(ROOT/'objects/load_variants.json', dict(schema='cadgrasp_all_load_variants_v1', complete=True,
            objects=len(names), poses=len(tasks), variants=6*len(tasks), count_per_variant=L.DEFAULT_SAMPLE_COUNT,
            sampled_loads=batch['sampled_loads'], cone_half_angles_deg=list(ANGLES),
            states=['grounded', 'airborne'], airborne_clearance_m=args.airborne_clearance_m,
            legacy_default='angle_30_grounded', pose_manifest='objects/{object}/poses/{pose}/load_variants.json',
            batch_report=str((directory/'batch.json').relative_to(ROOT)),
            verification_scope='sample preprocessing and wrench/contact metadata; no fixture design acceptance'))
    print(json.dumps({key: value for key, value in batch.items() if key not in ('results', 'sources', 'options')}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
