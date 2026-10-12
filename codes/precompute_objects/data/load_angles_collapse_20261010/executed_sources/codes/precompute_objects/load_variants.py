"""Three angle-only COM demand domains; contact availability belongs to layout.

Native legacy input bytes stay unchanged. No airborne demand copies, fixed
system-floor wrenches or floor availability flags are stored per angle.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
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
ARRAY_KEYS = ('work_face_index', 'parameters', 'pt_m', 'force_push_mg', 'need_wrench')
LEGACY_FILES = ('needs.json', 'samples.json', 'floor_contact.npz')
STATE_KEYS = {'grounded', 'workpiece_grounded', 'workpiece_floor_contact_allowed',
              'workpiece_floor_contact_points_m', 'airborne_clearance_m',
              'native_to_variant_translation_m', 'floor_contact_m'}
SCHEMA = 'cadgrasp_pose_load_angles_v2'

def variant_name(half_angle_deg):
    if isinstance(half_angle_deg, (bool, np.bool_)) or half_angle_deg not in ANGLES:
        raise ValueError('Cone half angle must be 15, 30 or 60 degrees')
    return f'angle_{int(half_angle_deg)}'

def variant_folder(name, pose, half_angle_deg):
    pose = normalize_pose(pose)
    if pose not in task_poses(name):
        raise ValueError(f'{name}/{pose}: unknown saved pose')
    return ROOT/'objects'/name/'poses'/pose/variant_name(half_angle_deg)

def angle_domain(original, half_angle_deg, setup_path, source_hash):
    variant_name(half_angle_deg)
    data = copy.deepcopy(original)
    data.pop('contact_model', None)
    data['load']['cone_half_deg'] = float(half_angle_deg)
    data['parameters']['theta_rad'] = [0., float(np.deg2rad(half_angle_deg))]
    p = data['provenance']
    p.pop('metadata_alignment', None)
    p.update(setup_snapshot=str(setup_path.relative_to(ROOT)), setup_snapshot_sha256=source_hash,
        setup_snapshot_cone_half_deg=float(half_angle_deg),
        native_setup_snapshot=p.get('native_setup_snapshot', original['provenance']['setup_snapshot']),
        native_setup_snapshot_sha256=p.get('native_setup_snapshot_sha256', original['provenance']['setup_snapshot_sha256']),
        native_setup_snapshot_cone_half_deg=30.,
        variant_generator='codes/precompute_objects/load_variants.py',
        variant_generator_sha256=digest(__file__))
    data['demand_semantics'] = dict(moment_origin='workpiece center of mass',
        reference_geometry='native pose; points and COM translate together',
        ground_state_independent=True, gravity_retained=True,
        contact_availability='chosen by the current layout, not by this demand file',
        system_floor_wrench='derive from final world placement in Step5')
    return data

def as_arrays(samples):
    return {key: np.asarray(samples[key], dtype=np.int64 if key == 'work_face_index' else np.float64)
            for key in ARRAY_KEYS}

def wrench_at_world_origin(wrench_com, com):
    result = np.asarray(wrench_com).copy()
    result[:, 3:] += np.cross(com, result[:, :3])
    return result

def array_fingerprints(arrays):
    return {key: hashlib.sha256(np.ascontiguousarray(arrays[key]).tobytes()).hexdigest()
            for key in ARRAY_KEYS}

def check_arrays(domain, arrays, check_visibility=True):
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
    wanted_height = 0.
    if abs(float(vertices[:, 2].min())-wanted_height) > 1e-9:
        raise ValueError('Load reference geometry differs from the native pose')
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


def save_angle(destination, final, name, pose, angle, domain, setup, arrays, sample_metadata,
               originals, source_record, visibility_checked):
    """Save unchanged COM samples with neutral contact/placement semantics."""
    destination.mkdir()
    clean_setup = {key: value for key, value in setup.items() if key not in STATE_KEYS}
    clean_setup['cone_half_deg'] = np.asarray(float(angle))
    np.savez_compressed(destination/'setup.npz', **clean_setup)
    neutral = angle_domain(domain, angle, final/'setup.npz', digest(destination/'setup.npz'))
    write(destination/'needs.json', neutral)
    arrays = {key: np.array(arrays[key], copy=True) for key in ARRAY_KEYS}
    audit = check_arrays(neutral, arrays, check_visibility=False)
    audit.update(tool_visibility_checked=bool(visibility_checked),
        tool_visibility_inherited_from_identical_samples=(bool(visibility_checked)
            and source_record['kind']=='existing_angle_arrays_without_resampling'))
    com = np.asarray(neutral['frame']['moment_origin_m'])
    np.savez_compressed(destination/'samples.npz', **arrays,
        moment_origin_m=com, gravity_force_mg=np.array([0., 0., -1.]),
        cone_half_deg=np.asarray(float(angle)), weight_per_sample=np.asarray(1./L.DEFAULT_SAMPLE_COUNT),
        sample_count=np.asarray(L.DEFAULT_SAMPLE_COUNT), sample_seed=np.asarray(L.DEFAULT_SAMPLE_SEED))
    excluded = STATE_KEYS | set(ARRAY_KEYS) | {
        'provenance', 'paired_states_have_identical_com_wrenches',
        'state_changes_reaction_availability_not_external_load'}
    meta = {key: value for key, value in sample_metadata.items() if key not in excluded}
    meta.update(schema='cadgrasp_angle_samples_v2', object=name, pose_id=pose,
        array_file='samples.npz', array_file_sha256=digest(destination/'samples.npz'),
        count=L.DEFAULT_SAMPLE_COUNT, seed=L.DEFAULT_SAMPLE_SEED,
        physical_domain_file='needs.json', physical_domain_sha256=digest(destination/'needs.json'),
        cone_half_deg=angle, moment_origin_m=com.tolist(), ground_state_independent=True,
        moment_origin='workpiece center of mass', reference_geometry='native pose')
    write(destination/'sample_metadata.json', meta)
    variant = dict(schema='cadgrasp_load_angle_v2', object=name, pose_id=pose,
        folder=final.name, cone_half_deg=angle, cone_opening_deg=2*angle,
        ground_state_independent=True, gravity_retained=True,
        count=L.DEFAULT_SAMPLE_COUNT, seed=L.DEFAULT_SAMPLE_SEED,
        sample_arrays='samples.npz', sample_metadata='sample_metadata.json',
        original_inputs=originals, audit=audit, source_record=source_record,
        sample_array_fingerprints=array_fingerprints(arrays),
        whole_fixture_verified=False,
        provenance=dict(generator='codes/precompute_objects/load_variants.py',
            code_sha256=digest(__file__), load_sampler_sha256=digest(L.__file__),
            mesh_sha256=domain['provenance']['mesh_sha256'],
            poses_sha256=domain['provenance']['poses_sha256']),
        artifacts={file: digest(destination/file) for file in
            ('needs.json', 'setup.npz', 'samples.npz', 'sample_metadata.json')})
    write(destination/'variant.json', variant)
    return dict(folder=final.name, cone_half_deg=angle, count=L.DEFAULT_SAMPLE_COUNT,
                variant_sha256=digest(destination/'variant.json'))

def pose_manifest(name, pose, originals, rows):
    return dict(schema=SCHEMA, object=name, pose_id=pose, complete=True,
        cone_half_angles_deg=list(ANGLES), ground_state_independent=True,
        variant_count=3, count_per_variant=L.DEFAULT_SAMPLE_COUNT,
        total_sampled_loads=3*L.DEFAULT_SAMPLE_COUNT, legacy_default='angle_30',
        legacy_inputs='native root files retained for existing consumers',
        original_inputs=originals, variants=rows)

def global_manifest(names, poses, batch_report):
    return dict(schema='cadgrasp_all_load_angles_v2', complete=True,
        objects=len(names), poses=poses, variants=3*poses,
        count_per_variant=L.DEFAULT_SAMPLE_COUNT, sampled_loads=3*poses*L.DEFAULT_SAMPLE_COUNT,
        cone_half_angles_deg=list(ANGLES), ground_state_independent=True,
        legacy_default='angle_30', pose_manifest='objects/{object}/poses/{pose}/load_variants.json',
        batch_report=batch_report,
        verification_scope='angle-only COM demand preprocessing; contact state chosen by layout')

def read_variant(name, pose, half_angle_deg):
    """Read an angle demand; never infer or add a workpiece floor reaction."""
    folder = variant_folder(name, pose, half_angle_deg)
    manifest = json.loads((folder.parent/'load_variants.json').read_text())
    if manifest['schema'] != SCHEMA or not manifest['complete']:
        raise ValueError('Migrate the six-state inputs to angle-only storage first')
    row = next(r for r in manifest['variants'] if r['folder'] == folder.name)
    if digest(folder/'variant.json') != row['variant_sha256']:
        raise ValueError(f'Stale angle metadata: {folder}')
    meta = json.loads((folder/'variant.json').read_text())
    if (meta['object'] != name or meta['pose_id'] != normalize_pose(pose)
            or meta['cone_half_deg'] != half_angle_deg or not meta['ground_state_independent']):
        raise ValueError('Angle identity mismatch')
    for file, expected in meta['artifacts'].items():
        if digest(folder/file) != expected: raise ValueError(f'Stale angle artifact: {folder/file}')
    for file, expected in meta['original_inputs'].items():
        if digest(folder.parent/file) != expected: raise ValueError(f'Stale native pose: {folder.parent/file}')
    for file, key in [('mesh.stl', 'mesh_sha256'), ('poses.json', 'poses_sha256')]:
        if digest(folder.parents[2]/file) != meta['provenance'][key]:
            raise ValueError(f'Stale object: {file}')
    domain = json.loads((folder/'needs.json').read_text())
    with np.load(folder/'samples.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if (STATE_KEYS & arrays.keys() or 'need_wrench_world_origin' in arrays
            or 'contact_model' in domain or domain['load']['cone_half_deg'] != half_angle_deg
            or int(arrays['sample_count']) != L.DEFAULT_SAMPLE_COUNT
            or float(arrays['cone_half_deg']) != half_angle_deg
            or array_fingerprints(arrays) != meta['sample_array_fingerprints']):
        raise ValueError('Angle array/domain mismatch')
    with np.load(folder/'setup.npz', allow_pickle=False) as saved:
        if STATE_KEYS & set(saved.files): raise ValueError('Contact state leaked into demand setup')
        np.testing.assert_array_equal(saved['T_world_mesh'], domain['frame']['T_world_mesh'])
        np.testing.assert_array_equal(saved['com_m'], domain['frame']['moment_origin_m'])
    if digest(folder/'setup.npz') != domain['provenance']['setup_snapshot_sha256']:
        raise ValueError('Stale demand setup')
    return dict(folder=folder, metadata=meta, domain=domain, arrays=arrays,
                ground_state_independent=True)

def generate_pose(task):
    name, pose, resume = task
    folder=ROOT/'objects'/name/'poses'/pose; manifest_path=folder/'load_variants.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['schema'] != SCHEMA:
            raise ValueError('Use collapse_load_variants to migrate the old six-state folders')
        if not resume: raise FileExistsError('Use --resume to verify/reuse existing angles')
        for angle in ANGLES: read_variant(name, pose, angle)
        return dict(object=name, pose=pose, passed=True, reused=True, variants=3, sampled_loads=3*32768)
    originals={file:digest(folder/file) for file in ('setup.npz','setup.json',*LEGACY_FILES)}
    domain=json.loads((folder/'needs.json').read_text())
    samples=json.loads((folder/'samples.json').read_text())
    with np.load(folder/'setup.npz',allow_pickle=False) as saved:
        setup={key:saved[key].copy() for key in saved.files}
    rows=[]
    with tempfile.TemporaryDirectory(prefix='.load_angles_',dir=folder) as temporary:
        stage=Path(temporary)
        for angle in ANGLES:
            current=copy.deepcopy(domain)
            current['load']['cone_half_deg']=float(angle)
            current['parameters']['theta_rad']=[0.,float(np.deg2rad(angle))]
            data=samples if angle==30 else L.sample_needs(L.ContinuousNeeds(current))
            arrays=as_arrays(data)
            check_arrays(current,arrays,check_visibility=True)
            name_dir=variant_name(angle)
            rows.append(save_angle(stage/name_dir,folder/name_dir,name,pose,angle,current,setup,
                arrays,data,originals,dict(kind='original_rows' if angle==30 else 'new_angle_samples'),True))
        for row in rows: (stage/row['folder']).rename(folder/row['folder'])
        write(manifest_path,pose_manifest(name,pose,originals,rows))
    return dict(object=name,pose=pose,passed=True,reused=False,variants=3,sampled_loads=3*32768)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects',nargs='+');p.add_argument('--poses',nargs='+')
    p.add_argument('--jobs',type=int,default=4);p.add_argument('--resume',action='store_true')
    p.add_argument('--run-name',default='load_angles_v2')
    args=p.parse_args();names=args.objects or list(active_objects())
    if args.jobs<1 or any(name not in active_objects() for name in names): p.error('Invalid jobs/object')
    poses=[normalize_pose(pose) for pose in args.poses] if args.poses else None
    tasks=[(name,pose,args.resume) for name in names for pose in task_poses(name)
           if poses is None or pose in poses]
    directory=ROOT/'codes/precompute_objects/data'/args.run_name
    directory.mkdir(parents=True,exist_ok=args.resume)
    began=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(generate_pose,task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%30==0: print('ANGLES',len(rows),'/',len(tasks),flush=True)
    batch=dict(complete=True,passed=True,objects=names,poses=len(rows),variants=3*len(rows),
        sampled_loads=3*32768*len(rows),results=rows,seconds=time.monotonic()-began)
    write(directory/'batch.json',batch)
    if poses is None and set(names)==set(active_objects()):
        write(ROOT/'objects/load_variants.json',global_manifest(names,len(rows),
            str((directory/'batch.json').relative_to(ROOT))))
    print('COMPLETE',len(rows),'poses',3*len(rows),'angles',flush=True)

if __name__=='__main__': main()
