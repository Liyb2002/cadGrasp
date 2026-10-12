"""Independently replay every published load row and grounded/airborne pair.

No pose search, new load sampling or fixture optimization is performed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from codes.precompute_objects.load_variants import ANGLES, read_variant
from codes.precompute_objects.registry import active_objects, task_poses
from codes.precompute_objects.work_regions import digest, write


def verify_pose(task):
    name, pose = task
    folder = ROOT/'objects'/name/'poses'/pose
    manifest = json.loads((folder/'load_variants.json').read_text())
    assert manifest['variant_count'] == len(manifest['variants']) == 6
    assert {row['folder'] for row in manifest['variants']} == {
        f'angle_{angle}_{state}' for angle in ANGLES for state in ('grounded', 'airborne')}
    default = json.loads((folder/'samples.json').read_text())
    maximum = 0.
    pairs = []
    for angle in ANGLES:
        ground = read_variant(name, pose, angle, True)
        air = read_variant(name, pose, angle, False)
        for case, grounded in [(ground, True), (air, False)]:
            domain, arrays, metadata = case['domain'], case['arrays'], case['metadata']
            assert metadata['count'] == 32768 and arrays['need_wrench'].shape == (32768, 6)
            assert bool(arrays['grounded']) == grounded
            assert domain['load']['cone_half_deg'] == angle
            assert domain['provenance']['setup_snapshot_cone_half_deg'] == angle
            assert domain['load']['magnitude_range_mg'] == [0., .5]
            np.testing.assert_array_equal(domain['load']['gravity_force_mg'], [0., 0., -1.])
            c = np.asarray(domain['frame']['moment_origin_m'])
            q, f = arrays['pt_m'], arrays['force_push_mg']
            # Direct six balance equations, independent of the generating map.
            residual = np.c_[arrays['need_wrench'][:, :3]+f-[0., 0., 1.],
                arrays['need_wrench'][:, 3:]+np.cross(q-c, f)]
            maximum = max(maximum, float(np.abs(residual).max()))
            np.testing.assert_allclose(residual, 0., atol=1e-12, rtol=0)
            world_moment = np.cross(c, [0., 0., 1.])-np.cross(q, f)
            np.testing.assert_allclose(arrays['need_wrench_world_origin'][:, 3:],
                world_moment, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(arrays['need_wrench_world_origin'][:, :3],
                                          arrays['need_wrench'][:, :3])
            normals = np.asarray(domain['geometry']['inward_normals'])[arrays['work_face_index']]
            magnitude = np.linalg.norm(f, axis=1)
            assert np.all((magnitude >= 0.) & (magnitude <= .5+1e-13))
            cos_angle = np.sum(normals*f, axis=1)/magnitude
            assert np.all(cos_angle >= np.cos(np.deg2rad(angle))-1e-12)
            np.testing.assert_allclose(magnitude, arrays['parameters'][:, 4], atol=1e-13, rtol=0)
            with np.load(case['folder']/'setup.npz', allow_pickle=False) as setup:
                np.testing.assert_array_equal(setup['T_world_mesh'], domain['frame']['T_world_mesh'])
                np.testing.assert_array_equal(setup['com_m'], c)
                assert float(setup['cone_half_deg']) == angle
                if grounded:
                    assert setup['floor_contact_m'].shape == (3,)
                    assert abs(setup['floor_contact_m'][2]) < 1e-9
                else:
                    assert setup['floor_contact_m'].shape == (0, 3)
                    assert not bool(setup['workpiece_floor_contact_allowed'])
            assert case['workpiece_floor_contact_points_m'].shape == ((1, 3) if grounded else (0, 3))
            assert case['workpiece_floor_contact_allowed'] == grounded
            vertices = np.asarray(domain['geometry']['vertices_m'])
            np.testing.assert_allclose(vertices[:, 2].min(),
                0. if grounded else manifest['airborne_clearance_m'], atol=1e-9, rtol=0)
            with np.load(case['folder']/'floor_contact.npz', allow_pickle=False) as floor:
                xy = np.c_[-world_moment[:, 1], world_moment[:, 0]]/arrays['need_wrench'][:, 2, None]
                np.testing.assert_allclose(floor['floor_demands_xy_m'], xy, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(floor['load_wrenches'], arrays['need_wrench'])
                if not grounded:
                    assert 'original_pivot_m' not in floor.files
                    assert not bool(floor['workpiece_floor_contact_allowed'])
            # Actual full ray checks already happened before publication. The
            # paired rigid translation is checked below, preserving visibility.
            if grounded:
                assert metadata['audit']['tool_visibility_checked'] is True
            assert arrays['need_wrench'][:, 2].min() >= .5-1e-12
        a, g = air['arrays'], ground['arrays']
        shift = np.asarray(air['metadata']['native_to_variant_translation_m'])
        np.testing.assert_array_equal(a['need_wrench'], g['need_wrench'])
        np.testing.assert_array_equal(a['force_push_mg'], g['force_push_mg'])
        np.testing.assert_array_equal(a['parameters'], g['parameters'])
        np.testing.assert_array_equal(a['work_face_index'], g['work_face_index'])
        np.testing.assert_allclose(a['pt_m'], g['pt_m']+shift, atol=1e-15, rtol=0)
        np.testing.assert_allclose(air['domain']['geometry']['vertices_m'],
            np.asarray(ground['domain']['geometry']['vertices_m'])+shift, atol=1e-15, rtol=0)
        np.testing.assert_allclose(a['need_wrench_world_origin'][:, 3:]-g['need_wrench_world_origin'][:, 3:],
            np.cross(shift, g['need_wrench'][:, :3]), atol=1e-12, rtol=0)
        if angle == 30:
            np.testing.assert_array_equal(g['need_wrench'], default['need_wrench'])
            np.testing.assert_array_equal(g['pt_m'], default['pt_m'])
            np.testing.assert_array_equal(g['force_push_mg'], default['force_push_mg'])
        pairs.append(dict(cone_half_deg=angle, paired_com_wrenches_identical=True,
                          airborne_object_floor_contact_count=0))
    for file, expected in manifest['original_inputs'].items():
        assert digest(folder/file) == expected
    return dict(object=name, pose=pose, passed=True, variants=6, sampled_loads=6*32768,
                maximum_balance_residual=maximum, pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects', nargs='+')
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--run-name', default='load_variants_all_20261010')
    args = parser.parse_args()
    names = args.objects or list(active_objects())
    directory = ROOT/'codes/precompute_objects/data'/args.run_name
    batch = json.loads((directory/'batch.json').read_text())
    assert batch['complete'] and batch['passed']
    for relative, expected in batch['sources'].items():
        assert digest(directory/'executed_sources'/relative) == expected
    tasks = [(name, pose) for name in names for pose in task_poses(name)]
    began, rows = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(verify_pose, task) for task in tasks]
        for future in as_completed(pending):
            rows.append(future.result())
            if len(rows) % 60 == 0 or len(rows) == len(tasks):
                print(f'AUDIT {len(rows)}/{len(tasks)} PASS', flush=True)
    protected = json.loads((directory/'protected.json').read_text())
    for relative, expected in protected.items():
        assert digest(ROOT/relative) == expected, relative
    report = dict(schema='cadgrasp_load_variants_audit_v1', passed=True, objects=len(names),
        poses=len(rows), variants=sum(row['variants'] for row in rows),
        sampled_loads=sum(row['sampled_loads'] for row in rows),
        maximum_balance_residual=max(row['maximum_balance_residual'] for row in rows),
        original_inputs_unchanged=True, all_com_wrench_pairs_identical=True,
        executed_source_snapshots_verified=True,
        all_airborne_workpiece_floor_contacts_removed=True,
        ground_origin_moments_and_required_pressure_centers_checked=True,
        generation_tool_visibility_checks_verified=True,
        fixture_designs_not_solved=True, seconds=time.monotonic()-began,
        auditor='codes/precompute_objects/verify_load_variants.py', auditor_sha256=digest(__file__),
        cases=sorted(rows, key=lambda row: (row['object'], int(row['pose'].split('_')[1]))))
    write(directory/'verification.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}))


if __name__ == '__main__':
    main()
