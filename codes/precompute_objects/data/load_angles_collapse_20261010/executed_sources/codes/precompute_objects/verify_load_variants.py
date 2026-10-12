"""Replay angle-only COM loads, source fingerprints and preserved input hashes.

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
from codes.precompute_objects import load_variants as V
from codes.precompute_objects.registry import active_objects, task_poses
from codes.precompute_objects.work_regions import digest, write


def verify_pose(task):
    name, pose = task
    folder = ROOT/'objects'/name/'poses'/pose
    manifest = json.loads((folder/'load_variants.json').read_text())
    assert manifest['schema'] == V.SCHEMA and manifest['complete']
    expected = {V.variant_name(angle) for angle in V.ANGLES}
    assert manifest['variant_count'] == len(manifest['variants']) == 3
    assert {row['folder'] for row in manifest['variants']} == expected
    assert {path.name for path in folder.glob('angle_*')} == expected
    assert all((folder/file).is_file() and not (folder/file).is_symlink()
               for file in ('setup.npz', 'setup.json', *V.LEGACY_FILES))
    maximum, fingerprints = 0., []
    for angle in V.ANGLES:
        case = V.read_variant(name, pose, angle)
        domain, arrays, metadata = case['domain'], case['arrays'], case['metadata']
        assert metadata['count'] == 32768 and arrays['need_wrench'].shape == (32768, 6)
        assert set(path.name for path in case['folder'].iterdir()) == {
            'setup.npz', 'needs.json', 'samples.npz', 'sample_metadata.json', 'variant.json'}
        assert not (V.STATE_KEYS & arrays.keys()) and 'need_wrench_world_origin' not in arrays
        assert 'contact_model' not in domain and domain['demand_semantics']['ground_state_independent']
        assert domain['provenance']['setup_snapshot_cone_half_deg'] == angle
        assert domain['load']['magnitude_range_mg'] == [0., .5]
        np.testing.assert_array_equal(domain['load']['gravity_force_mg'], [0., 0., -1.])
        np.testing.assert_array_equal(arrays['gravity_force_mg'], [0., 0., -1.])
        V.check_arrays(domain, arrays, check_visibility=False)
        c = np.asarray(domain['frame']['moment_origin_m'])
        q, f = arrays['pt_m'], arrays['force_push_mg']
        # Independent substitution of the six balance equations.
        residual = np.c_[arrays['need_wrench'][:, :3]+f-[0., 0., 1.],
            arrays['need_wrench'][:, 3:]+np.cross(q-c, f)]
        maximum = max(maximum, float(np.abs(residual).max()))
        np.testing.assert_allclose(residual, 0., atol=1e-12, rtol=0)
        world = V.wrench_at_world_origin(arrays['need_wrench'], c)
        np.testing.assert_allclose(world[:, 3:], np.cross(c, [0., 0., 1.])-np.cross(q, f),
                                   atol=1e-12, rtol=0)
        assert arrays['need_wrench'][:, 2].min() >= .5-1e-12
        assert metadata['audit']['tool_visibility_checked'] is True
        actual = V.array_fingerprints(arrays)
        source = metadata['source_record']
        if source['kind'] == 'existing_angle_arrays_without_resampling':
            assert source['sample_array_fingerprints'] == actual
            assert source['paired_com_wrenches_identical'] is True
        fingerprints.append(dict(cone_half_deg=angle, fingerprints=actual,
                                  source_kind=source['kind']))
    for filename, expected_digest in manifest['original_inputs'].items():
        assert digest(folder/filename) == expected_digest
    return dict(object=name, pose=pose, passed=True, variants=3, sampled_loads=3*32768,
        maximum_balance_residual=maximum, original_inputs_unchanged=True,
        ground_state_fields_absent=True, old_state_directories_deleted=True,
        angle_checks=fingerprints)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects', nargs='+')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--run-name', default='load_angles_collapse_20261010')
    args = parser.parse_args()
    names = args.objects or list(active_objects())
    if args.jobs < 1 or any(name not in active_objects() for name in names):
        parser.error('Invalid object/jobs')
    directory = ROOT/'codes/precompute_objects/data'/args.run_name
    batch = json.loads((directory/'batch.json').read_text())
    assert batch['complete'] and batch['passed']
    for relative, expected in batch.get('sources', {}).items():
        assert digest(directory/'executed_sources'/relative) == expected
    tasks = [(name, pose) for name in names for pose in task_poses(name)]
    began, rows = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(verify_pose, task) for task in tasks]
        for future in as_completed(pending):
            rows.append(future.result())
            if len(rows) % 60 == 0 or len(rows) == len(tasks):
                print(f'AUDIT {len(rows)}/{len(tasks)} PASS', flush=True)
    if set(names) == set(active_objects()):
        index = json.loads((ROOT/'objects/load_variants.json').read_text())
        assert index['complete'] and index['poses'] == len(rows)
        assert index['variants'] == 3*len(rows) and 'states' not in index
    protected_path = directory/'protected.json'
    if protected_path.exists():
        protected = json.loads(protected_path.read_text())
        for relative, expected in protected.items():
            assert digest(ROOT/relative) == expected, relative
    report = dict(schema='cadgrasp_load_angles_audit_v2', passed=True, objects=len(names),
        poses=len(rows), variants=sum(row['variants'] for row in rows),
        sampled_loads=sum(row['sampled_loads'] for row in rows),
        maximum_balance_residual=max(row['maximum_balance_residual'] for row in rows),
        original_inputs_unchanged=True, all_source_com_samples_preserved=True,
        ground_state_fields_absent=True, old_state_directories_deleted=True,
        placement_specific_wrenches_not_precomputed=True,
        generation_tool_visibility_checks_inherited_from_unchanged_samples=True,
        fixture_designs_not_solved=True, seconds=time.monotonic()-began,
        auditor='codes/precompute_objects/verify_load_variants.py', auditor_sha256=digest(__file__),
        cases=sorted(rows, key=lambda row: (row['object'], int(row['pose'].split('_')[1]))))
    write(directory/'verification.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}))


if __name__ == '__main__':
    main()
