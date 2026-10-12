"""Preserve three angle-only COM demand sets, then delete six state folders.

No resampling. Only code and metadata are archived, never duplicate arrays.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from codes.precompute_objects import load_variants as V
from codes.precompute_objects.cases import normalize_pose
from codes.precompute_objects.registry import active_objects, task_poses
from codes.precompute_objects.work_regions import digest, write

OLD_FOLDERS = tuple(f'angle_{a}_{s}' for a in V.ANGLES for s in ('grounded', 'airborne'))
DEFAULT_RUN = 'load_angles_collapse_20261010'


def atomic_json(path, data):
    temporary = path.with_name(path.name+'.pending')
    write(temporary, data)
    temporary.replace(path)


def detach_legacy_aliases(folder):
    """Keep native file bytes alive before removing former alias targets."""
    for filename in V.LEGACY_FILES:
        path = folder/filename
        if path.is_symlink():
            target = path.resolve(strict=True)
            if target.parent != folder/'angle_30_grounded':
                raise ValueError(f'Unexpected legacy alias: {path}')
            temporary = folder/(filename+'.detached')
            os.link(target, temporary)
            temporary.replace(path)


def archive_metadata(folder, directory):
    destination = directory/'before'/folder.relative_to(ROOT)
    files = ['load_variants.json'] + [f'{name}/{file}' for name in OLD_FOLDERS
                                    for file in ('variant.json', 'sample_metadata.json')]
    for relative in files:
        source, final = folder/relative, destination/relative
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            if digest(final) != digest(source):
                raise ValueError(f'Conflicting archive: {final}')
        else:
            shutil.copyfile(source, final)


def checked_source(folder, row):
    path = folder/row['folder']
    if digest(path/'variant.json') != row['variant_sha256']:
        raise ValueError(f'Stale source metadata: {path}')
    metadata = json.loads((path/'variant.json').read_text())
    for filename, expected in metadata['artifacts'].items():
        if digest(path/filename) != expected:
            raise ValueError(f'Stale source artifact: {path/filename}')
    with np.load(path/'samples.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in V.ARRAY_KEYS}
        origin = saved['moment_origin_m'].copy()
        np.testing.assert_allclose(saved['need_wrench_world_origin'],
            V.wrench_at_world_origin(arrays['need_wrench'], origin), atol=1e-12, rtol=0)
    return path, metadata, arrays, origin


def collapse_pose(task):
    name, pose, run_name = task
    directory = ROOT/'codes/precompute_objects/data'/run_name
    folder = ROOT/'objects'/name/'poses'/pose
    record_path = directory/'cases'/name/pose/'migration.json'
    manifest = json.loads((folder/'load_variants.json').read_text())
    originals = manifest['original_inputs']
    for filename, expected in originals.items():
        if digest(folder/filename) != expected:
            raise ValueError(f'Changed native input: {folder/filename}')
    if manifest['schema'] == V.SCHEMA:
        if not record_path.exists():
            raise ValueError(f'Already migrated by another run: {folder}')
        for angle in V.ANGLES:
            case = V.read_variant(name, pose, angle)
            source = case['metadata']['source_record']
            if source['sample_array_fingerprints'] != V.array_fingerprints(case['arrays']):
                raise ValueError('Migrated samples differ from their source fingerprints')
        record = json.loads(record_path.read_text())
        for old in OLD_FOLDERS:
            if (folder/old).exists():
                shutil.rmtree(folder/old)
        record.update(passed=True, complete=True)
        atomic_json(record_path, record)
        return record
    if manifest['schema'] != 'cadgrasp_pose_load_variants_v1' or not manifest['complete']:
        raise ValueError(f'Unexpected source manifest: {folder}')
    if {r['folder'] for r in manifest['variants']} != set(OLD_FOLDERS):
        raise ValueError('Expected six source folders')
    if any((folder/V.variant_name(angle)).exists() for angle in V.ANGLES):
        raise ValueError(f'Unpublished angle directory requires inspection: {folder}')
    archive_metadata(folder, directory)
    source_rows = {row['folder']: row for row in manifest['variants']}
    rows, proofs = [], []
    with tempfile.TemporaryDirectory(prefix='.collapse_angles_', dir=folder) as temporary:
        stage = Path(temporary)
        for angle in V.ANGLES:
            ground_path, ground_meta, ground, ground_com = checked_source(
                folder, source_rows[f'angle_{angle}_grounded'])
            air_path, air_meta, air, air_com = checked_source(
                folder, source_rows[f'angle_{angle}_airborne'])
            for key in set(V.ARRAY_KEYS)-{'pt_m'}:
                np.testing.assert_array_equal(ground[key], air[key])
            shift = np.asarray(air_meta['native_to_variant_translation_m'])
            np.testing.assert_allclose(air['pt_m'], ground['pt_m']+shift, atol=1e-15, rtol=0)
            np.testing.assert_allclose(air_com, ground_com+shift, atol=1e-15, rtol=0)
            if not ground_meta['audit']['tool_visibility_checked']:
                raise ValueError('Source lacks a completed visibility audit')
            domain = json.loads((ground_path/'needs.json').read_text())
            with np.load(ground_path/'setup.npz', allow_pickle=False) as saved:
                setup = {key: saved[key].copy() for key in saved.files}
            source = dict(kind='existing_angle_arrays_without_resampling',
                source_grounded_folder=ground_path.name, source_airborne_folder=air_path.name,
                grounded_variant_sha256=digest(ground_path/'variant.json'),
                airborne_variant_sha256=digest(air_path/'variant.json'),
                grounded_samples_sha256=digest(ground_path/'samples.npz'),
                airborne_samples_sha256=digest(air_path/'samples.npz'),
                paired_com_wrenches_identical=True, sample_array_fingerprints=V.array_fingerprints(ground))
            new_name = V.variant_name(angle)
            sample_meta = json.loads((ground_path/'sample_metadata.json').read_text())
            rows.append(V.save_angle(stage/new_name, folder/new_name, name, pose, angle,
                domain, setup, ground, sample_meta, originals, source, True))
            with np.load(stage/new_name/'samples.npz', allow_pickle=False) as saved:
                for key in V.ARRAY_KEYS:
                    np.testing.assert_array_equal(saved[key], ground[key])
            proofs.append(dict(cone_half_deg=angle, paired_com_wrenches_identical=True,
                all_five_sample_arrays_preserved_exactly=True,
                source_sample_array_fingerprints=source['sample_array_fingerprints']))
        detach_legacy_aliases(folder)
        for filename, expected in originals.items():
            if digest(folder/filename) != expected or (folder/filename).is_symlink():
                raise ValueError(f'Native file detachment failed: {folder/filename}')
        record = dict(object=name, pose=pose, passed=False, complete=False,
            variants=3, sampled_loads=3*32768, original_inputs_unchanged=True,
            legacy_aliases_replaced_by_independent_files=True, no_resampling=True,
            removed_folders=list(OLD_FOLDERS), checks=proofs)
        atomic_json(record_path, record)
        for row in rows:
            (stage/row['folder']).rename(folder/row['folder'])
        atomic_json(folder/'load_variants.json', V.pose_manifest(name, pose, originals, rows))
        for angle in V.ANGLES:
            V.read_variant(name, pose, angle)
        for old in OLD_FOLDERS:
            shutil.rmtree(folder/old)
        record.update(passed=True, complete=True)
        atomic_json(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects', nargs='+')
    parser.add_argument('--poses', nargs='+')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--run-name', default=DEFAULT_RUN)
    args = parser.parse_args()
    names = args.objects or list(active_objects())
    if args.jobs < 1 or any(name not in active_objects() for name in names):
        parser.error('Invalid object/jobs')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be a single directory name')
    poses = [normalize_pose(pose) for pose in args.poses] if args.poses else None
    tasks = [(name, pose, args.run_name) for name in names for pose in task_poses(name)
             if poses is None or pose in poses]
    if not tasks:
        parser.error('No saved poses match the selection')
    directory = ROOT/'codes/precompute_objects/data'/args.run_name
    directory.mkdir(parents=True, exist_ok=True)
    previous_protected = ROOT/'codes/precompute_objects/data/load_variants_all_20261010/protected.json'
    if not (directory/'protected.json').exists():
        shutil.copyfile(previous_protected, directory/'protected.json')
    sources = {}
    for filename in ('load_variants.py', 'collapse_load_variants.py', 'verify_load_variants.py'):
        relative = 'codes/precompute_objects/'+filename
        path = directory/'executed_sources'/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and digest(path) != digest(ROOT/relative):
            parser.error('Source differs from this run snapshot; select a new run-name')
        shutil.copyfile(ROOT/relative, path)
        sources[relative] = digest(path)
    rows, errors, began = [], [], time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(collapse_pose, task): task for task in tasks}
        for future in as_completed(pending):
            try:
                rows.append(future.result())
            except Exception as error:
                task = pending[future]
                errors.append(dict(object=task[0], pose=task[1], error=repr(error)))
                print('ERROR', task[:2], repr(error), flush=True)
            done = len(rows)+len(errors)
            if done % 30 == 0 or done == len(tasks):
                print(f'COLLAPSE {done}/{len(tasks)} verified={len(rows)} errors={len(errors)}', flush=True)
    batch = dict(schema='cadgrasp_angle_collapse_batch_v2', complete=not errors,
        passed=not errors, objects=names, poses=len(rows), variants=3*len(rows),
        sampled_loads=3*32768*len(rows), original_inputs_unchanged=not errors,
        no_resampling=True, sources=sources, results=rows, errors=errors,
        seconds=time.monotonic()-began)
    atomic_json(directory/'batch.json', batch)
    if not errors and poses is None and set(names) == set(active_objects()):
        atomic_json(ROOT/'objects/load_variants.json', V.global_manifest(names, len(rows),
            str((directory/'batch.json').relative_to(ROOT))))
    print(json.dumps({k: v for k, v in batch.items() if k not in ('results', 'sources')}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
