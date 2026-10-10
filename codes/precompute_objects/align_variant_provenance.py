"""Historical six-state migration: correct inherited setup-angle labels.

Preserve prior JSONs and the earlier audit. Existing default inputs, sample
arrays, setups and floor arrays remain byte-identical. New generation already
uses the corrected labels; this migration applies to the initial completed run.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codes.precompute_objects.registry import active_objects, task_poses
from codes.precompute_objects.work_regions import digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='load_variants_all_20261010')
    args = parser.parse_args()
    for name in active_objects():
        for pose in task_poses(name):
            source = ROOT/'objects'/name/'poses'/pose/'load_variants.json'
            if json.loads(source.read_text())['schema'] != 'cadgrasp_pose_load_variants_v1':
                parser.error('This historical migration applies only to six-state v1 inputs')
    directory = ROOT/'codes/precompute_objects/data'/args.run_name
    archive = directory/'provenance_before_alignment'
    archive.mkdir(exist_ok=True)
    old_audit = directory/'verification.json'
    if old_audit.exists() and not (archive/'verification.json').exists():
        shutil.copy2(old_audit, archive/'verification.json')
    snapshot = directory/'metadata_alignment_sources'/Path(__file__).relative_to(ROOT)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copy2(__file__, snapshot)
    else:
        assert digest(snapshot) == digest(__file__)
    changed, preserved_arrays = 0, 0
    for name in active_objects():
        for pose in task_poses(name):
            folder = ROOT/'objects'/name/'poses'/pose
            manifest_path = folder/'load_variants.json'
            manifest = json.loads(manifest_path.read_text())
            backup_pose = archive/name/pose
            backup_pose.mkdir(parents=True, exist_ok=True)
            if not (backup_pose/'load_variants.json').exists():
                shutil.copy2(manifest_path, backup_pose/'load_variants.json')
            for row in manifest['variants']:
                case = folder/row['folder']
                if row['folder'] == 'angle_30_grounded':
                    continue
                variant = json.loads((case/'variant.json').read_text())
                if 'metadata_alignment' in variant:
                    for file, expected in variant['artifacts'].items():
                        assert digest(case/file) == expected
                    continue
                assert digest(case/'variant.json') == row['variant_sha256']
                for file, expected in variant['artifacts'].items():
                    assert digest(case/file) == expected
                backup = backup_pose/row['folder']
                backup.mkdir(exist_ok=True)
                for file in ('needs.json', 'sample_metadata.json', 'variant.json'):
                    if not (backup/file).exists():
                        shutil.copy2(case/file, backup/file)
                domain = json.loads((case/'needs.json').read_text())
                numerical_before = copy.deepcopy(domain)
                numerical_before.pop('provenance')
                previous_angle = domain['provenance']['setup_snapshot_cone_half_deg']
                # The old field was inherited from the original setup. State
                # that source separately and make the active field match setup.
                domain['provenance'].update(
                    native_setup_snapshot_cone_half_deg=30.,
                    setup_snapshot_cone_half_deg=float(row['cone_half_deg']),
                    metadata_alignment=dict(generator='codes/precompute_objects/align_variant_provenance.py',
                        code_sha256=digest(__file__), previous_snapshot_cone_half_deg=previous_angle,
                        previous_domain_sha256=variant['artifacts']['needs.json']))
                numerical_after = copy.deepcopy(domain)
                numerical_after.pop('provenance')
                assert numerical_before == numerical_after
                write(case/'needs.json', domain)
                sample_meta = json.loads((case/'sample_metadata.json').read_text())
                sample_meta['physical_domain_sha256'] = digest(case/'needs.json')
                write(case/'sample_metadata.json', sample_meta)
                variant['metadata_alignment'] = dict(generator='codes/precompute_objects/align_variant_provenance.py',
                    code_sha256=digest(__file__), previous_variant_sha256=row['variant_sha256'],
                    numerical_domains_and_arrays_unchanged=True,
                    previous_jsons=str(backup.relative_to(ROOT)))
                for file in ('needs.json', 'sample_metadata.json'):
                    variant['artifacts'][file] = digest(case/file)
                for file in ('samples.npz', 'setup.npz', 'floor_contact.npz'):
                    assert digest(case/file) == variant['artifacts'][file]
                    preserved_arrays += 1
                write(case/'variant.json', variant)
                row['variant_sha256'] = digest(case/'variant.json')
                changed += 1
            write(manifest_path, manifest)
            for file, expected in manifest['original_inputs'].items():
                assert digest(folder/file) == expected
    report = dict(passed=True, metadata_cases_aligned=changed, numerical_array_files_preserved=preserved_arrays,
                  original_default_inputs_unchanged=True, numerical_domains_unchanged=True,
                  code_sha256=digest(__file__), prior_metadata=str(archive.relative_to(ROOT)))
    write(directory/'metadata_alignment.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
