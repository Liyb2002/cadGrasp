"""Compare the rebuilt Z-up B design with its transported pre-migration reference.

The stored reference already uses Z-up. Check scientific loads and physical
surfaces independently of triangle numbering in the boolean union.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step5_connect_support.surface_check import surface_distances
from step3_scheculer import contacts as I

OUTPUT = Path(__file__).resolve().parent.parent / 'output'
# One micrometre permits boolean retessellation noise; this is a geometry
# comparison threshold, independent of the physical collision tolerances.
SURFACE_TOLERANCE_M = 1e-6


def run(pose):
    case = OUTPUT / 'B' / pose
    out = case / 'step5_connect_support'
    reference_path = out / 'coordinate_reference.npz'
    with np.load(reference_path) as data:
        ref = {k: data[k] for k in data.files}
    assert str(ref['coordinate_system']) == 'z_up_xy_floor'
    samples = json.loads((case / 'step_1_needs/samples.json').read_text())
    schedule = I.check_report(case / 'step3_scheculer/schedule.json')
    connection = I.check_report(out / 'connection.json')
    audit = I.check_report(out / 'audit.json')
    with np.load(out / 'geometry.npz') as data:
        current = {k: data[k] for k in data.files}
    with np.load(case / 'step3_scheculer/final_contacts.npz') as data:
        contacts = {k: data[k] for k in data.files}
    original = trimesh.Trimesh(ref['support_union_vertices_m'], ref['support_union_faces'], process=False)
    rebuilt = trimesh.Trimesh(current['union_vertices_m'], current['union_faces'], process=False)
    # Vertices AND face centroids sample both surfaces. This is a measured
    # discrepancy, not a claimed continuous Hausdorff-distance certificate.
    old_points = np.vstack([original.vertices, original.triangles_center])
    new_points = np.vstack([rebuilt.vertices, rebuilt.triangles_center])
    distances = [float(surface_distances(rebuilt, old_points).max()),
                 float(surface_distances(original, new_points).max())]
    sample_errors = {label: float(np.abs(np.asarray(samples[field]) - ref[key]).max())
                     for label, field, key in [
                         ('position_m', 'pt_m', 'sample_pt'),
                         ('force_mg', 'force_push_mg', 'sample_force'),
                         ('wrench_mg_and_mgm', 'need_wrench', 'sample_wrench')]}
    assert schedule['selected_ids'] == ref['selected_ids'].tolist()
    np.testing.assert_array_equal(schedule['common_withdrawal_directions']['ids'], ref['common_ids'])
    np.testing.assert_array_equal(current['part_labels'], ref['support_part_labels'])
    np.testing.assert_array_equal(contacts['candidate_ids'], ref['contact_candidate_ids'])
    assert max(sample_errors.values()) < 1e-12
    assert max(distances) < SURFACE_TOLERANCE_M
    assert original.is_watertight and rebuilt.is_watertight
    assert original.volume > 0 and rebuilt.volume > 0
    assert schedule['continuous_coverage_proved'] and connection['passed'] and audit['passed']
    report = dict(
        object='B', pose=pose, coordinate_system='z_up_xy_floor', complete=True, passed=True,
        algorithm_reexecuted=True, selected_ids=schedule['selected_ids'],
        selected_ids_unchanged=True, common_direction_ids_unchanged=True,
        common_direction_count=len(ref['common_ids']), part_labels_unchanged=True,
        sample_count=samples['count'], maximum_sample_absolute_error=sample_errors,
        surface_comparison=dict(
            method='Bidirectional vertex and triangle-centroid distances to the other oriented triangle surface',
            continuous_hausdorff_bound_claimed=False,
            maximum_reference_to_rebuilt_m=distances[0],
            maximum_rebuilt_to_reference_m=distances[1],
            comparison_tolerance_m=SURFACE_TOLERANCE_M,
            reference_query_count=len(old_points), rebuilt_query_count=len(new_points)),
        bounds_maximum_absolute_error_m=float(np.abs(original.bounds-rebuilt.bounds).max()),
        reference_volume_m3=float(original.volume), rebuilt_volume_m3=float(rebuilt.volume),
        maximum_contact_center_difference_m=float(np.abs(contacts['centers_m']-ref['contact_centers_m']).max()),
        maximum_contact_radius_difference_m=float(np.abs(contacts['radius_m']-ref['contact_radius_m']).max()),
        continuous_load_and_one_body_insertion_passed=True,
        pre_migration_reference_sha256=str(ref['pre_migration_reference_sha256']),
        provenance=dict(inputs=I.hashes([reference_path, out/'geometry.npz', out/'connection.json', out/'audit.json',
                                        case/'step3_scheculer/schedule.json', case/'step3_scheculer/final_contacts.npz',
                                        case/'step_1_needs/samples.json']),
                        code=I.hashes([Path(__file__), Path(__file__).with_name('surface_check.py')]))
    )
    I.save(out/'coordinate_equivalence.json', report)
    print(pose, 'coordinate equivalence passed;', 'sampled surface discrepancy', max(distances), 'm', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', choices=['pose_1', 'pose_2'])
    args = parser.parse_args()
    for pose in ([args.pose] if args.pose else ['pose_1', 'pose_2']):
        run(pose)
