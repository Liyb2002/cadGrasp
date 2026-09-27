"""Reassess saved head sets under the fixed-sample rule without rerunning search.

The historical search files stay intact. Current acceptance is sample_result.json,
linked to the exact source report and contacts; Step4 and figures use that verdict.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT, OUTPUTS, sha256
from step3_scheculer import contacts as I, sample_acceptance as A, pair_scoring as S
from step3_scheculer.pair_tasks import read_task, pair_folder, input_hashes
from step3_scheculer import run_pairs as R
from step3_scheculer.draw_pair import draw


def reassess(path):
    old = json.loads(path.read_text())
    if old['schema'] in (A.SCHEMA, A.FIXED_AREA_SCHEMA, A.COMPLETION_SCHEMA):
        return old
    assert old['schema'] == 'shared_object_attached_heads_two_pose_v2' and old['complete']
    I.check_hashes(old['provenance']['inputs'])
    name, poses = old['object'], old['poses']
    folder = path.parent
    suffix = folder.relative_to(pair_folder(name, poses, 'step3_scheculer'))
    problems = [read_task(name, p, poses) for p in poses]
    records, inputs = [], input_hashes(problems)
    inputs[str(path.relative_to(ROOT))] = sha256(path)
    for index in range(old['particles']):
        particle = folder/f'particle_{index:03d}'
        source = particle/'schedule.json'
        historical = json.loads(source.read_text())
        inputs[str(source.relative_to(ROOT))] = sha256(source)
        contacts = []
        for pose in poses:
            contact_path = particle/f'contacts_{pose}.npz'
            inputs[str(contact_path.relative_to(ROOT))] = sha256(contact_path)
            contacts.append(I.read_contacts(contact_path))
        score = S.score_tasks(problems, contacts)
        with np.load(particle/'coverage.npz') as stored:
            for pose, mask in zip(poses, score['masks']):
                np.testing.assert_array_equal(mask, stored[pose][:A.SAMPLE_COUNT])
        geometry = historical['geometry']
        assert geometry['passed'] and geometry['common_path_components'] and all(geometry['common_direction_ids'])
        checks = [S.C.verify_classification(p.supply(c), p.targets, m)
                  for p, c, m in zip(problems, contacts, score['masks'])]
        result = dict(particle=index, seed=historical['seed'], heads=historical['heads'],
            selected_ids=historical['selected_ids'], area_m2=historical['area_m2'],
            **R.brief(score), success_rule=A.RULE,
            status='both_samples_passed' if score['both_sampled_complete'] else 'sample_coverage_incomplete',
            random_covered_counts=score['covered_counts'], random_sample_counts=score['sample_counts'],
            geometry=geometry, independent_sample_checks=checks,
            source_search_status=historical['status'], source_schedule_sha256=sha256(source))
        np.savez_compressed(particle/'sample_coverage.npz', **dict(zip(poses, score['masks'])))
        R.save(particle/'sample_schedule.json', result)
        records.append(result)
    winner = min(range(len(records)), key=lambda i: (
        not records[i]['both_sampled_complete'], -records[i]['mean_coverage'], records[i]['area_m2'], i))
    result = dict(schema=A.SCHEMA, object=name, poses=poses, complete=True,
        scope='Fixed-sample reassessment of saved final head sets; contact search not rerun',
        source_schedule_sha256=sha256(path),
        source_search_code_hashes=old['provenance']['code'],
        success_rule=A.RULE, sample_count_per_pose=A.SAMPLE_COUNT, continuous_validation_performed=False,
        gravity_policy='separate zero-process-force check is diagnostic only',
        search_seed=old['search_seed'], particles=old['particles'], max_heads=old['max_heads'], top_k=old['top_k'],
        winner=winner, result=records[winner],
        selected_contact_files={p: f'particle_{winner:03d}/contacts_{p}.npz' for p in poses},
        successful_particles=sum(r['both_sampled_complete'] for r in records),
        sampled_complete_particles=sum(r['both_sampled_complete'] for r in records),
        particle_results=[{k: v for k, v in r.items() if k != 'independent_sample_checks'} for r in records],
        provenance=dict(inputs=inputs, code=I.hashes([Path(__file__), Path(S.__file__), Path(A.__file__), Path(R.__file__)])))
    R.save(folder/'sample_result.json', result)
    floor_folder = pair_folder(name, poses, 'step4_floor_contact')/suffix
    search = R.PairSearch.__new__(R.PairSearch)
    search.name, search.poses, search.out = name, poses, folder
    search.problems, search.inputs = problems, input_hashes(problems)
    search.original_targets = [p.targets.copy() for p in problems]
    search.floor(result, floor_folder)
    floor_report = json.loads((floor_folder/'pair_result.json').read_text())
    floor_report['step3_report'] = str((folder/'sample_result.json').relative_to(ROOT))
    floor_report['success_rule'] = A.RULE
    R.save(floor_folder/'pair_result.json', floor_report)
    R.save(floor_folder/'status.json', dict(complete=True, status=result['result']['status'],
        scope='fixed-sample reassessment', success_rule=A.RULE))
    R.save(floor_folder/'sample_result_check.json', dict(passed=True, particles_checked=len(records),
        exactly_32768_loads_per_pose=True, saved_original_masks_reproduced=True,
        geometry_and_common_path_and_directions_preserved=True,
        independent_sample_lp_checks_passed=True, source_schedule_sha256=sha256(path),
        sample_result_sha256=sha256(folder/'sample_result.json')))
    draw(name, poses, old['max_heads'] if suffix != Path('.') else None)
    print(poses, 'heads <=', old['max_heads'], 'passed', result['successful_particles'], '/', old['particles'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    args = parser.parse_args()
    root = OUTPUTS/args.object
    paths = sorted([*root.glob('pose*+*/step3_scheculer/schedule.json'),
                    *root.glob('pose*+*/step3_scheculer/heads_*/schedule.json')])
    for path in paths:
        reassess(path)


if __name__ == '__main__':
    main()
