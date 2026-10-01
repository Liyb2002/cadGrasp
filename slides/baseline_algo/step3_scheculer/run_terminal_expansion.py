"""Apply the terminal step to saved fixed-area chains without rerunning sampling."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import run_pairs as R, sample_acceptance as A, contacts as I
from step3_scheculer import terminal_expansion as T
from step3_scheculer.pair_tasks import read_task, fixed_area_folder, completion_folder, input_hashes
from step3_scheculer.pair_geometry import PairGeometry
from step3_scheculer.draw_pair import draw


def apply_saved(source):
    old = json.loads(source.read_text())
    if old['schema'] != A.FIXED_AREA_SCHEMA or not old['complete']:
        raise ValueError('A completed fixed-area v4 search is required')
    I.check_hashes(old['provenance']['inputs'])
    name, poses = old['object'], old['poses']
    suffix = source.parent.relative_to(fixed_area_folder(name, poses, 'step3_scheculer'))
    step2 = fixed_area_folder(name, poses, 'step2_local_support')/suffix
    cat_path, contacts_path = step2/'shared_candidates.json', step2/'shared_candidates.npz'
    catalogue = json.loads(cat_path.read_text())
    I.check_hashes(catalogue['provenance']['inputs'])
    candidates = {c['candidate_id']: c for c in I.read_contacts(contacts_path)}
    search = R.PairSearch.__new__(R.PairSearch)
    search.name, search.poses = name, tuple(poses)
    search.seed, search.particles, search.max_heads = old['search_seed'], old['particles'], old['max_heads']
    search.out = completion_folder(name, poses, 'step3_scheculer')/suffix
    search.out.mkdir(parents=True, exist_ok=True)
    search.problems = [read_task(name, p, poses) for p in poses]
    for problem in search.problems:
        problem.name = name+'/'+problem.pose
    search.original_targets = [p.targets.copy() for p in search.problems]
    search.inputs = input_hashes(search.problems)
    search.inputs.update(I.hashes([source, cat_path, contacts_path]))
    search.geometry = PairGeometry(search.problems, count=catalogue['candidate_count'], initialize_candidates=False)
    if search.geometry.catalogues != catalogue['direction_catalogues'] or search.geometry.paths.record != catalogue['path_roadmap']:
        raise ValueError('Saved direction/path catalogue no longer matches the task geometry')
    entries_by_id = {}
    for row in catalogue['candidates']:
        if row['valid']:
            contact = candidates[row['id']]
            np.testing.assert_array_equal(contact['center_m'], search.geometry.centers[row['index']])
            assert contact['center_face'] == search.geometry.faces[row['index']]
            assert abs(I.area([contact])/search.geometry.target_area-1) <= R.AREA_REL_TOL
            entries_by_id[row['id']] = dict(row, contact=contact, area=I.area([contact]))
    search.geometry.initial = catalogue['candidates']
    search.scores = {}
    results, designs, checks = [], [], []
    for index in range(search.particles):
        previous = source.parent/f'particle_{index:03d}'
        prior_path = previous/'schedule.json'
        prior = json.loads(prior_path.read_text())
        search.inputs.update(I.hashes([prior_path, previous/'coverage.npz']))
        entries = [entries_by_id[i] for i in prior['selected_ids']]
        before = search.evaluate(entries)
        assert before['covered_counts'] == prior['covered_counts']
        with np.load(previous/'coverage.npz') as masks:
            for pose, mask in zip(poses, before['masks']):
                np.testing.assert_array_equal(mask, masks[pose])
        restored = search.geometry.contacts_by_pose(entries)
        for pose, contacts in zip(poses, restored):
            path = previous/f'contacts_{pose}.npz'
            search.inputs.update(I.hashes([path]))
            saved = I.read_contacts(path)
            assert len(contacts) == len(saved)
            for actual, expected in zip(contacts, saved):
                for field in actual:
                    np.testing.assert_array_equal(actual[field], expected[field])
        entries, expansion = T.run(search, entries)
        score = search.evaluate(entries)
        geometry = search.geometry.group_check(entries)
        assert geometry['passed']
        folder = search.out/f'particle_{index:03d}'
        R.save(folder/'terminal_expansion.json', expansion)
        contacts = search.geometry.contacts_by_pose(entries)
        independent = [R.C.verify_classification(p.supply(c), p.targets, m)
                       for p, c, m in zip(search.problems, contacts, score['masks'])]
        for pose, transformed in zip(poses, contacts):
            I.save_contacts(folder/f'contacts_{pose}.npz', transformed)
        np.savez_compressed(folder/'coverage.npz', **dict(zip(poses, score['masks'])))
        reread = [I.read_contacts(folder/f'contacts_{p}.npz') for p in poses]
        replay = R.score_tasks(search.problems, reread)
        for a, b in zip(replay['masks'], score['masks']):
            np.testing.assert_array_equal(a, b)
        for entry, initial_id in zip(entries, prior['selected_ids']):
            start = entries_by_id[initial_id]['contact']
            np.testing.assert_array_equal(entry['contact']['center_m'], start['center_m'])
            assert entry['contact']['radius_m'] >= start['radius_m']
            assert entry['area'] <= search.geometry.target_area*max(T.AREA_FACTORS)*(1+R.AREA_REL_TOL)
        if not T.eligible(before) or before['both_sampled_complete']:
            assert not expansion['attempted']
            assert score['covered_counts'] == before['covered_counts']
        status = 'both_samples_passed' if score['both_sampled_complete'] else expansion['status']
        result = dict(particle=index, seed=prior['seed'], status=status, heads=len(entries),
            selected_ids=prior['selected_ids'], **R.brief(score),
            random_covered_counts=score['covered_counts'], random_sample_counts=score['sample_counts'],
            rounds=prior['rounds'], source_search_report=str(prior_path.relative_to(I.ROOT)),
            source_search_sha256=I.sha256(prior_path), search_stop_status=prior['status'],
            terminal_expansion=expansion, independent_sample_checks=independent,
            geometry=geometry, success_rule=A.RULE)
        R.save(folder/'schedule.json', result)
        results.append(result)
        designs.append(entries)
        checks.append(dict(particle=index, status=status, before_counts=before['covered_counts'],
                           after_counts=score['covered_counts'], expansion_attempted=expansion['attempted']))
        print(poses, 'chain', index+1, expansion['status'], score['covered_counts'], flush=True)
    result = search.finish(results, designs)
    result['source_fixed_search'] = dict(report=str(source.relative_to(I.ROOT)), sha256=I.sha256(source),
        sampling_rerun=False, original_successful_particles=old['successful_particles'])
    result['provenance']['code'].update(I.hashes([Path(__file__)]))
    R.save(search.out/'schedule.json', result)
    step4 = completion_folder(name, poses, 'step0_pose_selection')/suffix
    search.floor(result, step4)
    R.save(step4/'terminal_expansion_check.json', dict(passed=True, particles_checked=len(checks),
        source_schedule_sha256=I.sha256(source), schedule_sha256=I.sha256(search.out/'schedule.json'),
        original_fixed_masks_reproduced=True, exported_final_masks_recomputed=True,
        threshold_checked_per_pose=True, already_complete_heads_unchanged=True,
        fixed_centers_and_head_count=True, expansion_bounded_to_area_factor=max(T.AREA_FACTORS),
        independent_sample_lp_checks_passed=True, geometry_checks_passed=True,
        no_added_loads=True, particles=checks))
    draw(name, poses, max_heads=old['max_heads'] if suffix != Path('.') else None, terminal_expansion=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    parser.add_argument('--poses', nargs=2)
    parser.add_argument('--max-heads', type=int, default=5)
    args = parser.parse_args()
    if args.poses:
        paths = [fixed_area_folder(args.object, args.poses, 'step3_scheculer')/f'heads_{args.max_heads}'/'schedule.json']
    else:
        paths = sorted((R.OUTPUTS/args.object).glob(f'pose*+*/step3_scheculer/fixed_area_1pct/heads_{args.max_heads}/schedule.json'))
    if not paths:
        raise ValueError('No saved fixed-area searches found')
    for source in paths:
        saved = json.loads(source.read_text())
        step4 = completion_folder(args.object, saved['poses'], 'step0_pose_selection')/f'heads_{args.max_heads}'
        R.save(step4/'status.json', dict(complete=False, status='running', scope='terminal expansion of saved chains'))
        started = time.monotonic()
        try:
            with (step4/'run.log').open('w') as log, redirect_stdout(log):
                result = apply_saved(source)
            elapsed = time.monotonic()-started
            R.save(step4/'timing.json', dict(elapsed_seconds=elapsed, sampling_rerun=False))
            R.save(step4/'status.json', dict(complete=True, status=result['result']['status'], elapsed_seconds=elapsed))
            print(saved['poses'], 'passed', result['successful_particles'], '/', result['particles'],
                  'newly completed', result['expansion_completed_particles'], flush=True)
        except Exception as error:
            R.save(step4/'status.json', dict(complete=False, status='error', error=str(error)))
            raise


if __name__ == '__main__':
    main()
