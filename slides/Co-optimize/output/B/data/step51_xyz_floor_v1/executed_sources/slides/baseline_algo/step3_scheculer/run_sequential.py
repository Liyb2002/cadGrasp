"""Hardcoded ten-particle pose1(3) -> best shared head -> pose2(+2) search."""
import argparse
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I, terminal_expansion as T, strict_lp_retry
from step3_scheculer import run_pairs as R
from step3_scheculer.pair_tasks import read_task, sample_pairs, pair_folder, input_hashes, prepare_pair_inputs
from step3_scheculer.pair_scoring import C, J, coverage_summary
from step3_scheculer.random_search import chain_seed
from step3_scheculer.sequential_geometry import SequentialGeometry, active_entries
from step3_scheculer.pair_geometry import transform_contact
from step2_local_support import surface as S, circles as P

SCHEMA = 'sequential_three_heads_share_one_add_two_v1'


def output_folder(name, poses, stage, expansion=True):
    return pair_folder(name, poses, stage)/'sequential_3plus2'/f'from_{poses[0]}'/(
        'terminal_expansion' if expansion else 'fixed_only')


def top5(rows, base_count):
    ranked = sorted((r for r in rows if r['eligible']), key=lambda r: (-r['covered_count'], r['id']))[:5]
    gains = np.array([r['covered_count']-base_count for r in ranked], float)
    if np.any(gains < 0):
        raise RuntimeError('Adding a fixed active head reduced coverage')
    probabilities = gains/gains.sum() if gains.sum() else np.full(len(ranked), 1/len(ranked)) if ranked else []
    return [r['index'] for r in ranked], probabilities


def best_shared(rows):
    legal = [r for r in rows if r['eligible']]
    return min(legal, key=lambda r: (-r['covered_count'], r['id'])) if legal else None


@contextmanager
def numerical_recovery(folder, records):
    original = C.W.solve
    def solve(full, target):
        try:
            return original(full, target)
        except RuntimeError as error:
            target = C.U.target(target, full.shape[1])
            token = hashlib.sha256(full.tobytes()+target.tobytes()).hexdigest()
            folder.mkdir(parents=True, exist_ok=True)
            arrays = folder/(token+'.npz')
            np.savez_compressed(arrays, full=full, target=target)
            R.save(folder/(token+'.json'), dict(complete=False, original_error=str(error)))
            witness, evidence = strict_lp_retry.recover(full, target)
            if witness is None:
                separator = C.W.exact_separator(full, target)
                if separator is None:
                    raise RuntimeError('Recovery has no primal or exact dual witness') from error
                evidence['exact_separator'] = separator
            R.save(folder/(token+'.json'), dict(complete=True, original_error=str(error),
                same_equations=True, tolerance_relaxed=False, witness=witness, **evidence,
                artifacts={arrays.name: I.sha256(arrays)}))
            records[token] = evidence['status']
            print('Original-equation recovery', evidence['status'], flush=True)
            return witness
    C.W.solve = solve
    try:
        yield
    finally:
        C.W.solve = original


class SequentialSearch:
    def __init__(self, name, poses, seed, particles=10, count=200, expansion=True):
        self.name, self.poses = name, tuple(poses)
        self.seed, self.particles, self.expansion = seed, particles, expansion
        self.out = output_folder(name, poses, 'step3_scheculer', expansion)
        self.out.mkdir(parents=True, exist_ok=True)
        prepare_pair_inputs(name, poses)
        self.problems = [read_task(name, p, poses) for p in poses]
        self.original_targets = [p.targets.copy() for p in self.problems]
        self.inputs = input_hashes(self.problems)
        self.geometry = SequentialGeometry(self.problems, count)
        self.geometry.save(output_folder(name, poses, 'step2_local_support', expansion), R.save)
        self.scores, self.recoveries = {}, {}

    def task_score(self, entries, task):
        active = active_entries(entries, task)
        key = (task, tuple(sorted((e['contact']['candidate_id'], e['contact']['radius_m']) for e in active)))
        if key not in self.scores:
            contacts = [self.geometry.view(e, task)['contact'] for e in active]
            problem = self.problems[task]
            mask, _ = J.classify(problem.supply(contacts), problem.targets)
            self.scores[key] = dict(mask=mask, covered_count=int(mask.sum()), sample_count=len(mask))
        return self.scores[key]

    def evaluate(self, entries):
        masks = [self.task_score(entries, k)['mask'] for k in range(2)]
        return dict(**coverage_summary(masks), masks=masks, area_m2=sum(e['area'] for e in entries))

    def extend(self, entries, task, additions, rng, folder):
        entries, rounds = list(entries), []
        for step in range(additions):
            base = self.task_score(entries, task)
            rows = []
            for index, candidate in enumerate(self.geometry.pools[task]):
                row = dict(index=index, id=candidate['contact']['candidate_id'], eligible=False)
                if not candidate['valid']:
                    row['reason'] = candidate['reason']
                elif self.geometry.same_center(candidate, entries):
                    row['reason'] = 'center_already_selected'
                else:
                    check = self.geometry.task_check(entries+[candidate], task)
                    row['reason'] = check['reason']
                    if check['passed']:
                        tested = self.task_score(entries+[candidate], task)
                        if np.any(base['mask'] & ~tested['mask']):
                            raise RuntimeError('An already covered sample was lost')
                        row.update(eligible=True, covered_count=tested['covered_count'])
                rows.append(row)
            ids, probabilities = top5(rows, base['covered_count'])
            draw = float(rng.random()) if ids else None
            R.save(folder/f'pose{task+1}_round_{step+1:02d}_scores.json', dict(rows=rows,
                base_count=base['covered_count'], top5_indices=ids, probabilities=probabilities, uniform_draw=draw))
            if not ids:
                break
            pick = min(int(np.searchsorted(np.cumsum(probabilities), draw, side='right')), len(ids)-1)
            entries.append(self.geometry.pools[task][ids[pick]])
            score = self.task_score(entries, task)
            row = dict(step=step+1, selected_id=entries[-1]['contact']['candidate_id'],
                covered_count=score['covered_count'], sample_count=score['sample_count'],
                active_ids=[e['contact']['candidate_id'] for e in active_entries(entries, task)],
                geometry=self.geometry.task_check(entries, task))
            rounds.append(row)
            R.save(folder/f'pose{task+1}_round_{step+1:02d}.json', row)
            print(self.poses, folder.name, 'task', task+1, 'heads', len(row['active_ids']),
                  'coverage', score['covered_count'], '/', score['sample_count'], flush=True)
        return entries, rounds

    def choose_shared(self, entries):
        baseline = self.task_score([], 1)['covered_count']
        rows, proposals = [], []
        for index, entry in enumerate(entries):
            shared = self.geometry.shared(entry)
            row = dict(index=index, id=entry['contact']['candidate_id'],
                       eligible=shared['valid'], reason=shared['reason'])
            if shared['valid']:
                score = self.task_score([shared], 1)
                row.update(covered_count=score['covered_count'], gain=score['covered_count']-baseline)
            rows.append(row)
            proposals.append(shared)
        winner = best_shared(rows)
        if winner is None:
            return entries, dict(candidates=rows, selected_id=None)
        selected = list(entries)
        selected[winner['index']] = proposals[winner['index']]
        return selected, dict(candidates=rows, selected_id=winner['id'],
                              rule='maximum pose2 singleton coverage gain over its original floor contact; ID breaks ties')

    def run(self):
        initial, results, designs = [], [], []
        # Finish all ten pose1 particles first, then transfer one head per chain.
        for particle in range(self.particles):
            folder = self.out/f'particle_{particle:03d}'
            entries, rounds = self.extend([], 0, 3,
                np.random.default_rng(chain_seed(self.seed, 2*particle)), folder)
            initial.append((entries, rounds))
        for particle, (entries, first_rounds) in enumerate(initial):
            folder = self.out/f'particle_{particle:03d}'
            first = self.task_score(entries, 0)
            first_brief = {k: v for k, v in first.items() if k != 'mask'}
            shared, second_rounds = dict(selected_id=None, candidates=[]), []
            # A <=98% pose1 prefix cannot pass the retained terminal rule.
            eligible_prefix = first['covered_count'] == first['sample_count'] or (
                self.expansion and 100*first['covered_count'] > 98*first['sample_count'])
            if len(entries) == 3 and eligible_prefix:
                entries, shared = self.choose_shared(entries)
                R.save(folder/'shared_head.json', shared)
                if shared['selected_id'] is not None:
                    entries, second_rounds = self.extend(entries, 1, 2,
                        np.random.default_rng(chain_seed(self.seed, 2*particle+1)), folder)
            counts = [len(active_entries(entries, k)) for k in range(2)]
            layout_ok = len(entries) == 5 and counts == [3, 3] and sum(len(e['active_tasks']) == 2 for e in entries) == 1
            before = self.evaluate(entries)
            if layout_ok and self.expansion:
                entries, expansion = T.run(self, entries)
            else:
                expansion = dict(attempted=False, status='disabled' if layout_ok else 'incomplete_3plus2_layout',
                                 before=R.brief(before), after=R.brief(before), updates=[], trials=[])
            score = self.evaluate(entries)
            geometry = self.geometry.group_check(entries)
            passed = layout_ok and geometry['passed'] and score['both_sampled_complete']
            status = ('active_contact_sets_passed' if passed else 'pose1_failed' if not eligible_prefix
                      else 'no_valid_shared_head' if shared['selected_id'] is None
                      else 'pose2_candidates_exhausted' if not layout_ok else 'sample_coverage_failed')
            contacts = self.geometry.contacts_by_pose(entries)
            checks, exported = [], []
            for k, (pose, problem, contact) in enumerate(zip(self.poses, self.problems, contacts)):
                path = folder/f'contacts_{pose}.npz'
                I.save_contacts(path, contact)
                restored = I.read_contacts(path)
                exported.append(restored)
                replay, _ = J.classify(problem.supply(restored), problem.targets)
                np.testing.assert_array_equal(replay, score['masks'][k])
                checks.append(C.verify_classification(problem.supply(restored), problem.targets, replay))
            np.savez_compressed(folder/'coverage.npz', **dict(zip(self.poses, score['masks'])))
            if layout_ok:
                one, two = [{c['candidate_id']: c for c in group} for group in exported]
                assert set(one) & set(two) == {shared['selected_id']}
                moved = transform_contact(one[shared['selected_id']], self.geometry.transforms[0][1])
                np.testing.assert_allclose(two[shared['selected_id']]['triangles_m'], moved['triangles_m'], atol=1e-13, rtol=0)
                np.testing.assert_array_equal(one[shared['selected_id']]['source_faces'], two[shared['selected_id']]['source_faces'])
            for e in entries:
                actual_area = float(S.areas(e['contact']['triangles_m']).sum())
                target = self.geometry.local[e['owner_task']].target_area
                np.testing.assert_allclose(actual_area, e['area'], atol=1e-14, rtol=1e-12)
                assert target*(1-P.AREA_REL_TOL) <= actual_area <= target*max(T.AREA_FACTORS)*(1+P.AREA_REL_TOL)
            result = dict(particle=particle, status=status, passed=passed, heads=len(entries),
                selected_ids=[e['contact']['candidate_id'] for e in entries],
                active_ids_by_pose=[[c['candidate_id'] for c in group] for group in contacts],
                shared_head=shared, pose1_stage=first_brief, pose1_rounds=first_rounds, pose2_rounds=second_rounds,
                pose1_seed=chain_seed(self.seed, 2*particle), pose2_seed=chain_seed(self.seed, 2*particle+1),
                **R.brief(score), geometry=geometry, terminal_expansion=expansion,
                independent_sample_checks=checks, exported_masks_reproduced=True,
                head_areas=[dict(id=e['contact']['candidate_id'], area_m2=e['area'],
                                 area_fraction=e['area_fraction'], radius_m=e['contact']['radius_m']) for e in entries])
            R.save(folder/'terminal_expansion.json', expansion)
            R.save(folder/'schedule.json', result)
            results.append(result)
            designs.append(entries)
            R.save(self.out/'progress.json', dict(complete=False, particles_completed=len(results)))
            print(self.poses, 'particle', particle+1, status, score['covered_counts'], flush=True)
        winner = min(range(len(results)), key=lambda i: (not results[i]['passed'], -results[i]['mean_coverage'], i))
        for pose, contacts in zip(self.poses, self.geometry.contacts_by_pose(designs[winner])):
            I.save_contacts(self.out/f'final_contacts_{pose}.npz', contacts)
        code = [Path(__file__), Path(T.__file__), Path(strict_lp_retry.__file__),
                Path(__file__).with_name('sequential_geometry.py'), Path(__file__).with_name('pair_geometry.py'),
                Path(__file__).with_name('pair_scoring.py'), Path(__file__).with_name('pair_tasks.py')]
        report = dict(schema=SCHEMA, complete=True, object=self.name, poses=self.poses, search_seed=self.seed,
            particles=self.particles, candidate_count_per_pose=self.geometry.count, winner=winner, result=results[winner],
            algorithm='pose1 top5 sampling for 3 heads; maximum pose2 contribution shared head; pose2 top5 for 2 new heads',
            pose1_successful_particles=sum(r['pose1_stage']['covered_count'] == r['pose1_stage']['sample_count'] for r in results),
            transferred_particles=sum(r['shared_head']['selected_id'] is not None for r in results),
            successful_particles=sum(r['passed'] for r in results),
            sampled_complete_particles=sum(r['passed'] for r in results),
            unique_successful_combinations=len({tuple(sorted(r['selected_ids'])) for r in results if r['passed']}),
            terminal_expansion_policy=T.policy() if self.expansion else None,
            expansion_attempted_particles=sum(r['terminal_expansion']['attempted'] for r in results),
            scope='active three-head contact sets only; inactive material, relative fixture placement and feet deferred to Step5',
            shared_head_geometry='same unmodified contact surface mapped by the object task transform',
            contact_area_fraction=.01, during_selection_area_optimization_performed=False,
            sample_count_per_pose=32768, continuous_validation_performed=False,
            complete_fixture_verified=False, inactive_parts_are_feet=False,
            numerical_recovery=dict(cases=self.recoveries, tolerance_relaxed=False),
            particle_results=results, candidate_reasons=[dict(Counter(e['reason'] for e in pool)) for pool in self.geometry.pools],
            provenance=dict(inputs=self.inputs, code=I.hashes(code)))
        R.save(self.out/'schedule.json', report)
        R.save(self.out/'progress.json', dict(complete=True, particles_completed=len(results)))
        return report


def run_case(args, poses):
    step4 = output_folder(args.object, poses, 'step0_pose_selection', not args.no_expansion)
    R.save(step4/'status.json', dict(complete=False, status='running'))
    started = time.monotonic()
    try:
        with (step4/'run.log').open('w') as log, redirect_stdout(log):
            search = SequentialSearch(args.object, poses, args.seed, args.particles, args.candidates, not args.no_expansion)
            with numerical_recovery(search.out/'numerical_retries', search.recoveries):
                result = search.run()
            R.PairSearch.floor(search, result, step4)
            from step3_scheculer.draw_sequential import draw
            draw(search.out/'schedule.json', step4)
        elapsed = time.monotonic()-started
        R.save(step4/'status.json', dict(complete=True, status=result['result']['status'], elapsed_seconds=elapsed))
        R.save(step4/'sequential_check.json', dict(passed=True, particles_checked=args.particles,
            exported_masks_recomputed=True, independent_sample_lp_checks=True,
            per_pose_active_sets_only=True, exactly_one_shared_head_in_complete_layouts=True,
            complete_fixture_verified=False, source_schedule_sha256=I.sha256(search.out/'schedule.json')))
        print(poses, 'pose1 passed', result['pose1_successful_particles'], 'transferred', result['transferred_particles'],
              'both passed', result['successful_particles'], '/', args.particles,
              'unique', result['unique_successful_combinations'], 'seconds', round(elapsed, 1), flush=True)
    except Exception as error:
        R.save(step4/'status.json', dict(complete=False, status='error', error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    parser.add_argument('--poses', nargs=2)
    parser.add_argument('--pairs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--particles', type=int, default=10)
    parser.add_argument('--candidates', type=int, default=200)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--no-expansion', action='store_true')
    args = parser.parse_args()
    if min(args.particles, args.candidates, args.workers) < 1:
        parser.error('Positive particle, candidate and worker counts required')
    if args.poses:
        from step1.cases import normalize_pose
        poses = tuple(normalize_pose(p) for p in args.poses)
        if poses[0] == poses[1]:
            parser.error('Distinct poses required')
        run_case(args, poses)
        return
    pairs, _ = sample_pairs(args.object, args.pairs, args.seed)
    print('Sequential ordered pairs:', pairs, 'seed', args.seed, flush=True)
    from concurrent.futures import ThreadPoolExecutor
    def launch(poses):
        command = [sys.executable, str(Path(__file__).resolve()), args.object, '--poses', *poses,
                   '--seed', str(args.seed), '--particles', str(args.particles), '--candidates', str(args.candidates)]
        if args.no_expansion:
            command.append('--no-expansion')
        return subprocess.run(command).returncode
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(launch, pairs))
    if any(codes):
        raise RuntimeError(f'Sequential cases failed: {list(zip(pairs, codes))}')


if __name__ == '__main__':
    main()
