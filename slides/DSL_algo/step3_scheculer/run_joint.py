"""Simultaneous N-pose Step3: fixed 1% heads, append only, uncovered-weighted gain."""
import argparse
import json
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.joint_tasks import canonical_poses, prepare_tasks, output_folder
from step3_scheculer.joint_geometry import JointGeometry
from step3_scheculer.joint_prepared import load_prepared, store_prepared, geometry_sources, scoring_sources
from step3_scheculer import joint_workers
from step3_scheculer.joint_bounds import remaining_pool_certificate
from step3_scheculer.joint_scoring import coverage_summary, weighted_gain, top5_distribution
from step3_scheculer.pair_tasks import input_hashes, sample_pairs
from step3_scheculer.pair_scoring import C, J
from step3_scheculer.sequential_geometry import active_entries
from step3_scheculer.random_search import chain_seed
from step3_scheculer.run_pairs import save, brief
from step3_scheculer.run_sequential import numerical_recovery
from step3_scheculer.sample_acceptance import SAMPLE_COUNT, RULE
from step1.registry import task_poses
from step2_local_support.circles import AREA_FRACTION, AREA_REL_TOL
from step0_pose_selection.floor_points import pressure_centers

SCHEMA = 'joint_uncovered_weighted_add_only_v1'


def assignment_key(result):
    """Insertion order does not make a different final task assignment."""
    return tuple(tuple(sorted(ids)) for ids in result['active_ids_by_pose'])


def begin_run(args, poses):
    """A failed rerun must not inherit the previous run's completion evidence.

    Keep existing geometry/data files; only a newly completed manifest can make
    them current again. Invalidate before input loading or candidate generation.
    """
    poses = canonical_poses(poses)
    pending = dict(schema=SCHEMA, complete=False, status='preparing', object=args.object,
                   poses=poses, search_seed=args.seed, particles=args.particles,
                   candidate_count_per_pose=args.candidates, max_heads=args.max_heads)
    step3 = output_folder(args.object, poses, 'step3_scheculer')
    step4 = output_folder(args.object, poses, 'step0_pose_selection')
    save(step3/'schedule.json', pending)
    save(step3/'progress.json', dict(complete=False, particles_completed=0))
    save(step4/'joint_result.json', pending)
    save(step4/'joint_check.json', dict(passed=None, status='not_revalidated', poses=poses))
    if (step4/'review_check.json').exists():
        save(step4/'review_check.json', dict(passed=None, status='not_revalidated', poses=poses))


class JointSearch:
    def __init__(self, name, poses, seed=20260926, particles=10, count=200, max_heads=None, workers=1, resume=False):
        if particles < 1 or count < 1 or seed < 0 or (max_heads is not None and max_heads < 1):
            raise ValueError('Positive search sizes and a nonnegative seed required')
        self.name, self.poses = name, canonical_poses(poses)
        self.seed, self.particles, self.max_heads = seed, particles, max_heads
        self.out = output_folder(name, self.poses, 'step3_scheculer')
        self.out.mkdir(parents=True, exist_ok=True)
        self.problems = prepare_tasks(name, self.poses)
        self.original_targets = [p.targets.copy() for p in self.problems]
        self.inputs = input_hashes(self.problems)
        self.geometry_folder = output_folder(name, self.poses, 'step2_local_support')
        self.geometry = load_prepared(self.geometry_folder, self.problems, count)
        self.prepared_cache_hit = self.geometry is not None
        if self.geometry is None:
            self.geometry = JointGeometry(self.problems, count=count)
        self.initial_scores = (self.geometry.first_scores if
            getattr(self.geometry, 'first_scores_valid', False) else None)
        self.workers, self.pool = workers, None
        self.resume, self.certify_exhaustion = resume, True
        self.scores, self.recoveries = OrderedDict(), {}

    def run_identity(self):
        sources = [Path(__file__).with_name(f) for f in (
            'run_joint.py', 'joint_bounds.py', 'joint_geometry.py', 'joint_prepared.py',
            'joint_workers.py', 'joint_scoring.py', 'joint_tasks.py')]
        return dict(schema=SCHEMA, object=self.name, poses=list(self.poses), seed=self.seed,
            particles=self.particles, count=self.geometry.count, max_heads=self.max_heads,
            inputs=self.inputs, code=I.hashes(sources+geometry_sources()+scoring_sources()),
            candidates={p.name: I.sha256(p) for p in self.geometry_folder.glob('candidates_*')})

    def restore_prefix(self, folder, particle):
        paths = [p for p in folder.glob('round_*.json') if p.stem.removeprefix('round_').isdecimal()]
        paths.sort(key=lambda p: int(p.stem.removeprefix('round_')))
        rounds, previous = [], []
        candidates = {e['contact']['candidate_id']: e for e in self.geometry.initial}
        rng = np.random.default_rng(chain_seed(self.seed, particle))
        for number, path in enumerate(paths, 1):
            record = json.loads(path.read_text())
            if record['step'] != number or record['contacts'][:-1] != previous:
                raise ValueError('Resume prefix is not a contiguous append-only chain')
            scores = json.loads(path.with_name(path.stem+'_scores.json').read_text())
            if float(rng.random()) != scores['uniform_draw']:
                raise ValueError('Resume prefix uses another search seed')
            previous = record['contacts']
            rounds.append(record)
        entries = []
        for record in previous:
            candidate = candidates[record['id']]
            if candidate['contact']['radius_m'] != record['radius_m'] or candidate['area'] != record['area_m2']:
                raise ValueError('Resume candidate geometry changed')
            entries.append(dict(candidate, active_tasks=tuple(record['active_tasks'])))
        if rounds:
            score = self.evaluate(entries)
            np.testing.assert_array_equal(score['covered_counts'], rounds[-1]['score']['covered_counts'])
            if self.geometry.group_check(entries) != rounds[-1]['geometry']:
                raise ValueError('Resume active-group geometry changed')
        return entries, rounds

    def task_score(self, entries, task, known=None):
        active = active_entries(entries, task)
        key = (task, tuple(sorted((e['contact']['candidate_id'], e['contact']['radius_m']) for e in active)))
        if key not in self.scores:
            if known is not None and np.all(known):
                # Zero coefficients on the added head retain every old witness.
                mask = known.copy()
            else:
                contacts = [self.geometry.view(e, task)['contact'] for e in active]
                problem = self.problems[task]
                options = {} if known is None else dict(known_covered=known)
                mask, _ = J.classify(problem.supply(contacts), problem.targets, **options)
            self.scores[key] = mask
            # Bound memory across many tasks, candidates and independent chains.
            if len(self.scores) > 2048:
                self.scores.popitem(last=False)
        self.scores.move_to_end(key)
        return self.scores[key]

    def evaluate(self, entries, known_masks=None):
        masks = [self.task_score(entries, k, None if known_masks is None else known_masks[k])
                 for k in range(len(self.problems))]
        return dict(**coverage_summary(masks), masks=masks,
                    area_m2=float(sum(e['area'] for e in entries)))

    def candidate_row(self, entries, candidate, base, index):
        row = dict(index=index, id=candidate['contact']['candidate_id'], eligible=False)
        proposal, check = self.geometry.propose(entries, candidate)
        row.update(reason=check['reason'], per_pose_geometry=check['per_pose'])
        if proposal is None:
            return row, None
        tested = self.evaluate(entries+[proposal], base['masks'])
        if any(np.any(old & ~new) for old, new in zip(base['masks'], tested['masks'])):
            raise RuntimeError('Adding a head lost an already covered sample')
        row.update(eligible=True, active_tasks=list(proposal['active_tasks']),
                   active_poses=[self.poses[k] for k in proposal['active_tasks']],
                   **brief(tested), **weighted_gain(base['covered_fractions'], tested['covered_fractions']))
        return row, proposal

    def candidate_rows(self, entries, base, particle, step):
        initial = getattr(self, 'initial_scores', None)
        if not entries and initial is not None:
            np.testing.assert_array_equal(initial['base']['covered_counts'], base['covered_counts'])
            rows = initial['rows']
            return rows, [dict(self.geometry.initial[i], active_tasks=tuple(row['active_tasks']))
                          if row['eligible'] else None for i, row in enumerate(rows)]
        total = len(self.geometry.initial)
        rows, proposals = [None]*total, [None]*total
        if getattr(self, 'workers', 1) > 1 and entries:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=self.workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=joint_workers.initialize,
                    initargs=(self.name, self.poses, self.geometry.count))
            futures = [self.pool.submit(joint_workers.score_batch, entries, base, indices.tolist())
                       for indices in np.array_split(np.arange(total), self.workers) if len(indices)]
            completed = 0
            for future in as_completed(futures):
                batch, recovery = future.result()
                self.recoveries.update(recovery)
                for row, active in batch:
                    index = row['index']
                    rows[index] = row
                    proposals[index] = (None if active is None else
                        dict(self.geometry.initial[index], active_tasks=tuple(active)))
                completed += len(batch)
                print(self.poses, 'particle', particle, 'round', step, 'scored', completed, flush=True)
        else:
            for index, candidate in enumerate(self.geometry.initial):
                rows[index], proposals[index] = self.candidate_row(entries, candidate, base, index)
                if (index+1) % 40 == 0:
                    print(self.poses, 'particle', particle, 'round', step, 'scored', index+1, flush=True)
        if not entries and hasattr(self, 'geometry_folder'):
            self.initial_scores = dict(base=brief(base), rows=rows)
            store_prepared(self.geometry_folder, self.problems, self.geometry.count, self.initial_scores)
        return rows, proposals

    def search_particle(self, particle):
        rng = np.random.default_rng(chain_seed(self.seed, particle))
        folder = self.out/f'particle_{particle:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        entries, rounds = (self.restore_prefix(folder, particle) if getattr(self, 'resume', False) else ([], []))
        for _ in rounds:
            rng.random()
        certificate = None
        limit = len(self.geometry.initial) if self.max_heads is None else min(self.max_heads, len(self.geometry.initial))
        score = self.evaluate(entries)
        while True:
            if score['all_sampled_complete']:
                status = 'all_samples_passed'
                break
            if len(entries) >= limit:
                status = ('head_budget_reached' if self.max_heads is not None and len(entries) >= self.max_heads
                          else 'candidates_exhausted')
                break
            step, base = len(entries)+1, score
            rows, proposals = self.candidate_rows(entries, base, particle, step)
            indices, probabilities = top5_distribution(rows, base['covered_fractions'])
            draw = float(rng.random()) if indices else None
            save(folder/f'round_{step:03d}_scores.json', dict(base=brief(base), rows=rows,
                top5_indices=indices, probabilities=probabilities, uniform_draw=draw))
            if not indices:
                status = 'candidates_exhausted'
                break
            if getattr(self, 'certify_exhaustion', False) and rows[indices[0]]['value'] == 0:
                certificate = remaining_pool_certificate(self, entries, proposals, base, folder, step)
                if certificate is not None:
                    status = 'remaining_pool_cannot_cover'
                    break
            pick = min(int(np.searchsorted(np.cumsum(probabilities), draw, side='right')), len(indices)-1)
            index = indices[pick]
            entries.append(proposals[index])
            score = self.evaluate(entries, base['masks'])
            np.testing.assert_array_equal(score['covered_counts'], rows[index]['covered_counts'])
            record = dict(step=step, selected_id=rows[index]['id'], value=rows[index]['value'],
                uncovered_weights=rows[index]['uncovered_weights'],
                coverage_gains=rows[index]['coverage_gains'], value_by_pose=rows[index]['value_by_pose'],
                added_active_poses=rows[index]['active_poses'],
                active_ids_by_pose=[[c['candidate_id'] for c in group]
                                   for group in self.geometry.contacts_by_pose(entries)],
                contacts=[dict(id=e['contact']['candidate_id'], radius_m=e['contact']['radius_m'],
                               area_m2=e['area'], active_tasks=list(e['active_tasks'])) for e in entries],
                score=brief(score), geometry=self.geometry.group_check(entries))
            rounds.append(record)
            save(folder/f'round_{step:03d}.json', record)
            save(self.out/'progress.json', dict(complete=False, particles_completed=particle,
                current_particle=particle, heads=len(entries), poses=self.poses, **brief(score)))
            print(self.poses, 'particle', particle, 'heads', len(entries),
                  'covered', score['covered_counts'], flush=True)
        geometry = self.geometry.group_check(entries)
        return entries, dict(particle=particle, seed=chain_seed(self.seed, particle), status=status,
            passed=score['all_sampled_complete'] and geometry['passed'],
            heads=len(entries), selected_ids=[e['contact']['candidate_id'] for e in entries],
            active_ids_by_pose=[[c['candidate_id'] for c in group]
                               for group in self.geometry.contacts_by_pose(entries)],
            **brief(score), geometry=geometry, rounds=rounds, remaining_pool_certificate=certificate)

    def export_and_check(self, entries, result):
        folder = self.out/f'particle_{result["particle"]:03d}'
        score = self.evaluate(entries)
        checks, artifacts = [], {}
        contacts = self.geometry.contacts_by_pose(entries)
        for k, (problem, group) in enumerate(zip(self.problems, contacts)):
            path = folder/f'contacts_{problem.pose}.npz'
            I.save_contacts(path, group)
            restored = I.read_contacts(path)
            for original, exported in zip(group, restored):
                np.testing.assert_array_equal(original['triangles_m'], exported['triangles_m'])
                area_fraction = I.area([exported])/problem.domain.mesh.area
                if abs(area_fraction/AREA_FRACTION-1) > AREA_REL_TOL+1e-12:
                    raise RuntimeError('Exported head does not have the fixed 1% area')
            full = problem.supply(restored)
            replay, _ = J.classify(full, problem.targets)
            np.testing.assert_array_equal(replay, score['masks'][k])
            np.testing.assert_array_equal(problem.targets, self.original_targets[k])
            checks.append(C.verify_classification(full, problem.targets, replay))
            artifacts[path.name] = I.sha256(path)
        path = folder/'coverage.npz'
        np.savez_compressed(path, **dict(zip(self.poses, score['masks'])))
        artifacts[path.name] = I.sha256(path)
        result.update(independent_sample_checks=checks, exported_masks_reproduced=True,
                      fixed_area_verified=True, artifacts=artifacts)
        save(folder/'schedule.json', result)

    def run(self):
        results, designs = [], []
        if not getattr(self, 'resume', False):
            self.geometry.save(output_folder(self.name, self.poses, 'step2_local_support'), save)
        if hasattr(self, 'geometry_folder'):
            identity = self.run_identity()
            path = self.out/'run_identity.json'
            if self.resume and (not path.exists() or json.loads(path.read_text()) != identity):
                raise ValueError('Cannot resume: inputs, candidates, settings or code changed')
            save(path, identity)
        # Also protect callers that invoke the search directly, without the CLI.
        save(self.out/'schedule.json', dict(schema=SCHEMA, complete=False, status='running',
            object=self.name, poses=self.poses, search_seed=self.seed))
        save(self.out/'progress.json', dict(complete=False, particles_completed=0))
        if getattr(self, 'resume', False):
            self.geometry.save(output_folder(self.name, self.poses, 'step2_local_support'), save)
        for particle in range(self.particles):
            old = self.out/f'particle_{particle:03d}'/'schedule.json'
            previous = json.loads(old.read_text()) if getattr(self, 'resume', False) and old.exists() else None
            if previous is not None and previous['passed']:
                entries, rounds = self.restore_prefix(old.parent, particle)
                if previous['seed'] != chain_seed(self.seed, particle) or not self.evaluate(entries)['all_sampled_complete']:
                    raise ValueError('Saved completed particle did not revalidate')
                result = dict(previous, rounds=rounds, resumed_and_revalidated=True)
            else:
                entries, result = self.search_particle(particle)
            self.export_and_check(entries, result)
            results.append(result)
            designs.append(entries)
            save(self.out/'progress.json', dict(complete=False, particles_completed=len(results)))
        winner = min(range(len(results)), key=lambda i: (not results[i]['passed'], -results[i]['mean_coverage'], i))
        selected = designs[winner]
        artifacts = {}
        for pose, group in zip(self.poses, self.geometry.contacts_by_pose(selected)):
            path = self.out/f'final_contacts_{pose}.npz'
            I.save_contacts(path, group)
            artifacts[path.name] = I.sha256(path)
        if (self.out/'run_identity.json').exists():
            artifacts['run_identity.json'] = I.sha256(self.out/'run_identity.json')
        code = [Path(__file__).with_name(f) for f in (
            'run_joint.py', 'joint_bounds.py', 'joint_geometry.py', 'joint_prepared.py', 'joint_workers.py', 'joint_scoring.py', 'joint_tasks.py',
            'sequential_geometry.py', 'pair_geometry.py', 'pair_scoring.py', 'pair_tasks.py',
            'passive_support.py', 'floor_support.py', 'run_sequential.py', 'strict_lp_retry.py')]
        code += [Path(C.__file__), Path(J.__file__), Path(C.W.__file__)]
        heads = [dict(id=e['contact']['candidate_id'], owner_pose=self.poses[e['owner_task']],
            active_poses=[self.poses[k] for k in e['active_tasks']],
            radius_m=e['contact']['radius_m'], area_m2=e['area'],
            normal_depth_m=self.geometry.local[e['owner_task']].depth) for e in selected]
        report = dict(schema=SCHEMA, complete=True, object=self.name, poses=self.poses,
            search_seed=self.seed, particles=self.particles, top_k=5,
            score_workers=getattr(self, 'workers', 1),
            prepared_geometry_cache_hit=getattr(self, 'prepared_cache_hit', False),
            known_covered_samples_reused=True,
            resumed_search=getattr(self, 'resume', False),
            exact_remaining_pool_certificates_enabled=getattr(self, 'certify_exhaustion', False),
            max_heads=self.max_heads, finite_candidate_bound=len(self.geometry.initial),
            candidate_count_per_pose=self.geometry.count,
            score='mean_k((1 - coverage_before[k]) * (coverage_after[k] - coverage_before[k]))',
            sampling='top5 by weighted value; value-proportional sampling; uniform when all top5 values are zero',
            activation='add the same owner head to every geometrically compatible current task group; freeze assignments after selection',
            selection='append only; no removal, replacement, resizing or terminal expansion',
            shared_head_count_prescribed=False, heads_per_pose_prescribed=False,
            contact_area_fraction=AREA_FRACTION, area_optimization_performed=False,
            continuous_validation_performed=False, sample_count_per_pose=SAMPLE_COUNT,
            success_rule=RULE, complete_fixture_verified=False, inactive_parts_are_feet=False,
            scope='active contact groups only; placements, inactive material, feet and complete bodies remain Step5',
            winner_rule='all tasks passed, then mean coverage, then particle index',
            winner=winner, result=results[winner], heads=heads,
            successful_particles=sum(r['passed'] for r in results),
            unique_successful_assignments=len({assignment_key(r) for r in results if r['passed']}),
            particle_results=[{k: v for k, v in r.items() if k != 'rounds'} for r in results],
            numerical_recovery=dict(cases=self.recoveries, tolerance_relaxed=False),
            provenance=dict(inputs=self.inputs, code=I.hashes(code)), artifacts=artifacts)
        save(self.out/'schedule.json', report)
        save(self.out/'progress.json', dict(complete=True, particles_completed=len(results)))
        return report

    def floor(self, result, folder):
        folder.mkdir(parents=True, exist_ok=True)
        reports = []
        for problem, targets in zip(self.problems, self.original_targets):
            loads = targets/problem.scale
            if len(loads) != SAMPLE_COUNT:
                raise ValueError('Step4 must use exactly the original fixed loads')
            cloud, normal = pressure_centers(loads, problem.domain.com)
            required = loads[:, 3:]+np.cross(problem.domain.com, loads[:, :3])
            supplied = np.cross(np.c_[cloud, np.zeros(len(cloud))], np.c_[np.zeros((len(cloud), 2)), normal])
            np.testing.assert_allclose(supplied[:, :2], required[:, :2], atol=1e-12, rtol=1e-12)
            path = folder/f'floor_contact_{problem.pose}.npz'
            np.savez_compressed(path, load_wrenches=loads, floor_demands_xy_m=cloud,
                total_floor_normal_mg=normal, original_pivot_m=problem.floor, moment_origin_m=problem.domain.com)
            reports.append(dict(pose=problem.pose, load_count=len(loads), arrays=path.name,
                sha256=I.sha256(path), pressure_formula_checked=True, actual_floor_contacts_designed=False))
        save(folder/'joint_result.json', dict(object=self.name, poses=self.poses, complete=True,
            step3_report=str((self.out/'schedule.json').relative_to(I.ROOT)),
            step3_status=result['result']['status'], successful_particles=result['successful_particles'],
            floor=reports, provenance=dict(inputs=self.inputs)))


def run_case(args, poses):
    folder = output_folder(args.object, poses, 'step0_pose_selection')
    started = time.monotonic()
    save(folder/'status.json', dict(complete=False, status='running'))
    try:
        begin_run(args, poses)
        search = JointSearch(args.object, poses, args.seed, args.particles, args.candidates, args.max_heads,
                             workers=getattr(args, 'workers', 1), resume=getattr(args, 'resume', False))
        try:
            with numerical_recovery(search.out/'numerical_retries', search.recoveries):
                result = search.run()
        finally:
            if search.pool is not None:
                search.pool.shutdown()
        search.floor(result, folder)
        from step3_scheculer.draw_joint import draw
        draw(search, result, folder)
        elapsed = time.monotonic()-started
        save(folder/'status.json', dict(complete=True, status=result['result']['status'], elapsed_seconds=elapsed))
        print('JOINT RESULT', search.poses, result['result']['status'], 'passed', result['successful_particles'],
              '/', args.particles, 'heads', result['result']['heads'],
              'coverage', result['result']['covered_counts'], 'seconds', round(elapsed, 2), flush=True)
    except Exception as error:
        save(folder/'status.json', dict(complete=False, status='error', error=str(error)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--poses', nargs='+', help='two or more task poses, evaluated simultaneously')
    selection.add_argument('--all-poses', action='store_true', help='jointly search all registered task poses (default)')
    selection.add_argument('--pairs', type=int, help='explicitly run this many sampled two-pose experiments')
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--particles', type=int, default=10)
    parser.add_argument('--candidates', type=int, default=200, help='candidate centers per pose')
    parser.add_argument('--workers', type=int, default=4, help='independent candidate scoring processes; parent alone samples heads')
    parser.add_argument('--resume', action='store_true', help='revalidate and resume an identical saved search')
    parser.add_argument('--max-heads', type=int, help='optional total search budget; default is finite candidate exhaustion')
    args = parser.parse_args(argv)
    if (min(args.particles, args.candidates, args.workers) < 1 or args.seed < 0 or
            (args.pairs is not None and args.pairs < 1) or
            (args.max_heads is not None and args.max_heads < 1)):
        parser.error('Positive search sizes and a nonnegative seed required')
    try:
        if args.pairs is not None:
            cases, _ = sample_pairs(args.object, args.pairs, args.seed)
        else:
            available = task_poses(args.object)
            if args.poses is None:
                args.poses = available
            cases = [canonical_poses(args.poses)]
            unknown = set(cases[0])-set(available)
            if unknown:
                raise ValueError('Unregistered task poses: '+', '.join(sorted(unknown)))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    for poses in cases:
        print('JOINT SEARCH', args.object, poses, 'particles', args.particles,
              'candidates per pose', args.candidates, 'max heads', args.max_heads, flush=True)
        run_case(args, poses)


if __name__ == '__main__':
    main()
