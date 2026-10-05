"""Fixed 1% head selection, optional >98% terminal expansion, then floor demands.

One shared object-attached head set is scored in two fixed task poses. No dock,
connector solid, independently reoriented fixture, or robot motion is generated.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT, OUTPUTS, sha256
from step1.cases import selected_pose
from step3_scheculer import contacts as I
from step3_scheculer import sample_acceptance as A
from step3_scheculer import terminal_expansion as T
from step3_scheculer.pair_tasks import read_task, sample_pairs, completion_folder as pair_folder, canonical_pair, input_hashes, prepare_pair_inputs
from step3_scheculer.pair_geometry import PairGeometry
from step2_local_support.circles import AREA_FRACTION, AREA_REL_TOL
from step3_scheculer.pair_scoring import score_tasks, top5_distribution, C
from step3_scheculer.random_search import chain_seed
from step4_floor_contact.whole_assembly import pressure_centers


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=convert)+'\n')
    tmp.replace(path)


def brief(score):
    return {k: v for k, v in score.items() if k not in ('masks', 'classifiers', 'gravity')}


def shape_key(entries):
    return tuple(sorted((e['contact']['candidate_index'], e['contact']['radius_m']) for e in entries))


class PairSearch:
    def __init__(self, name, poses, seed, particles=10, max_heads=3, count=200):
        self.name, self.poses = name, canonical_pair(poses)
        self.seed, self.particles, self.max_heads = seed, particles, max_heads
        self.out = pair_folder(name, poses, 'step3_scheculer')
        self.out.mkdir(parents=True, exist_ok=True)
        prepare_pair_inputs(name, self.poses)
        self.problems = [read_task(name, p, self.poses) for p in self.poses]
        for problem in self.problems:
            problem.name = name+'/'+problem.pose
        self.original_targets = [p.targets.copy() for p in self.problems]
        self.inputs = input_hashes(self.problems)
        self.geometry = PairGeometry(self.problems, count=count)
        self.scores = {}
        folder = pair_folder(name, poses, 'step2_local_support')
        folder.mkdir(parents=True, exist_ok=True)
        I.save_contacts(folder/'shared_candidates.npz', [e['contact'] for e in self.geometry.initial
                                                        if len(e['contact']['triangles_m'])])
        save(folder/'shared_candidates.json', dict(object=name, poses=self.poses,
            coordinate_frame=f'{self.poses[0]} world; same object-attached material in both tasks',
            normal_depth_m=self.geometry.depth, candidate_count=count,
            target_area_fraction=AREA_FRACTION, area_relative_tolerance=AREA_REL_TOL,
            area_policy='fixed target area; reject unavailable or invalid geometry, never shrink to rescue',
            sampling=self.geometry.sampling, path_roadmap=self.geometry.paths.record,
            direction_catalogues=self.geometry.catalogues,
            transforms_from_first_task=self.geometry.transforms,
            work_face_policy='exclude union of both task work surfaces',
            candidates=[dict(id=e['contact']['candidate_id'], index=e['contact']['candidate_index'],
                radius_m=e['contact']['radius_m'], area_m2=e['area'], valid=e['valid'], reason=e['reason'],
                area_fraction=e['area_fraction'], relative_area_error=e['relative_area_error'], area_fit=e['area_fit'],
                path_components=e.get('path_components'), directions=e.get('directions')) for e in self.geometry.initial],
            rejection_counts=dict(Counter(e['reason'] for e in self.geometry.initial)),
            provenance=dict(inputs=self.inputs)))

    def evaluate(self, entries):
        key = shape_key(entries)
        if key not in self.scores:
            score = score_tasks(self.problems, self.geometry.contacts_by_pose(entries))
            score['area_m2'] = float(sum(e['area'] for e in entries))
            self.scores[key] = score
        return self.scores[key]

    def candidate_row(self, entries, candidate):
        contact = candidate['contact']
        row = dict(index=contact['candidate_index'], id=contact['candidate_id'],
                   eligible=False, covered_fractions=[0., 0.])
        if contact['candidate_index'] in {e['contact']['candidate_index'] for e in entries}:
            row['reason'] = 'already_selected'
            return row
        proposed = entries+[candidate]
        check = self.geometry.group_check(proposed)
        row['reason'] = candidate['reason'] if not candidate['valid'] else check['reason']
        if check['passed']:
            # Mechanical completion is required only for the final design.
            # A head improving either task remains eligible for joint sampling.
            row.update(brief(self.evaluate(proposed)), eligible=True, reason='scored_joint')
        return row

    def run(self):
        results, designs = [], []
        first_scoring = None
        for particle in range(self.particles):
            for p, targets in zip(self.problems, self.original_targets):
                p.targets = targets.copy()
            rng = np.random.default_rng(chain_seed(self.seed, particle))
            folder = self.out/f'particle_{particle:03d}'
            entries, rounds = [], []
            status = 'contact_limit_reached'
            for step in range(1, self.max_heads+1):
                base = self.evaluate(entries)
                rows = []
                if step == 1 and first_scoring is not None:
                    rows = first_scoring
                else:
                    for candidate in self.geometry.initial:
                        rows.append(self.candidate_row(entries, candidate))
                        if len(rows) % 40 == 0:
                            print(self.poses, 'particle', particle+1, 'step', step, 'scored', len(rows), flush=True)
                    if step == 1:
                        first_scoring = rows
                ids, probabilities = top5_distribution(rows, base['covered_fractions'])
                draw = float(rng.random()) if ids else None
                save(folder/f'round_{step:02d}_scores.json', dict(base=brief(base), rows=rows,
                    top5_indices=ids, probabilities=probabilities, uniform_draw=draw))
                if not ids:
                    status = 'candidates_exhausted'
                    break
                index = ids[min(int(np.searchsorted(np.cumsum(probabilities), draw, side='right')), len(ids)-1)]
                entries.append(self.geometry.initial[index])
                score = self.evaluate(entries)
                if any(np.any(old & ~new) for old, new in zip(base['masks'], score['masks'])):
                    raise RuntimeError('Adding a fixed head reduced sampled coverage')
                row = dict(step=step, selected_id=self.geometry.initial[index]['contact']['candidate_id'],
                    contacts=[dict(id=e['contact']['candidate_id'], radius_m=e['contact']['radius_m'],
                                   area_m2=e['area']) for e in entries],
                    score=brief(score), area_optimization_performed=False, geometry=self.geometry.group_check(entries))
                rounds.append(row)
                save(folder/f'round_{step:02d}.json', row)
                print(self.poses, 'particle', particle+1, 'heads', len(entries),
                      'coverage', [round(100*x, 4) for x in score['covered_fractions']], flush=True)
                if score['both_sampled_complete']:
                    status = 'both_samples_passed'
                    break
            search_stop_status = status
            entries, expansion = T.run(self, entries)
            save(folder/'terminal_expansion.json', expansion)
            score = self.evaluate(entries)
            status = 'both_samples_passed' if score['both_sampled_complete'] else expansion['status']
            contacts = self.geometry.contacts_by_pose(entries)
            checks = [C.verify_classification(p.supply(c), p.targets, m)
                      for p, c, m in zip(self.problems, contacts, score['masks'])]
            for pose, transformed in zip(self.poses, contacts):
                I.save_contacts(folder/f'contacts_{pose}.npz', transformed)
            np.savez_compressed(folder/'coverage.npz', **{p.pose: m for p, m in zip(self.problems, score['masks'])})
            random_counts = [int(m[:len(t)].sum()) for m, t in zip(score['masks'], self.original_targets)]
            result = dict(particle=particle, seed=chain_seed(self.seed, particle), status=status,
                heads=len(entries), selected_ids=[e['contact']['candidate_id'] for e in entries],
                **brief(score), random_covered_counts=random_counts,
                random_sample_counts=[len(t) for t in self.original_targets],
                rounds=rounds, independent_sample_checks=checks,
                search_stop_status=search_stop_status, terminal_expansion=expansion,
                geometry=self.geometry.group_check(entries), success_rule=A.RULE)
            save(folder/'schedule.json', result)
            results.append(result)
            designs.append(entries)
            save(self.out/'progress.json', dict(complete=False, particles_completed=len(results),
                 statuses=[r['status'] for r in results]))
        return self.finish(results, designs)

    def finish(self, results, designs):
        def rank(i):
            r = results[i]
            return (not r['both_sampled_complete'],
                    -r['mean_coverage'], i)
        winner = min(range(len(results)), key=rank)
        for pose, contacts in zip(self.poses, self.geometry.contacts_by_pose(designs[winner])):
            I.save_contacts(self.out/f'final_contacts_{pose}.npz', contacts)
        result = dict(schema=A.COMPLETION_SCHEMA, object=self.name, poses=self.poses,
            complete=True, scope='two-pose contact model through Step4; connected support not constructed',
            search_seed=self.seed, particles=self.particles, top_k=5, max_heads=self.max_heads,
            sampling='independent top5 gain-weighted chains, uniform if all top5 gains are zero',
            score='equal-weight mean of per-task load coverage; separate reactions; both tasks required for success',
            gravity_policy='separate zero-process-force check is diagnostic only; gravity is included in every sampled load',
            success_rule=A.RULE, sample_count_per_pose=A.SAMPLE_COUNT,
            continuous_validation_performed=False,
            contact_area_fraction=AREA_FRACTION, contact_area_relative_tolerance=AREA_REL_TOL,
            during_selection_area_optimization_performed=False,
            area_optimization_performed=any(r['terminal_expansion']['attempted'] for r in results),
            terminal_expansion_policy=T.policy(),
            expansion_attempted_particles=sum(r['terminal_expansion']['attempted'] for r in results),
            expansion_completed_particles=sum(r['terminal_expansion']['status'] == 'completed_by_expansion' for r in results),
            sizing='fixed 1% during selection; bounded terminal growth only if every task exceeds 98%; no shrinking or area-efficiency objective',
            winner_rule='both tasks complete first, then mean coverage, then chain index; no area objective',
            geometry_model='one shared head set attached to the object; independent fixture placements not optimized',
            insertion_model='independent horizontal object insertion directions per task, full head ray collision checks',
            connectivity_model=self.geometry.paths.record,
            winner=winner, result=results[winner],
            successful_particles=sum(r['both_sampled_complete'] for r in results),
            sampled_complete_particles=sum(r['both_sampled_complete'] for r in results),
            candidate_reasons=dict(Counter(e['reason'] for e in self.geometry.initial)),
            particle_results=[{k: v for k, v in r.items() if k not in ('rounds', 'independent_sample_checks')}
                              for r in results],
            provenance=dict(inputs=self.inputs, code=I.hashes(list(Path(__file__).parent.glob('pair_*.py'))+[Path(__file__), Path(A.__file__), Path(T.__file__)])))
        save(self.out/'schedule.json', result)
        save(self.out/'progress.json', dict(complete=True, particles_completed=len(results)))
        return result

    def floor(self, result, folder=None):
        folder = Path(folder) if folder is not None else pair_folder(self.name, self.poses, 'step4_floor_contact')
        reports = []
        for k, problem in enumerate(self.problems):
            base = self.original_targets[k]/problem.scale
            loads = base
            assert len(loads) == A.SAMPLE_COUNT
            cloud, normal = pressure_centers(loads, problem.domain.com)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/f'floor_contact_{problem.pose}.npz'
            np.savez_compressed(path, load_wrenches=loads, floor_demands_xy_m=cloud,
                total_floor_normal_mg=normal,
                original_pivot_m=problem.floor, moment_origin_m=problem.domain.com)
            # Independently substitute the ground moment equations.
            required = loads[:, 3:]+np.cross(problem.domain.com, loads[:, :3])
            supplied = np.cross(np.c_[cloud, np.zeros(len(cloud))], np.c_[np.zeros((len(cloud), 2)), normal])
            np.testing.assert_allclose(supplied[:, :2], required[:, :2], atol=1e-12, rtol=1e-12)
            report = dict(pose=problem.pose, load_count=len(loads), success_rule=A.RULE,
                arrays=path.name, sha256=sha256(path),
                pressure_formula_checked=True, actual_floor_contacts_designed=False)
            save(folder/f'floor_contact_{problem.pose}.json', report)
            reports.append(report)
        save(folder/'pair_result.json', dict(object=self.name, poses=self.poses, complete=True,
            step3_report=str((self.out/'schedule.json').relative_to(ROOT)),
            step3_status=result['result']['status'], successful_particles=result['successful_particles'],
            sampled_complete_particles=result['sampled_complete_particles'], floor=reports,
            provenance=dict(inputs=self.inputs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    parser.add_argument('--poses', nargs=2)
    parser.add_argument('--pairs', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--particles', type=int, default=10)
    parser.add_argument('--candidates', type=int, default=200)
    parser.add_argument('--max-heads', type=int, default=3)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.particles < 1 or args.max_heads < 1 or args.candidates < 1 or args.workers < 1:
        parser.error('Positive search sizes required')
    if args.poses:
        pairs, seed = [canonical_pair(args.poses)], args.seed if args.seed is not None else 0
    else:
        pairs, seed = sample_pairs(args.object, args.pairs, args.seed)
    if len(pairs) > 1:
        import subprocess
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print('Batch seed:', seed, 'pairs:', pairs, flush=True)
        def launch(pair):
            folder = pair_folder(args.object, pair, 'step4_floor_contact')
            folder.mkdir(parents=True, exist_ok=True)
            save(folder/'batch_plan.json', dict(object=args.object, seed=seed, pair=pair,
                 pairs=pairs, particles=args.particles, max_heads=args.max_heads, candidates=args.candidates))
            command = [sys.executable, str(Path(__file__).resolve()), args.object,
                '--poses', *pair, '--seed', str(seed), '--particles', str(args.particles),
                '--max-heads', str(args.max_heads), '--candidates', str(args.candidates)]
            with (folder/'run.log').open('w') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            return pair, code
        errors = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(launch, p) for p in pairs]):
                pair, code = future.result()
                print('Finished pair', pair, 'exit', code, flush=True)
                if code:
                    errors.append(pair)
        if errors:
            raise RuntimeError(f'Pair execution errors: {errors}; see per-pair run.log')
        return
    print('Pair sampling seed:', seed, 'pairs:', pairs, flush=True)
    for pair in pairs:
        started = time.monotonic()
        folder = pair_folder(args.object, pair, 'step4_floor_contact')
        save(folder/'status.json', dict(complete=False, status='running', poses=pair))
        try:
            search = PairSearch(args.object, pair, seed, args.particles, args.max_heads, args.candidates)
            result = search.run()
            search.floor(result)
            elapsed = time.monotonic()-started
            save(folder/'timing.json', dict(elapsed_seconds=elapsed, scope='paired Step2–4; existing Step1 inputs reused'))
            save(folder/'status.json', dict(complete=True, status=result['result']['status'], elapsed_seconds=elapsed))
            print('PAIR RESULT', pair, result['result']['status'],
                'sample-passed particles', result['successful_particles'], '/', args.particles,
                'coverage', result['result']['covered_fractions'], 'seconds', round(elapsed, 1), flush=True)
        except Exception as error:
            save(folder/'status.json', dict(complete=False, status='error', error_type=type(error).__name__, error=str(error)))
            raise


if __name__ == '__main__':
    main()
