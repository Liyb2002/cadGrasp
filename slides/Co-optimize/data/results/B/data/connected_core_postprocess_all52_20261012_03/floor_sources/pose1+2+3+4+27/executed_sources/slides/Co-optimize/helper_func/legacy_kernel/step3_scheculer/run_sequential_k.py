"""Solve K poses in order: 3–4 active heads, one best prior head per new pose."""
import argparse
from collections import OrderedDict
import itertools
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.joint_geometry import JointGeometry
from step3_scheculer.joint_tasks import task_set_folder
from step0_pose_selection.select_poses import accepted_tasks
from step3_scheculer.joint_prepared import geometry_sources, scoring_sources
from step3_scheculer.pair_tasks import input_hashes
from step3_scheculer.random_search import chain_seed
from step3_scheculer.run_joint import JointSearch
from step3_scheculer.run_pairs import brief
from step3_scheculer.run_sequential import top5, best_shared, numerical_recovery
from step3_scheculer.sequential_geometry import active_entries
from step3_scheculer.sequential_withdrawal import SavedCandidateGeometry, WholeHeadWithdrawal
from step1.registry import task_poses

SCHEMA = 'sequential_k_share_best_one_three_or_four_whole_withdrawal_v3'
VARIANT = 'sequential_k_global'


def sample_groups(available, seed):
    available = sorted(available, key=lambda p: int(p.split('_')[1]))
    rng = random.Random(seed)
    return [group for size in (2, 3, 4)
            for group in rng.sample(list(itertools.combinations(available, size)), 2)]


def folder(name, poses, stage, *, variant=VARIANT):
    if variant not in ('sequential_k', VARIANT):
        raise ValueError('Unknown sequential result variant')
    return task_set_folder(name, poses, stage)/variant/('from_'+'_'.join(p.split('_')[1] for p in poses))


class SequentialKSearch(JointSearch):
    def __init__(self, name, poses, seed=20260928, particles=10, count=200, *, reuse_candidates=False):
        if len(poses) < 2 or len(set(poses)) != len(poses) or particles < 1 or count < 1:
            raise ValueError('Distinct poses and positive search sizes required')
        self.name, self.poses, self.seed, self.particles = name, tuple(poses), seed, particles
        self.max_heads = None
        # Step0 must pass before writing a search or generating any candidate.
        by_pose = {p.pose:p for p in accepted_tasks(name, poses)}
        self.out = folder(name, poses, 'step3_scheculer')
        self.out.mkdir(parents=True, exist_ok=True)
        I.save(self.out/'schedule.json', dict(schema=SCHEMA, complete=False, status='preparing', poses=poses))
        self.problems = [by_pose[p] for p in poses]
        self.original_targets = [p.targets.copy() for p in self.problems]
        self.inputs = input_hashes(self.problems)
        self.geometry_folder = folder(name, poses, 'step2_local_support')
        print('REUSE CANDIDATES' if reuse_candidates else 'PREPARE CANDIDATES', poses, count, flush=True)
        if reuse_candidates:
            self.geometry = SavedCandidateGeometry(self.geometry_folder, self.problems, count)
        else:
            self.geometry = JointGeometry(self.problems, count=count, global_head_exclusions=True)
            self.geometry.save(self.geometry_folder, I.save)
        self.inputs.update(I.hashes(list(self.geometry_folder.glob('candidates_*.json'))+
                                   list(self.geometry_folder.glob('candidates_*.npz'))))
        self.withdrawal = WholeHeadWithdrawal(self.geometry)
        self.direction_state = self.withdrawal.initial()
        self.scores, self.recoveries = OrderedDict(), {}

    def choose_shared_for(self, entries, task):
        baseline = int(self.task_score([], task).sum())
        rows, proposals, states = [], [], []
        for index, entry in enumerate(entries):
            proposal = dict(entry, active_tasks=tuple(entry['active_tasks'])+(task,))
            check = self.geometry.task_check([proposal], task)
            state = self.withdrawal.restrict(self.direction_state, task, check.get('common_direction_ids', []))
            eligible = check['passed'] and all(state)
            row = dict(index=index, id=entry['contact']['candidate_id'], eligible=eligible,
                       reason=check['reason'] if not check['passed'] or eligible else 'no_inherited_withdrawal_direction',
                       owner_pose=self.poses[entry['owner_task']])
            if eligible:
                count = int(self.task_score([proposal], task).sum())
                row.update(covered_count=count, gain=count-baseline)
            rows.append(row); proposals.append(proposal); states.append(state)
        chosen = best_shared(rows)
        selected = list(entries)
        if chosen is not None:
            selected[chosen['index']] = proposals[chosen['index']]
            self.direction_state = states[chosen['index']]
        return selected, dict(candidates=rows, baseline_count=baseline,
            selected_id=None if chosen is None else chosen['id'],
            whole_head_withdrawal=self.withdrawal.record(self.direction_state),
            rule='Maximum current-pose singleton coverage gain among ALL earlier physical heads; ID tie-break')

    def extend_stage(self, entries, task, rng, destination):
        rounds = []
        while len(active_entries(entries, task)) < 4:
            base = self.task_score(entries, task)
            if len(active_entries(entries, task)) >= 3 and base.all():
                break
            rows, states = [], {}
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
                        inherited = self.withdrawal.restrict(self.direction_state, task, check['common_direction_ids'])
                        state, whole = self.withdrawal.append(inherited, candidate)
                        row.update(reason=whole['reason'], whole_head_withdrawal=whole)
                        if state is not None:
                            states[index] = state
                            mask = self.task_score(entries+[candidate], task, base)
                            assert not np.any(base & ~mask)
                            row.update(eligible=True, covered_count=int(mask.sum()))
                rows.append(row)
            ids, probabilities = top5(rows, int(base.sum()))
            draw = float(rng.random()) if ids else None
            number = len(rounds)+1
            I.save(destination/f'{self.poses[task]}_round_{number:02d}_scores.json',
                dict(rows=rows, base_count=int(base.sum()), top5_indices=ids,
                     inherited_withdrawal=self.withdrawal.record(self.direction_state),
                     probabilities=np.asarray(probabilities).tolist(), uniform_draw=draw))
            if not ids:
                break
            choice = min(int(np.searchsorted(np.cumsum(probabilities), draw, side='right')), len(ids)-1)
            entries = entries+[self.geometry.pools[task][ids[choice]]]
            self.direction_state = states[ids[choice]]
            mask = self.task_score(entries, task, base)
            row = dict(step=number, selected_id=entries[-1]['contact']['candidate_id'],
                active_ids=[e['contact']['candidate_id'] for e in active_entries(entries, task)],
                covered_count=int(mask.sum()), sample_count=len(mask),
                geometry=self.geometry.task_check(entries, task),
                whole_head_withdrawal=self.withdrawal.record(self.direction_state))
            rounds.append(row)
            I.save(destination/f'{self.poses[task]}_round_{number:02d}.json', row)
            I.save(self.out/'progress.json', dict(complete=False, particle=destination.name,
                current_pose=self.poses[task], active_heads=len(row['active_ids']),
                covered_count=int(mask.sum()), sample_count=len(mask)))
            print(self.poses, destination.name, self.poses[task], len(row['active_ids']), int(mask.sum()), flush=True)
        return entries, rounds

    def search_particle(self, particle):
        destination = self.out/f'particle_{particle:03d}'
        rng = np.random.default_rng(chain_seed(self.seed, particle))
        entries, stages = [], []
        self.direction_state = self.withdrawal.initial()
        status = 'all_contacts_and_whole_head_withdrawal_passed'
        for task, pose in enumerate(self.poses):
            shared = None
            if task:
                entries, shared = self.choose_shared_for(entries, task)
                I.save(destination/f'{pose}_shared_head.json', shared)
                if shared['selected_id'] is None:
                    status = 'no_valid_prior_shared_head'
                    stages.append(dict(pose=pose, passed=False, shared_head=shared, rounds=[]))
                    break
            entries, rounds = self.extend_stage(entries, task, rng, destination)
            mask = self.task_score(entries, task)
            passed = 3 <= len(active_entries(entries, task)) <= 4 and bool(mask.all())
            stages.append(dict(pose=pose, passed=passed, shared_head=shared, rounds=rounds,
                               covered_count=int(mask.sum()), sample_count=len(mask)))
            if not passed:
                status = ('current_pose_not_covered_with_four_heads' if len(active_entries(entries, task)) == 4
                          else 'no_eligible_head_with_inherited_withdrawal')
                break
        score = self.evaluate(entries)
        geometry = self.geometry.group_check(entries)
        whole = self.withdrawal.verify(entries, self.direction_state)
        geometry.update(whole_head_withdrawal=whole, inactive_head_withdrawal_checked=True,
                        passed=geometry['passed'] and whole['passed'])
        for local, physical in zip(geometry['per_pose'], whole['per_pose']):
            local['active_common_direction_ids'] = local['common_direction_ids']
            local['common_direction_ids'] = physical['common_direction_ids']
        result = dict(particle=particle, seed=chain_seed(self.seed, particle), status=status,
            passed=len(stages)==len(self.poses) and all(s['passed'] for s in stages)
                   and geometry['passed'] and score['all_sampled_complete'],
            heads=len(entries), selected_ids=[e['contact']['candidate_id'] for e in entries],
            active_ids_by_pose=[[c['candidate_id'] for c in group] for group in self.geometry.contacts_by_pose(entries)],
            stages=stages, geometry=geometry, **brief(score))
        return entries, result

    def run(self):
        results, designs = [], []
        for particle in range(self.particles):
            entries, result = self.search_particle(particle)
            self.export_and_check(entries, result)
            results.append(result); designs.append(entries)
            if result['passed']:
                break
        winner = min(range(len(results)), key=lambda i:(not results[i]['passed'], -results[i]['mean_coverage'], i))
        selected = designs[winner]
        artifacts = {}
        for pose, group in zip(self.poses, self.geometry.contacts_by_pose(selected)):
            path = self.out/f'final_contacts_{pose}.npz'
            I.save_contacts(path, group); artifacts[path.name] = I.sha256(path)
        report = dict(schema=SCHEMA, complete=True, object=self.name, poses=self.poses,
            search_seed=self.seed, particles=len(results), particle_budget=self.particles,
            stop_after_first_success=True, min_heads_per_pose=3, max_heads_per_pose=4,
            shared_from_all_prior_heads=True, new_shared_heads_per_later_pose=1,
            global_head_exclusions=True,
            whole_head_withdrawal_checked_each_addition=True, inherited_direction_sets=True,
            head_exclusion_rule='Every head avoids every supplied pose work surface and ground, including inactive poses',
            candidate_count_per_pose=self.geometry.count, terminal_expansion=False,
            complete_fixture_verified=False, winner=winner, result=results[winner],
            successful_particles=sum(r['passed'] for r in results),
            heads=[dict(id=e['contact']['candidate_id'], owner_pose=self.poses[e['owner_task']],
                active_poses=[self.poses[k] for k in e['active_tasks']], radius_m=e['contact']['radius_m'],
                area_m2=e['area'], normal_depth_m=self.geometry.local[e['owner_task']].depth) for e in selected],
            particle_results=[{k:v for k,v in r.items() if k!='stages'} for r in results],
            numerical_recovery=self.recoveries,
            provenance=dict(inputs=self.inputs, code=I.hashes(
                [Path(__file__), Path(__file__).with_name('sequential_withdrawal.py')]+geometry_sources()+scoring_sources())),
            artifacts=artifacts)
        I.save(self.out/'schedule.json', report)
        I.save(self.out/'progress.json', dict(complete=True, particles_completed=len(results), passed=results[winner]['passed']))
        return report


def run_case(name, poses, seed, particles=10, count=200, *, reuse_candidates=False):
    began = time.monotonic()
    search = SequentialKSearch(name, poses, seed, particles, count, reuse_candidates=reuse_candidates)
    with numerical_recovery(search.out/'numerical_retries', search.recoveries):
        report = search.run()
    print('STEP3 COMPLETE', poses, report['result']['passed'], report['result']['heads'],
          report['result']['covered_counts'], 'seconds', time.monotonic()-began, flush=True)
    return search.out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    parser.add_argument('--poses', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--particles', type=int, default=10)
    parser.add_argument('--candidates', type=int, default=200)
    parser.add_argument('--reuse-candidates', action='store_true', help='Read exact saved Step2 geometry')
    args = parser.parse_args()
    if any(p not in task_poses(args.object) for p in args.poses):
        parser.error('Unknown pose')
    run_case(args.object, args.poses, args.seed, args.particles, args.candidates, reuse_candidates=args.reuse_candidates)
