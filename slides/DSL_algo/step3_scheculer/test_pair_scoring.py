"""Two-task success and sampling must retain the individual task obligations."""
import unittest
import numpy as np
from step3_scheculer.pair_scoring import coverage_summary, top5_distribution, score_tasks
from step3_scheculer.pair_tasks import canonical_pair, sample_pairs, pair_folder, task_folder, prepare_pair_inputs
from unittest.mock import patch, Mock
from types import SimpleNamespace
from step3_scheculer.run_pairs import PairSearch
from step3_scheculer.sample_acceptance import SAMPLE_COUNT, SCHEMA, FIXED_AREA_SCHEMA, COMPLETION_SCHEMA, read_report
from pathlib import Path
from tempfile import TemporaryDirectory
import json


class PairScoringTests(unittest.TestCase):
    def partial_problems(self):
        floor = np.array([[0., 0., 1., 1., 0., 0.]])
        head = np.array([[0., 0., 1., -1., 0., 0.]])
        return [SimpleNamespace(
            supply=lambda contacts, active=active: np.vstack([floor, head]) if active in contacts else floor,
            domain=SimpleNamespace(gravity=np.array([0., 0., -1.])), scale=np.ones(6),
            targets=np.array([[0., 0., 1., 1., 0., 0.], [0., 0., 1., -.5, 0., 0.]]))
            for active in ('A', 'B')]

    def test_gravity_failure_does_not_erase_real_load_coverage(self):
        report = score_tasks(self.partial_problems(), [[], []])
        self.assertFalse(report['gravity_passed'])
        self.assertEqual(report['covered_counts'], [1, 1])
        for mask in report['masks']:
            np.testing.assert_array_equal(mask, [True, False])

    def test_one_pose_gain_enters_scheduler_and_complementary_head_completes(self):
        problems = self.partial_problems()
        base = score_tasks(problems, [[], []])
        first = score_tasks(problems, [['A'], ['A']])
        self.assertEqual([g['passed'] for g in first['gravity']], [True, False])
        self.assertEqual(first['covered_fractions'], [1., .5])
        search = PairSearch.__new__(PairSearch)
        search.geometry = SimpleNamespace(group_check=lambda entries: dict(passed=True, reason='valid'))
        search.evaluate = lambda entries: first
        candidate = dict(contact=dict(candidate_index=0, candidate_id='A'), valid=True, reason='valid')
        row = search.candidate_row([], candidate)
        self.assertTrue(row['eligible'])
        ids, weights = top5_distribution([row], base['covered_fractions'])
        self.assertEqual(ids, [0])
        np.testing.assert_array_equal(weights, [1.])
        final = score_tasks(problems, [['A', 'B'], ['A', 'B']])
        self.assertTrue(final['both_sampled_complete'])
        self.assertTrue(final['gravity_passed'])

    def test_sample_coverage_alone_determines_load_success(self):
        problems = self.partial_problems()
        for p in problems:
            p.targets = p.targets[:1]
        report = score_tasks(problems, [[], []])
        self.assertTrue(report['load_samples_complete'])
        self.assertEqual(report['covered_fractions'], [1., 1.])
        self.assertFalse(report['gravity_passed'])
        self.assertTrue(report['both_sampled_complete'])

    def test_fixed_samples_stop_search_and_generate_only_fixed_floor_loads(self):
        problems = self.partial_problems()
        for index, p in enumerate(problems):
            p.pose = f'pose_{index+1}'
            p.targets = np.concatenate([np.tile(p.targets[:1], (SAMPLE_COUNT-1, 1)), p.targets[1:]])
            p.domain.com = np.zeros(3)
            p.floor = np.zeros(3)
        candidates = [dict(contact=dict(candidate_index=i, candidate_id=name, radius_m=.2+.1*i),
                           valid=True, reason='valid', area=.01) for i, name in enumerate(('A', 'B'))]
        search = PairSearch.__new__(PairSearch)
        search.name, search.poses = 'test', ('pose_1', 'pose_2')
        search.seed, search.particles, search.max_heads = 0, 1, 5
        search.problems, search.inputs, search.scores = problems, {}, {}
        search.original_targets = [p.targets.copy() for p in problems]
        search.geometry = SimpleNamespace(initial=candidates, paths=SimpleNamespace(record={}),
            make=Mock(side_effect=AssertionError('Selected heads must not be resized')),
            expand=Mock(side_effect=AssertionError('Near-complete intermediate rounds must not expand')),
            group_check=lambda entries: dict(passed=True, reason='valid'),
            contacts_by_pose=lambda entries: [[e['contact']['candidate_id'] for e in entries]]*2)
        search.size = Mock(side_effect=AssertionError('Step3 must not optimize contact area'))
        with TemporaryDirectory() as temporary, \
             patch('step3_scheculer.verification.continuous_check', side_effect=AssertionError('No continuum check allowed')), \
             patch('step3_scheculer.enclosure.contain', side_effect=AssertionError('No enclosure check allowed')), \
             patch('step3_scheculer.contacts.save_contacts'):
            search.out = Path(temporary)/'step3'
            result = search.run()
            self.assertEqual(result['result']['status'], 'both_samples_passed')
            self.assertEqual(result['result']['heads'], 2)
            self.assertEqual(result['successful_particles'], 1)
            self.assertEqual(result['schema'], COMPLETION_SCHEMA)
            self.assertFalse(result['area_optimization_performed'])
            self.assertEqual(result['result']['terminal_expansion']['status'], 'already_complete')
            search.geometry.expand.assert_not_called()
            search.size.assert_not_called()
            search.geometry.make.assert_not_called()
            initial_radii = {e['contact']['candidate_id']: e['contact']['radius_m'] for e in candidates}
            previous = [0., 0.]
            for row in result['result']['rounds']:
                self.assertNotIn('efficiency', row['score'])
                self.assertFalse(row['area_optimization_performed'])
                for contact in row['contacts']:
                    self.assertEqual(contact['radius_m'], initial_radii[contact['id']])
                    self.assertEqual(contact['area_m2'], .01)
                self.assertTrue(all(a >= b for a, b in zip(row['score']['covered_fractions'], previous)))
                previous = row['score']['covered_fractions']
            self.assertFalse(result['continuous_validation_performed'])
            self.assertEqual(result['result']['sample_counts'], [SAMPLE_COUNT]*2)
            floor = Path(temporary)/'step4'
            with patch('step3_scheculer.run_pairs.ROOT', Path(temporary)):
                search.floor(result, floor)
            for p in problems:
                with np.load(floor/f'floor_contact_{p.pose}.npz') as data:
                    np.testing.assert_array_equal(data['load_wrenches'], p.targets)
                    self.assertEqual(len(data['floor_demands_xy_m']), SAMPLE_COUNT)

    def test_native_fixed_area_and_previous_sample_reports_ignore_old_relabels(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'sample_result.json').write_text('{"stale": true}')
            for schema in (SCHEMA, FIXED_AREA_SCHEMA, COMPLETION_SCHEMA):
                report = dict(schema=schema, result=dict(status='both_samples_passed'))
                (folder/'schedule.json').write_text(json.dumps(report))
                self.assertEqual(read_report(folder), report)

    def test_each_pose_solves_its_own_reactions(self):
        problems = []
        for sign in (1., -1.):
            rays = np.array([[0, 0, 1, 0, 0, 0, 0],
                             [sign, 0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, -1]], float)
            targets = np.array([[0, 0, 1, 0, 0, 0], [.5*sign, 0, 1, 0, 0, 0]])
            problems.append(SimpleNamespace(supply=lambda contacts, r=rays: r,
                domain=SimpleNamespace(gravity=np.array([0., 0., -1.])),
                scale=np.ones(6), targets=targets))
        report = score_tasks(problems, [[], []])
        self.assertTrue(report['gravity_passed'])
        self.assertTrue(report['both_sampled_complete'])
        self.assertEqual(report['covered_counts'], [2, 2])

    def test_one_complete_pose_is_not_joint_success(self):
        report = coverage_summary([np.ones(8, bool), np.array([True, False])])
        self.assertEqual(report['covered_counts'], [8, 1])
        self.assertAlmostEqual(report['mean_coverage'], .75)
        self.assertFalse(report['both_sampled_complete'])

    def test_zero_gain_in_one_pose_is_allowed(self):
        rows = [dict(index=0, id='A', eligible=True, covered_fractions=[.8, .2]),
                dict(index=1, id='B', eligible=True, covered_fractions=[.4, .6])]
        ids, weights = top5_distribution(rows, [.4, .2])
        self.assertEqual(ids, [0, 1])
        np.testing.assert_allclose(weights, [.5, .5])

    def test_top_five_and_zero_gain_fallback(self):
        rows = [dict(index=i, id=f'C{i:03d}', eligible=True, covered_fractions=[.5, .5])
                for i in range(7)]
        ids, weights = top5_distribution(rows, [.5, .5])
        self.assertEqual(ids, list(range(5)))
        np.testing.assert_allclose(weights, np.full(5, .2))

    def test_ineligible_candidate_cannot_be_rescued_by_score(self):
        rows = [dict(index=0, id='A', eligible=False, covered_fractions=[1., 1.]),
                dict(index=1, id='B', eligible=True, covered_fractions=[.5, .5])]
        ids, weights = top5_distribution(rows, [0., 0.])
        self.assertEqual(ids, [1])
        np.testing.assert_array_equal(weights, [1.])

    def test_pair_sampling_is_unique_and_reproducible(self):
        with patch('step3_scheculer.pair_tasks.task_poses', return_value=tuple(f'pose_{i}' for i in range(1, 11))):
            first, seed = sample_pairs('B', 5, 20260926)
            second, _ = sample_pairs('B', 5, seed)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(all(a != b for a, b in first))

    def test_reversed_pair_uses_same_stage_directory(self):
        self.assertEqual(canonical_pair(['pose_10', 'pose_2']), ('pose_2', 'pose_10'))
        self.assertEqual(pair_folder('B', ['pose_10', 'pose_2'], 'step3_scheculer').parts[-2:],
                         ('pose2+10', 'step3_scheculer'))
        self.assertEqual(pair_folder('B', ['pose_2', 'pose_10'], 'step3_scheculer'),
                         pair_folder('B', ['pose_10', 'pose_2'], 'step3_scheculer'))
        with self.assertRaises(ValueError):
            pair_folder('B', ['pose_1', 'pose_2'], 'step5_base')

    def test_new_pair_reuses_exact_samples_without_single_pose_output(self):
        with TemporaryDirectory() as tmp, patch('step3_scheculer.pair_tasks.OUTPUTS', Path(tmp)), patch('step3_scheculer.pair_tasks.ROOT', Path(tmp)):
            for pose, pair in [('pose_1', ['pose_1', 'pose_3']), ('pose_4', ['pose_4', 'pose_6'])]:
                folder = task_folder('B', pose, pair)
                folder.mkdir(parents=True)
                for filename in ('needs.json', 'samples.json', 'examples.json'):
                    (folder/filename).write_bytes((pose+filename+'\n').encode())
            pair = ['pose_4', 'pose_1']
            prepare_pair_inputs('B', pair)
            for pose in pair:
                folder = task_folder('B', pose, pair)
                self.assertEqual((folder/'samples.json').read_bytes(), (pose+'samples.json\n').encode())
            self.assertFalse((Path(tmp)/'B/pose_1').exists())
            self.assertFalse((Path(tmp)/'B/pose_4').exists())

    def test_unqualified_task_read_rejects_conflicting_pair_samples(self):
        with TemporaryDirectory() as tmp, patch('step3_scheculer.pair_tasks.OUTPUTS', Path(tmp)), patch('step3_scheculer.pair_tasks.ROOT', Path(tmp)):
            for other in ('pose_3', 'pose_6'):
                folder = task_folder('B', 'pose_1', ['pose_1', other])
                folder.mkdir(parents=True)
                (folder/'needs.json').write_text('{}')
                (folder/'samples.json').write_text(other)
            with self.assertRaisesRegex(ValueError, 'Conflicting Step1 inputs'):
                task_folder('B', 'pose_1')
            self.assertEqual(task_folder('B', 'pose_1', ['pose_1', 'pose_6']).parent.parent.name, 'pose1+6')


if __name__ == '__main__':
    unittest.main()
