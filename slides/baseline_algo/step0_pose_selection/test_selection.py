"""Step0 retries, exhaustion, immutable inputs, and downstream stage gates."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import nullcontext

from step0_pose_selection.select_poses import choose, choose_many, shuffled_combinations, materialize_step1
from step3_scheculer.run_sequential_k import SequentialKSearch
from run_sequential_batch import run_pipeline


class SelectionTests(unittest.TestCase):
    def test_batch_selects_distinct_passes_and_stops_at_requested_count(self):
        seen = []
        def evaluate(group):
            seen.append(group)
            return dict(passed=len(seen) in (2, 4, 5))
        result = choose_many([f'pose_{i}' for i in range(1, 5)], 2, 9, evaluate, 2)
        self.assertTrue(result['passed'])
        self.assertEqual(result['selected_groups'], [list(seen[1]), list(seen[3])])
        self.assertEqual(len(seen), 4)
        self.assertEqual(len(set(seen)), 4)

    def test_batch_preserves_partial_selection_on_exhaustion(self):
        seen = []
        def evaluate(group):
            seen.append(group)
            return dict(passed=len(seen) == 1)
        result = choose_many(['pose_1', 'pose_2', 'pose_3'], 2, 9, evaluate, 2)
        self.assertFalse(result['passed'])
        self.assertTrue(result['exhausted'])
        self.assertEqual(result['selected_groups'], [list(seen[0])])
        self.assertEqual(len(seen), 3)

    def test_reject_then_pass_stops_at_first_compatible_set(self):
        seen = []
        def evaluate(poses):
            seen.append(poses)
            return dict(passed=len(seen) == 4, violating_counts=[0, 0] if len(seen) == 4 else [1, 0])
        result = choose(['pose_1', 'pose_2', 'pose_3', 'pose_4'], 2, 81, evaluate)
        self.assertTrue(result['passed'])
        self.assertEqual(result['attempted_count'], 4)
        self.assertEqual(result['selected_poses'], list(seen[-1]))
        self.assertEqual(len(set(seen)), 4)
        self.assertTrue(all(len(set(poses)) == 2 for poses in seen))

    def test_exhaustion_terminates_without_repeating_or_reducing_n(self):
        seen = []
        def reject(poses):
            seen.append(poses)
            return dict(passed=False, violating_counts=[1]*len(poses))
        result = choose([f'pose_{i}' for i in range(1, 11)], 4, 18, reject)
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'], 'no_floor_compatible_pose_set')
        self.assertIsNone(result['selected_poses'])
        self.assertEqual(len(seen), 210)
        self.assertEqual(len(set(seen)), 210)
        self.assertTrue(all(len(g) == 4 for g in seen))

    def test_seed_is_reproducible_independent_of_registry_order(self):
        available = ['pose_10', 'pose_2', 'pose_7', 'pose_1']
        a = shuffled_combinations(available, 2, 99)
        self.assertEqual(a, shuffled_combinations(available[::-1], 2, 99))
        self.assertNotEqual(a, shuffled_combinations(available, 2, 100))

    def test_invalid_n_never_evaluates_a_pose_set(self):
        evaluate = Mock()
        for n in (0, 1, 4, -1, True):
            with self.assertRaises(ValueError):
                choose(['pose_1', 'pose_2', 'pose_3'], n, 0, evaluate)
        evaluate.assert_not_called()

    def test_failed_step0_never_starts_step1_or_head_generation(self):
        with TemporaryDirectory() as tmp:
            selection = SimpleNamespace(report=dict(passed=False, selected_poses=None,
                status='no_floor_compatible_pose_set'), ledger_path=Path(tmp)/'selection.json')
            with patch('run_sequential_batch.I.ROOT', Path(tmp)), \
                 patch('run_sequential_batch.select', return_value=selection), \
                 patch('run_sequential_batch.materialize_step1') as inputs, \
                 patch('step3_scheculer.run_sequential_k.JointGeometry') as candidates:
                result = run_pipeline('B', 3)
            self.assertEqual(result['completed_through'], 0)
            self.assertFalse(result['constructed'])
            inputs.assert_not_called(); candidates.assert_not_called()

    def test_step0_only_does_not_publish_step1_even_on_success(self):
        with TemporaryDirectory() as tmp:
            selection = SimpleNamespace(report=dict(passed=True, selected_poses=['pose_1', 'pose_2'],
                status='pose_set_selected'), ledger_path=Path(tmp)/'selection.json')
            with patch('run_sequential_batch.I.ROOT', Path(tmp)), \
                 patch('run_sequential_batch.select', return_value=selection), \
                 patch('run_sequential_batch.materialize_step1') as inputs:
                result = run_pipeline('B', 2, through_step=0)
            self.assertTrue(result['step0_passed']); inputs.assert_not_called()

    def test_direct_step3_requires_step0_before_any_candidate_work(self):
        with patch('step3_scheculer.run_sequential_k.accepted_tasks', side_effect=ValueError('Step0 failed')), \
             patch('step3_scheculer.run_sequential_k.JointGeometry') as candidates, \
             patch('step3_scheculer.run_sequential_k.folder') as folder:
            with self.assertRaisesRegex(ValueError, 'Step0 failed'):
                SequentialKSearch('B', ['pose_1', 'pose_2'])
        candidates.assert_not_called(); folder.assert_not_called()

    def test_step1_refuses_different_samples_in_existing_group(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'inputs'; target = root/'step1/pose_1'
            source.mkdir(); target.mkdir(parents=True)
            (source/'needs.json').write_text('original')
            (target/'needs.json').write_text('different')
            report = dict(passed=True, object='B', poses=['pose_1', 'pose_2'],
                          load_input_folders={'pose_1':'inputs', 'pose_2':'inputs'})
            with patch('step0_pose_selection.select_poses.I.ROOT', root), \
                 patch('step0_pose_selection.select_poses.I.check_report', return_value=report), \
                 patch('step0_pose_selection.select_poses.task_set_folder', return_value=root/'step1'):
                with self.assertRaisesRegex(ValueError, 'differs from Step0'):
                    materialize_step1('B', ['pose_1', 'pose_2'], root/'report.json')
            self.assertEqual((target/'needs.json').read_text(), 'different')

    def test_step1_rejects_failed_or_different_pose_check(self):
        for report in [dict(passed=False, object='B', poses=['pose_1','pose_2']),
                       dict(passed=True, object='B', poses=['pose_1','pose_3'])]:
            with patch('step0_pose_selection.select_poses.I.check_report', return_value=report):
                with self.assertRaisesRegex(ValueError, 'accepted Step0'):
                    materialize_step1('B', ['pose_1','pose_2'], '/not_used')

    def test_full_pipeline_includes_body_stage_diagnostics_on_incomplete_heads(self):
        for passed in (False, True):
            with self.subTest(step3_passed=passed), TemporaryDirectory() as tmp:
                root = Path(tmp); group = root/'pose1+2'
                selection = SimpleNamespace(report=dict(passed=True, selected_poses=['pose_1','pose_2'],
                    status='pose_set_selected'), ledger_path=root/'selection.json',
                    check_path=group/'step0_pose_selection/report.json')
                search = SimpleNamespace(out=group/'step3_scheculer/sequential_k_global/from_1_2',
                    recoveries={}, run=Mock(return_value=dict(result=dict(passed=passed,
                        status='tested', heads=5, covered_counts=[32768,32768 if passed else 9]))))
                with patch('run_sequential_batch.I.ROOT', root), \
                     patch('run_sequential_batch.select', return_value=selection), \
                     patch('run_sequential_batch.materialize_step1'), \
                     patch('step3_scheculer.run_sequential_k.SequentialKSearch', return_value=search), \
                     patch('step3_scheculer.run_sequential.numerical_recovery', return_value=nullcontext()), \
                     patch('step4_connect_support.run_sequential_k.run', return_value=dict(
                         constructed=passed, status='construction_only' if passed else 'step3_incomplete')) as construct, \
                     patch('step4_connect_support.draw_selected_heads.draw') as draw:
                    result = run_pipeline('B', 2)
                self.assertEqual(result['completed_through'], 4)
                self.assertEqual(result['constructed'], passed)
                construct.assert_called_once_with(search.out)
                draw.assert_called_once_with(group/'step4')


if __name__ == '__main__':
    unittest.main()
