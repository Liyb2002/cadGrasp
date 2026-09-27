"""The 98% gate and terminal growth cannot weaken per-task acceptance."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from step3_scheculer import terminal_expansion as T
from step3_scheculer.pair_scoring import coverage_summary
from step3_scheculer.pair_geometry import PairGeometry


def head(index, factor=1.):
    return dict(contact=dict(candidate_index=index, candidate_id=f'C{index}',
                             center_m=np.array([index, 0., 0.]), radius_m=np.sqrt(factor)),
                valid=True, reason='valid', area=.01*factor, area_fraction=.01*factor, factor=factor)


def problem(counts, geometry_check=None):
    def evaluate(entries):
        masks = [np.arange(100) < n for n in counts(entries)]
        return dict(**coverage_summary(masks), masks=masks, area_m2=sum(e['area'] for e in entries),
                    gravity_passed=False)
    geometry = SimpleNamespace(
        expand=Mock(side_effect=lambda e, f: head(e['contact']['candidate_index'], f)),
        group_check=geometry_check or (lambda entries: dict(passed=True, reason='passed')))
    return SimpleNamespace(evaluate=evaluate, geometry=geometry)


class TerminalExpansionTests(unittest.TestCase):
    def test_threshold_is_strict_and_applies_to_each_pose_not_the_average(self):
        for counts in ([100, 98], [98, 100], [100, 97]):
            with self.subTest(counts=counts):
                search = problem(lambda entries: counts)
                entries, report = T.run(search, [head(0)])
                self.assertEqual(report['status'], 'sample_coverage_below_expansion_threshold')
                self.assertFalse(report['attempted'])
                search.geometry.expand.assert_not_called()
                self.assertEqual(entries[0]['factor'], 1.)

    def test_already_complete_geometry_is_not_changed(self):
        search = problem(lambda entries: [100, 100])
        entries, report = T.run(search, [head(0)])
        self.assertEqual(report['status'], 'already_complete')
        self.assertFalse(report['area_changed'])
        search.geometry.expand.assert_not_called()

    def test_two_zero_immediate_gain_expansions_can_complete_together(self):
        search = problem(lambda entries: [100, 100] if all(e['factor'] >= 1.01 for e in entries) else [99, 99])
        entries, report = T.run(search, [head(0), head(1)])
        self.assertEqual(report['status'], 'completed_by_expansion')
        self.assertEqual(len(entries), 2)
        self.assertEqual([e['factor'] for e in entries], [1.01, 1.01])
        self.assertEqual(len(report['updates']), 2)
        self.assertEqual(report['updates'][0]['score']['covered_counts'], [99, 99])
        self.assertFalse(report['after']['gravity_passed'])
        for i, entry in enumerate(entries):
            np.testing.assert_array_equal(entry['contact']['center_m'], [i, 0., 0.])

    def test_geometry_blocks_completion_even_when_larger_head_would_cover(self):
        search = problem(lambda entries: [100, 100] if entries[0]['factor'] > 1 else [100, 99],
            lambda entries: dict(passed=entries[0]['factor'] == 1., reason='common_direction_blocked'))
        entries, report = T.run(search, [head(0), head(1)])
        self.assertEqual(report['status'], 'terminal_expansion_exhausted')
        self.assertEqual(entries[0]['factor'], 1.)
        self.assertLessEqual(max(e['factor'] for e in entries), 1.10)
        self.assertFalse(report['after']['both_sampled_complete'])

    def test_failure_stays_failure_at_the_area_cap(self):
        search = problem(lambda entries: [99, 100])
        entries, report = T.run(search, [head(0)])
        self.assertEqual(report['status'], 'terminal_expansion_exhausted')
        self.assertEqual(entries[0]['factor'], 1.10)
        self.assertEqual(report['after']['covered_counts'], [99, 100])
        self.assertEqual([call.args[1] for call in search.geometry.expand.call_args_list], list(T.AREA_FACTORS))

    def test_losing_an_original_sample_is_rejected_even_with_unchanged_counts(self):
        search = problem(lambda entries: [99, 100])
        original = search.evaluate
        def evaluate(entries):
            score = original(entries)
            if entries[0]['factor'] > 1:
                score['masks'][0] = np.roll(score['masks'][0], 1)
            return score
        search.evaluate = evaluate
        entries, report = T.run(search, [head(0)])
        self.assertEqual(entries[0]['factor'], 1.)
        self.assertEqual(report['status'], 'terminal_expansion_exhausted')
        self.assertTrue(all(r['reason'] == 'sample_coverage_regressed' for r in report['trials']))

    def test_geometry_api_rejects_growth_outside_the_small_area_menu(self):
        geometry = PairGeometry.__new__(PairGeometry)
        for factor in (.99, 1.11, 2.):
            with self.assertRaises(ValueError):
                geometry.expand(head(0), factor)


if __name__ == '__main__':
    unittest.main()
