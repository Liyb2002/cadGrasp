from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from step3_scheculer.run_independent import IndependentSearch
from step3_scheculer.run_joint import JointSearch


def toy(path, required):
    search = IndependentSearch.__new__(IndependentSearch)
    search.seed, search.out, search.poses = 29, Path(path), ('pose_5',)
    entries = [dict(contact=dict(candidate_id=f'pose_5_C{i:03d}', radius_m=.01),
                    active_tasks=(0,), owner_task=0, valid=True, reason='valid', area=.01)
               for i in range(8)]
    def check(selected, task=0):
        assert task == 0
        return dict(passed=True, reason='valid', common_direction_ids=[7])
    search.geometry = SimpleNamespace(pools=[entries], task_check=check,
        same_center=lambda e, old: e in old, group_check=check)
    search.task_score = lambda entries, task, known=None: np.arange(required) < len(entries)
    search.evaluate = lambda entries: dict(covered_counts=[min(required,len(entries))],
        covered_fractions=[min(required,len(entries))/required], mean_coverage=min(required,len(entries))/required,
        all_sampled_complete=len(entries)>=required, area_m2=.01*len(entries))
    return search


class IndependentTests(unittest.TestCase):
    def test_pose_uses_only_its_own_pool_force_and_exit(self):
        with TemporaryDirectory() as tmp:
            search = toy(tmp, 3)
            # Neither the whole-head cross-pose tracker nor shared-head picker
            # exists. Completing the search must not require either interface.
            entries, result = search.search_particle(0)
        self.assertTrue(result['passed'])
        self.assertEqual(len(entries), 3)
        self.assertTrue(all(e['active_tasks'] == (0,) for e in entries))
        self.assertEqual(result['geometry']['common_direction_ids'], [7])

    def test_fourth_head_used_only_when_needed(self):
        with TemporaryDirectory() as tmp:
            entries, result = toy(tmp, 4).search_particle(0)
        self.assertTrue(result['passed'])
        self.assertEqual(len(entries), 4)

    def test_finite_failure_preserves_partial_coverage(self):
        with TemporaryDirectory() as tmp:
            entries, result = toy(tmp, 5).search_particle(0)
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'], 'four_heads_incomplete')
        self.assertEqual(result['covered_counts'], [4])

    def test_constructor_never_receives_other_pose_geometry(self):
        with TemporaryDirectory() as tmp:
            problem = SimpleNamespace(targets=np.zeros((2,7)))
            geometry = SimpleNamespace(save=lambda *args:None)
            with patch('step3_scheculer.run_independent.root', return_value=Path(tmp)), \
                 patch('step3_scheculer.run_independent.task_poses', return_value=['pose_5']), \
                 patch('step3_scheculer.run_independent.prepare_task', return_value=problem), \
                 patch('step3_scheculer.run_independent.input_hashes', return_value={}), \
                 patch('step3_scheculer.run_independent.JointGeometry', return_value=geometry) as factory:
                IndependentSearch('B', 'pose_5')
            factory.assert_called_once_with([problem], count=200, global_head_exclusions=False)


if __name__ == '__main__':
    unittest.main()
