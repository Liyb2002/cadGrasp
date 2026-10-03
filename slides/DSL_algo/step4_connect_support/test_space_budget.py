"""Spatial-budget regressions independent of mesh construction dependencies."""
import unittest

import numpy as np

from step4_connect_support.space_budget import analyze, inclusion


class SpaceBudgetTests(unittest.TestCase):
    def test_far_apart_poses_keep_saved_world_origins(self):
        objects = [np.array([[0, 0, 0], [1, 2, 3]]),
                   np.array([[10, 0, 0], [11, 2, 3]])]
        result = analyze(objects, [np.array([[.5, 1, 0]]), np.array([[10.5, 1, 0]])],
                         ['pose_1', 'pose_2'])
        self.assertEqual(result['object_box']['extents_mm'], [11000., 2000., 3000.])
        self.assertEqual(result['extra_xy_area_cm2'], 0.)
        self.assertEqual(len(result['expansion_budgets']), 1)

    def test_asymmetric_growth_preserves_every_original_point(self):
        objects = [np.array([[0, 0, 0], [.1, .2, .3]])]
        # An extremal point that a sparse plotting sample could miss.
        demands = [np.vstack([np.tile([.05, .1, 0], (32767, 1)), [-.005, .207, 0]])]
        result = analyze(objects, demands, ['pose_1'], expansion_step_mm=2)
        self.assertEqual(result['per_pose'][0]['outside_count'], 1)
        self.assertEqual(len(result['expansion_budgets']), 5)
        np.testing.assert_allclose(result['lower_expansion_mm'], [5, 0, 0])
        np.testing.assert_allclose(result['upper_expansion_mm'], [0, 7, 0])
        budgets = result['expansion_budgets']
        self.assertFalse(budgets[-2]['all_demands_inside'])
        self.assertTrue(budgets[-1]['all_demands_inside'])
        self.assertEqual(budgets[-1]['box'], result['object_and_demands_box'])
        for budget in budgets:
            self.assertEqual(budget['box']['max_m'][0], .1)
            self.assertEqual(budget['box']['max_m'][2], .3)

    def test_closed_box_boundaries_and_invalid_step(self):
        result = analyze([np.array([[0, 0, 0], [1, 1, 1]])],
                         [np.array([[0, 1, 0], [1, 0, 0]])], ['pose_1'])
        self.assertTrue(inclusion(np.array([[0, 1, 0], [1, 0, 0]]), result['object_box'])['all_inside'])
        for step in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                analyze([np.zeros((1, 3))], [np.zeros((1, 3))], ['pose_1'], step)


if __name__ == '__main__':
    unittest.main()
