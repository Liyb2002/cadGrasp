"""Guard the workstation aggregation and the inverse saved-placement transform."""
import unittest

import numpy as np

from step5_evaluate.evaluate import measure


class FootprintTests(unittest.TestCase):
    def test_all_poses_share_one_box_not_sum_or_largest_area(self):
        tall = np.array([[0., 0., 0.], [1., 4., 1.]])
        wide = np.array([[0., 0., 0.], [4., 1., 1.]])
        tiny_support = np.array([[0., 0., 0.], [.5, .5, .5]])
        result, rows, _ = measure([tall, wide], tiny_support, [np.eye(3)]*2, np.zeros((2, 3)))
        self.assertEqual(result['object_poses']['xy_area_m2'], 16.)
        self.assertEqual([r['object_poses']['xy_area_m2'] for r in rows], [4., 4.])
        self.assertEqual(result['extra_area_ratio'], 0.)

    def test_support_is_transformed_back_to_saved_task_world(self):
        angle = np.deg2rad(37)
        basis = np.array([[np.cos(angle), -np.sin(angle), 0.],
                          [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        offset = np.array([7., -3., 2.])
        # An asymmetric triangle catches row/column inversion and bbox-corner shortcuts.
        actual_support = np.array([[-1., 0., 0.], [2., 0., 1.], [0., 5., 2.]])
        fixture = actual_support @ basis + offset
        obj = np.array([[0., 0., 0.], [1., 1., 1.]])
        result, _, world = measure([obj], fixture, [basis], [offset])
        np.testing.assert_allclose(world[0], actual_support, atol=1e-12)
        self.assertAlmostEqual(result['object_and_support_poses']['xy_area_m2'], 15.)
        self.assertAlmostEqual(result['extra_area_ratio'], 14.)


if __name__ == '__main__':
    unittest.main()
