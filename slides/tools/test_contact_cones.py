"""Low-rank cone regressions; tests the shared support-feasibility helpers.

    python -m unittest discover -s tools -p 'test_contact_cones.py'
"""
import unittest

import numpy as np
from scipy.optimize import linprog

from contact_cones import fewest, in_cone


class ConeMembershipTests(unittest.TestCase):
    def test_empty_cone_zero_and_single_ray(self):
        queries = [[0, 0, 0], [0, 0, 1], [0, 0, -1], [1, 0, 0]]
        np.testing.assert_array_equal(in_cone(queries, []), [True, False, False, False])
        np.testing.assert_array_equal(in_cone(queries, [[0, 0, 0]]),
                                      [True, False, False, False])
        np.testing.assert_array_equal(in_cone(queries, [[0, 0, 3]]),
                                      [True, True, False, False])
        np.testing.assert_array_equal(in_cone(queries, [[0, 0, 1], [0, 0, -1]]),
                                      [True, True, True, False])
        self.assertEqual(in_cone([], [[0, 0, 1]]).shape, (0,))

    def test_plane_inside_outside_and_off_plane(self):
        generators = [[1, 0, 0], [0, 1, 0], [2, 2, 0]]
        queries = [[1, 2, 0], [1, 0, 0], [-1, 2, 0], [1, 2, 0.1]]
        np.testing.assert_array_equal(in_cone(queries, generators),
                                      [True, True, False, False])

    def test_full_rank_includes_lower_dimensional_faces(self):
        queries = [[1, 2, 3], [1, 0, 0], [1, 2, 0], [-1, 2, 3]]
        np.testing.assert_array_equal(in_cone(queries, np.eye(3)),
                                      [True, True, True, False])
        # A non-pointed full cone and positive rescalings represent the same set.
        generators = np.vstack([np.eye(3), -np.eye(3)])
        np.testing.assert_array_equal(in_cone(queries, generators), [True] * 4)
        np.testing.assert_array_equal(in_cone(queries, generators * np.arange(1, 7)[:, None]),
                                      [True] * 4)

    def test_membership_against_independent_linear_programs(self):
        rng = np.random.default_rng(20260906)
        for rank in (1, 2, 3):
            basis = np.linalg.qr(rng.normal(size=(3, 3)))[0][:, :rank]
            generators = rng.normal(size=(5, rank)) @ basis.T
            queries = np.vstack([rng.normal(size=(15, rank)) @ basis.T,
                                 rng.normal(size=(10, 3)), generators])
            expected = []
            for query in queries:
                result = linprog(np.zeros(len(generators)), A_eq=generators.T,
                                 b_eq=query, bounds=(0, None), method='highs')
                self.assertIn(result.status, (0, 2))
                expected.append(result.success)
            np.testing.assert_array_equal(in_cone(queries, generators), expected)

    def test_search_checks_zero_one_and_two_extra_contacts(self):
        chosen, share, pairs = fewest([[0, 0, 1]], [[1, 0, 0]], [[0, 0, 1]])
        self.assertEqual((chosen, share, pairs), ([], 1.0, 0))
        chosen, share, pairs = fewest([[1, 0, 1]], [[1, 0, 0]], [[0, 0, 1]])
        self.assertEqual((chosen, share, pairs), ([0], 1.0, 0))
        chosen, share, pairs = fewest([[1, 1, 1]], [[1, 0, 0], [0, 1, 0]],
                                     [[0, 0, 1]])
        self.assertEqual((chosen, share, pairs), ([0, 1], 1.0, 1))
        self.assertEqual(fewest([[1, 0, 0]], [], [[0, 0, 1]]), ([], 0.0, 0))
        self.assertEqual(fewest([[1, 0, 1]], [[1, 0, 0]], [[0, 0, 1]], kmax=0),
                         ([], 0.0, 0))


if __name__ == '__main__':
    unittest.main()
