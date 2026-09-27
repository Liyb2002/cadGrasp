"""A 3-D drawing must not invent an intersection or discard a residual couple."""
import unittest
import numpy as np
from row3 import crossing, construction, examples, load_case, WEIGHT, landings


class SpatialConstructionTests(unittest.TestCase):
    def test_intersection_and_full_moment_equivalence(self):
        c = np.array([.01, -.02, .10])
        q = np.array([.05, .03, .18])
        X = np.array([c[0], c[1], .27])
        F = .5 * (q - X) / np.linalg.norm(q - X)
        case = construction(c, q, F)
        np.testing.assert_allclose(case['X'], X, atol=1e-12)
        np.testing.assert_allclose(np.cross(case['p'] - X, case['W']), 0, atol=1e-12)
        self.assertAlmostEqual(case['p'][2], 0.)
        reaction = -case['W']
        np.testing.assert_allclose(WEIGHT + F + reaction, 0., atol=1e-12)
        np.testing.assert_allclose(np.cross(c, WEIGHT) + np.cross(q, F)
                                   + np.cross(case['p'], reaction), 0., atol=1e-12)

    def test_skew_force_lines_have_no_X(self):
        c, q, F = [0., 0., .1], [.04, 0., .2], [0., .2, -.3]
        with self.assertRaisesRegex(ValueError, 'skew'):
            crossing(c, q, F)
        # A pressure-center point still balances pitch/roll, but not yaw.
        p, _ = landings(q, F, c)
        residual = np.cross(c, WEIGHT) + np.cross(q, F) - np.cross(p, WEIGHT + F)
        np.testing.assert_allclose(residual[:2], 0., atol=1e-12)
        self.assertGreater(abs(residual[2]), 1e-4)

    def test_parallel_vertical_loads_do_not_define_a_unique_X(self):
        with self.assertRaisesRegex(ValueError, 'Parallel'):
            crossing([0., 0., .1], [.04, .02, .2], [0., 0., -.5])

    def test_two_distinct_loads_then_same_point_and_direction_at_lower_magnitude(self):
        domain = load_case()
        cases = examples(domain)
        self.assertEqual(len(cases), 3)
        self.assertEqual(len({item['face'] for item in cases}), 2)
        for item, magnitude in zip(cases, (.5, .5, .1)):
            self.assertIn(item['face'], domain.work_ids)
            self.assertLessEqual(item['cone_angle_deg'], domain.cone_half_deg)
            self.assertAlmostEqual(np.linalg.norm(item['force']), magnitude)
            np.testing.assert_allclose(item['q'], domain.mesh.triangles_center[item['face']])
            np.testing.assert_allclose(item['moment_residual'], 0., atol=1e-12)
        self.assertGreater(np.linalg.norm(cases[0]['q'] - cases[1]['q']), .15 * domain.mesh.extents.max())
        angle = np.degrees(np.arccos(np.clip(cases[0]['force'] @ cases[1]['force'] / .25, -1, 1)))
        self.assertGreater(angle, 25.)
        np.testing.assert_array_equal(cases[2]['q'], cases[1]['q'])
        np.testing.assert_allclose(cases[2]['force'], .2 * cases[1]['force'], atol=1e-12)
        np.testing.assert_allclose(cases[2]['X'], cases[1]['X'], atol=1e-12)
        plumb = np.r_[domain.com[:2], 0.]
        self.assertLess(np.linalg.norm(cases[2]['p'] - plumb), np.linalg.norm(cases[1]['p'] - plumb))


if __name__ == '__main__':
    unittest.main()
