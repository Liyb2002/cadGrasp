"""Known uplift must be distinguished from a compressive support reaction."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step5_connect_support.bearing_failure import reaction_check


class BearingFailureTests(unittest.TestCase):
    def test_upper_contact_requires_tensile_ground(self):
        points = np.array([[0., 0., 0.], [1., 0., 1.]])
        normals = np.array([[0., 0., 1.], [0., 0., -1.]])
        target = np.array(np.asarray([0., 0., 1., 0., 1., 0.]))
        result, weights = reaction_check(np.asarray(points), np.asarray(normals), np.array([-1, 0]), np.zeros(3), np.ones(6), target)
        np.testing.assert_allclose(weights, [2., 1.])
        self.assertAlmostEqual(result['maximum_support_floor_normal_mg'], -1.)
        self.assertTrue(result['uplift_required_even_with_unrestricted_support_footprint'])

    def test_lower_contact_permits_compression(self):
        points = np.array([[0., 0., 0.], [1., 0., 1.]])
        normals = np.array([[0., 0., 1.], [0., 0., 1.]])
        target = np.array(np.asarray([0., 0., 2., 0., -1., 0.]))
        result, weights = reaction_check(np.asarray(points), np.asarray(normals), np.array([-1, 0]), np.zeros(3), np.ones(6), target)
        np.testing.assert_allclose(weights, [1., 1.])
        self.assertAlmostEqual(result['maximum_support_floor_normal_mg'], 1.)
        self.assertFalse(result['uplift_required_even_with_unrestricted_support_footprint'])


if __name__ == '__main__':
    unittest.main()
