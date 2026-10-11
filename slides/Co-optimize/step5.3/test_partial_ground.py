"""Sparse material is sufficient; a separating floor still limits its hull."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partial_ground as P

np = P.np


class PartialGroundTests(unittest.TestCase):
    def test_disconnected_vertices_can_enclose_demands_without_a_rim(self):
        feet = np.array([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]])
        demands = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        self.assertTrue(P.covered(demands, feet).all())

    def test_retaining_legal_half_boundary_does_not_cover_other_half(self):
        square = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        segments, _ = P.clip_boundary(square, np.array([[1., 0.]]), np.array([0.]))
        self.assertAlmostEqual(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1).sum(), 4.)
        self.assertTrue((segments[..., 0] >= -P.TOL).all())
        np.testing.assert_array_equal(P.covered(square, segments.reshape(-1, 2)), [False, True, True, False])

    def test_unlimited_legal_feet_cannot_cross_a_separating_line(self):
        a, b = np.array([[1., 0.]]), np.array([0.])
        self.assertFalse(P.unlimited_contact_test([-1., 0.], a, b, [])['unlimited_relaxation_feasible'])
        self.assertTrue(P.unlimited_contact_test([1.e5, -1.e5], a, b, [])['finite_contact_witness'])

    def test_object_ground_can_help_enclose_an_otherwise_illegal_point(self):
        result = P.unlimited_contact_test([-1., 0.], np.array([[1., 0.]]), np.array([0.]), [[-2., 0.]])
        self.assertTrue(result['finite_contact_witness'])
        self.assertAlmostEqual(result['fixture_ground_weight'], .5)
        np.testing.assert_allclose(result['mean_fixture_contact_xy_m'], [0., 0.])

    def test_an_infinite_recession_limit_is_not_a_finite_foot_arrangement(self):
        result = P.unlimited_contact_test([-2., 1.], np.array([[1., 0.]]), np.array([0.]), [[-2., 0.]])
        self.assertTrue(result['unlimited_relaxation_feasible'])
        self.assertFalse(result['finite_contact_witness'])
        self.assertEqual(result['status'], 'infinite_recession_limit_only')

    def test_floor_transform_composes_states_in_the_correct_order(self):
        first = np.eye(4)
        second = np.eye(4)
        second[:3, :3] = [[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]
        second[2, 3] = -.01
        a, b = P.floor_inequalities([first, second], 0)
        np.testing.assert_allclose(a[1], [1., 0.])
        self.assertAlmostEqual(b[1], -.01)
        points = np.array([[.02, .01, 0., 1.], [-.02, .01, 0., 1.]])
        np.testing.assert_allclose(points[:, :2] @ a[1]+b[1], (points @ second.T)[:, 2])


if __name__ == '__main__':
    unittest.main()
