"""Physical counterexamples and invariants for the fixed-registration gate."""
from types import SimpleNamespace
import unittest
import numpy as np

from step4_connect_support.shared_preflight import ground_bound, registration_rank, sibling
from step1.needs import OUTPUTS


def about_origin(points, forces, origin):
    return np.r_[forces.sum(axis=0), np.cross(points-origin, forces).sum(axis=0)][None]


def quarter_turn():
    # h(x,y,0)=y: both the direction and its inverse matter.
    return np.array([[1., 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


class SharedPreflightTests(unittest.TestCase):
    def test_real_nonnegative_ground_forces_obey_bound_even_with_tangential_forces(self):
        points = np.array([[0., 0, 0], [-2., 1, 0], [2., 3, 0]])
        forces = np.array([[.3, -.7, .2], [-.4, .3, .5], [.2, -.1, .3]])
        origin = np.array([.4, -.2, .6])
        row, data = ground_bound(about_origin(points, forces, origin), origin, points[0], quarter_turn())
        self.assertTrue(row['necessary_condition_passed'])
        np.testing.assert_allclose(data['pressure_centers_xy_m'][0], forces[:, 2]@points[:, :2])

    def test_original_pivot_may_be_outside_support_foot_region(self):
        points = np.array([[0., -2, 0], [0., 1, 0]])
        forces = np.array([[0., 0, .75], [0., 0, .25]])
        row, data = ground_bound(about_origin(points, forces, np.zeros(3)), np.zeros(3), points[0], quarter_turn())
        self.assertLess(data['other_height_m'][0], 0)
        self.assertEqual(row['lower_bound_m'], -2.)
        self.assertTrue(row['necessary_condition_passed'])

    def test_forbidden_pressure_center_rejected_and_transform_inverse_not_confused(self):
        loads = np.array([[0., 0, 1., -1., 0., 0.]])  # CoP=(0,-1)
        forward, _ = ground_bound(loads, np.zeros(3), np.zeros(3), quarter_turn())
        reverse, _ = ground_bound(loads, np.zeros(3), np.zeros(3), np.linalg.inv(quarter_turn()))
        self.assertFalse(forward['necessary_condition_passed'])
        self.assertEqual(forward['worst_sample']['violation_m'], 1.)
        self.assertTrue(reverse['necessary_condition_passed'])

    def test_coplanar_identity_is_not_a_fixture_success_certificate(self):
        # Arbitrary yaw torque is deliberately invisible to this necessary test.
        row, _ = ground_bound([[0, 0, 1, 5, 7, 1e9]], np.zeros(3), np.zeros(3), np.eye(4))
        self.assertTrue(row['necessary_condition_passed'])
        self.assertNotIn('complete_fixture_verified', row)

    def test_translation_of_other_ground_is_included(self):
        transform = quarter_turn(); transform[2, 3] = 2.
        row, _ = ground_bound([[0, 0, 1, -1, 0, 0]], np.zeros(3), np.zeros(3), transform)
        self.assertTrue(row['necessary_condition_passed'])
        self.assertEqual(row['minimum_slack_m'], 1.)

    def test_nonpositive_normal_and_off_floor_pivot_are_rejected(self):
        for normal in (0, -1):
            with self.assertRaises(ValueError):
                ground_bound([[0, 0, normal, 0, 0, 0]], np.zeros(3), np.zeros(3), np.eye(4))
        with self.assertRaises(ValueError):
            ground_bound([[0, 0, 1, 0, 0, 0]], np.zeros(3), [0, 0, .1], np.eye(4))

    def test_planar_contact_has_three_tangential_rigid_dofs(self):
        contact = dict(triangles_m=np.array([[[0., 0, 0], [1., 0, 0], [0., 1, 0]]]),
                       source_faces=np.array([0]), center_m=np.array([.3, .3, 0]))
        mesh = SimpleNamespace(face_normals=np.array([[0., 0, 1]]), extents=np.ones(3))
        self.assertEqual(registration_rank(contact, mesh)['local_twist_nullity'], 3)

    def test_output_preserves_pair_order_branch_without_new_top_level_folder(self):
        source = OUTPUTS/'B/pose1+6/step3_scheculer/sequential_3plus2/from_pose_6/fixed_only'
        target = sibling(source, 'step4_connect_support')
        self.assertEqual(target, OUTPUTS/'B/pose1+6/step4')


if __name__ == '__main__':
    unittest.main()
