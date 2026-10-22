"""Check real object rotation, exact continuous bounds and gravity frames."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .poses import spin_pose, orbit_bounds
from whole_task_2.step1.pose_coverage import spatial_workspace_mask


class PoseFamilyTests(unittest.TestCase):
    def test_rotation_moves_translation_about_world_axis(self):
        reference = np.eye(4); reference[:3, 3] = [2., 0., 3.]
        actual = spin_pose(reference, [0., 0., 1.], [1., 0., 0.], 90.)
        np.testing.assert_allclose(actual[:3, 3], [1., 1., 3.], atol=1e-12)
        np.testing.assert_allclose(actual[:3, :3].T @ actual[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(spin_pose(reference, [0, 0, 1], [1, 0, 0], 360), reference, atol=1e-12)

    def test_selected_points_stay_in_finite_cone_and_coaxial_ball(self):
        workspace = dict(center_world_m=[0., 0., 2.], usable_radius_m=3.,
            spatial_cone=dict(apex_world_m=[0., 0., 0.], axis_world=[0., 0., 1.],
                maximum_axial_distance_m=3., usable_half_angle_deg=25.))
        triangle = np.array([[[.1, .2, 1.], [.4, .1, 2.], [-.3, .2, 1.5]]])
        for angle in np.arange(0, 360, 7):
            transform = spin_pose(np.eye(4), [0, 0, 1], [0, 0, 0], angle)
            world = triangle @ transform[:3, :3].T+transform[:3, 3]
            self.assertTrue(spatial_workspace_mask(world, workspace).all())

    def test_continuous_bounds_match_dense_orbit_extrema(self):
        axis = np.array([.3, -.6, -.7]); axis /= np.linalg.norm(axis)
        pivot = np.array([.2, .3, .5]); base = np.array([.7, -.4, .1])
        points = np.array([[.4, .1, .6], [-.1, .2, .9], [.8, -.2, .4]])
        bound = orbit_bounds(points, axis, pivot, base)
        rotation = Rotation.from_rotvec(np.radians(np.arange(0, 360, .05))[:, None]*axis).as_matrix()
        orbit = np.einsum('aij,pj->api', rotation, points-pivot)+pivot
        actual_radius = np.linalg.norm(orbit-base, axis=2).max()
        actual_height = orbit[:, :, 2].min()
        self.assertGreaterEqual(bound['maximum_distance_from_reference_m']+1e-12, actual_radius)
        self.assertLessEqual(bound['minimum_world_z_m']-1e-12, actual_height)
        self.assertAlmostEqual(bound['maximum_distance_from_reference_m'], actual_radius, places=6)
        self.assertAlmostEqual(bound['minimum_world_z_m'], actual_height, places=6)

    def test_gravity_changes_in_object_frame_for_tilted_world_axis(self):
        reference = np.eye(4)
        rotated = spin_pose(reference, [0., -1., -1.], [0., 0., 0.], 90.)
        gravity = np.array([0., 0., -9.81])
        self.assertFalse(np.allclose(rotated[:3, :3].T @ gravity, gravity))
        np.testing.assert_allclose(rotated[:3, :3] @ (rotated[:3, :3].T @ gravity), gravity, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
