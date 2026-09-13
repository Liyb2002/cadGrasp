"""Discriminating checks for a handedness-changing world-axis migration."""
import unittest
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import coordinate_transport as C
import coordinates as WORLD


class Coordinates(unittest.TestCase):
    def test_positions_and_ground_exchange_only_y_z(self):
        np.testing.assert_array_equal(C.polar([2., 3., 7.]), [2., 7., 3.])
        np.testing.assert_array_equal(C.polar([2., 0., 3.])[WORLD.UP_AXIS], 0.)
        np.testing.assert_array_equal(C.polar([0., -1., 0.]), [0., 0., -1.])

    def test_torque_is_axial_not_just_a_permutation(self):
        r = np.array([.23, -.13, .37]); force = np.array([.4, .2, -.8])
        expected = np.cross(C.polar(r), C.polar(force))
        np.testing.assert_array_equal(C.axial(np.cross(r, force)), expected)
        self.assertFalse(np.allclose(C.polar(np.cross(r, force)), expected))

    def test_rigid_pose_and_quaternion_conjugation(self):
        rot = Rotation.from_rotvec([.3, -.7, .2])
        pose = np.eye(4); pose[:3, :3] = rot.as_matrix(); pose[:3, 3] = [.2, .6, -.1]
        q = rot.as_quat()[[3, 0, 1, 2]]
        moved = C.transform(pose)
        np.testing.assert_allclose(np.linalg.det(moved[:3, :3]), 1.)
        np.testing.assert_allclose(Rotation.from_quat(C.quaternion_wxyz(q)[[1, 2, 3, 0]]).as_matrix(), moved[:3, :3])
        point = np.array([.1, .4, .8])
        np.testing.assert_allclose(moved[:3, :3]@C.polar(point)+moved[:3, 3], C.polar(pose[:3, :3]@point+pose[:3, 3]))

    def test_mesh_keeps_outward_normals_volume_and_triangle_order(self):
        old = trimesh.creation.box(extents=[.2, .3, .4])
        old.apply_translation([.05, .08, .2])
        new = C.mesh(old)
        self.assertTrue(new.is_watertight and new.is_winding_consistent)
        self.assertGreater(new.volume, 0.)
        self.assertAlmostEqual(new.volume, old.volume)
        np.testing.assert_allclose(new.face_normals, C.polar(old.face_normals))
        np.testing.assert_array_equal(new.triangles, C.triangles(old.triangles))
        np.testing.assert_array_equal(C.mesh(new).triangles, old.triangles)

    def test_shared_reactions_keep_equilibrium(self):
        rng = np.random.default_rng(927)
        points, normals = rng.normal(size=(2, 20, 3))
        weights = rng.random(20)
        supply = np.c_[normals, np.cross(points, normals)]
        world_supply = np.c_[C.polar(normals), np.cross(C.polar(points), C.polar(normals))]
        np.testing.assert_allclose(C.wrench(supply), world_supply)
        np.testing.assert_allclose(weights@world_supply, C.wrench(weights@supply))
        np.testing.assert_array_equal(C.wrench(C.wrench(np.c_[supply, weights])), np.c_[supply, weights])

    def test_floor_pressure_center_under_axis_exchange(self):
        force = np.array([.12, -.17, -.92]); point = np.array([.4, .3, .7])
        gravity = np.array([0., 0., -1.]); com = np.array([.1, .2, .4])
        moment = np.cross(point, force)+np.cross(com, gravity)
        normal = -(force+gravity)[2]
        old_landing = np.array([moment[1]/normal, -moment[0]/normal, 0.])
        world_moment = np.cross(C.polar(point), C.polar(force))+np.cross(C.polar(com), C.polar(gravity))
        world_landing = np.array([-world_moment[2]/normal, 0., world_moment[0]/normal])
        np.testing.assert_allclose(world_landing, C.polar(old_landing))

    def test_camera_projection_preserves_existing_design_view(self):
        rng = np.random.default_rng(2)
        points = rng.normal(size=(20, 3)); camera_rows = Rotation.random(random_state=rng).as_matrix()
        # Camera rows are screen axes expressed in world coordinates, not a body rotation.
        np.testing.assert_allclose(C.polar(points)@C.polar(camera_rows).T, points@camera_rows.T)

    def test_native_world_is_z_up(self):
        self.assertEqual(WORLD.UP_AXIS, 2)
        np.testing.assert_array_equal(WORLD.WORLD_UP, [0., 0., 1.])
        np.testing.assert_array_equal(WORLD.floor([2., 3., 7.]), [2., 3.])
        np.testing.assert_array_equal(WORLD.lift_floor([2., 3.]), [2., 3., 0.])


if __name__ == '__main__':
    unittest.main()
