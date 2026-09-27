"""Contact identity, relative-motion sign and weak 3-D connectivity checks."""
import unittest
import numpy as np
import trimesh
from types import SimpleNamespace
from unittest.mock import Mock
from scipy.spatial.transform import Rotation

from step3_scheculer.pair_geometry import transform_contact, horizontal_catalogue, FreePaths, PairGeometry


class PairGeometryTests(unittest.TestCase):
    def fixed_area_geometry(self):
        geometry = PairGeometry.__new__(PairGeometry)
        geometry.mesh = SimpleNamespace(area=100.)
        geometry.target_area = 1.
        geometry.centers = np.array([[.1, .1, 0.]])
        geometry.faces = np.array([0])
        geometry.cache = {}
        geometry.clearance = SimpleNamespace(check=Mock(return_value=dict(valid=False, status='wrap_angle_exceeded')))
        return geometry

    def test_smaller_geometric_fallback_is_not_a_fixed_one_percent_candidate(self):
        geometry = self.fixed_area_geometry()
        triangle = np.array([[0., 0., 0.], [1.6, 0., 0.], [0., 1., 0.]])
        entry = geometry.make(0, 2., {0: triangle})
        self.assertAlmostEqual(entry['area_fraction'], .008)
        self.assertFalse(entry['valid'])
        self.assertEqual(entry['reason'], 'fixed_area_unavailable')
        geometry.clearance.check.assert_not_called()

    def test_invalid_target_sized_head_is_rejected_without_radius_rescue(self):
        geometry = self.fixed_area_geometry()
        triangle = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]])
        entry = geometry.make(0, 2., {0: triangle})
        self.assertAlmostEqual(entry['area_fraction'], .01)
        self.assertFalse(entry['valid'])
        self.assertEqual(entry['reason'], 'wrap_angle_exceeded')
        self.assertEqual(entry['contact']['radius_m'], 2.)
        geometry.clearance.check.assert_called_once()

    def test_rigid_transform_preserves_surface_and_material_identity(self):
        contact = dict(candidate_id='A', center_m=np.array([1., 2., 3.]),
            triangles_m=np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]),
            triangle_areas_m2=np.array([.5]), source_faces=np.array([7]), radius_m=.2)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', [.3, -.4, .7]).as_matrix()
        transform[:3, 3] = [.1, .5, -.2]
        moved = transform_contact(contact, transform)
        restored = transform_contact(moved, np.linalg.inv(transform))
        np.testing.assert_allclose(restored['triangles_m'], contact['triangles_m'], atol=1e-14)
        np.testing.assert_array_equal(moved['source_faces'], contact['source_faces'])
        self.assertEqual(moved['candidate_id'], 'A')
        self.assertEqual(moved['radius_m'], .2)

    def test_direction_menu_is_horizontal(self):
        catalogue = horizontal_catalogue(trimesh.creation.box())
        vectors = np.asarray(catalogue['vectors'])
        np.testing.assert_array_equal(vectors[:, 2], 0.)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.)

    def test_paths_can_leave_surface_but_cannot_cross_object_or_ground(self):
        mesh = trimesh.creation.box(extents=[1., 1., 1.]).apply_translation([0, 0, .5])
        paths = FreePaths(mesh, [[0, 0, 1, 0]], resolution=7)
        left, right = np.array([-.55, 0., .5]), np.array([.55, 0., .5])
        self.assertFalse(paths.clear_segments([left], [right])[0])
        self.assertTrue(set(paths.ports(left)) & set(paths.ports(right)))
        self.assertEqual(paths.ports(np.array([0., 0., -.1])), [])


if __name__ == '__main__':
    unittest.main()
