import unittest
import numpy as np
import trimesh

from .first_layer import first_layer_faces, incidence_quality
from .step1_3 import _visible_faces


class FirstLayerTests(unittest.TestCase):
    def test_small_occluder_between_the_old_four_rays_is_not_missed(self):
        mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],
            [.1,.1,1],[.12,.1,1],[.1,.12,1]], faces=[[0,1,2],[3,4,5]], process=False)
        self.assertTrue(_visible_faces(mesh, [0], np.array([[0.,0.,1.]]))[0])
        np.testing.assert_array_equal(first_layer_faces(mesh, [0,0,1]), [False,True])
        # The occluder is a whole metre beyond the target plane: it still
        # blocks even if a later work-window test excludes the occluder.

    def test_rear_sheet_and_back_faces_are_excluded(self):
        mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],
            [0,0,1],[1,0,1],[0,1,1]], faces=[[0,1,2],[3,4,5]], process=False)
        np.testing.assert_array_equal(first_layer_faces(mesh, [0,0,1]), [False,True])
        np.testing.assert_array_equal(first_layer_faces(mesh, [0,0,-1]), [False,False])

    def test_adjacent_triangles_and_grazing_visible_surface_are_kept(self):
        mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]], faces=[[0,1,2],[0,2,3]], process=False)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(80), [1,0,0]))
        self.assertTrue(first_layer_faces(mesh, [0,0,1]).all())
        quality = incidence_quality(mesh, [0,1], np.array([0,0,1]))
        self.assertAlmostEqual(quality['maximum_incidence_angle_deg'], 80.)
        self.assertFalse(quality['used_as_hard_angle_limit'])

    def test_depth_order_is_local_not_centroid_order(self):
        # A sloped sheet crosses a horizontal one; each is partly occluded.
        mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],
            [0,0,-.1],[1,0,.1],[0,1,.1]], faces=[[0,1,2],[3,4,5]], process=False)
        np.testing.assert_array_equal(first_layer_faces(mesh, [0,0,1]), [False,False])


if __name__ == '__main__':
    unittest.main()
