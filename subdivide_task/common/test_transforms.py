import unittest
import numpy as np
from .transforms import rigid_transform,inverse,points,directions

class TransformTests(unittest.TestCase):
    def test_frame_direction_translation_and_normals(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        transform = rigid_transform(rotation, [2., 3., 4.])
        np.testing.assert_allclose(points(transform, [[1., 0, 0]]), [[2., 4., 4.]])
        np.testing.assert_allclose(directions(transform, [[1., 0, 0]]), [[0., 1., 0.]])
        np.testing.assert_allclose(points(inverse(transform), [[2., 4., 4.]]), [[1., 0., 0.]])

    def test_rejects_reflection_scaling_and_nonfinite_transforms(self):
        for rotation in (np.diag([-1., 1, 1]), np.eye(3)*2, np.eye(3)*np.nan):
            with self.assertRaises(ValueError):
                rigid_transform(rotation, [0, 0, 0])
        with self.assertRaises(ValueError):
            inverse(np.eye(4)*2)

