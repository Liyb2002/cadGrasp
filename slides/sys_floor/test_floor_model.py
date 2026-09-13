"""Regression checks for the slides' half-body-weight floor model.

Run with the cadgrasp environment:
    python -m unittest discover -s slides/sys_floor -p 'test_floor_model.py'
"""
import unittest
from types import SimpleNamespace

import numpy as np

import support_polygon as S
from on_the_floor import folds


class FloorModelTests(unittest.TestCase):
    def test_default_load_and_explicit_body_weights(self):
        q, com = [1, 0, 1], [0, 0, 1]
        p, reaction = S.cop(q, [0, 0, -1], com)
        np.testing.assert_allclose(p, [[1 / 3, 0]])
        np.testing.assert_allclose(reaction, [1.5])
        p, reaction = S.cop(q, [0, 0, -1], com, t=0.25)
        np.testing.assert_allclose(p, [[0.2, 0]])
        np.testing.assert_allclose(reaction, [1.25])

    def test_landings_balance_external_force_and_horizontal_moment(self):
        rng = np.random.default_rng(5)
        q = rng.normal(size=(300, 3))
        q[:, 2] = np.abs(q[:, 2]) + 0.1
        d = rng.normal(size=(300, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        com = np.array([0.3, -0.2, 1.0])
        for magnitude in (0.25, 0.5, 0.75):
            p, reaction = S.cop(q, d, com, t=magnitude)
            floor_force = -magnitude * d - S.DOWN
            floor_point = np.c_[p, np.zeros(len(p))]
            moment = (np.cross(com, S.DOWN) + np.cross(q, magnitude * d)
                      + np.cross(floor_point, floor_force))
            np.testing.assert_allclose(moment[:, :2], 0, atol=1e-12)
            np.testing.assert_allclose(reaction, floor_force[:, 2])
            self.assertGreaterEqual(reaction.min(), 1 - magnitude)

    def test_normal_reaction_extrema_include_cap_interior(self):
        alpha = np.radians(S.CONE_HALF_DEG)
        for outward, expected in (([0, 0, -1], 0.5),
                                  ([1, 0, 0], 1 - 0.5 * np.sin(alpha)),
                                  ([0, 0, 1], 1 + 0.5 * np.cos(alpha))):
            mesh = SimpleNamespace(face_normals=np.array([outward], float))
            r_min, _, loses_contact = S.exact_R_min(mesh, np.eye(4), [0])
            self.assertAlmostEqual(r_min, expected)
            self.assertFalse(loses_contact)
        # Explicit historical magnitude: upward push can cancel gravity at K=1.
        mesh = SimpleNamespace(face_normals=np.array([[0, 0, -1]], float))
        r_min, _, loses_contact = S.exact_R_min(mesh, np.eye(4), [0], t=1)
        self.assertEqual(r_min, 0)
        self.assertTrue(loses_contact)

    def test_half_weight_stereographic_formula_matches_moment_balance(self):
        rng = np.random.default_rng(19)
        u = rng.normal(size=(300, 2))
        square = np.sum(u * u, axis=1)
        direction = np.c_[2 * u, square - 1] / (1 + square[:, None])
        q = rng.normal(size=(300, 3))
        q[:, 2] = np.abs(q[:, 2]) + 0.1
        com = np.array([0.3, -0.2, 1.0])
        g = np.r_[com[:2], 0.]
        r = q - g
        projected = (g[:2] + ((1 - square[:, None]) * r[:, :2]
                     + 2 * r[:, 2, None] * u) / (3 + square[:, None]))
        balanced, _ = S.cop(q, direction, com)
        np.testing.assert_allclose(projected, balanced, atol=1e-12)

    def test_fold_circle_changes_with_load_magnitude(self):
        points = np.array([[0., 0., 1.]])
        com = np.array([0., 0., 2.])
        # r is vertical. The cone is centred at d_z=0.5, so it meets the
        # K=0.5 critical circle, but excludes the K=1 critical direction, up.
        outward = -np.array([[np.sqrt(3) / 2, 0., 0.5]])
        self.assertEqual(folds(points, outward, com), 1)
        self.assertEqual(folds(points, outward, com, t=1), 0)
        # A cone about straight up has the reverse classification.
        outward = np.array([[0., 0., -1.]])
        self.assertEqual(folds(points, outward, com), 0)
        self.assertEqual(folds(points, outward, com, t=1), 1)


if __name__ == '__main__':
    unittest.main()
