"""Analytic geometry and integration cases, independent of the object demos."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import cubature
import coverage as R


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.lib = R.native()

    def evaluate(self, angles, coefficients, blockers=()):
        coefficients = np.ascontiguousarray(coefficients, dtype=float).reshape(-1, 3, 4)
        out = np.zeros((len(angles), 2))
        vertices, offsets = [], [0]
        for blocker in blockers:
            vertices.extend(blocker)
            offsets.append(len(vertices))
        self.lib.evaluate(
            len(angles), np.ascontiguousarray(angles), 1, 1,
            np.array([len(coefficients)], np.int32), np.array([0], np.int64), coefficients.ravel(),
            np.array([[0 if len(coefficients) else 1]], np.int32), np.ones(1),
            np.array([0, len(blockers)], np.int32), np.array(offsets, np.int32),
            np.array(vertices, float).reshape(-1, 3), np.eye(2).ravel(), np.zeros(2), out)
        return out

    def test_polygon_area_includes_partial_triangles(self):
        examples = [([], 1.), ([[1., 0., -.5]], .75),
                    ([[1., 1., -.5]], .25), ([[1., 0., 1.]], 0.)]
        for planes, expected in examples:
            data = np.array(planes, float).reshape(-1, 3)
            self.assertAlmostEqual(self.lib.clipped_area(len(data), data), expected, places=13)

    def test_solid_angle_integral_matches_closed_form(self):
        # x+y <= cos(psi) cuts a triangle with relative area cos(psi)^2.
        # Integrating in solid angle gives (1+c+c²)/3, c=cos(theta).
        coeff = np.array([[[0., -1., 0., 0.], [1., -1., 0., 0.], [1., -1., 0., 0.]]])
        result = cubature(lambda x: self.evaluate(x, coeff), [0., 0.],
                          [R.C.THETA, 2*np.pi], rule='gk21', rtol=1e-11, atol=1e-13)
        c = np.cos(R.C.THETA)
        np.testing.assert_allclose(result.estimate, [(1+c+c*c)/3, 1.], atol=1e-12)

    def test_occluder_union_does_not_double_count_overlap(self):
        psi, phi = .2, 0.
        direction_shift = np.tan(psi)
        shadow = np.array([[0., 0.], [.5, 0.], [0., .5]])
        blocker = np.c_[shadow-np.array([direction_shift, 0.]), np.ones(3)]
        angle = np.array([[psi, phi]])
        density = np.sin(psi)/(2*np.pi*(1-np.cos(R.C.THETA)))
        once = self.evaluate(angle, [], [blocker])[0]/density
        twice = self.evaluate(angle, [], [blocker, blocker])[0]/density
        np.testing.assert_allclose(once, [.75, .75], atol=1e-12)
        np.testing.assert_allclose(twice, once, atol=1e-12)

    def test_contact_cone_halfspaces_recover_known_orthant(self):
        full = np.vstack([np.eye(6), np.ones(6), np.arange(1., 7.)])
        with patch.object(R.C, 'patch_vertex_columns', return_value=full):
            H, _, _, error = R.contact_halfspaces(None, None)
        self.assertLess(error, 1e-9)
        self.assertLessEqual((H@np.arange(1., 7.)).max(), 1e-10)
        for i in range(6):
            demand = np.ones(6); demand[i] = -.1
            self.assertGreater((H@demand).max(), .01)

    def test_rescaled_vector_integral_and_complement_keep_original_units(self):
        cap_area = 2*np.pi*(1-np.cos(R.C.THETA))
        def f(x):
            density = np.sin(x[:, 0])/cap_area
            return density[:, None]*np.array([1e-4, 1-2e-3, 1.])
        result = R.integrate(f, 'gk15', 1e-6, np.array([False, True]))
        np.testing.assert_allclose(result.estimate, [1e-4, 2e-3, 1.], atol=1e-12)
        np.testing.assert_allclose(sum(r.estimate for r in result.regions), result.estimate, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
