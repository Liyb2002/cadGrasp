
import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
import numpy as np
from physics_guided_cone import cone_projection, missing_values


class ConeProjectionTests(unittest.TestCase):
    def test_feasible_target_duplicates_and_zero_rays(self):
        rays = np.vstack([np.eye(7), np.eye(7)[0] * 2, np.zeros(7)])
        target = np.arange(1., 8.)
        result = cone_projection(rays, target)
        self.assertLess(result['loss'], 1e-22)
        np.testing.assert_allclose(result['coefficients'] @ rays, target)
        self.assertLess(result['kkt_max_violation'], 1e-10)

    def test_target_gradient_matches_finite_difference(self):
        rays = np.eye(7)[:3]
        target = np.array([1., -2., .5, 3., -1., 2., .8])
        scale = np.array([1., 2., 3., .5, 1., 2., .7])
        result = cone_projection(rays, target, scale)
        eps = 1e-6
        derivative = []
        for axis in np.eye(7):
            plus = cone_projection(rays, target + eps * axis, scale)['loss']
            minus = cone_projection(rays, target - eps * axis, scale)['loss']
            derivative.append((plus - minus) / (2 * eps))
        np.testing.assert_allclose(derivative, -result['dual'], atol=1e-8)

    def test_missing_generator_value_is_actual_force_derivative(self):
        target = np.array([1., 2., 0., 0., 0., 0., 0.])
        result = cone_projection(np.eye(7)[:1], target)
        missing = np.eye(7)[1]
        value = missing_values([missing], result)[0]
        self.assertEqual(value, 2.)
        eps = 1e-6
        loss_at_eps = .5 * np.sum((result['residual'] + eps * missing)**2)
        self.assertAlmostEqual((loss_at_eps - result['loss']) / eps, -value, places=5)
        self.assertLess(cone_projection(np.eye(7)[:2], target)['loss'], 1e-22)

    def test_positive_area_scaling_does_not_change_cone(self):
        rng = np.random.default_rng(42)
        rays = rng.normal(size=(6, 7))
        target = rng.normal(size=7)
        plain = cone_projection(rays, target)
        scaled = cone_projection(rays * np.logspace(-3, 3, 6)[:, None], target)
        self.assertAlmostEqual(plain['loss'], scaled['loss'], places=11)

    def test_empty_cone_and_invalid_metric(self):
        result = cone_projection([], np.ones(7))
        self.assertEqual(result['loss'], 3.5)
        with self.assertRaises(ValueError):
            cone_projection([], np.ones(7), np.zeros(7))

    def test_large_cone_column_generation_kkt(self):
        rng = np.random.default_rng(9)
        rays = rng.uniform(.05, 1., size=(600, 7))
        target = np.array([-1., 2., 1., 0., .5, 3., 1.])
        result = cone_projection(rays, target)
        self.assertEqual(result['solver'], 'column_generation_nnls')
        self.assertLess(result['kkt_max_violation'], 1e-8)
        self.assertTrue(np.all(result['coefficients'] >= 0))


if __name__ == '__main__':
    unittest.main()
