"""Check the Y-up force/moment mapping used in the current floor diagrams."""
import unittest
import numpy as np
from presentation import landings


class LandingTests(unittest.TestCase):
    def test_parallel_downward_forces_have_weighted_floor_intercept(self):
        com = np.array([.03, .12, -.02])
        q = np.array([-.04, .08, .06])
        p, normal = landings(q, [0., -.5, 0.], com)
        np.testing.assert_allclose(p[[0, 2]], (com[[0, 2]]+.5*q[[0, 2]])/1.5)
        self.assertEqual(p[1], 0.)
        self.assertEqual(normal, 1.5)

    def test_general_loads_balance_horizontal_moments_in_y_up_world(self):
        rng = np.random.default_rng(20260913)
        q = rng.uniform(-.1, .1, (200, 3))
        q[:, 1] += .15
        push = rng.normal(size=(200, 3))
        push *= rng.uniform(0., .5, (200, 1))/np.linalg.norm(push, axis=1)[:, None]
        com = np.array([.025, .08, -.003])
        p, normal = landings(q, push, com)
        gravity = np.array([0., -1., 0.])
        reaction = -(gravity+push)
        moment = np.cross(com, gravity)+np.cross(q, push)+np.cross(p, reaction)
        np.testing.assert_allclose(moment[:, [0, 2]], 0., atol=1e-14)
        np.testing.assert_allclose(normal, reaction[:, 1])
        self.assertTrue(np.all(normal >= .5))

    def test_zero_process_force_and_loss_of_compression(self):
        com = np.array([.03, .12, -.02])
        p, normal = landings([1., 2., 3.], [0., 0., 0.], com)
        np.testing.assert_allclose(p, [.03, 0., -.02])
        self.assertEqual(normal, 1.)
        with self.assertRaises(ValueError):
            landings([0., .1, 0.], [0., 1., 0.], com)


if __name__ == '__main__':
    unittest.main()
