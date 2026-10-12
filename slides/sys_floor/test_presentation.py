"""Check the Z-up force/moment mapping used in the current floor diagrams."""
import unittest
import numpy as np
from pathlib import Path
import importlib.util
_spec = importlib.util.spec_from_file_location('floor_presentation', Path(__file__).with_name('presentation.py'))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
landings = _module.landings


class LandingTests(unittest.TestCase):
    def test_parallel_downward_forces_have_weighted_floor_intercept(self):
        com = np.array([.03, .12, -.02])
        q = np.array([-.04, .08, .06])
        p, normal = landings(q, [0., 0., -.5], com)
        np.testing.assert_allclose(p[[0, 1]], (com[[0, 1]]+.5*q[[0, 1]])/1.5)
        self.assertEqual(p[2], 0.)
        self.assertEqual(normal, 1.5)

    def test_general_loads_balance_horizontal_moments_in_z_up_world(self):
        rng = np.random.default_rng(20260913)
        q = rng.uniform(-.1, .1, (200, 3))
        q[:, 2] += .15
        push = rng.normal(size=(200, 3))
        push *= rng.uniform(0., .5, (200, 1))/np.linalg.norm(push, axis=1)[:, None]
        com = np.array([.025, .08, -.003])
        p, normal = landings(q, push, com)
        gravity = np.array([0., 0., -1.])
        reaction = -(gravity+push)
        moment = np.cross(com, gravity)+np.cross(q, push)+np.cross(p, reaction)
        np.testing.assert_allclose(moment[:, [0, 1]], 0., atol=1e-14)
        np.testing.assert_allclose(normal, reaction[:, 2])
        self.assertTrue(np.all(normal >= .5))

    def test_zero_process_force_and_loss_of_compression(self):
        com = np.array([.03, .12, -.02])
        p, normal = landings([1., 2., 3.], [0., 0., 0.], com)
        np.testing.assert_allclose(p, [.03, .12, 0.])
        self.assertEqual(normal, 1.)
        with self.assertRaises(ValueError):
            landings([0., .1, 0.], [0., 0., 1.], com)


if __name__ == '__main__':
    unittest.main()
