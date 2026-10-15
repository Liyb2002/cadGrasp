"""Acute routed legs remain solid while the previous shaft is enlarged."""
import sys
from pathlib import Path
import unittest
import numpy as np

COOPT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COOPT / name) for name in ['step5.3', 'step5.2', 'helper_func']]
from stocky_legs_capsule_recovery_v3 import flared_path
from continuous_legs import capsule, no_tunnels
from deterministic_base import union, volume


class SharpCapsuleTests(unittest.TestCase):
    def test_enlarged_bent_solid_retains_previous_shaft(self):
        path = np.array([[0., 0., 0.], [0., .015, 0.], [.010, .015, 0.],
                         [.010, .035, .005], [.026, .035, .005]])
        old = union([capsule(a, b, .003) for a, b in zip(path[:-1], path[1:])])
        wider = flared_path(path, .0033, .00363)
        self.assertTrue(no_tunnels(wider))
        self.assertLess(volume(old - wider), 1e-12)
        self.assertGreater(volume(wider), volume(old))


if __name__ == '__main__':
    unittest.main()
