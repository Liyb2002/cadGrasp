"""Boundary-terminal recovery must preserve real connected, solid material."""
import sys
from pathlib import Path
import unittest
import numpy as np
import manifold3d as md

COOPT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COOPT / name) for name in ['step5.3', 'step5.2', 'helper_func']]
import stocky_legs_resumed_profiles as R
from deterministic_base import box, volume, SCALE
from continuous_legs import no_tunnels


class BoundaryTerminalTests(unittest.TestCase):
    def test_clipped_corner_with_small_positive_probe_is_accepted(self):
        builder = R.Builder.__new__(R.Builder)
        builder.domain = box(np.zeros(3), np.full(3, .1))
        current = box(np.array([.025, 0., 0.]), np.array([.045, .012, .012]))
        point = np.full(3, -.0001)
        proposal = R.flared_path([np.array([.035, .006, .006]), point], .0045, .007)
        probe = md.Manifold.sphere(.0002/SCALE, 12).translate(tuple(point/SCALE))
        overlap = volume(proposal ^ builder.domain ^ probe)
        self.assertGreater(overlap, 1e-18)
        self.assertLess(overlap, 1e-14)
        chosen = builder.part(proposal, point, current)
        self.assertIsNotNone(chosen)
        self.assertTrue(no_tunnels(chosen[1]))

    def test_disconnected_material_is_rejected(self):
        builder = R.Builder.__new__(R.Builder)
        builder.domain = box(np.zeros(3), np.full(3, .1))
        current = box(np.full(3, .07), np.full(3, .09))
        point = np.array([.015, .015, .015])
        proposal = R.flared_path([np.array([.025, .015, .015]), point], .0045, .007)
        self.assertIsNone(builder.part(proposal, point, current))


if __name__ == '__main__':
    unittest.main()
