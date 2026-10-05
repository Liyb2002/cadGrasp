"""The shared ring-ray intersection handles withdrawal and misses outside."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step5_connect_support import ring as R


class RingRayTests(unittest.TestCase):
    def test_boundary_ray_uses_withdrawal_and_can_miss_from_outside(self):
        square=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        ring=R.make(square,1.,1.)
        np.testing.assert_allclose(R.ray_exit(ring,[0,0],[-1,0])[0],[-1,0])
        self.assertIsNone(R.ray_exit(ring,[-2,0],[-1,0]))


if __name__ == '__main__': unittest.main()
