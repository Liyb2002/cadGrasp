"""A tiny sphere cap must fail generation, not wait for final acceptance."""
import unittest
import numpy as np
import trimesh
from step3_scheculer import absolute_floor_recovery as R

class FloorTests(unittest.TestCase):
    def test_small_penetration_rejected_without_cutting_the_complete_core(self):
        R.activate();F=R.A.F
        grow=F.Grow.__new__(F.Grow)
        grow.bases=np.array([np.eye(3)]);grow.offsets=np.zeros((1,3))
        grow.mask=F.F.bounded_space(dict(min_m=[-.01,-.01,0.],max_m=[.01,.01,.02]),grow.bases,grow.offsets)
        mesh=trimesh.creation.icosphere(subdivisions=2,radius=.0026)
        mesh.vertices[:,2]+=.002599
        sphere=F.S.solid(mesh)
        # The cap is below the existing volume tolerance: this was the gap.
        self.assertLess(abs(float((sphere-grow.mask).volume()))*F.S.SCALE**3,8e-14)
        self.assertFalse(grow.legal(sphere))
        mesh.vertices[:,2]+=.000501
        self.assertTrue(grow.legal(F.S.solid(mesh)))
if __name__=='__main__':unittest.main()
