"""Routing window recovery must expand search without relaxing physics."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import trimesh
from step3_scheculer import operation_navigation_recovery as N,operation_dsl as F


class NavigationTests(unittest.TestCase):
    def test_expands_finite_window_and_keeps_all_pose_floor_halfspaces(self):
        def initialize(g,*args,**kwargs):
            g.navigation_window=dict(min_m=[0,0,0],max_m=[.08,.08,.08])
            g.bases=np.array([np.eye(3)]);g.offsets=np.zeros((1,3))
            g.case=SimpleNamespace(root_solids=[]);g.forbidden=F.F.md.Manifold()
        with patch.object(N.R.RecoveryGrow,'__init__',initialize):g=N.RoomGrow(None)
        self.assertAlmostEqual(g.navigation_window['max_m'][0],.13)
        self.assertEqual(g.navigation_window['min_m'][2],0.)
        sphere=trimesh.creation.icosphere(subdivisions=2,radius=.0026);sphere.apply_translation([.115,.04,.03])
        self.assertTrue(g.legal(F.S.solid(sphere)))
        sphere.apply_translation([0,0,-.04])
        self.assertFalse(g.legal(F.S.solid(sphere)))

if __name__=='__main__':unittest.main()
