"""Growth must connect around an obstacle without inheriting the whole block."""
from types import SimpleNamespace
import unittest

import numpy as np
from shapely.geometry import MultiPoint
import trimesh

from step4_connect_support.growing_support import Grow, components, prune_ground
from step4_connect_support import build_coupled_saddle as S


def block(low, high):
    low, high = np.asarray(low), np.asarray(high)
    mesh = trimesh.creation.box(high-low)
    mesh.apply_translation((low+high)/2)
    return mesh


class GrowingSupportTests(unittest.TestCase):
    def test_sparse_growth_routes_around_wall_and_preserves_both_roots(self):
        barrier = S.solid(block([.028,0,0],[.032,.05,.04]))
        growth = Grow.__new__(Grow)
        growth.group = SimpleNamespace(name='synthetic')
        growth.mask = S.solid(block([0,0,0],[.06]*3))-barrier
        growth.reference = S.unpack(growth.mask)
        growth.bead = trimesh.creation.icosphere(subdivisions=1,radius=.004).vertices
        growth.terminals, growth.paths = [], []
        growth.graph(.008)
        roots = [block([x,.028,.008],[x+.004,.032,.012]) for x in (.008,.048)]
        for index, root in enumerate(roots):
            growth.attach(S.solid(root),root.vertices,str(index))
        solid, _ = growth.connect()
        self.assertEqual(len(components(solid)),1)
        self.assertLess(solid.volume(),.25*growth.mask.volume())
        self.assertLess(abs((solid^barrier).volume())*S.SCALE**3,8e-14)
        for root in roots:
            self.assertLess(abs((S.solid(root)-solid).volume())*S.SCALE**3,8e-14)
        self.assertTrue(S.unpack(solid).is_watertight)

    def test_pruning_ground_retains_every_demand_and_removes_unneeded_corners(self):
        angle=np.arange(8)*np.pi/4
        ground=np.c_[np.cos(angle),np.sin(angle)]
        demands=np.array([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
        selected=prune_ground(ground,demands)
        self.assertLess(len(selected),len(ground))
        self.assertLess(MultiPoint(demands).convex_hull.difference(MultiPoint(selected).convex_hull).area,1e-12)


if __name__ == '__main__':
    unittest.main()
