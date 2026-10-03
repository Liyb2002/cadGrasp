"""Local postprocessing retains a border/rib and skips undersized feet."""
import unittest
import numpy as np
from shapely.geometry import Polygon,Point

from step4_connect_support.baseline_current import hollow_feet as H


class PocketTests(unittest.TestCase):
    def test_separate_pockets_keep_border_and_central_rib(self):
        xy=np.array([[-.03,-.02],[.03,-.02],[.03,.02],[-.03,.02]])
        boundary=Polygon(xy)
        pockets=[Polygon(p) for p in H.pockets(xy)]
        self.assertEqual(len(pockets),2)
        self.assertGreaterEqual(pockets[0].distance(pockets[1]),H.RIB-1e-12)
        for pocket in pockets:
            self.assertGreaterEqual(pocket.distance(boundary.boundary),H.WALL-1e-12)
            self.assertFalse(pocket.covers(Point(0,0)))
        self.assertLess(sum(p.area for p in pockets),boundary.area*.65)

    def test_small_foot_stays_solid(self):
        xy=np.array([[-.004,-.004],[.004,-.004],[.004,.004],[-.004,.004]])
        self.assertEqual(H.pockets(xy),[])


if __name__=='__main__':unittest.main()
