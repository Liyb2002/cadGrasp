"""Prevent corner-cutting and non-progressing route shortcut reduction."""
import sys
from pathlib import Path
import unittest
import numpy as np
COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/x)for x in ['step5.2','step5.3','helper_func']]
from continuous_legs_safe_recovery import RouteGuide
from deterministic_base import box


class SafeRouteTests(unittest.TestCase):
    def test_every_raw_neighbor_is_axis_aligned(self):
        stock=box(np.zeros(3),np.full(3,.020))
        core=box(np.array([.014,.004,.004]),np.array([.019,.010,.010]))
        guide=RouteGuide(stock,core,pitch=.002)
        self.assertEqual(len(guide.directions),6)
        self.assertTrue((abs(guide.directions).sum(1)==1).all())

    def test_shortcuts_make_progress_around_a_corner(self):
        stock=box(np.zeros(3),np.array([.035,.022,.010]))
        obstacle=box(np.array([.014,0.,-.001]),np.array([.021,.014,.011]))
        stock=stock-obstacle
        core=box(np.array([.027,.003,.002]),np.array([.034,.008,.008]))
        guide=RouteGuide(stock,core,pitch=.002)
        path,_=guide.route(np.array([.004,.004,.005]))
        self.assertLess(len(path),100)
        self.assertGreater(path[:,1].max(),.014)
        for a,b in zip(path[1:-1],path[2:]):
            points=np.linspace(a,b,80)
            ids=np.rint((points-guide.origin)/guide.pitch).astype(int)
            self.assertTrue(guide.valid[tuple(ids.T)].all())


if __name__=='__main__':unittest.main()
