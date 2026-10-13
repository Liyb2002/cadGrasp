"""Regressions for perforation rejection and obstacle-respecting route guides."""
import sys
from pathlib import Path
import unittest
import numpy as np
import manifold3d as md

COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/x)for x in ['step5.2','step5.3','helper_func']]
from continuous_legs import no_tunnels, raster, RouteGuide, capsule
from deterministic_base import box, SCALE, unpack, volume


class ContinuousLegTests(unittest.TestCase):
    def test_perforated_blade_is_rejected(self):
        blade=box(np.array([0.,0.,0.]),np.array([.025,.018,.006]))
        hole=md.Manifold.cylinder(.020/SCALE,.002/SCALE,circular_segments=24)
        hole=hole.translate((.012/SCALE,.009/SCALE,-.005/SCALE))
        self.assertTrue(no_tunnels(blade))
        self.assertFalse(no_tunnels(blade-hole))

    def test_enclosed_cavity_is_not_filled_or_certified_as_a_rod(self):
        block=box(np.zeros(3),np.full(3,.020))
        cavity=md.Manifold.sphere(.003/SCALE,24).translate(tuple(np.full(3,.010)/SCALE))
        self.assertFalse(no_tunnels(block-cavity))

    def test_slice_grid_preserves_a_real_through_hole(self):
        block=box(np.zeros(3),np.array([.020,.020,.008]))
        hole=box(np.array([.007,.007,-.001]),np.array([.013,.013,.009]))
        guide=raster(block-hole,np.array([.001,.001,.001]),(10,10,3),.002)
        self.assertFalse(guide[5,5,1])
        self.assertTrue(guide[1,1,1])

    def test_route_detours_around_an_obstacle(self):
        stock=box(np.zeros(3),np.array([.040,.024,.012]))
        obstacle=box(np.array([.016,0.,-.001]),np.array([.024,.015,.013]))
        stock=stock-obstacle
        core=box(np.array([.032,.003,.003]),np.array([.039,.010,.010]))
        guide=RouteGuide(stock,core,pitch=.0015)
        path,record=guide.route(np.array([.005,.006,.006]))
        self.assertGreater(float(path[:,1].max()),.015)
        self.assertGreater(record['path_length_m'],.030)
        for a,b in zip(path[:-1],path[1:]):
            centers=np.linspace(a,b,40)
            ids=np.rint((centers-guide.origin)/guide.pitch).astype(int)
            self.assertTrue(guide.valid[tuple(ids.T)].all())


if __name__=='__main__':unittest.main()
