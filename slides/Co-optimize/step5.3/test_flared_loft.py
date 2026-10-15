"""Real shaft/root dimensions, thin clipping and closed bent-leg regressions."""
import sys
from pathlib import Path
import unittest
import numpy as np
COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/x)for x in ['step5.3','step5.2','helper_func']]
from flared_loft import flared_path
from stocky_legs import cross_section_widths
from continuous_legs import no_tunnels
from deterministic_base import box,SCALE


class FlaredLoftTests(unittest.TestCase):
    def test_root_flare_is_local_and_larger_than_shaft(self):
        value=flared_path(np.array([[0.,0.,0.],[0.,0.,.080]]),.0045,.0075,.020)
        root=value.slice(0.).to_polygons()[0];middle=value.slice(.040/SCALE).to_polygons()[0]
        self.assertGreater(np.ptp(np.asarray(root)[:,0])*SCALE,.014)
        shaft=np.ptp(np.asarray(middle)[:,0])*SCALE
        self.assertGreater(shaft,.0085);self.assertLess(shaft,.0092)
        self.assertTrue(no_tunnels(value))

    def test_measurement_detects_a_thin_clipped_shaft(self):
        path=np.array([[0.,0.,0.],[0.,0.,.060]])
        value=flared_path(path,.005,.0075,.020)
        clip=box(np.array([-.0005,-.020,-.020]),np.array([.0005,.020,.080]))
        widths=cross_section_widths(value^clip,path)
        self.assertEqual(len(widths),5)
        self.assertTrue(all(abs(w-.001)<1e-7 for w in widths))

    def test_bent_flare_is_a_single_unperforated_solid(self):
        value=flared_path(np.array([[0.,0.,0.],[0.,0.,.022],[.018,0.,.040]]),.0045,.007,.018)
        self.assertTrue(no_tunnels(value))


if __name__=='__main__':unittest.main()
