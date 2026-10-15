"""Current rounded surfaces: real root growth, retained envelope and thin cuts."""
import sys
from pathlib import Path
import unittest
import numpy as np
COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/x)for x in ['step5.3','step5.2','helper_func']]
from flared_leg_surface import flared_path
from stocky_legs_production import cross_section_widths
from continuous_legs import no_tunnels
from deterministic_base import box,SCALE


class CurrentFlareTests(unittest.TestCase):
    def test_root_is_broader_and_far_shaft_stays_compact(self):
        value=flared_path([[0.,0.,0.],[0.,0.,.080]],.0045,.0075,.020)
        root=value.slice(0.).to_polygons()[0];middle=value.slice(.040/SCALE).to_polygons()[0]
        self.assertGreater(np.ptp(np.asarray(root)[:,0])*SCALE,.014)
        width=np.ptp(np.asarray(middle)[:,0])*SCALE
        self.assertGreater(width,.0085);self.assertLess(width,.0092)
        self.assertTrue(no_tunnels(value))

    def test_zero_length_attachment_is_a_closed_root(self):
        self.assertTrue(no_tunnels(flared_path([[0.,0.,0.],[0.,0.,0.]],.0045,.007)))

    def test_previous_wider_shaft_envelope_is_retained(self):
        value=flared_path([[0.,0.,0.],[0.,0.,.120]],.00425,.006,.014,minimum_envelope=(.004,.0054))
        contour=value.slice(.030/SCALE).to_polygons()[0]
        width=np.ptp(np.asarray(contour)[:,0])*SCALE
        self.assertGreater(width,.010)

    def test_real_clipped_width_detects_a_thin_sheet(self):
        path=np.array([[0.,0.,0.],[0.,0.,.060]])
        value=flared_path(path,.005,.0075,.020)
        clipped=value^box(np.array([-.0005,-.020,-.020]),np.array([.0005,.020,.080]))
        widths=cross_section_widths(clipped,path)
        self.assertEqual(len(widths),5)
        self.assertTrue(all(abs(w-.001)<1e-7 for w in widths))

    def test_bent_surface_has_no_tunnel(self):
        self.assertTrue(no_tunnels(flared_path([[0.,0.,0.],[0.,0.,.022],[.018,0.,.040]],.0045,.007,.018)))


if __name__=='__main__':unittest.main()
