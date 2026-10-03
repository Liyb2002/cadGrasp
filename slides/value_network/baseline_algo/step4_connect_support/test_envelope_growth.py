"""Fixed guidance and deterministic legal greedy growth."""
import unittest
import numpy as np
from step4_connect_support.envelope_growth import EnvelopeGrow
from step4_connect_support import test_coverage_growth as fixtures
from step4_connect_support import build_coupled_saddle as S,growing_support as L

class EnvelopeTests(unittest.TestCase):
    def fixture(self):
        helper=fixtures.CoverageTests();base=helper.fixture([[.015,.015],[.06,.015],[.03,.06]])
        self.addCleanup(helper.doCleanups)
        g=EnvelopeGrow.__new__(EnvelopeGrow);g.__dict__.update(base.__dict__)
        g.timings={'contact_starts':0.};return g

    def test_guidance_does_not_expand_with_accepted_branches(self):
        g=self.fixture();g.envelope_lo=np.array([0.,0.,0.]);g.envelope_hi=np.array([.06,.06,.06])
        lo=np.array([.05,.01,.01]);hi=np.array([.08,.02,.02])
        first=g.overflow(lo,hi)
        g.occupied_lo=np.array([-.1,-.1,-.1]);g.occupied_hi=np.array([.2,.2,.2])
        self.assertEqual(first,g.overflow(lo,hi));self.assertAlmostEqual(first,.02)

    def test_repeated_growth_covers_demands_with_complete_cores(self):
        a=self.fixture();b=self.fixture();a.ground();b.ground()
        solid,ra=a.connect();_,rb=b.connect()
        self.assertEqual(ra['growth_journal'],rb['growth_journal'])
        self.assertEqual(ra['beam_width'],1);self.assertTrue(ra['envelope_fixed_during_growth'])
        self.assertFalse(ra['fixed_ground_targets'])
        self.assertEqual(len(L.components(solid)),1)
        self.assertTrue(a.coverage(a.floor_points(solid,0),0)[0])
        for _,core in a.core_solids:
            self.assertLessEqual(abs((core-solid).volume())*S.SCALE**3,8e-14)
        for rod in a.beams:self.assertTrue(a.legal(rod))

if __name__=='__main__':unittest.main()
