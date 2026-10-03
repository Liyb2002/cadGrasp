"""Joint growth can land before merging heads and preserves actual coverage."""
import unittest
import numpy as np
from step4_connect_support.joint_growth import JointGrow
from step4_connect_support.test_coverage_growth import CoverageTests
from step4_connect_support import build_coupled_saddle as S,growing_support as L

class JointTests(unittest.TestCase):
    def fixture(self):
        helper=CoverageTests();base=helper.fixture([[.015,.015],[.06,.015],[.03,.06]])
        self.addCleanup(helper.doCleanups)
        g=JointGrow.__new__(JointGrow);g.__dict__.update(base.__dict__);g.beam_width=4
        return g

    def test_joint_ground_can_precede_head_connectivity(self):
        g=self.fixture();g.ground();solid,report=g.connect()
        self.assertTrue(report['ground_started_before_head_connectivity'])
        self.assertEqual(len(L.components(solid)),1)
        self.assertTrue(g.coverage(g.floor_points(solid,0),0)[0])
        self.assertFalse(report['fixed_ground_targets'])
        self.assertIsNone(report['ground_minimum_relative_gain'])
        for _,core in g.core_solids:
            self.assertLessEqual(abs((core-solid).volume())*S.SCALE**3,8e-14)

    def test_joint_growth_repeats_identical_decisions(self):
        a=self.fixture();b=self.fixture();a.ground();b.ground()
        _,ra=a.connect();_,rb=b.connect()
        self.assertEqual(ra['growth_journal'],rb['growth_journal'])
        self.assertEqual(ra['complete_candidates'],rb['complete_candidates'])

if __name__=='__main__':unittest.main()
