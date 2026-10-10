import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from contact_event_steps import minimum_step

class ContactEventTests(unittest.TestCase):
    def test_minimum_translation_is_computed_from_boundary(self):
        step,stats=minimum_step([[1.,0.],[0.,1.]],[.2,.3],2)
        np.testing.assert_allclose(step,[.2,.3],atol=1e-8)
        self.assertLess(stats['violation_after'],1e-12)
    def test_large_required_motion_is_only_advanced_one_small_step(self):
        step,stats=minimum_step([[1.,0.]],[3.],2)
        np.testing.assert_allclose(step,[1.,0.],atol=1e-8)
        self.assertAlmostEqual(stats['required_norm'],3.)
        self.assertLess(stats['violation_after'],stats['violation_before'])
    def test_conflicting_boundary_branches_are_not_reported_feasible(self):
        self.assertIsNone(minimum_step([[1.,0.],[-1.,0.]],[1.,1.],2))

if __name__=='__main__':unittest.main()
