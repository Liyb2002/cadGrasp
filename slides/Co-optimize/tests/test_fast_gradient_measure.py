"""Regression: near-feasible uncovered regions must retain their measure."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from continuous_support.adaptive_fast_gradient import demand_strata,AdaptiveGradientSearch
from continuous_support.fast_gradient import turn
from whole_search.model import Layout
from types import SimpleNamespace
from unittest.mock import Mock


class FastGradientMeasureTests(unittest.TestCase):
    def test_rare_failed_demands_are_not_lost_and_covered_region_is_retained(self):
        targets=np.arange(32768)[:,None];mask=np.ones(32768,bool);mask[100:294]=False
        points,weights=demand_strata(targets,mask)
        selected=points[:,0].astype(int)
        self.assertEqual(set(range(100,294)),set(selected[~mask[selected]]))
        self.assertTrue(mask[selected].any());self.assertTrue(np.all(weights>0))
        self.assertAlmostEqual(weights[~mask[selected]].sum(),194/32768)
        self.assertAlmostEqual(weights[mask[selected]].sum(),1-194/32768)
        self.assertAlmostEqual(weights.sum(),1.)

    def test_both_large_regions_keep_original_measure(self):
        targets=np.arange(32768)[:,None];mask=np.arange(32768)%3!=0
        points,weights=demand_strata(targets,mask);selected=points[:,0].astype(int)
        self.assertAlmostEqual(weights[mask[selected]].sum(),mask.mean())
        self.assertAlmostEqual(weights[~mask[selected]].sum(),1-mask.mean())

    def test_spherical_step_preserves_ground_legality_and_original_placement(self):
        layout=Layout(np.eye(4)[None],np.array([[1.,0.,1e-6]]),np.array([0]),(0,))
        layout.directions/=np.linalg.norm(layout.directions,axis=1)[:,None]
        model=SimpleNamespace(floor_normal=lambda q,k:np.array([0.,0.,1.]))
        trial=turn(model,layout,0,np.array([0.,0.,-.1]))
        self.assertGreaterEqual(trial.directions[0,2],0.)
        self.assertAlmostEqual(np.linalg.norm(trial.directions[0]),1.)
        np.testing.assert_array_equal(trial.placements,layout.placements)

    def test_near_feasible_contact_diagnostic_does_not_repeat_seat_search(self):
        current=dict(masks={0:np.array([True,False])})
        search=AdaptiveGradientSearch(SimpleNamespace(),Path('.'),iterations=10)
        search.refine=Mock(return_value=current);search.fine_refine=Mock(return_value=current)
        search.rescue=Mock(side_effect=AssertionError('Must not repeat a seat for one conservative diagnostic'))
        self.assertIs(search.solve_frontier(current),current)
        self.assertTrue(search.fine_refine.called);self.assertFalse(search.rescue.called)


if __name__=='__main__':unittest.main()
