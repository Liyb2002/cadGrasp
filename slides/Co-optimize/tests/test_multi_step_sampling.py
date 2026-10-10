import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from multi_step_sampling import shortlist_multiscale,improvement_allowed,MultiStepSamplingSearch
from joint_placement_target import JointPlacementTarget
from local_placement_sampling import LocalPlacementSearch
from physics_guided_geometry import tangent_frames
from types import SimpleNamespace

class MultiStepTests(unittest.TestCase):
    def test_shortlist_keeps_both_tools_and_different_step_sizes(self):
        rows=[]
        for tool in ('direction','translation'):
            for f in (1.,.5,.25,.125,.0625):
                rows.append(((),tool,None,None,dict(loss=1-f,sum_loss=1-f),dict(step_fraction=f)))
        chosen=shortlist_multiscale(rows,6)
        for tool in ('direction','translation'):
            fractions=[r[5]['step_fraction'] for r in chosen if r[1]==tool]
            self.assertEqual(len(set(fractions)),3)

    def test_proxy_or_count_gain_does_not_override_worst_gap_or_protection(self):
        self.assertFalse(improvement_allowed(1.,.5,False,0.,.2))
        self.assertFalse(improvement_allowed(1.,1.,True,0.,.2))
        self.assertFalse(improvement_allowed(1.,.5,True,.21,.2))
        self.assertTrue(improvement_allowed(1.,.5,True,.19,.2))

    def test_cumulative_caps_and_native_floor(self):
        search=MultiStepSamplingSearch.__new__(MultiStepSamplingSearch)
        search.normals=np.tile([0.,0,1.],(2,1));search.initial_directions=search.normals.copy()
        d=search.initial_directions;o=np.zeros((2,3))
        self.assertIsNone(search.layout_limit(d,o))
        o[1,0]=.0101;self.assertEqual(search.layout_limit(d,o),'cumulative translation cap')
        o[1]=[0,0,.0001];self.assertEqual(search.layout_limit(d,o),'translation leaves native floor plane')
        o[:]=0;o[0,0]=.0001;self.assertEqual(search.layout_limit(d,o),'reference pose moved')

    def test_actual_footprint_breaks_near_equal_gain_tie(self):
        search=MultiStepSamplingSearch.__new__(MultiStepSamplingSearch)
        search.initial_directions=np.tile([0.,0,1.],(2,1));d=search.initial_directions;o=np.zeros((2,3))
        compact=search.trial_rank(.5005,1.,dict(maximum_expansion_fraction=.01),d,o)
        large=search.trial_rank(.5001,1.,dict(maximum_expansion_fraction=.1),d,o)
        self.assertLess(compact,large)

    def test_gradient_candidate_has_multiple_computed_step_sizes(self):
        search=LocalPlacementSearch.__new__(LocalPlacementSearch);search.n=2
        search.normals=np.array([[0.,0,1.],[0.,0,1.]])
        search.frames=tangent_frames(search.normals);search.step_fractions=MultiStepSamplingSearch.step_fractions
        search.boundary=SimpleNamespace(evaluate=lambda d,o,t:dict(loss=0.,sum_loss=0.))
        target=JointPlacementTarget.__new__(JointPlacementTarget);target.search=search
        wanted=np.array([.0003,.0004,0.]);target.value=lambda d,o:float(np.sum(((o[1]-wanted)/.001)**2))
        target.selected=[(0,np.array([0]),np.array([1.]))];target.key=((0,(0,)),)
        options,_=target.propose(search.normals,np.zeros((2,3)),'translation',[1],[],{})
        self.assertTrue({1.,.5,.25,.125,.0625}.issubset({r[5]['step_fraction'] for r in options}))
        self.assertTrue(all(r[5]['guidance_after']<r[5]['guidance_before'] for r in options))
        self.assertTrue(all(np.linalg.norm(r[3][1])<=.004 for r in options))

if __name__=='__main__':unittest.main()
