"""The continuous rewrite must retain successful discrete seating basins."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'helper_func'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'step4.2'))
import _bootstrap
import numpy as np
import trimesh
from continuous_support.complete_grid import position_rule
from whole_search.model import Layout
from continuous_support.objective import Evaluation
from juxtapose.reuse_search import proposals, balanced_subset, operation_key, juxtapose
from optimizer_reuse import preserve_sampled_feasibility


class ReuseJumpTests(unittest.TestCase):
    def setUp(self):
        self.layout=Layout(np.repeat(np.eye(4)[None],3,axis=0),
            np.asarray([[0.,0.,1.],[.8,0.,.6],[-.8,0.,.6]]),np.arange(3),(0,1,2))
        native=np.repeat(np.eye(4)[None],3,axis=0);native[:,0,3]=[0.,.15,-.15]
        state=SimpleNamespace(locks={(a,b):np.zeros(1,bool) for a in range(3) for b in range(3)},
                              coverage_counts={k:np.ones(1) for k in range(3)})
        model=SimpleNamespace(native=native,poses=['a','b','c'],mesh=trimesh.creation.box(),
            extent=1.,point_areas=np.ones(1),contact_delta=SimpleNamespace(state=lambda q:state),
            floor_normal=lambda q,k:np.array([0.,0.,1.]))
        def reseat(q,guest,host):
            trial=q.copy();trial.hosts[guest]=host
            trial.placements[guest]=np.linalg.inv(native[host])@native[guest]
            trial.directions[guest]=q.directions[host]
            return trial
        model.juxtapose=reseat;self.model=model
        self.current=Evaluation(self.layout,np.zeros(3),[],1.,0,np.array([1.,.1,.2]),np.ones(3))
        self.objective=SimpleNamespace(model=model)

    def test_original_native_midpoint_seed_is_not_discarded(self):
        rows=proposals(self.objective,self.current)
        chosen=[(q,d) for _,q,d in rows if d['guest']=='a' and d['host']=='b'
                and d['direction_host_weight']==.5
                and np.linalg.norm(d['horizontal_offset_fixture_m'])==0.]
        self.assertTrue(chosen)
        q,_=chosen[0];expected=self.layout.directions[0]+self.layout.directions[1]
        np.testing.assert_allclose(q.directions[0],expected/np.linalg.norm(expected))
        np.testing.assert_allclose(self.model.native[q.hosts[0]]@q.placements[0],self.model.native[0])

    def test_visit_memory_keeps_three_direction_basins_distinct(self):
        rows=proposals(self.objective,self.current)
        native=[row for row in rows if row[2]['guest']=='a' and row[2]['host']=='b'
                and np.linalg.norm(row[2]['horizontal_offset_fixture_m'])==0.]
        self.assertEqual(len({operation_key(self.layout,q,d,1.) for _,q,d in native}),3)

    def test_limited_screen_reaches_overlapping_offset_scales(self):
        rows=proposals(self.objective,self.current);selected=balanced_subset(rows,30)
        self.assertTrue(any(np.linalg.norm(d['horizontal_offset_fixture_m'])>.05 for _,_,d in selected))
        self.assertEqual({d['direction_host_weight'] for _,_,d in selected},{0.,.5,1.})

    def test_branch_refinement_has_all_pose_direction_freedom(self):
        rows=proposals(self.objective,self.current);self.model.contact_delta.commit=lambda q:None
        class Screen:
            seconds=0.
            def evaluate(self,q):return dict(loss=1.,sampled_guidance_only=True)
        obj=self.objective;obj._jump_screen=Screen()
        obj.evaluate=lambda q:Evaluation(q,np.zeros(3),[],1.,0,np.ones(3),np.ones(3))
        obj.covered=lambda r:False;obj.coverage_loss=lambda r:float(r.residual_loss.mean())
        obj.better=lambda a,b:False;owners=[]
        def refine(candidate,chosen):owners.append(chosen);return candidate,[]
        with patch('juxtapose.reuse_search.proposals',return_value=rows):
            _,report=juxtapose(obj,self.current,budget=12,full_budget=2,max_refined=2,refine=refine)
        self.assertEqual(owners,[None,None]);self.assertFalse(report['screening_is_acceptance'])

    def test_fine_position_rule_has_positive_torque_extremes_and_same_measure(self):
        points,weights=position_rule(2)
        self.assertTrue(np.all(weights>0));self.assertAlmostEqual(weights.sum(),1.)
        for vertex in np.eye(3):self.assertTrue(np.any(np.all(points==vertex,axis=1)))
        for axis in range(3):
            self.assertAlmostEqual(weights@points[:,axis],1/3)
            self.assertAlmostEqual(weights@(points[:,axis]**2),1/6)
        self.assertAlmostEqual(weights@(points[:,0]*points[:,1]),1/12)

    def test_volume_descent_retains_sampled_feasible_incumbent(self):
        before=self.current;layout=self.layout.copy();layout.placements[0,0,3]+=.01
        after=Evaluation(layout,np.ones(3),[],.9,0,np.zeros(3),np.zeros(3))
        obj=self.objective;obj.covered=lambda r:True
        obj.model.evaluate=lambda q:dict(masks={0:np.asarray([True,q.key()==self.layout.key()])},counts={0:1})
        kept,record=preserve_sampled_feasibility(obj,before,after,dict(accepted=True,operation='translation'))
        self.assertIs(kept,before);self.assertFalse(record['accepted'])
        self.assertTrue(record['sampled_original_load_guard_rejected'])


if __name__=='__main__':unittest.main()
