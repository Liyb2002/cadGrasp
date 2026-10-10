"""Regression for world-horizontal joint updates and plateau progress."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.model import Layout
from continuous_support.cached_fast_gradient import CachedGradientSearch
from continuous_support.descent_fast_gradient import DescentGradientSearch
from continuous_support.efficient_fast_gradient import EfficientGradientSearch


class JointGradientRepairTests(unittest.TestCase):
    def test_shared_translation_preserves_world_height_with_different_hosts(self):
        native=np.repeat(np.eye(4)[None],2,axis=0)
        native[1,:3,:3]=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([[0.,0.,1.]]*2),np.array([1,0]),(0,1))
        def loss(q):
            points=np.array([native[q.hosts[k],:3,:3]@q.placements[k,:3,3] for k in q.active])
            return dict(loss=float(np.sum((points[:,:2]-[.1,.2])**2)))
        model=SimpleNamespace(native=native,extent=1.,poses=['a','b'],proxy=loss,gradient_seconds=0.)
        search=CachedGradientSearch(model,Path('.'))
        with patch('continuous_support.adaptive_fast_gradient.AdaptiveGradientSearch.local_proposals',return_value=([],loss(layout))):
            rows,_=search.local_proposals(dict(layout=layout),[],{},translation_guests=(0,1))
        gradients=[q for kind,q,_ in rows if kind=='translation-gradient-coherent']
        self.assertTrue(gradients)
        for trial in gradients:
            displacements=np.array([native[trial.hosts[k],:3,:3]@trial.placements[k,:3,3] for k in trial.active])
            np.testing.assert_allclose(displacements[:,2],0.,atol=1e-14)
            np.testing.assert_allclose(displacements[0],displacements[1],atol=1e-14)
            self.assertLess(loss(trial)['loss'],loss(layout)['loss'])

    def test_equal_failure_count_prefers_demand_progress_before_material(self):
        model=SimpleNamespace(volume_delta=None,proxy=lambda q:dict(loss=q))
        search=DescentGradientSearch(model,Path('.'))
        def state(loss,volume):
            return dict(layout=loss,masks={0:np.array([True,False])},volume_cm3=volume,
                        maximum_projected_footprint_m2=0.)
        with patch('whole_search.reuse_first.juxtaposed_count',return_value=1):
            self.assertLess(search.score(state(.1,12.)),search.score(state(.2,10.)))
            fully=state(1.,15.);fully['masks'][0][:]=True
            self.assertLess(search.score(fully),search.score(state(.01,1.)))

    def test_infeasible_flat_loss_does_not_spend_rounds_on_material_ties(self):
        current=dict(layout=1.,masks={0:np.array([False])},counts={'0':0},serial=1,
                     volume_cm3=10.,maximum_projected_footprint_m2=0.)
        candidate=dict(current,volume_cm3=1.,serial=2)
        model=SimpleNamespace(volume_delta=None,proxy=lambda q:dict(loss=1.),
                              evaluate=Mock(return_value=candidate),commit=Mock())
        search=DescentGradientSearch(model,Path('.'))
        search.targets=Mock(return_value=([],{},[]))
        search.local_proposals=Mock(return_value=([],dict(loss=1.)))
        search.shortlist=Mock(return_value=([(1.,1.,0.,'direction-sample',1.,{},dict(loss=1.))],1))
        search.record=Mock()
        with patch('whole_search.reuse_first.juxtaposed_count',return_value=1):
            result=search.refine(current,3)
        self.assertIs(result,current);self.assertEqual(model.evaluate.call_count,1)

    def test_passed_new_guest_still_repairs_other_damaged_pose(self):
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([[0.,0.,1.]]*2),np.array([1,1]),(0,1))
        def state(a,b,serial):
            masks={0:np.array([a]),1:np.array([b])}
            return dict(layout=layout,masks=masks,counts={str(k):int(v.sum()) for k,v in masks.items()},serial=serial)
        current=state(False,True,1);branch=state(True,False,2);repaired=state(True,True,3)
        model=SimpleNamespace(poses=['a','b'],evaluate=Mock(return_value=branch),commit=Mock())
        search=DescentGradientSearch(model,Path('.'))
        search.targets=Mock(return_value=([],{},[]));search.juxtapose_proposals=Mock(return_value=[])
        search.shortlist=Mock(return_value=([(1.,1.,0.,'juxtapose',layout,dict(guest_index=0),{})],1))
        search.refine=Mock(return_value=repaired);search.score=lambda r,a=None:(sum(int((~v).sum()) for v in r['masks'].values()),)
        search.checkpoint=Mock();search.record=Mock()
        self.assertIs(search.rescue(current,[0]),repaired)
        search.refine.assert_called_once()
        self.assertIn(0,search.refine.call_args.kwargs['translation_guests'])

    def test_first_improving_gradient_avoids_other_full_load_checks(self):
        current=dict(layout=1.,masks={0:np.array([False,False])},counts={'0':0},serial=1,
                     volume_cm3=10.,maximum_projected_footprint_m2=0.)
        candidate=dict(current,layout=.4,masks={0:np.array([True,False])},counts={'0':1},serial=2)
        model=SimpleNamespace(volume_delta=None,proxy=lambda q:dict(loss=q),
                              evaluate=Mock(return_value=candidate),commit=Mock())
        search=EfficientGradientSearch(model,Path('.'))
        search.targets=Mock(return_value=([],{},[]));search.local_proposals=Mock(return_value=([],dict(loss=1.)))
        search.shortlist=Mock(return_value=([
            (.3,.3,0.,'direction-sample',.3,{},dict(loss=.3)),
            (.4,.4,0.,'direction-gradient',.4,{},dict(loss=.4)),
            (.5,.5,0.,'translation-gradient',.5,{},dict(loss=.5))],3))
        search.record=Mock();search.checkpoint=Mock()
        with patch('whole_search.reuse_first.juxtaposed_count',return_value=1):
            self.assertIs(search.refine(current,1),candidate)
        model.evaluate.assert_called_once_with(.4)


if __name__=='__main__':unittest.main()
