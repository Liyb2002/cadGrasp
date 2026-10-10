"""Exclude unreleasable self locks and retain a joint jump branch."""
import sys,unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.model import Layout
from continuous_support.releasable_blockers import ReleasableBlockerModel
from continuous_support.paired_blocker_gradient import PairedBlockerGradientSearch


class PairedBlockerTests(unittest.TestCase):
    def test_other_poses_get_no_credit_for_a_ray_locked_by_the_owner(self):
        layout=Layout(np.repeat(np.eye(4)[None],3,axis=0),np.array([[0.,0.,1.]]*3),np.arange(3),(0,1,2))
        e=np.eye(7);locks={(i,j):np.zeros(2,bool) for i in layout.active for j in layout.active}
        locks[0,0]=np.array([True,False]);locks[0,1]=np.array([True,False]);locks[0,2]=np.array([False,True])
        state=SimpleNamespace(available={k:np.zeros(2,bool) for k in layout.active},
                              coverage_counts={k:np.ones(2,int) for k in layout.active},locks=locks,
                              lock_counts={0:np.array([2,1]),1:np.ones(2,int),2:np.ones(2,int)})
        model=SimpleNamespace(contact_delta=SimpleNamespace(state=lambda q:state),quadrature_epoch=0,
            coordinate_cache=OrderedDict(),coordinate_seconds=0.,points=np.zeros((2,3)),sources=np.array([0,1]),
            allowed=[np.array([0,1])]*3,point_areas=np.array([1000.,1.]),
            point_rays=[np.array([10*e[0],e[0]])]*3,
            supply_at_points=lambda k,f,layout=None:(e[1:2],np.array([0])),adaptive_demands={},
            demand_screen=SimpleNamespace(demands=[(e[0:1],np.ones(1),1.),(e[1:2],np.ones(1),1.),(e[1:2],np.ones(1),1.)]))
        _,scores=ReleasableBlockerModel.useful_coordinates(model,dict(layout=layout))
        self.assertEqual(scores[1],0.)
        self.assertGreater(scores[2],0.)
        self.assertEqual(model.jump_ray_weights[0][0],0.)

    def test_joint_jump_is_not_removed_by_better_single_candidate_ties(self):
        model=SimpleNamespace(proxy=Mock(return_value=dict(loss=.2,sum_loss=.2,span_m=0.)))
        search=PairedBlockerGradientSearch(model,Path('.'),finalists=3)
        search.screen_budget=96;search.pending_pair_rows=[('juxtapose-pair','pair',{})]
        singles=[(v,v,0.,'juxtapose',str(v),{},dict(loss=v)) for v in [.1,.15,.18]]
        with patch('continuous_support.blocker_gradient.BlockerGradientSearch.shortlist',return_value=(singles,95)):
            selected,count=search.shortlist([],[])
        self.assertIn('juxtapose-pair',[r[3] for r in selected])
        self.assertEqual(len(selected),3);self.assertEqual(count,96)
        self.assertEqual(search.screen_budget,96)


if __name__=='__main__':unittest.main()
