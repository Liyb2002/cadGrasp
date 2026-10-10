"""Seat choice need not force the anchor's fixture orientation."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.model import Layout
from whole_search.reuse_first import registered
from continuous_support.state_seat_gradient import StateSeatGradientSearch


class StateSeatTests(unittest.TestCase):
    def test_explicit_new_seat_preserves_state_rotation_and_world_height(self):
        native=np.repeat(np.eye(4)[None],2,axis=0)
        c=s=np.sqrt(.5)
        native[0,:3,:3]=[[1.,0.,0.],[0.,c,-s],[0.,s,c]]
        placements=np.repeat(np.eye(4)[None],2,axis=0);placements[1,0,3]=.2
        layout=Layout(placements,np.array([[0.,0.,1.]]*2),np.arange(2),(0,1))
        model=SimpleNamespace(mesh=trimesh.creation.box(),extent=1.,native=native,poses=['p0','p1'])
        model.floor_normal=lambda q,k:native[q.hosts[k],:3,:3].T@np.array([0.,0.,1.])
        search=StateSeatGradientSearch(model,Path('.'))
        rows=search.current_state_seats(dict(layout=layout,masks={0:np.array([False]),1:np.array([True])}),[0])
        self.assertTrue(rows)
        for kind,trial,detail in rows:
            self.assertEqual(kind,'juxtapose-current-state')
            self.assertEqual(detail['host'],'p1')
            self.assertEqual(trial.hosts[0],layout.hosts[0])
            self.assertFalse(registered(trial,0))
            np.testing.assert_array_equal(trial.placements[0,:3,:3],layout.placements[0,:3,:3])
            before=native[trial.hosts[0]]@layout.placements[0]
            after=native[trial.hosts[0]]@trial.placements[0]
            self.assertAlmostEqual(before[2,3],after[2,3])
            self.assertGreaterEqual(detail['body_bbox_overlap_fraction'],.03)

    def test_new_seat_and_pair_share_three_branches_and_96_screens(self):
        model=SimpleNamespace(proxy=Mock(return_value=dict(loss=.2,sum_loss=.2,span_m=0.)))
        search=StateSeatGradientSearch(model,Path('.'),finalists=3);search.screen_budget=96
        rows=[('juxtapose-current-state','seat'+str(k),dict(guest_index=k,host='p1')) for k in range(3)]
        old=[(v,v,0.,kind,kind+str(v),{},dict(loss=v)) for v,kind in
             [(.1,'juxtapose'),(.15,'juxtapose'),(.3,'juxtapose-pair')]]
        def ordinary(*args):
            self.assertEqual(search.screen_budget,93)
            return old,93
        with patch('continuous_support.state_seat_gradient.BalancedGradientSearch.shortlist',side_effect=ordinary):
            chosen,count=search.shortlist(rows,[])
        self.assertEqual(count,96)
        self.assertEqual(len(chosen),3)
        self.assertIn('juxtapose-pair',[r[3] for r in chosen])
        self.assertIn('juxtapose-current-state',[r[3] for r in chosen])
        self.assertEqual(search.screen_budget,96)


if __name__=='__main__':unittest.main()
