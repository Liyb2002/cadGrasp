"""Passing work/body blockers must enter the same discrete jump budget."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.model import Layout
from continuous_support.blocker_gradient import BlockerGradientSearch


class BlockerGradientTests(unittest.TestCase):
    def test_passing_blockers_enter_before_failed_guests_stop_improving(self):
        layout=Layout(np.repeat(np.eye(4)[None],4,axis=0),np.array([[0.,0.,1.]]*4),np.arange(4),tuple(range(4)))
        masks={k:np.array([k!=0]) for k in layout.active}
        current=dict(layout=layout,masks=masks,counts={str(k):int(v.sum()) for k,v in masks.items()})
        complete=dict(current,masks={k:np.array([True]) for k in layout.active},counts={str(k):1 for k in layout.active})
        model=SimpleNamespace(poses=list('abcd'),set_working_measure=Mock(),
                              proxy=Mock(return_value=dict(residual_loss=[1.,0.,0.,0.])),
                              useful_coordinates=Mock(return_value=([2,1,3,0],{0:1.,1:9.,2:10.,3:0.})))
        search=BlockerGradientSearch(model,Path('.'),iterations=1)
        search.refine=Mock(side_effect=lambda state,*a,**kw:state);search.rescue=Mock(return_value=complete);search.record=Mock()
        self.assertIs(search.solve_frontier(current),complete)
        self.assertEqual(search.rescue.call_args.args[1],[0,2,1])
        self.assertEqual(search.rescue.call_args.kwargs['blocking_guests'],[2,1])
        # Even though these two tasks themselves pass, their geometry matters.
        self.assertTrue(current['masks'][2].all());self.assertTrue(current['masks'][1].all())


if __name__=='__main__':unittest.main()
