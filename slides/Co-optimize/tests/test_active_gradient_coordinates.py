"""Inactive demands still participate when only a few coordinates move."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.model import Layout
from continuous_support.active_fast_gradient import ActiveGradientSearch


class ActiveCoordinateTests(unittest.TestCase):
    def test_loss_on_an_unchanged_owner_controls_a_blockers_gradient(self):
        n=5;layout=Layout(np.repeat(np.eye(4)[None],n,axis=0),np.array([[0.,0.,1.]]*n),np.arange(n),tuple(range(n)))
        def proxy(q):
            x=q.directions[0,0]
            losses=[.5*(x-.01)**2,0.,0.,100*(x+.01)**2,1000.]
            return dict(loss=sum(losses)/n,residual_loss=losses)
        model=SimpleNamespace(poses=list('abcde'),extent=1.,native=np.repeat(np.eye(4)[None],n,axis=0),
                              proxy=proxy,gradient_seconds=0.,floor_normal=lambda q,k:np.array([0.,0.,1.]),
                              useful_coordinates=lambda current:(list(range(n)),dict.fromkeys(range(n),1.)))
        search=ActiveGradientSearch(model,Path('.'))
        rows,_=search.local_proposals(dict(layout=layout),[],{})
        directions=search.last_coordinate_selection['direction_poses']
        self.assertEqual(directions,['e','a','b','c']);self.assertNotIn('d',directions)
        trial=next(q for kind,q,detail in rows if kind=='direction-gradient-joint')
        # Owner a alone asks for +x, but the more strongly affected unchanged
        # owner d needs -x. The derivative must include d's loss.
        self.assertLess(trial.directions[0,0],0.)
        np.testing.assert_array_equal(trial.directions[3],layout.directions[3])
        self.assertLess(proxy(trial)['loss'],proxy(layout)['loss'])


if __name__=='__main__':unittest.main()
