import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from types import SimpleNamespace
from contact_lock_chain import ContactLocks
from contact_area_sensitivity import area_changes,area_gradient,ContactAreaSensitivity

class AreaSensitivityTests(unittest.TestCase):
    def test_union_requires_every_pose_to_release_and_counts_new_locks(self):
        locks=ContactLocks.__new__(ContactLocks)
        # One region remains locked by pose2; one releases; one newly locks.
        locks.locked=lambda d: np.array([False,False,True])
        previous=dict(directions=np.array([[0.,0.,1.],[0.,1.,0.]]),
                      locks=np.array([[True,True,False],[True,False,False]]))
        d=previous['directions'].copy();d[0]=[1.,0.,0.]
        union,count,active=locks.update(d,previous)
        np.testing.assert_array_equal(active,[False,True,False])
        changes=area_changes(np.array([False,False,True]),active,np.array([2.,3.,5.]))
        self.assertEqual(changes['released_area_m2'],3.)
        self.assertEqual(changes['newly_locked_area_m2'],5.)
        self.assertEqual(changes['available_area_m2'],3.)
    def test_losing_reaction_increases_wrench_deficit(self):
        model=ContactAreaSensitivity.__new__(ContactAreaSensitivity)
        model.floors=[np.empty((0,7))];model.rays=[np.eye(7)[:2]]
        target=np.array([1.,1.,0,0,0,0,0])
        self.assertAlmostEqual(model.loss(dict(active=np.array([True,True])),0,target),0.)
        self.assertAlmostEqual(model.loss(dict(active=np.array([True,False])),0,target),.5)
    def test_gradient_never_calls_full_geometry(self):
        search=SimpleNamespace(normals=np.array([[0.,0.,1.]]),states=[(SimpleNamespace(targets=np.array([[1.,0,0,0,0,0,0]])),None)])
        model=SimpleNamespace(areas=np.array([1.]),state=lambda d,previous=None:dict(directions=d,active=np.array([True])),
            loss=lambda state,k,w: float(1+state['directions'][0,0]))
        g,probes=area_gradient(search,model,np.array([[0.,0.,1.]]),dict(pose_index=0,load_index=0),1e-5)
        np.testing.assert_allclose(g,[[0.,-1.]],atol=1e-8)
        self.assertEqual(len(probes),2)

if __name__=='__main__':unittest.main()
