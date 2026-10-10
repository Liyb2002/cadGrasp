"""Cooperative reaction selection and bounded plateau acceptance."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from contact_recovery import cooperating_reactions,plateau_allowed,RecoveryTarget
from types import SimpleNamespace
from unittest.mock import patch

class RecoveryTests(unittest.TestCase):
    def state(self,mask):return dict(masks=[np.array(mask,bool)])
    def test_joint_missing_contacts_balance_torque(self):
        rays=np.array([[1,0,0,1,0,0,0],[0,1,0,-1,0,0,0]],float)
        target=np.array([1,1,0,0,0,0,0],float)
        reactions,error=cooperating_reactions(np.empty((0,7)),rays,target,np.ones(2))
        self.assertTrue(np.all(reactions>0))
        np.testing.assert_allclose((rays/np.linalg.norm(rays,axis=1)[:,None]).T@reactions,target)
        self.assertLess(error,1e-9)
    def test_existing_real_reactions_are_preserved_as_free_supply(self):
        target=np.array([1,0,0,0,0,0,0],float)
        reactions,_=cooperating_reactions(target[None,:],target[None,:],target,np.ones(1))
        self.assertEqual(reactions[0],0.)
    @patch('contact_recovery.NominalContactSweep')
    def test_full_sweep_pricing_and_deterministic_target_switch(self,model):
        # Identical reaction rays: normals cannot distinguish them. Only the
        # full trajectory detects the obstruction of the first contact.
        ray=np.array([1.,0,0,0,0,0,0])
        search=SimpleNamespace(points=np.zeros((2,3)),rays=[np.vstack([ray,ray])],
            ray_normals=np.array([[1.,0,0],[1.,0,0]]),states=[(SimpleNamespace(targets=ray[None,:]),None)],
            group=dict(poses=['pose_1']),clearance=None,length=.5,mesh=SimpleNamespace(extents=np.ones(3)),
            distance_model=SimpleNamespace(distances=lambda d:np.array([.1,0.])),
            choose_loads=lambda state:[(0,0)],projection=lambda *a:dict(loss=1.))
        state=dict(masks=[np.array([False])],supplies=[np.empty((0,7))])
        direction=np.array([[0.,0,1.]])
        preferred=RecoveryTarget(search,state,direction)
        self.assertEqual(preferred.ids.tolist(),[1])
        alternate=RecoveryTarget(search,state,direction,avoided=[1])
        self.assertEqual(alternate.ids.tolist(),[0])
    def test_plateau_requires_same_passing_loads(self):
        before=self.state([1,0]);after=self.state([0,1])
        self.assertFalse(plateau_allowed(before,after,1,.5,2,0))
    def test_plateau_requires_progress_and_bounds(self):
        s=self.state([1,0])
        self.assertTrue(plateau_allowed(s,s,1,.9,4,0))
        self.assertFalse(plateau_allowed(s,s,1,1,4,0))
        self.assertFalse(plateau_allowed(s,s,1,.9,13,0))
        self.assertFalse(plateau_allowed(s,s,1,.9,4,6))
    def test_joint_descent_stays_on_legal_hemisphere(self):
        # An actual normal at the floor boundary must be adjusted tangentially
        # without leaving the legal hemisphere; finite differences verify the
        # target derivative inside the optimizer.
        target=RecoveryTarget.__new__(RecoveryTarget)
        target.search=SimpleNamespace(normals=np.array([[0.,0.,1.]]))
        target.normals=np.array([[0.,1.,0.]])
        target.weights=np.ones(1);target.ids=np.array([0]);target.records=[];target.locked_poses=[]
        target.model=SimpleNamespace(linearize=lambda d,f:(np.zeros((1,1)),np.zeros((1,1,2))))
        start=np.array([[1.,0.,0.]])
        endpoint,stats=target.step(start)
        self.assertTrue(stats['valid_step'])
        self.assertLess(stats['derivative_relative_error'],1e-3)
        self.assertGreaterEqual(endpoint[0,2],-1e-12)
        self.assertLess(endpoint[0,1],0)
        np.testing.assert_allclose(np.linalg.norm(endpoint,axis=1),1.)
if __name__=='__main__':unittest.main()
