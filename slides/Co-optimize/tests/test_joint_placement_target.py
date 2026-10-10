import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from types import SimpleNamespace
from joint_placement_target import JointPlacementTarget
from local_placement_sampling import LocalPlacementSearch
from contact_boundary_model import prism_planes
from physics_guided_geometry import tangent_frames
from unittest.mock import patch

class JointTargetTests(unittest.TestCase):
    def test_computed_translation_step_minimizes_local_cost_inside_one_mm(self):
        search=LocalPlacementSearch.__new__(LocalPlacementSearch);search.n=2
        search.normals=np.array([[0.,0,1.],[0.,0,1.]])
        search.frames=tangent_frames(search.normals)
        search.boundary=SimpleNamespace(evaluate=lambda d,o,t:dict(loss=0.,sum_loss=0.))
        target=JointPlacementTarget.__new__(JointPlacementTarget);target.search=search
        wanted=np.array([.0003,.0004,0.]);target.value=lambda d,o:float(np.sum(((o[1]-wanted)/.001)**2))
        target.selected=[(0,np.array([0]),np.array([1.]))];target.key=((0,(0,)),)
        options,stats=target.propose(search.normals,np.zeros((2,3)),'translation',[1],[],{})
        best=min(options,key=lambda row:row[0]);np.testing.assert_allclose(best[3][1],wanted,atol=1e-7)
        self.assertLess(best[5]['guidance_after'],best[5]['guidance_before'])
        self.assertLessEqual(np.linalg.norm(best[3][1]),.001)
    @patch('joint_placement_target.igl.signed_distance',return_value=(np.array([1.]),None,None,None))
    def test_shadow_cost_accounts_for_new_interference(self,ignored):
        triangle=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        planes=prism_planes(triangle,np.array([0.,0,1.]),np.array([0.,0,1.]))
        search=SimpleNamespace(n=2,mesh=SimpleNamespace(vertices=np.zeros((3,3)),faces=np.array([[0,1,2]])),
            boundary=SimpleNamespace(shadows=lambda d:(None,[planes])))
        target=JointPlacementTarget.__new__(JointPlacementTarget);target.search=search
        p=np.array([[.2,.2,.2]]);n=np.array([[1.,0,0.]])
        directions=np.array([[0.,0,1.],[0.,0,1.]])
        clear=target.costs(0,p,n,directions,np.array([[0.,0,0.],[2.,0,0.]]))
        locked=target.costs(0,p,n,directions,np.zeros((2,3)))
        # Owner remains the same blocker in both; second pose newly locks.
        self.assertGreater(locked[0],clear[0])

if __name__=='__main__':unittest.main()
