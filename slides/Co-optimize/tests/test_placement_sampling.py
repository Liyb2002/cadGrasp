import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from types import SimpleNamespace
from placement_sampling import separation_layout,fixture_transform,PlacementSearch
from physics_guided_geometry import tangent_frames
from co_common import transform_points

class PlacementTests(unittest.TestCase):
    def test_separated_layout_respects_each_floor_and_fixed_reference(self):
        rng=np.random.default_rng(8);normals=rng.normal(size=(6,3));normals/=np.linalg.norm(normals,axis=1)[:,None]
        offsets=separation_layout(normals,.4)
        np.testing.assert_allclose((offsets*normals).sum(axis=1),0.,atol=1e-12)
        np.testing.assert_array_equal(offsets[0],0.)
        dist=np.linalg.norm(offsets[:,None]-offsets[None,:],axis=2);dist+=np.eye(6)*100
        self.assertGreaterEqual(dist.min(),.4)
    def test_translation_preserves_native_wrench_point_and_moment_arm(self):
        angle=.7;T=np.eye(4);T[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
        T[:3,3]=[.1,.2,.3];points=np.array([[.01,.03,.05],[-.02,.07,.03]])
        offset=np.array([.05,-.02,0.])
        np.testing.assert_allclose(transform_points(points+offset,fixture_transform(T,offset)),transform_points(points,T),atol=1e-14)
    def test_candidates_include_both_tools_and_stay_floor_legal(self):
        search=PlacementSearch.__new__(PlacementSearch);search.n=3;search.normals=np.array([[0,0,1],[1,0,0],[0,1,0]],float)
        search.frames=tangent_frames(search.normals);search.rng=np.random.default_rng(42);search.targets=[(1,0,None),(2,0,None)]
        rows=search.candidates(search.normals,np.zeros((3,3)),0,16)
        kinds={kind for kind,d,o in rows};self.assertIn('translation',kinds);self.assertIn('direction',kinds);self.assertIn('coherent-direction',kinds)
        for kind,d,o in rows:
            self.assertGreaterEqual(float((d*search.normals).sum(axis=1).min()),-1e-12)
            np.testing.assert_allclose((o*search.normals).sum(axis=1),0.,atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(d,axis=1),1.,atol=1e-12)
            np.testing.assert_array_equal(o[0],0.)

if __name__=='__main__':unittest.main()
