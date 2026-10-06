
import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys
import unittest
from pathlib import Path
import igl
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
from physics_guided_field_fast import FastObjectDistanceField
from physics_guided_field import ObjectDistanceField


class FastDistanceTests(unittest.TestCase):
    def test_oriented_box_sign_and_exact_distance(self):
        mesh=trimesh.creation.box()
        points=np.array([[0.,0.,0.],[.75,0.,0.],[.5,.5,0.],[.75,.75,0.],[-.4,.2,.1]])
        distances,*_=igl.signed_distance(points,np.asarray(mesh.vertices),np.asarray(mesh.faces),igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
        np.testing.assert_allclose(-distances,[.5,-.25,0.,-np.sqrt(.125),.1],atol=1e-12)

    def test_nonconvex_missing_quadrant_is_exterior(self):
        a=trimesh.creation.box([2.,1.,1.]);a.apply_translation([0.,-.5,0.])
        b=trimesh.creation.box([1.,2.,1.]);b.apply_translation([-.5,0.,0.])
        mesh=trimesh.boolean.union([a,b],engine='manifold')
        points=np.array([[.5,.5,0.],[-.5,-.5,0.],[-1.5,-.5,0.],[.25,-.25,0.]])
        distances,*_=igl.signed_distance(points,np.asarray(mesh.vertices),np.asarray(mesh.faces),igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
        np.testing.assert_allclose(-distances,[-.5,.5,-.5,.25],atol=1e-12)

    def test_cache_separation_and_trilinear_boundary_extension(self):
        mesh=trimesh.creation.box();fast=FastObjectDistanceField(mesh,resolution=16)
        reference=ObjectDistanceField.__new__(ObjectDistanceField,mesh,resolution=16)
        self.assertNotEqual(fast.key,reference.key)
        points=np.array([[.45,0.,0.],[.5,0.,0.],[.55,0.,0.],[.75,0.,0.]])
        np.testing.assert_allclose(fast.sample(points),[.05,0.,-.05,-.25],atol=1e-12)

if __name__=='__main__':unittest.main()
