"""Regression cases for long/oblique access rays and whole working triangles."""
import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"helper_func"))
import _bootstrap
from co_common import trimesh
from work_access import unpack_solid
from work_access import WorkAccess


class WorkAccessTests(unittest.TestCase):
    def setUp(self):
        self.mesh=trimesh.Trimesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],process=False)
        self.access=WorkAccess(self.mesh,[0],30.)

    def test_far_and_oblique_rays_are_reserved_without_length_limit(self):
        # The old5mm band misses both of these actual allowed rays.
        points=[[.2,.2,10000.],[.2+10*np.tan(np.radians(29.)),.2,10.]]
        np.testing.assert_array_equal(self.access.contains_points(points),[True,True])

    def test_whole_source_area_is_used_and_inward_halfspace_is_clear(self):
        points=[[.01,.98,.05],[-.2,.2,.01],[.2,.2,-.01]]
        np.testing.assert_array_equal(self.access.contains_points(points),[True,False,False])

    def test_finite_mesh_matches_search_within_seed_and_caps_beyond_seed(self):
        points=np.array([[.2,.2,.2],[.9,.05,.1],[1.3,.2,.1],[-.1,.2,.5]])
        length=self.access.required_length(points,minimum=.5)
        self.assertGreater(length,.5)
        solid=self.access.solid(length)
        ray=trimesh.ray.ray_pyembree.RayMeshIntersector(unpack_solid(solid))
        np.testing.assert_array_equal(ray.contains_points(points),self.access.contains_points(points))

    def test_circular_boundary_is_inside_conservative_envelope(self):
        angles=np.linspace(0,2*np.pi,259,endpoint=False)
        points=np.c_[.2+10*np.tan(np.radians(30))*np.cos(angles),
                     .2+10*np.tan(np.radians(30))*np.sin(angles),np.full(len(angles),10.)]
        self.assertTrue(self.access.contains_points(points).all())
        outside=np.array([[20.,.2,10.],[.2,20.,10.]])
        self.assertFalse(self.access.contains_points(outside).any())


if __name__=='__main__':unittest.main()
