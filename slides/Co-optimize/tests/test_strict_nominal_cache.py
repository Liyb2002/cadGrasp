"""A microscopic positive leading sweep still removes shared seating."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.model import Layout
from continuous_support.strict_nominal_cache import StrictNominalCache


class StrictNominalTests(unittest.TestCase):
    def setUp(self):
        mesh=trimesh.creation.box(extents=[.05,.04,.03])
        self.model=SimpleNamespace(mesh=mesh,length=.5,sources=np.arange(len(mesh.faces)))
        self.cache=StrictNominalCache(self.model)
        self.layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([[1e-12,0.,1.]]*2),np.array([0,1]),(0,1))

    def test_positive_leading_thinner_than_ray_offset_is_still_locked(self):
        with patch('continuous_support.geometry_cache.GeometryCache.locks',return_value=np.zeros(len(self.model.sources),bool)):
            locks=self.cache.locks(self.layout,0,1)
        normal=self.model.mesh.face_normals
        self.assertTrue(locks[normal[:,0]>.9].all())
        self.assertFalse(locks[normal[:,0]<-.9].any())

    def test_exactly_tangent_surface_is_not_added_to_the_sweep(self):
        self.layout.directions[:]=[0.,0.,1.]
        with patch('continuous_support.geometry_cache.GeometryCache.locks',return_value=np.zeros(len(self.model.sources),bool)):
            locks=self.cache.locks(self.layout,0,1)
        self.assertFalse(locks[abs(self.model.mesh.face_normals[:,0])>.9].any())

    def test_distinct_seats_do_not_share_a_surface_face_rule(self):
        self.layout.placements[1,0,3]=.001
        with patch('continuous_support.geometry_cache.GeometryCache.locks',return_value=np.zeros(len(self.model.sources),bool)):
            self.assertFalse(self.cache.locks(self.layout,0,1).any())


if __name__=='__main__':unittest.main()
