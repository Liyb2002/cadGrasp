"""Work still locks points that a changed exit direction newly releases."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from whole_search.model import Layout
from continuous_support.geometry_cache import GeometryCache


class Ray:
    def __init__(self):self.body_calls=0;self.exit_calls=0
    def contains_points(self,points):self.body_calls+=1;return np.zeros(len(points),bool)
    def intersects_location(self,origins,directions,multiple_hits=False):
        self.exit_calls+=1
        if directions[0,0]>0:return origins[:1]+[.1,0,0],np.array([0]),np.array([0])
        return np.zeros((0,3)),np.zeros(0,int),np.zeros(0,int)


class Work:
    def __init__(self,first=True):self.first=first;self.calls=[]
    def contains_points(self,points):
        self.calls.append(points.copy());return points[:,0]<.5 if self.first else points[:,0]>.5


class SharedGeometryTests(unittest.TestCase):
    def setUp(self):
        self.ray=Ray();self.work=[Work(),Work(False)]
        self.model=SimpleNamespace(points=np.array([[0.,0.,0.],[1.,0.,0.]]),
            point_normals=np.array([[0.,0.,1.]]*2),epsilon=1e-6,length=10.,ray=self.ray,work_rays=self.work)
        self.layout=Layout(np.array([np.eye(4)]*2),np.array([[-1.,0.,0.]]*2),np.arange(2),(0,1))
        self.cache=GeometryCache(self.model)

    def test_changed_exit_newly_queries_previously_hidden_work_point(self):
        np.testing.assert_array_equal(self.cache.locks(self.layout,0,0),[True,False])
        self.assertEqual(len(self.work[0].calls[0]),1)
        changed=self.layout.copy();changed.directions[0]=[1.,0.,0.]
        np.testing.assert_array_equal(self.cache.locks(changed,0,0),[True,False])
        self.assertEqual(self.work[0].calls[-1][0,0],0.)

    def test_shared_body_and_exit_keep_pose_specific_work_regions(self):
        np.testing.assert_array_equal(self.cache.locks(self.layout,0,0),[True,False])
        np.testing.assert_array_equal(self.cache.locks(self.layout,0,1),[True,True])
        self.assertEqual(self.ray.body_calls,1);self.assertEqual(self.ray.exit_calls,1)


if __name__=='__main__':unittest.main()
