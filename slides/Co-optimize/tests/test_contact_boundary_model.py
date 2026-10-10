import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from types import SimpleNamespace
from contact_boundary_model import available_polygons,prism_planes
from placement_sampling import colocated_allowed
from local_placement_sampling import LocalPlacementSearch
from physics_guided_geometry import tangent_frames

class BoundaryTests(unittest.TestCase):
    def test_area_moves_continuously_when_boundary_crosses_triangle(self):
        triangle=np.array([[[0.,0,0],[1.,0,0],[0,1.,0]]])
        def area(c):return available_polygons(triangle,np.array([[[-c,1-c,-c]]]),np.array([True]))[2][0]
        self.assertAlmostEqual(area(.3),.3-.3**2/2)
        self.assertAlmostEqual((area(.300001)-area(.299999))/2e-6,.7,places=7)
    def test_all_blockers_clip_contact_and_new_lock_can_remove_area(self):
        triangle=np.array([[[0.,0,0],[1.,0,0],[0,1.,0]]])
        levels=np.array([[[-.3,.7,-.3]],[[-.4,-.4,.6]]])
        points,indices,area=available_polygons(triangle,levels,np.array([True]))
        self.assertAlmostEqual(area[0],.12)
        self.assertTrue(np.all(points[:,0]<=.3+1e-12));self.assertTrue(np.all(points[:,1]<=.4+1e-12))
    def test_prism_planes_contain_vertices_and_exclude_outside_point(self):
        tri=np.array([[0.,0,0],[1.,0,0],[0,1.,0]]);delta=np.array([.2,.3,1.])
        n,b=prism_planes(tri,np.array([0.,0,1.]),delta)
        self.assertLessEqual(float((np.vstack([tri,tri+delta])@n.T-b).max()),1e-12)
        self.assertGreater(float((np.array([0.,0,-1.])@n.T-b).max()),0.)
    def test_colocated_contacts_require_all_exit_directions(self):
        normals=np.eye(3);allowed=np.arange(3);directions=np.array([[0.,0,1.],[1.,0,0.]])
        offsets=np.zeros((2,3));np.testing.assert_array_equal(colocated_allowed(normals,allowed,0,directions,offsets),[1])
        offsets[1]=[.01,0,0];np.testing.assert_array_equal(colocated_allowed(normals,allowed,0,directions,offsets),[0,1])
    def test_step_is_computed_inside_small_trust_region_without_geometry(self):
        search=LocalPlacementSearch.__new__(LocalPlacementSearch);search.n=2
        search.normals=np.array([[0.,0,1.],[0.,0,1.]])
        search.frames=tangent_frames(search.normals)
        target=np.array([.0003,.0004,0.])
        def evaluate(d,o,targets):
            loss=float(np.sum(((o[1]-target)/.001)**2))
            return dict(loss=loss,sum_loss=loss)
        search.boundary=SimpleNamespace(evaluate=evaluate)
        directions=search.normals.copy();offsets=np.zeros((2,3));base=evaluate(directions,offsets,[])
        proposed,info=search.propose(directions,offsets,[],'translation',[1],base)
        self.assertTrue(proposed)
        best=min(proposed,key=lambda row:row[0]);np.testing.assert_allclose(best[3][1],target,atol=1e-7)
        self.assertLessEqual(np.linalg.norm(best[3][1]),.001)

if __name__=='__main__':unittest.main()
