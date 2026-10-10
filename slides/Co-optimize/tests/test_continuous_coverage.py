"""Continuous magnitude intervals, moving boundaries and physical invariants."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
import trimesh
from types import SimpleNamespace
from continuous_support.coverage import magnitude_intervals
from continuous_support.contacts import clip_field,exit_planes,union_field
from whole_search.model import Layout
from continuous_support.objective import CoverageObjective,Evaluation
from translation import translation


class ContinuousCoverageTests(unittest.TestCase):
    def test_full_partial_and_empty_magnitude_intervals(self):
        rays=np.eye(7);base=np.r_[1.,np.zeros(6)]
        slopes=np.array([[-2.,1,0,0,0,0,0],[-.5,1,0,0,0,0,0],[0,-1,0,0,0,0,0]])
        result=magnitude_intervals(rays,base,slopes,1.)
        np.testing.assert_allclose(result['fraction'],[.5,1.,0.],atol=1e-7)
        self.assertTrue(result['gravity_supported'])

    def test_nonzero_lower_endpoint_and_no_feasible_interval(self):
        base=np.r_[-.2,1.,np.zeros(5)];rays=np.eye(7)
        slopes=np.array([[1.,-1.,0,0,0,0,0],[-1.,0,0,0,0,0,0]])
        result=magnitude_intervals(rays,base,slopes,1.)
        np.testing.assert_allclose(result['fraction'],[.8,0.],atol=1e-7)
        self.assertFalse(result['gravity_supported'])

    def test_positive_contact_scaling_does_not_invent_pressure_capacity(self):
        rays=np.eye(7);base=np.r_[1.,np.zeros(6)];slopes=np.array([[-2.,1,0,0,0,0,0]])
        original=magnitude_intervals(rays,base,slopes,1.)
        scaled=magnitude_intervals(rays*np.array([1e-3,20,1,2,3,4,5])[:,None],base,slopes,1.)
        np.testing.assert_allclose(original['fraction'],scaled['fraction'],atol=1e-7)

    def test_continuous_cone_change_changes_coverage_without_pass_count_change(self):
        base=np.r_[1.,np.zeros(6)];slope=np.array([[-2.,1,0,0,0,0,0]])
        coverage=[]
        for parameter in [.2,.21]:
            rays=np.eye(7);rays[1,0]=-parameter
            coverage.append(magnitude_intervals(rays,base,slope,1.)['fraction'][0])
        # Both endpoint demands fail, yet their supported magnitude fraction
        # increases continuously: m <= 1/(2-parameter).
        np.testing.assert_allclose(coverage,[1/1.8,1/1.79],atol=1e-7)

    def test_contact_contour_moves_and_has_correct_area_derivative(self):
        tri=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        areas=[];extremes=[]
        for cut in [.2,.201]:
            polygon=clip_field(tri,tri[:,0]-cut)
            areas.append(sum(np.linalg.norm(np.cross(polygon[i]-polygon[0],polygon[i+1]-polygon[0]))/2
                             for i in range(1,len(polygon)-1)))
            extremes.append(polygon[:,0].min())
        np.testing.assert_allclose(extremes,[.2,.201],atol=1e-12)
        np.testing.assert_allclose(areas,[.8**2/2,.799**2/2],atol=1e-12)
        self.assertAlmostEqual((areas[1]-areas[0])/.001,-.7995,places=9)

    def test_exit_field_uses_full_length_and_union_of_triangle_prisms(self):
        mesh=trimesh.creation.box();planes,counts=exit_planes(mesh,np.array([0.,0,1.]),10.)
        samples=np.array([[.2,.1,5.],[.8,.1,5.],[.2,.1,11.]])
        values=union_field(samples,planes,counts)
        self.assertLess(values[0],0);self.assertGreater(values[1],0);self.assertGreater(values[2],0)

    def test_translation_counts_gains_and_losses_and_preserves_world_height(self):
        native=np.array([trimesh.transformations.rotation_matrix(.4,[1,0,0]),np.eye(4)])
        native[1,2,3]=.3
        placement=np.linalg.inv(native[0])@native[1]
        layout=Layout(np.array([np.eye(4),placement]),np.array([[0,0,1.],[0,0,1.]]),np.array([0,0]),(0,1))
        model=SimpleNamespace(poses=['pose_a','pose_b'],native=native,extent=2.,
            floor_normal=lambda q,k:native[q.hosts[k],:3,:3].T@np.array([0,0,1.]))
        class Objective:
            covered=staticmethod(CoverageObjective.covered)
            def __init__(self):self.model=model
            def evaluate(self,q):
                placed=native[q.hosts[1]]@q.placements[1]
                np.testing.assert_allclose(placed[:3,:3],native[1,:3,:3],atol=1e-12)
                if abs(placed[2,3]-.3)>1e-12:raise AssertionError('Height changed')
                x=placed[0,3]
                return Evaluation(q,np.array([.8+.02*x,.9-.01*x]),[],10.,0)
            def value(self,r,phase):return float(np.mean(1-r.coverage))
            def acceptable(self,a,b,phase):return self.value(b,phase)<self.value(a,phase)
        objective=Objective();before=objective.evaluate(layout);after,record=translation(objective,before)
        self.assertTrue(record['accepted'])
        self.assertGreater(after.coverage[0],before.coverage[0])
        self.assertLess(after.coverage[1],before.coverage[1])
        self.assertLess(objective.value(after,'coverage'),objective.value(before,'coverage'))


if __name__=='__main__':unittest.main()
