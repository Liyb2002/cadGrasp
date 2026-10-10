"""Independent projections, dead demands, coupled gains/losses and boundaries."""
import sys
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from scipy.optimize import nnls
from continuous_support.projection import project_demands, DemandDistance
from continuous_support.coverage import DemandGrid, magnitude_intervals
from continuous_support.objective import Evaluation, CoverageObjective
from whole_search.model import Layout
from direction_choice import direction_choice
from translation import translation
from continuous_support.surface import prism_slices


class DistanceGradientTests(unittest.TestCase):
    def test_batched_projection_matches_independent_nnls(self):
        rng = np.random.default_rng(37)
        rays = rng.normal(size=(80, 7));rays[:, 0]=abs(rays[:, 0])
        rays = np.vstack([rays, rays[:5], np.zeros((1, 7))])
        targets = rng.normal(size=(40, 7))
        result = project_demands(rays, targets)
        for index, target in enumerate(targets):
            coefficients, _ = nnls(rays.T, target, maxiter=3000)
            np.testing.assert_allclose(result['residuals'][index], coefficients@rays-target, atol=2e-8)
        self.assertLess(result['maximum_kkt_violation'], 2e-8)
        scaled = project_demands(rays*np.exp(rng.uniform(-8, 8, len(rays)))[:, None], targets)
        np.testing.assert_allclose(result['losses'], scaled['losses'], atol=2e-8)

    def test_empty_cone_and_degenerate_generators(self):
        target = np.array([[1., -2., 0, 0, 0, 0, 0]])
        np.testing.assert_allclose(project_demands(np.empty((0, 7)), target)['losses'], [2.5])
        rays = np.array([[1., 0, 0, 0, 0, 0, 0], [1., 0, 0, 0, 0, 0, 0]])
        np.testing.assert_allclose(project_demands(rays, target)['residuals'], [[0, 2., 0, 0, 0, 0, 0]])

    def test_zero_coverage_still_has_correct_distance_derivative(self):
        base=np.r_[-1., 1., np.zeros(5)]
        slopes=np.array([[0, .5, 0, 0, 0, 0, 0]])
        grid=DemandGrid(base,slopes,np.ones(1),1.,{})
        distance=DemandDistance(grid)
        angle=1.;h=1e-5
        def loss(theta):
            rays=np.array([np.r_[np.cos(theta),np.sin(theta),np.zeros(5)]])
            self.assertEqual(magnitude_intervals(rays,base,slopes,1.)['fraction'][0],0.)
            return distance.evaluate(rays)['residual_loss']
        derivative=(loss(angle+h)-loss(angle-h))/(2*h)
        direction=np.r_[np.cos(angle),np.sin(angle),np.zeros(5)]
        tangent=np.r_[-np.sin(angle),np.cos(angle),np.zeros(5)]
        normalization=max(1.,np.linalg.norm(distance.targets,axis=1).max())
        analytic=-distance.weights@((distance.targets@direction)*(distance.targets@tangent))/normalization**2
        self.assertAlmostEqual(derivative,analytic,places=7)
        self.assertLess(derivative,0.)

    def test_polygon_slice_detects_shadow_between_free_corners(self):
        triangle=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        # A small interior prism: all three original corners are outside.
        planes=np.array([[[1.,0,0,-.4],[-1.,0,0,.2],[0,1.,0,-.4],[0,-1.,0,.2],[0,0,1.,-.1],[0,0,-1.,-.1]]])
        polygons,sizes=prism_slices(triangle,np.array([0.,0,1.]),planes,np.array([6]),1e-8)
        polygon=polygons[0,:sizes[0]]
        area=sum(np.linalg.norm(np.cross(polygon[j]-polygon[0],polygon[j+1]-polygon[0]))/2 for j in range(1,len(polygon)-1))
        self.assertAlmostEqual(area,.04,places=10)

    def test_translation_gradient_accounts_for_other_pose_loss(self):
        native=np.repeat(np.eye(4)[None],2,axis=0)
        layout=Layout(native.copy(),np.array([[0.,0,1.],[0,0,1.]]),np.array([0,0]),(0,1))
        model=SimpleNamespace(native=native,poses=['a','b'],extent=2.,floor_normal=lambda q,k:np.array([0.,0,1.]))
        class Objective:
            covered=staticmethod(CoverageObjective.covered)
            def __init__(self):self.model=model
            def evaluate(self,q):
                x=q.placements[1,0,3]
                return Evaluation(q,np.zeros(2),[],1.,0,np.array([(x-1)**2,(x+.8)**2]),np.ones(2))
            def value(self,r,phase):return float(np.mean(r.residual_loss))
            def acceptable(self,a,b,phase):return self.value(b,phase)<self.value(a,phase)
        objective=Objective()
        before=objective.evaluate(layout)
        after,record=translation(objective,before,difference_fraction=1e-4)
        self.assertTrue(record['accepted'])
        self.assertLess(objective.value(after,'coverage'),objective.value(before,'coverage'))
        self.assertLess(after.residual_loss[0],before.residual_loss[0])
        self.assertGreater(after.residual_loss[1],before.residual_loss[1])
        # World derivative is -0.2; extent-normalized derivative is -0.4.
        self.assertAlmostEqual(np.linalg.norm(record['gradient']),.4,places=7)
        json.dumps(record)

    def test_coplanar_repeated_planes_do_not_duplicate_vertices(self):
        triangle=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        planes=np.array([[[-1.,0,0,0],[0,-1.,0,0],[1.,1.,0,-1.],[0,0,-1.,0]]*8])
        polygons,sizes=prism_slices(triangle,np.array([0.,0,1.]),planes,np.array([32]),1e-8)
        self.assertEqual(sizes[0],3)
        np.testing.assert_allclose(polygons[0,:3],triangle)

    def test_direction_boundary_uses_feasible_derivative(self):
        layout=Layout(np.eye(4)[None],np.array([[1.,0,0]]),np.array([0]),(0,))
        model=SimpleNamespace(native=np.eye(4)[None],poses=['a'],extent=1.,floor_normal=lambda q,k:np.array([0.,0,1.]))
        class Objective:
            covered=staticmethod(CoverageObjective.covered)
            def __init__(self):self.model=model
            def evaluate(self,q):
                assert q.directions[0,2]>=0
                z=q.directions[0,2]
                return Evaluation(q,np.zeros(1),[],1.,0,np.array([(z-.2)**2]),np.ones(1))
            def value(self,r,phase):return float(r.residual_loss[0])
            def acceptable(self,a,b,phase):return self.value(b,phase)<self.value(a,phase)
        objective=Objective();before=objective.evaluate(layout)
        after,record=direction_choice(objective,before,difference_degrees=.0001)
        self.assertTrue(record['probes'][0]['one_sided'])
        self.assertAlmostEqual(record['gradient'][0],-.4,places=5)
        self.assertGreater(after.layout.directions[0,2],0.)
        np.testing.assert_allclose(np.linalg.norm(after.layout.directions[0]),1.,atol=1e-12)
        json.dumps(record)

    def test_identical_exit_shadows_use_joint_direction_derivative(self):
        angle=.3
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),
                      np.repeat(np.array([[np.cos(angle),0.,np.sin(angle)]]),2,axis=0),np.arange(2),(0,1))
        model=SimpleNamespace(native=np.repeat(np.eye(4)[None],2,axis=0),poses=['a','b'],extent=1.,
                              floor_normal=lambda q,k:np.array([0.,0,1.]))
        class Objective:
            covered=staticmethod(CoverageObjective.covered)
            def __init__(self):self.model=model
            def evaluate(self,q):
                released=min(q.directions[:,2])
                return Evaluation(q,np.zeros(2),[],1.,0,np.repeat((1-released)**2,2),np.ones(2))
            def value(self,r,phase):return float(np.mean(r.residual_loss))
            def acceptable(self,a,b,phase):return self.value(b,phase)<self.value(a,phase)
        objective=Objective();before=objective.evaluate(layout)
        after,record=direction_choice(objective,before,difference_degrees=.0001)
        self.assertEqual(record['coordinate_groups'],[['a','b']])
        self.assertAlmostEqual(record['gradient'][0],-2*(1-np.sin(angle))*np.cos(angle),places=7)
        self.assertTrue(record['accepted'])
        np.testing.assert_allclose(after.layout.directions[0],after.layout.directions[1],atol=1e-12)
        self.assertGreater(min(after.layout.directions[:,2]),min(layout.directions[:,2]))

    def test_contact_birth_is_not_claimed_as_a_local_gradient(self):
        layout=Layout(np.eye(4)[None],np.array([[1.,0,0]]),np.array([0]),(0,))
        model=SimpleNamespace(native=np.eye(4)[None],poses=['a'],extent=1.,floor_normal=lambda q,k:np.array([0.,0,1.]))
        class Objective:
            covered=staticmethod(CoverageObjective.covered)
            def __init__(self):self.model=model
            def evaluate(self,q):
                free=q.directions[0,2]>np.sin(.012)
                return Evaluation(q,np.array([float(free)]),[dict(pose='a',contact_normal_signature='new' if free else 'old')],
                                  1.,0,np.array([0. if free else .5]),np.array([0. if free else 1.]))
            def value(self,r,phase):return float(r.residual_loss[0])
            def acceptable(self,a,b,phase):return self.value(b,phase)<self.value(a,phase)
        objective=Objective();before=objective.evaluate(layout)
        after,record=direction_choice(objective,before)
        self.assertTrue(record['accepted'])
        self.assertEqual(record['update_kind'],'contact_event_continuation')
        self.assertFalse(record['derivative_check']['local_derivative_consistent'])
        self.assertTrue(record['derivative_check']['contact_normal_event'])
        self.assertLess(np.arccos(after.layout.directions[0,0]),np.radians(2.))
        self.assertTrue(any(r.get('event_step_refinement') for r in record['line_search']))


if __name__=='__main__':unittest.main()
