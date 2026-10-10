"""A checked but unpenalized magnitude endpoint can trap the old objective."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import unittest
import numpy as np
from types import SimpleNamespace
from continuous_support.coverage import DemandGrid
from continuous_support.projection import project_demands
from continuous_support.projection import DemandDistance
from continuous_support.boundary_projection import BoundaryDistance
from continuous_support.boundary_grid import BoundaryGrid


class BoundaryTests(unittest.TestCase):
    def test_maximum_force_failure_has_nonzero_loss(self):
        grid=DemandGrid(np.r_[.95,1.,np.zeros(5)],np.array([np.r_[-1.,np.zeros(6)]]),np.ones(1),1.,{})
        old=DemandDistance(grid).evaluate(np.eye(7));new=BoundaryDistance(grid).evaluate(np.eye(7))
        self.assertLess(old['residual_loss'],1e-25);self.assertGreater(old['maximum_demand_residual'],.049)
        self.assertGreater(new['residual_loss'],1e-5)

    def test_zero_process_force_failure_has_nonzero_loss(self):
        grid=DemandGrid(np.r_[-.05,1.,np.zeros(5)],np.array([np.r_[1.,np.zeros(6)]]),np.ones(1),1.,{})
        old=DemandDistance(grid).evaluate(np.eye(7));new=BoundaryDistance(grid).evaluate(np.eye(7))
        self.assertLess(old['residual_loss'],1e-25);self.assertFalse(old['gravity_supported'])
        self.assertGreater(new['residual_loss'],1e-5)

    def test_original_uniform_magnitude_measure_degree_seven(self):
        grid=DemandGrid(np.zeros(7),np.ones((1,7)),np.ones(1),1.,{})
        distance=BoundaryDistance(grid)
        self.assertTrue((distance.weights>0).all())
        for degree in range(8):
            self.assertAlmostEqual(float(distance.weights@(distance.magnitudes**degree)),1/(degree+1),places=13)

    def test_angular_boundary_retains_original_uniform_cosine_measure(self):
        def evaluate(faces,u,v,theta,phi,**kwargs):
            theta=np.asarray(theta);phi=np.asarray(phi)
            return dict(tool_reachable=np.ones(len(theta),bool),q_m=np.column_stack([u,v,np.zeros(len(theta))]),
                        d=np.column_stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),-np.cos(theta)]))
        domain=SimpleNamespace(half_angle=np.pi/6,com=np.zeros(3),gravity=np.array([0.,0.,-1.]),k=1.,
            data={'geometry':{'work_face_areas_m2':[1.]}},evaluate=evaluate)
        task=SimpleNamespace(domain=domain,scale=np.ones(6))
        for level in [1,2,3]:
            grid=BoundaryGrid.from_task(task,level)
            self.assertTrue((grid.weights>0).all())
            self.assertAlmostEqual(float(grid.weights@grid.slopes[:,2]),(1+np.cos(np.pi/6))/2,places=12)
            self.assertAlmostEqual(float(grid.slopes[:,2].min()),np.cos(np.pi/6),places=12)

    def test_deduplication_preserves_all_demand_weights_and_loss(self):
        grid=DemandGrid(np.r_[.95,1.,np.zeros(5)],np.array([np.r_[-1.,np.zeros(6)]]*2),np.array([.2,.8]),1.,{})
        distance=BoundaryDistance(grid);direct=project_demands(np.eye(7),distance.targets)
        self.assertAlmostEqual(distance.evaluate(np.eye(7))['residual_loss'],float(distance.weights@direct['normalized_losses']),places=15)
        self.assertEqual(len(distance.unique_targets),5)


if __name__=='__main__':unittest.main()
