"""Final numerical samples may only move explicit seats horizontally."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.model import Layout
from continuous_support.joint_conditioning import joint_perturbation


class JointConditioningTests(unittest.TestCase):
    def test_registered_pose_stays_and_other_hosts_receive_world_horizontal_shifts(self):
        native=np.repeat(np.eye(4)[None],3,axis=0)
        native[1,:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
        layout=Layout(np.repeat(np.eye(4)[None],3,axis=0),np.array([[0.,0.,1.]]*3),np.array([0,0,1]),(0,1,2))
        model=SimpleNamespace(native=native,extent=.1,poses=['a','b','c'],
                              floor_normal=lambda q,k:native[q.hosts[k],:3,:3].T@np.array([0.,0.,1.]))
        trial,detail=joint_perturbation(model,layout,.125,1/512)
        np.testing.assert_array_equal(trial.placements[0],layout.placements[0])
        for k in [1,2]:
            world=native[trial.hosts[k],:3,:3]@(trial.placements[k,:3,3]-layout.placements[k,:3,3])
            self.assertAlmostEqual(world[2],0.,places=14)
            self.assertAlmostEqual(np.linalg.norm(world),.1/512,places=14)
        self.assertFalse(detail['gradient_step'])


if __name__=='__main__':unittest.main()
