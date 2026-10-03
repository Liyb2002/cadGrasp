"""Regression: absolute support frame, force sign and head-count independence."""
import unittest
from types import SimpleNamespace
import numpy as np
from step3_scheculer.absolute_dsl import directions,diagnostic
from step3_scheculer.operation_dsl import State,ray

class DirectionTests(unittest.TestCase):
    def test_standing_and_lying_saved_objects_use_fixture_frame(self):
        arbitrary_object_rotation=np.array([[0,0,1],[0,1,0],[-1,0,0.]])
        task=SimpleNamespace(domain=SimpleNamespace(mesh=SimpleNamespace(face_normals=np.array([[0,0,-1.]])),data={'frame':{'T_world_mesh':np.eye(4)}}))
        other=SimpleNamespace(domain=SimpleNamespace(mesh=task.domain.mesh,data={'frame':{'T_world_mesh':np.block([[arbitrary_object_rotation,np.zeros((3,1))],[np.zeros((1,3)),np.ones((1,1))]])}}))
        contact={'source_faces':np.array([0]),'triangle_areas_m2':np.array([1.]),'candidate_id':'a'}
        state=State(((contact,),(contact,)),np.array([np.eye(3)]*2),np.zeros((2,3)),(ray([0,0,-1]),)*2)
        force,exit=directions([task,other],state)
        np.testing.assert_allclose(force,[[0,0,1]]*2)
        np.testing.assert_allclose(exit,[[0,0,1]]*2)
        self.assertEqual(diagnostic([task,other],state)['exit_mean_pair_angle_deg'],0.)
    def test_rotated_fixture_changes_absolute_direction(self):
        task=SimpleNamespace(domain=SimpleNamespace(mesh=SimpleNamespace(face_normals=np.array([[0,0,-1.]]))))
        c={'source_faces':np.array([0]),'triangle_areas_m2':np.array([1.])}
        b=np.array([[0,0,-1],[0,1,0],[1,0,0.]])
        state=State(((c,),),b[None],np.zeros((1,3)),(ray([0,0,-1]),))
        force,exit=directions([task],state)
        np.testing.assert_allclose(force,[[1,0,0]])
        np.testing.assert_allclose(exit,force)
if __name__=='__main__':unittest.main()
