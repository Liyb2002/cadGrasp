from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys,unittest
import numpy as np
import trimesh

CO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(CO/'helper_func'),str(CO/'step4.2')]
import _bootstrap
from material_force import certify_material_contacts
from whole_search.model import Layout


class MaterialForceTests(unittest.TestCase):
    def test_actual_contact_wrenches_use_original_com_and_cancel_seat_translation(self):
        triangle=np.array([[0.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        mesh=trimesh.Trimesh(triangle,[[0,1,2]],process=False)
        native=np.repeat(np.eye(4)[None],2,axis=0)
        native[1,:3,:3]=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]];native[1,:3,3]=[.2,.3,.4]
        task=SimpleNamespace(domain=SimpleNamespace(com=native[1,:3,3],
            mesh=SimpleNamespace(face_normals=np.array([[0.,1.,0.]]))),scale=np.ones(6),
            targets=np.zeros((32768,6)))
        m=SimpleNamespace(mesh=mesh,native=native,allowed=[np.array([0])]*2,tasks=[task,task],
            floor_columns=lambda q,k:np.array([[0.,0.,0.,0.,0.,0.,-1.]]))
        q=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.tile([0.,0.,1.],(2,1)),np.array([0,0]),(1,))
        q.placements[1,:3,:3]=native[1,:3,:3];cones=[]
        for translation in [[1.,2.,3.],[-2.,4.,1.]]:
            q.placements[1,:3,3]=translation
            actual=triangle@q.placements[1,:3,:3].T+q.placements[1,:3,3]
            with patch('material_force.C.contact_boundary',return_value=(actual[None],np.array([0]))),\
                 patch('material_force.classify',return_value=(np.ones(32768,bool),{})):
                cones.append(certify_material_contacts(m,q,mesh)['supplies'][1])
        np.testing.assert_allclose(cones[0],cones[1],atol=1e-12)
        self.assertEqual(cones[0].shape[1],7)


if __name__=='__main__':unittest.main()
