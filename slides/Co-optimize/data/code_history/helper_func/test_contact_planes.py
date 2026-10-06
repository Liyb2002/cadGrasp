"""Exact tangent proposals must preserve every requested contact and floor."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
import physics_guided
from physics_guided_contact_planes import contact_plane_directions


class ContactPlaneTests(unittest.TestCase):
    def check(self,normals,floors):
        proposals=list(contact_plane_directions(normals,floors))
        self.assertTrue(proposals)
        for directions in proposals:
            np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1.,atol=1e-12)
            np.testing.assert_allclose(np.asarray(normals)@directions.T,0.,atol=1e-12)
            self.assertGreaterEqual(np.min(np.sum(directions*floors,axis=1)),-1e-12)
        return proposals

    def test_opposite_contacts_require_exact_plane(self):
        self.check([[1,0,0],[-1,0,0]],np.array([[0,0,1],[0,0,-1],[1,0,0]],float))

    def test_two_normals_allow_opposite_exits_per_pose(self):
        proposals=self.check([[1,0,0],[0,1,0]],np.array([[0,0,1],[0,0,-1]],float))
        for d in proposals:np.testing.assert_allclose(d,[[0,0,1],[0,0,-1]],atol=1e-12)

    def test_floor_projection_stays_inside_contact_plane(self):
        floors=np.array([[1,1,1],[-1,-2,3],[0,-1,-2]],float)
        floors/=np.linalg.norm(floors,axis=1)[:,None]
        self.check([[2,-1,3]],floors)

    def test_three_independent_normals_have_no_nonzero_tangent(self):
        self.assertEqual(list(contact_plane_directions(np.eye(3),np.array([[0,0,1.]]))),[])

if __name__=='__main__':unittest.main()

class AllocatedContactTests(unittest.TestCase):
    def test_contact_recovery_uses_moment_and_excludes_floor_and_slack(self):
        from types import SimpleNamespace
        from physics_guided_contact_planes import allocated_contacts
        from co_common import U
        points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        normals=np.array([[0.,0.,-1.]])
        forces=np.tile(-normals[0],(3,1));raw=np.c_[forces,np.cross(points,forces)]
        heads=U.heads(raw,np.ones(6))
        supplies=np.vstack([heads,[0,0,1,0,0,0,0],[0,0,0,0,0,0,-1]])
        coefficients=np.array([0.,2.,0.,7.,2.])
        target=coefficients@supplies
        np.testing.assert_allclose(target,[0,0,9,0,-2,0,0])
        task=SimpleNamespace(domain=SimpleNamespace(mesh=SimpleNamespace(face_normals=normals),com=np.zeros(3)),scale=np.ones(6))
        search=SimpleNamespace(mesh=SimpleNamespace(face_normals=normals),states=[(task,np.eye(4))],
                               proxy_indices=[{0}],projection=lambda *args:dict(coefficients=coefficients))
        result=dict(triangles=points[None],sources=np.array([0]),supplies=[supplies])
        old,recovered,normal,value=allocated_contacts(search,result)
        np.testing.assert_allclose(recovered,[points[1]])
        np.testing.assert_allclose(normal,normals)
        np.testing.assert_allclose(value,[2.])
