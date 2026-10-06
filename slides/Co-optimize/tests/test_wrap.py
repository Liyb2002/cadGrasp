"""Regress real contact extraction, registration, moment generators and fail evidence."""

import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
from fractions import Fraction
from co_common import *

class SurfaceWrapTests(unittest.TestCase):
    def test_registered_world_round_trip(self):
        task,T,mesh=state('B','pose_3')
        np.testing.assert_allclose(transform_points(mesh.vertices,T),task.domain.mesh.vertices,atol=1e-12,rtol=0)

    def test_full_shell_preserves_real_tetrahedron_contact_area(self):
        mesh=G.hull_mesh(np.array([[0.,0.,0.],[.03,0.,0.],[0.,.04,0.],[0.,0.,.05]]))
        offsets,valid=G.vertex_offsets(mesh,.002)
        self.assertTrue(valid.all())
        shell=union([S.solid(G.hull_mesh(G.head_cell(mesh,t,i,offsets))) for i,t in enumerate(mesh.triangles)])-S.solid(mesh)
        patch,source=contact_boundary(mesh,S.unpack(shell),np.arange(len(mesh.faces)))
        area=np.linalg.norm(np.cross(patch[:,1]-patch[:,0],patch[:,2]-patch[:,0]),axis=1).sum()/2
        self.assertAlmostEqual(area,mesh.area,places=11)
        for triangle,face in zip(patch,source):
            self.assertLess(np.max(np.abs((triangle-mesh.triangles[face,0])@mesh.face_normals[face])),1e-10)

    def test_wrap_has_bounded_outward_vertex_displacements(self):
        task,T,mesh=state('B','pose_3')
        offsets=wrap_offsets(mesh,.005)
        self.assertLessEqual(np.linalg.norm(offsets,axis=1).max(),.005+1e-14)
        self.assertTrue(np.all(np.einsum('fvc,fc->fv',offsets[mesh.faces],mesh.face_normals)>0))

    def test_triangle_vertices_span_all_normal_pressure_locations(self):
        tri=np.array([[0.,0.,0.],[.03,0.,0.],[0.,.04,0.]])
        normal=np.array([0.,0.,-1.]);com=np.array([.01,.02,.03]);weights=np.array([.2,.3,.5])
        vertices=np.c_[np.tile(normal,(3,1)),np.cross(tri-com,normal)]
        p=weights@tri
        np.testing.assert_allclose(weights@vertices,np.r_[normal,np.cross(p-com,normal)],atol=1e-15)

    def test_failure_separator_covers_entire_contact_cone(self):
        full=np.eye(7);target=np.array([-1.,0.,0.,0.,0.,0.,0.])
        proof=C.W.exact_separator(full,target)
        self.assertIsNotNone(proof)
        h=[Fraction(v) for v in proof['normal_exact']]
        self.assertGreater(sum(Fraction(float(t))*v for t,v in zip(target,h)),0)
        self.assertTrue(all(sum(Fraction(float(x))*v for x,v in zip(row,h))<=0 for row in full))

if __name__=='__main__':unittest.main()
