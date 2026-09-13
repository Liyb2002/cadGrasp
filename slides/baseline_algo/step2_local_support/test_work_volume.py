"""Analytic counterexamples for continuous work-volume clearance."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step2_local_support import work_volume as W
from step5_connect_support import layout as L
from step5_connect_support.test_connection import contact
from step2_local_support import insertion as D


def box(center,extent=.02):
    mesh=trimesh.creation.box(np.broadcast_to(extent,(3,)))
    mesh.apply_translation(center)
    return mesh


def volume():
    return W.WorkVolume([[[0,0,0],[1,0,0],[0,1,0]]],[[0,0,1]],[17],30.,1.)


class WorkVolumeTests(unittest.TestCase):
    def test_sign_is_outward_and_rays_are_unbounded(self):
        work=volume()
        self.assertTrue(work.check_parts([box([.2,.2,-1.])])['passed'])
        self.assertFalse(work.check_parts([box([.2,.2,1.])])['passed'])
        self.assertFalse(work.check_parts([box([.2,.2,100.])])['passed'])

    def test_triangle_interior_is_reserved_not_only_its_vertices(self):
        check=volume().check_parts([box([.3,.3,.01],.001)])
        self.assertFalse(check['passed'])
        self.assertEqual(check['witness']['classification'],'circular_cone_intersection')

    def test_all_solid_vertices_can_be_clear_while_its_interior_is_blocked(self):
        work=volume();beam=box([0.,.3,.2],[4.,.02,.02])
        for point in beam.vertices:
            self.assertTrue(work.check_parts([box(point,.0001)])['passed'])
        self.assertFalse(work.check_parts([beam])['passed'])

    def test_between_direction_samples_and_tangent_contact_are_rejected(self):
        work=volume();angle=np.pi+np.pi/W.SIDES
        point=np.r_[.999*np.tan(np.pi/6)*np.array([np.cos(angle),np.sin(angle)]),1.]
        self.assertFalse(work.check_parts([box(point,.0001)])['passed'])
        self.assertFalse(work.check_parts([box([.3,.3,-.01],.02)])['passed'])

    def test_circular_directions_are_contained_in_the_outer_polyhedral_cone(self):
        work=volume();angles=np.linspace(0,2*np.pi,10003)
        rays=np.c_[np.tan(np.pi/6)*np.cos(angles),np.tan(np.pi/6)*np.sin(angles),np.ones(len(angles))]
        self.assertLessEqual(np.max(rays@work.outer[0].T),1e-12)

    def test_union_does_not_fill_the_gap_between_disjoint_patches(self):
        triangles=np.array([[[0,0,0],[.1,0,0],[0,.1,0]]])+np.array([[[-3,0,0]],[[3,0,0]]])
        work=W.WorkVolume(triangles,[[0,0,1],[0,0,1]],[1,2],30.,1.)
        self.assertTrue(work.check_parts([box([0.,0.,.1])])['passed'])

    def test_rotation_translation_and_metric_scale_do_not_change_the_decision(self):
        source=volume();solid=box([.2,.2,.3]);rotation=trimesh.transformations.rotation_matrix(.8,[1,2,3])[:3,:3]
        for scale in [.01,1.,100.]:
            shift=np.array([14.,-3.,8.])
            tri=source.triangles@rotation.T*scale+shift
            work=W.WorkVolume(tri,source.normals@rotation.T,[17],30.,scale)
            moved=trimesh.Trimesh(solid.vertices@rotation.T*scale+shift,solid.faces,process=False)
            self.assertFalse(work.check_parts([moved])['passed'])

    def test_solver_error_cannot_be_reported_as_clear(self):
        error=SimpleNamespace(status=4,success=False,message='test numerical failure')
        with patch.object(W,'linprog',return_value=error):
            check=volume().check_parts([box([.2,.2,.5])])
        self.assertFalse(check['passed'])
        self.assertEqual(check['classification'],'solver_unresolved')

    def test_actual_contact_in_work_surface_is_rejected_before_ground_search(self):
        mesh=box([0,0,1.],[1,1,1]);head=contact(mesh,[0,0,1],0)
        face=head['center_face']
        work=W.WorkVolume(mesh.triangles[[face]],mesh.face_normals[[face]],[face],30.,1.)
        directions=[dict(certified_directions=D.normalize(isolated=[0.]))]
        points=np.array([[-.6,-.6],[.6,-.6],[.6,.6],[-.6,.6]])
        result,_=L.search(mesh,[head],directions,points,np.zeros(3),.01,work_volume=work)
        self.assertEqual(result['status'],'contact_surfaces_block_work_volume')
        self.assertEqual(result['attempts'],[])
        self.assertFalse(result['work_volume_head_precheck']['passed'])


if __name__=='__main__':unittest.main()
