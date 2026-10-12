"""Outward-only exact normal prisms, independent of force cone settings."""
from pathlib import Path
from types import SimpleNamespace
import sys, unittest
import numpy as np
import trimesh

COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/'helper_func'),str(COOPT/'step4.2')]
import _bootstrap
from normal_work_access import NormalWorkAccess, normal_work_access
from work_access import WorkAccess, unpack_solid
from base_oracle import BaseOracle
from pose_oracle import inside_union, transform_points
from whole_search.model import Layout
from trimesh.ray.ray_pyembree import RayMeshIntersector


class NormalAccessTests(unittest.TestCase):
    def source(self):
        return trimesh.Trimesh([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],[[0,1,2]],process=False)

    def test_far_normal_rays_and_whole_area_without_inward_or_angular_rays(self):
        mesh=self.source();access=NormalWorkAccess(mesh,[0])
        points=np.array([[.01,.98,10000.],[.2,.2,-.01],[1.2,.2,10.],[.2,.2,.1]])
        np.testing.assert_array_equal(access.contains_points(points),[True,False,False,True])
        self.assertTrue(WorkAccess(mesh,[0],30).contains_points(points[[2]])[0])
        self.assertEqual(access.clearance,0.)

    def test_prism_halfspaces_agree_with_actual_geometry_and_have_exact_volume(self):
        access=NormalWorkAccess(self.source(),[0]);mesh=unpack_solid(access.solid(.7))
        self.assertAlmostEqual(mesh.volume,.35,places=12)
        points=np.array([[.1,.1,.6],[.8,.8,.4],[-.02,.1,.1],[.1,.1,-.01]])
        np.testing.assert_array_equal(RayMeshIntersector(mesh).contains_points(points),access.contains_points(points))
        cap=access.required_length([[.2,.2,4.]],minimum=.001)
        self.assertGreater(cap,4.)

    def test_union_of_different_normals_does_not_fill_between_faces(self):
        mesh=trimesh.Trimesh([[-2.,0.,0.],[-1.,0.,0.],[-2.,1.,0.],
                             [2.,0.,0.],[2.,1.,0.],[2.,0.,1.]],[[0,1,2],[3,4,5]],process=False)
        access=NormalWorkAccess(mesh,[0,1])
        points=np.array([[-1.8,.1,.5],[2.5,.1,.1],[0.,.1,.1]])
        np.testing.assert_array_equal(access.contains_points(points),[True,True,False])
        np.testing.assert_array_equal(RayMeshIntersector(access.mesh(1.)).contains_points(points),[True,True,False])

    def test_subset_of_original_angular_model_and_no_force_angle_mutation(self):
        mesh=self.source();original=WorkAccess(mesh,[0],30.)
        access=normal_work_access(mesh,original)
        rng=np.random.default_rng(7);uv=rng.dirichlet([1,1,1],200)
        points=uv@mesh.triangles[0]+rng.uniform(0.,100.,(200,1))*mesh.face_normals[0]
        self.assertTrue(access.contains_points(points).all())
        self.assertTrue(original.contains_points(points).all())
        self.assertEqual(original.half_angle_deg,30.)
        self.assertIs(access,normal_work_access(mesh,original))
        self.assertEqual(access.definition()['mode'],'normal_ray_volume')

    def test_both_oracle_screen_and_connection_input_use_normal_geometry(self):
        mesh=self.source();original=WorkAccess(mesh,[0],30.)
        task=SimpleNamespace(domain=SimpleNamespace(data={'load':{'cone_half_deg':30.}}))
        # Supply no tasks while constructing: no objects arrays are needed.
        model=SimpleNamespace(extent=1.,tasks=[],poses=[],inputs=[],work_rays=[],mesh=mesh,
                              native=np.eye(4)[None],length=.5)
        oracle=BaseOracle(model)
        model.tasks=[task];model.poses=['A'];model.work_rays=[original]
        oracle.work_access=[normal_work_access(mesh,original)]
        layout=Layout(np.eye(4)[None],np.array([[0.,0.,1.]]),np.array([0]),(0,))
        layout.placements[0,:3,3]=[.4,.5,.6]
        constraints=oracle.pose_constraints(layout)
        self.assertIs(constraints.poses[0].work_access,oracle.work_access[0])
        families=oracle.families_by_pose(layout)
        points=np.array([[.2,.2,3.],[1.2,.2,3.],[.2,.2,-.1]])
        common=transform_points(points,layout.placements[0])
        np.testing.assert_array_equal(inside_union(common,families[0]['work']),[True,False,False])
        np.testing.assert_array_equal(model.work_rays[0].contains_points(points),[True,True,False])


if __name__=='__main__':unittest.main()
