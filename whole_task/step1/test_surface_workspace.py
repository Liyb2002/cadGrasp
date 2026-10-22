"""The inward work box constrains surface points; the head may remain outside."""
import unittest
from types import SimpleNamespace

import numpy as np
import trimesh

from .workspace import surface_workspace, points_inside_box, box_edges, tabletop_placements
from .areas import VIEWS, areas
from .arm import Check, POINT_OUTSIDE_BOX


class SurfaceWorkspaceTests(unittest.TestCase):
    def reference(self):
        mesh=trimesh.creation.box(extents=[.12,.18,.08])
        mesh.apply_translation([.04,.09,-.02])
        setup=dict(work_point_m=[0.,-.71,.3],table_height_m=.21,
                   working_direction_world=[0.,-2**-.5,-2**-.5])
        pose=tabletop_placements(mesh,VIEWS,setup)[0]
        vertices=mesh.vertices@np.asarray(pose['rotation_world_object']).T+pose['translation_world_m']
        return mesh,pose,np.array([vertices.min(axis=0),vertices.max(axis=0)])

    def test_cube_uses_half_longest_side_and_shares_near_face_center(self):
        mesh,pose,reference=self.reference();workspace=surface_workspace(mesh,pose)
        bounds=np.asarray(workspace['bounds_world_m'])
        edge=.5*np.max(reference[1]-reference[0])
        np.testing.assert_allclose(bounds[1]-bounds[0],np.full(3,edge))
        self.assertEqual(workspace['shape'],'cube');self.assertAlmostEqual(workspace['cube_edge_m'],edge)
        self.assertEqual(workspace['adjoining_axis'],'y');self.assertEqual(workspace['adjoining_sign'],1)
        shared=reference.mean(axis=0);shared[1]=reference[1,1]
        work_face=bounds.mean(axis=0);work_face[1]=bounds[1,1]
        np.testing.assert_allclose(work_face,shared,atol=1e-12)
        np.testing.assert_allclose(workspace['shared_face_center_world_m'],shared)
        self.assertAlmostEqual(bounds[0,1],shared[1]-edge)
        self.assertLess(bounds[0,1],reference[1,1])
        self.assertEqual(workspace['applies_to'],'surface_process_point')
        self.assertEqual(len(box_edges(bounds)),12)

    def test_all_six_robot_sides_extend_inward(self):
        mesh,pose,reference=self.reference()
        for axis in range(3):
            for sign in (-1,1):
                base=reference.mean(axis=0);base[axis]+=10*sign
                workspace=surface_workspace(mesh,pose,base_position_world=base)
                bounds=np.asarray(workspace['bounds_world_m'])
                self.assertEqual(workspace['adjoining_axis'],'xyz'[axis])
                self.assertEqual(workspace['adjoining_sign'],sign)
                edge=.5*np.max(reference[1]-reference[0])
                np.testing.assert_allclose(bounds[1]-bounds[0],np.full(3,edge))
                self.assertAlmostEqual(bounds[1 if sign>0 else 0,axis],reference[1 if sign>0 else 0,axis])
                self.assertAlmostEqual(bounds[0 if sign>0 else 1,axis],reference[1 if sign>0 else 0,axis]-sign*edge)

    def test_thin_object_does_not_shrink_or_clip_the_cube_short_axes(self):
        mesh=trimesh.creation.box(extents=[.02,.20,.04])
        pose=dict(rotation_world_object=np.eye(3).tolist(),translation_world_m=[0.,0.,0.])
        workspace=surface_workspace(mesh,pose,base_position_world=[0.,1.,0.])
        bounds=np.asarray(workspace['bounds_world_m'])
        np.testing.assert_allclose(bounds[1]-bounds[0],[.1,.1,.1])
        np.testing.assert_allclose(workspace['shared_face_center_world_m'],[0.,.1,0.])
        np.testing.assert_allclose(bounds[:,1],[0.,.1])
        self.assertLess(bounds[0,0],mesh.bounds[0,0]);self.assertGreater(bounds[1,2],mesh.bounds[1,2])

    def test_nonpositive_or_nonfinite_sizes_are_rejected(self):
        mesh,pose,_=self.reference()
        for fraction in (0.,-1.,np.inf,np.nan):
            with self.assertRaises(ValueError):surface_workspace(mesh,pose,fraction)

    def test_surface_point_inside_is_valid_even_when_the_head_root_is_outside(self):
        check=Check.__new__(Check);check.work_bounds=np.array([[-.2,-.2,-.2],[.2,.2,.2]])
        check.length=1.;check.seeds=[np.zeros(7)]
        check.arm=SimpleNamespace(solve=lambda *args:np.zeros(7),
            forward=lambda q:(np.zeros(3),np.array([0,0,-1]),None,None))
        check.hits=lambda *args:False
        result=check.solve(np.zeros(3),np.array([0,0,-1]),np.eye(4),None)
        np.testing.assert_array_equal(result,np.zeros(7))
        self.assertFalse(points_inside_box([0,0,1],check.work_bounds))

    def test_surface_point_outside_is_rejected_even_if_its_head_root_would_be_inside(self):
        check=Check.__new__(Check);check.work_bounds=np.array([[-.2,-.2,-.2],[.2,.2,.2]])
        check.length=.2
        tip=np.array([0,0,.3]);axis=np.array([0,0,1])
        self.assertTrue(points_inside_box(tip-check.length*axis,check.work_bounds))
        self.assertIsNone(check.solve(tip,axis,np.eye(4),None))

    def test_pair_and_replay_reject_a_surface_vertex_outside_before_ik(self):
        check=Check.__new__(Check)
        check.mesh=trimesh.Trimesh(vertices=[[0,0,.3],[.01,0,.3],[0,.01,.3]],faces=[[0,1,2]],process=False)
        check.work_bounds=np.array([[-.1,-.1,-.1],[.1,.1,.1]])
        check.length=.2
        code,_=check.pair(0,np.array([0,0,1]),np.eye(4))
        self.assertEqual(code,POINT_OUTSIDE_BOX)
        self.assertFalse(check.replay(0,np.array([0,0,1]),np.eye(4),np.zeros((4,7))))

    def test_actual_fk_process_point_must_also_stay_inside(self):
        check=Check.__new__(Check);check.work_bounds=np.array([[-.2,-.2,-.2],[.2,.2,.2]])
        check.length=.2;check.seeds=[np.zeros(7)]
        check.arm=SimpleNamespace(solve=lambda *args:np.zeros(7),
            forward=lambda q:(np.array([0,0,.3]),np.array([0,0,1]),None,None))
        check.hits=lambda *args:False
        self.assertIs(check.solve(np.zeros(3),np.array([0,0,1]),np.eye(4),None),False)


class StrictIntersectionTests(unittest.TestCase):
    def test_unreachable_enclosed_face_is_never_added_to_a_task(self):
        mesh=trimesh.creation.box().subdivide().subdivide().subdivide()
        tilt=np.where(mesh.face_normals@(-VIEWS).T<-.9,0.,np.inf)
        top=np.flatnonzero(mesh.face_normals[:,2]>.9)
        missing=top[np.argmin(np.linalg.norm(mesh.triangles_center[top]-[0,0,.5],axis=1))]
        tilt[missing]=np.inf
        label,summary=areas(mesh,np.arange(len(mesh.faces)),tilt,-VIEWS,25.,.1,strict_reach=True)
        self.assertEqual(label[missing],-1)
        self.assertEqual(summary['enclosed_face_ids'],[])
        assigned=np.flatnonzero(label>=0)
        self.assertTrue(np.isfinite(tilt[assigned,label[assigned]]).all())
        np.testing.assert_array_equal(label>=0,np.isfinite(tilt).any(axis=1))

    def test_small_but_reachable_islands_are_not_discarded(self):
        large=trimesh.creation.box();small=trimesh.creation.box(extents=[.01]*3)
        small.apply_translation([3,0,0]);mesh=trimesh.util.concatenate([large,small])
        label,summary=areas(mesh,np.arange(len(mesh.faces)),np.zeros((len(mesh.faces),1)),
                            np.array([[0,0,-1]]),25.,.1,strict_reach=True)
        self.assertTrue((label==0).all());self.assertEqual(summary['small_pieces_dropped'],0)


if __name__=='__main__':unittest.main()
