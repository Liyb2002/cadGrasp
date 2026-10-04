"""Regressions for continuous world exits and geometry-dependent force moments."""
import unittest
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import trimesh
from step3_scheculer.co_descent_v39 import turn,tangent_basis,paths,contact_key,improves
from step3_scheculer.pair_scoring import TaskProblem

class CoDescentTests(unittest.TestCase):
    def test_oblique_updates_leave_axes_and_are_not_quantized(self):
        a=turn([0,0,1],np.deg2rad(3.713),.271828)
        b=turn([0,0,1],np.deg2rad(3.713),.271829)
        self.assertTrue(np.all(np.abs(a)>1e-3));self.assertGreater(np.linalg.norm(a-b),1e-9)
        self.assertAlmostEqual(np.linalg.norm(a),1.,places=14)
        self.assertAlmostEqual(np.arccos(a[2]),np.deg2rad(3.713),places=12)
    def test_zero_angle_keeps_direction_and_tangent_is_valid_at_poles(self):
        for d in ([0,0,1],[0,0,-1],[0,1,0],[1,2,3]):
            n=np.asarray(d,float);n/=np.linalg.norm(n);u,v=tangent_basis(n)
            np.testing.assert_allclose([u@n,v@n,u@v],[0,0,0],atol=1e-14)
            np.testing.assert_allclose(turn(n,0,.4),n,atol=1e-14)
    def test_all_poses_share_native_world_ray_including_nonaxial(self):
        d=np.array([.11,.23,.91]);d/=np.linalg.norm(d)
        for p in paths(d,5):
            np.testing.assert_allclose(p['initial_object_exit_world'],d,atol=1e-14)
            np.testing.assert_allclose(p['object_translation_waypoints_world_m'][1],.5*d,atol=1e-14)
            self.assertFalse(p['rotation_allowed'])
    def test_moving_true_patch_changes_moment_and_force_cache_key(self):
        mesh=trimesh.creation.box();face=0;tri=mesh.triangles[[face]].copy()
        contact=dict(triangles_m=tri,source_faces=np.array([face]))
        moved=dict(contact,triangles_m=tri+np.array([.01,.02,0]))
        self.assertNotEqual(contact_key([contact]),contact_key([moved]))
        normal=-mesh.face_normals[face]
        old=np.cross(tri.reshape(-1,3),normal);new=np.cross(moved['triangles_m'].reshape(-1,3),normal)
        np.testing.assert_allclose(new-old,np.broadcast_to(np.cross([.01,.02,0],normal),old.shape),atol=1e-14)
    def test_nonfinite_or_worse_actual_volume_cannot_improve(self):
        for new in (np.nan,np.inf,101.,100.):self.assertFalse(improves(100.,new))
        self.assertTrue(improves(100.,99.))

if __name__=='__main__':unittest.main()
