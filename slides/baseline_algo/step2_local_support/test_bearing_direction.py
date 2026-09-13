"""Admit all hemispheres while retaining geometry and invalid-input checks."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step2_local_support import bearing_direction as H, circles as P


class BearingDirectionTests(unittest.TestCase):
    def test_oblique_upward_and_horizontal_are_admitted(self):
        for normal in ([.6,0,.8],[0,0,1],[1,0,0]):
            with self.subTest(normal=normal):
                self.assertTrue(H.check(SimpleNamespace(face_normals=np.array([normal])),[0])['passed'])
        self.assertTrue(H.check(SimpleNamespace(face_normals=np.asarray(np.array([[.6,0,-.8]]))),[0])['passed'])

    def test_mixed_normals_are_admitted_for_one_rigid_support(self):
        normals=np.asarray(np.array([[0,0,-1]]*99+[[.8,0,.6]]))
        self.assertLess(normals.mean(axis=0)[2],0)
        result=H.check(SimpleNamespace(face_normals=normals),range(100))
        self.assertTrue(result['passed']);self.assertEqual(result['non_downward_face_count'],1)
        self.assertEqual(result['weakest_source_face'],99)

    def test_horizontal_band_does_not_filter_candidates(self):
        for z in (1e-16,0.,-1e-16,-H.NORMAL_Z_GUARD):
            self.assertTrue(H.check(SimpleNamespace(face_normals=np.array([[1.,0,z]])),[0])['passed'])
        self.assertTrue(H.check(SimpleNamespace(face_normals=np.array([[1.,0,-2*H.NORMAL_Z_GUARD]])),[0])['passed'])

    def test_real_geometry_accepts_top_and_bottom_without_changing_them(self):
        mesh=trimesh.creation.box([1, 1, 1]);mesh.apply_translation([0, 0, 2])
        top=int(np.argmax(mesh.face_normals[:,2]));bottom=int(np.argmin(mesh.face_normals[:,2]))
        for size in (.01,1,100):
            scaled=mesh.copy();scaled.apply_scale(size)
            patch={top:scaled.triangles[top].copy()};before=patch[top].copy()
            check=P.LocalClearance(scaled,.01*size).check(patch)
            self.assertTrue(check['valid']);self.assertEqual(check['status'],'valid')
            np.testing.assert_array_equal(patch[top],before)
            self.assertTrue(P.LocalClearance(scaled,.01*size).check({bottom:scaled.triangles[bottom]})['valid'])

    def test_empty_patch_is_not_a_contact(self):
        self.assertFalse(H.check(SimpleNamespace(face_normals=np.zeros((0,3))),[])['passed'])


if __name__=='__main__':unittest.main()
