"""Local subdivision must preserve the solid and shared edge correspondence."""
import unittest
import numpy as np
import trimesh
from codes.precompute_objects.lowpoly import split_large_faces
from codes.precompute_objects.work_regions import refine

class LowpolyTests(unittest.TestCase):
    def test_local_bisection_preserves_closed_solid_and_area(self):
        raw=trimesh.creation.box([.08,.12,.16])
        low=split_large_faces(raw)
        self.assertTrue(low.is_watertight)
        self.assertTrue(low.is_winding_consistent)
        self.assertEqual(low.euler_number,raw.euler_number)
        self.assertTrue(np.all(np.bincount(low.edges_unique_inverse)==2))
        self.assertLessEqual(len(low.faces),5000)
        np.testing.assert_allclose(low.volume,raw.volume,atol=1e-15)
        np.testing.assert_allclose(low.area,raw.area,atol=1e-15)
        self.assertLessEqual(low.area_faces.max()/low.area,.005*(1+1e-12))

    def test_work_mesh_does_not_expand_a_prepared_lowpoly_mesh(self):
        low=split_large_faces(trimesh.creation.box([.08,.12,.16]))
        working,rounds=refine(low)
        self.assertEqual(rounds,0)
        np.testing.assert_array_equal(working.faces,low.faces)
        np.testing.assert_array_equal(working.vertices,low.vertices)

    def test_budget_cannot_create_t_junctions_to_force_completion(self):
        with self.assertRaisesRegex(ValueError,'budget'):
            split_large_faces(trimesh.creation.box(),cap=12)

if __name__=='__main__':unittest.main()
