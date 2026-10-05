"""Internal void shells must not be confused with separate printed bodies."""
import sys
from pathlib import Path
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step5_connect_support import solids as S,ground as G,belt_geometry as B

class SolidComponentTests(unittest.TestCase):
    def hollow(self):
        value=G.manifold.Manifold.cube([2.,2.,2.])-G.manifold.Manifold.cube([1.,1.,1.]).translate([.5,.5,.5])
        data=value.to_mesh64()
        return trimesh.Trimesh(np.asarray(data.vert_properties[:,:3]),np.asarray(data.tri_verts),process=False)

    def test_hollow_body_is_one_solid_and_void_is_preserved(self):
        mesh,record=B.union_parts([self.hollow()],2.)
        self.assertTrue(record['one_solid'],record)
        self.assertEqual(record['component_count'],1)
        self.assertEqual(record['boundary_component_count'],2)
        self.assertEqual(record['cavity_count'],1)
        self.assertTrue(record['cavity_containment_verified'])
        self.assertAlmostEqual(mesh.volume,7.)
        self.assertFalse(mesh.contains([[1.,1.,1.]])[0])

    def test_solid_island_inside_a_void_is_still_a_separate_body(self):
        island=trimesh.creation.box([.2, .2, .2]);island.apply_translation([1., 1., 1.])
        _,record=B.union_parts([self.hollow(),island],2.)
        self.assertFalse(record['one_solid'])
        self.assertEqual(record['positive_boundary_shells'],2)

    def test_disconnected_positive_piece_is_not_erased(self):
        tiny=trimesh.creation.box([.001, .001, .001]);tiny.apply_translation([3., 0, 0])
        _,record=B.union_parts([self.hollow(),tiny],3.)
        self.assertFalse(record['one_solid'])
        self.assertEqual(record['positive_boundary_shells'],2)

if __name__=='__main__':unittest.main()
