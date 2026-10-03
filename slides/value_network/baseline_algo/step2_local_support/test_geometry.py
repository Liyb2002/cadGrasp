"""Counterexamples for the actual geometric predicates, independent of C5."""
import sys
from pathlib import Path
import unittest
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as G


class GeometryTests(unittest.TestCase):
    def test_entire_volume_catches_edge_obstacle_between_clear_endpoints(self):
        base = trimesh.creation.box([4, 4, 1])
        base.apply_translation([0, 0, -.5])
        obstacle = trimesh.creation.box([.2, .3, .1])
        obstacle.apply_translation([.8, 0, .45])
        mesh = trimesh.util.concatenate([base, obstacle])
        poly = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0.]])
        cell = np.vstack([poly, poly+[0,0,1]])
        # The center ray and both endpoint faces miss the obstacle.
        self.assertFalse(mesh.ray.intersects_any([[0,0,1e-6]], [[0,0,1]])[0])
        self.assertGreaterEqual(G.Clearance(mesh, 1e-9).obstruction(cell), len(base.faces))

    def test_design_contact_on_boundary_is_allowed(self):
        mesh = trimesh.creation.box([4, 4, 1])
        mesh.apply_translation([0, 0, -.5])
        poly = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0.]])
        self.assertEqual(G.Clearance(mesh, 1e-9).obstruction(np.vstack([poly, poly+[0,0,1]])), -1)

    def test_joined_skin_has_required_face_normal_depth(self):
        mesh = trimesh.creation.box([1., 1., 1.])
        offsets, valid = G.vertex_offsets(mesh, .01)
        self.assertTrue(valid.all())
        projections = np.einsum('fvc,fc->fv', offsets[mesh.faces], mesh.face_normals)
        self.assertGreaterEqual(projections.min(), .01-1e-12)


if __name__ == '__main__':
    unittest.main()
