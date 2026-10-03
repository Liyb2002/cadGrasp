"""Nearest-plane targeting, including real clipped-solid fallback."""
from pathlib import Path
import sys
import unittest

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step4_connect_support.baseline_current import build_local_bodies as L


class NearestFloorTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box(extents=[.02, .02, .02])
        self.mesh.apply_translation([.04, .025, .08])
        self.bases = np.array([np.eye(3), [[0.,1,0],[0,0,1],[1,0,0]]])
        self.offsets = np.zeros((2,3))

    def test_own_pose_is_eligible_and_rotation_is_respected(self):
        targets = L.floor_targets(self.mesh.vertices, self.bases, self.offsets, 1)
        self.assertEqual([f for _,f in targets], [1,0])
        np.testing.assert_allclose([d for d,_ in targets], [.03,.07])
        self.assertEqual(L.floor_targets(self.mesh.vertices, self.bases, self.offsets,
                                        1, 'opposite'), [(targets[1][0],0)])

    def test_all_three_planes_are_ranked_and_ties_are_stable(self):
        bases = np.concatenate([self.bases, [[[0.,0,1],[1,0,0],[0,1,0]]]])
        offsets = np.vstack([self.offsets, [0,.005,0]])
        targets = L.floor_targets(self.mesh.vertices, bases, offsets, 0)
        self.assertEqual([f for _,f in targets], [2,1,0])
        np.testing.assert_allclose([d for d,_ in targets], [.01,.03,.07])
        tied = L.floor_targets(self.mesh.vertices, np.array([np.eye(3)]*2), self.offsets, 1)
        self.assertEqual([f for _,f in tied], [0,1])
        with self.assertRaises(ValueError):
            L.floor_targets(self.mesh.vertices, bases, offsets, 0, 'opposite')

    def test_blocked_nearest_floor_falls_back_to_a_real_connected_sole(self):
        original = L.S.solid(self.mesh)
        def connected_piece(value, required):
            for component in value.decompose():
                if all(abs((item-component).volume())*L.S.SCALE**3 <= 8e-14 for item in required):
                    return component
        # The x=0 sole is removed; the z=0 sole remains reachable.
        body, selection = L.grow_initial_body(self.mesh.vertices,
            self.mesh.vertices+[.008,0,0], original, self.bases, self.offsets, 1,
            carve=lambda value:value.trim_by_plane([1,0,0], .01/L.S.SCALE),
            connected_piece=connected_piece)
        self.assertEqual(selection['target_floor_index'], 0)
        self.assertEqual(selection['rejected_nearer_floors'][0]['floor_index'], 1)
        self.assertIn('no_connected_landing_area', selection['rejected_nearer_floors'][0]['reasons'])
        self.assertEqual(len(body.decompose()), 1)
        self.assertLess(abs((original-body).volume())*L.S.SCALE**3, 8e-14)
        vertices = L.S.unpack(body).vertices
        landing = vertices[np.abs(vertices[:,2]) < 1e-9,:2]
        self.assertGreaterEqual(L.S.MultiPoint(landing).convex_hull.area, 1e-6)


if __name__ == '__main__':
    unittest.main()
