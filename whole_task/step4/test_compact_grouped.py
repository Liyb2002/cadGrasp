"""Regression checks for real occupancy and roundoff-safe installation counts."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from .angular_model import AngularLayout
from .compact_grouped import execution_box, shared_static_bounds


class CompactGroupingTests(unittest.TestCase):
    def test_signed_zero_does_not_create_an_extra_support_installation(self):
        vertices = np.array([[x, y, z] for x in [-.02, .02] for y in [-.01, .01] for z in [0., .04]])
        q = AngularLayout(np.repeat(np.eye(4)[None], 2, axis=0), np.zeros((2, 3)), np.zeros(2, int), (0, 1))
        t1 = np.eye(4); t2 = np.eye(4); t2[0, 3] = -0.
        result = shared_static_bounds(vertices, q.arrays(np.repeat(np.eye(4)[None], 2, axis=0)),
            full_vertices=vertices, pose_names=['A', 'B'],
            execution=[dict(world_from_support=t.tolist()) for t in [t1, t2]])
        self.assertEqual(result['support_pose_count'], 1)
        self.assertEqual(result['object_pose_counts_per_support_pose'], [2])
        self.assertAlmostEqual(result['with_full_support']['volume_litres'], .032)

    def test_execution_envelope_is_invariant_to_material_coordinate_gauge(self):
        vertices = np.array([[x, y, z] for x in [-.07, .07] for y in [-.02, .02] for z in [0., .03]])
        q = AngularLayout(np.repeat(np.eye(4)[None], 2, axis=0), np.zeros((2, 3)), np.arange(2), (0, 1))
        q.placements[1, 0, 3] = .13
        installs = np.repeat(np.eye(4)[None], 2, axis=0)
        installs[1, :3, :3] = Rotation.from_euler('y', 60, degrees=True).as_matrix()
        model = SimpleNamespace(mesh=SimpleNamespace(vertices=vertices), activate=lambda q: None,
            execution=lambda q, k: dict(world_from_support=installs[k]))
        before = execution_box(model, q)
        gauge = np.eye(4); gauge[:3, :3] = Rotation.from_euler('xyz', [23, -41, 17], degrees=True).as_matrix()
        gauge[:3, 3] = [.1, -.05, .2]
        changed = q.copy(); changed.placements = gauge @ changed.placements
        installs = installs @ np.linalg.inv(gauge)
        after = execution_box(model, changed)
        np.testing.assert_allclose(before[0], after[0], atol=1e-14)
        np.testing.assert_allclose(before[1], after[1], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
