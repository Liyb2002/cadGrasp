"""Physical invariants of selecting a common tool/ground reference."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from subdivide_task.operations.state import Layout, transform
from subdivide_task.step3.independent_geometry import pressure_points
from .height_reference import lower_layout, floor_limit


class HeightReferenceTests(unittest.TestCase):
    def test_lowering_preserves_seats_exit_and_tool_relative_work_area(self):
        rng = np.random.default_rng(135)
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        material = poses.copy()
        poses[:, :3, :3] = Rotation.random(3, random_state=rng).as_matrix()
        material[:, :3, :3] = Rotation.random(3, random_state=rng).as_matrix()
        poses[:, :3, 3] = rng.normal(size=(3, 3))
        material[:, :3, 3] = rng.normal(size=(3, 3))
        q = Layout(poses, material, np.tile([0., 0., 1.], (3, 1)))
        original = q.copy()
        h = .085
        new = lower_layout(q, h)
        np.testing.assert_allclose(new.material @ new.poses, q.material @ q.poses, atol=1e-14, rtol=0.)
        head = np.array([0., -.4, .37])
        points = rng.normal(size=(12, 3))
        for i in range(3):
            np.testing.assert_allclose(transform(points, new.poses[i]) - (head - [0., 0., h]),
                                       transform(points, q.poses[i]) - head, atol=1e-14, rtol=0.)
            np.testing.assert_allclose(new.material[i, :3, :3] @ new.exits[i],
                                       q.material[i, :3, :3] @ q.exits[i], atol=1e-14, rtol=0.)
        for key, array in q.arrays().items():
            np.testing.assert_array_equal(array, original.arrays()[key])

    def fixture(self):
        M = np.repeat(np.eye(4)[None], 2, axis=0)
        M[0, :3, 3] = [0., 0., 1.]
        M[1, :3, :3] = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        M[1, :3, 3] = [2., 0., 0.]
        q = Layout(np.repeat(np.eye(4)[None], 2, axis=0), M, np.tile([0., 0., 1.], (2, 1)))
        com = np.array([.05, .02, .4])
        force = np.array([.5, .3, 2.])
        origin_moment = np.array([.02 * force[2], -2.15 * force[2], 0.])
        targets = np.r_[force, origin_moment - np.cross(com, force)][None]
        return q, com, targets

    def test_pressure_floor_bound_predicts_recomputed_ground_demands(self):
        q, com, targets = self.fixture()
        result, slack, slope = floor_limit(q, 0, targets, com)
        self.assertAlmostEqual(result['max_lowering_m'], .2, places=12)
        self.assertEqual(result['other_index'], 1)
        for h in (0., .05, .2, .21):
            new = lower_layout(q, h)
            xy = pressure_points(targets, com - [0., 0., h])
            points = transform(np.c_[xy, np.zeros(len(xy))], new.material[0])
            n = new.material[:, :3, 2]
            b = np.einsum('ij,ij->i', n, new.material[:, :3, 3])
            np.testing.assert_allclose(points @ n.T - b, slack + h * slope, atol=1e-14, rtol=0.)
        self.assertLess((slack + .21 * slope).min(), -.007)

    def test_infeasible_original_pressures_are_rejected(self):
        q, com, targets = self.fixture()
        q.material[1, 0, 3] = 3.
        with self.assertRaisesRegex(ValueError, 'starting pressure'):
            floor_limit(q, 0, targets, com)


if __name__ == '__main__':
    unittest.main()
