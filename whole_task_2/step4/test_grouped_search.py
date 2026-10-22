"""Checks that grouping means actual installation reuse, not a host label."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from .angular_model import AngularLayout
from .grouped_search import balanced_partition, memberships, move_group_material, installation_summary
from fixture_state import installed_state


class GroupingTests(unittest.TestCase):
    def test_balanced_partition_covers_all_thirteen_once_for_each_allowed_count(self):
        rotations = Rotation.random(13, random_state=3).as_matrix()
        for count in [3, 4, 5]:
            groups = balanced_partition(rotations, np.array([0., 0., 1.]), count)
            self.assertEqual(len(groups), count)
            self.assertEqual(sorted(k for ids in groups for k in ids), list(range(13)))
            self.assertLessEqual(max(map(len, groups))-min(map(len, groups)), 1)

    def test_whole_group_move_preserves_all_world_tasks(self):
        native = np.repeat(np.eye(4)[None], 3, axis=0)
        native[0, :3, :3] = Rotation.from_euler('xyz', [20, 30, 40], degrees=True).as_matrix()
        model = SimpleNamespace(native=native, activate=lambda q: None)
        q = AngularLayout(np.repeat(np.eye(4)[None], 3, axis=0), np.zeros((3, 3)),
                          np.array([0, 0, 2]), (0, 1, 2))
        before = [installed_state(native, q, q.hosts[k]) @ q.placements[k] for k in q.active]
        trial = move_group_material(model, q, 0, np.array([.03, -.02, .1]))
        for k in q.active:
            after = installed_state(native, trial, trial.hosts[k]) @ trial.placements[k]
            np.testing.assert_allclose(after, before[k], atol=1e-12)
        np.testing.assert_array_equal(trial.placements[2], q.placements[2])
        self.assertEqual(memberships(trial), {0: [0, 1], 2: [2]})

    def test_count_uses_full_execution_transform(self):
        q = SimpleNamespace(active=(0, 1, 2))
        transforms = np.repeat(np.eye(4)[None], 3, axis=0)
        transforms[2, 0, 3] = .01
        model = SimpleNamespace(activate=lambda q: None,
            execution=lambda q, k: dict(pose=f'W{k}', world_from_support=transforms[k]))
        summary = installation_summary(model, q)
        self.assertEqual(summary['actual_support_pose_count'], 2)
        self.assertEqual(summary['support_orientation_count'], 1)
        self.assertEqual(summary['group_sizes'], [2, 1])


if __name__ == '__main__':
    unittest.main()

class FloorPlacementTests(unittest.TestCase):
    def test_opposite_floor_constraints_gain_clearance_without_moving_tasks(self):
        from .grouped_placement import fit_groups
        native = np.repeat(np.eye(4)[None], 2, axis=0)
        native[1, :3, :3] = np.diag([-1., 1., -1.])
        q = AngularLayout(np.repeat(np.eye(4)[None], 2, axis=0), np.zeros((2, 3)), np.arange(2), (0, 1))
        vertices = np.array([[x, y, z] for x in [-.02, .02] for y in [-.02, .02] for z in [0., .04]])
        hull = np.array([[-.01, -.01], [.01, -.01], [.01, .01], [-.01, .01]])
        rows = [dict(index=k, T=native[k].copy(), hull=hull) for k in range(2)]
        model = SimpleNamespace(native=native, activate=lambda q: None, mesh=SimpleNamespace(vertices=vertices),
            poses=['A', 'B'], base_oracle=SimpleNamespace(rows=lambda q: rows),
            execution=lambda q, k: dict(world_from_support=installed_state(native, q, k)))
        trial, record = fit_groups(model, q, margin=.002)
        for k in range(2):
            np.testing.assert_allclose(installed_state(native, trial, k) @ trial.placements[k], native[k], atol=1e-12)
            ground = np.c_[hull, np.zeros(4)]
            inverse = np.linalg.inv(installed_state(native, trial, k))
            material = ground @ inverse[:3, :3].T+inverse[:3, 3]
            other = installed_state(native, trial, 1-k)
            heights = material @ other[2, :3]+other[2, 3]
            self.assertGreaterEqual(float(heights.min()), .002-1e-10)
