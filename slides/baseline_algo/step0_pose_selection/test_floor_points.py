"""Direct rigid mapping and unrestricted-floor regressions for current Step4."""
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from step0_pose_selection import floor_points as C
from step0_pose_selection.floor_points import pressure_centers


def frame(rotation=None, translation=(0., 0., 0.)):
    t = np.eye(4)
    if rotation is not None:
        t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def task(mesh, transform, pose):
    world = mesh.copy(); world.apply_transform(transform)
    return SimpleNamespace(pose=pose, floor=np.zeros(3), domain=SimpleNamespace(mesh=world))


class FloorPointTests(unittest.TestCase):
    def test_mapping_matches_independent_object_roundtrip(self):
        rng = np.random.default_rng(892)
        frames = [frame(r, t) for r, t in zip(Rotation.random(5, random_state=rng).as_matrix(),
                                             rng.normal(size=(5, 3)))]
        xy = [rng.normal(size=(57, 2)) for _ in frames]
        result = C.check_floor_points(xy, frames, list('abcde'))
        for i, source in enumerate(frames):
            world = np.c_[xy[i], np.zeros(len(xy[i]))]
            obj = (world-source[:3, 3])@source[:3, :3]
            for j, target in enumerate(frames):
                expected = obj@target[:3, :3].T+target[:3, 3]
                np.testing.assert_allclose(C.map_points(world, source, target), expected, atol=1e-14)
                np.testing.assert_allclose(result.heights[i][:, j], expected[:, 2], atol=1e-14)
            np.testing.assert_allclose(result.heights[i][:, i], 0, atol=1e-14)

    def test_no_lateral_bound_or_object_hole_rejection(self):
        frames = [frame(), frame(Rotation.from_euler('z', 85, degrees=True).as_matrix(), (17, -2, 0))]
        clouds = [np.array([[1.e6, -1.e6], [0., 0.]]), np.array([[-1.e6, 1.e6]])]
        result = C.check_floor_points(clouds, frames, ['a', 'b'])
        self.assertTrue(all(row['passed'] for row in result.rows))

    def test_directed_pair_reports_source_and_target(self):
        # Target b's world height is minus a's x, with identical object origin.
        frames = [frame(), frame(Rotation.from_euler('y', 90, degrees=True).as_matrix())]
        result = C.check_floor_points([np.array([[2., 0], [-1., 0]]), np.array([[1., 0]])],
                                      frames, ['a', 'b'])
        pair = result.rows[0]['per_other_pose'][0]
        self.assertEqual(pair['other_pose'], 'b')
        self.assertEqual(pair['violating_sample_count'], 1)
        self.assertAlmostEqual(pair['minimum_height_m'], -2)
        self.assertEqual(pair['worst_sample_index'], 0)
        np.testing.assert_allclose(pair['worst_point_target_world_m'], [0, 0, -2], atol=1e-14)
        self.assertTrue(result.rows[1]['passed'])

    def test_every_target_checked_and_pose_order_irrelevant(self):
        frames = [frame(), frame(), frame(), frame(Rotation.from_euler('y', 90, degrees=True).as_matrix())]
        clouds = [np.array([[1., 0]])]+[np.array([[0., 0]])]*3
        result = C.check_floor_points(clouds, frames, list('abcd'))
        self.assertEqual([r['violating_sample_count'] for r in result.rows], [1, 0, 0, 0])
        self.assertEqual(result.rows[0]['per_other_pose'][-1]['other_pose'], 'd')
        order = [3, 1, 0, 2]
        reordered = C.check_floor_points([clouds[i] for i in order], [frames[i] for i in order],
                                         [list('abcd')[i] for i in order])
        self.assertEqual([r['violating_sample_count'] for r in reordered.rows], [0, 0, 1, 0])

    def test_unique_samples_not_sum_of_pair_violations(self):
        rotation = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        result = C.check_floor_points([np.array([[1., 0]])]*3,
                                      [frame(), frame(rotation), frame(rotation)], ['a', 'b', 'c'])
        self.assertEqual(result.rows[0]['violating_sample_count'], 1)
        self.assertEqual(sum(p['violating_sample_count'] for p in result.rows[0]['per_other_pose']), 2)

    def test_tangent_and_tolerance(self):
        frames = [frame(), frame(translation=(0, 0, -.5*C.TOL)), frame(translation=(0, 0, -2*C.TOL))]
        result = C.check_floor_points([np.zeros((1, 2))]*3, frames, ['a', 'b', 'c'])
        self.assertEqual([r['violating_sample_count'] for r in result.rows[0]['per_other_pose']], [0, 1])

    def test_malformed_inputs_rejected(self):
        for bad in [frame(np.diag([2., 1, 1])), frame(np.diag([-1., 1, 1])), np.full((4, 4), np.nan)]:
            with self.assertRaises(ValueError): C.validate_frames([bad])
        for cloud in [np.empty((0, 2)), np.array([[np.nan, 0]]), np.zeros((2, 3))]:
            with self.assertRaises(ValueError): C.check_floor_points([cloud], [frame()], ['a'])
        with self.assertRaises(ValueError):
            C.check_floor_points([np.zeros((1, 2))]*2, [frame()]*2, ['a', 'a'])

    def test_invalid_object_or_pivot_is_input_error(self):
        mesh = trimesh.creation.box([1., 1, 1])
        with self.assertRaisesRegex(ValueError, 'Object itself penetrates'):
            C.validate_task_geometry([task(mesh, frame(), 'a')], [frame()])
        mesh.apply_translation([0, 0, .5])
        valid = task(mesh, frame(), 'a')
        C.validate_task_geometry([valid], [frame()])
        valid.floor = np.array([0., 0, -.01])
        with self.assertRaisesRegex(ValueError, 'pivot'):
            C.validate_task_geometry([valid], [frame()])

    def test_cop_formula_matches_independent_external_force_formula(self):
        rng = np.random.default_rng(28)
        origin = np.array([.2, -.3, .4])
        q = rng.normal(size=(91, 3)); f = rng.normal(size=(91, 3))
        f[:, 2] = rng.uniform(-.5, .5, len(q))
        reaction = np.array([0., 0., 1.])-f
        loads = np.c_[reaction, -np.cross(q-origin, f)]
        xy, normal = pressure_centers(loads, origin)
        expected = (origin[:2]-q[:, :2]*f[:, 2, None]+q[:, 2, None]*f[:, :2])/(1-f[:, 2, None])
        np.testing.assert_allclose(xy, expected, atol=1e-14)
        np.testing.assert_array_equal(normal, 1-f[:, 2])


if __name__ == '__main__':
    unittest.main()
