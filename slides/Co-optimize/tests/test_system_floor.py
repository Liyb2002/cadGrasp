"""Physical checks for final-position system-floor pressure centers."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helper_func'))
from system_floor import final_task_frame, system_floor_demands, transform_points


class SystemFloorTests(unittest.TestCase):
    def test_pure_gravity_projects_com_to_floor_at_any_height(self):
        result = system_floor_demands([[0., 0., 1., 0., 0., 0.]], [.12, -.08, .30])
        np.testing.assert_allclose(result['floor_demands_world_m'], [[.12, -.08, 0.]])

    def test_translation_includes_height_lever_arm(self):
        loads = np.array([[.3, -.2, 1.2, .04, -.03, .02],
                          [-.1, .25, .8, -.01, .04, -.03]])
        delta = np.array([.02, -.03, .07])
        first = system_floor_demands(loads, [.1, .2, .15])
        second = system_floor_demands(loads, np.array([.1, .2, .15]) + delta)
        expected = delta[:2] - delta[2] * loads[:, :2] / loads[:, 2, None]
        np.testing.assert_allclose(second['floor_demands_xy_m'] - first['floor_demands_xy_m'], expected)
        self.assertFalse(np.allclose(expected, np.broadcast_to(delta[:2], expected.shape)))

    def test_projection_matches_external_force_and_gravity_balance(self):
        com = np.array([.06, -.02, .14])
        q = np.array([[.08, .03, .19], [.02, -.04, .11]])
        pushes = np.array([[.12, -.18, -.2], [-.08, .06, .1]])
        force = np.array([0., 0., 1.]) - pushes
        moments = -np.cross(q - com, pushes)
        loads = np.c_[force, moments]
        result = system_floor_demands(loads, com)
        direct = -np.cross(q, pushes) - np.cross(com, [0., 0., -1.])
        np.testing.assert_allclose(result['wrench_world_origin'][:, 3:], direct)
        p = result['floor_demands_world_m']
        at_pressure_center = result['wrench_world_origin'][:, 3:] - np.cross(p, force)
        np.testing.assert_allclose(at_pressure_center[:, :2], 0., atol=1e-14)
        # Torsional demand is deliberately retained, not declared satisfied.
        self.assertTrue(np.any(np.abs(at_pressure_center[:, 2]) > 1e-5))

    def test_fixture_weight_changes_pressure_center_by_vertical_weight_average(self):
        bare = system_floor_demands([[.1, 0., 1., .01, .02, 0.]], [.1, .05, .2])
        weighted = system_floor_demands([[.1, 0., 1., .01, .02, 0.]], [.1, .05, .2],
                                       fixture_weight_ratio=.3, fixture_com_world_m=[-.05, .08, .02])
        expected = (bare['floor_demands_xy_m'] + .3 * np.array([-.05, .08])) / 1.3
        np.testing.assert_allclose(weighted['floor_demands_xy_m'], expected)
        np.testing.assert_allclose(weighted['total_floor_normal_mg'], [1.3])

    def test_grounded_and_airborne_have_the_same_input_demands(self):
        loads = np.array([[.2, .1, 1., .01, -.02, .03]])
        original = loads.copy()
        grounded = system_floor_demands(loads, [0., 0., .1])
        raised = system_floor_demands(loads, [0., 0., .2])
        np.testing.assert_array_equal(loads, original)
        np.testing.assert_allclose(raised['floor_demands_xy_m'] - grounded['floor_demands_xy_m'], [[-.02, -.01]])

    def test_seating_uses_host_frame_not_native_object_frame_for_fixture(self):
        native = np.array([np.eye(4), np.eye(4)])
        native[1, :3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        placements = np.array([np.linalg.inv(native[1]), np.eye(4)])
        placements[0, :3, 3] = [.03, 0., .02]
        fixture, obj = final_task_frame(native, placements, [1, 1], 0)
        np.testing.assert_array_equal(fixture, native[1])
        np.testing.assert_allclose(obj[:3, :3], np.eye(3))
        world = np.array([[.1, .2, 0.]])
        local = transform_points(world, np.linalg.inv(fixture))
        np.testing.assert_allclose(transform_points(local, fixture), world)
        with self.assertRaises(ValueError):
            final_task_frame(native, np.array([np.eye(4), np.eye(4)]), [1, 1], 0)

    def test_invalid_vertical_demands_and_weight_are_explicit_errors(self):
        for force_z in [0., -1.]:
            with self.assertRaises(ValueError):
                system_floor_demands([[0., 0., force_z, 0., 0., 0.]], [0., 0., .1])
        with self.assertRaises(ValueError):
            system_floor_demands([[0., 0., 1., 0., 0., 0.]], [0., 0., .1], fixture_weight_ratio=-.1)
        with self.assertRaises(ValueError):
            system_floor_demands([[0., 0., 1., 0., 0., 0.]], [0., 0., .1], fixture_weight_ratio=.1)


if __name__ == '__main__':
    unittest.main()
