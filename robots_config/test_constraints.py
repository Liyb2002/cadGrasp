"""Check sign, frames and the difference between existence and full coverage."""
import copy
import unittest

import numpy as np

from robots_config.constraints import (
    direction_allowed, load_configs, process_cone_fully_covered,
    rotate_direction, surface_normal_compatible, tool_direction_allowed_at_surface,
    unit, validate_config, nominal_tool_positions, surface_point_allowed, standoff_allowed,
)


class DirectionConstraintsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = load_configs()
        cls.top = cls.configs[0]

    def test_all_configs_and_axis_sign(self):
        self.assertEqual(len(self.configs), 10)
        for config in self.configs:
            axis = np.array(config["task_constraint"]["working_direction_world"])
            self.assertTrue(direction_allowed(config, axis))
            self.assertFalse(direction_allowed(config, -axis))
            self.assertTrue(surface_normal_compatible(config, -axis))
            self.assertFalse(surface_normal_compatible(config, axis))
            self.assertTrue(tool_direction_allowed_at_surface(config, axis, -axis))

    def test_closed_cone_boundary(self):
        half = self.top["task_constraint"]["cone_half_angle_deg"]
        def tilted(angle):
            angle = np.radians(angle)
            return [np.sin(angle), 0, -np.cos(angle)]
        self.assertTrue(direction_allowed(self.top, tilted(half)))
        self.assertFalse(direction_allowed(self.top, tilted(half + 0.01)))

    def test_zero_half_angle_with_oblique_axis(self):
        config = copy.deepcopy(self.configs[1])
        config["task_constraint"]["cone_half_angle_deg"] = 0
        config["task_constraint"]["surface_incidence_half_angle_deg"] = 0
        axis = np.array(config["task_constraint"]["working_direction_world"])
        self.assertTrue(direction_allowed(config, axis))
        self.assertTrue(tool_direction_allowed_at_surface(config, axis, -axis))

    def test_intersection_is_not_full_coverage(self):
        config = copy.deepcopy(self.top)
        config["task_constraint"]["cone_half_angle_deg"] = 25
        config["task_constraint"]["surface_incidence_half_angle_deg"] = 10
        normal = [-0.5, 0, np.sqrt(3) / 2]  # inward axis is tilted by 30 deg
        self.assertTrue(surface_normal_compatible(config, normal))
        self.assertFalse(tool_direction_allowed_at_surface(config, [0, 0, -1], normal))
        self.assertFalse(process_cone_fully_covered(config, -np.array(normal), 10))
        self.assertTrue(process_cone_fully_covered(config, [0, 0, -1], 25))
        self.assertFalse(process_cone_fully_covered(config, [0, 0, -1], 30))

    def test_object_rotation_does_not_rotate_robot_cone(self):
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        normal_world = rotate_direction([0, 0, 1], rotation)
        np.testing.assert_allclose(normal_world, [1, 0, 0])
        self.assertFalse(surface_normal_compatible(self.top, normal_world))

    def test_invalid_inputs(self):
        for vector in ([0, 0, 0], [np.nan, 0, 1], [1, 2]):
            with self.assertRaises(ValueError):
                unit(vector)
        with self.assertRaises(ValueError):
            rotate_direction([1, 0, 0], np.diag([1, 1, -1]))
        with self.assertRaises(ValueError):
            rotate_direction([1, 0, 0], np.eye(3) * 2)
        bad = copy.deepcopy(self.top)
        bad["task_constraint"]["cone_half_angle_deg"] = 90
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_standoff_and_nominal_geometry(self):
        for config in self.configs:
            task = config['task_constraint']
            axis = np.array(task['working_direction_world'])
            center = np.array(task['world_position_constraint']['center_world_m'])
            gap = task['tcp_to_surface_distance_m']
            positions = nominal_tool_positions(config)
            np.testing.assert_allclose(center - positions['tcp_world_m'], axis * gap['nominal'], atol=1e-12)
            length = config['feasibility']['tcp_transform']['translation_flange_m'][2]
            np.testing.assert_allclose(positions['tcp_world_m'] - positions['flange_world_m'], axis * length, atol=1e-12)
            self.assertTrue(standoff_allowed(config, gap['min']))
            self.assertTrue(standoff_allowed(config, gap['max']))
            self.assertFalse(standoff_allowed(config, gap['max'] + .001))
            self.assertFalse(standoff_allowed(config, -.001))
        np.testing.assert_allclose(nominal_tool_positions(self.top)['tcp_world_m'],
                                   self.top['task_constraint']['world_position_constraint']['center_world_m'])
        spray = self.configs[-1]
        np.testing.assert_allclose(nominal_tool_positions(spray)['tcp_world_m'], [0, .62, .4])

    def test_work_window_on_oblique_surface(self):
        config = self.configs[3]
        patch = config['task_constraint']['world_position_constraint']
        center = np.array(patch['center_world_m'])
        normal = np.array(patch['outward_normal_world'])
        tangent = np.cross(normal, [0, 0, 1])
        tangent /= np.linalg.norm(tangent)
        self.assertTrue(surface_point_allowed(config, center))
        self.assertTrue(surface_point_allowed(config, center + patch['radius_m'] * tangent))
        self.assertFalse(surface_point_allowed(config, center + (patch['radius_m'] + .001) * tangent))
        self.assertFalse(surface_point_allowed(config, center + (patch['normal_tolerance_m'] + .001) * normal))

    def test_invalid_station_and_standoff(self):
        bad = copy.deepcopy(self.top)
        bad['task_constraint']['tcp_to_surface_distance_m']['min'] = .01
        with self.assertRaises(ValueError):
            validate_config(bad)
        bad = copy.deepcopy(self.top)
        bad['task_constraint']['world_position_constraint']['center_world_m'][0] += .1
        with self.assertRaises(ValueError):
            validate_config(bad)


if __name__ == "__main__":
    unittest.main()
