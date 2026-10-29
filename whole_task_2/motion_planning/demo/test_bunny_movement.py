"""Model, search and output regressions. Run: python -m unittest -v test_bunny_movement."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from bunny_movement import HERE, Planner, load_request, package_pose, verify
from collision import World
from robot import rigid, points
from search import PlanningFailure, edge_samples, rrt_connect, grasp_candidates


class CarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request, cls.robot, cls.body, cls.directory = load_request(HERE / 'example_request.json')

    def test_full_jacobian_matches_finite_difference(self):
        r = self.robot
        q = (r.lower + r.upper) / 2
        t, j = r.frames(q, jacobian=True)
        for i in range(5):
            d = np.zeros(5)
            d[i] = 1e-7
            b = r.frames(q + d)[0]
            position = (b[:3, 3] - t[:3, 3]) / 1e-7
            rotation = Rotation.from_matrix(b[:3, :3] @ t[:3, :3].T).as_rotvec() / 1e-7
            np.testing.assert_allclose(j[:, i], np.r_[position, rotation], atol=1e-6)

    def test_jaw_changes_geometry_and_contact_span(self):
        r = self.robot
        spans = [np.linalg.norm(r.moving_pad(np.radians(a)) - r.fixed_pad) for a in [0, 10, 20, 30]]
        self.assertTrue(np.all(np.diff(spans) > 0))
        for width in [.012, .025, .05]:
            angle = r.jaw_for_width(width)
            self.assertAlmostEqual(np.linalg.norm(r.moving_pad(angle) - r.fixed_pad), width, places=9)
        q = (r.lower + r.upper) / 2
        _, a = r.frames(q, 0)
        _, b = r.frames(q, .5)
        np.testing.assert_allclose(a['gripper_link'], b['gripper_link'])
        self.assertFalse(np.allclose(a['moving_jaw_so101_v1_link'], b['moving_jaw_so101_v1_link']))

    def test_full_pose_ik_roundtrip_and_unreachable(self):
        r = self.robot
        q = (r.lower + r.upper) / 2
        wanted = r.frames(q)[0]
        solved = r.ik(wanted, [q + .01])
        self.assertIsNotNone(solved)
        np.testing.assert_allclose(r.frames(solved)[0], wanted, atol=1e-5)
        impossible = wanted.copy()
        impossible[:3, 3] += [2, 0, 0]
        self.assertIsNone(r.ik(impossible, [q]))

    def test_package_mm_and_print_lift(self):
        with tempfile.TemporaryDirectory() as folder:
            t = np.eye(4)
            t[:3, 3] = [100, 200, 300]
            info = dict(units='length mm', tasks=[dict(id='S1', world_from_object=t.tolist(), object_lift_mm=2)])
            (Path(folder) / 'info.json').write_text(json.dumps(info))
            p, _ = package_pose(folder, 'S1')
            np.testing.assert_allclose(p[:3, 3], [.1, .2, .302])
            info['units'] = 'm'
            (Path(folder) / 'info.json').write_text(json.dumps(info))
            with self.assertRaises(ValueError):
                package_pose(folder, 'S1')

    def test_rigid_transform_rejects_scale_reflection_nan(self):
        for value in (np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                rigid(value)

    def test_hardware_mapping_roundtrip(self):
        r = copy.copy(self.robot)
        r.signs = np.array([1, -1, 1, -1, 1, -1])
        r.offsets = np.array([0, 12, -3, 0, 0, 5])
        model = np.radians([10, 20, 30, 40, 50, 60])
        np.testing.assert_allclose(r.hardware_degrees(model) * r.signs + r.offsets, np.degrees(model))

    def test_table_and_limits_are_not_ignored(self):
        w = World(self.robot, self.body, self.request, self.directory)
        q = np.radians(self.request['robot']['initial_joints_deg'])
        pose = np.array(self.request['source']['world_from_object'])
        pose[2, 3] -= .1
        self.assertEqual(w.check(q, np.radians(55), pose), 'bunny_table')
        q[0] = self.robot.upper[0] + .1
        self.assertEqual(w.check(q, 0, pose), 'joint_limits')

    def test_box_obstacle_catches_enclosed_bunny(self):
        request = copy.deepcopy(self.request)
        pose = np.array(request['source']['world_from_object'])
        centre = points(self.body.centroid, pose)
        obstacle = np.eye(4)
        obstacle[:3, 3] = centre
        request['obstacles'] = [dict(name='blocked', box_extents_m=[.2, .2, .2], world_from_mesh=obstacle.tolist())]
        w = World(self.robot, self.body, request, self.directory)
        reason = w.check(np.radians(request['robot']['initial_joints_deg']), np.radians(55), pose)
        self.assertIsNotNone(reason)

    def test_floating_placement_has_no_support(self):
        w = World(self.robot, self.body, self.request, self.directory)
        pose = np.array(self.request['source']['world_from_object'])
        self.assertTrue(w.has_support(pose))
        pose[2, 3] += .10
        self.assertFalse(w.has_support(pose))

    def test_rrt_routes_around_obstacle_and_checks_start(self):
        def valid(q):
            return not (.35 <= q[0] <= .65 and -.3 <= q[1] <= .3)
        def edge(a, b):
            return all(valid(q) for q in edge_samples(a, b, .03, 0, .01))
        policy = dict(rrt_step_rad=.15, rrt_iterations=3000, rrt_seconds=3., shortcut_tries=30)
        a, b = np.array([0., 0.]), np.array([1., 0.])
        path = rrt_connect(a, b, valid, edge, np.array([0., -1.]), np.array([1., 1.]), np.random.default_rng(8), policy)
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[0], a)
        np.testing.assert_allclose(path[-1], b)
        self.assertTrue(all(edge(x, y) for x, y in zip(path, path[1:])))
        self.assertIsNone(rrt_connect(np.array([.5, 0]), b, valid, edge, a - 1, b + 1, np.random.default_rng(8), policy))

    def test_retiming_obeys_velocity_and_acceleration(self):
        planner = Planner(self.request, self.robot, self.body, self.directory)
        pose = np.array(self.request['source']['world_from_object'])
        q = np.radians(self.request['robot']['initial_joints_deg'])
        frames = planner.retime([('test', [q, q + .3, q - .2], .5, pose, False)])
        t = np.array([f['t'] for f in frames])
        x = np.array([f['q'] for f in frames])
        v = np.diff(np.degrees(x), axis=0) / np.diff(t)[:, None]
        a = np.diff(v, axis=0) / ((np.diff(t)[1:] + np.diff(t)[:-1]) / 2)[:, None]
        self.assertLessEqual(np.abs(v).max(), self.request['planner']['maximum_joint_speed_deg_s'] + 1e-8)
        self.assertLessEqual(np.abs(a).max(), self.request['planner']['maximum_joint_acceleration_deg_s2'] + 1e-8)

    def replay_fixture(self):
        request = copy.deepcopy(self.request)
        pose = np.array(request['source']['world_from_object'])
        request['target'] = dict(world_from_object=pose.tolist())
        request['robot']['park_joints_deg'] = request['robot']['initial_joints_deg']
        q = np.radians([*request['robot']['initial_joints_deg'], request['robot']['initial_jaw_deg']])
        frames = [dict(t=i * .05, q=q.copy(), object_pose=pose.copy(), phase='prepare_gripper', attached=False)
                  for i in range(3)]
        result = dict(gripper_from_object=np.eye(4), actual_target_world_from_object=pose,
                      grasp=dict(contacts_object_m=self.body.vertices[:2]))
        return request, frames, result

    def test_replay_rejects_unattached_teleport(self):
        request, frames, result = self.replay_fixture()
        frames[1]['object_pose'][0, 3] += .01
        validation = verify(request, self.robot, self.body, self.directory, frames, result)
        self.assertEqual(validation['status'], 'failed')
        self.assertIn((1, 'unattached_bunny_moved'), validation['errors'])

    def test_replay_rejects_broken_attachment(self):
        request, frames, result = self.replay_fixture()
        for f in frames:
            f['attached'] = True
            f['phase'] = 'carry'
        validation = verify(request, self.robot, self.body, self.directory, frames, result)
        self.assertEqual(validation['status'], 'failed')
        self.assertIn((0, 'broken_rigid_attachment'), validation['errors'])

    def test_oblique_contacts_respect_both_normal_cones(self):
        import trimesh
        plate = trimesh.creation.box(extents=[.05, .05, .013])
        policy = dict(self.request['planner'], grasp_samples=80, grasp_rolls=4,
                      contact_cone_half_angle_deg=20., contact_normal_tolerance_deg=20.,
                      contact_directions_per_sample=5, contact_pair_limit=0)
        candidates = grasp_candidates(plate, self.robot, np.eye(4), np.eye(4), np.random.default_rng(3), policy)
        self.assertTrue(candidates)
        self.assertTrue(any(max(g['contact_normal_angles_deg']) > 15 for g in candidates))
        self.assertTrue(all(max(g['contact_normal_angles_deg']) <= 20.00001 for g in candidates))

    def test_moving_pad_box_rotates_and_rejects_contacts_outside_measured_width(self):
        robot = copy.copy(self.robot)
        robot.gripper = copy.deepcopy(robot.gripper)
        robot.gripper['contact_patch_radius_m'] = .015
        robot.gripper['contact_patch_half_extents_m'] = {'moving': [.002, .010, .010]}
        world = World(robot, self.body, self.request, self.directory)
        jaw = np.radians(30)
        frame = robot.pad_frame(np.eye(4), jaw, 'moving')
        np.testing.assert_allclose(frame[:3, 3], robot.moving_pad(jaw), atol=1e-9)
        self.assertFalse(np.allclose(frame[:3, :3], np.eye(3)))
        grasp = dict(contacts_object_m=[robot.fixed_pad, frame[:3, 3]])
        part = next(p for p in robot.parts if p['contact'] == 'moving')
        inside = SimpleNamespace(pos=points([0, .009, 0], frame), penetration_depth=0.)
        outside = SimpleNamespace(pos=points([0, .011, 0], frame), penetration_depth=0.)
        self.assertIsNone(world.pad_contact_failure([inside], part, np.eye(4), grasp, np.eye(4), jaw))
        detail = world.pad_contact_failure([outside], part, np.eye(4), grasp, np.eye(4), jaw)
        self.assertEqual(detail['reason'], 'outside_pad_contact')
        self.assertLess(detail['maximum_radius_m'], .015)
        self.assertGreater(detail['maximum_pad_local_abs_m'][1], .010)

    def test_multistart_refinement_checks_distinct_seeds_with_full_pose_residual(self):
        grasp = json.loads((HERE / 'inputs/refinement_regression.json').read_text())
        request = copy.deepcopy(self.request)
        request['planner'].update(refine_multistart=2, refine_max_variants=4)
        planner = Planner(request, self.robot, self.body, self.directory)
        variants = list(planner.refine_grasps(grasp))
        self.assertTrue(variants)
        self.assertGreater(planner.report['refinement_seeds'], 1)
        for variant in variants:
            actual = self.robot.frames(variant['source_seed'])[0]
            desired = planner.a @ variant['object_from_gripper']
            self.assertLessEqual(np.linalg.norm(actual[:3, 3] - desired[:3, 3]), request['planner']['position_tolerance_m'])
            self.assertLessEqual(Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).magnitude(),
                                 np.radians(request['planner']['orientation_tolerance_deg']))

    def test_late_replay_failure_resumes_grasp_search(self):
        grasp = json.loads((HERE / 'inputs/refinement_regression.json').read_text())
        planner = Planner(self.request, self.robot, self.body, self.directory)
        validations = [dict(status='failed', errors=[(290, 'outside_pad_contact:moving_jaw')]), dict(status='passed')]
        with patch('bunny_movement.grasp_candidates', return_value=[grasp, grasp]), \
             patch.object(planner, 'refine_grasps', side_effect=lambda g: iter([g])), \
             patch.object(planner, 'attempt', return_value=([dict()], {})) as attempt, \
             patch('bunny_movement.verify', side_effect=validations):
            _, result = planner.plan()
        self.assertEqual(attempt.call_count, 2)
        self.assertEqual(result['failures_before_success']['retimed_replay'], 1)
        self.assertEqual(planner.report['failed_attempts'][0]['detail']['errors'][0][0], 290)

    def test_cartesian_failure_identifies_the_motion_phase(self):
        planner = Planner(self.request, self.robot, self.body, self.directory)
        planner.policy = dict(planner.policy, cartesian_step_m=5.)
        q = np.radians(self.request['robot']['initial_joints_deg'])
        start = self.robot.frames(q)[0]
        impossible = start.copy()
        impossible[0, 3] += 2.
        with self.assertRaises(PlanningFailure) as caught:
            planner.line(q, start, impossible, 0., planner.a, None, phase='extract')
        self.assertEqual(str(caught.exception), 'cartesian_ik')
        self.assertEqual(caught.exception.detail['phase'], 'extract')

    def test_yaw_target_rotates_about_pan_not_base_origin(self):
        from prepare_station_request import yaw_target
        source = np.asarray(self.request['source']['world_from_object'])
        pan = (self.robot.base @ self.robot.origins[0])[:3, 3]
        target = yaw_target(source, self.robot, -25.)
        self.assertAlmostEqual(np.linalg.norm(source[:2, 3] - pan[:2]), np.linalg.norm(target[:2, 3] - pan[:2]), places=9)
        self.assertAlmostEqual(source[2, 3], target[2, 3], places=9)
        self.assertAlmostEqual(Rotation.from_matrix(target[:3, :3] @ source[:3, :3].T).magnitude(), np.radians(25), places=9)

    def test_station_profile_preserves_measurements_and_contact_radius(self):
        from prepare_station_request import prepare
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'request.json'
            prepare(HERE / 'example_request.json', output, HERE / 'station_search_profile.json')
            raw = json.loads(output.read_text())
            original = json.loads((HERE / 'example_request.json').read_text())
            self.assertEqual(raw['robot']['world_from_base'], original['robot']['world_from_base'])
            self.assertEqual(raw['robot']['hardware_zero_offsets_deg'], original['robot']['hardware_zero_offsets_deg'])
            self.assertEqual(raw['gripper']['contact_patch_radius_m'], original['gripper']['contact_patch_radius_m'])
            self.assertEqual(raw['target'], original['target'])
            self.assertEqual(raw['planner']['refine_multistart'], 12)
            self.assertEqual((output.parent / raw['object']['mesh']).resolve(), (HERE / original['object']['mesh']).resolve())


if __name__ == '__main__':
    unittest.main()
