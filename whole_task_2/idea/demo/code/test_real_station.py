"""Physical coordinate and TCP regressions for the separate real-station planner."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from whole_task_2.step5.video.so101 import Arm
from .real_arm import RealArm, SceneCheck, tcp_matrix
from .real_plan import load_station, placed_on_table, within_table, export, core_hashes, STATION


class RealStationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.station = load_station()
        cls.arm = RealArm(cls.station)

    def test_nominal_tcp_matches_existing_kinematics_without_changing_core(self):
        old = Arm(np.zeros(3), np.eye(3))
        for q in np.random.default_rng(13).uniform(old.lower, old.upper, (8, 5)):
            for before, after in zip(old.forward(q), self.arm.forward(q)):
                np.testing.assert_allclose(before, after, atol=1e-13)

    def test_calibrated_full_tcp_has_correct_position_direction_and_jacobians(self):
        s = deepcopy(self.station)
        tcp = tcp_matrix(s['feasibility']['tcp_transform'])
        correction = np.eye(4); correction[:3, :3] = Rotation.from_euler('xyz', [.02, -.03, .01]).as_matrix()
        correction[:3, 3] = [.003, -.005, .007]
        s['feasibility']['tcp_transform'] = (tcp @ correction).tolist()
        arm = RealArm(s); q = np.array([.2, -.5, .8, .4, .1]); p, d, jp, ja = arm.forward(q)
        eps = 1e-6
        for i in range(5):
            step = np.eye(5)[i]*eps; plus = arm.forward(q+step); minus = arm.forward(q-step)
            np.testing.assert_allclose((plus[0]-minus[0])/(2*eps), jp[:, i], atol=1e-8)
            np.testing.assert_allclose((plus[1]-minus[1])/(2*eps), ja[:, i], atol=1e-8)
        self.assertGreater(np.linalg.norm(p-self.arm.forward(q)[0]), .001)
        self.assertGreater(np.linalg.norm(d-self.arm.forward(q)[1]), .001)

    def test_tabletop_translation_keeps_fixture_seat_and_gravity_orientation(self):
        source = trimesh.creation.box(extents=[.03, .04, .05])
        support = trimesh.creation.box(extents=[.10, .10, .03])
        seat = np.eye(4); seat[2, 3] = .05
        P, W = placed_on_table(source, np.eye(3), [.20, 0], self.station, support, seat)
        np.testing.assert_allclose(np.linalg.inv(W) @ P, seat, atol=1e-14)
        self.assertAlmostEqual(trimesh.transform_points(support.vertices, W)[:, 2].min(), -.0024)
        self.assertTrue(within_table(source, P, self.station, support, W))
        bad = P.copy(); bad[0, 3] += .3
        self.assertFalse(within_table(source, bad, self.station, support, W))

    def test_tool_table_check_uses_measured_plane(self):
        source = trimesh.creation.box(extents=[.02, .02, .02])
        check = SceneCheck(source, self.arm, self.station)
        face = int(np.argmax(source.face_normals[:, 2])); axis = np.array([0., 0., -1.])
        P = np.eye(4); P[2, 3] = -.03
        self.assertFalse(check.tool_above_table(face, axis, P))
        P[2, 3] = .02
        self.assertTrue(check.tool_above_table(face, axis, P))

    def test_invalid_tcp_scale_or_speed_is_rejected(self):
        bad = np.eye(4); bad[0, 0] = 1.1
        with self.assertRaises(ValueError): tcp_matrix(bad)
        s = deepcopy(self.station); s['maximum_joint_speed_deg_s'] = 61
        import json
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'station.json'; path.write_text(json.dumps(s))
            with self.assertRaises(ValueError): load_station(path)

    def test_no_support_package_preserves_every_stop_in_mm_and_five_joint_angles(self):
        source = trimesh.creation.box(extents=[.03, .04, .05]); P, W = placed_on_table(source, np.eye(3), [.2, 0], self.station)
        task = dict(id='S1', original_faces=np.arange(len(source.faces)), faces=np.array([2, 4]),
                    axes=np.array([[0., 0., -1.], [0., 0., -1.]]), joints=np.zeros((2, 4, 5)),
                    pose=P, support_pose=W, seat=None, exit=np.array([0., 0., 1.]), exit_length=0., bounds=None)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            info = export(source, [task], None, dict(mode='test'), self.station, out, STATION, core_hashes())
            self.assertIsNone(info['support']); self.assertIsNone(info['tasks'][0]['support_from_object'])
            self.assertFalse((out/'support.stl').exists())
            self.assertEqual(info['station']['arm_base_offsets_mm'], [0.])
            self.assertEqual(info['station']['table_top_z_mm'], -2.4)
            with np.load(out/'tasks.npz') as z:
                self.assertEqual(z['S1_arm_joints_deg'].shape, (2, 4, 5))
                np.testing.assert_allclose(z['S1_stop_tip_world_mm'],
                    trimesh.transform_points(source.triangles_center[[2, 4]], P)*1e3)
                self.assertEqual(z['S1_world_from_support_mm'].size, 0)


if __name__ == '__main__': unittest.main()
