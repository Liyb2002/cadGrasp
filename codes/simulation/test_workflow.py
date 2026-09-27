"""Check sequence boundaries and geometry independently of the renderer."""
import unittest

import numpy as np
import mujoco
import trimesh

from scene import build
from workflow import Workflow, TEST_START
from kuka import Arm, TOOL


class WorkflowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = build()
        cls.workflow = Workflow(cls.model)
        cls.workflow.prepare_arms(fps=15)

    def test_grounded_roll_never_lifts_and_finishes_at_exact_target(self):
        w = self.workflow
        initial = trimesh.transform_points(w.object.vertices, w.object_pose(0))
        self.assertAlmostEqual(initial[:, 2].min(), 0., places=10)
        self.assertLess(np.ptp(initial[:, 2]), .7 * w.object.extents[2])
        for t in np.linspace(0, 9, 181):
            points = trimesh.transform_points(w.object.vertices, w.object_pose(t))
            self.assertLess(abs(points[:, 2].min()), 1e-9)
        np.testing.assert_array_equal(w.object_pose(9), np.eye(4))
        self.assertFalse(w.state(8.999)['working_area_visible'])
        self.assertTrue(w.state(9)['working_area_visible'])

    def test_fixture_keeps_saved_path_and_a_holds_until_b_is_clear(self):
        w = self.workflow
        direction = np.asarray(w.path['motion'][:3])
        distance = w.path['length_scale_m'] * w.path['final_withdrawal_amount']
        amounts = []
        for t in np.linspace(11, 16, 101):
            state = w.state(t)
            self.assertTrue(state['arm_a_holding'])
            np.testing.assert_array_equal(state['object_pose'], np.eye(4))
            position = state['support_translation']
            np.testing.assert_allclose(np.cross(position, direction), 0., atol=1e-12)
            amounts.append(np.dot(position, direction))
        self.assertAlmostEqual(amounts[0], distance)
        self.assertAlmostEqual(amounts[-1], 0.)
        self.assertTrue(np.all(np.diff(amounts) <= 0))
        self.assertTrue(w.state(18.999)['arm_a_holding'])
        self.assertFalse(w.state(18.999)['arm_b_holding'])

    def test_real_robot_limits_and_no_robot_reactions_in_load_checks(self):
        w = self.workflow
        self.assertEqual(self.model.nv, 12)
        self.assertEqual(self.model.nbody, 3)
        self.assertIsNone(w.state(TEST_START - .001)['load_index'])
        for j, name in enumerate(('A','B')):
            arm = Arm(name)
            self.assertTrue(np.all(w.arm_q[:,j] >= arm.lower))
            self.assertTrue(np.all(w.arm_q[:,j] <= arm.upper))
            self.assertTrue(np.all(abs(np.diff(w.arm_q[:,j],axis=0))*w.arm_fps <= arm.velocity))
            for i in range(0,len(w.arm_times),5):
                state=w.state(w.arm_times[i]);key=name.lower()
                point,axis,_,_=arm.forward(w.arm_q[i,j])
                np.testing.assert_allclose(point,state[key],atol=1e-4)
                np.testing.assert_allclose(axis,-state['normal_'+key],atol=1e-3)
        for t in np.linspace(TEST_START,w.duration,50):
            state=w.state(t)
            self.assertFalse(state['arm_a_holding'])
            self.assertFalse(state['arm_b_holding'])
            np.testing.assert_array_equal(state['object_pose'],np.eye(4))
            np.testing.assert_array_equal(state['support_translation'],np.zeros(3))
        self.assertEqual(len(w.results),10)
        for result in w.results:
            self.assertEqual(result['status'],'equilibrium_feasible')
            self.assertLess(result['max_force_balance_residual_N'],1e-7)

    def test_contact_pivot_does_not_slide_and_continuous_floor_check_passes(self):
        roll=self.workflow.roll
        roll.verify_continuous_floor_contact()
        for segment in roll.segments:
            for angle in np.linspace(segment['low'],segment['high'],7):
                transform=roll.pose(1-angle/roll.angle)
                point=transform[:3,:3]@segment['pivot']+transform[:3,3]
                np.testing.assert_allclose(point,segment['contact'],atol=1e-9)

    def test_rendered_robot_frames_agree_with_upstream_kinematics(self):
        model,_=build(robots=True)
        self.assertEqual(model.nv,26)
        data=mujoco.MjData(model)
        for index in (0,60,100,200,330):
            for j,name in enumerate(('A','B')):
                addresses=[model.jnt_qposadr[model.joint(f'robot_{name}_joint_{i}').id] for i in range(1,8)]
                q=self.workflow.arm_q[index,j]
                data.qpos[addresses]=q
                mujoco.mj_forward(model,data)
                body=model.body(f'robot_{name}_link_7').id
                rotation=data.xmat[body].reshape(3,3)
                point=data.xpos[body]+rotation[:,2]*TOOL
                expected,axis,_,_=Arm(name).forward(q)
                np.testing.assert_allclose(point,expected,atol=1e-10)
                np.testing.assert_allclose(rotation[:,2],axis,atol=1e-10)


if __name__ == '__main__':
    unittest.main()
