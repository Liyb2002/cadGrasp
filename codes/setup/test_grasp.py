"""Physical calibration: a grasp must carry a free box and reject an empty hand."""
import unittest
from unittest.mock import patch
import numpy as np
import mujoco

from grasp import HERE, build, simulate, geometry_check


BOX = '''<mujoco><asset/><worldbody><body name="obj"><freejoint/>
<inertial mass=".3" pos="0 0 0" diaginertia=".0001 .0001 .0001"/>
<geom name="box" type="box" size=".03 .02 .025" group="3"/>
</body></worldbody></mujoco>'''


class GraspPhysics(unittest.TestCase):
    def test_thin_rim_geometry_never_crosses_joint_limits(self):
        model=build('B',BOX,hand_xml=HERE/'assets/parallel_jaw.xml')
        initial=np.eye(4);initial[2,3]=.025
        hand=np.eye(4);hand[:3,:3]=np.diag([-1.,1.,-1.]);hand[:3,3]=[.3,0.,.3]
        joints=[model.joint(f'finger_joint{i}') for i in (1,2)]
        forward=mujoco.mj_forward
        for width in (.001,.002,.003):
            observed=[]
            def inspect(model,data):
                observed.append([float(data.qpos[j.qposadr[0]]) for j in joints])
                return forward(model,data)
            with self.subTest(width=width),patch('grasp.mujoco.mj_forward',side_effect=inspect):
                self.assertTrue(geometry_check(model,initial,dict(hand=hand,width=width,opening=width/2+.01)))
            positions=np.asarray(observed)
            self.assertEqual(len(positions),30)
            for i,joint in enumerate(joints):
                low,high=model.jnt_range[joint.id]
                self.assertTrue(np.all(positions[:,i]>=low))
                self.assertTrue(np.all(positions[:,i]<=high))
                self.assertAlmostEqual(positions[-1,i],low)

    def test_generic_jaws_lift_and_turn_free_box(self):
        model = build('B', BOX, hand_xml=HERE/'assets/parallel_jaw.xml')
        initial = np.eye(4); initial[2,3] = .025
        hand = np.eye(4); hand[:3,:3] = np.diag([-1., 1., -1.])
        # Only the distal 40 mm of the straight fingers overlaps the box.
        hand[2,3] = .025 + .11
        candidate = dict(hand=hand, width=.04)
        self.assertTrue(geometry_check(model, initial, candidate))
        result, _ = simulate(model, initial, candidate, turn_axis=[0.,1.,0.])
        self.assertTrue(result['passed'], result)

    def setup_case(self):
        model = build('B', BOX)
        initial = np.eye(4); initial[2,3] = .025
        hand = np.eye(4); hand[:3,:3] = np.diag([-1., 1., -1.])
        hand[2,3] = .025 + .1029
        return model, initial, dict(hand=hand, width=.04)

    def test_box_lift_and_no_object_weld(self):
        model, initial, candidate = self.setup_case()
        obj = model.body('object').id
        for i in range(model.neq):
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                self.assertNotIn(obj, (model.eq_obj1id[i], model.eq_obj2id[i]))
        result, _ = simulate(model, initial, candidate)
        self.assertTrue(result['passed'], result)
        self.assertGreater(result['object_vertical_displacement_m'], .075)

    def test_missed_grasp_fails(self):
        model, initial, candidate = self.setup_case()
        candidate['hand'][0,3] = .15
        result, _ = simulate(model, initial, candidate)
        self.assertFalse(result['passed'], result)

    def test_box_turns_without_object_weld(self):
        model, initial, candidate = self.setup_case()
        result, _ = simulate(model, initial, candidate, turn_axis=[1.,0.,0.])
        self.assertTrue(result['passed'], result)
        self.assertEqual([c['angle_deg'] for c in result['turn_checkpoints']], [40,50,60])

    def test_contour_fingers_connected_disjoint_and_short(self):
        from imprint import prototype
        _, _, _, fingers = prototype('side')
        for boxes in fingers.values():
            low = np.array([np.array(c)-s for c,s in boxes])
            high = np.array([np.array(c)+s for c,s in boxes])
            self.assertGreaterEqual(low[:,1].min(), -1e-12)
            self.assertLessEqual(high[:,2].max(), .054 + 1e-12)
            overlaps = np.all((low[:,None] <= high[None]+1e-9) &
                              (low[None] <= high[:,None]+1e-9), axis=2)
            connected, frontier = {0}, [0]
            while frontier:
                for other in np.flatnonzero(overlaps[frontier.pop()]):
                    if other not in connected:
                        connected.add(other); frontier.append(other)
            self.assertEqual(len(connected), len(boxes))


if __name__ == '__main__':
    unittest.main()
