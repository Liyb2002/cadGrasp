"""Regression checks for collision repair of a real KUKA wrist posture."""
import unittest

import mujoco
import numpy as np

import grasp as G
import kuka_transfer as K
from kuka import Arm


class SceneRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=G.build('cuboid_baseline',hand_xml=G.HERE/'assets/parallel_jaw.xml',
                         render_arm=True,robot_collisions=True,floor_hull=True)
        cls.indices=[cls.model.joint(f'robot_A_joint_{i}').qposadr[0] for i in range(1,8)]

    def setUp(self):
        self.arm=Arm('A');self.arm.base[:]=[-.45,-.35,0.]
        # A real search failure: link_5 and link_7 overlap by about 0.8 mm.
        self.seed=np.array([1.03537215,1.45387847,-.29859246,-.47572941,
                            2.79263627,-2.05964343,2.40769026])
        reachable=self.seed.copy();reachable[5]+=.06
        position,rotation,_=K.flange(self.arm,reachable)
        self.target=np.eye(4);self.target[:3,:3]=rotation;self.target[:3,3]=position
        self.data=mujoco.MjData(self.model)
        rest=np.eye(4);rest[2,3]=.04109342841384651
        G.object_pose(self.model,self.data,rest)
        G.hand_pose(self.data,self.target,initialize=True)
        self.data.qpos[self.indices]=self.seed
        mujoco.mj_forward(self.model,self.data)

    def arm_contacts(self):
        return [float(c.dist) for c in self.data.contact
                if any(self.model.geom(int(g)).name.startswith('arm_collision_')
                       for g in (c.geom1,c.geom2))]

    def test_repairs_collision_without_moving_recorded_hand_or_object(self):
        self.assertLess(min(self.arm_contacts()),-.0002)
        fixed=np.ones(self.model.nq,dtype=bool);fixed[self.indices]=False
        before=self.data.qpos[fixed].copy()
        mocap_pos=self.data.mocap_pos.copy();mocap_quat=self.data.mocap_quat.copy()
        q=K.repair_scene_configuration(self.arm,self.target,self.seed,self.model,self.data,self.indices,
                                      previous=self.seed,dt=.297)
        position,rotation,_=K.flange(self.arm,q)
        self.assertLess(np.linalg.norm(position-self.target[:3,3]),1e-4)
        self.assertLess(K.Rotation.from_matrix(rotation@self.target[:3,:3].T).magnitude(),1e-3)
        self.assertGreaterEqual(min(self.arm_contacts(),default=0.),-.0002)
        self.assertTrue(np.all(q>=self.arm.lower) and np.all(q<=self.arm.upper))
        self.assertTrue(np.all(np.abs(q-self.seed)<=self.arm.velocity*.297+1e-10))
        np.testing.assert_array_equal(before,self.data.qpos[fixed])
        np.testing.assert_array_equal(mocap_pos,self.data.mocap_pos)
        np.testing.assert_array_equal(mocap_quat,self.data.mocap_quat)

    def test_unreachable_tool_target_is_rejected(self):
        self.target[:3,3]=[5.,5.,5.]
        with self.assertRaisesRegex(ValueError,'repair failed'):
            K.repair_scene_configuration(self.arm,self.target,self.seed,self.model,self.data,self.indices)


if __name__=='__main__':
    unittest.main()
