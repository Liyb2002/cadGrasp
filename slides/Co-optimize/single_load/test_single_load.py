import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
import trimesh,mujoco
from robot import Panda
from retention import check
from co_common import S

class PhysicalChecks(unittest.TestCase):
    def test_franka_ik_reconstructs_reachable_pose(self):
        robot=Panda();robot.data.qpos[:7]=robot.home
        mujoco.mj_forward(robot.model,robot.data)
        T=np.eye(4);T[:3,3]=robot.data.xpos[robot.hand];T[:3,:3]=robot.data.xmat[robot.hand].reshape(3,3)
        solution=robot.ik(T,robot.home+np.array([.1,0,0,0,0,0,0]))
        self.assertIsNotNone(solution)
        self.assertTrue(np.all((solution>=robot.lower)&(solution<=robot.upper)))
        robot.data.qpos[:7]=solution;mujoco.mj_forward(robot.model,robot.data)
        self.assertLess(np.linalg.norm(robot.data.xpos[robot.hand]-T[:3,3]),.0005)

    def test_ik_rejects_out_of_reach(self):
        robot=Panda();T=np.eye(4);T[:3,3]=[3,0,0]
        self.assertIsNone(robot.ik(T))

    def test_entire_fingertip_work_contact_is_checked(self):
        robot=Panda();hand=np.eye(4)
        pad=next(solid for _,is_pad,solid in robot.hand_solids(hand,.05-.0004) if is_pad)
        box=S.unpack(pad);triangle=box.triangles[0].copy()
        work=trimesh.Trimesh(triangle,[[0,1,2]],process=False)
        self.assertFalse(robot.work_contact_clear(hand,.05,work))
        work.apply_translation([2,0,0])
        self.assertTrue(robot.work_contact_clear(hand,.05,work))

    def test_contact_gap_is_not_finger_joint_separation(self):
        robot=Panda();parts=robot.hand_parts(np.eye(4),.05)
        pads=[p for p in parts if p['pad']]
        left=max([p for p in pads if p['body']=='left_finger'],key=lambda p:p['mesh'].volume)
        right=max([p for p in pads if p['body']=='right_finger'],key=lambda p:p['mesh'].volume)
        self.assertAlmostEqual(left['mesh'].bounds[0,1],.025,places=7)
        self.assertAlmostEqual(right['mesh'].bounds[1,1],-.025,places=7)

    def test_bolted_base_does_not_reject_all_robot_configurations(self):
        from co_common import F
        robot=Panda()
        self.assertTrue(robot.arm_clear(robot.home,.05,F.md.Manifold()))

    def test_asymmetric_contact_recenters_before_symmetric_closure(self):
        robot=Panda();mesh=trimesh.creation.box([.012,.022,.020]);mesh.apply_translation([0,.005,.1029])
        hand=np.eye(4);result=robot.center_on_first_contacts(hand,S.solid(mesh))
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['center_shift_m'],.005,places=5)
        self.assertTrue(robot.object_contacts(result['hand'],result['width'],S.solid(mesh))['passed'])
        np.testing.assert_array_equal(hand,np.eye(4))

    def test_retention_rejects_inverted_open_tray(self):
        mesh=trimesh.creation.box([.04,.04,.04]);fixture=trimesh.creation.box([.06,.06,.01]);fixture.apply_translation([0,0,-.025])
        ids=np.flatnonzero(mesh.face_normals[:,2]<-.99)
        points=np.array([[-.03,0,-.025],[.03,0,-.025]])
        normals=np.array([[1.,0,0],[-1.,0,0]])
        upright=check(mesh,fixture,mesh.triangles[ids],ids,points,normals,[np.eye(3)])
        flipped=check(mesh,fixture,mesh.triangles[ids],ids,points,normals,[np.diag([1.,-1.,-1.])])
        self.assertTrue(upright['passed']);self.assertFalse(flipped['passed'])

if __name__=='__main__':unittest.main()
