"""The demo assumption must switch off before an unsupported regrasp interval."""
import unittest
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import grasp as G
from regrasp_sequence import RegraspTrial
from sequence import contacts
from test_grasp import BOX


class IdealGraspLifecycle(unittest.TestCase):
    def test_grasp_then_real_unheld_release(self):
        mesh=trimesh.creation.box([.06,.04,.05])
        model=G.build('B',BOX,hand_xml=G.HERE/'assets/parallel_jaw_ideal.xml')
        rest=np.eye(4);rest[2,3]=.025
        hand=np.eye(4);hand[:3,:3]=np.diag([-1.,1.,-1.]);hand[2,3]=.137
        candidate=dict(hand=hand,width=.04,opening=.08)
        trial=RegraspTrial(model,mesh,rest,candidate)
        self.assertFalse(trial.data.eq_active[trial.lock_id])
        trial.pickup(candidate,rest,first=True)
        self.assertTrue(trial.data.eq_active[trial.lock_id])
        self.assertEqual(contacts(model,trial.data)[0],2)
        relative=np.linalg.inv(G.transform(trial.data,trial.hand))@G.transform(trial.data,trial.obj)
        target=G.transform(trial.data,trial.hand)
        target[:3,:3]=Rotation.from_euler('z',35,degrees=True).as_matrix()@target[:3,:3]
        trial.hand_move(target,opened=False)
        moved=np.linalg.inv(G.transform(trial.data,trial.hand))@G.transform(trial.data,trial.obj)
        self.assertLess(np.linalg.norm(moved[:3,3]-relative[:3,3]),.001)
        self.assertLess(Rotation.from_matrix(moved[:3,:3]@relative[:3,:3].T).magnitude(),.01)
        trial.release(rest)
        self.assertFalse(trial.data.eq_active[trial.lock_id])
        count,force=contacts(model,trial.data)
        self.assertEqual(count,0);self.assertGreater(force,.001)
        self.assertEqual(len(trial.history),len(trial.constraint_history))
        self.assertIn(True,trial.constraint_history)
        self.assertFalse(trial.constraint_history[-1])


if __name__=='__main__':unittest.main()
