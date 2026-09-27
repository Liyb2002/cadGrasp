"""Reject cosmetic pose/grasp changes that used to pass the sequence filter."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from regrasp_sequence import descriptor, differences, diverse_grasp, diverse_pose


class Diversity(unittest.TestCase):
    def setUp(self):
        self.rest=np.eye(4)
        self.candidate=dict(hand=np.eye(4),contacts=np.array([[0,-.02,0],[0,.02,0]]),width=.04)
        self.first=descriptor(self.candidate,self.rest)

    def test_jaw_label_swap_is_same_contact(self):
        swapped=descriptor(dict(self.candidate,contacts=self.candidate['contacts'][::-1]),self.rest)
        self.assertEqual(differences(self.first,swapped)[0],0.)
        self.assertFalse(diverse_grasp(swapped,[self.first],.1))

    def test_new_approach_at_same_spot_is_not_new_grasp(self):
        hand=np.eye(4);hand[:3,:3]=Rotation.from_euler('x',60,degrees=True).as_matrix()
        candidate=descriptor(dict(self.candidate,hand=hand),self.rest)
        self.assertFalse(diverse_grasp(candidate,[self.first],.1))

    def test_new_spot_with_same_approach_is_not_new_grasp(self):
        candidate=descriptor(dict(self.candidate,contacts=self.candidate['contacts']+[.02,0,0]),self.rest)
        self.assertFalse(diverse_grasp(candidate,[self.first],.1))

    def test_both_changes_pass(self):
        hand=np.eye(4);hand[:3,:3]=Rotation.from_euler('x',60,degrees=True).as_matrix()
        candidate=descriptor(dict(self.candidate,hand=hand,contacts=self.candidate['contacts']+[.02,0,0]),self.rest)
        self.assertTrue(diverse_grasp(candidate,[self.first],.1))

    def test_world_yaw_does_not_create_new_tilt(self):
        first=np.eye(4);first[:3,:3]=Rotation.from_euler('x',45,degrees=True).as_matrix()
        yawed=first.copy();yawed[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()@first[:3,:3]
        self.assertFalse(diverse_pose(yawed,[first],self.rest))


if __name__=='__main__':unittest.main()
