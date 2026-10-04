"""Yaw must preserve installed anchor, upward exit and shared floor geometry."""
import unittest
import numpy as np
from step3_scheculer.cavity_dsl import yaw_about,common_floor
from step3_scheculer.operation_dsl import State,ray

class CavityTests(unittest.TestCase):
    def test_yaw_keeps_the_contact_anchor_and_upward_opening(self):
        s=State(((),()),np.array([np.eye(3)]*2),np.array([[.1,.2,0.],[-.1,0.,0.]]),(ray([0,0,-1]),)*2)
        center=np.array([.03,-.02,.04]);p=yaw_about(s,0,90,center)
        np.testing.assert_allclose(center@p.bases[0]+p.offsets[0],center@s.bases[0]+s.offsets[0])
        np.testing.assert_allclose(np.array([0,0,1.])@p.bases[0],[0,0,1.])
        self.assertTrue(common_floor(p))
        self.assertAlmostEqual(p.offsets[0,2],0.)
        np.testing.assert_allclose(p.bases[0]@p.bases[0].T,np.eye(3),atol=1e-12)
        np.testing.assert_array_equal(s.bases[0],np.eye(3))
    def test_yaw_is_about_fixture_axis_with_existing_yaw(self):
        s=State(((),),np.eye(3)[None],np.zeros((1,3)),(ray([0,0,-1]),))
        a=yaw_about(s,0,35,np.zeros(3));b=yaw_about(a,0,-35,np.zeros(3))
        np.testing.assert_allclose(b.bases[0],np.eye(3),atol=1e-12)
if __name__=='__main__':unittest.main()
