"""Strict, boundary and impossible intersections must remain distinct."""
import unittest
import numpy as np
from codes.precompute_objects.common_directions import classify,build_counterexamples

class CommonDirectionTests(unittest.TestCase):
    def test_strict_shared_direction(self):
        normals=np.eye(3);r=classify(normals)
        self.assertEqual(r['status'],'strict_common_direction')
        self.assertGreater(np.min(normals@r['direction']),0.)
    def test_opposite_normals_still_have_boundary_directions(self):
        normals=np.array([[0.,0.,1.],[0.,0.,-1.]])
        r=classify(normals)
        self.assertEqual(r['status'],'boundary_only_common_direction')
        self.assertGreater(np.linalg.norm(r['direction']),0.)
        np.testing.assert_allclose(normals@r['direction'],0.,atol=1e-9)
    def test_positive_span_proves_no_nonzero_direction(self):
        normals=np.vstack([np.eye(3),-np.ones(3)/np.sqrt(3)])
        r=classify(normals);c=r['certificate']
        self.assertFalse(r['common_direction_exists']);self.assertEqual(c['normal_rank'],3)
        self.assertGreater(min(c['weights']),0.)
        np.testing.assert_allclose(np.array(c['weights'])@normals,0.,atol=1e-9)
    def test_six_axis_case_requires_more_than_four_pose_core(self):
        normals=np.vstack([np.eye(3),-np.eye(3)])
        sets,info=build_counterexamples(normals,np.zeros((6,6),int),per_size=1)
        self.assertEqual(len(sets),1);self.assertEqual(sets[0]['size'],6)
        self.assertFalse(sets[0]['common_direction']['common_direction_exists'])
    def test_subsets_cannot_be_counterexamples_if_all_share_direction(self):
        sets,info=build_counterexamples(np.eye(3),np.zeros((3,3),int))
        self.assertEqual(sets,[]);self.assertEqual(info['status'],'impossible_from_saved_poses')

if __name__=='__main__':unittest.main()
