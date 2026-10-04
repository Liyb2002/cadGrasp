"""Reject incompatible exits even when all heads pass individually."""
import unittest
from step3_scheculer.initialize_cached_v18 import compatibility


class CacheCompatibilityTests(unittest.TestCase):
    def test_individually_valid_different_rays_are_not_a_joint_exit(self):
        heads=[dict(valid=True,directions=[[0]],path_components=[1]),
               dict(valid=True,directions=[[1]],path_components=[1])]
        directions,ports=compatibility(heads,dict(vectors=[[1,0,0],[0,1,0]]))
        self.assertFalse(directions)
        self.assertEqual(ports,{1})

    def test_common_ray_does_not_connect_disjoint_roadmap_ports(self):
        heads=[dict(valid=True,directions=[[0]],path_components=[1]),
               dict(valid=True,directions=[[0]],path_components=[2])]
        directions,ports=compatibility(heads,dict(vectors=[[1,0,0]]))
        self.assertEqual(directions,{0})
        self.assertFalse(ports)

    def test_intersection_retains_only_common_certificates(self):
        heads=[dict(valid=True,directions=[[0,1]],path_components=[1,2]),
               dict(valid=True,directions=[[1,2]],path_components=[2,3])]
        self.assertEqual(compatibility(heads,dict(vectors=[0,1,2])),({1},{2}))


if __name__=='__main__':unittest.main()
