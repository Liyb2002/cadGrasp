"""The selected region must be exactly the discrete ball/cone intersection."""
import unittest
import numpy as np
import trimesh
from .needle_geometry import geometric_mask,evaluate,workspace
from robots_config.constraints import load_config
from .common import ROOT


def sheets(*triangles):
    vertices=np.asarray(triangles,dtype=float).reshape(-1,3)
    return trimesh.Trimesh(vertices,np.arange(len(vertices)).reshape(-1,3),process=False)


def patch(distance,angle=0.):
    t=np.deg2rad(angle);center=distance*np.array([np.sin(t),0,np.cos(t)])
    return center+np.array([[-.00001,-.00001,0],[0,.00001,0],[.00001,-.00001,0]])


class NeedleTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'robots_config/04_kuka_lbr_med14_r820.json')
        self.ws=workspace(self.config)
        self.head=np.zeros(3);self.forward=np.array([0.,0.,1.])

    def test_inner_distances_and_old_angle_and_radius_margins_do_not_carve_holes(self):
        mesh=sheets(patch(.02),patch(.06),patch(.0995),patch(.09,24.5))
        self.assertTrue(geometric_mask(mesh,self.head,self.forward,self.ws).all())
        self.assertEqual(self.ws['half_angle_deg'],25)
        self.assertEqual(self.ws['maximum_reference_distance_m'],.1)

    def test_occluded_and_opposite_normal_sheets_are_all_included(self):
        front=patch(.04);rear=patch(.09)
        mesh=sheets(front,front[::-1],rear,rear[::-1])
        rows=evaluate(mesh,self.head,-self.forward,self.config,self.ws)
        self.assertEqual(len(rows),1)
        np.testing.assert_array_equal(rows[0]['face_ids'],np.arange(4))

    def test_disconnected_patches_stay_in_the_same_pose_region(self):
        mesh=sheets(patch(.08,15),patch(.08,-15))
        rows=evaluate(mesh,self.head,-self.forward,self.config,self.ws)
        self.assertEqual(len(rows),1)
        np.testing.assert_array_equal(rows[0]['face_ids'],[0,1])

    def test_outside_radius_angle_and_backward_points_are_excluded(self):
        mesh=sheets(patch(.1005),patch(.09,25.5),patch(-.05),patch(.09,24))
        np.testing.assert_array_equal(geometric_mask(mesh,self.head,self.forward,self.ws),[False,False,False,True])

    def test_radius_is_spherical_not_an_axial_depth_limit(self):
        # axial depth is below 100mm, but Euclidean distance exceeds 100mm.
        tri=np.array([[.02,0,.099],[.0201,.0001,.099],[.0199,.0001,.099]])
        self.assertFalse(geometric_mask(sheets(tri),self.head,self.forward,self.ws)[0])

    def test_exact_cone_and_sphere_boundary_is_included(self):
        angle=np.deg2rad(25);edge=np.array([.1*np.sin(angle),0,.1*np.cos(angle)])
        tri=np.array([edge,[0,0,.08],[0,.001,.085]])
        self.assertTrue(geometric_mask(sheets(tri),self.head,self.forward,self.ws)[0])

    def test_proposal_margin_does_not_change_surface_membership(self):
        mesh=sheets(patch(.099,24.5))
        a=geometric_mask(mesh,self.head,self.forward,workspace(self.config,0))
        b=geometric_mask(mesh,self.head,self.forward,workspace(self.config,.1))
        np.testing.assert_array_equal(a,b)
        self.assertTrue(a.all())


if __name__=='__main__':
    unittest.main()
