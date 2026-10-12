from pathlib import Path
import sys
import unittest
import numpy as np
from collections import OrderedDict
from types import SimpleNamespace
import trimesh

COOPT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COOPT/'helper_func'), str(COOPT/'step4.2')]
import _bootstrap
from base_oracle import BaseOracle, clip, cover, change_planes, foot_obstacle_offsets, slice_polygon, contact_probe_points
from system_floor import system_floor_demands
from pose_oracle import PoseOracleConstraints, inside_union, transform_points
from whole_search.model import Layout


class OracleGeometry(unittest.TestCase):
    def test_legal_submicron_contact_probe_never_crosses_an_installed_floor(self):
        p=np.array([[0.,0.,9.2e-7],[0.,0.,1e-9]])
        n=np.array([[0.,0.,-1.],[0.,0.,-1.]])
        floors=np.array([[0.,0.,1.,0.]])
        anchors=contact_probe_points(p,n,floors,1e-6)
        self.assertTrue(np.all(anchors[:,2]>0.))
        np.testing.assert_allclose(anchors[:,2],p[:,2]/2,atol=1e-18)
        np.testing.assert_array_equal(p[:,2],[9.2e-7,1e-9])

    def test_probe_uses_all_final_floors_and_does_not_inflate_tangent_contacts(self):
        p=np.array([[1e-9,0.,0.]])
        n=np.array([[-1.,0.,0.]])
        floors=np.array([[0.,0.,1.,0.],[1.,0.,0.,0.]])
        a=contact_probe_points(p,n,floors,1e-6)
        np.testing.assert_allclose(a,[[.5e-9,0.,0.]],atol=1e-18)

    def test_zero_margin_default_and_arbitrarily_thin_slice(self):
        model=SimpleNamespace(extent=.1,tasks=[],poses=[],inputs=[],work_rays=[],
                              mesh=trimesh.creation.box([.1]*3))
        self.assertEqual(BaseOracle(model).foot_width,0.)
        thin=np.array([[0.,0.],[1.,0.],[1.,1e-7],[0.,1e-7]])
        shape=slice_polygon(thin)
        self.assertTrue(shape.is_valid)
        self.assertAlmostEqual(shape.area,1e-7,places=15)
        np.testing.assert_array_equal(foot_obstacle_offsets(np.array([[0.,1.,0.,-1e-7]]),0.),[-1e-7])

    def test_coincident_slice_orders_existing_vertices_without_buffer(self):
        points=np.array([[0.,0.],[1.,0.],[1.,1.],[1.,1.-1e-14],[0.,1.]])
        shape=slice_polygon(points)
        self.assertTrue(shape.is_valid)
        self.assertEqual(shape.area,1.)
        np.testing.assert_array_equal(shape.bounds,[0.,0.,1.,1.])

    def test_tolerant_clip_never_extrapolates_past_edge(self):
        points=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        cut=clip(points,np.array([-1.,0.]),-5e-13)
        self.assertTrue(np.all(cut>=0.))
        self.assertTrue(np.all(cut<=1.))

    def test_foot_clearance_is_per_face_and_tessellation_independent(self):
        planes=np.array([[0.,0.,-1.,.5], [1.,0.,0.,-.1]])
        offsets=foot_obstacle_offsets(planes,.005)
        np.testing.assert_allclose(offsets,[.495,-.1025])
        repeated=np.repeat(planes,100,axis=0)
        np.testing.assert_allclose(foot_obstacle_offsets(repeated,.005),np.repeat(offsets,100))
        # Ground under an object 0.5m high cannot overlap its bottom plane.
        self.assertGreater(offsets[0],0.)

    def test_exit_envelope_reaches_far_query_stock(self):
        oracle=object.__new__(BaseOracle)
        oracle.family_cache=OrderedDict()
        oracle.constraint_cache=OrderedDict()
        oracle.model=SimpleNamespace(length=.5,mesh=trimesh.creation.box([.1,.1,.1]),
            native=np.eye(4)[None],poses=['pose_1'],
            work_rays=[SimpleNamespace(planes=[],counts=[],face_ids=[],half_angle_deg=30)])
        layout=Layout(np.eye(4)[None],np.array([[0.,0.,1.]]),np.array([0]),(0,))
        point=np.array([0.,0.,1.8])
        short=oracle.families(layout)[0]
        self.assertGreater((short[:,:3]@point+short[:,3]).max(),0.)
        stock=np.array([[-.1,-.1,-.1],[.1,.1,2.]])
        full=oracle.families(layout,bounds=stock)[0]
        self.assertLessEqual((full[:,:3]@point+full[:,3]).max(),0.)

    def test_open_feet_enclose_without_material_rim(self):
        feet = np.array([[-2., -1.], [2., -1.], [0., 2.]])
        required = np.array([[-.2, -.2], [.2, -.2], [0., .2]])
        self.assertTrue(cover(required, feet)[0])

    def test_missing_extreme_is_rejected(self):
        required = np.array([[-2., 0.], [1., -1.], [1., 1.]])
        feet = np.array([[0., -2.], [2., 0.], [0., 2.]])
        passed, gap, _ = cover(required, feet)
        self.assertFalse(passed); self.assertGreater(gap, 0.)

    def test_nearly_collinear_contacts_are_not_jittered_into_area(self):
        contacts=np.array([[.8,.96],[.795,.975],[.79,.99]])
        required=np.array([[.7,.97],[.9,.97],[.8,1.1]])
        passed,gap,_=cover(required,contacts)
        self.assertFalse(passed);self.assertGreater(gap,.01)

    def test_final_state_floor_halfspace_and_contact_equality(self):
        rectangle = np.array([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]])
        legal = clip(rectangle, np.array([1., 0.]), .5)
        self.assertAlmostEqual(legal[:, 0].min(), -.5)
        # Pose1 uses z=0; pose2 has height x+.5. No intermediate rotation.
        self.assertTrue(np.all(legal[:, 0]+.5 >= -1e-12))

    def test_plane_coordinate_change(self):
        planes = np.array([[1., 0., 0., -2.]])
        transform = np.eye(4); transform[0, 3] = .7
        point = np.array([.3, 1., 0.])
        converted = change_planes(planes, transform)
        self.assertAlmostEqual(float(converted[0,:3]@point+converted[0,3]), -1.)

    def test_z_changes_each_load_ground_target_differently(self):
        loads = np.array([[.2, 0., 1., 0., 0., 0.], [-.3, .1, 1., 0., 0., 0.]])
        before = system_floor_demands(loads, np.array([0., 0., .1]))
        after = system_floor_demands(loads, np.array([.02, -.01, .15]))
        expected = np.array([.02, -.01])-.05*loads[:,:2]/loads[:,2,None]
        np.testing.assert_allclose(after['floor_demands_xy_m']-before['floor_demands_xy_m'], expected)

    def constraints(self):
        # Task A's exclusion is local x>=0; B's is also x>=0 in its own
        # object, but B is seated 2m to the left in the common fixture.
        work=SimpleNamespace(planes=np.array([[[-1.,0,0,0]]]),counts=np.array([1]),
            face_ids=np.array([0]),half_angle_deg=30.)
        native=np.repeat(np.eye(4)[None],2,axis=0)
        native[1,:3,:3]=[[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
        native[1,2,3]=1.
        Q=np.repeat(np.eye(4)[None],2,axis=0);Q[1,0,3]=-2.
        model=SimpleNamespace(native=native,poses=['A','B'],work_rays=[work,work])
        layout=Layout(Q,np.array([[0.,0.,1.],[1.,0.,0.]]),np.array([0,1]),(0,1))
        return PoseOracleConstraints(model,layout)

    def test_current_pose_cone_does_not_include_other_task(self):
        constraints=self.constraints();points=np.array([[-1.,0.,.2]])
        masks=constraints.classify_world(0,points)
        self.assertFalse(masks['current_pose_work'][0])
        self.assertTrue(masks['shared_material_work'][0])
        self.assertFalse(masks['global_floor'][0])

    def test_own_world_cone_and_common_cone_agree_after_state_change(self):
        constraints=self.constraints();q=np.array([[-3.,0.,.2],[-1.,0.,.2],[.2,0.,.2]])
        for task in constraints.poses:
            points=transform_points(q,task.T_world_fixture)
            np.testing.assert_array_equal(inside_union(points,task.work_world),inside_union(q,task.work_common))

    def test_other_final_floor_excludes_material_without_intermediate_rotations(self):
        constraints=self.constraints();masks=constraints.classify_world(0,np.array([[1.2,0.,.2],[.8,0.,.2]]))
        np.testing.assert_array_equal(masks['global_floor'],[True,False])
        self.assertFalse(constraints.definition()['intermediate_transfer_states_constrained'])

    def test_object_z_updates_own_frame_floor_while_fixed_fixture_floor_stays_put(self):
        work=SimpleNamespace(planes=[],counts=[],face_ids=[],half_angle_deg=30.)
        model=SimpleNamespace(native=np.eye(4)[None],poses=['A'],work_rays=[work])
        layout=Layout(np.eye(4)[None],np.array([[0.,0.,1.]]),np.array([0]),(0,))
        before=PoseOracleConstraints(model,layout)
        layout.placements[0,2,3]=.05
        after=PoseOracleConstraints(model,layout)
        self.assertAlmostEqual(after.floor_planes_object[0,3]-before.floor_planes_object[0,3],.05)
        np.testing.assert_allclose(after.floor_planes_common,before.floor_planes_common,atol=1e-12,rtol=0)

    def test_other_fixture_state_z_moves_its_floor_constraint(self):
        work=SimpleNamespace(planes=[],counts=[],face_ids=[],half_angle_deg=30.)
        native=np.repeat(np.eye(4)[None],2,axis=0)
        model=SimpleNamespace(native=native,poses=['A','B'],work_rays=[work,work])
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([[0.,0.,1.],[0.,0.,1.]]),np.array([0,1]),(0,1))
        native[1,2,3]=-.10
        before=PoseOracleConstraints(model,layout)
        self.assertTrue(before.classify_world(0,np.array([[0.,0.,.05]]))['global_floor'][0])
        native[1,2,3]=.10
        after=PoseOracleConstraints(model,layout)
        self.assertFalse(after.classify_world(0,np.array([[0.,0.,.05]]))['global_floor'][0])

    def test_support_contact_margin_does_not_raise_physical_base_floor(self):
        work=SimpleNamespace(planes=[],counts=[],face_ids=[],half_angle_deg=30.)
        native=np.repeat(np.eye(4)[None],2,axis=0)
        native[1,:3,:3]=[[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]]
        model=SimpleNamespace(native=native,poses=['A','B'],work_rays=[work,work],support_floor_clearance_m=.005)
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([[0.,0.,1.],[1.,0.,0.]]),np.array([0,1]),(0,1))
        constraints=PoseOracleConstraints(model,layout)
        points=np.array([[.01,0.,0.],[.01,0.,.004],[.002,0.,.02],[.005,0.,.005]])
        masks=constraints.classify_world(0,points)
        np.testing.assert_array_equal(masks['global_floor'],[False]*4)
        np.testing.assert_array_equal(masks['support_contact_floor'],[True,True,True,False])
        self.assertEqual(constraints.definition()['support_contact_clearance_m'],.005)


if __name__ == '__main__': unittest.main()
