"""Invariants that distinguish a side installation from planar yaw."""
import unittest
import numpy as np
import trimesh
from subdivide_task.step3 import independent_geometry as G
from .state import Layout,transform
from .support_turn import support_turn
from .floors import planes,clip,connected_material
from .geometry import OperationModel


class SideTurnTests(unittest.TestCase):
    def test_boolean_remnant_clearance_preserves_other_actual_body_contacts(self):
        from .clearance import local_clearance
        body=G.box([0.,0.,0.],[.1,.1,.1])
        base=G.box([0.,0.,-.01],[.1,.1,0.])
        remnant=G.box([.04,.04,0.],[.045,.045,.001])
        current=G.union([base,remnant])
        actual=current-local_clearance(current,body,1e-5)
        retained_contact=G.box([0.,0.,-.001],[.01,.01,0.])
        self.assertLess(G.volume(actual^body),1e-15)
        self.assertLess(G.volume(retained_contact-actual),1e-15)

    def test_ribbed_casting_retains_the_contact_skin_and_real_floor_material(self):
        from types import SimpleNamespace
        from .ribbed_stock import world_stock
        dense=G.box([0.,0.,0.],[.1,.1,.2])
        # A needed contact layer may extend beyond its own old casting;
        # another task's stock could previously have supplied that material.
        skin=G.box([0.,0.,.12],[.12,.1,.2])
        model=SimpleNamespace(outer=G.mesh(skin),policy={'plate_thickness_m':.008})
        result=G.solid(world_stock(model,{'stock_world':G.mesh(dense)},np.eye(4),.035,.012))
        floor=G.box([0.,0.,0.],[.1,.1,.008])
        self.assertLess(G.volume(result),.75*G.volume(dense))
        self.assertLess(G.volume(skin-result),1e-15)
        self.assertLess(G.volume(floor-result),1e-15)
        self.assertEqual(len(result.decompose()),1)

    def test_reference_floor_follows_the_main_installation_if_task_zero_stands(self):
        from .construction import CompactConstructionModel
        q=Layout(np.tile(np.eye(4),(3,1,1)),np.tile(np.eye(4),(3,1,1)),np.tile([0.,0.,1.],(3,1)))
        q=support_turn(q,0,[1.,0.,0.],90.,[0.,0.,0.])
        reference=CompactConstructionModel.reference_material(q.material)
        np.testing.assert_allclose(reference[:3,2],[0.,0.,1.],atol=1e-14)

    def test_required_standing_installation_rejects_yaw_and_partial_tilt(self):
        from .installations import standing_pairs
        q=Layout(np.tile(np.eye(4),(2,1,1)),np.tile(np.eye(4),(2,1,1)),np.tile([0.,0.,1.],(2,1)))
        self.assertEqual(standing_pairs(support_turn(q,1,[0.,0.,1.],90.,[0.,0.,0.])),[])
        self.assertEqual(standing_pairs(support_turn(q,1,[1.,0.,0.],63.,[0.,0.,0.])),[])
        self.assertEqual(standing_pairs(support_turn(q,1,[1.,0.,0.],90.,[0.,0.,0.])),[(0,1)])

    def test_standing_installation_moves_every_vertex_of_the_same_body(self):
        q=Layout(np.tile(np.eye(4),(2,1,1)),np.tile(np.eye(4),(2,1,1)),np.tile([0.,0.,1.],(2,1)))
        # An asymmetric fixture has a wide base and a distinct tall seat.
        mesh=G.mesh(G.union([G.box([0.,0.,0.],[.3,.1,.02]),G.box([0.,0.,.02],[.04,.1,.2])]))
        q=support_turn(q,1,[0.,1.,0.],-90.,[0.,0.,0.])
        flat=transform(mesh.vertices,np.linalg.inv(q.material[0]))
        side=transform(mesh.vertices,np.linalg.inv(q.material[1]))
        np.testing.assert_allclose(np.ptp(flat,axis=0)[[2,1,0]],np.ptp(side,axis=0),atol=1e-14)
        np.testing.assert_allclose(transform(side,q.material[1]),mesh.vertices,atol=1e-14)
        self.assertEqual(len(flat),len(side))

    def test_needle_spin_matches_the_entire_seat_and_an_exact_side_installation(self):
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation
        from subdivide_task.step4.aligned_side import needle_angles,propose
        model=SimpleNamespace(source=SimpleNamespace(center_mass=np.array([.02,.03,.01])),
                              axis=np.array([0.,-1.,-1.])/np.sqrt(2),head=np.array([0.,0.,.22]))
        rng=np.random.default_rng(42)
        checked=0
        for _ in range(20):
            q=Layout(np.tile(np.eye(4),(2,1,1)),np.tile(np.eye(4),(2,1,1)),np.tile([0.,0.,1.],(2,1)))
            q.poses[:,:3,:3]=Rotation.random(2,random_state=rng).as_matrix()
            q.poses[:,:3,3]=rng.uniform(-.05,.05,(2,3))
            for angle in needle_angles(q,model.axis,0,1):
                p=propose(model,q,0,1,angle)
                np.testing.assert_allclose(p.material[0]@p.poses[0],p.material[1]@p.poses[1],atol=2e-13)
                self.assertAlmostEqual(p.material[0,:3,2]@p.material[1,:3,2],0.,places=12)
                for world in(model.head,model.head+.1*model.axis):
                    np.testing.assert_allclose(transform(world,np.linalg.inv(p.poses[0])),
                                               transform(world,np.linalg.inv(q.poses[0])),atol=1e-13)
                checked+=1
        self.assertGreater(checked,10)

    def test_side_turn_changes_gravity_in_material_but_not_working_pose(self):
        q=Layout(np.array([np.eye(4)]),np.array([np.eye(4)]),np.array([[0.,0.,1.]]))
        q.poses[0,:3,3]=[.05,-.03,.12]
        p=np.array([[0.,0.,0.],[.03,.02,.05]])
        r=support_turn(q,0,[1.,0.,0.],90.,[.05,0.,.12])
        np.testing.assert_allclose(r.poses,q.poses,atol=1e-14)
        np.testing.assert_allclose(r.exits,q.exits,atol=1e-14)
        p_material=transform(p,r.material[0]@r.poses[0])
        np.testing.assert_allclose(transform(p_material,np.linalg.inv(r.material[0])),transform(p,q.poses[0]),atol=1e-14)
        np.testing.assert_allclose(r.material[0,:3,2],[0.,-1.,0.],atol=1e-14)

    def test_every_installation_constrains_the_same_actual_solid(self):
        matrices=np.array([np.eye(4),np.eye(4)])
        q=Layout(matrices.copy(),matrices.copy(),np.array([[0.,0.,1.],[0.,0.,1.]]))
        q=support_turn(q,1,[1.,0.,0.],90.,[0.,0.,0.])
        solid=clip(G.box([-.1,-.1,-.1],[.1,.1,.1]),planes(q.material))
        mesh=G.mesh(solid)
        self.assertAlmostEqual(mesh.volume,.002,places=12)
        for M in q.material:
            self.assertGreaterEqual(transform(mesh.vertices,np.linalg.inv(M))[:,2].min(),-1e-14)

    def test_component_cleanup_has_no_assumed_global_floor(self):
        # A perfectly usable fixture may lie entirely above material z=0.
        main=G.box([0.,0.,.1],[.1,.1,.2])
        orphan=G.box([.3,0.,.1],[.31,.01,.11])
        actual=G.mesh(connected_material(G.union([main,orphan])))
        self.assertAlmostEqual(actual.volume,.001,places=12)
        self.assertGreater(actual.bounds[0,2],.09)

    def test_turning_one_cohort_preserves_all_its_relative_seats(self):
        q=Layout(np.tile(np.eye(4),(3,1,1)),np.tile(np.eye(4),(3,1,1)),np.tile([0.,0.,1.],(3,1)))
        q.material[1,:3,3]=[.2,.1,0.]
        relative=np.linalg.inv(q.material[0])@q.material[1]
        r=support_turn(q,[0,1],[0.,1.,0.],90.,[.1,.2,.3])
        np.testing.assert_allclose(np.linalg.inv(r.material[0])@r.material[1],relative,atol=1e-14)
        np.testing.assert_array_equal(r.material[2],q.material[2])

    def test_cached_exit_proof_is_invalidated_by_changed_geometry_or_path(self):
        m=OperationModel.__new__(OperationModel)
        m.tolerance=1e-10;m.path_cache={}
        support=G.box([-.05,-.05,-.05],[.05,.05,.05])
        body=G.mesh(G.box([.1,-.02,-.02],[.2,.02,.02]))
        up=[('test',body,np.array([0.,0.,1.]),.3)]
        rows,_=m.path_checks(support,up)
        self.assertTrue(rows[0]['passed'])
        self.assertEqual(len(m.path_cache),1)
        sideways=[('test',body,np.array([-1.,0.,0.]),.3)]
        rows,_=m.path_checks(support,sideways)
        self.assertFalse(rows[0]['passed'])
        overlapping=G.box([.12,-.01,-.01],[.15,.01,.01])
        rows,_=m.path_checks(overlapping,up)
        self.assertFalse(rows[0]['passed'])

    def test_one_task_can_leave_a_previous_cohort_without_losing_other_tasks(self):
        from subdivide_task.step4.quarter_turns import reassigned_groups
        groups=[[0,1,2],[3,4],[5,6]]
        actual=reassigned_groups(groups,[1],4)
        self.assertEqual(actual,[[0,2],[5,6],[1,3,4]])
        self.assertEqual(sorted(k for group in actual for k in group),list(range(7)))
        self.assertEqual(groups,[[0,1,2],[3,4],[5,6]])

    def test_side_alignment_optimizes_bearing_without_changing_world_pose(self):
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation
        from subdivide_task.step4.quarter_turns import propose
        model=SimpleNamespace(source=SimpleNamespace(center_mass=np.zeros(3)))
        rng=np.random.default_rng(29)
        for _ in range(6):
            q=Layout(np.tile(np.eye(4),(2,1,1)),np.tile(np.eye(4),(2,1,1)),np.tile([0.,0.,1.],(2,1)))
            q.poses[:,:3,:3]=Rotation.random(2,random_state=rng).as_matrix()
            q.poses[:,2,3]=.15
            side=propose(model,q,[0],0,1,'closest_side',90.,[0.,0.,0.])
            np.testing.assert_array_equal(side.poses,q.poses)
            self.assertAlmostEqual(side.material[0,:3,2]@side.material[1,:3,2],0.,places=12)
            target=q.material[1,:3,:3]@q.poses[1,:3,:3]
            error=np.linalg.norm(side.material[0,:3,:3]@q.poses[0,:3,:3]-target)
            for axis in(0,1):
                for degrees in(-90.,90.):
                    other=propose(model,q,[0],0,1,axis,degrees,[0.,0.,0.])
                    self.assertLessEqual(error,np.linalg.norm(other.material[0,:3,:3]@q.poses[0,:3,:3]-target)+1e-12)


if __name__=='__main__':
    unittest.main()
