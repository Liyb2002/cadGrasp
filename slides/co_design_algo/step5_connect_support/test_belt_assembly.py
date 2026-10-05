"""Discriminating geometry/trajectory checks for the one-piece belt baseline."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from types import SimpleNamespace
import numpy as np
import trimesh
from step2_local_support.surface import areas
from step5_connect_support import belt_geometry as B, rigid_path as P, whole_assembly as A
from step5_connect_support.fixtures import contacts


class BeltTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box([1., 1., 1.]); self.mesh.apply_translation([0, 0, 1.])
        self.scene = B.Scene(self.mesh)




    def test_real_detached_thin_material_is_not_removed(self):
        big = trimesh.creation.box([1., 1., 1.])
        thin = trimesh.creation.box([.1, .1, 1e-6]); thin.apply_translation([2., 0., 0.])
        joined, report = B.union_parts([big, thin], 1.)
        self.assertFalse(report['one_solid'])
        self.assertEqual(report['zero_thickness_surface_duplicates_removed'], [])

    def test_open_ring_retains_required_floor_hull(self):
        required = np.array([[-.7, -.7], [.7, -.7], [.7, .7], [-.7, .7]])
        value = B.open_ring(self.mesh, required, required, np.zeros(3), 0., 1.6, .6, self.scene)
        self.assertIsNotNone(value)
        parts, base = value
        check, _ = A.footprint(parts, np.zeros(3), required, 1.)
        self.assertTrue(check['passed'])
        self.assertLess(base['cut_offset_m'], np.max(np.asarray(base['outer_xy_m'])[:, 0]))
        self.assertTrue(all(self.scene.clear(p, ground=True) for p in parts))

    def test_local_form_closure_is_not_reported_as_a_trajectory(self):
        c = []
        for face, triangle in enumerate(self.mesh.triangles):
            center = triangle.mean(axis=0); patch = (center+.3*(triangle-center))[None]
            c.append(dict(source_faces=np.array([face]), triangles_m=patch))
        motions, report = P.local_motions(self.mesh, c, np.empty((0, 3)), self.mesh.center_mass, 1.)
        self.assertEqual(len(motions), 0)
        self.assertFalse(report['global_impossibility_claimed'])

    def test_clear_translation_is_found_and_continuously_replayed(self):
        piece = trimesh.creation.box([.1, .1, .1]); piece.apply_translation([-.6, 0, 1.])
        report = P.search(self.scene, contacts(self.mesh), [piece], self.mesh.center_mass)
        self.assertTrue(report['passed'], report)
        self.assertTrue(P.replay(self.scene, [piece], report))

    def test_rotational_midpath_collision_is_rejected(self):
        obstacle = trimesh.creation.box([.2, .2, .2]); obstacle.apply_translation([0., 1., .5])
        scene = B.Scene(obstacle)
        piece = trimesh.creation.box([.1, .1, .1]); piece.apply_translation([1., 0., .5])
        origin = np.array([0., 0., .5]); motion = np.array([0., 0., 0., 0., 0., np.pi])
        self.assertTrue(scene.clear(piece))
        end = piece.copy(); end.vertices = P.transform(piece.vertices, origin, scene.scale, motion, 1.)
        self.assertTrue(scene.clear(end))
        budget = dict(remaining=100, checked=0)
        self.assertFalse(P.segment(scene, [piece], origin, motion, 0., 1., budget))

    def test_clear_rotation_is_checked_as_a_continuous_arc(self):
        obstacle = trimesh.creation.box([.2, .2, .2]); obstacle.apply_translation([0., 0., .5])
        scene = B.Scene(obstacle)
        piece = trimesh.creation.box([.1, .1, .1]); piece.apply_translation([1., 0., .5])
        budget = dict(remaining=100, checked=0)
        self.assertTrue(P.segment(scene, [piece], np.array([0., 0., .5]),
            np.array([0., 0., 0., 0., 0., np.pi]), 0., 1., budget))
        self.assertGreater(budget['checked'], 1)


    def test_rotation_padding_preserves_tangent_contact_plane(self):
        piece=trimesh.creation.box([.1, .1, .1]);piece.apply_translation([.55, 0, 1])
        budget=dict(remaining=200,checked=0)
        self.assertTrue(P.segment(self.scene,[piece],np.array([0.,0.,1.]),
            np.array([0.,0.,0.,np.pi/3,0.,0.]),0.,1.,budget))

    def test_yaw_at_floor_does_not_invent_downward_motion(self):
        piece=trimesh.creation.box([.1, .1, .1]);piece.apply_translation([1., 0, .05])
        budget=dict(remaining=200,checked=0)
        self.assertTrue(P.segment(self.scene,[piece],np.zeros(3),
            np.array([0.,0.,0.,0.,0.,.5]),0.,1.,budget))

    def test_analytic_floor_minimum_contains_intermediate_rotated_vertices(self):
        rng=np.random.default_rng(710);points=rng.normal(size=(30,3))*.2+[0,0,1]
        origin=np.array([.1,.2,.3]);motion=np.array(np.asarray([.1,.2,-.15,.8,-.3,.6]))
        _,low=P.rotation_enclosure(points,origin,motion,1.,.1,.8)
        sampled=min(P.transform(points,origin,1.,motion,t)[:,2].min() for t in np.linspace(.1,.8,1001))
        self.assertLessEqual(low,sampled+1e-13)
        self.assertLess(sampled-low,1e-6)

    def test_piecewise_detour_replays_and_direct_shortcut_collides(self):
        from step5_connect_support import piecewise_path as PP
        piece=trimesh.creation.box([.1, .1, .1]);piece.apply_translation([-.7, 0., 1.])
        poses=[]
        for shift in ([0,0,0],[0,1,0],[1.4,1,0],[1.4,0,0]):
            pose=np.eye(4);pose[:3,3]=shift;poses.append(pose)
        report=dict(passed=True,continuous_sweep_verified=True,origin_m=[0.,0.,1.],poses=poses)
        self.assertTrue(PP.replay(self.scene,[piece],report))
        self.assertFalse(PP.edge(self.scene,[piece],np.array([0.,0.,1.]),poses[0],poses[-1])[0])

    def test_failed_trajectory_cannot_produce_success_video(self):
        from step5_connect_support import video
        with self.assertRaises(ValueError):video.render(None,None,dict(passed=False),None)

    def test_original_object_floor_point_has_the_same_friction_cone(self):
        from step5_connect_support import belt_assembly as BA
        domain=SimpleNamespace(mesh=self.mesh)
        points,normals,owners=BA.bearing_rays(domain,contacts(self.mesh),np.zeros(3),4.)
        pivot=owners<0
        self.assertEqual(int(pivot.sum()),4)
        np.testing.assert_array_equal(points[pivot],np.zeros((4,3)))
        np.testing.assert_array_equal(normals[pivot],[[4,0,1],[-4,0,1],[0,4,1],[0,-4,1]])
        self.assertTrue(np.all(np.linalg.norm(COORD.floor(normals[pivot]),axis=1)<=4*normals[pivot,2]))

    def test_current_shared_body_reactions_replay_with_friction_at_both_feet(self):
        from step5_connect_support import belt_assembly as BA
        from step4_floor_contact import equilibrium as Q
        from step4_floor_contact.audit import replay
        mesh=trimesh.creation.box([.2, .2, .2]);mesh.apply_translation([0, 0, .5])
        heads=[dict(triangles_m=(t.mean(axis=0)+.3*(t-t.mean(axis=0)))[None],source_faces=np.array([i]))
               for i,t in enumerate(mesh.triangles)]
        domain=SimpleNamespace(mesh=mesh,com=mesh.center_mass)
        points,normals,_=Q.contact_rays(domain,heads,np.zeros(3))
        weights=np.random.default_rng(774).uniform(.01,.02,(12,len(points)))
        weights[:,normals[:,2]>0]+=.1
        loads=weights@Q.wrench(points,normals,domain.com)
        floor=dict(original_pivot_m=np.zeros(3),load_wrenches=loads,continuous_outer_load_wrenches=loads)
        report,arrays=BA.bearing(domain,heads,floor,dict(pads_xy_m=[[[-1,-1],[1,-1],[1,1],[-1,1]]]))
        self.assertTrue(report['continuous_passed'],report)
        self.assertTrue(np.all(arrays['contact_owners'][:4]==-1))
        self.assertTrue(np.all(arrays['contact_owners'][4:]==0))
        for prefix in ('sample','continuous'):
            checked=replay(arrays,prefix,loads,1,report['sufficient_friction_coefficient'])
            self.assertLess(checked['maximum_body_equilibrium_residual_conditioned'],1e-10)


if __name__ == '__main__': unittest.main()
