"""Geometric counterexamples for deletion, using the unchanged physical guards."""
from types import SimpleNamespace
import unittest
import numpy as np
import trimesh
from shapely.geometry import Polygon
from wrap_geometry import Validator, Deleter, box, solid, unpack, rim, enclosure, SCALE
from wrap_routes import simplify_path
from wrap_dimensions import (dimensions, physical_component_count,
                             INITIAL_FOOT_WIDTH_M, MINIMUM_WIDTH_M)
from wrap_width import ConnectionWidthGuard,feature_for_solid,link_frames,projection,foot_tracks
import manifold3d as md


class FullDemandStub:
    def evaluate_candidate(self, mesh):
        return dict(passed=True,counts={'0':32768},masks={0:np.ones(32768,bool)})


class Guards(unittest.TestCase):
    def setUp(self):
        self.row=dict(pose='pose_test',fixture_transform=np.eye(4),object_transform=np.eye(4),
            hull=np.array([[-.01,-.01,0],[.01,-.01,0],[.01,.01,0],[-.01,.01,0]]),
            object_ground=np.empty((0,2)))
        self.v=Validator.__new__(Validator)
        self.v.rows=[self.row]
        self.v.witness=box(np.array([-.1,-.1,0]),np.array([.1,.1,.1]))
        self.v.objects=md.Manifold()
        self.v.data={'body':trimesh.creation.box([.001]*3)}
        self.v.data['body'].apply_translation([0,0,.02])
        self.v.checker=FullDemandStub()

    def test_open_perimeter_keeps_ground_hull(self):
        band=rim(self.row)
        # Opening the middle of one edge keeps four outer corners and leaves
        # a single U-shaped body. This must not be rejected for lacking a ring.
        cut=box(np.array([-.004,-.03,-.001]),np.array([.004,-.009,.008]))
        opened=band-cut
        value,proof,checks=self.v.check(unpack(opened),band)
        self.assertTrue(checks['floor_passed'])
        self.assertEqual(checks['components'],1)
        self.assertTrue(proof['masks'][0].all())

    def test_loss_of_hull_rejected(self):
        too_small=box(np.array([-.005,-.005,0]),np.array([.005,.005,.005]))
        with self.assertRaisesRegex(ValueError,'Original solid guards'):
            self.v.check(unpack(too_small))

    def test_disconnected_stock_rejected(self):
        parts=rim(self.row)+box(np.array([.04,.04,0]),np.array([.045,.045,.005]))
        with self.assertRaisesRegex(ValueError,'Original solid guards'):
            self.v.check(unpack(parts))

    def test_tiny_loose_chip_cannot_hide_below_old_volume_cutoff(self):
        parts=rim(self.row)+box(np.array([.04,.04,0]),np.array([.04002,.04002,.00002]))
        self.assertEqual(physical_component_count(parts),2)
        with self.assertRaisesRegex(ValueError,'Original solid guards'):
            self.v.check(unpack(parts))

    def test_new_foot_and_minimum_width_are_both_eight_mm(self):
        self.assertEqual(INITIAL_FOOT_WIDTH_M,.008)
        self.assertEqual(MINIMUM_WIDTH_M,.008)
        mesh=unpack(rim(self.row))
        self.assertAlmostEqual(mesh.bounds[1,0]-mesh.bounds[0,0],.028,places=8)
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='foot',kind='foot_band',solid=solid(mesh),frames=[np.eye(4)])])
        self.assertTrue(guard.assess(mesh)['passed'])
        self.assertAlmostEqual(mesh.extents[2],.005,places=8)

    def test_supplemental_feet_are_widened_without_increasing_height(self):
        from wrap_ground_stock import prepare
        core=box(np.array([-.003,-.003,.02]),np.array([.003,.003,.04]))
        domain=box(np.array([-.1,-.1,0]),np.array([.1,.1,.1]))
        builder,patches=prepare(core,domain,[self.row])
        self.assertTrue(patches)
        for patch in patches:
            extents=unpack(patch).extents
            np.testing.assert_allclose(extents,[.008,.008,.005],atol=1e-10,rtol=0)

    def test_just_under_eight_mm_connection_width_rejected(self):
        value=box(np.array([0,0,0]),np.array([.04,.0079,.005]))
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='strip',kind='foot_band',solid=value,frames=[np.eye(4)])])
        self.assertFalse(guard.assess(value)['passed'])

    def test_five_mm_plate_thickness_is_not_a_width_failure(self):
        for thickness in [.005,.003]:
            value=box(np.array([0,0,0]),np.array([.04,.020,thickness]))
            guard=ConnectionWidthGuard(md.Manifold(),[dict(name='plate',kind='foot_band',solid=value,frames=[np.eye(4)])])
            self.assertTrue(guard.assess(value)['passed'])

    def test_thick_lobes_do_not_hide_a_thin_connected_neck(self):
        a=box(np.array([0,0,0]),np.array([.02,.02,.02]))
        b=box(np.array([.04,0,0]),np.array([.06,.02,.02]))
        neck=box(np.array([.019,.007,0]),np.array([.041,.013,.005]))
        mesh=unpack(a+b+neck)
        self.assertEqual(physical_component_count(solid(mesh)),1)
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='neck',kind='foot_band',solid=solid(mesh),frames=[np.eye(4)])])
        self.assertFalse(guard.assess(mesh)['passed'])

    def test_short_narrow_bridge_cannot_hide_inside_attachment_caps(self):
        a=box(np.array([0,0,0]),np.array([.02,.02,.005]))
        b=box(np.array([.022,0,0]),np.array([.042,.02,.005]))
        neck=box(np.array([.019,.007,0]),np.array([.023,.013,.005]))
        value=a+b+neck
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='short_neck',kind='foot_band',solid=neck,frames=[np.eye(4)])])
        self.assertEqual(physical_component_count(value),1)
        self.assertFalse(guard.assess(value)['passed'])

    def test_rotated_six_mm_square_is_not_an_eight_mm_connection(self):
        rod=box(np.array([0,-.003,-.003]),np.array([.06,.003,.003]))
        T=trimesh.transformations.rotation_matrix(np.pi/4,[1,0,0])
        from deterministic_base import moved
        rod=moved(rod,T)
        Q=np.eye(4);Q[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='square',kind='round_link',solid=rod,frames=[np.eye(4),Q])])
        self.assertFalse(guard.assess(rod)['passed'])

    def test_thin_rotated_wide_link_is_checked_by_width(self):
        from deterministic_base import moved
        value=box(np.array([0,-.00405,-.00025]),np.array([.06,.00405,.00025]))
        value=moved(value,trimesh.transformations.rotation_matrix(np.pi/4,[1,0,0]))
        Q=np.eye(4);Q[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='wide_thin',kind='round_link',solid=value,frames=[np.eye(4),Q])])
        self.assertTrue(guard.assess(value)['passed'])

    def test_compound_plate_short_narrow_neck_is_rejected(self):
        a=box(np.array([0,0,0]),np.array([.02,.02,.005]))
        b=box(np.array([.022,0,0]),np.array([.042,.02,.005]))
        neck=box(np.array([.019,.007,0]),np.array([.023,.013,.005]))
        value=a+b+neck
        Q=np.eye(4);Q[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='plate',kind='flat_stock',solid=value,frames=[np.eye(4),Q])])
        self.assertFalse(guard.assess(value)['passed'])

    def test_new_wing_on_five_mm_contact_wall_does_not_make_a_thickness_gate(self):
        core=box(np.array([0,0,0]),np.array([.08,.005,.02]))
        wing=box(np.array([0,.004,0]),np.array([.08,.006,.005]))
        value=core+wing
        guard=ConnectionWidthGuard(core,[dict(name='wing',kind='ground_patch',solid=wing,frames=[np.eye(4)])])
        self.assertTrue(guard.assess(value)['passed'])

    def test_polygon_scan_lines_match_geos_with_holes_and_boundary_edges(self):
        from wrap_width_profiles import sections,intervals
        import shapely
        from shapely.geometry import MultiPolygon
        outer=Polygon([(0,0),(5,0),(5,4),(3,4),(3,2),(2,2),(2,4),(0,4)],
            holes=[[(.5,.5),(.5,1.5),(1.5,1.5),(1.5,.5)]])
        shape=MultiPolygon([outer,Polygon([(7,0),(9,1),(8,3)])])
        xs=np.unique(np.r_[np.linspace(-1,10,73),[0,.5,1.5,2,3,5,7,8,9]])
        lines=shapely.linestrings(np.array([[[x,-1],[x,5]] for x in xs]))
        reference=[intervals(line) for line in shapely.intersection(lines,shape)]
        actual=sections(shape,xs,-1,5)
        for a,b in zip(actual,reference):np.testing.assert_allclose(np.array(a).reshape(-1,2),np.array(b).reshape(-1,2),atol=1e-12,rtol=0)

    def test_each_member_cap_starts_at_its_own_boundary(self):
        from shapely.geometry import LineString,box as rectangle
        from shapely.ops import unary_union
        from wrap_width import footprint_solid
        distant=rectangle(0,-.04,.06,-.03)
        closer=LineString([(.01,0),(.05,0)]).buffer(.004,quad_segs=32)
        shape=unary_union([distant,closer])
        value=footprint_solid(shape,0,.005)
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='feet',kind='ground_patch',solid=value,frames=[np.eye(4)])])
        self.assertTrue(guard.assess(value)['passed'])

    def test_twisting_five_mm_sheet_preserves_its_lateral_width(self):
        vertices=[];faces=[]
        for x in np.linspace(0,.08,13):
            angle=x/.08*np.pi/2;c,s=np.cos(angle),np.sin(angle)
            for y,z in [(-.01,-.0025),(.01,-.0025),(.01,.0025),(-.01,.0025)]:
                vertices.append([x,c*y-s*z,s*y+c*z])
        faces.extend([[0,2,1],[0,3,2]])
        for k in range(12):
            for j in range(4):
                a=4*k+j;b=4*k+(j+1)%4;c=a+4;d=b+4
                faces.extend([[a,b,d],[a,d,c]])
        n=len(vertices)-4;faces.extend([[n,n+1,n+2],[n,n+2,n+3]])
        value=solid(trimesh.Trimesh(vertices,faces,process=False))
        Q=np.eye(4);Q[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='sheet',kind='flat_stock',solid=value,frames=[np.eye(4),Q])])
        self.assertTrue(guard.assess(value)['passed'])

    def test_width_goal_is_not_a_hard_deletion_gate(self):
        self.row['hull'][:,:2]*=.1
        before=box(np.array([-.02,-.006,0]),np.array([.02,.006,.005]))
        after=box(np.array([-.02,-.00395,0]),np.array([.02,.00395,.005]))
        self.v.width_guard=ConnectionWidthGuard(md.Manifold(),[dict(name='foot',kind='foot_band',solid=before,frames=[np.eye(4)])])
        _,_,checks=self.v.check(unpack(after),before)
        self.assertTrue(checks['passed'])
        self.assertNotIn('minimum_width',checks)

    def test_conservative_budget_stops_over_deletion_before_expensive_checks(self):
        import tempfile
        from pathlib import Path
        from deterministic_base import volume
        self.row['hull'][:,:2]*=.1
        before=box(np.array([-.02,-.006,0]),np.array([.02,.006,.005]))
        mesh=unpack(before);value,proof,checks=self.v.check(mesh)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            deleter=Deleter(mesh,value,proof,checks,self.v,Path(folder))
            after=box(np.array([-.02,-.003,0]),np.array([.02,.003,.005]))
            self.assertFalse(deleter.attempt(after,'outside_cell',{}))
            self.assertEqual(len(deleter.states),1)
            self.assertAlmostEqual(volume(deleter.value),volume(before))
            self.assertIn('material budget',deleter.history[-1]['reason'])

    def test_original_five_mm_contact_body_never_fails_the_width_gate(self):
        core=box(np.array([0,0,0]),np.array([.04,.04,.005]))
        guard=ConnectionWidthGuard(core,[])
        result=guard.assess(core)
        self.assertTrue(result['passed'])
        self.assertFalse(result['wall_thickness_constrained'])

    def test_unneeded_dependent_branch_can_be_removed_in_the_same_connected_cut(self):
        import tempfile
        from pathlib import Path
        from deterministic_base import volume
        self.row['hull'][:,:2]*=.1
        main=box(np.array([-.02,-.01,0]),np.array([.02,.01,.005]))
        bridge=box(np.array([.019,-.002,0]),np.array([.04,.002,.005]))
        branch=box(np.array([.039,-.006,0]),np.array([.052,.006,.005]))
        before=main+bridge+branch
        cut=box(np.array([.027,-.02,-.001]),np.array([.033,.02,.006]))
        mesh=unpack(before);value,proof,checks=self.v.check(mesh)
        self.assertEqual(physical_component_count(before-cut),2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            deleter=Deleter(mesh,value,proof,checks,self.v,Path(folder))
            self.assertTrue(deleter.attempt(before-cut,'outside_cell',{},cut))
            self.assertEqual(physical_component_count(deleter.value),1)
            self.assertTrue(deleter.states[-1]['operation'].endswith('_and_dependent_branches'))
            self.assertLess(volume((deleter.initial-deleter.mask)-deleter.value),1e-10)
            self.assertLess(volume(deleter.value^branch),1e-10)

    def test_existing_legally_clipped_strip_can_be_deleted_without_waiving_new_necks(self):
        old=box(np.array([0,0,0]),np.array([.04,.006,.005]))
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='clipped',kind='foot_band',solid=old,frames=[np.eye(4)])])
        self.assertFalse(guard.assess(old)['passed'])
        self.assertTrue(guard.assess(old,old)['passed'])
        narrower=box(np.array([0,0,0]),np.array([.04,.0055,.005]))
        self.assertFalse(guard.assess(narrower,old)['passed'])
        self.assertTrue(guard.assess(md.Manifold(),old)['passed'])

    def test_full_feature_may_be_deleted(self):
        feature=box(np.array([0,0,0]),np.array([.04,.012,.005]))
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='strip',kind='foot_band',solid=feature,frames=[np.eye(4)])])
        self.assertTrue(guard.assess(md.Manifold())['passed'])

    def test_shortening_a_wide_plate_is_allowed(self):
        value=box(np.array([0,0,0]),np.array([.06,.012,.005]))
        shorter=box(np.array([0,0,0]),np.array([.03,.012,.005]))
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='strip',kind='foot_band',solid=value,frames=[np.eye(4)])])
        self.assertTrue(guard.assess(shorter)['passed'])

    def test_long_thin_branch_is_not_hidden_by_a_wide_body(self):
        body=box(np.array([0,0,0]),np.array([.02,.02,.005]))
        branch=box(np.array([.019,.0085,0]),np.array([.04,.0115,.005]))
        value=body+branch
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='branch',kind='foot_band',solid=value,frames=[np.eye(4)])])
        self.assertFalse(guard.assess(value)['passed'])

    def test_flat_link_thickness_is_not_an_additional_width_gate(self):
        value=box(np.array([0,-.006,0]),np.array([.04,.006,.005]))
        frames=link_frames([0,0,.0025],[.04,0,.0025])
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='link',kind='round_link',solid=value,frames=frames)])
        self.assertTrue(guard.assess(value)['passed'])

    def test_extra_height_cannot_hide_a_narrow_foot(self):
        value=box(np.array([0,0,0]),np.array([.04,.0079,.02]))
        guard=ConnectionWidthGuard(md.Manifold(),[dict(name='foot',kind='foot_band',solid=value,frames=[np.eye(4)])])
        self.assertFalse(guard.assess(value)['passed'])

    def test_internal_cavity_is_not_a_second_material_body(self):
        outer=box(np.zeros(3),np.array([.03]*3))
        void=box(np.array([.01]*3),np.array([.02]*3))
        self.assertEqual(physical_component_count(outer-void),1)

    def test_projection_retains_a_real_island_inside_a_ring(self):
        outer=box(np.array([0,0,0]),np.array([.1,.1,.005]))
        hole=box(np.array([.01,.01,-.001]),np.array([.09,.09,.006]))
        island=box(np.array([.04,.04,0]),np.array([.06,.06,.005]))
        result=projection((outer-hole)+island)
        self.assertAlmostEqual(result.area,.1**2-.08**2+.02**2,places=9)

    def test_full_width_polygon_members_do_not_fail_at_miter_corners(self):
        for xy in [np.array([[0,0],[.06,0],[.045,.025],[.015,.04]]),
                   np.array([[0,0],[.06,0],[.01,.015]])]:
            row=dict(self.row,hull=np.c_[xy,np.zeros(len(xy))])
            value=rim(row)
            f=dict(name='frame',kind='foot_band',solid=value,frames=[np.eye(4)],tracks=foot_tracks(row))
            self.assertTrue(ConnectionWidthGuard(md.Manifold(),[f]).assess(value)['passed'])

    def test_contiguous_section_pieces_are_one_width(self):
        from shapely.geometry import MultiLineString
        from wrap_width_profiles import intervals
        self.assertEqual(intervals(MultiLineString([[(0,0),(0,.004)],[(0,.004),(0,.008)]])),[(0.,.008)])

    def test_added_material_rejected(self):
        band=rim(self.row)
        extension=box(np.array([-.004,.012,0]),np.array([.004,.025,.005]))
        with self.assertRaisesRegex(ValueError,'Original solid guards'):
            self.v.check(unpack(band+extension),band)

    def test_diagonal_route_always_progresses(self):
        valid=np.zeros((3,3,3),bool);path=np.array([[0,1,0],[1,2,0]])
        valid[tuple(path.T)]=True
        self.assertFalse(valid[tuple(np.rint(np.mean(path,axis=0)).astype(int))])
        result,fallbacks=simplify_path(path,valid)
        np.testing.assert_array_equal(result,path)
        self.assertEqual(len(fallbacks),1)

    def test_clear_route_keeps_original_shortcut(self):
        path=np.array([[0,1,0],[1,2,0],[2,2,0]])
        result,fallbacks=simplify_path(path,np.ones((3,3,3),bool))
        np.testing.assert_array_equal(result,path[[0,-1]])
        self.assertEqual(fallbacks,[])

if __name__=='__main__':unittest.main()
