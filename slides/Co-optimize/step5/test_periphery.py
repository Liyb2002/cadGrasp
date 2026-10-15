"""Geometry regressions for exterior peeling and a resolved connector guard."""
import sys,unittest
from pathlib import Path
COOPT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(COOPT/name)for name in ['step5','helper_func','step5.2','step5.3','step4.2']]
import numpy as np
import trimesh
from periphery import clip_triangles,ThicknessGuard,structural_profile
from deterministic_base import solid,unpack,volume,SCALE


class PeripheryTests(unittest.TestCase):
    def test_clipped_contacts_gain_correct_boundary_vertices(self):
        triangles=np.array([[[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]])
        clipped,source=clip_triangles(triangles,np.array([17]),np.array([1.,0.,0.]),1.)
        self.assertTrue(np.all(clipped[:,:,0]<=1.))
        area=np.linalg.norm(np.cross(clipped[:,1]-clipped[:,0],clipped[:,2]-clipped[:,0]),axis=1).sum()/2
        self.assertAlmostEqual(area,1.5)
        self.assertTrue(np.all(source==17))
        self.assertTrue(any(np.allclose(v,[1,1,0])for v in clipped.reshape(-1,3)))

    def test_plane_cut_is_an_exterior_intersection(self):
        original=solid(trimesh.creation.box([.02,.02,.02]))
        cut=original.trim_by_plane((-1.,0.,0.),-.003/SCALE)
        self.assertLessEqual(volume(cut-original),1e-14)
        self.assertAlmostEqual(unpack(cut).bounds[1,0],.003)
        self.assertEqual(len(unpack(cut).split()),1)

    def test_thin_neck_separates_structural_cores(self):
        a=trimesh.creation.box([.012,.012,.012]);a.apply_translation([-.012,0,0])
        b=a.copy();b.apply_translation([.024,0,0])
        narrow=trimesh.creation.box([.025,.002,.002])
        wide=trimesh.creation.box([.025,.008,.008])
        thin=unpack(solid(a)+solid(b)+solid(narrow))
        thick=unpack(solid(a)+solid(b)+solid(wide))
        weak=ThicknessGuard(thin,.002,.01,pitch=.001)
        good=ThicknessGuard(thick,.002,.01,pitch=.001)
        self.assertGreater(weak.baseline,.2)
        self.assertLess(good.baseline,.01)

    def test_inward_plane_cannot_create_a_narrow_neck(self):
        mesh=trimesh.creation.box([.02,.02,.02])
        guard=ThicknessGuard(mesh,.002,.01,pitch=.001)
        passed,separated=guard.assess([(np.array([1.,0.,0.]),.005)])
        self.assertTrue(passed)
        self.assertEqual(separated,0.)

    def test_round_band_repairs_an_existing_neck_without_leaving_stock(self):
        from connector_bands import reinforce
        a=trimesh.creation.box([.012,.012,.012]);a.apply_translation([-.012,0,0])
        b=a.copy();b.apply_translation([.024,0,0])
        neck=trimesh.creation.box([.025,.002,.002])
        mesh=unpack(solid(a)+solid(b)+solid(neck))
        witness=trimesh.creation.box([.05,.03,.03])
        candidate,history,profile=reinforce(mesh,witness,trimesh.creation.box([.001]*3),[],mesh.bounds,
            dict(pitch_m=.001,guard_radius_m=.002,maximum_separated_core_fraction=.01,
                references=[dict(profiles=[dict(radius_m=.002,component_volumes_cm3=[1.,.01])])]),lambda m:(m,{}))
        self.assertTrue(any(row['accepted']for row in history))
        self.assertLessEqual(profile['profiles'][0]['separated_core_fraction'],.01)
        self.assertLessEqual(volume(solid(candidate)-solid(witness)),1e-10)

    def test_connector_measurement_preserves_an_internal_void(self):
        outer=solid(trimesh.creation.box([.02]*3))
        inner=solid(trimesh.creation.box([.012]*3))
        mesh=unpack(outer-inner)
        record,voxels,_=structural_profile(mesh,pitch=.001,radii=(.001,))
        center=np.rint(-voxels.transform[:3,3]/.001).astype(int)
        self.assertFalse(voxels.matrix[tuple(center)])
        self.assertTrue(record['cavities_preserved'])

    def test_inward_skin_keeps_a_selected_contact_and_removes_the_opposite_one(self):
        from rounded_front import inner_envelope,round_ports
        box=trimesh.creation.box([.02]*3)
        _,voxels,distance=structural_profile(box,pitch=.001,radii=(.001,))
        guide=inner_envelope(distance,voxels.transform[:3,3],.001,.0025,.0015)
        port=round_ports([dict(center_m=[.01,0,0],extent_m=.001)],.002)
        candidate=solid(box)^(guide+port)
        mesh=unpack(candidate)
        self.assertAlmostEqual(mesh.bounds[1,0],.01,places=10)
        self.assertGreater(mesh.bounds[0,0],-.0098)
        self.assertLess(volume(candidate),volume(solid(box)))
        self.assertLessEqual(volume(candidate-solid(box)),1e-10)
        self.assertTrue(mesh.is_watertight)

    def test_shared_round_junction_retains_reference_width_when_caps_shrink(self):
        from anchor_network import ball,path_body
        core=ball([0.,0.,0.],.0015)
        paths=[np.array([[0.,0.,0.],end])for end in [[.02,0.,0.],[0.,.02,0.],[0.,0.,.02]]]
        for path in paths:core=core+path_body(path,.0015)
        envelope=ball([0.,0.,0.],.004)
        for path in paths:envelope=envelope+path_body(path,.0016)
        self.assertLessEqual(volume(core-envelope),1e-10)
        mesh=unpack(envelope)
        profile,_,_=structural_profile(mesh,pitch=.0005,radii=(.001,))
        self.assertLess(profile['profiles'][0]['separated_core_fraction'],.01)

    def test_round_capsules_remain_solid_at_coincident_and_near_coincident_ends(self):
        from tube_support import capsule
        for separation in [0.,1e-15,1e-10,.001,.02]:
            value=capsule([.013,.027,.041],[.013+separation,.027,.041],.0015)
            mesh=unpack(value)
            self.assertTrue(mesh.is_watertight)
            self.assertGreater(volume(value),1.3e-8)
            self.assertEqual(len(mesh.split()),1)

    def test_ground_ports_preserve_point_contacts_without_planar_floor_faces(self):
        from tube_support import ground_ports
        from deterministic_base import enclosure
        cone=trimesh.creation.cone(radius=.01,height=.02,sections=32)
        cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[1,0,0]))
        cone.apply_translation([0,0,.02])
        row=dict(fixture_transform=np.eye(4),object_ground=np.empty((0,2)),
            hull=np.array([[0.,0.,0.]]),pose='test_pose')
        ports,_=ground_ports(cone,[row])
        self.assertGreaterEqual(len(ports),1)
        self.assertTrue(enclosure(cone,[row])[0]['passed'])
        self.assertTrue(any(abs(p['center_m'][2])<1e-8 for p in ports))

    def test_detached_numerical_shell_removal_preserves_every_inner_connector(self):
        from anchor_network import ball,path_body
        from backbone_regularization import regularize
        from prune_support import material_components
        nodes=np.array([[0.,0.,0.],[.02,0.,0.]])
        paths=[(0,1,nodes)]
        core=path_body(nodes,.0015)
        mesh=unpack(core+ball([.04,0.,0.],.00015))
        cleaned,radius,evidence=regularize(mesh,nodes,paths,.0015,.001)
        self.assertEqual(len(material_components(solid(cleaned))),1)
        self.assertAlmostEqual(radius,.00145)
        self.assertTrue(evidence['every_inner_ball_and_capsule_checked'])
        self.assertLessEqual(volume(path_body(nodes,radius)-solid(cleaned)),1e-12)

    def test_projected_backbone_proves_both_actual_products_above_reference_width(self):
        from anchor_network import path_body
        from project_backbone import project
        from prune_support import material_components
        nodes=np.array([[0.,0.,0.],[.02,0.,0.],[.02,.015,0.]])
        paths=[(0,1,nodes[:2]),(1,2,nodes[1:])]
        tree=lambda r:path_body(nodes[:2],r)+path_body(nodes[1:],r)
        domain=solid(trimesh.creation.box([.1]*3))
        for inset in [.00005,.00015]:
            packet=unpack(tree(.00145));full=unpack(tree(.0018));core=unpack(tree(.0017))
            radius,record=project(packet,full,core,nodes,paths,.00145,.001,domain,proof_inset_m=inset)
            self.assertGreater(radius,.001)
            self.assertAlmostEqual(radius,.00145-inset)
            for mesh in [packet,full,core]:
                self.assertEqual(len(material_components(solid(mesh))),1)
                self.assertLessEqual(volume(tree(radius)-solid(mesh)),1e-12)
            self.assertTrue(record['every_inner_ball_and_capsule_checked'])
        with self.assertRaisesRegex(ValueError,'below reference'):
            project(packet,full,core,nodes,paths,.0011,.001,domain,proof_inset_m=.00015)


if __name__=='__main__':unittest.main()
