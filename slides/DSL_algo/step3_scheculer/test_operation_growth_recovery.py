"""A thin valid contact must not become disconnected by search-only padding."""
import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
from step3_scheculer.operation_growth_recovery import RecoveryGrow
from step3_scheculer import operation_dsl as F
from step4_connect_support import growing_support as L


class ContactTransitionTests(unittest.TestCase):
    def test_exact_safe_transition_bridges_search_padding_only(self):
        g=RecoveryGrow.__new__(RecoveryGrow)
        obj=trimesh.creation.box(extents=(.04,.04,.04));obj.apply_translation([0,0,.03])
        ids=np.flatnonzero(obj.face_normals[:,0]>.99);face=int(ids[0])
        triangles=np.array([[[.02,-.003,.027],[.02,.003,.027],[.02,.003,.033]],[[.02,-.003,.027],[.02,.003,.033],[.02,-.003,.033]]])
        c=dict(candidate_id='H',center_m=np.array([.02,0,.03]),center_face=face,source_faces=np.array([face,face]),triangles_m=triangles,triangle_areas_m2=np.array([.000018,.000018]))
        root=trimesh.creation.box(extents=(.0001,.006,.006));root.apply_translation([.02005,0,.03]);original=F.S.solid(root)
        pad=trimesh.creation.box(extents=(.0408,.0408,.0408));pad.apply_translation([0,0,.03])
        outer=F.S.solid(trimesh.creation.box(extents=(.2,.2,.2)))
        g.mask=(outer-F.S.solid(pad))+original
        g.bases=np.array([np.eye(3)]);g.offsets=np.zeros((1,3))
        g.sweep_meshes=[obj];g.navigation_window=dict(min_m=[-.1,-.1,-.1],max_m=[.1,.1,.1])
        g.case=SimpleNamespace(poses=['pose_1'],groups=[[c]],tasks=[SimpleNamespace(domain=SimpleNamespace(mesh=obj))])
        g.group=SimpleNamespace(name='synthetic-padding-gap')
        sphere=trimesh.creation.icosphere(subdivisions=2,radius=.0026);g.bead=sphere.vertices
        g.guaranteed_radius=float(np.min(np.abs(np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0]))))
        g.contact_vertices=triangles.reshape(-1,3);g.buried_contact_rejections=0
        g.original_roots=[];g.original_soles=[];g.terminals=[];g.seed_positions=[];g.core_solids=[];g.thickness=[]
        g.attach(original,root.vertices,'pose_1:H')
        self.assertEqual(len(g.terminals),1)
        solid=g.terminals[0]['solid']
        self.assertEqual(len(L.components(solid)),1)
        self.assertLess(abs(float((solid^F.S.solid(obj)).volume()))*F.S.SCALE**3,8e-14)
        self.assertLess(abs(float((original-solid).volume()))*F.S.SCALE**3,8e-14)
        self.assertGreater(g.guaranteed_radius*2000,5.)
        self.assertTrue(g.thickness[0]['actual_continuous_sweep_enforced'])

if __name__=='__main__':unittest.main()
