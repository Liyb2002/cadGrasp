"""Surface-head regression: exact patches, bounded new roots, strict floor replay."""
from types import SimpleNamespace
import unittest

import numpy as np
import trimesh

from step4_connect_support.baseline_current import zero_thickness_heads as Z
from step4_connect_support.baseline_current import head_registration as H
from step4_connect_support.baseline_current import build_coupled_saddle as S
from step2_local_support import geometry as G


def case_at_height(height):
    mesh = trimesh.creation.box(extents=[.02, .02, .02])
    mesh.apply_translation([.02, .02, .01+height])
    faces = np.flatnonzero(mesh.face_normals[:, 2] < -.5)
    triangles, source = [], []
    for face in faces:
        polygon = mesh.triangles[face]
        triangles.extend([[polygon.mean(0), polygon[j], polygon[(j+1)%3]] for j in range(3)])
        source.extend([face]*3)
    tasks, groups = [], []
    for k in range(2):
        tasks.append(SimpleNamespace(pose=f'pose_{k}', domain=SimpleNamespace(mesh=mesh.copy(),
            data={'frame': {'T_world_mesh': np.eye(4)}})))
        groups.append([dict(candidate_id=f'head_{k}', triangles_m=np.asarray(triangles),
            source_faces=np.asarray(source))])
    return SimpleNamespace(tasks=tasks, groups=groups, heads=[['old_probe_0'],['old_probe_1']],schedule={})


class SurfaceHeadTests(unittest.TestCase):
    def test_floor_margin_bounds_generated_material_and_preserves_exact_patch(self):
        case = case_at_height(.002)
        originals = [g[0]['triangles_m'].copy() for g in case.groups]
        Z.prepare(case,.0004)
        self.assertEqual(case.head_model,Z.MODEL)
        self.assertEqual(case.probe_heads,[['old_probe_0'],['old_probe_1']])
        for k, record in enumerate(case.support_seed_records):
            np.testing.assert_array_equal(np.asarray(case.heads[k][0]), originals[k])
            self.assertEqual(record['mandatory_input_head_thickness_m'],0.)
            self.assertLessEqual(record['constructor_root_normal_depth_m'],.0008)
            self.assertLessEqual(record['maximum_root_vertex_displacement_bound_m'],.001+1e-15)
            self.assertGreaterEqual(min(record['minimum_root_height_by_pose_m']),.001-1e-15)
            self.assertFalse(record['original_probe_volume_retained'])
            for cell in case.support_seeds[k][0]:
                self.assertGreater(S.solid(G.hull_mesh(cell)).volume(),0.)
        b,o = H.fixed_placements(case.tasks)
        heads,_ = H.register(case.groups,case.heads,b,o,general_layout=True)
        for k, head in enumerate(heads):
            np.testing.assert_allclose(Z.patch_mesh(head.contact_points).triangles,originals[k],atol=1e-16)

    def test_floor_touching_surface_cannot_spawn_unbounded_root(self):
        with self.assertRaisesRegex(ValueError,'no positive all-pose floor clearance'):
            Z.prepare(case_at_height(0.),.0004)

    def test_surface_mesh_keeps_nonplanar_triangles_without_a_convex_cap(self):
        triangles=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
                            [[0.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
        mesh=Z.patch_mesh(triangles)
        self.assertEqual(len(mesh.faces),2)
        self.assertFalse(mesh.is_watertight)
        np.testing.assert_array_equal(mesh.triangles,triangles)


if __name__=='__main__':
    unittest.main()
