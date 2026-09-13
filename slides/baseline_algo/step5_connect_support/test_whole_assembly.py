"""One-body construction must keep every head and sweep all connecting material."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step2_local_support import insertion as D
from step2_local_support.surface import areas
from step5_connect_support import whole_assembly as A, motion as M


def contacts(mesh):
    ids = np.flatnonzero(mesh.face_normals[:, 0] < -.9)
    result = []
    for i, face in enumerate(ids):
        tri = mesh.triangles[face]; center = tri.mean(axis=0)
        tri = (center+.3*(tri-center))[None]
        result.append(dict(candidate_id=f'C{i}', candidate_index=i, center_m=center,
            center_face=int(face), radius_m=float(np.linalg.norm(tri-center, axis=2).max()),
            source_faces=np.array([face]), triangles_m=tri, triangle_areas_m2=areas(tri)))
    return result


class WholeConnectionTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box([1., 1., 1.]); self.mesh.apply_translation([0, 1., 0])
        self.contacts = contacts(self.mesh)
        self.required = np.array([[-.7, -.7], [.7, -.7], [.7, .7], [-.7, .7]])
        self.records = [dict(candidate_id=c['candidate_id'], certified_directions=D.normalize(isolated=[0.])) for c in self.contacts]

    def test_opposing_individual_directions_do_not_make_one_insertable_body(self):
        self.records[1]['certified_directions'] = D.normalize(isolated=[180.])
        report, module = A.search(self.mesh, self.contacts, self.records, self.required, np.zeros(3), .01)
        self.assertIsNone(module)
        self.assertEqual(report['status'], 'no_common_certified_head_direction')
        self.assertEqual(report['selected_ids'], ['C0', 'C1'])
        self.assertFalse(report['step3_selection_changed'])

    def test_open_base_covers_demand_and_sweeps_clear_but_reverse_is_blocked(self):
        parts, base = A.open_base(self.mesh, self.required, 0.)
        self.assertTrue(M.sweep_check(self.mesh, parts, np.array([1., 0, 0]))['passed'])
        check, _ = A.footprint(parts, np.zeros(3), self.required, 1.)
        self.assertTrue(check['passed'])
        # Put an obstacle down to the floor so it intersects a reverse sweep.
        grounded = trimesh.creation.box([1., 1., 1.]); grounded.apply_translation([0, .5, 0])
        self.assertFalse(M.sweep_check(grounded, parts, np.array([-1., 0, 0]))['passed'])

    def test_two_heads_and_common_base_form_one_swept_solid(self):
        report, module = A.search(self.mesh, self.contacts, self.records, self.required, np.zeros(3), .01)
        self.assertTrue(report['passed'], report)
        self.assertTrue(report['solid']['one_solid'])
        self.assertTrue(report['ground']['passed'])
        self.assertEqual([r['candidate_id'] for r in report['routes']], ['C0', 'C1'])
        self.assertGreater(len(module['floor_triangles']), 0)

    def test_budget_exhaustion_retains_failure_without_reselection(self):
        report, module = A.search(self.mesh, self.contacts, self.records, self.required, np.zeros(3), .01, edge_budget=1)
        self.assertIsNone(module)
        self.assertEqual(report['status'], 'whole_support_search_budget_exhausted')
        self.assertEqual(report['new_edge_checks'], 1)
        self.assertFalse(report['global_impossibility_claimed'])

    def test_no_heads_does_not_construct_an_empty_success(self):
        report, module = A.search(self.mesh, [], [], self.required, np.zeros(3), .01)
        self.assertEqual(report['status'], 'no_selected_heads'); self.assertIsNone(module)

    def test_mixed_direction_heads_share_one_body_reaction_certificate(self):
        from step4_floor_contact import equilibrium as Q
        from step4_floor_contact.audit import replay
        mesh = trimesh.creation.box([.2, .2, .2]); mesh.apply_translation([0, .5, 0])
        heads = []
        for i, triangle in enumerate(mesh.triangles):
            center = triangle.mean(axis=0)
            heads.append(dict(triangles_m=(center+.3*(triangle-center))[None], source_faces=np.array([i])))
        domain = SimpleNamespace(mesh=mesh, com=mesh.center_mass)
        p, n, _ = Q.contact_rays(domain, heads, np.zeros(3))
        weights = np.random.default_rng(774).uniform(.01, .02, (12, len(p)))
        weights[:, n[:, 1] > 0] += .1
        loads = weights@Q.wrench(p, n, domain.com)
        # Constructed wrench-polytope test; not a CAD machining-domain verdict.
        floor = dict(original_pivot_m=np.zeros(3), load_wrenches=loads, continuous_outer_load_wrenches=loads)
        base = dict(pads_xz_m=[[[-1, -1], [1, -1], [1, 1], [-1, 1]]])
        report, arrays = A.bearing(domain, heads, floor, base)
        self.assertTrue(report['continuous_passed'], report)
        self.assertEqual(report['body_count'], 1)
        self.assertTrue(np.all(arrays['contact_owners'][1:] == 0))
        for prefix in ('sample', 'continuous'):
            checked = replay(arrays, prefix, loads, 1, report['sufficient_friction_coefficient'])
            self.assertLess(checked['maximum_body_equilibrium_residual_conditioned'], 1e-10)


if __name__ == '__main__': unittest.main()
