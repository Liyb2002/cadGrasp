"""Free-space feasibility tests: no thin-body or minimum-material assumption."""
import unittest
import contextlib
import io
from types import SimpleNamespace

import manifold3d as md
import numpy as np
import trimesh

from step4_connect_support.baseline_current.boxed_support import bounded_space, maximal_component, box_checks
from step4_connect_support.baseline_current import build_coupled_saddle as S
from step4_connect_support.baseline_current import working_surface as SURFACE


def cube(lo, hi):
    lo, hi = np.array(lo), np.array(hi)
    return md.Manifold.cube(((hi-lo)/S.SCALE).tolist()).translate((lo/S.SCALE).tolist())


class BoxedSupportTests(unittest.TestCase):
    def test_keep_all_connected_available_material(self):
        allowed = cube([0, 0, 0], [.1, .1, .1])
        forbidden = cube([.04, .04, 0], [.06, .06, .08])
        roots = [cube([.01, .01, .01], [.02, .02, .02]), cube([.08, .08, .08], [.09, .09, .09])]
        result, check = maximal_component(allowed, forbidden, roots)
        self.assertIsNotNone(result)
        self.assertEqual(check['free_component_count'], 1)
        self.assertAlmostEqual(result.volume()*S.SCALE**3, .001-.02*.02*.08, places=12)
        self.assertAlmostEqual((result ^ forbidden).volume(), 0., places=12)

    def test_no_bridge_through_exit_space(self):
        allowed = cube([0, 0, 0], [.1, .1, .1])
        wall = cube([.04, -.01, -.01], [.06, .11, .11])
        roots = [cube([.01, .01, .01], [.02, .02, .02]), cube([.08, .08, .08], [.09, .09, .09])]
        result, check = maximal_component(allowed, wall, roots)
        self.assertIsNone(result)
        self.assertEqual(check['reason'], 'roots_in_different_free_components')

    def test_closed_forbidden_cavity_is_not_filled_by_component_selection(self):
        allowed = cube([0,0,0], [.1]*3)
        forbidden = cube([.04]*3, [.06]*3)
        root = cube([.01]*3, [.02]*3)
        result, check = maximal_component(allowed, forbidden, [root])
        self.assertIsNotNone(result)
        self.assertEqual(check['preserved_cavity_count'], 1)
        self.assertAlmostEqual(result.volume()*S.SCALE**3, .001-.02**3, places=12)
        self.assertAlmostEqual((result^forbidden).volume(), 0., places=12)

    def test_discard_unused_component_without_trimming_used_one(self):
        allowed = cube([0, 0, 0], [.1, .1, .1])
        wall = cube([.04, -.01, -.01], [.06, .11, .11])
        root = cube([.01, .01, .01], [.02, .02, .02])
        result, check = maximal_component(allowed, wall, [root])
        self.assertIsNotNone(result)
        self.assertEqual(check['free_component_count'], 2)
        self.assertAlmostEqual(result.volume()*S.SCALE**3, .04*.1*.1, places=12)

    def test_installed_box_constraints_apply_to_every_pose(self):
        bounds = dict(min_m=[0, 0, 0], max_m=[.1, .2, .3])
        bases = np.array([np.eye(3), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]], float)
        offsets = np.array([[0, 0, 0], [0, .1, 0]])
        allowed = bounded_space(bounds, bases, offsets)
        self.assertGreater(allowed.volume(), 0.)
        self.assertTrue(all(r['all_inside'] for r in box_checks(S.unpack(allowed), bounds, bases, offsets, ['a', 'b'])))

    def test_root_outside_budget_is_not_clipped_or_accepted(self):
        allowed = cube([0, 0, 0], [.1, .1, .1])
        root = cube([.09, .01, .01], [.11, .02, .02])
        result, check = maximal_component(allowed, md.Manifold(), [root])
        self.assertIsNone(result)
        self.assertEqual(check['reason'], 'mandatory_roots_outside_box_or_floor')

    def test_disjoint_installed_boxes_have_no_material_space(self):
        bounds = dict(min_m=[0,0,0], max_m=[.1]*3)
        bases = np.repeat(np.eye(3)[None], 2, axis=0)
        offsets = np.array([[0,0,0], [.2,0,0]])
        self.assertTrue(bounded_space(bounds, bases, offsets).is_empty())

    def test_complete_solid_preserves_contact_and_allows_continuous_exit(self):
        obj = trimesh.creation.box(extents=[.02]*3,
            transform=trimesh.transformations.translation_matrix([.05]*3))
        root = trimesh.creation.box(extents=[.002, .01, .01],
            transform=trimesh.transformations.translation_matrix([.061, .05, .05]))
        patch = np.array([[[.06,.045,.045],[.06,.055,.045],[.06,.055,.055]],
                          [[.06,.045,.045],[.06,.055,.055],[.06,.045,.055]]])
        task = SimpleNamespace(pose='synthetic', targets=np.zeros((32768, 6)),
            domain=SimpleNamespace(mesh=obj, work_ids=np.where(obj.face_normals[:, 0] < -.9)[0]))
        bases, offsets, directions = np.eye(3)[None], np.zeros((1, 3)), np.array([[1., 0, 0]])
        bounds = dict(min_m=[0,0,0], max_m=[.1]*3)
        sweep = S.swept_solid(obj, -S.SWEEP_LENGTH*directions[0])
        pad = md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
        full, _ = maximal_component(bounded_space(bounds, bases, offsets),
            S.solid(sweep).minkowski_sum(pad), [S.solid(root)])
        self.assertIsNotNone(full)
        mesh = S.unpack(full)
        with contextlib.redirect_stdout(io.StringIO()):
            checks, _ = S.verify([task], [[dict(triangles_m=patch)]], [[[root.vertices]]],
                directions, bases, offsets, mesh, [sweep], check_equilibrium=False)
        self.assertTrue(checks[0]['withdrawal']['clear'])
        self.assertTrue(checks[0]['independent_head_withdrawal']['clear'])
        self.assertLess(checks[0]['max_active_contact_gap_m'], 1e-8)
        self.assertTrue(SURFACE.check(mesh, task)['passed'])
        self.assertTrue(box_checks(mesh, bounds, bases, offsets, ['synthetic'])[0]['all_inside'])


if __name__ == '__main__':
    unittest.main()
