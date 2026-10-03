"""Tests for physical loss semantics, grammar operations and actual descent."""
import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
from step3_scheculer import contact_dsl as D
from step3_scheculer import passive_support as U


class GrammarTests(unittest.TestCase):
    def test_indeterminate_lp_proposal_cannot_be_accepted(self):
        from unittest.mock import patch, Mock
        from step3_scheculer import run_dsl as R
        search = R.Search.__new__(R.Search)
        contact = dict(triangles_m=np.zeros((1,3,3)), source_faces=np.array([0]))
        search.compiler = SimpleNamespace(compile=lambda p:[contact])
        search.task = SimpleNamespace(targets=np.zeros((32768,6)), supply=lambda c:np.eye(7))
        search.lp_cache = {}; search.sample_ids = np.array([2,7]); search.emit = Mock()
        with patch.object(R.J, 'classify', side_effect=RuntimeError('HiGHS Status 15: Unknown')):
            score = search.classify(D.Program((D.Patch(1,(0.,0.,0.),.1),)), refine=True)
        self.assertTrue(score['indeterminate'])
        self.assertFalse(score['passed'])
        self.assertFalse(score['mask'].any())
        np.testing.assert_array_equal(search.sample_ids, [2,7])
        self.assertEqual(search.emit.call_args.kwargs['operation'], 'reject_numerically_indeterminate_lp_proposal')

    def test_cuda_padded_cones_match_cpu_residuals(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('No CUDA device')
        rng = np.random.default_rng(51)
        rays = [rng.normal(size=(m, 7)) for m in (8, 12, 24, 48)]
        targets = rng.normal(size=(24, 7))
        expected = np.array([D.cone_distance(f, targets) for f in rays])
        actual = D.ConeBatch('cuda', 600).solve(rays, targets)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-7)
        chunked = D.ConeBatch('cuda', 600).solve(rays, np.tile(targets, (4, 1)))
        np.testing.assert_allclose(chunked, np.tile(expected, (1, 4)), atol=1e-8, rtol=1e-7)
        # Normalizing rays must preserve capacity even when units differ widely.
        scaled = [f*np.geomspace(.001, 1000, len(f))[:, None] for f in rays]
        np.testing.assert_allclose(D.ConeBatch('cuda', 600).solve(scaled, targets), actual, atol=1e-12)
        head = np.array([[0., 0., -1., 0., 0., 0.]])
        augmented = np.vstack([U.heads(head, np.ones(6)), np.r_[np.zeros(6), -1.]])
        self.assertGreater(D.ConeBatch('cuda', 600).solve([augmented], head)[0, 0], .1)
        almost_opposed = np.zeros((2, 7))
        almost_opposed[0, 0] = 1.
        almost_opposed[1, :2] = [-1., .001]
        target = np.eye(7)[1:2]
        self.assertLess(D.ConeBatch('cuda', 600).solve([almost_opposed], target)[0, 0], 1e-10)

    def test_copied_packages_do_not_import_original_baseline(self):
        import sys
        from step3_scheculer import run_dsl
        from step4_connect_support import try_dsl_growth
        bad = {n:getattr(m,'__file__','') for n,m in list(sys.modules.items())
               if '/slides/baseline_algo/' in (getattr(m,'__file__','') or '')}
        self.assertEqual(bad, {})

    def test_identity_and_reversible_count_changes(self):
        p = D.Patch(1, (0., 0., 0.), .1)
        q = D.Patch(2, (1., 0., 0.), .2)
        program = D.Program((p,))
        self.assertEqual(program.add(q).delete(1), program)
        self.assertEqual(program.substitute(0, q).patches, (q,))
        with self.assertRaises(ValueError):
            program.add(p)
        x = D.parameters(program, 2.)
        self.assertEqual(D.from_parameters(program, x, 2.), program)

    def test_unbounded_ray_scaling_cannot_act_as_a_soft_gate(self):
        full = np.vstack([np.eye(7), np.eye(7)[0]])
        target = np.array([[1., 2., 0., 0., 0., 0., 0.]])
        a = D.cone_distance(full, target)
        b = D.cone_distance(full*np.linspace(.001, 2., len(full))[:, None], target)
        np.testing.assert_allclose(a, b, atol=1e-16)

    def test_shared_uplift_condition_changes_feasibility(self):
        # A head that pushes the workpiece down can balance this demand in R6,
        # but its force lifts the massless fixture and must fail in R7.
        head = np.array([[0., 0., -1., 0., 0., 0.]])
        full = np.vstack([U.heads(head, np.ones(6)), np.r_[np.zeros(6), -1.]])
        self.assertGreater(D.cone_distance(full, head)[0], .1)

    def test_object_frame_direction_round_trip(self):
        rotation = trimesh.transformations.rotation_matrix(.7, [1., 0., 0.])
        task = SimpleNamespace(domain=SimpleNamespace(data={'frame':{'T_world_mesh':rotation.tolist()}}))
        vector = np.array([0., 1., 0.])
        np.testing.assert_allclose(D.object_direction(vector, task)@rotation[:3,:3].T, -vector, atol=1e-15)

    def test_descent_decreases_loss_with_backtracking(self):
        class Quadratic:
            scale = 1.
            min_radius = .001
            max_radius = 1.
            def loss(self, program, *args):
                c = np.asarray(program.patches[0].center)
                value = float(np.sum((c-np.array([.1, .2, .3]))**2))
                return value, dict(force=value, normal=0., alignment=0.)
        compiler = Quadratic()
        before = D.Program((D.Patch(1, (.6, .6, .6), .1),))
        after, history = D.descend(compiler, before, [], steps=10)
        self.assertLess(compiler.loss(after)[0], compiler.loss(before)[0])
        values = [r['loss'] for r in history]
        self.assertTrue(all(b <= a for a, b in zip(values, values[1:])))

    def test_compiler_keeps_entire_patch_above_floor_and_off_work(self):
        mesh = trimesh.creation.box(extents=[1., 1., 1.]); mesh.apply_translation([0, 0, .5])
        task = SimpleNamespace(pose='pose_1', floor=np.array([0., 0., 0.]), scale=np.ones(6),
            domain=SimpleNamespace(mesh=mesh, com=np.array([0., 0., .5]),
                work_ids=np.flatnonzero(mesh.face_normals[:,2] > .9),
                data={'frame':{'T_world_mesh':np.eye(4).tolist()}}))
        compiler = D.Compiler(task, [task])
        contact = compiler.compile_patch(D.Patch(1, (.5, 0., .002), .15))
        self.assertIsNotNone(contact)
        self.assertGreaterEqual(contact['triangles_m'][:,:,2].min(), .002-1e-12)
        self.assertEqual(len(np.intersect1d(contact['source_faces'], task.domain.work_ids)), 0)


if __name__ == '__main__':
    unittest.main()
