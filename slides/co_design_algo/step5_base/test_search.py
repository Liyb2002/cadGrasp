"""Independent checks for the floor-cone reduction and bounded search."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog
from step5_base import base as C, bearing as M


class SearchTests(unittest.TestCase):
    def test_floor_cone_reduction_preserves_all_original_rays(self):
        # A triangulated concave contact patch with repeated/interior vertices.
        foot = dict(pads_xy_m=[np.array([[0., 0.], [2., 0.], [1., .3]]),
                              np.array([[0., 0.], [1., .3], [0., 2.]]),
                              np.array([[0., 2.], [1., .3], [2., 1.]])])
        points = np.empty((0, 3)); owners = np.empty(0, int)
        args = (points, points, owners, np.array([.5, .5, 1.]), np.ones(6), [foot], 3.)
        original, ground = M.Q.grounded_matrix(*args)
        reduced, extreme = M.grounded_matrix(*args)
        self.assertLess(reduced.shape[1], original.shape[1])
        actual = np.unique(np.concatenate(foot['pads_xy_m']), axis=0)
        for p in extreme['points_m'][:, :2]:
            self.assertTrue(np.any(np.all(p == actual, axis=1)))
        for ray in original.T:
            # Independent nonnegative cone-membership LP, not reusing hull code.
            result = linprog(np.zeros(reduced.shape[1]), A_eq=reduced[6:], b_eq=ray[6:],
                             bounds=(0, None), method='highs')
            self.assertTrue(result.success)
            np.testing.assert_allclose(reduced@result.x, ray, atol=1e-10)

    def test_different_holes_same_extremes_share_mechanics(self):
        from shapely.geometry import Polygon
        outer = [[0,0],[2,0],[2,2],[0,2]]
        a = Polygon(outer)
        b = Polygon(outer, holes=[[[.5,.5],[1.5,.5],[1.5,1.5],[.5,1.5]]])
        groups = C.equivalent_ground_groups([(4.,4.,0,0,None,a,{}),(4.,3.,1,1,None,b,{})])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

    def test_refinement_keeps_certified_incumbent(self):
        calls = []
        def evaluate(rank):
            calls.append(rank)
            return rank if rank >= 791 else None
        result, trials = C.adaptive_search(4097, evaluate)
        self.assertEqual(result, 896)  # retain a slightly larger certified base
        self.assertEqual(trials[trials.index(1024)+1:], [768, 896])
        self.assertEqual(len(trials), 14)
        self.assertEqual(len(trials), len(set(trials)))
        self.assertEqual(trials, calls)

    def test_failed_refinements_return_the_original_certificate(self):
        certified = object()
        result, trials = C.adaptive_search(4097,
            lambda rank: certified if rank == 1024 else None)
        self.assertIs(result, certified)
        self.assertEqual(trials[trials.index(1024)+1:], [768, 896])

    def test_zero_refinement_budget_returns_first_certificate(self):
        result, trials = C.adaptive_search(4097,
            lambda rank: rank if rank >= 791 else None, max_refinements=0)
        self.assertEqual(result, 1024)
        self.assertEqual(trials[-1], 1024)

    def test_no_certificate_stays_unresolved_after_logarithmic_probes(self):
        result, trials = C.adaptive_search(100000, lambda rank: None)
        self.assertIsNone(result)
        self.assertEqual(trials[-1], 99999)
        self.assertLess(len(trials), 20)

    def test_continuous_probe_uses_certificate_and_remembers_original_index(self):
        loads = np.zeros((100,6)); loads[:,0] = np.arange(100)
        floor = dict(original_pivot_m=np.zeros(3), load_wrenches=loads,
                     continuous_outer_load_wrenches=loads)
        solver = SimpleNamespace(lp_count=1, solve=None)
        from unittest.mock import Mock
        solver.solve = Mock(return_value=dict(passed=False,diagnostics=[dict(index=1,status='coefficient_certificate_unresolved')]))
        witnesses = {('continuous', 87)}
        domain = SimpleNamespace(mesh=SimpleNamespace(extents=np.ones(3)), com=np.zeros(3))
        with patch.object(M,'bearing_rays',return_value=([],[],[])), \
             patch.object(M,'grounded_matrix',return_value=(np.zeros((12,12)),{})), \
             patch.object(M.Q,'BatchSolver',return_value=solver):
            report,_ = M.bearing(domain, [], floor, {}, (64.,), witnesses)
        self.assertTrue(solver.solve.call_args.kwargs['certified'])
        indices = report['attempts'][0]['probe_load_indices']
        self.assertIn(87,indices)
        self.assertIn(('continuous',indices[1]),witnesses)
        self.assertFalse(report['continuous_passed'])


if __name__ == '__main__': unittest.main()
