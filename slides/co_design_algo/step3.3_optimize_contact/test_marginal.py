"""Physical residual semantics and strict gates for local area proposals."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.stage_imports import load_stage
M = load_stage('optimize', 'marginal')


class MarginalTests(unittest.TestCase):
    def test_exactly_feasible_and_infeasible_residuals(self):
        full = np.vstack([np.c_[np.eye(6), np.zeros(6)], np.r_[np.zeros(6), -1.]])
        self.assertLess(M.residual_gap(full, [1, 0, 0, 0, 0, 0]), 1e-8)
        self.assertAlmostEqual(M.residual_gap(full, [-1, 0, 0, 0, 0, 0]), 1.)

    def test_no_uplift_is_hard_even_in_the_proxy(self):
        # Head plus floor can balance Fx=1 in six dimensions, but the head
        # pulls up on the support; no positive head weight obeys lifted equality.
        full = np.array([[1, 0, -1, 0, 0, 0, -1],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, -1]], float)
        self.assertAlmostEqual(M.residual_gap(full, [1, 0, 0, 0, 0, 0]), 1.)

    def test_stratified_weights_preserve_mass_and_include_counterexamples(self):
        mask = np.r_[np.ones(97, bool), np.zeros(3, bool), True, False]
        ids, weights = M.representative_loads(mask, 100, 4)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertAlmostEqual(weights[ids < 97].sum(), 97/102)
        self.assertTrue({100, 101}.issubset(set(ids)))
        np.testing.assert_array_equal(ids, M.representative_loads(mask, 100, 4)[0])

    def test_proxy_cannot_trade_away_full_coverage(self):
        before = dict(row=dict(covered_count=100, total_area_m2=2., efficiency=50.))
        tempting = dict(row=dict(covered_count=99, total_area_m2=1., efficiency=99.))
        self.assertFalse(M.acceptable(before, tempting, 100))

    def test_unresolved_proxy_lp_uses_zero_reaction_residual_and_records_failure(self):
        proxy = M.Proxy.__new__(M.Proxy)
        proxy.indices = np.array([17])
        proxy.targets = np.array([[1., -2., 0., 0., 0., 0.]])
        proxy.lp_count = 0
        proxy.seconds = 0.
        proxy.lp_failures = []
        with patch.object(M, 'residual_gap', side_effect=M.ProposalLPError('unresolved')):
            np.testing.assert_array_equal(proxy.gaps(np.zeros((1, 7))), [3.])
        self.assertEqual(proxy.lp_count, 1)
        self.assertEqual(proxy.lp_failures[0]['load_index'], 17)

    def test_full_coverage_has_priority_over_partial_efficiency(self):
        before = dict(row=dict(covered_count=90, total_area_m2=1., efficiency=90.))
        full = dict(row=dict(covered_count=100, total_area_m2=2., efficiency=50.))
        self.assertTrue(M.acceptable(before, full, 100))
        self.assertGreater(M.result_key(full, 100), M.result_key(before, 100))

    def test_local_search_never_uses_global_radius_bounds(self):
        class Scalar:
            initial_radius = 1.
            minimum_radius = .01
            minimum_area = .01
            cap = 4.
            tolerance = 1e-5
            targets = np.zeros((100, 6))
            def __init__(self): self.cache = {}
            def score(self, r):
                if r not in self.cache:
                    count = 100 if r >= .9 else 90
                    self.cache[r] = dict(mask=np.arange(100) < count,
                        row=dict(radius_m=r, area_m2=r*r, total_area_m2=r*r,
                                 covered_count=count, efficiency=count/(r*r)))
                return self.cache[r]
            def insertion_directions(self, r): return {'ids': [1]}
            def maximum_radius(self): raise AssertionError('Global radius search is forbidden')
            def insertion_radius_cap(self, r): raise AssertionError('Global direction search is forbidden')
        class MisleadingProxy:
            def __init__(self, problem, initial):
                self.values = {}; self.indices = np.array([0]); self.weights = np.array([1.])
                self.tau = .02; self.lp_count = 0; self.seconds = 0.
                self.evaluate(problem.initial_radius)
            def evaluate(self, r):
                self.values[r] = dict(radius_m=r, area_m2=r*r,
                                     contribution=1., utility=1/(r*r))
                return self.values[r]
        scalar = Scalar()
        with patch.object(M, 'Proxy', MisleadingProxy):
            radius, report = M.search(scalar, 12)
        self.assertGreaterEqual(radius, .9)
        self.assertLess(radius, 1.)
        self.assertTrue(scalar.score(radius)['mask'].all())
        self.assertLessEqual(report['evaluated_sizes'], M.EXACT_TRIALS)


if __name__ == '__main__': unittest.main()
