"""Ratio optimization across multiple peaks, plateaus, shrinkage and expansion."""
import unittest
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal
import adjust as A


class SizeAdjustmentTests(unittest.TestCase):
    def test_hard_rest_constraint_sets_a_feasible_shrink_limit(self):
        radius,report=A.size_search.feasible_radius_floor(.1,1.,1e-6,lambda r:r>=.73)
        self.assertGreaterEqual(radius,.73)
        self.assertLess(radius,.730001)
        self.assertTrue(report['enforced'])
        with self.assertRaises(ValueError):
            A.size_search.feasible_radius_floor(.1,1.,1e-6,lambda r:False)

    def search(self, counts, radii, initial, budget=512):
        return A.maximize_efficiency(lambda r: dict(area_m2=r, covered_count=counts(r)),
                                    radii, initial, 1e-7, 1e-3, budget)

    def test_more_coverage_with_lower_efficiency_keeps_original(self):
        radius, report = self.search(lambda r: 10+int(5*(r-1)), [1., 2.], 1.)
        self.assertEqual(radius, 1.)
        self.assertFalse(report['improved'])
        self.assertTrue(report['converged'])

    def test_shrink_when_area_savings_outweigh_coverage_loss(self):
        radius, report = self.search(lambda r: 0 if r < .6 else 80 if r < 1 else 100,
                                     [.1, 1.], 1.)
        self.assertGreaterEqual(radius, .6)
        self.assertLess(radius, .601)
        self.assertGreater(report['best_value_count_per_m2'], 100)
        self.assertTrue(report['converged'])

    def test_expand_when_coverage_pays_for_area(self):
        radius, report = self.search(lambda r: int(100*r*r), [.5, 1., 2.], 1.)
        self.assertEqual(radius, 2.)
        self.assertEqual(report['best_value_count_per_m2'], 200)

    def test_search_does_not_assume_a_single_efficiency_peak(self):
        def count(r):
            return 0 if r < .8 else 100 if r < 2.2 else 400
        radius, report = self.search(count, [.4, 1., 3.], 1.)
        self.assertGreaterEqual(radius, 2.2)
        self.assertLess(radius, 2.203)
        self.assertTrue(report['converged'])
        self.assertGreaterEqual(report['upper_bound_count_per_m2'], 400/2.2)

    def test_equal_zero_efficiency_keeps_original(self):
        radius, report = self.search(lambda r: 0, [.1, 1., 2.], 1.)
        self.assertEqual(radius, 1.)
        self.assertEqual(report['relative_gap'], 0.)

    def test_budget_exhaustion_does_not_claim_convergence(self):
        radius, report = self.search(lambda r: int(r*100), [.1, 1., 2.], 1., budget=3)
        self.assertFalse(report['converged'])
        self.assertEqual(radius, 1.)
        self.assertTrue(any(b['reason'] == 'evaluation_budget' for b in report['bounds']))

    def test_zero_area_is_not_an_infinite_efficiency_candidate(self):
        with self.assertRaises(ValueError):
            A.efficiency(dict(area_m2=0., covered_count=1))

    def test_previous_area_is_included_in_efficiency_and_interval_bounds(self):
        # Using only the new area would prefer a useless tiny new patch.
        evaluate = lambda r: dict(area_m2=r, total_area_m2=1+r,
                                  covered_count=10 if r < .5 else 100)
        radius, report = A.maximize_efficiency(evaluate,[.01,.5,1.],1.,1e-7)
        self.assertAlmostEqual(radius,.5,places=3)
        self.assertLess(report['upper_bound_count_per_m2'],67.)
        self.assertAlmostEqual(A.efficiency(evaluate(.5)),100/1.5)

    def test_real_unmodified_radius_preserves_step3_sample_mask(self):
        if not (A.I.folder('B',A.M.OUTPUT_NAME,1)/'selection.json').exists():
            self.skipTest('Requires a current B Step3 selection; Step2-only reruns archive that fixture')
        import json
        saved=A.I.folder('B',A.OUTPUT_NAME,1)/'state.json'
        if not saved.exists() or not json.loads(saved.read_text()).get('rest_equilibrium_radius_constraint'):
            self.skipTest('Saved B selection predates the shared no-uplift force model')
        problem = A.SizeProblem('B')
        entry = problem.score(problem.initial_radius)
        assert_array_equal(entry['mask'], problem.initial_mask)
        full = A.C.columns(problem.domain, problem.data, problem.index, problem.floor, problem.scale)
        # Different polygon triangulations must generate the same feasible set.
        targets = problem.targets[::1024]
        actual, _, _ = A.C.classify(entry['full'], targets)
        expected, _, _ = A.C.classify(full, targets)
        assert_array_equal(actual, expected)
        self.assertAlmostEqual(entry['row']['efficiency'],
                               entry['row']['covered_percent']/entry['row']['object_area_percent'])

    def test_floor_reaction_remains_when_patch_is_empty(self):
        domain=SimpleNamespace(com=np.array([.2,-.1,.3]))
        floor=np.array([.05,.02,0.]);scale=np.array([1.,1.,1.,2.,2.,2.])
        full = A.patch_columns(domain, {}, floor, scale)
        self.assertEqual(full.shape, (5, 7))
        expected = A.C.U.floor(A.C.F.columns(floor, domain.com),scale)
        assert_array_equal(full, expected)

if __name__ == '__main__':
    unittest.main()
