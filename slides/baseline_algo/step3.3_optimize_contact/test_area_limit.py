"""Prevent the constant-coverage objective from shrinking a contact to a point."""
import unittest
import area_limit as L
import size_search as S


class AreaLimitTests(unittest.TestCase):
    def test_constant_coverage_stops_at_actual_surface_area_boundary(self):
        # A clipped, nonplanar patch need not have disk area pi*r*r.
        area = lambda r: r*r+r**3
        minimum = L.MIN_AREA_FRACTION*100.
        lower, report = L.radius_floor(1e-6, 1., 1e-8, area, minimum)
        evaluated = []
        def score(radius):
            evaluated.append(area(radius))
            return dict(area_m2=area(radius), total_area_m2=2+area(radius), covered_count=100)
        radius, _ = S.maximize_efficiency(score, [lower, 1.], 1., 1e-8)
        self.assertEqual(radius, lower)
        self.assertTrue(all(value>minimum for value in evaluated))
        self.assertLess(area(lower)-minimum, 1e-7)
        self.assertLessEqual(area(report['lower_infeasible_radius_m']), minimum)

    def test_component_join_keeps_feasible_side_of_area_jump(self):
        area = lambda r: .2*r if r<.6 else 1.+r
        lower, report = L.radius_floor(1e-6, 1., 1e-8, area, .5)
        self.assertGreaterEqual(lower, .6)
        self.assertLess(lower, .60000001)
        self.assertGreater(report['boundary_area_m2'], .5)

    def test_initial_contact_at_or_below_threshold_is_rejected(self):
        for area in [.49, .5]:
            with self.subTest(area=area), self.assertRaises(ValueError):
                L.radius_floor(1e-6, 1., 1e-8, lambda r: area*r*r, .5)


if __name__ == '__main__':
    unittest.main()
