"""Keep efficiency search within the certified insertion radius interval."""
import unittest
import insertion_limit as L


class InsertionLimitTests(unittest.TestCase):
    def test_expansion_stops_before_it_erases_the_individual_direction(self):
        radius,report=L.radius_cap(1.,3.,1e-6,lambda r:r<=1.75)
        self.assertLessEqual(radius,1.75)
        self.assertGreater(radius,1.75-1e-6)
        self.assertTrue(report['limited'])

    def test_all_feasible_expansion_keeps_geometric_maximum(self):
        radius,report=L.radius_cap(1.,3.,1e-6,lambda r:True)
        self.assertEqual(radius,3.)
        self.assertFalse(report['limited'])

    def test_infeasible_initial_geometry_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            L.radius_cap(1.,3.,1e-6,lambda r:False)


if __name__=='__main__':unittest.main()
