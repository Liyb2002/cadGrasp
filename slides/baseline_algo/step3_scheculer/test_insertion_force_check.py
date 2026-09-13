"""Distinguish a genuine all-bearing obstruction from one-sided/collinear loads."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.insertion_force_check import positive_triangle


class InsertionForceTests(unittest.TestCase):
    def test_two_dimensional_positive_zero_combination_is_exact(self):
        points=np.array([[1.,0.],[-.5,1.],[-.5,-1.]])
        proof=positive_triangle(points)
        weights=[Fraction(x) for x in proof['positive_weights_exact']]
        self.assertGreater(min(weights),0)
        self.assertEqual(sum(weights),1)
        for j in range(2):
            self.assertEqual(sum(w*Fraction(float(points[i,j])) for i,w in zip(proof['sample_indices'],weights)),0)

    def test_one_sided_loads_do_not_prove_all_bearings_impossible(self):
        self.assertIsNone(positive_triangle(np.array([[1.,-1.],[1.,1.],[2.,0.]])))

    def test_collinear_opposing_loads_allow_a_perpendicular_direction(self):
        self.assertIsNone(positive_triangle(np.array([[-1.,0.],[0.,0.],[1.,0.]])))


if __name__=='__main__':unittest.main()
