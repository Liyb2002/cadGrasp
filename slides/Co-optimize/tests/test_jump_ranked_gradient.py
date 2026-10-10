"""Discrete escape must not change continuous descent's objective."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from continuous_support.jump_ranked_gradient import JumpRankedGradientSearch


class JumpRankedGradientTests(unittest.TestCase):
    def test_branch_gradient_uses_integral_even_inside_discrete_rescue(self):
        search = JumpRankedGradientSearch.__new__(JumpRankedGradientSearch)
        def refine(*args, **kwargs):
            self.assertFalse(search.ranking_discrete_jump)
            return 'refined'
        def rescue(*args, **kwargs):
            self.assertTrue(search.ranking_discrete_jump)
            result = search.refine('branch', 2)
            self.assertTrue(search.ranking_discrete_jump)
            return result
        with patch('continuous_support.jump_ranked_gradient.BalancedGradientSearch.refine', side_effect=refine):
            with patch('continuous_support.jump_ranked_gradient.BalancedGradientSearch.rescue', side_effect=rescue):
                self.assertEqual(search.rescue('parent', []), 'refined')
        self.assertFalse(search.ranking_discrete_jump)

    def test_exception_restores_continuous_ranking(self):
        search = JumpRankedGradientSearch.__new__(JumpRankedGradientSearch)
        with patch('continuous_support.jump_ranked_gradient.BalancedGradientSearch.rescue', side_effect=ValueError):
            with self.assertRaises(ValueError):
                search.rescue('parent', [])
        self.assertFalse(search.ranking_discrete_jump)


if __name__ == '__main__':
    unittest.main()
