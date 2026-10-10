"""A small gain cannot hide a larger gradient scale from comparison."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from continuous_support.balanced_gradient import BalancedGradientSearch


class BalancedGradientTests(unittest.TestCase):
    def test_all_gradient_tools_and_scales_compete_before_sampling(self):
        rows = [('direction-gradient', 'small d', dict(step_degrees=.5)),
                ('direction-gradient', 'large d', dict(step_degrees=8.)),
                ('translation-gradient', 't', dict(step_m=.003)),
                ('direction-coherent-sample', 'sample', {})]
        groups = BalancedGradientSearch.proposal_groups(rows)
        self.assertEqual([r[1] for r in groups[0][1]], ['small d', 'large d', 't'])
        self.assertEqual(groups[1][1][0][1], 'sample')


if __name__ == '__main__':
    unittest.main()
