"""Adaptive branch comparisons cannot change their own integration nodes."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from continuous_support.stable_gradient import StableGradientModel, StableGradientSearch


class StableGradientTests(unittest.TestCase):
    def test_nested_branch_and_line_steps_use_parent_measure(self):
        model = StableGradientModel.__new__(StableGradientModel)
        with patch('continuous_support.stable_gradient.ReleasableBlockerModel.set_working_measure') as update:
            with model.hold_measure('parent'):
                with model.hold_measure('competing branch'):
                    model.set_working_measure('descended branch')
                model.set_working_measure('second branch')
            update.assert_called_once_with('parent')
            model.set_working_measure('next outer decision')
            self.assertEqual(update.call_count, 2)

    def test_failed_branch_releases_measure_hold(self):
        model = StableGradientModel.__new__(StableGradientModel)
        with patch('continuous_support.stable_gradient.ReleasableBlockerModel.set_working_measure') as update:
            with self.assertRaises(ValueError):
                with model.hold_measure('parent'):
                    raise ValueError('unresolved branch')
            model.set_working_measure('next decision')
            self.assertEqual(model.measure_hold_depth, 0)
            self.assertEqual(update.call_count, 2)

    def test_rescue_pins_measure_across_all_branch_refinements(self):
        model = StableGradientModel.__new__(StableGradientModel)
        search = StableGradientSearch.__new__(StableGradientSearch)
        search.model = model
        def branches(*args, **kwargs):
            model.set_working_measure('branch one')
            model.set_working_measure('branch two')
            return 'chosen'
        with patch('continuous_support.stable_gradient.ReleasableBlockerModel.set_working_measure') as update:
            with patch('continuous_support.stable_gradient.PairedBlockerGradientSearch.rescue', side_effect=branches):
                self.assertEqual(search.rescue('parent', []), 'chosen')
            update.assert_called_once_with('parent')


if __name__ == '__main__':
    unittest.main()
