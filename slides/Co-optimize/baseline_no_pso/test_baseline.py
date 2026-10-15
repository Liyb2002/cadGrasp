"""Check that the ablation cannot silently run continuous proposals."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
COOPT = HERE.parent
sys.path[:0] = [str(COOPT/'helper_func'), str(COOPT/'tests'), str(COOPT/'step4.2')]
import _bootstrap
from test_population32 import SmallModel, initial
from test_population_soft import sample, evidence
from controller import PopulationNoPSOSearch, baseline_joint_discrete_move


class BaselineContract(unittest.TestCase):
    def controller(self, adjusted=False):
        with patch.object(PopulationNoPSOSearch, 'local_adjustment', adjusted):
            return PopulationNoPSOSearch(SimpleNamespace(model=SmallModel(),
                frontier_width=32, population_continuous_policy='differential',
                allow_height_repair=True))

    def test_strict_mode_has_no_continuous_policy_or_height_refinement(self):
        controller = self.controller()
        self.assertEqual(controller.continuous_policy, 'none')
        self.assertFalse(controller.allow_height_repair)
        raw = object()
        with patch('population_soft.PopulationSoftSearch.checked_adjustment',
                   side_effect=AssertionError('Continuous adjustment was invoked')):
            self.assertIs(controller.checked_adjustment(raw, 0, 0), raw)

    def test_adjusted_mode_calls_original_adjustment(self):
        with patch.object(PopulationNoPSOSearch, 'local_adjustment', True):
            controller = self.controller(True)
            raw = object()
            with patch('population_soft.PopulationSoftSearch.checked_adjustment', return_value=raw) as adjust:
                self.assertIs(controller.checked_adjustment(raw, 0, 0), raw)
                adjust.assert_called_once()
            self.assertTrue(controller.allow_height_repair)

    def test_full_loop_never_calls_pso_differential_or_local_adjustment(self):
        model = SmallModel()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'case/step4/step4.2/experiment'
            out.mkdir(parents=True)
            search = SimpleNamespace(model=model, frontier_width=32, out=out, iterations=2,
                rng=np.random.default_rng(42), record=Mock(), process_snapshot=Mock(),
                population_continuous_policy='differential')
            controller = PopulationNoPSOSearch(search)
            def preview(layout, operation='parent', parent='', detail=None, result=None):
                item = sample(layout)
                item.operation = operation
                item.detail = detail or {}
                return item
            controller.preview = preview
            controller.checked = lambda item: item
            controller.certify = Mock()
            with patch('controller.freeze_population_measure'), \
                 patch('controller.pso_move', side_effect=AssertionError('PSO called')), \
                 patch('controller.differential_move', side_effect=AssertionError('DE called')), \
                 patch.object(controller, 'refine_preview', side_effect=AssertionError('Refinement called')):
                controller.solve(evidence(initial()))
            self.assertEqual(controller.branch_counts, dict(discrete=64, differential=0, pso=0, hold=0))
            for row in controller.audit:
                for event in row['events']:
                    self.assertNotIn('error', event)
                    self.assertEqual(event['adjustment_passes'], 0)

    def test_strict_joint_reseating_does_not_apply_lp_height_offsets(self):
        model = SmallModel()
        model.registered_floor_prior = SimpleNamespace(assess=lambda reuse:
            dict(status='necessary_feasible', heights_m=np.full(len(reuse), .03)))
        layout, detail = baseline_joint_discrete_move(model, initial(),
            dict.fromkeys(range(4), 1.), np.random.default_rng(42), 1)
        np.testing.assert_array_equal(layout.state_offsets, np.zeros((4, 3)))
        self.assertFalse(detail['floor_prior_height_adjustment_enabled'])


if __name__ == '__main__':
    unittest.main()
