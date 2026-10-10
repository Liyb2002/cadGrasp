"""A tiny accepted move must not indefinitely suppress discrete repairs."""
import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'helper_func'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'step4.2'))
import _bootstrap
import numpy as np
import optimizer_v3
from whole_search.model import Layout
from continuous_support.objective import Evaluation


class OuterLoopTests(unittest.TestCase):
    def run_search(self, amount):
        layout = Layout(np.eye(4)[None], np.array([[0., 0., 1.]]), np.array([0]), (0,))
        class Objective:
            evaluations = lp_calls = 0
            seconds = 0.
            def evaluate(self, q):
                self.evaluations += 1
                return Evaluation(q, np.zeros(1), [], 1., 0,
                                  np.array([1-q.placements[0, 0, 3]]), np.ones(1))
            def coverage_loss(self, r): return float(r.residual_loss.mean())
            def covered(self, r): return False
            def timing(self): return {}
        objective = Objective()
        def direction(obj, current, **kwargs):
            trial = current.layout.copy(); trial.placements[0, 0, 3] += amount
            return (obj.evaluate(trial) if amount else current), dict(operation='direction', accepted=bool(amount))
        def translation(obj, current, **kwargs):
            return current, dict(operation='translation', accepted=False)
        def jump(obj, current, **kwargs):
            return current, dict(operation='juxtapose', accepted=False)
        options = dict(iterations=2, jumps=2, jump_trials=2, jump_refine=1, backtracks=1,
                       minimum_progress=.005, quadrature_level=1, contact_depth=1)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(optimizer_v3, 'CoverageObjective', return_value=objective), \
             patch.object(optimizer_v3, 'direction_choice', side_effect=direction), \
             patch.object(optimizer_v3, 'translation', side_effect=translation), \
             patch.object(optimizer_v3, 'juxtapose', side_effect=jump) as jumps:
            optimizer_v3.optimize(SimpleNamespace(), layout, Path(directory), options)
            return jumps.call_count

    def test_small_accepted_moves_allow_jumps(self):
        self.assertEqual(self.run_search(.001), 2)

    def test_substantial_progress_keeps_continuous_search(self):
        self.assertEqual(self.run_search(.2), 0)

    def test_rejected_first_jump_does_not_end_discrete_budget(self):
        self.assertEqual(self.run_search(0.), 2)


if __name__ == '__main__': unittest.main()
