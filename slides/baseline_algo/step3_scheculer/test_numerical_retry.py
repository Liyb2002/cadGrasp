"""Original-equation retry for the recorded A1-f pose_3 HiGHS unknown state."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step3_scheculer import numerical_retry as R
class NumericalRetryTests(unittest.TestCase):
    def test_recorded_unknown_case_resolves_as_infeasible(self):
        with np.load(Path(__file__).parent/'fixtures/unknown_lp.npz') as z:
            witness, evidence = R.retry_original_equations(z['full'], z['target'])
        self.assertIsNone(witness)
        self.assertEqual(evidence['highs_status'], 2)

    def test_feasible_retry_preserves_original_equations(self):
        full = np.vstack([np.eye(6), -np.eye(6)])
        target = np.array(np.asarray([.5, -.3, 1., -.1, .2, .3]))
        witness, evidence = R.retry_original_equations(full, target)
        self.assertIsNotNone(witness)
        np.testing.assert_allclose(np.array(witness['coefficients'])@full[witness['indices']], target, atol=1e-12)

    def test_unknown_retry_is_never_counted_as_infeasible(self):
        with patch.object(R, 'linprog', return_value=SimpleNamespace(status=4, success=False, message='unknown')):
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                R.retry_original_equations(np.eye(6), np.ones(6))


if __name__ == '__main__': unittest.main()
