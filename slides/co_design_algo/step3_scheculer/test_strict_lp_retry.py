"""Recovery preserves augmented equilibrium equations and unresolved verdicts."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import strict_lp_retry as R


class StrictRetryTests(unittest.TestCase):
    def test_seven_equations_are_preserved(self):
        full=np.vstack([np.eye(7),-np.eye(7)])
        target=np.array([.5,-.3,1.,-.1,.2,.3,1.])
        witness,evidence=R.recover(full,target)
        np.testing.assert_allclose(np.array(witness['coefficients'])@full[witness['indices']],target,atol=2e-9)
        self.assertEqual(evidence['status'],'feasible')

    def test_infeasibility_requires_explicit_solver_verdicts(self):
        witness,evidence=R.recover(np.eye(7),-np.ones(7))
        self.assertIsNone(witness)
        self.assertEqual(evidence['status'],'infeasible_numeric')
        self.assertGreaterEqual(evidence['agreeing_infeasible_formulations'],2)

    def test_unknown_is_not_infeasible(self):
        with patch.object(R,'linprog',return_value=SimpleNamespace(status=4,success=False)):
            with self.assertRaisesRegex(RuntimeError,'unresolved'):
                R.recover(np.eye(7),-np.ones(7))

    def test_unverified_primal_is_not_overridden_by_infeasible_retries(self):
        calls=[SimpleNamespace(status=0,success=True,x=np.zeros(7))]+[SimpleNamespace(status=2,success=False)]*7
        with patch.object(R,'linprog',side_effect=calls):
            with self.assertRaisesRegex(RuntimeError,'unresolved'):
                R.recover(np.eye(7),-np.ones(7))


if __name__=='__main__':unittest.main()
