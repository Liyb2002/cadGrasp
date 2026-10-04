"""Bound both ordinary LPs and recovery LPs without changing tolerances."""
import contextlib
import unittest
from unittest.mock import patch
from step3_scheculer import initialize_sampling4_greedy4_v29 as M

class RetryTests(unittest.TestCase):
    def test_limit_applies_to_wrench_and_recovery_and_restores_modules(self):
        original_wrench=M.C.W.linprog
        with patch.object(M.strict_lp_retry,'linprog') as lp,patch.object(M,'original_recovery',return_value=contextlib.nullcontext()):
            with M.numerical_recovery(None,{}):
                M.C.W.linprog([0],options=dict(primal_feasibility_tolerance=1e-9))
                M.strict_lp_retry.linprog([0],options=dict(dual_feasibility_tolerance=1e-10))
            self.assertIs(M.C.W.linprog,original_wrench)
            for call in lp.call_args_list:
                self.assertEqual(call.kwargs['options']['time_limit'],1.)
            self.assertEqual(lp.call_args_list[0].kwargs['options']['primal_feasibility_tolerance'],1e-9)
            self.assertEqual(lp.call_args_list[1].kwargs['options']['dual_feasibility_tolerance'],1e-10)

if __name__=='__main__':unittest.main()
