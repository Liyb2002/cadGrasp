"""Proof reordering must preserve certificates and the original fallback."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import primal_first_retry as P, enclosure as E, contacts as I


class PrimalFirstTests(unittest.TestCase):
    def setUp(self):
        self.original = Mock(return_value={'status': 'counterexample'})
        self.verification = SimpleNamespace(continuous_check=self.original,
                                           C=SimpleNamespace(code_hashes=lambda: {}))
        self.problem = SimpleNamespace(name='test', supply=Mock(return_value=np.eye(7)))

    def test_success_saves_auditable_certificate_without_dual_hull(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(P.continuous_retry, 'install', return_value=self.verification), \
                patch.object(E, 'contain', return_value=({'status': 'verified'}, {'bases': np.eye(7)})), \
                patch.object(I, 'ROOT', Path(directory)):
            verification = P.install(['test'])
            path = Path(directory)/'certificate.npz'
            result = verification.continuous_check(self.problem, [object()], path)
            self.assertEqual(result['status'], 'verified')
            self.assertEqual(result['certificate_sha256'], I.sha256(path))
            self.assertEqual(result['enclosure_code_sha256'], I.sha256(E.__file__))
            self.original.assert_not_called()

    def test_inconclusive_outer_approximation_uses_original_verdict(self):
        with patch.object(P.continuous_retry, 'install', return_value=self.verification), \
                patch.object(E, 'contain', return_value=({'status': 'not_certified'}, None)) as contain:
            verification = P.install(['test'])
            contacts = [object()]
            result = verification.continuous_check(self.problem, contacts, Path('unused.npz'))
            self.assertEqual(contain.call_count, 3)
            self.assertEqual(result, self.original.return_value)
            self.original.assert_called_once_with(self.problem, contacts, Path('unused.npz'))

    def test_no_heads_preserves_original_no_design_verdict(self):
        with patch.object(P.continuous_retry, 'install', return_value=self.verification), \
                patch.object(E, 'contain') as contain:
            verification = P.install(['test'])
            verification.continuous_check(self.problem, [], Path('unused.npz'))
            contain.assert_not_called()
            self.original.assert_called_once()


if __name__ == '__main__':
    unittest.main()
