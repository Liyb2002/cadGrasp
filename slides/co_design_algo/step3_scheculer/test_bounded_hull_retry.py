"""Spawned hull lifecycle and uncertainty semantics, including large results."""
import multiprocessing
from pathlib import Path
import sys
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import bounded_hull_retry as B


def sleepy(connection):
    time.sleep(30.)


def large_result(connection):
    connection.send(('ok', np.arange(500000, dtype=np.float64)))
    connection.close()


def failed_result(connection):
    connection.send(('error', 'QhullError: diagnostic'))
    connection.close()


class BoundedHullTests(unittest.TestCase):
    def test_timeout_terminates_and_joins_child_and_is_unresolved(self):
        before = {p.pid for p in multiprocessing.active_children()}
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'unresolved: timeout'):
            B._run_worker(sleepy, (), .15)
        self.assertLess(time.monotonic()-started, 5.)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_large_result_received_before_join_without_deadlock(self):
        before = {p.pid for p in multiprocessing.active_children()}
        result = B._run_worker(large_result, (), 10.)
        np.testing.assert_array_equal(result, np.arange(500000, dtype=np.float64))
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_actual_original_hull_success(self):
        rays = np.vstack([np.eye(6), -np.eye(6)])
        result = B.bounded_cone(rays, timeout=10.)
        self.assertEqual(result.shape, (0, 6))

    def test_cli_runpy_stage_preserves_spawn_target_import(self):
        folder = B.HERE/'output/A2/pose_1/step3_scheculer'
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='hull_cli_test_', dir=folder) as directory:
            stage = Path(directory)/'probe.py'
            stage.write_text("from step3_scheculer.stage_imports import load_stage\n"
                             "import numpy as np\n"
                             "C=load_stage('score','contribution')\n"
                             "if __name__ == '__main__':\n"
                             "    assert C.W.cone(np.vstack([np.eye(6),-np.eye(6)])).shape == (0,6)\n")
            run = subprocess.run([sys.executable, str(Path(B.__file__)), '--hull-timeout',
                                  '10', str(stage), 'A2'], capture_output=True, text=True, timeout=15.)
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)

    def test_child_failure_is_unresolved_not_a_verdict(self):
        with self.assertRaisesRegex(RuntimeError, 'unresolved: QhullError'):
            B._run_worker(failed_result, (), 10.)

    def test_nonpositive_or_nonfinite_timeouts_are_rejected(self):
        for timeout in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                B._run_worker(sleepy, (), timeout)

    def test_wrapper_records_protocol_and_preserves_unresolved_status(self):
        solver = SimpleNamespace(cone=None)
        def check(*args):
            try:
                solver.cone(np.eye(7))
            except RuntimeError:
                return dict(status='unresolved', reason='RuntimeError')
            self.fail('Expected timeout')
        verification = SimpleNamespace(C=SimpleNamespace(W=solver, code_hashes=lambda: {}),
                                       continuous_check=check)
        with patch.object(B.primal_first_retry, 'install', return_value=verification), \
                patch.object(B, 'bounded_cone', side_effect=RuntimeError('unresolved: timeout')):
            wrapped = B.install(['A2'], 3.)
            result = wrapped.continuous_check(None, [object()])
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['bounded_hull_protocol']['timeout_seconds'], 3.)
        self.assertEqual(result['bounded_hull_protocol']['calls'][0]['status'], 'unresolved')
        self.assertTrue(any(path.endswith('bounded_hull_retry.py') for path in wrapped.C.code_hashes()))


if __name__ == '__main__':
    unittest.main()
