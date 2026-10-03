"""Distinguish completed failures, interrupted stages and joint-command timings."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.timing import StageTimings


class TimingTests(unittest.TestCase):
    def test_stage_totals_and_joint_scope_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name/'timing.json' for name in ('A','B')]
            timing = StageTimings(paths, dict(resume=False))
            result = timing.call('step3_scheculer/scheduler.py', ['A','B'],
                                 lambda: SimpleNamespace(returncode=2))
            self.assertEqual(result.returncode, 2)
            timing.finish(returncode=2)
            a, b = [json.loads(path.read_text()) for path in paths]
            self.assertEqual(a,b)
            self.assertTrue(a['complete'])
            self.assertEqual(a['step_seconds']['3'], a['stages'][0]['elapsed_seconds'])
            self.assertEqual(a['stages'][0]['objects'], ['A','B'])
            self.assertGreaterEqual(a['outside_stage_seconds'], 0.)

    def test_exception_retains_completed_and_failed_stage_durations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'timing.json'
            timing = StageTimings([path], {})
            timing.call('step1/needs.py', ['A'], lambda: SimpleNamespace(returncode=0))
            def fail():
                raise RuntimeError('worker failed')
            try:
                timing.call('step2_local_support/insertion_directions.py', ['A'], fail)
            except RuntimeError as error:
                timing.finish(error=error)
            record = json.loads(path.read_text())
            self.assertFalse(record['complete'])
            self.assertEqual(len(record['stages']), 2)
            self.assertFalse(record['stages'][1]['complete'])
            self.assertIn('worker failed', record['error'])
            self.assertIn('2', record['step_seconds'])


if __name__ == '__main__':
    unittest.main()
