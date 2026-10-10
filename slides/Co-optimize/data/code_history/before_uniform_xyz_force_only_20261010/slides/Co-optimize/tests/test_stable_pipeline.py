"""Conditioning is a bounded fallback for this run's feasible own states."""
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'step4.2'))
from stable_pipeline import needs_conditioning
import stable_pipeline


class StablePipelineTests(unittest.TestCase):
    def test_no_extra_sampling_after_success(self):
        self.assertFalse(needs_conditioning({'passed': True}, Path('unused')))

    def test_unresolved_loads_do_not_enter_final_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'sampled_layout.npz').touch()
            report = out / 'search_report.json'
            report.write_text(json.dumps({'counts': {'p1': 32768, 'p2': 32767}}))
            self.assertFalse(needs_conditioning({'passed': False}, out))
            report.write_text(json.dumps({'counts': {'p1': 32768, 'p2': 32768}}))
            self.assertTrue(needs_conditioning({'passed': False}, out))

    def test_feasible_export_failure_uses_its_own_state_and_one_bounded_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'output/B'
            def simulate(main, arguments):
                name = arguments[arguments.index('--output-name')+1]
                data = base / 'data' / name
                data.mkdir(parents=True)
                conditioning = main is stable_pipeline.run_stable_conditioning.main
                if conditioning:
                    self.assertEqual(arguments[arguments.index('--resume-experiment')+1], 'new_run')
                (data / 'batch.json').write_text(json.dumps({'results': [
                    {'id': 'p1+p2+p3', 'passed': conditioning}]}))
                if not conditioning:
                    out = base / 'p1+p2+p3/step4/step4.2' / name
                    out.mkdir(parents=True)
                    (out / 'sampled_layout.npz').touch()
                    (out / 'search_report.json').write_text(json.dumps({'counts': {'p1': 32768, 'p2': 32768, 'p3': 32768}}))
            with patch.object(stable_pipeline, 'HERE', Path(directory)), \
                 patch.object(stable_pipeline, 'invoke', side_effect=simulate) as invoke, \
                 patch.object(sys, 'argv', ['run.py', '--sets', 'p1+p2+p3', '--output-name', 'new_run']):
                stable_pipeline.main()
            self.assertEqual(invoke.call_count, 2)
            report=json.loads((base / 'data/new_run/pipeline.json').read_text())
            self.assertEqual(report['passed'],1)
            self.assertEqual(report['base_budget_passes'],0)
            self.assertEqual(report['conditional_passes'],1)

    def test_load_failure_repairs_own_state_with_fast_gradient_and_keeps_successful_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'output/B'
            def simulate(main, arguments):
                name = arguments[arguments.index('--output-name')+1]
                data = base / 'data' / name
                data.mkdir(parents=True)
                repair = main is stable_pipeline.run_fast_state_seat_gradient.main
                if repair:
                    self.assertEqual(arguments[arguments.index('--sets')+1:arguments.index('--jobs')], ['failed'])
                    self.assertEqual(arguments[arguments.index('--resume-experiment')+1], 'new_run')
                    self.assertEqual(arguments[arguments.index('--iterations')+1], '8')
                    rows = [{'id': 'failed', 'passed': True}]
                else:
                    rows = [{'id': 'passed', 'passed': True}, {'id': 'failed', 'passed': False}]
                    out = base / 'failed/step4/step4.2' / name
                    out.mkdir(parents=True)
                    (out / 'sampled_layout.npz').touch()
                    (out / 'search_report.json').write_text(json.dumps({'counts': {'p1': 32767}}))
                (data / 'batch.json').write_text(json.dumps({'results': rows}))
            with patch.object(stable_pipeline, 'HERE', Path(directory)), \
                 patch.object(stable_pipeline, 'invoke', side_effect=simulate) as invoke, \
                 patch.object(sys, 'argv', ['run.py', '--sets', 'passed', 'failed', '--output-name', 'new_run']):
                stable_pipeline.main()
            self.assertEqual(invoke.call_count, 2)
            report=json.loads((base / 'data/new_run/pipeline.json').read_text())
            self.assertEqual(report['passed'],2)
            self.assertEqual(report['base_budget_passes'],1)
            self.assertEqual(report['gradient_repair_passes'],1)
            self.assertEqual(report['conditioning_guests'],[])
            self.assertEqual(report['selected']['passed']['experiment'],'new_run')
            self.assertEqual(report['selected']['failed']['experiment'],'new_run_repair')


if __name__ == '__main__':
    unittest.main()
