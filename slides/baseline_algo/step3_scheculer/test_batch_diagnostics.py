"""A failed full pipeline retains floor and partial-head diagnostics."""
import sys,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import batch_cases as B
class BatchDiagnosticTests(unittest.TestCase):
    def test_numerical_stop_keeps_original_failure_and_runs_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(B,'OUTPUTS',Path(folder)):
            path=B.stage_folder(('A1-f','pose_2'),3);path.mkdir(parents=True)
            (path/'status.json').write_text(json.dumps(dict(complete=False,status='running')))
            with patch.object(B.subprocess,'run',side_effect=[SimpleNamespace(returncode=c) for c in [1,0,0]]) as run:
                record=B.run_case(('A1-f','pose_2'),2,5)
            self.assertEqual(record['returncode'],1)
            self.assertTrue(record['skipped']);self.assertEqual(record['diagnostic_returncode'],0)
            self.assertIn('--through-step',run.call_args_list[1].args[0])
            self.assertTrue(run.call_args_list[2].args[0][1].endswith('failure_visuals.py'))
            self.assertTrue((B.stage_folder(('A1-f','pose_2'),5)/'batch_run.json').is_file())

    def test_step3_only_request_does_not_create_step4_or_step5_outputs(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(B,'OUTPUTS',Path(folder)),patch.object(B.subprocess,'run',return_value=SimpleNamespace(returncode=1)) as run:
            record=B.run_case(('B','pose_1'),2,3)
            self.assertEqual(run.call_count,1);self.assertEqual(record['returncode'],1)
            self.assertFalse(B.stage_folder(('B','pose_1'),4).exists())
            self.assertFalse(B.stage_folder(('B','pose_1'),5).exists())

if __name__=='__main__':unittest.main()
