"""A failed full pipeline retains floor and partial-head diagnostics."""
import sys,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import batch_cases as B
class BatchDiagnosticTests(unittest.TestCase):
    def test_particle_search_configuration_reaches_worker(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(B,'OUTPUTS',Path(folder)),patch.object(B.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
            record=B.run_case(('B','pose_1'),3,3,no_round_drawings=True,
                              search_mode='smc',trajectories=10,search_seed=17,
                              top_k=7,sizing_sweeps=3,sizing_budget=120)
            command=run.call_args.args[0]
            for flag,value in [('--search-mode','smc'),('--trajectories','10'),('--search-seed','17'),
                               ('--top-k','7'),('--sizing-sweeps','3'),('--sizing-budget','120')]:
                self.assertEqual(command[command.index(flag)+1],value)
            self.assertIn('--no-round-drawings',command)
            self.assertEqual(record['command'],command)
            self.assertEqual(record['returncode'],0)

    def test_numerical_stop_keeps_original_failure_and_runs_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(B,'OUTPUTS',Path(folder)):
            path=B.stage_folder(('A2','pose_2'),3);path.mkdir(parents=True)
            (path/'status.json').write_text(json.dumps(dict(complete=False,status='running')))
            with patch.object(B.subprocess,'run',side_effect=[SimpleNamespace(returncode=c) for c in [1,0,0]]) as run:
                record=B.run_case(('A2','pose_2'),2,6)
            self.assertEqual(record['returncode'],1)
            self.assertTrue(record['skipped']);self.assertEqual(record['diagnostic_returncode'],0)
            self.assertIn('--through-step',run.call_args_list[1].args[0])
            self.assertTrue(run.call_args_list[2].args[0][1].endswith('failure_visuals.py'))
            self.assertTrue((B.stage_folder(('A2','pose_2'),6)/'batch_run.json').is_file())

    def test_step3_only_request_does_not_create_step4_or_step5_outputs(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(B,'OUTPUTS',Path(folder)),patch.object(B.subprocess,'run',return_value=SimpleNamespace(returncode=1)) as run:
            record=B.run_case(('B','pose_1'),2,3)
            self.assertEqual(run.call_count,1);self.assertEqual(record['returncode'],1)
            self.assertFalse(B.stage_folder(('B','pose_1'),4).exists())
            self.assertFalse(B.stage_folder(('B','pose_1'),5).exists())

if __name__=='__main__':unittest.main()
