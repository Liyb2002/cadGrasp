"""Interrupted files and failed worker exits must never masquerade as success."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helper_func'))
import run_pso_campaign as campaign


class CampaignRecovery(unittest.TestCase):
    def test_interrupted_artifacts_are_preserved_before_cold_restart(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            system = root / 'system'
            old = root / 'case/step4/run'
            old.mkdir(parents=True)
            (old / 'layout.npz').write_bytes(b'partial layout')
            with patch.object(campaign, 'ROOT', root):
                campaign.archive_partial(system, 'attempt02', 'pso', 'case', [old])
            self.assertFalse(old.exists())
            saved = system / 'interrupted_attempts/attempt02/case/step4/run/layout.npz'
            self.assertEqual(saved.read_bytes(), b'partial layout')
            record = json.loads((system / 'interrupted_attempts/attempt02/records/pso/case.json').read_text())
            self.assertIn('no checkpoint reuse', record['restart'])

    def test_nonzero_worker_exit_overrides_saved_pass(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / 'result.json'
            result.write_text(json.dumps(dict(id='case', passed=True)))
            with patch.object(campaign, 'ROOT', root), patch.object(campaign.subprocess, 'call', return_value=-11):
                row = campaign.execute(root / 'worker.py',
                    (root / 'payload.json', dict(id='case')), result, root / 'worker.log')
            self.assertFalse(row['passed'])
            self.assertEqual(row['worker_exit_code'], -11)
            self.assertEqual(row['status'], 'RUNTIME_FAIL')

    def test_missing_worker_result_is_a_recorded_failure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(campaign, 'ROOT', root), patch.object(campaign.subprocess, 'call', return_value=1):
                row = campaign.execute(root / 'worker.py',
                    (root / 'payload.json', dict(case=dict(id='case'))),
                    root / 'result.json', root / 'worker.log')
            self.assertEqual(row['id'], 'case')
            self.assertFalse(row['passed'])
            self.assertTrue((root / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
