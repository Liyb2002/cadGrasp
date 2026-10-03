"""Verify group-held-out experiment boundaries and frozen checkpoints."""
import json
from pathlib import Path
import unittest
from fixed_data import collect
from train import DATA,digest

HERE=Path(__file__).parent
ROOT=HERE/'transfer'

class TransferChecks(unittest.TestCase):
    def setUp(self):self.plan=json.loads((ROOT/'plan.json').read_text())
    def test_five_group_split_is_novel_and_pose_covered(self):
        train={tuple(ps) for _,ps in self.plan['train']};test={tuple(ps) for _,ps in self.plan['test']};old={tuple(ps) for _,ps in self.plan['original']}
        self.assertEqual(len(train),5);self.assertEqual(len(test),5)
        self.assertFalse(train&test);self.assertFalse((train|test)&old)
        self.assertEqual({p for ps in train for p in ps},{p for ps in test for p in ps})
    def test_loader_only_reads_training_groups(self):
        poses,arrays,keys,records,sources=collect(ROOT/'train_on_policy',ROOT/'train_config.json')
        allowed={tuple(ps) for _,ps in self.plan['train']}
        self.assertTrue(records)
        self.assertTrue(all(tuple(r['poses']) in allowed for r in records.values()))
        self.assertTrue(all(tuple(json.loads(Path(p).read_text())['poses']) in allowed for p in sources))
        self.assertTrue(all(not (Path(self.plan['state_root'])/name).exists() for name,_ in self.plan['test']))
    def test_original_checkpoint_was_not_changed(self):
        self.assertEqual(digest(HERE/'current/best.pt'),self.plan['original_checkpoint_sha256'])
        frozen=json.loads((ROOT/'frozen/report.json').read_text())
        self.assertTrue(frozen['checkpoint_unchanged'])
        self.assertEqual(frozen['checkpoint_sha256'],self.plan['original_checkpoint_sha256'])
    def test_heldout_terminal_evidence_and_checkpoint_freeze(self):
        e=json.loads((ROOT/'experiment.json').read_text())
        for report_path in [e['test_report'],e['original_ten_test_report'],e['training_report']]:
            report=json.loads(Path(report_path).read_text())
            self.assertEqual(report['checkpoint_sha256'],e['checkpoint_sha256'])
            for r in report['results']:
                if not r['passed']:continue
                for p,g in zip(r['poses'],r['selected_indices']):
                    record=json.loads((ROOT/'mechanics'/f'terminal_checks_{p}.json').read_text())[','.join(map(str,sorted(g)))]
                    self.assertTrue(record['passed'])
                    self.assertEqual(record['covered_count'],32768)
                    self.assertTrue(record['no_uplift_in_same_reaction_solve'])
        self.assertEqual(digest(Path(e['checkpoint'])),e['checkpoint_sha256'])
    def test_five_group_model_started_from_random_weights(self):
        first=json.loads((ROOT/'fit_00/report.json').read_text())
        self.assertIsNone(first['args']['resume'])
        allowed={tuple(ps) for _,ps in self.plan['train']}
        self.assertTrue(all(tuple(json.loads(Path(p).read_text())['poses']) in allowed for p in first['source_hashes']))

if __name__=='__main__':unittest.main()
