"""Check actual dataset separation and trained-checkpoint inference consistency."""
import unittest
import json
from pathlib import Path
import numpy as np
import torch
from train import load_data,DATA
from model import ValueNetwork
from predict import predict

class TrainingChecks(unittest.TestCase):
    def test_whole_state_split_and_duplicate_state(self):
        d=load_data(20261004)
        partitions={s:set(d['state_ids'][d['splits']==s]) for s in ['train','validation','test']}
        self.assertFalse(partitions['train']&partitions['test']);self.assertFalse(partitions['train']&partitions['validation']);self.assertFalse(partitions['validation']&partitions['test'])
        self.assertEqual(len(d['labels']),30868)
        for name in ['pose1+3','pose1+3copied']:
            record=json.loads((DATA/name/'pilot20/state_000.json').read_text());self.assertEqual(record['selected_indices'],[[],[]])
        # Canonical feature state is encoded exactly once across repeated groups.
        mask=(d['membership'][:,d['poses'].index('pose_1')]==1)&(d['membership'][:,d['poses'].index('pose_3')]==1)&(d['membership'].sum(axis=1)==2)&(d['counts'].sum(axis=1)==0)
        self.assertEqual(int(mask.sum()),1)
    def test_checkpoint_matches_saved_predictions(self):
        out=Path(__file__).parent/'output';c=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        m=ValueNetwork(c['candidates'],len(c['poses']));m.load_state_dict(c['model']);m.eval()
        d=load_data(c['seed']);saved=np.load(out/'predictions.npz');ii=np.flatnonzero(saved['splits']=='test')[:100];uid=d['state_ids'][ii]
        with torch.no_grad():p=m(torch.tensor(d['states'][uid]),torch.tensor(d['membership'][uid]),torch.tensor(d['counts'][uid]),torch.tensor(d['actions'][ii])).numpy()
        np.testing.assert_allclose(p,saved['prediction'][ii],rtol=1e-5,atol=1e-5)
        rows=predict(DATA/'pose1+3/pilot20/state_000.json',out/'best.pt')
        self.assertTrue(rows);self.assertTrue(all(np.isfinite(r['predicted_value']) and r['predicted_value']>=0 for r in rows))
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
if __name__=='__main__':unittest.main()
