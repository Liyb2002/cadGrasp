"""Audit actual neural Step3 trajectories, exports, and terminal physical evidence."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from fixed_model import FixedValueNetwork
from fixed_data import features
from train import DATA, digest

HERE=Path(__file__).parent
OUTPUT=HERE.parent/'baseline_algo/output/B/value_network_step3'

class FixedStep3Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.report=json.loads((OUTPUT/'report.json').read_text())
        cls.ckpt=torch.load(HERE/'current/best.pt',map_location='cpu',weights_only=True)
        c=cls.ckpt;cls.model=FixedValueNetwork(c['candidates'],len(c['poses']),c['width'],c['depth'])
        cls.model.load_state_dict(c['model']);cls.model.eval()
        cls.pools={p:json.loads((DATA/'pilot20_shared/inputs'/f'paths_{p}.json').read_text())['candidates'] for p in c['poses']}

    def test_complete_state_keeps_head_identity(self):
        a=features(['pose_1'],[[16]],self.ckpt['poses'])
        b=features(['pose_1'],[[184]],self.ckpt['poses'])
        np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
        self.assertEqual(int(np.count_nonzero(a[0]!=b[0])),2)

    def test_all_ten_actual_results_and_full_load_evidence(self):
        self.assertEqual(self.report['passed_groups'],10)
        self.assertIn('/value_network/baseline_algo/',self.report['mechanics_module'])
        self.assertEqual(digest(HERE/'current/best.pt'),self.report['checkpoint_sha256'])
        for r in self.report['results']:
            self.assertTrue(r['passed']);self.assertAlmostEqual(r['cost_gap'],0,places=8)
            self.assertEqual(r['load_counts'],[32768]*len(r['poses']))
            for p,g in zip(r['poses'],r['selected_indices']):
                evidence=json.loads((OUTPUT/'checks'/f'terminal_checks_{p}.json').read_text())[','.join(map(str,sorted(g)))]
                self.assertTrue(evidence['passed']);self.assertEqual(evidence['covered_count'],32768)
                self.assertTrue(evidence['no_uplift_in_same_reaction_solve'])
                artifact=OUTPUT/r['group']/f'final_contacts_{p}.npz'
                self.assertTrue(artifact.is_file())
            schedule=json.loads((OUTPUT/r['group']/'schedule.json').read_text())
            for filename,expected in schedule['artifacts'].items():self.assertEqual(digest(OUTPUT/r['group']/filename),expected)

    def test_actions_are_model_argmin_under_hard_masks(self):
        universe=self.ckpt['poses'];pindex={p:k for k,p in enumerate(universe)}
        for r in self.report['results']:
            state=[[] for _ in r['poses']]
            for step in r['trace']:
                self.assertEqual(state,step['state'])
                x,m,c=features(r['poses'],state,universe)
                with torch.no_grad():v,l=self.model(torch.tensor(x)[None],torch.tensor(m)[None],torch.tensor(c)[None])
                legal=[]
                for k,p in enumerate(r['poses']):
                    pool=self.pools[p]
                    for i,e in enumerate(pool):
                        if i in state[k] or not e['valid']:continue
                        group=state[k]+[i]
                        ds=set(pool[group[0]]['directions']);cs=set(pool[group[0]]['components'])
                        for j in group[1:]:ds &= set(pool[j]['directions']);cs &= set(pool[j]['components'])
                        if ds and cs:legal.append((k,i,pindex[p]*200+i))
                covered=[t for t in legal if l[0,t[2]]>=0]
                pick=min(covered,key=lambda t:(float(v[0,t[2]]),r['poses'][t[0]],t[1])) if covered else max(legal,key=lambda t:float(l[0,t[2]]))
                k,i,j=pick;self.assertEqual((r['poses'][k],i),(step['action']['pose'],step['action']['index']))
                self.assertAlmostEqual(float(v[0,j]),step['predicted_value'],places=5)
                state[k]=sorted(state[k]+[i])
            self.assertEqual(state,r['selected_indices'])

if __name__=='__main__':unittest.main()
