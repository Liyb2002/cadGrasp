"""Every raw/descent candidate is checked and true regressions are rolled back."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import json,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from force_candidate_chain import force_candidate_search

class ChainRollbackTests(unittest.TestCase):
    def test_all_32_pairs_checked_and_worse_descent_rejected(self):
        calls=[]
        def state(search,model,d,previous=None):
            calls.append(d.copy());count=1 if len(calls)==1 else (3 if len(calls)%2==0 else 2)
            masks=[np.arange(10)<count]*2
            return dict(directions=d.copy(),counts=[count]*2,masks=masks,locks=np.zeros((2,1),bool),active=np.ones(1,bool))
        with tempfile.TemporaryDirectory() as out:
            search=SimpleNamespace(out=Path(out),chain_seed=42,mesh=None,points=None,ray_normals=None,length=1.,
                normals=np.tile([0.,0.,1.],(2,1)),sample_angle=30.,max_proposals=32,proposals=0,
                candidates_per_round=32,group={'poses':[1,2],'id':'probe'},
                states=[(SimpleNamespace(targets=np.zeros((10,7)),inputs=[]),None)]*2,
                remember=lambda *args:None,seed_path=None,contact_path=None,additional_code=[])
            with patch('force_candidate_chain.ContactLocks'),patch('force_candidate_chain.contact_state',side_effect=state),\
                 patch('force_candidate_chain.force_descent',side_effect=lambda search,raw,d:(d.copy(),{})),\
                 patch('force_candidate_chain.provenance',return_value={}):
                report=force_candidate_search(search)
            rows=json.loads((Path(out)/'chain_trajectory.json').read_text())[0]['candidates']
            self.assertEqual(len(calls),65)
            self.assertEqual(report['final_counts'],[3,3])
            self.assertTrue(all(r['descent_worsened'] for r in rows))
            self.assertTrue(all(r['kept_counts']==[3,3] for r in rows))
            calls.clear();search.proposals=0;search.descent_enabled=False
            with patch('force_candidate_chain.ContactLocks'),patch('force_candidate_chain.contact_state',side_effect=state),\
                 patch('force_candidate_chain.force_descent') as descent,\
                 patch('force_candidate_chain.provenance',return_value={}):
                report=force_candidate_search(search)
                descent.assert_not_called()
            self.assertEqual(len(calls),33)
            self.assertFalse(report['gradient_descent_enabled'])
            self.assertEqual(report['candidates'],32)


if __name__=='__main__':unittest.main()
