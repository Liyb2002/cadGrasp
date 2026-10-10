"""Candidates branch from one incumbent and cannot replace it with worse loss."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import tempfile
import unittest
import json
import numpy as np
from candidate_chain import candidate_chain_search

class Probe:
    def __init__(self,out,improve):
        self.out=Path(out);self.normals=np.tile([0.,0.,1.],(4,1));self.group={'poses':[2,3,4,7]}
        self.chain_seed=42;self.sample_angle=30.;self.candidates_per_round=32
        self.max_proposals=32;self.proposals=0;self.trace=[];self.selection_trace=[]
        self.exact_calls=0;self.load_bank={(0,0)};self.improve=improve
    def exact(self,d):
        self.exact_calls+=1
        return dict(directions=d.copy(),counts=[0]*4,masks=[np.zeros(1,bool)]*4,
                    loss=10.,overlap=0.,partition=0.,endpoint_overlap=[0.]*4)
    def remember(self,r):pass
    def refine(self,r,*args):return dict(r,loss=5. if self.improve else 20.)
    def actual_loss(self,r,loads):return r['loss'],[]
    def finish(self,r,*args,**kwargs):return r

class CandidateTests(unittest.TestCase):
    def test_same_origin_for_32_candidates(self):
        with tempfile.TemporaryDirectory() as out:
            p=Probe(out,True);result=candidate_chain_search(p)
            events=json.loads((Path(out)/'chain_trajectory.json').read_text())
            choices=[e for e in events if e['stage']=='direction_choice']
            self.assertEqual(len(choices),32)
            for e in choices:
                np.testing.assert_array_equal(e['before'],p.normals)
                self.assertEqual(np.count_nonzero(np.linalg.norm(np.array(e['directions'])-p.normals,axis=1)>1e-10),1)
            self.assertEqual(sum(e.get('selected',False) for e in choices),1)
            self.assertEqual(result['loss'],5.)
    def test_worse_candidates_leave_incumbent_unchanged(self):
        with tempfile.TemporaryDirectory() as out:
            p=Probe(out,False);result=candidate_chain_search(p)
            np.testing.assert_array_equal(result['directions'],p.normals)
            self.assertEqual(result['loss'],10.)

if __name__=='__main__':unittest.main()
