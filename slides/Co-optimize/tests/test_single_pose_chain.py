"""Check single-pose transitions, independent cycle order and failure recovery."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import tempfile
import unittest
import json
import numpy as np
from single_pose_chain import sample_one, single_pose_search


class Probe:
    def __init__(self, out, fail=False):
        self.out=Path(out); self.normals=np.tile([0.,0.,1.],(4,1))
        self.group={'poses':[2,3,4,7]}; self.chain_seed=42; self.sample_angle=30.
        self.max_proposals=8; self.proposals=0; self.trace=[]; self.selection_trace=[]
        self.best=None; self.calls=0; self.fail=fail; self.refined=[]
    def exact(self, d):
        self.calls+=1
        if self.fail and self.calls==2: raise ValueError('invalid construction')
        return dict(directions=d.copy(), counts=[0]*4, masks=[np.zeros(1,bool)]*4,
                    overlap=0., partition=0., endpoint_overlap=[0.]*4)
    def remember(self, result):
        if self.best is None: self.best=result
    def refine(self, result, iterations, label):
        self.refined.append(result['directions'].copy())
        d=result['directions'].copy(); d[:,0]+=.001; d/=np.linalg.norm(d,axis=1)[:,None]
        self.trace.extend([{'accepted':True},{'accepted':False}])
        return dict(result,directions=d)
    def finish(self, result, *args, **kwargs): return self.report_extra


class ChainTests(unittest.TestCase):
    def test_only_selected_pose_moves_and_stays_in_hemisphere(self):
        rng=np.random.default_rng(7); normals=np.tile([0.,0.,1.],(4,1)); d=normals.copy()
        for _ in range(100):
            candidate, axis, angle=sample_one(d,normals,2,rng)
            np.testing.assert_array_equal(candidate[[0,1,3]],d[[0,1,3]])
            self.assertGreaterEqual(candidate[2]@normals[2],0)
            np.testing.assert_allclose(np.linalg.norm(candidate,axis=1),1.)
            self.assertTrue(5.<=angle<=30.); self.assertIn(axis,range(4)); d=candidate
    def test_cycle_and_continuation_and_counts(self):
        with tempfile.TemporaryDirectory() as out:
            probe=Probe(out); single_pose_search(probe)
            events=json.loads((Path(out)/'chain_trajectory.json').read_text())
            samples=[e for e in events if e['stage']=='sample']
            for start in (0,4): self.assertEqual({e['pose'] for e in samples[start:start+4]},{2,3,4,7})
            for i,e in enumerate(samples):
                before=np.array(e['before']); after=np.array(e['directions'])
                self.assertEqual(np.count_nonzero(np.linalg.norm(after-before,axis=1)>1e-10),1)
                if i: np.testing.assert_allclose(before,events[2*i]['directions'])
                self.assertEqual(e['descent_iterations'],2)
                self.assertEqual(e['accepted_descent_iterations'],1)
            self.assertEqual(len(probe.refined),8)
    def test_refinement_failure_keeps_valid_sample(self):
        with tempfile.TemporaryDirectory() as out:
            probe=Probe(out); probe.max_proposals=1
            def fail(*args): raise ValueError('descent failed')
            probe.refine=fail
            single_pose_search(probe)
            events=json.loads((Path(out)/'chain_trajectory.json').read_text())
            self.assertIn('error',events[1])
            np.testing.assert_array_equal(events[1]['directions'],events[2]['directions'])

    def test_failed_construction_preserves_chain(self):
        with tempfile.TemporaryDirectory() as out:
            probe=Probe(out,fail=True); single_pose_search(probe)
            events=json.loads((Path(out)/'chain_trajectory.json').read_text())
            self.assertIn('error',events[1])
            np.testing.assert_array_equal(events[0]['directions'],events[2]['directions'])
            self.assertEqual(events[1]['accepted_descent_iterations'],0)

if __name__=='__main__': unittest.main()
