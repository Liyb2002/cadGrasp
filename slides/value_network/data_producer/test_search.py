"""Checks for the physical-frame and completion-count label semantics."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from search import object_directions, pair_dispersion, best_pair, prune, TerminalOracle
from prepare import compatible


def entry(directions=(0,), components=(0,), valid=True):
    return dict(valid=valid, directions=list(directions), components=list(components))


class LabelTests(unittest.TestCase):
    def test_common_paths_die_monotonically(self):
        pool=[entry((0,1)),entry((1,)),entry((0,))]
        self.assertTrue(compatible(pool,[0,1]))
        self.assertFalse(compatible(pool,[1,2]))
        self.assertFalse(compatible(pool,[0,1,2]))
        pool[2]['components']=[2]
        self.assertFalse(compatible(pool,[0,2]))

    def test_exit_comparison_uses_object_frame(self):
        r=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        t=np.eye(4);t[:3,:3]=r
        problem=SimpleNamespace(domain=SimpleNamespace(data={'frame':{'T_world_mesh':t}}))
        np.testing.assert_allclose(object_directions(problem,np.array([[0.,1,0]])),[[1,0,0]])

    def test_antipodal_exits_are_not_equivalent(self):
        a=[entry((0,))];b=[entry((0,1))]
        d,ids,angle=pair_dispersion(a,(0,),b,(0,),np.array([[np.pi,0.]]))
        self.assertEqual(d,0);self.assertEqual(ids,(0,1))
        b[0]['directions']=[0]
        d,ids,angle=pair_dispersion(a,(0,),b,(0,),np.array([[np.pi,0.]]))
        self.assertEqual(d,1);self.assertEqual(angle,180)

    def test_count_and_dispersion_come_from_same_completion(self):
        pools=[[entry((0,)),entry((1,))],[entry((0,))]]
        own=[(0,), (1,)];other=[(0,)]
        result=best_pair(own,other,0,pools,np.array([[np.pi],[0.]]),1.)
        self.assertEqual(result['completion_indices'],[[1],[0]])
        self.assertEqual(result['remaining_heads'],1)
        self.assertEqual(result['value'],1)
        self.assertIsNone(best_pair([],other,0,pools,np.array([[np.pi],[0.]]),1.))

    def test_probe_pass_cannot_replace_all_load_acceptance(self):
        pool=[dict(entry(),contact={})]
        full=np.array([[0.,0,1,0,0,0,1],[0.,0,0,0,0,0,-1]])
        problem=SimpleNamespace(supply=lambda contacts: full,targets=np.zeros((32768,6)))
        with TemporaryDirectory() as directory:
            oracle=TerminalOracle(problem,pool,Path(directory)/'checks.json')
            final=np.ones(32768,dtype=bool);final[-1]=False
            with patch('search.J.classify',side_effect=[(np.ones(96,dtype=bool),{}),(final,{})]) as classify:
                self.assertFalse(oracle.check([0]))
                self.assertEqual(classify.call_count,2)
                self.assertEqual(oracle.records['0']['failed_sample_indices'],[32767])

    def test_unknown_numeric_status_is_not_infeasibility(self):
        pool=[dict(entry(),contact={})]
        problem=SimpleNamespace(supply=lambda contacts: np.eye(7),targets=np.zeros((32768,6)))
        with TemporaryDirectory() as directory:
            oracle=TerminalOracle(problem,pool,Path(directory)/'checks.json')
            with patch('search.J.classify',side_effect=RuntimeError('unknown solver status')):
                self.assertIsNone(oracle.check([0]))
                self.assertEqual(oracle.records['0']['method'],'numerical_unresolved')

    def test_pruning_preserves_forced_head(self):
        oracle=SimpleNamespace(check=lambda xs: len(xs)>=2)
        group=prune(oracle,(0,1,2,3),2,np.random.default_rng(1))
        self.assertEqual(len(group),2);self.assertIn(2,group)


if __name__=='__main__':
    unittest.main()
