"""Physical synergy, exit compatibility, and monotone fallback regression."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from step3_scheculer import contacts as I
from step3_scheculer.additive_gpu_v24 import GPUClassifier
from step3_scheculer.initialize_residual_v25 import AdditiveSearch, extendable, witness_proposals as pair_proposals
from step3_scheculer.pair_scoring import J


def entry(name,center,directions=(0,),ports=(1,)):
    return dict(valid=True,directions=[list(directions)],path_components=list(ports),
                contact=dict(candidate_id=name,center_m=np.array(center,float)))


class AdditiveTests(unittest.TestCase):
    def test_two_heads_solve_load_neither_can_solve_alone(self):
        selected=[entry('seed',[0,0,0])]
        heads=[entry('a',[1,0,0]),entry('b',[-1,0,0])]
        columns=dict(seed=np.array([[0,0,0,1,0,0,0.]]),
                     a=np.array([[1,0,1,0,0,0,1.]]),
                     b=np.array([[-1,0,1,0,0,0,1.]]))
        floor=np.array([[0,0,0,0,0,0,-1.]])
        targets=np.array([[0,0,1,0,0,0.]])
        for head in heads:
            full=I.merge_columns(floor,columns['seed'],columns[head['contact']['candidate_id']])
            self.assertFalse(J.classify(full,targets)[0].all())
        proposals,evidence=pair_proposals(selected,heads,columns,floor,targets,
            np.array([False]),dict(vectors=[[0,0,-1]]),1.)
        self.assertEqual(len(proposals),1)
        self.assertEqual({h['contact']['candidate_id'] for h in proposals[0]},{'a','b'})
        full=I.merge_columns(floor,*columns.values())
        self.assertTrue(J.classify(full,targets)[0].all())
        self.assertEqual(evidence['probe_indices'],[0])

    def test_joint_pair_must_share_exit_and_distinct_centers(self):
        seed=[entry('seed',[0,0,0],(0,1))]
        a=entry('a',[1,0,0],(0,));b=entry('b',[-1,0,0],(1,))
        catalogue=dict(vectors=[[0,0,-1],[1,0,0]])
        self.assertTrue(extendable(seed,[a],catalogue,1.))
        self.assertTrue(extendable(seed,[b],catalogue,1.))
        self.assertFalse(extendable(seed,[a,b],catalogue,1.))
        self.assertFalse(extendable(seed,[entry('duplicate',[0,0,0])],catalogue,1.))

    def test_grow_keeps_original_heads_and_covered_loads(self):
        search=object.__new__(AdditiveSearch)
        old=entry('old',[0,0,0]);new=entry('new',[1,0,0])
        search.pool=[old,new];search.by_id={'old':old,'new':new}
        search.catalogue=dict(vectors=[[0,0,-1]]);search.scale=1.
        search.config=dict(max_heads=12,load_probes=8,pair_budget=128)
        search.unresolved={};search.name='synthetic';search.pose='pose_1'
        def classify(entries,known=None):
            return np.array([True,any(e['contact']['candidate_id']=='new' for e in entries)]),{}
        search.classify=classify
        with tempfile.TemporaryDirectory() as tmp:
            search.out=Path(tmp)
            selected,mask,result=search.grow(dict(selected_ids=['old'],trajectory=0))
        self.assertTrue(mask.all())
        self.assertEqual([e['contact']['candidate_id'] for e in selected],['old','new'])
        self.assertEqual(result['rounds'][0]['covered_before'],1)
        self.assertEqual(result['rounds'][0]['covered_after'],2)

    def test_optimistic_cone_prunes_impossible_seed(self):
        from types import SimpleNamespace
        search=object.__new__(AdditiveSearch)
        old=entry('old',[0,0,0]);new=entry('new',[1,0,0])
        search.pool=[old,new];search.by_id={'old':old,'new':new}
        search.columns={k:np.array([[1,0,0,0,0,0,0.]]) for k in search.by_id}
        search.floor=np.array([[0,0,0,0,0,0,-1.]])
        search.task=SimpleNamespace(targets=np.array([[0,1,0,0,0,0.]]))
        search.catalogue=dict(vectors=[[0,0,-1]]);search.scale=1.
        search.config=dict(max_heads=12,load_probes=32,pair_budget=128)
        search.unresolved={};search.name='synthetic';search.pose='pose_1'
        search.classify=lambda entries,known=None:(np.array([False]),{})
        with tempfile.TemporaryDirectory() as tmp:
            search.out=Path(tmp)
            selected,mask,result=search.grow(dict(selected_ids=['old'],trajectory=0))
        self.assertFalse(mask.all())
        self.assertEqual(len(selected),1)
        self.assertEqual(result['upper_bound']['failed_original_load'],0)
        self.assertEqual(result['status'],'finite_candidate_superset_cannot_complete_seed')

    def test_three_head_synergy_requires_bundle(self):
        selected=[entry('seed',[0,0,0])]
        heads=[entry('a',[1,0,0]),entry('b',[0,1,0]),entry('c',[-1,-1,0])]
        columns=dict(seed=np.array([[0,0,0,1,0,0,0.]]),
                     a=np.array([[1,0,1,0,0,0,1.]]),
                     b=np.array([[0,1,1,0,0,0,1.]]),
                     c=np.array([[-1,-1,1,0,0,0,1.]]))
        floor=np.array([[0,0,0,0,0,0,-1.]])
        targets=np.array([[0,0,1,0,0,0.]])
        import itertools
        for subset in itertools.combinations(heads,2):
            full=I.merge_columns(floor,columns['seed'],*[columns[h['contact']['candidate_id']] for h in subset])
            self.assertFalse(J.classify(full,targets)[0].all())
        proposals,_=pair_proposals(selected,heads,columns,floor,targets,
            np.array([False]),dict(vectors=[[0,0,-1]]),1.)
        self.assertEqual(len(proposals),1)
        self.assertEqual(len(proposals[0]),3)
        self.assertTrue(J.classify(I.merge_columns(floor,*columns.values()),targets)[0].all())
        proposals,_=pair_proposals(selected,heads,columns,floor,targets,
            np.array([False]),dict(vectors=[[0,0,-1]]),1.,max_added=2)
        self.assertEqual(proposals,[])

    def test_cuda_preserves_no_uplift_and_nonnegative_certificate(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        targets=np.array([[0,0,1,0,0,0.],[0,0,-1,0,0,0.]])
        for full in [np.array([[0,0,1,0,0,0,1.],[0,0,0,0,0,0,-1.]]),
                     np.array([[0,0,-1,0,0,0,-1.],[0,0,0,0,0,0,-1.]])]:
            np.testing.assert_array_equal(GPUClassifier(targets).classify(full)[0],J.classify(full,targets)[0])


if __name__=='__main__':
    unittest.main()
