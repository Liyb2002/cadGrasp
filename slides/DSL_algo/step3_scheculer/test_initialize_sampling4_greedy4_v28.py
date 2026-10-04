"""Check the sampling-to-greedy boundary and eight-head termination."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from step3_scheculer.initialize_sampling4_greedy4_v28 import Search

class HybridTests(unittest.TestCase):
    def make_search(self):
        search=object.__new__(Search)
        pool=[dict(valid=True,directions=[[0]],path_components=[1],contact=dict(
            candidate_id=f'h{i:02d}',center_m=np.array([i,0.,0.]))) for i in range(12)]
        search.pools={.01:pool};search.base=np.zeros(12,bool)
        search.catalogue=dict(vectors=[[0,0,-1]])
        search.cache=dict(arrays=dict(roadmap_labels=np.array([1])))
        search.scale=1.;search.name='synthetic';search.pose='pose_1'
        search.reused_prefix_seconds=0.;search.inputs=[]
        def classify(entries,known=None):
            mask=np.zeros(12,bool)
            for e in entries:mask[int(e['contact']['candidate_id'][1:])]=True
            return mask,{}
        search.classify=classify
        return search

    def test_four_reused_draws_then_four_deterministic_choices(self):
        search=self.make_search()
        prefix=[dict(step=i+1,selected_id=f'h{i:02d}',seconds=0.,random_draw=.5) for i in range(6)]
        search.original=dict(trajectory_results=[dict(trajectory=0,rounds=prefix,
            selected_ids=[f'h{i:02d}' for i in range(6)])])
        with tempfile.TemporaryDirectory() as tmp,patch('step3_scheculer.initialize_sampling4_greedy4_v28.I.save_contacts'):
            search.out=Path(tmp)
            selected,result=search.trajectory(0)
        self.assertEqual([e['contact']['candidate_id'] for e in selected],[f'h{i:02d}' for i in range(8)])
        self.assertEqual(result['heads'],8)
        self.assertFalse(result['passed'])
        for row in result['rounds'][4:]:
            self.assertEqual(row['selection_mode'],'maximum_coverage_greedy')
            self.assertIsNone(row['random_draw'])
            self.assertEqual(row['selected_id'],row['top10'][0]['id'])

    def test_new_trajectory_samples_only_first_four_steps(self):
        search=self.make_search();search.original=dict(trajectory_results=[])
        with tempfile.TemporaryDirectory() as tmp,patch('step3_scheculer.initialize_sampling4_greedy4_v28.I.save_contacts'):
            search.out=Path(tmp)
            _,result=search.trajectory(0)
        self.assertEqual(len(result['rounds']),8)
        self.assertTrue(all(r['random_draw'] is not None and r['selection_mode']=='top10_sampling' for r in result['rounds'][:4]))
        self.assertTrue(all(r['random_draw'] is None and r['selection_mode']=='maximum_coverage_greedy' for r in result['rounds'][4:]))

if __name__=='__main__':unittest.main()
