"""Regression checks for unrestricted exits, weighted top10 and GPU LP certificates."""
import unittest
import numpy as np
import trimesh
from step3_scheculer.initialize_gpu_v12 import top10,catalogue,GPUClassifier
from step3_scheculer.pair_scoring import J

class InitializationTests(unittest.TestCase):
    def test_top10_really_exposes_ten_choices(self):
        rows=[dict(id=str(i),covered=i+10) for i in range(12)]
        ranked,p=top10(rows,0)
        self.assertEqual(len(ranked),10)
        self.assertEqual([r['covered'] for r in ranked],list(range(21,11,-1)))
        self.assertTrue(np.all(p>0));self.assertAlmostEqual(p.sum(),1)
        _,p=top10([dict(id=str(i),covered=0) for i in range(12)],0)
        np.testing.assert_allclose(p,.1)

    def test_exit_menu_not_upward_only(self):
        c=catalogue(trimesh.creation.box());v=np.asarray(c['vectors'])
        self.assertTrue(np.any(v[:,2]<-.99))
        self.assertTrue(np.any(np.abs(v[:,2])<1e-12))
        self.assertTrue(np.any((v[:,2]<-.1)&(v[:,2]>-.9)))
        self.assertTrue(np.all(v[:,2]<=1e-10))

    def test_cuda_matches_original_lp_including_no_uplift(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        # Six physical rows plus the no-uplift row. Without shared slack,
        # downward-only heads cannot solve an otherwise feasible demand.
        rng=np.random.default_rng(9)
        full=np.r_[np.c_[np.eye(6),np.ones(6)],np.array([[0,0,0,0,0,0,-1.]])]
        targets=rng.uniform(.1,2,(60,6));targets[::4,0]*=-1
        expected,_=J.classify(full,targets)
        got,_=GPUClassifier(targets).classify(full)
        np.testing.assert_array_equal(got,expected)
        upward=np.array([[0,0,1,0,0,0,1],[0,0,0,0,0,0,-1.]])
        downward=np.array([[0,0,-1,0,0,0,-1],[0,0,0,0,0,0,-1.]])
        target=np.array([[0,0,1,0,0,0],[0,0,-1,0,0,0]])
        np.testing.assert_array_equal(GPUClassifier(target).classify(upward)[0],[True,False])
        np.testing.assert_array_equal(GPUClassifier(target).classify(downward)[0],[False,False])

if __name__=='__main__':unittest.main()
