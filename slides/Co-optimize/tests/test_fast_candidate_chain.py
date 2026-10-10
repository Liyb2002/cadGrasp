"""Verify cheap guidance gradient and no exact evaluations inside candidate descent."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from fast_candidate_chain import prepare_guidance,cheap_candidates

class Model:
    def __init__(self,*args):pass
    def linearize(self,origin,frames):
        return np.ones((3,4))*.01,np.ones((3,4,2))*.002

class FastTests(unittest.TestCase):
    def probe(self):
        return SimpleNamespace(ray_normals=np.eye(3),points=np.eye(3),length=.5,clearance=None,
            mesh=SimpleNamespace(extents=np.ones(3)),normals=np.tile([0.,0.,1.],(4,1)),
            group={'poses':[2,3,4,7]},dual_weights=lambda *args:(np.array([.2,.3,.5]),{}),
            candidate_rng=np.random.default_rng(42),sample_angle=30.)
    @patch('fast_candidate_chain.NominalContactSweep',Model)
    def test_frozen_surrogate_derivative(self):
        p=self.probe();fun,*_=prepare_guidance(p,dict(directions=p.normals))
        z=np.linspace(-.02,.03,8);value,gradient=fun(z)
        for i in range(8):
            shift=np.eye(8)[i]*1e-6
            self.assertAlmostEqual(gradient[i],(fun(z+shift)[0]-fun(z-shift)[0])/2e-6,places=7)
    @patch('fast_candidate_chain.NominalContactSweep',Model)
    def test_32_candidates_without_constructing_solids(self):
        p=self.probe();rows,timing=cheap_candidates(p,dict(directions=p.normals),32)
        self.assertEqual(len(rows),32)
        for row in rows:
            choice=np.array(row['direction_choice']);endpoint=np.array(row['gradient_descent'])
            self.assertEqual(np.count_nonzero(np.linalg.norm(choice-p.normals,axis=1)>1e-10),1)
            np.testing.assert_allclose(np.linalg.norm(endpoint,axis=1),1.)
            self.assertGreaterEqual(np.min(np.sum(endpoint*p.normals,axis=1)),-1e-10)
        self.assertEqual([r['surrogate_after'] for r in rows],sorted(r['surrogate_after'] for r in rows))

if __name__=='__main__':unittest.main()
