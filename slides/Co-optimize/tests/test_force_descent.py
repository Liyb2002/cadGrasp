"""Reaction-reoptimized envelope derivative and candidate-local chart tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from force_descent import prepare_force_objective

class Model:
    def __init__(self,*args):pass
    def linearize(self,origin,frames):
        return np.full((2,len(origin)),.01),np.full((2,len(origin),2),.002)

class ForceDerivativeTests(unittest.TestCase):
    @patch('force_descent.NominalContactSweep',Model)
    def test_actual_material_drives_load_selection(self):
        actual=dict(marker='actual')
        seen=[]
        def choose(state):
            seen.append(state)
            return [(0,0)]
        search=SimpleNamespace(states=[(SimpleNamespace(targets=np.array([[1.,2.,0.,0.,0.,0.,0.]])),None)],
            floors=[np.empty((0,7))],rays=[np.eye(7)[:2]],ray_normals=np.eye(3)[:2],points=np.eye(3)[:2],
            clearance=None,length=.5,mesh=SimpleNamespace(extents=np.ones(3)),choose_loads=choose,
            # Real material has a different generator set than potential
            # contacts, so these coefficients cannot index potential rays.
            projection=lambda *args:dict(loss=.2,coefficients=np.ones(20)),
            dual_weights=lambda state,*args:(np.array([.5,.5]),{}))
        fun,_,info=prepare_force_objective(search,dict(active=np.ones(2,bool)),np.array([[0.,0.,1.]]),physical_state=actual)
        self.assertIs(seen[0],actual)
        self.assertTrue(np.isfinite(fun(np.zeros(2))[0]))
        self.assertEqual(info['critical_loads'],[(0,0)])

    @patch('force_descent.NominalContactSweep',Model)
    def test_envelope_derivative_reoptimizes_reactions(self):
        search=SimpleNamespace(states=[(SimpleNamespace(targets=np.array([[1.,2.,0.,0.,0.,0.,0.]])),None)],
            floors=[np.empty((0,7))],rays=[np.eye(7)[:2]],ray_normals=np.eye(3)[:2],points=np.eye(3)[:2],
            clearance=None,length=.5,mesh=SimpleNamespace(extents=np.ones(3)),
            choose_loads=lambda state:[(0,0)],
            projection=lambda *args:dict(loss=.2,coefficients=np.array([1.,2.])),
            dual_weights=lambda *args:(np.array([.5,.5]),{}))
        origin=np.array([[.2,.1,.97],[.1,.3,.95]]);origin/=np.linalg.norm(origin,axis=1)[:,None]
        fun,frames,info=prepare_force_objective(search,dict(active=np.ones(2,bool)),origin)
        z=np.array([.02,-.01,.03,-.02]);value,gradient=fun(z)
        self.assertEqual(gradient.shape,(4,))
        for i in range(4):
            shift=np.eye(4)[i]*1e-6
            finite=(fun(z+shift)[0]-fun(z-shift)[0])/2e-6
            self.assertAlmostEqual(gradient[i],finite,places=7)

if __name__=='__main__':unittest.main()
