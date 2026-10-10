"""A tiny continuous update must not repeat the first discrete host forever."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
from whole_search.model import Layout
from continuous_support.objective import Evaluation
from juxtapose.boundary_search import juxtapose


class StructureTests(unittest.TestCase):
    def test_host_memory_survives_a_tiny_direction_change(self):
        layout=Layout(np.repeat(np.eye(4)[None],3,axis=0),np.repeat([[0.,0.,1.]],3,axis=0),np.arange(3),(0,1,2))
        state=SimpleNamespace(locks=np.zeros((3,3,1),bool),coverage_counts=np.ones((3,1)))
        model=SimpleNamespace(native=np.repeat(np.eye(4)[None],3,axis=0),poses=['a','b','c'],mesh=trimesh.creation.box(),
            extent=1.,point_areas=np.ones(1),contact_delta=SimpleNamespace(state=lambda q:state),floor_normal=lambda q,k:np.array([0.,0.,1.]))
        def reseat(q,guest,host):
            trial=q.copy();trial.hosts[guest]=host
            t=.3*host;trial.placements[guest,:3,:3]=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1.]])
            return trial
        model.juxtapose=reseat
        class Objective:
            def __init__(self):self.model=model
            def evaluate(self,q):return Evaluation(q,np.zeros(3),[],1.,0,np.array([1.,.5,.2]),np.ones(3))
            def covered(self,r):return False
            def coverage_loss(self,r):return float(r.residual_loss.mean())
            def better(self,a,b):return False
        objective=Objective();excluded=set()
        _,a=juxtapose(objective,objective.evaluate(layout),budget=1,excluded=excluded,max_refined=1)
        moved=layout.copy();moved.directions[2]=[.001,0,np.sqrt(1-.001**2)]
        _,b=juxtapose(objective,objective.evaluate(moved),budget=1,excluded=excluded,max_refined=1)
        self.assertEqual(a['discrete_trials'][0]['host'],'b')
        self.assertEqual(b['discrete_trials'][0]['host'],'c')


if __name__=='__main__':unittest.main()
