"""Nearby exits need collective derivatives; full-shadow shortcuts preserve area."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import unittest
from types import SimpleNamespace
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from continuous_support.objective import Evaluation,CoverageObjective
from continuous_support.shared_descent import block_direction_choice
from continuous_support.surface import prism_slices as original
from continuous_support.shared_surface import prism_slices as accelerated
from whole_search.model import Layout


class SharedTests(unittest.TestCase):
    def update(self,angles,floors=None):
        floors=np.repeat([[0.,0.,1.]],2,axis=0) if floors is None else floors
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),
            np.array([[np.sin(t),0.,np.cos(t)] for t in angles]),np.arange(2),(0,1))
        model=SimpleNamespace(poses=['a','b'],extent=1.,floor_normal=lambda q,k:floors[k])
        class Objective:
            covered=staticmethod(CoverageObjective.covered)
            def __init__(self):self.model=model
            def evaluate(self,q):
                theta=np.arctan2(q.directions[:,0],q.directions[:,2])
                return Evaluation(q,np.zeros(2),[dict(pose=k,contact_normal_signature='fixed') for k in model.poses],
                                  1.,0,(theta-.1)**2,np.ones(2))
            def value(self,r,phase):return float(r.residual_loss.mean())
            def acceptable(self,a,b,phase):return self.value(b,phase)<self.value(a,phase)
        objective=Objective();before=objective.evaluate(layout)
        after,record=block_direction_choice(objective,before,None)
        if record['accepted']:self.assertLess(objective.value(after,'coverage'),objective.value(before,'coverage'))
        np.testing.assert_allclose(np.linalg.norm(after.layout.directions,axis=1),1.,atol=1e-12)
        self.assertGreaterEqual(float(np.min(np.einsum('ij,ij->i',after.layout.directions,floors))),-1e-12)
        return record

    def test_nearby_but_distinct_exits_share_two_coordinates(self):
        record=self.update([-.003,.003])
        self.assertTrue(record['accepted']);self.assertEqual(record['coordinate_groups'],[['a','b']])
        self.assertEqual(len(record['gradient']),2)
        self.assertTrue(record['derivative_check']['local_derivative_consistent'])

    def test_distant_exits_keep_independent_coordinates(self):
        record=self.update([-.1,.1])
        self.assertEqual(record['coordinate_groups'],[['a'],['b']])
        self.assertEqual(len(record['gradient']),4)

    def test_nearby_group_respects_each_original_floor(self):
        self.update([.003,-.003],np.array([[1.,0.,0.],[-1.,0.,0.]]))

    def test_whole_triangle_shortcut_preserves_union(self):
        triangle=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        box=np.array([[1.,0,0,-2],[-1.,0,0,-2],[0,1.,0,-2],[0,-1.,0,-2],[0,0,1.,-.1],[0,0,-1.,-.1]])
        partial=box.copy();partial[0,3]=-.4
        planes=np.array([partial,box]);counts=np.array([6,6]);shapes=[]
        for operation in [original,accelerated]:
            pieces,sizes=operation(triangle,np.array([0.,0,1.]),planes,counts,1e-8)
            shapes.append(unary_union([Polygon(pieces[k,:size,:2]) for k,size in enumerate(sizes) if size>=3]))
        self.assertLess(shapes[0].symmetric_difference(shapes[1]).area,1e-14)


if __name__=='__main__':unittest.main()
