"""Regression: global covering must retain physics-valued contact-normal axes."""

import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
from physics_guided_concurrent_search import concurrent_search


class Probe:
    def __init__(self,out):
        self.out=Path(out);self.name='C1';self.group={};self.start_directions=None
        self.normals=np.array([[0.,0.,1.]])
        self.ray_normals=np.eye(3);self.max_proposals=1200;self.proposals=0
        self.best=dict(serial=1,counts=[1],directions=self.normals.copy());self.beam=[]
        self.trace=[];self.global_trace=[];self.selection_trace=[];self.exact_calls=0
        self.winner=dict(masks=[np.ones(1,bool)],counts=[32768],directions=self.normals.copy(),
                         overlap=0.,partition=0.,endpoint_overlap=[0.])

    def warm(self):return self.normals.copy()
    def exact(self,d):raise RuntimeError('unresolved native construction')
    def dual_weights(self,result):return np.array([0.,2.,3.]),{}
    def ordered_common(self,axes):
        self.axes=axes
        return [(1.,self.normals.copy(),np.array([0.,0.,1.]))]
    def attempt(self,d,label,common=None,force=False):
        return None if label=='common floor cone' else self.winner
    def finish(self,*args,**kwargs):return self.report_extra


class ConcurrentCoveringTests(unittest.TestCase):
    def test_helpful_normals_precede_the_unmodified_covering_bank(self):
        with tempfile.TemporaryDirectory() as folder:
            search=Probe(folder);report=concurrent_search(search)
        self.assertEqual(len(search.axes),162)
        np.testing.assert_allclose(search.axes[:2],-np.eye(3)[1:])
        self.assertEqual(report['gradient_exact_evaluations'],0)
        self.assertEqual(report['total_exact_evaluations'],0)
        self.assertTrue(report['passed'])

if __name__=='__main__':unittest.main()
