"""Full-coverage protection and bounded sizing, including radius expansion."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer.stage_imports import load_stage
K=load_stage('optimize','coordinate')


class Scalar:
    def __init__(self,initial,counts,maximum=2.):
        self.initial_radius=initial;self.minimum_radius=.1;self.tolerance=1e-5
        self.targets=np.zeros((100,7));self.cache={};self.counts=counts;self.maximum=maximum
        self.fixed_full=np.zeros((1,7));self.domain=None;self.floor=None;self.scale=None

    def geometry(self,r):
        return dict(patch=None)

    def score(self,r):
        if r not in self.cache:
            count=self.counts(r)
            self.cache[r]=dict(mask=np.arange(100)<count,
                row=dict(area_m2=r,total_area_m2=1+r,covered_count=count))
        return self.cache[r]

    def maximum_radius(self):return self.maximum,{}
    def insertion_radius_cap(self,r):return r,{}


class CoordinateTests(unittest.TestCase):
    def test_invalid_geometry_cannot_be_certified_by_clear_sweep(self):
        problem=K.A.SizeProblem.__new__(K.A.SizeProblem)
        problem.direction_cache={}
        problem.geometry=lambda radius: {'row': {'geometry_valid': False}}
        problem.contact=lambda radius: self.fail('Invalid geometry must not enter sweep certification')
        self.assertEqual(problem.insertion_directions(.04),{'ids':[]})

    def search(self,problem,budget):
        with patch.object(K.A,'patch_columns',return_value=np.zeros((1,7))), \
                patch.object(K.C,'gravity_check',return_value=dict(passed=True)):
            return K.search_coordinate(problem,budget)

    def test_full_coverage_cannot_be_traded_for_better_ratio(self):
        # Incomplete tiny contact has a better count/area ratio than a full one.
        p=Scalar(1.,lambda r:90 if r<.8 else 100)
        radius,report=self.search(p,24)
        self.assertGreaterEqual(radius,.8)
        self.assertLess(radius,.8001)
        self.assertTrue(p.score(radius)['mask'].all())
        self.assertLessEqual(report['evaluated_sizes'],24)

    def test_expansion_can_reach_full_coverage_before_shrinking(self):
        p=Scalar(.5,lambda r:50 if r<1.2 else 100)
        radius,report=self.search(p,12)
        self.assertGreaterEqual(radius,1.2)
        self.assertLess(radius,1.21)
        self.assertEqual(report['evaluated_sizes'],12)

    def test_minimum_budget_keeps_a_known_feasible_size(self):
        p=Scalar(1.,lambda r:0 if r<.73 else 100)
        radius,report=self.search(p,3)
        self.assertEqual(radius,1.)
        self.assertFalse(report['converged'])
        self.assertEqual(len(p.cache),3)

    def test_full_coverage_report_with_numpy_radius_serializes(self):
        p=Scalar(1.,lambda r:90 if r<.8 else 100)
        p.minimum_radius=np.float64(.1)
        p.tolerance=np.float64(1e-5)
        _,report=self.search(p,24)
        json.dumps(report,allow_nan=False)

    def test_partial_coverage_uses_joint_area(self):
        p=Scalar(1.,lambda r:10 if r<.5 else 60)
        radius,report=self.search(p,20)
        self.assertGreaterEqual(radius,.5)
        self.assertLess(radius,.502)
        self.assertFalse(report['full_coverage_found'])
        self.assertLessEqual(len(p.cache),20)


if __name__=='__main__':unittest.main()
