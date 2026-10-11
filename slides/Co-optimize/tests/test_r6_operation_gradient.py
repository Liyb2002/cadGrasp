import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import unittest
from types import SimpleNamespace
import numpy as np
from direction_choice.r6_gradient import ProjectionCells,forward_strata
from continuous_support.projection import project_demands
from continuous_support.adaptive_fast_gradient import demand_strata
from current_task import current_task
from whole_search.common import C


class ProjectionCellTests(unittest.TestCase):
    def test_added_deleted_contacts_and_zero_supply(self):
        rng=np.random.default_rng(412);bank=rng.normal(size=(97,7));targets=rng.normal(size=(31,7))
        bank[:,-1]=np.abs(bank[:,-1]);bank[0]=np.r_[np.zeros(6),-1.]
        engine=ProjectionCells(bank)
        for ids in [np.arange(18),np.arange(19),np.arange(2,24),np.arange(5),np.array([0]),np.array([],int)]:
            result=engine.evaluate(ids,targets,('same_nodes',))
            expected=project_demands(bank[ids],targets)['residuals']
            np.testing.assert_allclose(result,expected,atol=3e-8,rtol=1e-8)
        self.assertGreater(engine.reused,0)

    def test_quadratic_cell_derivative(self):
        bank=np.eye(7);targets=np.array([[1.,-.2,.4,-.5,1.,-.1,.2],[.3,-.7,-.4,.5,.1,.2,-.8]])
        engine=ProjectionCells(bank);base=engine.evaluate(np.arange(7),targets,'fixed')
        delta=np.full_like(targets,.017);eps=1e-5
        a=engine.evaluate(np.arange(7),targets+eps*delta,'fixed')
        b=engine.evaluate(np.arange(7),targets-eps*delta,'fixed')
        numerical=(.5*np.sum(a*a,axis=1)-.5*np.sum(b*b,axis=1))/(2*eps)
        np.testing.assert_allclose(numerical,np.sum(-base*delta,axis=1),atol=1e-10)

    def test_rescaling_and_duplicate_ray_ids(self):
        bank=np.vstack([np.eye(7),np.eye(7)*3,np.zeros((1,7))]);targets=np.array([[-.3,.2,-.1,.5,-.7,.4,.1]])
        engine=ProjectionCells(bank)
        for ids in [np.arange(7),np.arange(7,14),np.arange(15)]:
            np.testing.assert_allclose(engine.evaluate(ids,targets,'fixed'),project_demands(bank[ids],targets)['residuals'],atol=1e-12)

    def test_floor_removal_retains_seventh_slack(self):
        floor=np.zeros((5,7));floor[:4,2]=1.;floor[:4,3:5]=[[1,0],[-1,0],[0,1],[0,-1]];floor[4,-1]=-1.
        heads=np.eye(7);bank=np.vstack([floor,heads]);needs=np.array([[0,0,1,0,0,0,0.],[0,0,1,.2,.3,0,0.]])
        engine=ProjectionCells(bank)
        for ids in [np.arange(12),np.arange(4,12),np.arange(12)]:
            np.testing.assert_allclose(engine.evaluate(ids,needs,'fixed'),project_demands(bank[ids],needs)['residuals'],atol=1e-10)


class ForwardStrataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.task=current_task('B','pose_1')

    def compare(self,mask):
        model=SimpleNamespace(tasks=[self.task],inputs=list(self.task.inputs),startup_sources={})
        expected,w=demand_strata(C.U.target(self.task.targets),mask)
        actual,aw=forward_strata(model,0,mask)
        np.testing.assert_array_equal(aw,w)
        np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=0)
        self.assertEqual(actual.shape[1],7)
        self.assertAlmostEqual(float(aw.sum()),1.)
        self.assertIn('slides/Co-optimize/helper_func/direction_choice/r6_gradient.py',model.startup_sources)

    def test_covered_uncovered_mass_preserved_with_forward_formula(self):
        self.compare(np.arange(32768)%3!=0)

    def test_all_rare_failures_retained(self):
        mask=np.ones(32768,bool);mask[71:89]=False;self.compare(mask)


if __name__=='__main__':unittest.main()
