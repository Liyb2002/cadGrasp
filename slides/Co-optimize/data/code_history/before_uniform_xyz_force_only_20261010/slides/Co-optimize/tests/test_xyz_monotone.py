"""Height freedom preserves XY proposals and cannot worsen checked material."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.fast_search import FastModel
from whole_search.model import Layout
from whole_search.checked_solution import select_no_worse
from continuous_support.cached_fast_gradient import CachedGradientSearch
from continuous_support.progressive_gradient import ProgressiveGradientSearch
from translation.volume_descent import polish_xyz


class XYZMonotoneTests(unittest.TestCase):
    def test_xyz_keeps_every_legacy_horizontal_candidate_and_sample_chunk(self):
        model=FastModel.__new__(FastModel)
        model.native=np.repeat(np.eye(4)[None],2,axis=0)
        model.native[1,:3,:3]=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        model.extent=1.;model.poses=['a','b'];model.gradient_seconds=0.
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([[0.,1.,0.],[0.,0.,1.]]),np.array([1,0]),(0,1))
        def proxy(q):
            positions=np.array([model.native[q.hosts[k],:3,:3]@q.placements[k,:3,3] for k in q.active])
            loss=float(np.sum((positions-[.1,.2,.03])**2)+np.sum((q.directions-[0.,0.,1.])**2))
            return dict(loss=loss,residual_loss=[loss/2]*2)
        model.proxy=proxy;search=CachedGradientSearch(model,Path('.'))
        for fine in [False,True]:
            search.fine_resolution=fine;model.allow_z_translation=False
            old,_=search.local_proposals(dict(layout=layout),[],{},translation_guests=(0,1))
            model.allow_z_translation=True
            new,_=search.local_proposals(dict(layout=layout),[],{},translation_guests=(0,1))
            legacy=[r for r in new if not r[2].get('supplementary_xyz')]
            self.assertEqual(len(old),len(legacy));self.assertGreater(len(new),len(old))
            for a,b in zip(old,legacy):
                self.assertEqual(a[0],b[0]);self.assertEqual(a[2],b[2])
                np.testing.assert_array_equal(a[1].placements,b[1].placements)
                np.testing.assert_array_equal(a[1].directions,b[1].directions)
            for _,_,detail in old+new:
                detail['body_extent_m']=model.extent
            old_groups=ProgressiveGradientSearch.proposal_groups(old,fine)
            new_groups=ProgressiveGradientSearch.proposal_groups(new,fine)
            self.assertGreater(len(new_groups),len(old_groups))
            for (a,aa),(b,bb) in zip(old_groups,new_groups):
                self.assertEqual(a,b)
                self.assertEqual([(r[0],r[1].key()) for r in aa],[(r[0],r[1].key()) for r in bb])

    def test_larger_or_unresolved_new_solution_cannot_replace_actual_incumbent(self):
        old=dict(passed=True,volume_cm3=10.,experiment='old')
        for candidate in [dict(passed=True,volume_cm3=10.00000001,experiment='new'),
                          dict(passed=True,volume_cm3=10.,experiment='new'),
                          dict(passed=False,volume_cm3=1.,experiment='new')]:
            chosen=select_no_worse(candidate,old)
            self.assertEqual(chosen['experiment'],'old');self.assertTrue(chosen['retained_incumbent'])
            self.assertEqual(chosen['volume_cm3'],10.)
        chosen=select_no_worse(dict(passed=True,volume_cm3=9.,experiment='new'),old)
        self.assertEqual(chosen['experiment'],'new');self.assertFalse(chosen['retained_incumbent'])

    def volume_fixture(self,fail_if_raised=False):
        model=FastModel.__new__(FastModel);model.allow_z_translation=True
        model.native=np.repeat(np.eye(4)[None],2,axis=0);model.extent=1.;model.poses=['a','b']
        q=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([[0.,0.,1.]]*2),np.array([0,0]),(0,1))
        estimate=lambda q:10.+100.*(model.workpiece_height(q,1)-.03)**2
        guidance=SimpleNamespace(estimate=estimate,epoch=0)
        model.volume_guidance=lambda layouts:guidance
        model.refresh_volume=lambda r:r.update(volume_cm3=estimate(r['layout']))
        def result(q,serial):
            mask=np.array([not(fail_if_raised and model.workpiece_height(q,1)>1e-9)])
            return dict(layout=q,masks={0:np.array([True]),1:mask},counts={0:1,1:int(mask[0])},
                        serial=serial,volume_cm3=estimate(q))
        model.evaluate=Mock(side_effect=lambda q:result(q,2));model.commit=Mock()
        search=SimpleNamespace(model=model,checkpoint=Mock(),record=Mock())
        return search,result(q,1)

    def test_material_gradient_can_use_z_after_feasibility(self):
        search,current=self.volume_fixture()
        result=polish_xyz(search,current,1)
        self.assertGreater(search.model.workpiece_height(result['layout'],1),0.)
        self.assertLess(result['volume_cm3'],current['volume_cm3'])
        accepted=[t for t in search.record.call_args.args[0]['trials'] if t['accepted']]
        self.assertEqual(accepted[0]['operation'],'translation-volume-gradient')

    def test_material_reduction_cannot_break_another_original_load(self):
        search,current=self.volume_fixture(fail_if_raised=True)
        result=polish_xyz(search,current,2)
        self.assertIs(result,current);self.assertTrue(search.model.evaluate.called)
        search.checkpoint.assert_not_called()


if __name__=='__main__':unittest.main()
