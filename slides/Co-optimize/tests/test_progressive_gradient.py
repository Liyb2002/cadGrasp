"""Lazy candidate scoring preserves the original whole-load safety gate."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.model import Layout
from continuous_support.progressive_gradient import ProgressiveGradientSearch


class ProgressiveGradientTests(unittest.TestCase):
    def setup_search(self, counts):
        layout=Layout(np.eye(4)[None],np.array([[0.,0.,1.]]),np.array([0]),(0,))
        directions=[]
        for x in [.01,.02,.03]:
            q=layout.copy();q.directions[0]=[x,0.,np.sqrt(1-x*x)];directions.append(q)
        def state(q,mask,serial):
            return dict(layout=q,masks={0:np.array(mask,bool)},counts={'0':sum(mask)},serial=serial,
                        volume_cm3=1.,maximum_projected_footprint_m2=0.)
        current=state(layout,[False,False],0)
        loss={layout.key():1.,directions[0].key():.5,directions[1].key():.4,directions[2].key():.3}
        model=SimpleNamespace(proxy=Mock(side_effect=lambda q:dict(loss=loss[q.key()])),extent=1.,
                              evaluate=Mock(side_effect=[state(directions[i],mask,i+1) for i,mask in enumerate(counts)]),
                              commit=Mock())
        search=ProgressiveGradientSearch(model,Path('.'))
        search.targets=Mock(return_value=([],{},[]));search.record=Mock();search.checkpoint=Mock()
        search.score=lambda s,a=None:(sum(int((~m).sum()) for m in s['masks'].values()),loss[s['layout'].key()])
        search.local_proposals=Mock(return_value=([
            ('direction-gradient-joint',directions[0],dict(step_degrees=2.)),
            ('direction-gradient-joint',directions[1],dict(step_degrees=.5)),
            ('direction-sample',directions[2],dict(step_degrees=2.))],dict(loss=1.)))
        return search,model,current,directions

    def test_accepted_line_step_skips_other_scales_and_all_sampling(self):
        search,model,current,directions=self.setup_search([[True,False]])
        result=search.refine(current,1)
        self.assertIs(result['layout'],directions[0])
        model.proxy.assert_called_once_with(directions[0]);model.evaluate.assert_called_once_with(directions[0])
        self.assertEqual(search.record.call_args.args[0]['skipped_candidates'],2)

    def test_lower_proxy_loss_cannot_commit_more_failed_original_demands(self):
        search,model,current,directions=self.setup_search([[False,False],[False,False],[False,False]])
        current['masks'][0][0]=True;current['counts']['0']=1
        self.assertIs(search.refine(current,1),current)
        self.assertEqual(model.evaluate.call_count,3)
        self.assertFalse(any(r['accepted'] for r in search.record.call_args.args[0]['trials']))

    def test_translation_steps_use_body_fraction(self):
        rows=[('translation-gradient',None,dict(step_m=.02/32,body_extent_m=.02)),
              ('translation-gradient',None,dict(step_m=.02/128,body_extent_m=.02))]
        groups=ProgressiveGradientSearch.proposal_groups(rows)
        self.assertEqual([g[1][0][2]['step_m'] for g in groups],[.02/32,.02/128])


if __name__=='__main__':unittest.main()
