"""Infeasible gradient descent is governed by the whole integral, not counts."""
import unittest
from unittest.mock import patch
import numpy as np
from test_progressive_gradient import ProgressiveGradientTests
from continuous_support.loss_first_gradient import LossFirstGradientSearch


class LossFirstGradientTests(unittest.TestCase):
    def test_lower_integral_can_temporarily_lose_near_boundary_demands(self):
        search,model,current,_=ProgressiveGradientTests().setup_search([[False,False]])
        search.__class__=LossFirstGradientSearch;model.volume_delta=None
        current['masks'][0][0]=True;current['counts']['0']=1
        # Use the actual loss-first score, not the mock factory's ranking.
        del search.score
        with patch('continuous_support.loss_first_gradient.juxtaposed_count',return_value=0):
            result=search.refine(current,1)
        self.assertEqual(result['counts']['0'],0)
        self.assertLess(model.proxy(result['layout'])['loss'],model.proxy(current['layout'])['loss'])
        self.assertTrue(search.record.call_args.args[0]['accepted'])

    def test_fully_feasible_incumbent_dominates_a_smaller_error_candidate(self):
        search,model,current,_=ProgressiveGradientTests().setup_search([[True,True]])
        search.__class__=LossFirstGradientSearch;model.volume_delta=None;del search.score
        candidate=model.evaluate(None);candidate['volume_cm3']=100.
        with patch('continuous_support.loss_first_gradient.juxtaposed_count',return_value=0):
            self.assertLess(search.score(candidate),search.score(current))
        model.evaluate.reset_mock()
        self.assertIs(search.refine(candidate,1),candidate)
        model.evaluate.assert_not_called()


if __name__=='__main__':unittest.main()
