"""Skip impossible full checks and include every owner's translation losses."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from whole_search.model import Layout
from continuous_support.fast_state_seat_gradient import FastStateSeatGradientSearch


class FastStateSeatTests(unittest.TestCase):
    def test_lazy_gate_keeps_zero_loss_count_repair_and_skips_ineligible_checks(self):
        losses={'bad':1.2,'better':.8,'zero':0.}
        model=SimpleNamespace(proxy=Mock(side_effect=lambda q:dict(loss=losses[q])),
                              timing={},evaluate=Mock())
        search=FastStateSeatGradientSearch(model,Path('.'));search.comparison_loss=1.
        rows=[('direction-gradient','gradient',{}),('direction-sample','bad',{}),
              ('direction-sample','better',{}),('direction-sample','zero',{})]
        groups=search.proposal_groups(rows)
        self.assertEqual(next(groups)[0],'gradient_multiscale')
        model.proxy.assert_not_called()
        remaining=list(groups)
        self.assertEqual([r[1] for _,group in remaining for r in group],['better','zero'])
        self.assertEqual(model.timing['ineligible_full_load_candidates_skipped'],1)
        model.evaluate.assert_not_called()

    def test_new_seat_gradient_accounts_for_other_owner_damage(self):
        layout=Layout(np.repeat(np.eye(4)[None],2,axis=0),np.array([[0.,0.,1.]]*2),np.arange(2),(0,1))
        def proxy(q):
            x=q.placements[0,0,3];y=q.placements[0,1,3]
            # Guest alone prefers +x; the ALL-owner sum prefers -x.
            return dict(loss=(x-1.)**2+3*(x+1.)**2+y*y)
        model=SimpleNamespace(mesh=trimesh.creation.box(),extent=1.,poses=['p0','p1'],
            native=np.repeat(np.eye(4)[None],2,axis=0),floor_normal=lambda q,k:np.array([0.,0.,1.]),proxy=proxy)
        search=FastStateSeatGradientSearch(model,Path('.'))
        current=dict(layout=layout,masks={0:np.array([False]),1:np.array([True])})
        rows=search.current_state_seats(current,[0])
        guided=[r for r in rows if r[2].get('proposed_by_all_pose_translation_gradient')]
        self.assertTrue(guided)
        for kind,trial,detail in guided:
            self.assertEqual(kind,'juxtapose-current-state')
            self.assertLess(trial.placements[0,0,3],0.)
            self.assertAlmostEqual(trial.placements[0,2,3],0.)
            self.assertFalse(detail['gradient_step'])
            self.assertTrue(detail['all_poses_in_objective'])


if __name__=='__main__':unittest.main()
