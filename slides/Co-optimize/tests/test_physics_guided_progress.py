"""Review intermediate progress on shared blockers and force tradeoffs."""

import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
import numpy as np
from physics_guided_progress import progress_decision
from physics_guided_geometry import distance_cost
from physics_guided import PhysicsSearch, tangent_frames
from physics_guided_objective import acquisition_equilibrium, aggregate


class IntermediateProgressTests(unittest.TestCase):
    def test_first_blocker_can_move_without_recovering_contact(self):
        # Both exits still cut a useful patch. Moving only the strongest
        # blocker reduces the nonlinear acquisition cost but restores no ray.
        initial=np.array([[.2,.15]])
        moved=np.array([[.18,.15]])
        zero=np.zeros((2,2));jac=np.zeros((1,2,2))
        before=distance_cost(initial,jac,zero)[0][0]
        after=distance_cost(moved,jac,zero)[0][0]
        self.assertTrue(np.all(moved>0))
        result=progress_decision(.5,.5,.5,before,after)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['acceptance_reason'],'geometry')

    def test_geometry_can_cross_a_discontinuous_contact_loss(self):
        result=progress_decision(.5,.57,.5,.2,.1)
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['force_loss_excess_above_best'],.07)

    def test_force_regression_without_geometry_progress_is_rejected(self):
        self.assertTrue(progress_decision(.5,.54,.5,.2,.15)['accepted'])
        self.assertFalse(progress_decision(.54,.58,.5,.15,.16)['accepted'])

    def test_force_improvement_can_trade_previously_feasible_loads(self):
        self.assertEqual(progress_decision(.5,.4,.5,.2,.3)['acceptance_reason'],'force')

    def test_no_actual_geometry_or_force_improvement_is_rejected(self):
        self.assertFalse(progress_decision(.5,.5,.5,.2,.2)['accepted'])

    def test_nonfinite_guidance_is_rejected(self):
        with self.assertRaises(ValueError):progress_decision(.5,.5,.5,.2,np.nan)

    def test_frozen_geometry_gradient_matches_physical_envelope(self):
        search=PhysicsSearch.__new__(PhysicsSearch)
        search.ray_normals=np.array([[1.,0,0],[0,1.,0],[-1.,0,0],[0,-1.,0]])
        d=np.array([[0.,0,1.],[.6,0,.8]])
        frames=tangent_frames(d);zero=np.zeros((2,2))
        costs,jac=search.local_costs(d,frames,zero)
        contacts=np.eye(7)[:4];floor=np.empty((0,7))
        targets=[np.array([.5,.3,0,.2,0,0,0]),np.array([.1,.6,.2,0,0,0,0])]
        physics=[acquisition_equilibrium(floor,contacts,t,costs) for t in targets]
        values=np.array([p['value'] for p in physics])
        factors=np.exp((values-values.max())/.005);factors/=factors.sum()
        frozen=sum(w*p['cost_gradient'] for w,p in zip(factors,physics))
        _,physical_gradient=aggregate(values,[np.einsum('j,jni->ni',p['cost_gradient'],jac) for p in physics])
        geometry_gradient=np.einsum('j,jni->ni',frozen/frozen.sum(),jac)
        np.testing.assert_allclose(geometry_gradient,physical_gradient/frozen.sum(),atol=1e-12)
        for i in range(2):
            for axis in range(2):
                step=np.zeros_like(zero);step[i,axis]=1e-6
                plus=search.local_costs(d,frames,step)[0]@frozen/frozen.sum()
                minus=search.local_costs(d,frames,-step)[0]@frozen/frozen.sum()
                self.assertAlmostEqual((plus-minus)/2e-6,geometry_gradient[i,axis],places=7)


if __name__=='__main__':unittest.main()
