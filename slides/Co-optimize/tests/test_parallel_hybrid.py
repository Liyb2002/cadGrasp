
import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
from physics_guided_hybrid_refined import interleaved_common_neighborhood
from physics_guided_parallel import independent_scales
from physics_guided_cone_iterative import cone_projection
from physics_guided_cone import cone_projection as reference


class ParallelHybridTests(unittest.TestCase):
    def test_large_cone_preserves_projection_and_kkt(self):
        rays=np.random.default_rng(7).uniform(.1,1.,size=(4000,7))
        target=np.array([-1.,2.,1.,0.,.5,3.,1.])
        expected=reference(rays,target);actual=cone_projection(rays,target)
        self.assertAlmostEqual(actual['loss'],expected['loss'],places=8)
        self.assertLess(actual['kkt_max_violation'],1e-7)

    def test_iterative_fallback_polishes_complementarity(self):
        from unittest.mock import patch
        import physics_guided_cone_iterative as kernel
        solve=kernel.nnls;calls=[0]
        def fail_first(*args,**kwargs):
            calls[0]+=1
            if calls[0]==1:raise RuntimeError('forced primary solver failure')
            return solve(*args,**kwargs)
        rays=np.random.default_rng(12).uniform(.1,1.,size=(600,7))
        target=np.array([-1.,2.,1.,0.,.5,3.,1.])
        with patch.object(kernel,'nnls',side_effect=fail_first):actual=kernel.cone_projection(rays,target)
        self.assertEqual(actual['solver'],'iterative_fallback_sparse_polish')
        self.assertAlmostEqual(actual['loss'],reference(rays,target)['loss'],places=8)
        self.assertLess(actual['kkt_max_violation'],1e-7)

    def test_metrics_and_duplicate_rays_keep_real_residual(self):
        rays=np.vstack([np.eye(7)[:3],np.eye(7)[:3]*3,np.zeros((1,7))])
        target=np.array([1.,-2.,.5,3.,-1.,2.,.8]);metric=np.array([1.,2.,3.,.5,1.,2.,.7])
        actual=cone_projection(rays,target,metric);expected=reference(rays,target,metric)
        np.testing.assert_allclose(actual['residual'],expected['residual'],atol=1e-9)
        self.assertLess(actual['kkt_max_violation'],1e-7)

    def test_local_budget_reaches_every_scale(self):
        center=np.array([0.,0.,1.]);bank=list(interleaved_common_neighborhood(center))[:20]
        angles=np.rad2deg(np.arccos([a@center for a,lift in bank]))
        np.testing.assert_allclose(sorted(set(np.round(angles,6))),[.5,2,5,10,20])

    def test_independent_budget_reaches_every_pose_and_stays_legal(self):
        directions=np.tile([0.,0.,1.],(6,1));bank=list(independent_scales(directions,directions,range(6)))
        self.assertEqual({k for candidate,k in bank[:6]},set(range(6)))
        for candidate,k in bank:
            self.assertTrue(np.all(np.sum(candidate*directions,axis=1)>=0))
            np.testing.assert_allclose(np.linalg.norm(candidate,axis=1),1.)

if __name__=='__main__':unittest.main()
