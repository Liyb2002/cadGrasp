import unittest
import numpy as np
from physics_guided_batch_projection import all_projection_losses
from physics_guided_cone_svd import cone_projection

class BatchProjectionTests(unittest.TestCase):
    def test_all_targets_match_individual_projections(self):
        rng=np.random.default_rng(826)
        for m in [0,3,6,30]:
            rays=rng.normal(size=(m,7));targets=rng.normal(size=(200,7))
            losses,info=all_projection_losses(rays,targets)
            reference=np.array([cone_projection(rays,t)['loss'] for t in targets])
            np.testing.assert_allclose(losses,reference,atol=1e-8,rtol=1e-7)
            self.assertTrue(info['all_failed_loads_projected'])

    def test_region_budget_falls_back_without_omitting_targets(self):
        rays=np.eye(7)[:4];targets=np.array([[1,-2,3,-4,5,-6,7.],[-1,2,-3,4,-5,6,-7.]])
        losses,info=all_projection_losses(rays,targets,region_limit=0)
        np.testing.assert_allclose(losses,[cone_projection(rays,t)['loss'] for t in targets])
        self.assertEqual(info['individual_fallbacks'],2)

    def test_proved_feasible_loads_are_zero_but_failures_all_projected(self):
        rays=np.eye(7);targets=np.vstack([np.ones(7),-np.ones(7)])
        losses,info=all_projection_losses(rays,targets,failed=[False,True])
        np.testing.assert_allclose(losses,[0.,3.5]);self.assertEqual(info['failed_loads'],1)

if __name__=='__main__':unittest.main()
