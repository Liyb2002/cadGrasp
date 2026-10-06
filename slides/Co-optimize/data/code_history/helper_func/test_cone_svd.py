import sys,unittest
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'step4.2'))
from physics_guided_cone_svd import cone_projection

class SVDProjectionTests(unittest.TestCase):
    def test_small_cones_match_reference(self):
        rng=np.random.default_rng(721)
        for n in [0,1,3,6,20,50]:
            for trial in range(10):
                rays=rng.normal(size=(n,7));target=rng.normal(size=7)
                result=cone_projection(rays,target)
                value=nnls(rays.T,target,maxiter=5000)[0] if n else np.empty(0)
                np.testing.assert_allclose(result['residual'],value@rays-target,atol=1e-8)

    def test_nearly_duplicate_rays_and_extreme_scaling(self):
        rng=np.random.default_rng(722);base=rng.uniform(.1,1,size=(6,7))
        rays=np.repeat(base,500,axis=0)+rng.normal(scale=1e-12,size=(3000,7))
        rays*=np.logspace(-4,4,len(rays))[:,None]
        target=np.array([-1,2,1,0,.5,3,1.])
        result=cone_projection(rays,target)
        self.assertLess(result['kkt_max_violation'],1e-7*np.linalg.norm(target))
        self.assertTrue(np.all(result['coefficients']>=0))
        reference=nnls(base.T,target)[0]@base-target
        np.testing.assert_allclose(result['residual'],reference,atol=1e-8)

    def test_large_feasible_and_metric(self):
        rng=np.random.default_rng(723);rays=rng.normal(size=(5000,7))
        target=rays[[5,501,4096]].sum(axis=0)
        result=cone_projection(rays,target,np.array([1,2,3,4,5,6,7.]))
        np.testing.assert_allclose(result['residual'],0.,atol=1e-8)

if __name__=='__main__':unittest.main()
