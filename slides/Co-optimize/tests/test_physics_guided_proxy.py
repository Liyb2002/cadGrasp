
import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
import numpy as np
from scipy.optimize import linprog
from physics_guided_proxy import CachedNecessaryCone
from physics_guided_cone import cone_projection


def solve(full,target):
    if not len(full):return None
    lp=linprog(np.zeros(len(full)),A_eq=full.T,b_eq=target,bounds=(0,None),method='highs')
    if not lp.success:return None
    indices=np.flatnonzero(lp.x>0)
    return dict(indices=indices.tolist(),coefficients=lp.x[indices].tolist())


class NecessaryConeCacheTests(unittest.TestCase):
    def test_primal_reuse_preserves_support_mapping(self):
        rays=np.eye(7);floor=np.eye(7)[:1];target=np.array([[1.,2.,0.,0.,0.,0.,0.]])
        cache=CachedNecessaryCone([rays],[floor],[target],solve)
        keep=np.ones(7,bool)
        self.assertTrue(cache.check(keep,[{0}]))
        keep[6]=False
        self.assertTrue(cache.check(keep,[{0}]))
        self.assertEqual(cache.stats['primal_solves'],1)
        self.assertEqual(cache.stats['primal_reused'],1)
        keep[1]=False
        self.assertFalse(cache.check(keep,[{0}]))

    def test_dual_reuse_invalidated_by_restoring_violating_ray(self):
        rays=np.eye(7);floor=np.eye(7)[:1];target=np.array([[1.,2.,0.,0.,0.,0.,0.]])
        cache=CachedNecessaryCone([rays],[floor],[target],solve)
        keep=np.zeros(7,bool)
        self.assertFalse(cache.check(keep,[{0}]))
        before=cache.stats['dual_solves']
        self.assertFalse(cache.check(keep,[{0}]))
        self.assertEqual(cache.stats['dual_solves'],before)
        self.assertEqual(cache.stats['dual_reused'],1)
        keep[1]=True
        self.assertTrue(cache.check(keep,[{0}]))

    def test_cached_rejection_never_discards_feasible_original_targets(self):
        rng=np.random.default_rng(33);rays=np.vstack([np.eye(7),-np.eye(7)])
        floors=np.eye(7)[:1];targets=rng.normal(size=(4,7))
        cache=CachedNecessaryCone([rays],[floors],[targets],solve)
        for _ in range(60):
            keep=rng.random(len(rays))<.5
            verdict=cache.check(keep,[set(range(4))])
            if not verdict:
                full=np.vstack([floors,rays[keep]])
                self.assertGreater(max(cone_projection(full,b)['loss'] for b in targets),1e-8)

if __name__=='__main__':unittest.main()
