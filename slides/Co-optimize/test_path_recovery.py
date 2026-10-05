import unittest
from co_common import *
from step42 import fibonacci,project_common
class Recovery(unittest.TestCase):
    def test_shared_tendency_gives_independent_floor_legal_exits(self):
        normals=fibonacci(31)
        for common in fibonacci(19):
            d=project_common(common,normals)
            np.testing.assert_allclose(np.linalg.norm(d,axis=1),1.,atol=1e-12)
            self.assertTrue(np.all(np.sum(d*normals,axis=1)>0))
        self.assertFalse(np.allclose(d[0],d[-1]))
    def test_degenerate_antiparallel_projection(self):
        n=np.array([[0.,0.,1.]])
        d=project_common(np.array([0.,0.,-1.]),n,lift=0.)
        self.assertTrue(np.isfinite(d).all());self.assertAlmostEqual(np.linalg.norm(d),1.);self.assertGreaterEqual(d[0]@n[0],0.)
    def test_changing_one_exit_restores_only_material_other_exits_release(self):
        mesh=trimesh.creation.box([.01,.01,.01]);seed=S.solid(trimesh.creation.box([.05,.05,.05]))
        up=S.solid(S.swept_solid(mesh,[0,0,.07]));right=S.solid(S.swept_solid(mesh,[.07,0,0]));left=S.solid(S.swept_solid(mesh,[-.07,0,0]))
        before=seed-union([up,right]);after=seed-union([up,left]);restored=after-before
        self.assertGreater(material_volume(restored),1e-9)
        self.assertLess(material_volume(restored^up),1e-12)
        self.assertLess(material_volume(restored^left),1e-12)
if __name__=='__main__':unittest.main()

class NumericalDeadline(unittest.TestCase):
    def test_deadline_escapes_solver_fallback_without_acceptance(self):
        import signal
        from unittest.mock import patch
        from step42 import Search
        from step42_timed import TimeLimitedSearch
        obj=object.__new__(TimeLimitedSearch);obj.timeouts=0;obj.trace=[];obj.evaluations=1;obj.best=None
        def fallback(*args,**kwargs):
            try:signal.raise_signal(signal.SIGALRM)
            except Exception:raise AssertionError('Deadline was swallowed by normal solver fallback')
        with patch.object(Search,'candidate',side_effect=fallback):
            self.assertIsNone(obj.candidate(np.array([[0.,0.,1.]])))
        self.assertEqual(obj.timeouts,1)
        self.assertIn('unresolved',obj.trace[0]['status'])
