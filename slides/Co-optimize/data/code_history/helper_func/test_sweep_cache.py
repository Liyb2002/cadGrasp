import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
from co_common import *
import step42_geometry_cache as cache
class ExactCache(unittest.TestCase):
    def test_reuse_and_geometry_or_direction_invalidation(self):
        m=trimesh.creation.box([.01,.02,.03]);d=np.array([0.,0.,.1]);a=cache.cached_sweep(m,d);b=cache.cached_sweep(m.copy(),d.copy());self.assertIs(a,b)
        moved=m.copy();moved.apply_translation([.2,0,0]);c=cache.cached_sweep(moved,d);self.assertIsNot(a,c)
        np.testing.assert_allclose(c.bounds-a.bounds,[[.2,0,0],[.2,0,0]],atol=1e-12)
        e=cache.cached_sweep(m,[.1,0,0]);self.assertIsNot(a,e)
        self.assertGreater(e.bounds[1,0],a.bounds[1,0]);self.assertGreater(a.bounds[1,2],e.bounds[1,2])
        self.assertAlmostEqual(material_volume(cache.cached_solid(a)),material_volume(cache._original_solid(a)),places=12)
if __name__=='__main__':unittest.main()
