
import sys as _test_sys
from pathlib import Path as _TestPath
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import unittest
from co_common import *
class ComponentValue(unittest.TestCase):
    def test_discard_is_not_inferred_from_single_component_size(self):
        # Complementary unilateral reactions: neither fragment alone carries
        # both demands, while both fragments together do.
        a=np.array([[1.,0,0,0,0,0,0]])
        b=np.array([[0.,1,0,0,0,0,0]])
        target_a=np.array([1.,0,0,0,0,0,0]);target_b=np.array([0.,1,0,0,0,0,0])
        self.assertIsNotNone(C.W.solve(np.vstack([a,b]),target_a))
        self.assertIsNotNone(C.W.solve(np.vstack([a,b]),target_b))
        self.assertIsNone(C.W.solve(a,target_b))
        self.assertIsNone(C.W.solve(b,target_a))
    def test_retained_component_is_connected_positive_material(self):
        large=F.md.Manifold.cube((1.,1.,1.));small=F.md.Manifold.cube((.1,.1,.1)).translate((2.,0,0))
        parts=union([large,small]).decompose();self.assertEqual(len(parts),2)
        keep=max(parts,key=material_volume);self.assertEqual(len(keep.decompose()),1)
        self.assertGreater(material_volume(keep),0.)
        self.assertLess(material_volume(keep^small),1e-12)
if __name__=='__main__':unittest.main()
