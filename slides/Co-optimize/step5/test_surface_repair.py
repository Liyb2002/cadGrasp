"""Physical counterexamples for the optional local surface expansion."""
import unittest
import json
import numpy as np
from wrap_geometry import box, solid, unpack, volume
from wrap_surface_repair import neck_regions, local_expansion


class SurfaceRepair(unittest.TestCase):
    def setUp(self):
        self.a = box(np.array([0.,0.,0.]), np.array([.012,.016,.005]))
        self.b = box(np.array([.022,0.,0.]), np.array([.034,.016,.005]))
        self.neck = box(np.array([.011,.007,0.]), np.array([.023,.009,.005]))
        self.value = self.a+self.neck+self.b
        self.region = dict(bounds_m=[[.011,.007,0.],[.023,.009,.005]])

    def test_narrow_connection_is_selected_and_widened_locally(self):
        regions, detection = neck_regions(unpack(self.value))
        json.dumps(dict(regions=regions,detection=detection))
        self.assertTrue(regions)
        domain = box(np.array([-.01,-.01,-.01]), np.array([.05,.03,.02]))
        grown, _ = local_expansion(self.value, regions, domain)
        middle = box(np.array([.014,.005,-.001]), np.array([.020,.011,.006]))
        self.assertGreater(volume((grown-self.value)^middle), 1e-10)
        self.assertLess(volume(self.value-grown), 1e-12)
        far = box(np.array([-.01,-.01,-.01]), np.array([.002,.03,.02]))
        self.assertLess(volume((grown-self.value)^far), 1e-12)

    def test_blocked_expansion_preserves_exact_input(self):
        grown, _ = local_expansion(self.value, [dict(self.region)], self.value)
        self.assertLess(volume(grown-self.value)+volume(self.value-grown), 1e-12)

    def test_workpiece_exclusion_is_preserved(self):
        domain = box(np.array([-.01,-.01,-.01]), np.array([.05,.03,.02]))
        obstacle = box(np.array([.013,.005,0.]), np.array([.021,.007,.005]))
        grown, _ = local_expansion(self.value, [dict(self.region)], domain-obstacle)
        self.assertLess(volume(grown^obstacle), 1e-12)
        self.assertLess(volume(self.value-grown), 1e-12)

    def test_wide_five_mm_plate_is_not_a_thin_neck(self):
        plate = box(np.array([0.,0.,0.]), np.array([.04,.03,.005]))
        regions, _ = neck_regions(unpack(plate))
        self.assertEqual(regions, [])

    def test_small_roundtrip_losses_cannot_accumulate_across_patches(self):
        domain = box(np.array([-.01,-.01,-.01]), np.array([.05,.03,.02]))
        regions = [dict(self.region), dict(self.region)]
        calls = 0
        def lossy_roundtrip(mesh):
            nonlocal calls
            calls += 1
            cut = box(np.array([-.01,-.01,-.01]), np.array([calls*1e-6,.03,.02]))
            return solid(mesh)-cut
        grown, _ = local_expansion(self.value, regions, domain,geometry_check=lossy_roundtrip)
        self.assertEqual(calls, 2)
        self.assertLessEqual(volume(self.value-grown), 1e-10)
        self.assertTrue(regions[0]['accepted'])
        self.assertFalse(regions[1]['accepted'])


if __name__ == '__main__':
    unittest.main()
