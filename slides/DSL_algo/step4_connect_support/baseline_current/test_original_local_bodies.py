"""Regressions for transferring the original local-body construction."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step4_connect_support.baseline_current.run_local_bodies import clip_floor, cheap_floor_test, original_case, original_placement, foot_menu
from step4_connect_support.baseline_current import build_coupled_saddle as S


class OriginalMethodTests(unittest.TestCase):
    def test_halfplane_clipping_preserves_actual_legal_region(self):
        square=np.array([[-1.,-1],[1,-1],[1,1],[-1,1]])
        clipped=clip_floor(square,[1.,0,0])
        self.assertAlmostEqual(Polygon(clipped).area,2.)
        self.assertTrue((clipped[:,0]>=0).all())
        self.assertEqual(len(clip_floor(square,[0.,0,-1.])),0)

    def test_independently_placed_object_pivot_is_not_fixture_material(self):
        bases=np.array([np.eye(3),[[0.,1,0],[0,0,1],[1,0,0]]])
        case=SimpleNamespace(tasks=[SimpleNamespace(floor=np.array([-1.,0,0])),SimpleNamespace(floor=np.array([0.,.1,0]))],
            demands=[np.array([[-.5,0],[.1,.2],[.1,-.2]]),np.array([[0.,.2],[.1,.3],[-.1,.3]])])
        self.assertTrue(cheap_floor_test(case,bases,np.zeros((2,3))))
        case.demands[0][0,0]=-1.1
        del case.demand_hulls
        self.assertFalse(cheap_floor_test(case,bases,np.zeros((2,3))))

    def test_original_successful_case_remains_in_the_search(self):
        import json
        case=original_case();placement=original_placement(case)
        self.assertTrue(cheap_floor_test(case,placement['bases'],placement['offsets']))
        np.testing.assert_allclose(placement['bases'][1],S.ROTATION,atol=0,rtol=0)
        expected=json.loads(Path(S.__file__).with_name('local_body_case.json').read_text())
        options=list(foot_menu(case,placement,case))
        self.assertEqual(len(options),1)
        self.assertEqual(options[0]['feet'],expected['feet_xy_m'])
        self.assertEqual(options[0]['assignments'],expected['assignments'])


if __name__=='__main__':unittest.main()
