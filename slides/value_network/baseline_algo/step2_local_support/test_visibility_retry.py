"""Regression for the actual C5 pose_3 near-coincident shadow overlay."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch
from shapely import wkb
from shapely.geometry import box
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step2_local_support import visibility_retry as R
class VisibilityRetryTests(unittest.TestCase):
    def test_actual_shadow_inputs_and_blocked_probe_remain_covered(self):
        data = json.loads((Path(__file__).parent/'fixtures/shadow_union.json').read_text())
        shapes = [wkb.loads(s, hex=True) for s in data['shapes']]
        point = wkb.loads(data['point'], hex=True)
        result = R.outer_union(shapes, 1e-7)
        self.assertTrue(all(result.covers(s) for s in shapes))
        self.assertTrue(result.covers(point))

    def test_failed_overlay_uses_all_input_coordinates(self):
        shapes = [box(0, 0, 1, 1), box(3, 0, 4, 1)]
        with patch.object(R, 'unary_union', return_value=shapes[0]):
            result = R.outer_union(shapes, 1e-7)
        self.assertTrue(all(result.covers(s) for s in shapes))


if __name__ == '__main__': unittest.main()
