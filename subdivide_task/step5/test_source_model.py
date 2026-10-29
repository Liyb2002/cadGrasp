"""A Step5 handoff must preserve every optimized Step4 task and installation."""
import json
from pathlib import Path
import unittest
import numpy as np

from subdivide_task.step1.common import DEFAULT_OUTPUT
from .source_model import restore


class ContinuousHandoffTests(unittest.TestCase):
    @unittest.skipUnless((DEFAULT_OUTPUT/'B/step4/continuous13/report.json').exists(),
        'Historical continuous13 output was retired; current membrane handoff is independently replayed')
    def test_every_optimized_pose_and_floor_state_survives_handoff(self):
        folder = DEFAULT_OUTPUT/'B/step4/continuous13'
        report = json.loads((folder/'report.json').read_text())
        saved = json.loads((folder/'execution.json').read_text())['poses']
        model, layout = restore(report, folder/'step4.2/layout.npz')
        self.assertEqual(len(layout.active), 13)
        np.testing.assert_array_equal(layout.angles_deg, report['final_axial_angles_deg'])
        for k in layout.active:
            actual = model.execution(layout, k)
            for key in ['world_from_object', 'world_from_support', 'head_world_m']:
                np.testing.assert_allclose(actual[key], saved[k][key], atol=1e-12, rtol=0)
            self.assertEqual(actual['object_floor_reaction_enabled'], saved[k]['object_floor_reaction_enabled'])
            self.assertAlmostEqual(actual['object_floor_gap_m'], saved[k]['object_floor_gap_m'], places=12)


if __name__ == '__main__':
    unittest.main()
