"""Integration: five shared physical placements survive Step4 → Step5 → movie."""
import json
import unittest
from unittest.mock import patch
import numpy as np
from whole_task_2.step1.common import DEFAULT_OUTPUT
from whole_task_2.step4.compact_grouped import shared_static_bounds
from .source_model import restore
from .grouped import SharedWorkflow, Workflow, installation_groups


class GroupedHandoffTests(unittest.TestCase):
    @unittest.skipUnless((DEFAULT_OUTPUT/'B/step4/grouped13_focused/report.json').exists(),
        'Historical five-placement output was retired; current membrane handoff is independently replayed')
    def test_certified_transforms_and_five_memberships_survive_handoff(self):
        folder = DEFAULT_OUTPUT/'B/step4/grouped13_focused'
        report = json.loads((folder/'report.json').read_text())
        saved = json.loads((folder/'execution.json').read_text())['poses']
        model, layout = restore(report, folder/'step4.2/layout.npz')
        execution = [model.execution(layout, k) for k in layout.active]
        for actual, expected in zip(execution, saved):
            for key in ('world_from_object', 'world_from_support', 'head_world_m'):
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=0)
        rows = {k: dict(fixture=x['world_from_support']) for k, x in enumerate(execution)}
        groups = installation_groups(rows)
        self.assertEqual(groups, [[0, 2, 5], [1, 4, 9], [3, 6], [7, 11, 12], [8, 10]])
        with np.load(folder/'step4.2/layout.npz') as z:
            arrays = {name: z[name] for name in z.files}
        metrics = shared_static_bounds(model.mesh.vertices, arrays, pose_names=model.poses, execution=execution)
        self.assertEqual(metrics['support_pose_count'], 5)
        self.assertEqual(metrics['object_pose_counts_per_support_pose'], [3, 3, 2, 3, 2])
        data = dict(rows=rows, order=list(rows))
        def native_workflow(instance, received):
            self.assertEqual(received['order'], [0, 2, 5, 1, 4, 9, 3, 6, 7, 11, 12, 8, 10])
            instance.segments = [dict(label='seat_fixture') for _ in groups]
        with patch.object(Workflow, '__init__', native_workflow):
            SharedWorkflow(data)


if __name__ == '__main__':
    unittest.main()
