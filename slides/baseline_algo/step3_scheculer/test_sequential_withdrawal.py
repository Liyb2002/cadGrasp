from types import SimpleNamespace
import unittest

import numpy as np
import trimesh

from step2_local_support import withdrawal as W, surface as S
from step3_scheculer.sequential_withdrawal import WholeHeadWithdrawal


class WholeWithdrawalTests(unittest.TestCase):
    def fixture(self):
        mesh = trimesh.creation.box(extents=[1,1,1]).apply_translation([0,0,1])
        vectors = [[-1.,0,0], [1.,0,0], [0,1.,0]]
        catalogue = dict(vectors=vectors, global_allowed_directions=dict(ids=[0,1,2]))
        local = SimpleNamespace(mesh=mesh, catalogues=[catalogue],
            analyzers=[W.Analyzer(mesh, .01, catalogue)],
            clearance=SimpleNamespace(offsets=np.zeros_like(mesh.vertices)))
        geometry = SimpleNamespace(local=[local,local],
            problems=[SimpleNamespace(pose='pose_1'),SimpleNamespace(pose='pose_2')],
            transforms=[[np.eye(4),np.eye(4)], [np.eye(4),np.eye(4)]])
        tracker = WholeHeadWithdrawal(geometry)
        def entry(ident, side):
            face = int(np.flatnonzero(mesh.face_normals[:,0] == side)[0])
            contact = dict(candidate_id=ident, source_faces=np.array([face]),
                           triangles_m=S.fan(mesh.triangles[face]))
            head = trimesh.creation.box(extents=[.05,.1,.1]).apply_translation([side*.525,0,1])
            for task in range(2):
                tracker.cells_cache[(ident,task)] = [SimpleNamespace(vertices=head.vertices)]
            return dict(contact=contact, owner_task=0 if side < 0 else 1,
                        active_tasks=(0,) if side < 0 else (1,))
        return tracker, entry('left', -1), entry('right', 1)

    def test_actual_inactive_solid_removes_direction_and_union_replays(self):
        tracker, left, right = self.fixture()
        state, _ = tracker.append(tracker.initial(), left)
        self.assertEqual(state, ((0,2), (0,2)))
        after, _ = tracker.append(state, right)
        self.assertEqual(after, ((2,), (2,)))
        self.assertEqual(state, ((0,2), (0,2)))
        report = tracker.verify([left,right], after)
        self.assertTrue(report['passed'])
        self.assertTrue(report['inactive_heads_included'])
        self.assertTrue(all(row['checks'][0]['clear'] for row in report['per_pose']))

    def test_candidate_rejected_if_it_removes_last_inherited_direction(self):
        tracker, left, right = self.fixture()
        state = ((0,), (0,))
        after, check = tracker.append(state, right)
        self.assertIsNone(after)
        self.assertFalse(check['passed'])
        self.assertEqual(state, ((0,), (0,)))
        # Rejection cannot poison another branch's direction set.
        other, _ = tracker.append(((2,), (2,)), right)
        self.assertEqual(other, ((2,), (2,)))

    def test_final_replay_detects_an_incorrectly_restored_direction(self):
        tracker, left, right = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'independent whole-head replay'):
            tracker.verify([left,right], ((0,), (2,)))


if __name__ == '__main__':
    unittest.main()
