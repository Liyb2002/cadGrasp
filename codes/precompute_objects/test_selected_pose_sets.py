"""Check selected database groups against immutable full-load floor data."""
import json
import unittest
from collections import Counter
import numpy as np
from codes.precompute_objects.common_directions import ROOT, digest
from codes.precompute_objects.select_pose_sets import TARGETS


class SelectedPoseSetTests(unittest.TestCase):
    def test_all_objects_have_certified_5_20_5_selection(self):
        folders = sorted((ROOT / 'objects').glob('*/selected_pose_sets.json'))
        self.assertEqual(len(folders), 21)
        for path in folders:
            with self.subTest(object=path.parent.name):
                data = json.loads(path.read_text()); groups = data['sets']
                self.assertEqual(len(groups), 30)
                self.assertEqual(Counter(g['category'] for g in groups), TARGETS)
                self.assertEqual(len({tuple(sorted(g['poses'])) for g in groups}), 30)
                matrix = np.asarray(json.loads((path.parent / 'pose_sets.json').read_text())['directed_violating_counts'])
                poses = json.loads((path.parent / 'poses.json').read_text())['poses']
                normals = {r['pose_id']: np.asarray(r['T_world_mesh'])[:3, :3].T @ [0., 0., 1.] for r in poses}
                for group in groups:
                    indices = [int(p.split('_')[1]) - 1 for p in group['poses']]
                    self.assertEqual(bool(np.any(matrix[np.ix_(indices, indices)])), group['category'] == 'illegal')
                    n = np.array([normals[p] for p in group['poses']]); result = group['common_direction']
                    if group['category'] == 'legal_without_common_direction':
                        self.assertFalse(result['common_direction_exists'])
                        cert = result['certificate']; weights = np.array(cert['weights'])
                        self.assertGreater(weights.min(), 0.)
                        np.testing.assert_allclose(weights @ n, 0., atol=1e-9)
                        self.assertEqual(np.linalg.matrix_rank(n, tol=1e-9), 3)
                    elif group['category'] == 'legal_with_common_direction':
                        self.assertTrue(result['common_direction_exists'])
                        self.assertGreaterEqual(np.min(n @ result['direction']), -1e-9)
                for relative, sha in data['immutable_input_sha256'].items():
                    self.assertEqual(digest(path.parent / relative), sha)


if __name__ == '__main__':
    unittest.main()
