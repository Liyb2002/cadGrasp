"""Exact subset search, immutable loads, and safe target-pose publication."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

import compatible_pose_search as S
import compatible_pose_export as E


class CliqueTests(unittest.TestCase):
    def test_matches_exhaustive_subsets_including_required_vertex(self):
        rng = np.random.default_rng(22)
        for _ in range(50):
            edges = rng.random((9, 9)) < .5
            edges = np.triu(edges, 1); edges |= edges.T
            adjacency = [set(np.flatnonzero(row)) for row in edges]
            for size in range(2, 6):
                for required in ((), (8,)):
                    valid = [g for g in itertools.combinations(range(9), size)
                             if set(required) <= set(g) and all(edges[i, j] for i, j in itertools.combinations(g, 2))]
                    found = S.find_clique(adjacency, size, required)
                    self.assertEqual(found is None, not valid)
                    if found is not None:
                        self.assertIn(tuple(sorted(found)), valid)

    def test_pairwise_success_without_triangle_cannot_certify_three(self):
        adjacency = [{1, 3}, {0, 2}, {1, 3}, {0, 2}]
        self.assertIsNone(S.find_clique(adjacency, 3))


class PosePublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.plan = Path(cls.temp.name)/'plan'
        with contextlib.redirect_stdout(io.StringIO()):
            cls.report = S.search('B', cls.plan, count=20, compatible_size=5, budget=120)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make_source(self, root):
        folder = root/'objects/B'; folder.mkdir(parents=True)
        for file in ('mesh.stl', 'poses.json'):
            shutil.copyfile(S.ROOT/'objects/B'/file, folder/file)
        (folder/'tasks.json').write_text('old task manifest')
        (folder/'tasks').mkdir(); (folder/'tasks/original.txt').write_text('old task')
        (folder/'video.mp4').write_bytes(b'old video')
        code = root/'codes/setup'; code.mkdir(parents=True)
        shutil.copyfile(S.ROOT/'codes/setup/compatible_pose_search.py', code/'compatible_pose_search.py')
        output = root/'slides/baseline_algo/output/B/pose3+4/step4'
        output.mkdir(parents=True); (output/'heads.png').write_bytes(b'old heads')
        return folder

    def test_small_sample_screen_is_prefix_of_full_fixed_loads(self):
        raw = trimesh.load(S.ROOT/'objects/B/mesh.stl', force='mesh')
        mesh = raw
        for _ in range(self.report['uniform_subdivision_rounds']):
            mesh = mesh.subdivide()
        with np.load(self.plan/'pose_1.npz') as arrays:
            transform, mask = arrays['T_world_mesh'], arrays['work_faces']
            _, _, small = S.local_demands(mesh, transform, mask, 'B', 1024)
            np.testing.assert_array_equal(small['need_wrench'], arrays['load_wrenches'][:1024])

    def test_full_export_replays_twenty_targets_and_preserves_history(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); folder = self.make_source(root)
            with patch.object(E, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                checked = E.publish('B', self.plan)
            self.assertEqual(checked['pose_count'], 20)
            self.assertGreater(checked['compatible_group_counts']['5'], 0)
            self.assertFalse(checked['placement_trajectory_verified'])
            self.assertFalse((folder/'video.mp4').exists())
            history = next((folder/'history').iterdir())
            self.assertEqual((history/'video.mp4').read_bytes(), b'old video')
            self.assertEqual((history/'tasks/original.txt').read_text(), 'old task')
            output = root/'slides/baseline_algo/output/B'
            self.assertFalse((output/'pose3+4').exists())
            self.assertEqual(next((output/'history').glob('*/pose3+4/step4/heads.png')).read_bytes(), b'old heads')

    def test_mid_publication_error_rolls_back_inputs_and_outputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); folder = self.make_source(root)
            old_poses = (folder/'poses.json').read_bytes()
            original = Path.rename
            def fail(source, target):
                if source.name == 'tasks' and source.parent.name.startswith('.B_pose_set_'):
                    raise OSError('injected installation failure')
                return original(source, target)
            with patch.object(E, 'ROOT', root), patch.object(Path, 'rename', fail), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, 'injected'):
                    E.publish('B', self.plan)
            self.assertEqual((folder/'poses.json').read_bytes(), old_poses)
            self.assertEqual((folder/'tasks.json').read_text(), 'old task manifest')
            self.assertEqual((folder/'tasks/original.txt').read_text(), 'old task')
            self.assertEqual((folder/'video.mp4').read_bytes(), b'old video')
            self.assertEqual((root/'slides/baseline_algo/output/B/pose3+4/step4/heads.png').read_bytes(), b'old heads')


if __name__ == '__main__':
    unittest.main()
