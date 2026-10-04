"""Dataset consumers must reuse exact loads and reject mismatched inputs."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from codes.precompute_objects import dataset, loads
from codes.precompute_objects.cases import selected_pose
from codes.precompute_objects.registry import active_cases, task_snapshot
from codes.precompute_objects.run import compatible_sets
import numpy as np


class DatasetTests(unittest.TestCase):
    def test_all_objects_expose_thirty_dataset_poses(self):
        cases=active_cases()
        self.assertEqual(len(cases),630)
        self.assertTrue(all('/poses/' in str(task_snapshot(*case)) for case in cases))

    def test_fixed_sample_reuse_never_calls_sampler(self):
        with tempfile.TemporaryDirectory() as tmp, selected_pose('pose_30'), patch.object(loads,'sample_needs',side_effect=AssertionError('resampled')):
            target=Path(tmp)
            loads.build('B',output_folder=target)
            source=ROOT/'objects/B/poses/pose_30'
            for filename in ('needs.json','samples.json'):
                self.assertEqual((target/filename).read_bytes(),(source/filename).read_bytes())

    def test_nonstandard_sample_counts_cannot_change_dataset(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            loads.build('B',count=100,output_folder=tmp)

    def test_size_six_sets_are_distinct_and_accepted(self):
        rows=dataset.selected_sets('B',6,19,4)
        self.assertEqual(len(rows),4)
        self.assertEqual(len({tuple(row['poses']) for row in rows}),4)
        self.assertTrue(all(row['passed'] and len(row['poses'])==6 for row in rows))

    def test_insufficient_cliques_do_not_relax_requirements(self):
        T=np.eye(4)
        with self.assertRaisesRegex(ValueError,'Only'):
            compatible_sets([T,T],[np.zeros((3,2)),np.zeros((3,2))])

    def test_domains_can_be_pickled_for_algorithm_workers(self):
        domain=loads.ContinuousNeeds.read(ROOT/'objects/B/poses/pose_1/needs.json')
        restored=pickle.loads(pickle.dumps(domain))
        np.testing.assert_array_equal(restored.mesh.vertices,domain.mesh.vertices)

    def test_manifest_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'objects/shape';folder.mkdir(parents=True)
            (folder/'poses.json').write_text('{}')
            (folder/'pose_sets.json').write_text(json.dumps(dict(passed=True,pose_manifest_sha256='wrong')))
            with patch.object(dataset,'ROOT',Path(tmp)),self.assertRaisesRegex(ValueError,'stale'):
                dataset.read_sets('shape')

if __name__=='__main__':unittest.main()
