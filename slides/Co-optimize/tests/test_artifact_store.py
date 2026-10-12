"""Verify final-artifact preservation and last-consumer cleanup."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helper_func'))
import artifact_store as A


class ArtifactLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.results = self.root / 'data/results'
        self.visuals = self.root / 'output'
        self.patches = [patch.object(A, 'RESULTS', self.results), patch.object(A, 'VISUALS', self.visuals)]
        for p in self.patches:
            p.start()
        self.addCleanup(self.directory.cleanup)
        for p in self.patches:
            self.addCleanup(p.stop)

    def test_old_stage_directory_resolves_to_numerical_results(self):
        stage = Path('B/pose1+2/step4/step4.2/final')
        (self.visuals / stage).mkdir(parents=True)
        numerical = self.results / stage
        numerical.mkdir(parents=True)
        (numerical / 'layout.npz').write_bytes(b'unchanged final layout')
        self.assertEqual(A.resolve_artifact(self.visuals / stage), numerical)
        self.assertEqual(A.digest(self.visuals / stage / 'layout.npz'), A.digest(numerical / 'layout.npz'))
        with self.assertRaises(FileNotFoundError):
            A.digest(self.visuals / stage / 'missing.npz')

    def test_publishing_moves_only_visuals_and_keeps_links_readable(self):
        stage = self.results / 'A/pose1+2/step5/step5.2/final'
        stage.mkdir(parents=True)
        (stage / 'overview.png').write_bytes(b'original image')
        (stage / 'support.obj').write_bytes(b'original solid')
        (stage / 'index.html').write_text('<img src="overview.png"><a href="support.obj">OBJ</a>')
        original = A.digest(stage / 'overview.png')
        A.publish_visuals(stage)
        visual = self.visuals / stage.relative_to(self.results)
        self.assertEqual(A.digest(visual / 'overview.png'), original)
        self.assertFalse((stage / 'overview.png').exists())
        self.assertEqual((stage / 'support.obj').read_bytes(), b'original solid')
        html = (visual / 'index.html').read_text()
        self.assertIn('src="overview.png"', html)
        self.assertIn(str(Path(__import__('os').path.relpath(stage / 'support.obj', visual))), html)
        self.assertEqual(A.resolve_artifact(stage / 'overview.png'), visual / 'overview.png')

    def test_b_publishes_images_without_html_and_preserves_numerical_artifacts(self):
        stage = self.results / 'B/pose1+2/step5/step5.2/final'
        stage.mkdir(parents=True)
        (stage / 'overview.png').write_bytes(b'accepted image')
        (stage / 'support.obj').write_bytes(b'accepted solid')
        (stage / 'index.html').write_text('<img src="overview.png">')
        A.publish_visuals(stage)
        visual = self.visuals / stage.relative_to(self.results)
        self.assertEqual((visual / 'overview.png').read_bytes(), b'accepted image')
        self.assertEqual((stage / 'support.obj').read_bytes(), b'accepted solid')
        gallery = self.results / 'B/current_index.html'
        gallery.write_text('obsolete HTML reader')
        A.publish_gallery(gallery)
        self.assertFalse(gallery.exists())
        self.assertFalse(list(self.visuals.rglob('*.html')))
        self.assertFalse(list(self.results.rglob('*.html')))

    def test_last_consumer_cleanup_retains_every_final_input(self):
        stage = self.results / 'B/pose1+2/step4/step4.2/final'
        stage.mkdir(parents=True)
        finals = ['layout.npz', 'pose_1_force.npz', 'support.obj', 'oracle_connected_witness.obj', 'base_oracle.json']
        for name in finals:
            (stage / name).write_bytes(name.encode())
        for folder in ['process_states', 'mesh_states', 'data/worker_logs']:
            (stage / folder).mkdir(parents=True)
            (stage / folder / 'temporary.npz').write_bytes(b'temporary')
        for name in ['trace.json', 'process.json', 'sampled_layout.npz']:
            (stage / name).write_bytes(b'consumed')
        A.cleanup_stage(stage)
        self.assertEqual(sorted(p.name for p in stage.iterdir() if p.is_file()), sorted(finals))
        self.assertFalse((stage / 'process_states').exists())
        A.cleanup_stage(stage)  # Safe after repeated cleanup.

    def test_batch_cleanup_keeps_verified_final_manifests(self):
        batch = self.results / 'B/data/completed'
        (batch / 'executed_sources').mkdir(parents=True)
        for name in ['summary.json', 'verification.json', 'progress.json']:
            (batch / name).write_bytes(b'bytes')
        (batch / 'executed_sources/run.py').write_bytes(b'consumed source copy')
        A.finish_batch(batch)
        self.assertTrue((batch / 'summary.json').exists())
        self.assertTrue((batch / 'verification.json').exists())
        self.assertFalse((batch / 'progress.json').exists())
        self.assertFalse((batch / 'executed_sources').exists())


if __name__ == '__main__':
    unittest.main()
