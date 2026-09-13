"""Serial/process equivalence and ordered publication with rejected candidates."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step2_local_support import parallel_directions as P, withdrawal as D
from step2_local_support import insertion_directions as I


class ParallelDirectionsTests(unittest.TestCase):
    def fixture(self):
        mesh = trimesh.creation.box(extents=[2., 2., 2.])
        mesh.apply_translation([0., 0., 1.1])
        catalogue = dict(vectors=[[1.,0,0], [-1.,0,0], [0,0,1]],
            global_allowed_directions=D.normalize([0,1,2]),
            preferred_withdrawal_direction=[1.,0,0])
        contacts = []
        for index, sign in [(0, 1), (2, -1)]:
            face = int(np.flatnonzero(mesh.face_normals[:,0]*sign > .9)[0])
            triangle = mesh.triangles[face]
            triangle = triangle.mean(axis=0)+.2*(triangle-triangle.mean(axis=0))
            contacts.append(dict(candidate_index=index, candidate_id=f'C{index+1:03d}',
                center_m=triangle.mean(axis=0), center_face=face, radius_m=.3,
                triangles_m=triangle[None], source_faces=np.array([face]),
                triangle_areas_m2=np.array([trimesh.triangles.area(triangle[None])[0]])))
        contacts.insert(1, dict(candidate_index=1, candidate_id='C002'))
        return mesh, .02, catalogue, contacts

    def test_all_direction_records_match_serial_with_rejected_slots(self):
        args = self.fixture()
        serial = list(P.rows(*args, workers=1))
        parallel = list(P.rows(*args, workers=2))
        self.assertEqual(serial, parallel)
        self.assertEqual([r['candidate_index'] for r in parallel], [0,1,2])
        self.assertEqual(parallel[1]['status'], 'rejected_step2_geometry')
        self.assertTrue(parallel[0]['has_certified_direction'])
        self.assertIn(1, parallel[0]['locked_direction_ids'])

    def test_empty_and_invalid_candidates_need_no_geometry(self):
        self.assertEqual(list(P.rows(None, None, None, [], workers=4)), [])
        self.assertEqual(list(P.rows(None, None, None,
            [dict(candidate_index=8,candidate_id='C009')], workers=4)), [P.rejected(8,'C009')])

    def test_resumed_suffix_keeps_original_indices(self):
        mesh, depth, catalogue, contacts = self.fixture()
        expected = list(P.rows(mesh, depth, catalogue, contacts, workers=1))
        suffix = list(P.rows(mesh, depth, catalogue, contacts[1:], workers=2))
        self.assertEqual(suffix, expected[1:])

    def test_invalid_worker_count_is_rejected(self):
        for value in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                list(P.rows(None, None, None, [], workers=value))

    def test_checkpoint_batches_and_force_preserve_a_resumable_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'progress.json'
            checkpoint = P.Checkpoint(path, 'test', {'inputs':{}})
            rows = []
            for i in range(23):
                rows.append(P.rejected(i, f'C{i+1:03d}'))
                checkpoint.save(rows)
            self.assertEqual(checkpoint.write_count, 2)
            self.assertEqual(len(json.loads(path.read_text())['candidates']), 20)
            checkpoint.save(rows, force=True)
            self.assertEqual(json.loads(path.read_text())['candidates'], rows)
            self.assertFalse(path.with_name('progress.json.tmp').exists())
            resumed = P.Checkpoint(path, 'test', {'inputs':{}}, initial_count=23)
            resumed.save(rows, force=True)
            self.assertEqual(resumed.write_count, 0)

    def test_failed_write_cannot_damage_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'progress.json'
            checkpoint = P.Checkpoint(path, 'test', {})
            checkpoint.save([P.rejected(0,'C001')], force=True)
            original = path.read_bytes()
            with patch.object(Path, 'write_text', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    checkpoint.save([P.rejected(0,'C001'),P.rejected(1,'C002')], force=True)
            self.assertEqual(path.read_bytes(), original)

    def test_interrupted_run_flushes_prefix_and_resume_finishes_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as mocks:
            out = Path(directory)/'insertion_directions.json'
            data = SimpleNamespace(valid=np.zeros(23, dtype=bool))
            source = dict(normal_depth_m=.02, patches=[dict(id=f'C{i+1:03d}') for i in range(23)])
            mocks.enter_context(patch.object(I, 'path', return_value=out))
            mocks.enter_context(patch.object(I.P, 'read', return_value=(SimpleNamespace(mesh=None,work_ids=[]),data,source)))
            mocks.enter_context(patch.object(I.I, 'hashes', return_value={}))
            mocks.enter_context(patch.object(I, 'code_hashes', return_value={}))
            mocks.enter_context(patch.object(I.D, 'make_catalogue', return_value={}))
            def interrupted(*args):
                for index in range(7):
                    yield P.rejected(index, f'C{index+1:03d}')
                raise RuntimeError('worker failed')
            with patch.object(I.PD, 'rows', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    I.run('test', workers=4)
            progress = out.with_name('insertion_directions_progress.json')
            self.assertEqual(len(json.loads(progress.read_text())['candidates']), 7)
            result = I.run('test', workers=4)
            self.assertEqual(result['execution']['resumed_candidate_count'], 7)
            self.assertEqual([r['candidate_index'] for r in result['candidates']], list(range(23)))
            self.assertFalse(progress.exists())


if __name__ == '__main__':
    unittest.main()
