"""Input discovery must not silently substitute poses or enable retired cases."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.registry import active_cases, active_objects, object_folder, task_snapshot


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, poses in [('new_shape', ['pose_2', 'pose_7']), ('retired', ['pose_1'])]:
            folder = self.root / name
            folder.mkdir()
            (folder / 'mesh.stl').write_bytes(b'input stub')
            (folder / 'tasks.json').write_text(json.dumps(dict(object=name, poses=poses)))
            for pose in poses:
                path = folder / 'tasks' / pose
                path.mkdir(parents=True)
                (path / 'setup.npz').write_bytes(b'snapshot stub')
        self.config(['new_shape'])

    def config(self, names):
        (self.root / 'cases.json').write_text(json.dumps(dict(active_objects=names)))

    def test_new_names_and_nonconsecutive_pose_ids_are_discovered(self):
        self.assertEqual(active_cases(self.root), [('new_shape', 'pose_2'), ('new_shape', 'pose_7')])
        self.assertEqual(task_snapshot('new_shape', 'pose_7', self.root),
                         self.root / 'new_shape/tasks/pose_7/setup.npz')

    def test_retired_object_stays_on_disk_but_is_not_runnable(self):
        self.assertTrue(object_folder('retired', self.root).is_dir())
        with self.assertRaisesRegex(ValueError, 'not in'):
            task_snapshot('retired', 'pose_1', self.root)

    def test_missing_pose_never_falls_back(self):
        with self.assertRaises(FileNotFoundError):
            task_snapshot('new_shape', 'pose_1', self.root)

    def test_manifest_snapshot_must_exist(self):
        (self.root / 'new_shape/tasks/pose_7/setup.npz').unlink()
        with self.assertRaises(FileNotFoundError):
            active_cases(self.root)

    def test_missing_mesh_duplicate_and_unsafe_names_are_rejected(self):
        for names, error in [(['missing'], FileNotFoundError),
                             (['new_shape', 'new_shape'], ValueError),
                             (['../escape'], ValueError)]:
            with self.subTest(names=names):
                self.config(names)
                with self.assertRaises(error):
                    active_objects(self.root)

    def test_other_objects_task_manifest_is_rejected(self):
        (self.root / 'new_shape/tasks.json').write_text(json.dumps(dict(object='retired', poses=['pose_2'])))
        with self.assertRaisesRegex(ValueError, 'another object'):
            active_cases(self.root)


if __name__ == '__main__':
    unittest.main()
