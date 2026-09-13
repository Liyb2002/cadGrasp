"""Case isolation and replay of the saved target geometry; no support searches."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.cases import normalize_pose, pose_name, selected_pose
from step1.needs import ROOT, ContinuousNeeds, setup_geometry
from step3_scheculer import contacts
from step4_floor_contact import floor_contact
from step3_scheculer import run_all
class CaseIsolationTests(unittest.TestCase):
    def test_environment_and_numeric_cli_are_normalized(self):
        self.assertEqual(normalize_pose('3'), 'pose_3')
        for invalid in ('pose_0', 'pose_01', '../pose_2', 'pose_2/step1', '-1'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                normalize_pose(invalid)

    def test_paths_switch_together_and_restore_after_failure(self):
        with selected_pose('pose_1'):
            first = contacts.folder('B', 'step3.3_optimize_contact', 1)
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with selected_pose('pose_3'):
                    second = contacts.folder('B', 'step3.3_optimize_contact', 1)
                    self.assertNotEqual(first, second)
                    self.assertIn('pose_3', second.parts)
                    self.assertIn('pose_3', floor_contact.output_folder('B').parts)
                    raise RuntimeError('test failure')
            self.assertEqual(pose_name(), 'pose_1')

    def test_pipeline_children_inherit_selected_case(self):
        seen = []
        def invoke(*args, **kwargs):
            seen.append(os.environ['CADGRASP_POSE'])
            return type('Result', (), {'returncode': 0})()
        with selected_pose('pose_1'), patch.object(run_all, 'read_passed'), \
                patch.object(run_all, 'connection_verified', return_value=True), \
                patch.object(run_all.subprocess, 'run', side_effect=invoke):
            self.assertEqual(run_all.run(['B'], from_step=5, pose='pose_4'), 0)
            self.assertEqual(pose_name(), 'pose_1')
        self.assertTrue(seen)
        self.assertEqual(set(seen), {'pose_4'})

    def test_missing_case_does_not_fall_back_to_pose_one(self):
        with selected_pose('pose_999'), self.assertRaises(FileNotFoundError):
            setup_geometry('B')


class ExportedTargetsTests(unittest.TestCase):
    def test_all_targets_replay_in_step_one_with_correct_floor_and_work_surface(self):
        for number in range(1, 5):
            pose = f'pose_{number}'
            with self.subTest(pose=pose), selected_pose(pose):
                data = setup_geometry('B')
                domain = ContinuousNeeds(data)
                self.assertEqual(data['pose_id'], pose)
                self.assertGreaterEqual(domain.mesh.vertices[:, 2].min(), -1e-10)
                fraction = data['geometry']['work_area_m2'] / data['geometry']['total_area_m2']
                self.assertTrue(.08 <= fraction <= .15)
                self.assertTrue((-domain.normals[:, 2] > .35).all())
                report = json.loads((ROOT/'slides/setup/poses/B'/pose/'setup.json').read_text())
                self.assertEqual(report['checks']['work_components'], 1)
                with np.load(ROOT / data['provenance']['setup_snapshot']) as z:
                    contact = z['floor_contact_m']
                distances = np.linalg.norm(domain.mesh.vertices - contact, axis=1)
                self.assertLess(distances.min(), 1e-9)
                self.assertLess(abs(contact[2]), 1e-10)

    def test_two_ear_targets_have_distinct_tilts_and_contact_different_ears(self):
        reports = [json.loads((ROOT/f'slides/setup/poses/B/pose_{n}/setup.json').read_text())
                   for n in (3, 4)]
        self.assertEqual([r['checks']['floor_contact_raw_vertex_ids'] for r in reports], [[8], [70]])
        gravity = [r['checks']['gravity_in_mesh_frame'] for r in reports]
        self.assertLess(np.dot(*gravity), np.cos(np.deg2rad(20)))
        # Both targets place the high-y ear ends below the body.
        self.assertTrue(all(g[1] > .7 for g in gravity))
        self.assertNotEqual(reports[0]['work_face_ids'], reports[1]['work_face_ids'])

    def test_pose_one_preserves_original_transform_work_mask_and_pivot(self):
        for name in ('A1-f', 'B', 'C5'):
            original = ROOT/f'slides/obj_supp/area/demand_{name}_tip1.npz'
            copied = ROOT/f'slides/setup/poses/{name}/pose_1/setup.npz'
            self.assertEqual(original.read_bytes(), copied.read_bytes())


if __name__ == '__main__':
    unittest.main()
