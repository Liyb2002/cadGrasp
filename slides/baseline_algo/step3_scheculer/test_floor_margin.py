"""A group floor filter checks the full finite patch, leaving local tasks independent."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from step3_scheculer.floor_margin import contact_check, FloorMarginGeometry
from step3_scheculer.run_independent import IndependentSearch


def contact(z=(.01, .01, .01)):
    return dict(candidate_id='pose_1_C001', triangles_m=np.array([[
        [0., 0., z[0]], [.01, 0., z[1]], [0., .01, z[2]]]]))


class ContactFloorMarginTests(unittest.TestCase):
    def test_centroid_above_margin_does_not_hide_low_vertex(self):
        c = contact((.001, .01, .01))
        self.assertGreater(c['triangles_m'].mean(axis=(0, 1))[2], .002)
        check = contact_check(c, ['pose_1'], [np.eye(4)])
        self.assertFalse(check['passed'])
        self.assertEqual(check['per_pose'][0]['minimum_contact_height_m'], .001)

    def test_exact_two_mm_is_allowed(self):
        check = contact_check(contact((.002, .002, .002)), ['pose_1'], [np.eye(4)])
        self.assertTrue(check['passed'])

    def test_rotated_inactive_pose_can_reject_safe_owner_patch(self):
        rotation = np.array([[0., 0., -1., 0.], [0., 1., 0., 0.],
                             [1., 0., 0., .001], [0., 0., 0., 1.]])
        check = contact_check(contact(), ['pose_1', 'pose_2'], [np.eye(4), rotation])
        self.assertTrue(check['per_pose'][0]['passed'])
        self.assertFalse(check['per_pose'][1]['passed'])
        self.assertFalse(check['passed'])

    def test_missing_geometry_never_passes(self):
        self.assertFalse(contact_check(dict(triangles_m=np.empty((0, 3, 3))),
                                       ['pose_1'], [np.eye(4)])['passed'])

    def test_final_check_recomputes_geometry_instead_of_trusting_candidate_flag(self):
        geometry = FloorMarginGeometry.__new__(FloorMarginGeometry)
        geometry.floor_poses = ['pose_1']
        geometry.floor_transforms = [np.eye(4)]
        geometry.floor_clearance_m = .002
        entry = dict(contact=contact((.001, .01, .01)), all_pose_floor_margin=dict(passed=True))
        self.assertFalse(geometry.floor_check([entry])['passed'])

    def test_group_mode_requires_separate_output_to_avoid_unfiltered_cache(self):
        with patch('step3_scheculer.run_independent.task_poses', return_value=['pose_1', 'pose_2']):
            with self.assertRaisesRegex(ValueError, 'separate output_root'):
                IndependentSearch('B', 'pose_1', floor_poses=['pose_1', 'pose_2'])

    def test_floor_context_does_not_enter_force_or_local_withdrawal_tasks(self):
        with TemporaryDirectory() as tmp:
            owner = SimpleNamespace(pose='pose_1', targets=np.zeros((2, 7)))
            other = SimpleNamespace(pose='pose_2', targets=np.ones((2, 7)))
            geometry = SimpleNamespace(save=lambda *args: None)
            with patch('step3_scheculer.run_independent.task_poses', return_value=['pose_1', 'pose_2']), \
                 patch('step3_scheculer.run_independent.prepare_task', side_effect=[owner, other]), \
                 patch('step3_scheculer.run_independent.input_hashes', return_value={}), \
                 patch('step3_scheculer.run_independent.F.FloorMarginGeometry', return_value=geometry) as factory:
                search = IndependentSearch('B', 'pose_1', floor_poses=['pose_1', 'pose_2'], output_root=Path(tmp))
            factory.assert_called_once_with(owner, [owner, other], count=200, clearance_m=.002)
            self.assertEqual(search.problems, [owner])
            np.testing.assert_array_equal(search.original_targets, [owner.targets])


if __name__ == '__main__':
    unittest.main()
