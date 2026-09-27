"""Regressions for the physical handoff assumptions of the concept film."""
import numpy as np
import unittest

from separate_loading import Sequence
from separate_scene import ground_balance


def check_single_arm_never_transports_both_bodies(sequence):
    for row in sequence.segments:
        for fraction in (.1, .5, .9):
            state = sequence.state(row['start']+fraction*(row['end']-row['start']))
            moved = [key for key in ('object', 'blue')
                     if not np.allclose(state[key], row['poses0'][key])]
            assert len(moved) <= 1
            if row['moving']:
                assert moved == [row['moving']]
                np.testing.assert_allclose(
                    state['hand'], state[row['moving']]@sequence.grasps[row['moving']], atol=1e-10)


def check_unheld_bodies_have_the_required_support(sequence):
    # Staging must not merely freeze an unstable object while the arm is busy.
    for key, mesh, park in (('object', sequence.obj, sequence.object_park),
                            ('blue', sequence.blue, sequence.blue_park)):
        balance = ground_balance(mesh, park)
        assert balance['ground_only_equilibrium'], key
        assert abs(balance['minimum_z_m']) < 1e-8
    for row in sequence.segments:
        state = sequence.state((row['start']+row['end'])/2)
        if not np.allclose(state['object'], sequence.object_park, atol=1e-10) and row['moving'] != 'object':
            assert row['case'] is not None
            np.testing.assert_allclose(state['blue'], sequence.cases[row['case']]['transform'], atol=1e-10)


def check_all_three_tasks_follow_separate_loading_and_reverse_unloading(sequence):
    for i, case in enumerate(sequence.cases):
        rows = [row for row in sequence.segments if row['case'] == i]
        releases = [r['label'] for r in rows if r['label'].endswith('_release')]
        assert releases == ['install_blue_release', 'install_object_release',
                            'remove_object_release', 'remove_blue_release']
        tasks = [r for r in rows if r['label'] == 'task']
        assert len(tasks) == 1
        task = sequence.state((tasks[0]['start']+tasks[0]['end'])/2)
        np.testing.assert_allclose(task['object'], case['transform'], atol=1e-10)
        np.testing.assert_allclose(task['blue'], case['transform'], atol=1e-10)
        assert task['gap'] == .07
        assert not task['moving']
    final = sequence.state(sequence.duration)
    np.testing.assert_allclose(final['object'], sequence.object_park, atol=1e-10)
    np.testing.assert_allclose(final['blue'], sequence.blue_park, atol=1e-10)


def check_ground_balance_diagnostic_against_box_counterexample():
    import trimesh
    box = trimesh.creation.box(extents=[1., 1., 1.])
    upright = np.eye(4); upright[2, 3] = .5
    assert ground_balance(box, upright)['ground_only_equilibrium']
    tilted = trimesh.transformations.rotation_matrix(.3, [1., .3, 0.])
    tilted[2, 3] = -(box.vertices@tilted[:3, :3].T)[:, 2].min()
    assert not ground_balance(box, tilted)['ground_only_equilibrium']


class SeparateLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sequence = Sequence()

    def test_original_shape_landmarks(self):
        # Guard against silently substituting another baseline result.
        from original_geometry import TERMINALS
        self.assertEqual(self.sequence.meta['selected_contact_ids'],['C140','C093'])
        self.assertEqual(self.sequence.meta['head_cell_count'],71)
        np.testing.assert_allclose(self.sequence.meta['blue_grip_local'],TERMINALS.mean(0),atol=1e-12)
        self.assertTrue(all(c['base_layout']=='shared_ground_frame_three_sockets'
                            for c in self.sequence.cases))
        # The full closed outer ring surrounds all three original stations.
        section=self.sequence.base.section(plane_origin=[0,0,.005],plane_normal=[0,0,1])
        self.assertIsNotNone(section)
        self.assertGreater(max(np.ptp(path[:,1]) for path in section.discrete),.40)

    def test_single_body_transport(self):
        check_single_arm_never_transports_both_bodies(self.sequence)

    def test_support_during_handoffs(self):
        check_unheld_bodies_have_the_required_support(self.sequence)

    def test_three_complete_cycles(self):
        check_all_three_tasks_follow_separate_loading_and_reverse_unloading(self.sequence)

    def test_floor_equilibrium_counterexample(self):
        check_ground_balance_diagnostic_against_box_counterexample()


if __name__ == '__main__':
    unittest.main()
