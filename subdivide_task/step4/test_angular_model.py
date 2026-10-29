"""Physical regression checks for task rotation during juxtaposition."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from .angular_model import AngularTaskModel, roll_to_match
from .station_seed import common_installation
from fixture_state import installed_state
from subdivide_task.step1.pose_coverage import spatial_workspace_mask
from copy import deepcopy


class RollMathTests(unittest.TestCase):
    def test_best_roll_matches_dense_orientation_search(self):
        rng = np.random.default_rng(42)
        axis = np.array([0., -2**-.5, -2**-.5])
        for _ in range(4):
            reference, target = Rotation.random(2, random_state=rng).as_matrix()
            angle = roll_to_match(reference, target, axis)
            result = Rotation.from_rotvec(axis*np.radians(angle)).as_matrix() @ reference
            score = np.trace(target.T @ result)
            rotations = Rotation.from_rotvec(np.radians(np.arange(0, 360, .1))[:, None]*axis).as_matrix()
            dense = np.einsum('ij,aji->a', target.T, rotations @ reference).max()
            self.assertGreaterEqual(score+1e-10, dense)


class AngularPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AngularTaskModel()
        cls.initial, _ = cls.model.initial()

    def setUp(self):
        self.model.activate(self.initial)

    def test_angle_is_part_of_state_identity_and_serialization(self):
        q = self.initial.copy(); q.angles_deg[2] = 13.72
        self.assertNotEqual(q.key(), self.initial.key())
        np.testing.assert_array_equal(q.copy().angles_deg, q.angles_deg)
        np.testing.assert_array_equal(q.arrays()['axial_angles_deg'], q.angles_deg)

    def test_changing_host_preserves_every_other_actual_world_task(self):
        m = self.model; q = common_installation(m, self.initial)
        before = [installed_state(m.native, q, int(q.hosts[k])) @ q.placements[k] for k in q.active]
        trial = m.spin_guest(q, 0, 37.25)
        for k in trial.active[1:]:
            actual = installed_state(m.native, trial, int(trial.hosts[k])) @ trial.placements[k]
            np.testing.assert_allclose(actual, before[k], atol=1e-12, rtol=0)
        self.assertFalse(np.allclose(m.native[0, :3, :3], before[0][:3, :3]))

    def test_loads_floor_and_cache_refresh_and_rollback_together(self):
        m = self.model; old = m.tasks[3].targets.copy()
        m.force_cache['stale_at_old_angle'] = old
        q = m.spin_guest(self.initial, 3, 23.17)
        self.assertNotIn('stale_at_old_angle', m.force_cache)
        task = m.tasks[3]
        self.assertFalse(np.allclose(task.targets, old))
        np.testing.assert_allclose(task.forward_nodes(np.arange(len(task.targets))), task.targets_7d, atol=1e-13)
        self.assertAlmostEqual(task.floor[2], 0., places=12)
        np.testing.assert_allclose(task.domain.mesh.vertices[m.floor_vertices[3]], task.floor, atol=1e-13)
        m.activate(self.initial)
        np.testing.assert_allclose(m.tasks[3].targets, old, atol=1e-13)

    def test_actual_centered_region_stays_in_the_same_translated_work_window(self):
        m = self.model; q = m.spin_guest(self.initial, 7, 113.48)
        row = m.execution(q, 7)
        workspace = deepcopy(m.station['workspace']); shift = np.asarray(row['work_window_translation_m'])
        workspace['center_world_m'] = (np.asarray(workspace['center_world_m'])+shift).tolist()
        workspace['spatial_cone']['apex_world_m'] = (np.asarray(workspace['spatial_cone']['apex_world_m'])+shift).tolist()
        transform = np.asarray(row['world_from_object'])
        surface = m.mesh.triangles[m.tasks[7].domain.work_ids] @ transform[:3, :3].T+transform[:3, 3]
        self.assertTrue(spatial_workspace_mask(surface, workspace).all())
        np.testing.assert_allclose(row['object_COM_world_m'][:2], m.center_xy, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
