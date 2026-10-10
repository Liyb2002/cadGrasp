"""Physical invariants for the six preprocessing inputs."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from codes.precompute_objects import loads as L
from codes.precompute_objects import load_variants as V


class LoadVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT/'objects/B/poses/pose_1'
        cls.domain = json.loads((cls.source/'needs.json').read_text())

    def test_half_angles_are_explicit_and_state_cannot_be_inferred_from_a_string(self):
        names = [V.variant_name(angle, grounded) for angle in V.ANGLES for grounded in V.STATES]
        self.assertEqual(len(set(names)), 6)
        for invalid in [True, 45, 120, float('nan')]:
            with self.assertRaises(ValueError):
                V.variant_name(invalid, True)
        with self.assertRaises(ValueError):
            V.variant_name(30, 'airborne')

    def test_airborne_translates_every_world_position_and_retains_gravity(self):
        original = copy.deepcopy(self.domain)
        path = self.source/'angle_60_airborne/setup.npz'
        domain, shift = V.shifted_domain(original, 60, False, .01, path, 'test')
        np.testing.assert_allclose(np.asarray(domain['geometry']['vertices_m']),
                                   np.asarray(original['geometry']['vertices_m'])+shift, atol=1e-15, rtol=0)
        np.testing.assert_allclose(np.asarray(domain['frame']['T_world_mesh'])[:3, 3],
            np.asarray(original['frame']['T_world_mesh'])[:3, 3]+shift, atol=1e-15, rtol=0)
        self.assertFalse(domain['contact_model']['workpiece_floor_contact_allowed'])
        self.assertEqual(domain['load']['gravity_force_mg'], [0., 0., -1.])
        self.assertAlmostEqual(np.asarray(domain['geometry']['vertices_m'])[:, 2].min(), .01)
        self.assertEqual(domain['load']['cone_half_deg'], 60)
        self.assertEqual(domain['provenance']['setup_snapshot_cone_half_deg'], 60)
        self.assertEqual(domain['provenance']['native_setup_snapshot_cone_half_deg'], 30)
        self.assertEqual(original, self.domain)
        with self.assertRaises(ValueError):
            V.shifted_domain(original, 60, False, 0., path, 'test')

    def test_lifting_preserves_com_demand_but_changes_ground_origin_moment(self):
        com = np.array([.02, .01, .07])
        points = np.array([[.03, .01, .10], [.05, .02, .09]])
        forces = np.array([[.25, 0., -.1], [0., -.2, -.1]])
        shift = np.array([0., 0., .01])
        demand = L.demand(points, forces, com)
        raised = L.demand(points+shift, forces, com+shift)
        np.testing.assert_allclose(demand, raised, atol=1e-16, rtol=0)
        before = V.wrench_at_world_origin(demand, com)
        after = V.wrench_at_world_origin(raised, com+shift)
        np.testing.assert_allclose(after[:, 3:]-before[:, 3:],
                                   np.cross(shift, demand[:, :3]), atol=1e-16, rtol=0)
        self.assertGreater(np.linalg.norm(after[:, 3:]-before[:, 3:]), 0.)

    def test_airborne_downward_push_requires_weight_plus_push_and_zero_gravity_torque(self):
        com = np.array([.01, -.02, .1])
        gravity_only = L.demand(com, np.zeros(3), com)
        np.testing.assert_array_equal(gravity_only, [0., 0., 1., 0., 0., 0.])
        downward = L.demand(com, [0., 0., -.5], com)
        np.testing.assert_array_equal(downward, [0., 0., 1.5, 0., 0., 0.])

    def test_sampler_obeys_each_declared_half_angle_with_same_magnitude_limit(self):
        for angle in V.ANGLES:
            domain, _ = V.shifted_domain(self.domain, angle, True, .01,
                self.source/f'angle_{angle}_grounded/setup.npz', 'test')
            samples = L.sample_needs(L.ContinuousNeeds(domain), count=128)
            parameters = np.asarray(samples['parameters'])
            self.assertLessEqual(parameters[:, 2].max(), np.deg2rad(angle))
            self.assertLessEqual(parameters[:, 4].max(), .5)
            self.assertEqual(len(samples['need_wrench']), 128)

    def test_published_variant_has_no_airborne_pivot_or_object_floor_reaction(self):
        if not (self.source/'load_variants.json').exists():
            self.skipTest('Run B/pose_1 preprocessing pilot first')
        for angle in V.ANGLES:
            ground = V.read_variant('B', 'pose_1', angle, True)
            air = V.read_variant('B', 'pose_1', angle, False)
            np.testing.assert_array_equal(ground['arrays']['need_wrench'], air['arrays']['need_wrench'])
            self.assertEqual(air['workpiece_floor_contact_points_m'].shape, (0, 3))
            with np.load(air['folder']/'setup.npz') as saved:
                self.assertEqual(saved['floor_contact_m'].shape, (0, 3))
                self.assertFalse(bool(saved['grounded']))
            with np.load(air['folder']/'floor_contact.npz') as saved:
                self.assertNotIn('original_pivot_m', saved.files)
                self.assertFalse(bool(saved['workpiece_floor_contact_allowed']))
        original = json.loads((self.source/'samples.json').read_text())
        default = V.read_variant('B', 'pose_1', 30, True)
        np.testing.assert_array_equal(original['need_wrench'], default['arrays']['need_wrench'])


if __name__ == '__main__':
    unittest.main()
