"""Physical and migration invariants for the three angle-only inputs."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from codes.precompute_objects import loads as L
from codes.precompute_objects import load_variants as V
from codes.precompute_objects.collapse_load_variants import detach_legacy_aliases


class LoadVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT/'objects/B/poses/pose_1'
        cls.domain = json.loads((cls.source/'needs.json').read_text())

    def test_angles_have_no_contact_state_dimension(self):
        self.assertEqual([V.variant_name(a) for a in V.ANGLES], ['angle_15','angle_30','angle_60'])
        for invalid in [True, 45, 120, float('nan')]:
            with self.assertRaises(ValueError): V.variant_name(invalid)
        with self.assertRaises(TypeError): V.variant_name(30, False)

    def test_angle_change_retains_geometry_gravity_and_native_source(self):
        original = copy.deepcopy(self.domain)
        original['contact_model'] = {'workpiece_floor_contact_allowed': True}
        domain = V.angle_domain(original, 60, self.source/'angle_60/setup.npz', 'test')
        self.assertEqual(domain['geometry'], self.domain['geometry'])
        self.assertEqual(domain['frame'], self.domain['frame'])
        self.assertNotIn('contact_model', domain)
        self.assertEqual(domain['load']['gravity_force_mg'], [0., 0., -1.])
        self.assertEqual(domain['provenance']['setup_snapshot_cone_half_deg'], 60)
        self.assertEqual(domain['provenance']['native_setup_snapshot_cone_half_deg'], 30)
        self.assertTrue(original['contact_model']['workpiece_floor_contact_allowed'])
        self.assertEqual(self.domain['load']['cone_half_deg'], 30)

    def test_translation_preserves_com_demand_but_changes_world_moment(self):
        com = np.array([.02, .01, .07])
        points = np.array([[.03, .01, .10], [.05, .02, .09]])
        forces = np.array([[.25, 0., -.1], [0., -.2, -.1]])
        shift = np.array([.015, -.004, .01])
        demand = L.demand(points, forces, com)
        translated = L.demand(points+shift, forces, com+shift)
        np.testing.assert_allclose(demand, translated, atol=1e-16, rtol=0)
        before = V.wrench_at_world_origin(demand, com)
        after = V.wrench_at_world_origin(translated, com+shift)
        np.testing.assert_allclose(after[:, 3:]-before[:, 3:],
                                   np.cross(shift, demand[:, :3]), atol=1e-16, rtol=0)
        self.assertGreater(np.linalg.norm(after[:, 3:]-before[:, 3:]), 0.)

    def test_downward_push_still_requires_weight_plus_push(self):
        com = np.array([.01, -.02, .1])
        np.testing.assert_array_equal(L.demand(com, np.zeros(3), com), [0.,0.,1.,0.,0.,0.])
        np.testing.assert_array_equal(L.demand(com, [0.,0.,-.5], com), [0.,0.,1.5,0.,0.,0.])

    def test_detached_native_files_survive_deleting_previous_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); old = folder/'angle_30_grounded'; old.mkdir()
            for i, filename in enumerate(V.LEGACY_FILES):
                (old/filename).write_bytes(bytes([i,10,255])*17)
                (folder/filename).symlink_to(Path(old.name)/filename)
            expected = {file:(folder/file).read_bytes() for file in V.LEGACY_FILES}
            detach_legacy_aliases(folder)
            shutil.rmtree(old)
            for filename, data in expected.items():
                self.assertFalse((folder/filename).is_symlink())
                self.assertEqual((folder/filename).read_bytes(), data)

    def test_published_angles_keep_default_samples_and_remove_state_fields(self):
        manifest = json.loads((self.source/'load_variants.json').read_text())
        if manifest['schema'] != V.SCHEMA: self.skipTest('Run the angle-only pilot first')
        for angle in V.ANGLES:
            case = V.read_variant('B', 'pose_1', angle)
            V.check_arrays(case['domain'], case['arrays'], check_visibility=False)
            self.assertFalse(V.STATE_KEYS & case['arrays'].keys())
            self.assertNotIn('need_wrench_world_origin', case['arrays'])
            self.assertFalse((case['folder']/'floor_contact.npz').exists())
            self.assertFalse((self.source/f'angle_{angle}_grounded').exists())
            self.assertFalse((self.source/f'angle_{angle}_airborne').exists())
        original = json.loads((self.source/'samples.json').read_text())
        default = V.read_variant('B', 'pose_1', 30)
        for key in V.ARRAY_KEYS:
            np.testing.assert_array_equal(np.asarray(original[key]), default['arrays'][key])


if __name__ == '__main__': unittest.main()
