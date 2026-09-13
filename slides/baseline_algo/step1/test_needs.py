"""Discriminating checks for signs, units, geometry and exported demand pairs."""
from needs import COORD
import json
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from needs import ContinuousNeeds, OBJECTS, OUTPUTS, demand, sample_needs, sha256
from step1.cases import pose_name


class WrenchMechanics(unittest.TestCase):
    def test_hand_calculated_lever_and_gravity(self):
        # 20 mm along +x, half-weight push along -y gives -10 mg*mm about z.
        result = demand([.02, 0, 0], [0, -.5, 0], np.zeros(3))
        assert_allclose(result, np.asarray([0, .5, 1, 0, 0, .01]), atol=1e-15)

    def test_reverse_force_does_not_reverse_gravity(self):
        down = demand([.1, 0, 0], [0, 0, -.5], np.zeros(3))
        up = demand([.1, 0, 0], [0, 0, .5], np.zeros(3))
        assert_allclose(down, np.asarray([0, 0, 1.5, 0, -.05, 0]))
        assert_allclose(up, np.asarray([0, 0, .5, 0, .05, 0]))

    def test_same_line_of_action_gives_same_pair(self):
        f = np.asarray(np.array([.3, .4, 0]))
        q, c = np.asarray(np.array([.1, -.02, .07])), np.asarray(np.array([.01, .02, .03]))
        assert_allclose(demand(q, f, c), demand(q+7*f, f, c), atol=2e-16)

    def test_translation_and_rotation_covariance(self):
        q, f, c = np.asarray(np.array([.02, -.03, .04])), np.asarray(np.array([.3, .4, 0])), np.asarray(np.array([.04, .02, -.01]))
        offset = np.asarray(np.array([.7, -1, 2]))
        assert_allclose(demand(q+offset, f, c+offset), demand(q, f, c), atol=1e-15)
        # Rotation preserving handedness; rotate gravity too for a frame change.
        R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        w = demand(q, f, c)
        rotated = demand(R@q, R@f, R@c, R@np.asarray(np.array([0., 0., -1.])))
        assert_allclose(rotated, np.r_[R@w[:3], R@w[3:]], atol=1e-15)

    def test_independent_moment_about_another_origin(self):
        q, c, o = np.asarray(np.array([.2, .3, .4])), np.asarray(np.array([.03, -.01, .02])), np.asarray(np.array([-.2, .1, 0]))
        f, g = np.asarray(np.array([.3, 0, -.4])), np.asarray(np.array([0, 0, -1]))
        w = demand(q, f, c)
        # Shift the equivalent reaction wrench from COM to o, then balance each
        # physical external force at its own application point about o.
        total = w[3:] + np.cross(c-o, w[:3]) + np.cross(q-o, f) + np.cross(c-o, g)
        assert_allclose(total, 0, atol=2e-16)


class SampledDemands(unittest.TestCase):
    @staticmethod
    def domain():
        # Disjoint horizontal triangles with a 1:4 area ratio, no occluders.
        return ContinuousNeeds({
            'object': 'two_triangles',
            'geometry': {
                'vertices_m': [[0, 0, 0], [1, 0, 0], [0, 1, 0],
                               [3, 0, 0], [5, 0, 0], [3, 2, 0]],
                'faces': [[0, 1, 2], [3, 4, 5]], 'work_face_ids': [0, 1],
                'inward_normals': [[0, 0, -1]]*2,
                'tangent1': [[1, 0, 0]]*2, 'tangent2': [[0, -1, 0]]*2,
                'work_face_areas_m2': [.5, 2.],
            },
            'frame': {'moment_origin_m': [0., 0., 0.]},
            'load': {'gravity_force_mg': [0., 0., -1.], 'K': .5,
                     'magnitude_range_mg': [0., .5], 'cone_half_deg': 30.},
            'reachability': {'ray_offset_m': 1e-5},
        })

    def test_area_solid_angle_and_magnitude_distributions(self):
        d = self.domain()
        samples = sample_needs(d, 8192, 24)
        self.assertAlmostEqual(np.mean(samples['work_face_index']), .8, delta=.02)
        u, v, theta, phi, magnitude = np.asarray(samples['parameters']).T
        self.assertTrue(np.all((u >= 0) & (v >= 0) & (u+v <= 1)))
        assert_allclose([u.mean(), v.mean()], [1/3, 1/3], atol=.015)
        cap_coordinate = (np.cos(theta)-np.cos(d.half_angle))/(1-np.cos(d.half_angle))
        for uniform in (cap_coordinate, phi/(2*np.pi), magnitude/d.k):
            assert_allclose(np.histogram(uniform, bins=np.linspace(0, 1, 5))[0]/len(uniform),
                            [.25]*4, atol=.02)
        force = np.asarray(samples['force_push_mg'])
        pt = np.asarray(samples['pt_m'])
        wrench = np.asarray(samples['need_wrench'])
        assert_allclose(wrench[:, :3]+force, np.tile([0, 0, 1], (len(pt), 1)), atol=1e-14)
        # Independent component expansion keeps force and moment paired.
        x, y, z = pt.T
        fx, fy, fz = force.T
        torque = np.c_[y*fz-z*fy, z*fx-x*fz, x*fy-y*fx]
        assert_allclose(wrench[:, 3:], -torque, atol=1e-14)

    def test_visibility_rejection_refills_and_preserves_prefix(self):
        d = self.domain()
        # Block every ray originating on the smaller triangle.
        def blocked(origins, directions):
            return origins[:, 0] < 2
        with patch.object(d.mesh.ray, 'intersects_any', side_effect=blocked):
            small = sample_needs(d, 97, 12)
            large = sample_needs(d, 5000, 12)
            other = sample_needs(d, 97, 13)
        self.assertEqual(large['count'], 5000)
        self.assertTrue(np.all(np.asarray(large['work_face_index']) == 1))
        self.assertGreater(large['rejected_proposal_count'], 0)
        self.assertEqual(large['proposal_count_until_last_sample'], 5000+large['rejected_proposal_count'])
        for key in ('work_face_index', 'parameters', 'pt_m', 'force_push_mg', 'need_wrench'):
            self.assertEqual(small[key], large[key][:97])
        self.assertNotEqual(small['pt_m'], other['pt_m'])
        self.assertEqual(large['weight_per_sample'], 1/5000)

    def test_every_scored_load_requires_visibility(self):
        d = self.domain()
        samples = sample_needs(d, 64, 0)
        self.assertEqual(len(samples['need_wrench']), 64)
        self.assertNotIn('boundary_checks', samples)
        self.assertEqual(samples['weight_per_sample']*samples['count'], 1.)
        with patch.object(d.mesh.ray, 'intersects_any', side_effect=lambda origins, directions: np.ones(len(origins), bool)):
            with self.assertRaisesRegex(RuntimeError, 'Only 0 of 8 reachable samples'):
                sample_needs(d, 8, 0)
        for count, seed in [(0, 1), (-1, 1), (2.5, 1), (True, 1), (8, -1)]:
            with self.subTest(count=count, seed=seed), self.assertRaises(ValueError):
                sample_needs(d, count, seed)


class ExportedContinuousDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.domains = {name: ContinuousNeeds.read(OUTPUTS/name/pose_name()/'step_1_needs/needs.json') for name in ('B',)}

    def test_frames_and_area_measure(self):
        for name, domain in self.domains.items():
            with self.subTest(object=name):
                data = domain.data
                self.assertFalse(data['finite_enumeration'])
                self.assertEqual(data['provenance']['sampled_push_or_coverage_fields_used'], [])
                basis = np.stack([domain.e1, domain.e2, domain.normals], axis=1)
                assert_allclose(basis@basis.transpose(0, 2, 1),
                                np.broadcast_to(np.eye(3), (len(basis), 3, 3)), atol=1e-12)
                assert_allclose(domain.normals, -domain.mesh.face_normals[domain.work_ids], atol=1e-11)
                assert_allclose(data['geometry']['work_area_m2'], domain.mesh.area_faces[domain.work_ids].sum())
                assert_allclose(data['geometry']['work_face_areas_m2'], domain.mesh.area_faces[domain.work_ids])

    def test_arbitrary_interior_and_boundary_parameters(self):
        # Not the three proof points. Entire triangle vertices/edges/interior and
        # cap pole/rim/interior are evaluable without consulting a finite table.
        for name, domain in self.domains.items():
            with self.subTest(object=name):
                i = np.linspace(0, len(domain.work_ids)-1, 11, dtype=int)
                for u, v, angle in [(0, 0, 0), (1, 0, 1), (0, 1, .5), (.23, .41, .73)]:
                    values = domain.evaluate(i, u, v, angle*domain.half_angle, np.arange(11)*.77)
                    assert_allclose(np.linalg.norm(values['force_push_mg'], axis=1), .5, atol=1e-14)
                    cosines = (values['d']*domain.normals[i]).sum(axis=1)
                    self.assertTrue(np.all(cosines >= np.cos(domain.half_angle)-1e-14))
                    tri = domain.mesh.triangles[domain.work_ids[i]]
                    assert_allclose(values['q_m'], (1-u-v)*tri[:, 0]+u*tri[:, 1]+v*tri[:, 2], atol=1e-15)
                    assert_allclose(values['need_wrench']+values['external_wrench'], 0, atol=1e-15)
                    periodic = domain.evaluate(i, u, v, angle*domain.half_angle, np.arange(11)*.77+2*np.pi)
                    assert_allclose(periodic['need_wrench'], values['need_wrench'], atol=1e-14)

    def test_reject_invalid_parameters(self):
        d = self.domains['B']
        for args in [(-1, 0, 0, 0, 0), (.5, 0, 0, 0, 0), (len(d.work_ids), 0, 0, 0, 0),
                     (0, .8, .8, 0, 0), (0, -.1, .5, 0, 0), (0, 0, 0, d.half_angle+.01, 0),
                     (0, 0, 0, -.01, 0), (0, 0, 0, 0, np.nan)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                d.evaluate(*args)
        for magnitude in [-.01, .50001, np.nan, np.inf]:
            with self.subTest(magnitude=magnitude), self.assertRaises(ValueError):
                d.evaluate(0, .2, .3, .1, .2, magnitude_mg=magnitude)

    def test_continuous_magnitudes_and_affine_endpoint_relation(self):
        for name, d in self.domains.items():
            with self.subTest(object=name):
                assert_allclose(d.magnitude_range, [0., .5])
                magnitudes = np.array([0., .0713, .25, .4991, .5])
                values = d.evaluate(0, .23, .41, .12, 1.7, magnitude_mg=magnitudes)
                assert_allclose(np.linalg.norm(values['force_push_mg'], axis=1), magnitudes, atol=1e-15)
                gravity = np.array(np.asarray([0., 0., 1., 0., 0., 0.]))
                assert_allclose(values['need_wrench'][0], gravity, atol=1e-15)
                fraction = (magnitudes/.5)[:, None]
                assert_allclose(values['need_wrench'],
                                (1-fraction)*gravity+fraction*values['need_wrench'][-1], atol=1e-15)

    def test_zero_force_is_not_injected_when_tool_ray_is_blocked(self):
        d = self.domains['B']
        with patch.object(d.mesh.ray, 'intersects_any', return_value=np.array([True, True])):
            values = d.evaluate(0, .2, .3, .1, .2, magnitude_mg=[0., .25])
        assert_allclose(values['need_wrench'][0], np.asarray([0., 0., 1., 0., 0., 0.]))
        self.assertEqual(values['reachable'].tolist(), [False, False])
        self.assertEqual(values['tool_reachable'].tolist(), [False, False])

    def test_proof_pairs_with_independent_skew_matrix(self):
        for name, domain in self.domains.items():
            folder = OUTPUTS/name/pose_name()/'step_1_needs'
            examples = json.loads((folder/'examples.json').read_text())
            self.assertEqual(examples['needs_sha256'], sha256(folder/'needs.json'))
            self.assertEqual(len(examples['cases']), 3)
            for case in examples['cases']:
                with self.subTest(object=name, case=case['id']):
                    p = case['parameters']
                    result = domain.evaluate(case['work_face_index'], p['u'], p['v'], p['theta_rad'], p['phi_rad'],
                                             magnitude_mg=p['magnitude_mg'])
                    self.assertTrue(result['reachable'])
                    assert_allclose(result['q_m'], case['pt_m'], atol=1e-15)
                    assert_allclose(result['need_wrench'], case['need_wrench'], atol=1e-15)
                    x, y, z = result['q_m']-domain.com
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    push_tau = skew@result['force_push_mg']
                    assert_allclose(result['need_wrench'][3:]+push_tau, 0, atol=1e-15)
                    # Catch the old ray-origin bug: tilted forces MUST be applied
                    # on the triangle, not at q + 10 micrometres*outward.
                    wrong = demand(result['q_m']-domain.ray_offset*domain.normals[case['work_face_index']],
                                   result['force_push_mg'], domain.com)
                    if p['magnitude_mg'] > 0:
                        self.assertGreater(np.linalg.norm(wrong[3:]-result['need_wrench'][3:]), 1e-9)

    def test_exported_sample_identity_units_and_reachability(self):
        for name, domain in self.domains.items():
            with self.subTest(object=name):
                folder = OUTPUTS/name/pose_name()/'step_1_needs'
                data = json.loads((folder/'samples.json').read_text())
                self.assertEqual(domain.data['load']['cone_half_deg'], 30.)
                self.assertNotIn('boundary_checks', data)
                self.assertEqual(data['provenance']['physical_domain_sha256'], sha256(folder/'needs.json'))
                self.assertEqual(data['provenance']['support_or_coverage_inputs_used'], [])
                self.assertEqual(data['provenance']['generator_sha256'], sha256(OUTPUTS.parent/'step1/needs.py'))
                index = json.loads((folder/'domain.json').read_text())
                self.assertEqual(index['sample_set']['source_sha256'], sha256(folder/'samples.json'))
                self.assertEqual(index['sample_set']['count'], data['count'])
                self.assertEqual(index['sample_set']['seed'], data['seed'])
                proof = json.loads((folder/'proof.json').read_text())
                self.assertEqual(proof['examples_sha256'], sha256(folder/'examples.json'))
                self.assertEqual(proof['draw_script_sha256'], sha256(OUTPUTS.parent/'step1/draw_proof.py'))
                n = data['count']
                self.assertEqual(np.shape(data['need_wrench']), (n, 6))
                self.assertEqual(np.shape(data['pt_m']), (n, 3))
                self.assertEqual(np.shape(data['parameters']), (n, 5))
                self.assertEqual(data['weight_per_sample']*n, 1.)
                ids = np.linspace(0, n-1, min(n, 73), dtype=int)
                u, v, theta, phi, magnitude = np.asarray(data['parameters'])[ids].T
                values = domain.evaluate(np.asarray(data['work_face_index'])[ids], u, v, theta, phi,
                                         magnitude_mg=magnitude)
                self.assertTrue(values['tool_reachable'].all())
                assert_allclose(values['q_m'], np.asarray(data['pt_m'])[ids], atol=1e-14)
                assert_allclose(values['force_push_mg'], np.asarray(data['force_push_mg'])[ids], atol=1e-14)
                assert_allclose(values['need_wrench'], np.asarray(data['need_wrench'])[ids], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
