"""Jointness, unilateral reactions, lower-dimensional cones and sampled scoring."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from fractions import Fraction
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import linprog
import contribution as V


class ContributionTests(unittest.TestCase):
    def test_unknown_primal_status_requires_an_exact_dual_certificate(self):
        full=np.eye(6);target=np.array([-1.,0,0,0,0,0])
        original=V.W.linprog
        def uncertain(*args,**kwargs):
            if 'A_eq' in kwargs:return SimpleNamespace(status=4,success=False,message='test unknown status')
            return original(*args,**kwargs)
        with patch.object(V.W,'linprog',side_effect=uncertain):
            self.assertIsNone(V.W.solve(full,target))
        proof=V.W.exact_separator(full,target)
        h=[Fraction(x) for x in proof['normal_exact']]
        self.assertTrue(all(sum(Fraction(float(v))*w for v,w in zip(row,h))<=0 for row in full))
        self.assertGreater(sum(Fraction(float(v))*w for v,w in zip(target,h)),0)
        self.assertIsNone(V.W.exact_separator(full,np.ones(6)))

    def test_separate_force_and_moment_success_is_not_joint_success(self):
        full = np.vstack([np.eye(6), [0,1,0,-1,0,0]])
        target = np.array([1.,0,0,-1,0,0])
        for block in [slice(0,3), slice(3,6)]:
            result = linprog(np.zeros(len(full)), A_eq=full[:,block].T, b_eq=target[block], bounds=(0,None))
            self.assertTrue(result.success)
        self.assertIsNone(V.W.solve(full,target))
        mask, _, _ = V.classify(full, target[None])
        self.assertFalse(mask[0])

    def test_positive_span_has_independent_witnesses(self):
        full = np.vstack([np.eye(6), -np.eye(6)])
        self.assertEqual(len(V.W.cone(full)), 0)
        proof = V.W.full_space_certificate(full)
        self.assertEqual(len(proof['witnesses']), 12)
        mask, _, _ = V.classify(full, np.random.default_rng(1).normal(size=(17,6)))
        self.assertTrue(mask.all())

    def test_cone_facets_match_nonnegative_joint_LP(self):
        rng = np.random.default_rng(4)
        full = np.c_[rng.normal(size=(18,5)), np.ones(18)]
        targets = np.vstack([rng.normal(size=(12,6)), rng.random((6,len(full)))@full])
        mask, _, _ = V.classify(full, targets)
        for target, verdict in zip(targets, mask):
            self.assertEqual(bool(verdict), V.W.solve(full, target) is not None)

    def test_supports_can_push_but_cannot_pull(self):
        full = np.eye(6)
        targets = np.array([[1,1,1,1,1,1], [-1,1,1,1,1,1], [0,0,0,0,0,0]])
        mask, _, _ = V.classify(full, targets)
        np.testing.assert_array_equal(mask, [True,False,True])

    def test_rank_deficient_cone_does_not_mean_zero_sample_score(self):
        full = np.array([[0.,0,1,0,0,0], [0,0,1,1,0,0]])
        targets = np.array([[0,0,1,.5,0,0], [0,0,1,2,0,0], [0,1,1,.5,0,0], [0,0,0,0,0,0]])
        mask, info, _ = V.classify(full, targets)
        np.testing.assert_array_equal(mask, [True,False,False,True])
        self.assertEqual(info['supply_rank'], 2)

    def test_batch_preserves_boundary_samples_and_duplicate_weights(self):
        H = -np.eye(6)
        targets = np.array([[0,0,0,0,0,0], [1,2,3,4,5,6], [-1,0,0,0,0,0], [1,2,3,4,5,6]])
        mask = V.halfspace_membership(H, targets)
        self.assertEqual(mask.sum()/len(mask), .75)
        np.testing.assert_array_equal(mask, [True,True,False,True])

    def test_separator_checks_every_stored_sample(self):
        full = np.eye(6)
        targets = np.array([[-1.,1,1,1,1,1], [-.1,2,2,2,2,2]])
        separator = V.sample_separator(full, targets)
        self.assertIsNotNone(separator)
        y = separator['vector']
        self.assertGreater((targets@y).min(), 1e-8)
        self.assertLessEqual((full@y).max(), 1e-12)
        self.assertIsNone(V.sample_separator(full, np.vstack([targets, np.ones(6)])))

    def test_conditioned_moment_uses_original_floor_contact_and_push_normals(self):
        domain, data, _, S, floor = V.context('B')
        index = int(np.flatnonzero(data.valid)[0])
        full = V.columns(domain, data, index, floor, S.scale)
        for force in ([64.,0.,1.],[-64.,0.,1.],[0.,64.,1.],[0.,-64.,1.]):
            expected = np.r_[np.r_[force, np.cross(floor-domain.com, force)]*S.scale,0.]
            self.assertLess(np.linalg.norm(full-expected, axis=1).min(), 1e-13)
        a,b = data.offsets[index:index+2]
        normal = -domain.mesh.face_normals[data.source_faces[a]]
        point = data.triangles[a,0]
        expected = np.r_[np.r_[normal, np.cross(point-domain.com, normal)]*S.scale,normal[2]]
        self.assertLess(np.linalg.norm(full-expected, axis=1).min(), 1e-13)

    def test_actual_stored_pairs_have_no_injected_gravity_case(self):
        for name in V.OBJECTS:
            domain, _, _, _, _ = V.context(name)
            samples, targets = V.read_samples(name, domain)
            self.assertEqual(len(targets), samples['count'])
            self.assertNotIn('boundary_checks', samples)
            assert_allclose(targets[:, :3]+samples['force_push_mg'], np.broadcast_to([0,0,1], (len(targets),3)))

    def test_real_circle_facets_keep_all_original_rays(self):
        domain, data, _, S, floor = V.context('B')
        full = V.columns(domain, data, int(np.flatnonzero(data.valid)[3]), floor, S.scale)
        H = V.W.cone(full)
        self.assertLess(float((H@full.T).max()), 2e-9)
        samples, targets = V.read_samples('B', domain)
        targets = targets[::2048]*S.scale
        mask = V.halfspace_membership(H, targets)
        verification = V.verify_classification(full, targets, mask)
        self.assertEqual(verification['disagreements'], 0)


if __name__ == '__main__':
    unittest.main()
