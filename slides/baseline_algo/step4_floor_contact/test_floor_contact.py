"""Mechanics regressions: ownership, tipping, reallocation, friction and evidence."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step4_floor_contact import equilibrium as Q, audit


def foot(x0, x1, y0=-1., y1=1.):
    return dict(pads_xy_m=[np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])])


def feasible(matrix, target):
    return linprog(np.ones(matrix.shape[1]), A_eq=matrix, b_eq=target,
                   bounds=(0, None), method='highs')


class CoupledEquilibriumTests(unittest.TestCase):

    def test_pressure_center_includes_height_times_horizontal_force(self):
        point = np.asarray(np.array([.2, 0, .8])); force = np.asarray(np.array([-.3, 0, 1.]))
        origin = np.asarray(np.array([.5, -.2, .7]))
        w = Q.wrench(point, force, origin)[None, None]
        cop, loaded = Q.pressure_centers(w, origin)
        np.testing.assert_allclose(cop[0, 0], [.44, 0], atol=1e-14)
        self.assertTrue(loaded[0, 0])

    def test_force_magnitude_does_not_change_pressure_center(self):
        w = np.array([[np.asarray([.2, .1, 1., .3, -.5, .7])]])
        a, _ = Q.pressure_centers(w, np.zeros(3))
        b, _ = Q.pressure_centers(w*100, np.zeros(3))
        np.testing.assert_allclose(a, b)

    def test_upward_pull_cannot_be_fixed_by_larger_feet(self):
        points = np.asarray(np.array([[0., 0., 1.]]))
        normals = np.asarray(np.array([[0., 0., -1.]]))
        for size in (1., 100.):
            matrix, _ = Q.grounded_matrix(points, normals, [0], np.zeros(3), np.ones(6), [foot(-size, size)], 10.)
            target = np.r_[normals[0], [0, 0, 0], np.zeros(6)]
            self.assertEqual(feasible(matrix, target).status, 2)

    def test_whole_hull_cannot_balance_two_separate_overturning_feet(self):
        points = np.asarray(np.array([[0., 0., 1.], [0., 0., 1.]]))
        normals = np.asarray(np.array([[0., 0., 1.], [0., 0., 1.]]))
        matrix, _ = Q.grounded_matrix(points, normals, [0, 1], np.zeros(3), np.ones(6),
                                      [foot(-2, -1), foot(1, 2)], 10.)
        target = np.r_[np.asarray([0., 0., 1., 0., 0., 0.]), np.zeros(12)]
        self.assertEqual(feasible(matrix, target).status, 2)
        # Those same feet can support the load if they are one rigid body.
        joined, _ = Q.grounded_matrix(points, normals, [0, 0], np.zeros(3), np.ones(6),
            [dict(pads_xy_m=foot(-2, -1)['pads_xy_m']+foot(1, 2)['pads_xy_m'])], 10.)
        self.assertTrue(feasible(joined, target[:12]).success)

    def test_reallocation_keeps_fixed_contacts_and_avoids_bad_foot(self):
        points = np.asarray(np.array([[0., 0., 1.], [0., 0., 1.]]))
        normals = np.asarray(np.array([[0., 0., 1.], [0., 0., 1.]]))
        matrix, _ = Q.grounded_matrix(points, normals, [0, 1], np.zeros(3), np.ones(6),
                                      [foot(-2, -1), foot(-.5, .5)], 2.)
        target = np.r_[np.asarray([0., 0., 1., 0., 0., 0.]), np.zeros(12)]
        solution = feasible(matrix, target)
        self.assertTrue(solution.success)
        self.assertAlmostEqual(solution.x[0], 0.)
        self.assertAlmostEqual(solution.x[1], 1.)
        # Freezing an arbitrary previous 50/50 split makes this design fail.
        forced = linprog(np.ones(matrix.shape[1]), A_eq=np.vstack([matrix, np.eye(matrix.shape[1])[:2]]),
                         b_eq=np.r_[target, .5, .5], bounds=(0, None), method='highs')
        self.assertEqual(forced.status, 2)

    def test_unloaded_floor_cannot_provide_pure_horizontal_force(self):
        point = np.asarray(np.array([[0., 0., 1.]]))
        normal = np.asarray(np.array([[1., 0., 0.]]))
        matrix, _ = Q.grounded_matrix(point, normal, [0], np.zeros(3), np.ones(6), [foot(-100, 100)], 64.)
        self.assertEqual(feasible(matrix, np.r_[Q.wrench(point[0], normal[0], np.zeros(3)), np.zeros(6)]).status, 2)

    def test_small_foot_tips_but_wider_foot_balances(self):
        p = np.asarray(np.array([[0., 0., 1.]]))
        n = np.asarray(np.array([[.5, 0., 1.]]))
        target = np.r_[Q.wrench(p[0], n[0], np.zeros(3)), np.zeros(6)]
        small, _ = Q.grounded_matrix(p, n, [0], np.zeros(3), np.ones(6), [foot(-.1, .1)], 2.)
        large, _ = Q.grounded_matrix(p, n, [0], np.zeros(3), np.ones(6), [foot(-1, 1)], 2.)
        self.assertEqual(feasible(small, target).status, 2)
        self.assertTrue(feasible(large, target).success)

    def test_batch_reuses_valid_bases_and_rejects_negative_target(self):
        matrix = np.eye(6)
        solver = Q.BatchSolver(matrix)
        targets = np.array([np.asarray([1., 2., 3., 4., 5., 6.]), np.asarray([2., 1., 4., 2., 3., 5.])])
        result = solver.solve(targets, certified=True)
        self.assertTrue(result['passed']); self.assertEqual(result['lp_count'], 1)
        self.assertFalse(solver.solve(-targets)['passed'])


    def test_duplicate_contact_locations_keep_their_owners(self):
        tri = np.asarray(np.array([[[0, 0, 1], [1, 0, 1], [0, 1, 1.]]]))
        domain = SimpleNamespace(mesh=SimpleNamespace(face_normals=np.asarray(np.array([[0., 0., -1.]]))))
        contacts = [dict(triangles_m=tri, source_faces=np.array([0])) for _ in range(2)]
        p, n, owners = Q.contact_rays(domain, contacts, np.zeros(3))
        self.assertEqual(len(p), 7)
        np.testing.assert_array_equal(np.bincount(owners[1:]), [3, 3])

    def test_mixed_up_and_down_contacts_are_allowed_within_one_body(self):
        p = np.asarray(np.array([[1., 0., 1.], [0., 0., 1.]]))
        n = np.asarray(np.array([[0., 0., -1.], [0., 0., 1.]]))
        target = np.array(np.asarray([0., 0., 1., 0., 1., 0.]))
        together, _ = Q.grounded_matrix(p, n, [0, 0], np.zeros(3), np.ones(6), [foot(-2, 2)], 2.)
        split, _ = Q.grounded_matrix(p, n, [0, 1], np.zeros(3), np.ones(6), [foot(-2, 2), foot(-3, 3)], 2.)
        self.assertTrue(feasible(together, np.r_[target, np.zeros(6)]).success)
        self.assertEqual(feasible(split, np.r_[target, np.zeros(12)]).status, 2)

    def test_exact_impossibility_certificate_replays_original_geometry(self):
        p = np.asarray(np.array([[.3, .2, 1.]]))
        n = np.asarray(np.array([[0., 0., -1.]]))
        origin = np.asarray(np.array([.1, -.3, .7])); scale = np.array([1., 1., 1., 3., 3., 3.])
        load = Q.wrench(p[0], n[0], origin)
        matrix = Q.relaxed_matrix(p, n, np.array([0]), origin, scale, 1)
        target = Q.padded_targets(load[None], scale, 7)[0]
        proof = Q.exact_relaxed_separator(matrix, target, p, n, np.array([0]), origin, scale, 1, load)
        self.assertIsNotNone(proof)
        arrays = dict(moment_origin_m=origin, contact_points_m=p, contact_normals=n,
                      contact_owners=np.array([0]), load_wrenches=load[None], failing_load_index=np.array(0))
        self.assertTrue(audit.replay_separator(arrays, proof)['exact_separation_replayed'])

    def test_three_dimensional_shared_reactions_replay_body_by_body(self):
        rng = np.random.default_rng(781)
        p = rng.uniform(-.2, .2, (24, 3)); p[:, 2] += 1.
        n = rng.uniform(-.2, .2, (24, 3)); n[:, 2] = 1.
        owner = np.repeat([0, 1], 12); origin = np.asarray(np.array([.1, .2, .8]))
        scale = np.ones(6)
        loads = rng.uniform(.01, .1, (12, 24))@Q.wrench(p, n, origin)
        matrix, ground = Q.grounded_matrix(p, n, owner, origin, scale, [foot(-2, 2), foot(-3, 3)], 2.)
        solver = Q.BatchSolver(matrix)
        result = solver.solve(Q.padded_targets(loads, scale, 18), certified=True)
        self.assertTrue(result['passed'], result['diagnostics'])
        arrays = dict(contact_points_m=p, contact_normals=n, contact_owners=owner,
            ground_points_m=ground['points_m'], ground_forces=ground['forces'], ground_owners=ground['owners'],
            moment_origin_m=origin, scale=scale, sample_assignment=result['assignment'],
            sample_basis_indices=np.asarray(solver.bases), sample_coefficients_mg=result['weights'])
        verified = audit.replay(arrays, 'sample', loads, 2, 2.)
        self.assertLess(verified['maximum_body_equilibrium_residual_conditioned'], 1e-9)



if __name__ == '__main__': unittest.main()
