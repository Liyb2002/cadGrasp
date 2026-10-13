"""Discriminating checks for the continuous-domain reduction and certificates."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linprog
import continuous as C


class ContinuousTests(unittest.TestCase):
    def test_cap_extrema_include_axial_and_interior_cases(self):
        n = np.tile([0., 0., 1.], (5, 1))
        v = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 0, 0], [.1, 0, 1.]])
        d = C.cap_maximizer(v, n)
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1, atol=1e-14)
        self.assertTrue((d[:, 2] >= np.cos(C.THETA)-1e-14).all())
        np.testing.assert_allclose(np.sum(v*d, axis=1),
                                   [1, -np.cos(C.THETA), np.sin(C.THETA), 0, np.sqrt(1.01)], atol=1e-14)

    def test_prism_encloses_rim_between_its_vertices(self):
        # An inscribed polygon would fail these midway rim directions.
        n = np.array([[0., 0., 1.]])
        vertices = C.cap_enclosure(n)[0]
        e1, e2 = C.frame(n)
        for angle in (np.pi/C.SIDES, .413, 1.832):
            d = np.cos(C.THETA)*n[0]+np.sin(C.THETA)*(np.cos(angle)*e1[0]+np.sin(angle)*e2[0])
            solution = linprog(np.zeros(len(vertices)), A_eq=np.vstack([vertices.T, np.ones(len(vertices))]),
                               b_eq=np.r_[d, 1], bounds=(0, None), method='highs')
            self.assertTrue(solution.success)

    def test_bilinear_demand_is_convex_combination_of_vertex_demands(self):
        S = SimpleNamespace(com=np.array([.2, -.3, .4]), scale=np.array([1, 1, 1, 2, 2, 2]))
        q = np.array([[0., 0., 0.], [1., .3, .1], [-.5, 1., .2]])
        d = C.cap_enclosure(np.array([[0., 0., 1.]]))[0]
        alpha = np.array([.13, .29, .58])
        beta = np.arange(1., len(d)+1); beta /= beta.sum()
        vertex_targets = C.wrench(S, q[:, None], d[None])
        combination = np.einsum('i,j,ijk->k', alpha, beta, vertex_targets)
        np.testing.assert_allclose(combination, C.wrench(S, alpha@q, beta@d), atol=1e-14)

    def test_residual_alone_does_not_certify_membership(self):
        B = np.eye(6)
        targets = np.ones((3, 6)); targets[1, 0] = -1e-12; targets[2, 0] = 0
        ok, _ = C.basis_membership(B, targets)
        np.testing.assert_array_equal(ok, [True, False, False])
        B[-1] = B[0]
        self.assertFalse(C.basis_membership(B, np.ones((1, 6)))[0][0])


if __name__ == '__main__':
    unittest.main()
