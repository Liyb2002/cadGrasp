"""Regression checks for the reductions used to score contact regions."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linprog
from area_search import Search


class SearchGeometryTests(unittest.TestCase):
    def test_new_contact_invalidates_old_infeasibility_separator(self):
        search = object.__new__(Search)
        search.target = np.array([[1., 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
                                  [-1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        first = np.eye(6)[[0]]
        old_ok, old_duals = search.classify(first)
        self.assertEqual(old_ok.tolist(), [False, True, False, False])
        together = np.eye(6)[[0, 1]]
        ok, _ = search.classify(together, old_ok, old_duals)
        expected = [linprog(np.zeros(2), A_eq=together.T, b_eq=b,
                            bounds=(0, None), method='highs').success for b in search.target]
        self.assertEqual(ok.tolist(), expected)
        self.assertEqual(ok.tolist(), [True, True, False, False])

    def test_rank_deficient_cone_accepts_zero_and_rejects_off_plane(self):
        search = object.__new__(Search)
        search.target = np.array([[0., 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0],
                                  [-1, 0, 0, 0, 0, 0], [1, 0, .01, 0, 0, 0]])
        ok, _ = search.classify(np.eye(6)[[0]])
        self.assertEqual(ok.tolist(), [True, True, False, False])

    def test_planar_reduction_preserves_all_contact_wrenches(self):
        # A tilted planar triangle with interior and collinear edge samples.
        u = np.array([1., 2, 3]); u /= np.linalg.norm(u)
        e1 = np.cross(u, [0., 0, 1]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        xy = np.array([[0., 0], [2, 0], [0, 3], [1, 0], [0, 1], [.2, .3], [.5, .5]])
        points = .01*(xy[:, [0]]*e1+xy[:, [1]]*e2)+[.1, -.2, .3]
        search = object.__new__(Search)
        search.S = SimpleNamespace(skin=SimpleNamespace(cs=points, src=np.zeros(len(points), int)))
        search.reduced = {}
        search.footprint = lambda seed, level: np.arange(len(points))
        ids = search.extreme_faces(0, 0)
        self.assertEqual(len(ids), 3)
        forces = np.tile(u, (len(points), 1))
        wrenches = np.c_[forces, np.cross(points-[.02, .03, .04], forces)]
        for target in wrenches:
            result = linprog(np.zeros(len(ids)), A_eq=wrenches[ids].T, b_eq=target,
                             bounds=(0, None), method='highs')
            self.assertTrue(result.success)
            self.assertLess(np.max(np.abs(wrenches[ids].T@result.x-target)), 1e-10)


class ContactVisibilityTests(unittest.TestCase):
    def test_geometry_beyond_camera_does_not_occlude_contact(self):
        import trimesh
        from views import visible
        skin = SimpleNamespace(cs=np.zeros((1, 3)), cm=np.zeros((1, 3)),
                               ns=np.array([[0., 0, 1]]), nm=np.array([[0., 0, 1]]))
        wall = trimesh.creation.box(bounds=[[-1, -1, 1.9], [1, 1, 2.1]])
        scene = SimpleNamespace(skin=skin, mesh=wall, R=np.eye(3))
        camera = SimpleNamespace(eye=np.array([0., 0, 1]))
        self.assertTrue(visible(scene, np.array([0]), camera)[0][0])
        wall.apply_translation([0, 0, -1.5])
        self.assertFalse(visible(scene, np.array([0]), camera)[0][0])


if __name__ == '__main__':
    unittest.main()
