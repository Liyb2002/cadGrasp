"""Step4 combine: seats, Rotational reuse and Juxtaposition placements, and the paired-load residual."""
import unittest
import numpy as np

from whole_task_2.operations.state import transform
from whole_task_2.step3 import independent_geometry as G
from .combine import (body_and_keepout, centre, default_exit, juxtaposed, largest, membrane, needle, pose, residual,
                      rises, seats, slot, turn, unit)
from .seats import context

OBJECT = 'B'


def group_of(*rows):
    return dict(name='g', slots=[slot(i, X, default_exit(OBJECT, i)) for i, X in rows])


class CombineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = context(OBJECT); cls.n = len(cls.c.base.ids)

    def test_default_exit_is_against_the_needle_and_rises(self):
        a = unit(self.c.base.axis)
        for i in range(self.n):
            np.testing.assert_allclose(pose(OBJECT, i)[:3, :3]@default_exit(OBJECT, i), -a, atol=1e-12)
            self.assertTrue(rises(OBJECT, i, default_exit(OBJECT, i)))
            self.assertFalse(rises(OBJECT, i, pose(OBJECT, i)[:3, :3].T@np.array([0., 0., -1.])))

    def test_seats_group_tasks_sharing_one_placement(self):
        X = np.eye(4); Y = np.eye(4); Y[:3, 3] = [.1, 0., 0.]
        g = group_of((0, X), (1, Y), (2, X))
        rows = seats(g)
        self.assertEqual([ks for _, ks in rows], [[0, 2], [1]])

    def test_juxtaposition_keeps_the_installation_and_puts_centre_on_centre(self):
        Xj = np.eye(4); g = group_of((0, Xj)); i = 3
        X = juxtaposed(OBJECT, g, 0, i, 0., np.zeros(3))
        Wj = pose(OBJECT, 0)[:3, :3]@Xj[:3, :3].T; Wi = pose(OBJECT, i)[:3, :3]@X[:3, :3].T
        np.testing.assert_allclose(Wi, Wj, atol=1e-12)                                  # the same support face down
        np.testing.assert_allclose(transform(centre(OBJECT, i), X), transform(centre(OBJECT, 0), Xj), atol=1e-12)

    def test_pose_rotation_turns_the_seat_about_its_needle_through_its_centre(self):
        g = group_of((0, np.eye(4))); i = 2; t = np.array([.01, -.02, .005])
        X0, X = juxtaposed(OBJECT, g, 0, i, 0., t), juxtaposed(OBJECT, g, 0, i, 25., t)
        np.testing.assert_allclose(transform(centre(OBJECT, i), X), transform(centre(OBJECT, i), X0), atol=1e-12)
        np.testing.assert_allclose(X[:3, :3]@needle(OBJECT, i), X0[:3, :3]@needle(OBJECT, i), atol=1e-12)
        # The task keeps its certified pose; the placing arm turns the support about the world needle direction.
        W0 = pose(OBJECT, i)[:3, :3]@X0[:3, :3].T; W = pose(OBJECT, i)[:3, :3]@X[:3, :3].T
        np.testing.assert_allclose(W, turn(unit(self.c.base.axis), -25.)@W0, atol=1e-12)

    def test_a_task_alone_holds_its_guidance_demands(self):
        i = 0; keep, share = largest(membrane(OBJECT, i)-body_and_keepout(OBJECT, i))
        self.assertGreater(share, .9)
        loss, covered, rows = residual(OBJECT, [slot(i, np.eye(4), default_exit(OBJECT, i))], G.mesh(keep), {})
        self.assertTrue(covered, rows); self.assertEqual(loss, 0.)


if __name__ == '__main__':
    unittest.main()
