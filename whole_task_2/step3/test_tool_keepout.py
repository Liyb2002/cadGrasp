"""The tool keep-out holds the needle and the head of every face, and no more than its polygons' excess."""
import unittest
import numpy as np
import trimesh

from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.tool_keepout import keepout, pieces

SETUP = dict(needle_length_m=.1, needle_radius_m=.002, head_radius_m=.02)


def inside(solid, points):
    return G.mesh(solid).contains(np.asarray(points))


class KeepoutTest(unittest.TestCase):
    def setUp(self):
        # One flat face in z = 0, painted from above: the needle points down (-z).
        self.mesh = trimesh.Trimesh([[0, 0, 0], [.01, 0, 0], [0, .01, 0]], [[0, 1, 2]], process=False)
        self.d = np.array([0., 0., -1.]); self.reach = .05
        self.solid = keepout(self.mesh, [0], [self.d], SETUP, self.reach)

    def test_needle_and_head_inside(self):
        rng = np.random.default_rng(0); u = rng.random((200, 2)); u[u.sum(1) > 1] = 1-u[u.sum(1) > 1]
        tips = np.c_[u*.01, np.zeros(200)]; angle = rng.random(200)*2*np.pi
        ring = np.c_[np.cos(angle), np.sin(angle), np.zeros(200)]
        needle = tips+.0019*rng.random((200, 1))*ring+rng.uniform(.001, .099, (200, 1))*[0, 0, 1]
        head = tips+.0199*rng.random((200, 1))*ring+rng.uniform(.101, .149, (200, 1))*[0, 0, 1]
        self.assertTrue(inside(self.solid, needle).all())
        self.assertTrue(inside(self.solid, head).all())

    def test_nothing_far_from_the_tool(self):
        centre = np.array([.01/3, .01/3, 0.])
        outside = [centre+[0, 0, -.001],                      # below the face, inside the object
                   centre+[.02, 0, .05],                      # beside the needle, farther than its radius
                   centre+[.05, 0, .12],                      # beside the head, farther than its radius
                   centre+[0, 0, .1+self.reach+.002]]         # beyond the head's reach
        self.assertFalse(inside(self.solid, outside).any())

    def test_tilted_axis_follows_the_face_axis(self):
        d = np.array([1., 0., -1.])/np.sqrt(2)
        solid = keepout(self.mesh, [0], [d], SETUP, self.reach)
        tip = np.array([.002, .002, 0.])
        self.assertTrue(inside(solid, [tip-.05*d, tip-.12*d]).all())
        self.assertFalse(inside(solid, [tip+[0, 0, .05]]).any())    # where an untilted needle would be

    def test_two_pieces_per_face(self):
        parts = pieces(self.mesh.triangles, [self.d], .1, .002, .02, self.reach, sides=8)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (2*3*8, 3))


if __name__ == '__main__':
    unittest.main()
