"""Force/moment landing and continuous enclosure regressions for a shared body."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step1.needs import demand
from step4_floor_contact import whole_assembly as F
from step5_connect_support.ground import hull_coverage
from step5_connect_support import floor_design as FD


class WholeFloorTests(unittest.TestCase):
    def test_closed_floor_polygon_contains_demand_and_external_pivot(self):
        required = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        pivot = COORD.polar(np.array([2., .5, 0.]))
        polygon, loop = FD.support_polygon(required, pivot)
        np.testing.assert_array_equal(loop[0], loop[-1])
        self.assertEqual(len(loop), len(polygon)+1)
        self.assertTrue(hull_coverage(np.vstack([required, COORD.floor(pivot)]), polygon, 1e-12)[0].all())
        self.assertTrue(np.any(np.all(polygon == COORD.floor(pivot), axis=1)))

    def test_degenerate_floor_demand_does_not_invent_a_width(self):
        with self.assertRaises(ValueError):
            FD.support_polygon([[0., 0.], [1., 0.]], [2., 0., 0.])

    def test_gravity_lands_below_com_and_horizontal_force_shifts_landing(self):
        com = COORD.polar(np.array([.2, .3, 1.]))
        q = COORD.polar(np.array([[0., 0., 2.], [0., 0., 2.]]))
        forces = COORD.polar(np.array([[0., 0., 0.], [.5, 0., 0.]]))
        p, normal = F.pressure_centers(demand(q, forces, com), com)
        np.testing.assert_allclose(p, [[.2, .3], [1.2, .3]])
        np.testing.assert_array_equal(normal, [1., 1.])

    def test_moment_is_not_lost_when_resultant_force_is_unchanged(self):
        loads = np.array([COORD.wrench([0., 0., 1., 0., 0., 0.]), COORD.wrench([0., 0., 1., .3, -.4, 1.])])
        p, _ = F.pressure_centers(loads, np.zeros(3))
        np.testing.assert_allclose(p, [[0., 0.], [.4, .3]])

    def test_positive_denominator_maps_convex_combinations_inside_vertex_hull(self):
        rng = np.random.default_rng(612)
        loads = rng.normal(size=(30, 6)); loads[:, 1] = rng.uniform(.5, 1.5, 30)
        weights = rng.uniform(size=(100, 30)); weights /= weights.sum(axis=1)[:, None]
        origin = COORD.polar(np.array([.1, -.2, .7]))
        p, _ = F.pressure_centers(loads, origin)
        mixed, _ = F.pressure_centers(weights@loads, origin)
        self.assertTrue(hull_coverage(mixed, p, 1e-12)[0].all())

    def test_continuous_outer_domain_contains_actual_cap_and_zero(self):
        mesh = trimesh.creation.box([1., 1., 1.]); mesh.apply_translation([0, .5, 0])
        ids = np.flatnonzero(mesh.face_normals[:, 1] > .9)
        domain = SimpleNamespace(mesh=mesh, work_ids=ids, normals=-mesh.face_normals[ids],
            half_angle=np.pi/6, k=.5, gravity=COORD.polar(np.array([0., 0., -1.])), com=mesh.center_mass)
        outer = F.outer_loads(domain); p, _ = F.pressure_centers(outer, domain.com)
        rng = np.random.default_rng(719)
        theta = rng.uniform(0, np.pi/6, 300); phi = rng.uniform(0, 2*np.pi, 300)
        forces = rng.uniform(0, .5, 300)[:, None]*np.c_[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), -np.cos(theta)]
        q = np.c_[rng.uniform(-.5, .5, (300, 2)), np.ones(300)]
        cloud, _ = F.pressure_centers(demand(COORD.polar(q), COORD.polar(forces), domain.com), domain.com)
        self.assertTrue(hull_coverage(cloud, p, 1e-12)[0].all())
        np.testing.assert_array_equal(outer[0], COORD.wrench([0, 0, 1, 0, 0, 0]))

    def test_zero_or_negative_ground_normal_is_not_a_finite_landing(self):
        for n in (0., -1.):
            with self.assertRaises(ValueError): F.pressure_centers([COORD.wrench([0, 0, n, 0, 0, 0])], np.zeros(3))


if __name__ == '__main__': unittest.main()
