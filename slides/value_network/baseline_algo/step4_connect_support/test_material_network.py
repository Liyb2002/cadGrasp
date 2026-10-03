"""Regressions for joint coverage, real connectivity and shared costs."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step4_connect_support import material_network as N


class MaterialNetworkTests(unittest.TestCase):
    def test_shared_trunk_is_paid_once_and_all_heads_reached(self):
        # Two expensive private routes compete with a common fork.
        cost = np.array([0., 3., 1., 1., 0., 0., 5., 5.])
        edges = np.array([[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [0, 6], [6, 4], [0, 7], [7, 5]])
        groups = [N.Group(np.array([h]), 'head') for h in (0, 4, 5)]
        groups += [N.Group(np.array([2, 6]), 'floor', 0), N.Group(np.array([3, 7]), 'floor', 1)]
        chosen, history = N.greedy(cost, edges, groups, 0)
        self.assertEqual(cost[chosen].sum(), 5.)
        self.assertEqual(sum(r['added_volume_cm3'] for r in history), 5.)
        self.assertTrue(N.connected_selection(chosen, edges, len(cost)))
        exact, report = N.flow_milp(cost, edges, groups, [0, 4, 5], 0, seconds=5)
        self.assertTrue(report['solver_converged_on_supplied_graph'])
        self.assertEqual(cost[exact].sum(), 5.)

    def test_separated_heads_are_not_free_teleport_sources(self):
        cost = np.array([0., 1., 0., 1.]); edges = np.array([[0, 1], [2, 3]])
        groups = [N.Group(np.array([0]), 'head'), N.Group(np.array([2]), 'head')]
        with self.assertRaisesRegex(RuntimeError, 'unreachable'):
            N.greedy(cost, edges, groups, 0)
        chosen, report = N.flow_milp(cost, edges, groups, [0, 2], 0, seconds=5)
        self.assertIsNone(chosen)
        self.assertEqual(report['status'], 2)

    def test_axis_coverage_does_not_accept_inscribed_diamond(self):
        diamond = np.array([[1., 0], [0, 1], [-1, 0], [0, -1]])
        square = np.array([[-1., -1], [-1, 1], [1, 1], [1, -1]])
        axes = np.array([[1., 0], [0, 1], [-1, 0], [0, -1]])
        self.assertTrue(np.all((diamond@axes.T).max(0) >= (square@axes.T).max(0)))
        check = N.containment(diamond, square)
        self.assertFalse(check['passed'])
        self.assertEqual(len(check['normals']), 4)

    def test_duplicate_geometry_groups_do_not_multiply_reward(self):
        floors = [[np.array([[0., 0], [2, 0], [0, 2]]), np.empty((0, 2))],
                  [np.array([[-2., 0], [0, 0], [0, -2]]), np.empty((0, 2))]]
        demands = [np.array([[.5, 0], [0, .5]]), np.empty((0, 2))]
        groups = []
        self.assertTrue(N.add_direction(groups, floors, demands, 0, [1., 0]))
        self.assertFalse(N.add_direction(groups, floors, demands, 0, [1., .1]))
        self.assertEqual(len(groups), 1)

    def test_feedback_growth_keeps_existing_material_and_only_pays_increment(self):
        cost = np.array([0., 1., 1., 2., 1.])
        edges = np.array([[0, 1], [1, 2], [0, 3], [3, 4], [2, 4]])
        seed = np.array([True, True, True, False, False])
        groups = [N.Group(np.array([0]), 'head'), N.Group(np.array([4]), 'mechanics', 0)]
        chosen, history = N.greedy(cost, edges, groups, 0, seed)
        chosen = N.prune(chosen, cost, edges, groups, [0], seed)
        self.assertTrue(chosen[seed].all())
        self.assertEqual(sum(x['added_volume_cm3'] for x in history), 1.)
        self.assertFalse(chosen[3])
        with self.assertRaisesRegex(ValueError, 'connected network'):
            N.greedy(cost, edges, groups, 0, np.array([True, False, True, False, False]))

    def test_diagonal_voxel_contact_survives_exact_stl_vertex_welding(self):
        import manifold3d as md
        import trimesh
        from step4_connect_support import material_graph as M
        from step4_connect_support import build_coupled_saddle as S
        cells = [(0, 0), (1, 1), (-1, 0), (-1, 1), (-1, 2), (0, 2), (1, 2)]
        full = M.union(md.Manifold.cube([.01/S.SCALE]*3).translate([x*.01/S.SCALE, y*.01/S.SCALE, 0]) for x, y in cells)
        head = md.Manifold.cube([.002/S.SCALE]*3).translate([.003/S.SCALE]*3)
        mesh = S.unpack(full); vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
        self.assertFalse(trimesh.Trimesh(vertices, inverse[mesh.faces], process=False).is_watertight)
        fixed, report = M.regularize_export(full, [head])
        mesh = S.unpack(fixed); vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
        welded = trimesh.Trimesh(vertices, inverse[mesh.faces], process=False)
        self.assertTrue(welded.is_watertight)
        self.assertEqual(len(welded.split()), 1)
        self.assertLess(report['removed_volume_cm3'], 1e-5)
        self.assertLess((head-fixed).volume(), 1e-16)


if __name__ == '__main__': unittest.main()
