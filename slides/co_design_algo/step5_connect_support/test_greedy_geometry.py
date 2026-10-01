"""Analytic regressions for assignment, actual feet, and graph connectivity."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step5_connect_support.greedy_geometry import (
    kruskal, nearest_order, foot_targets, footprint_coverage, actual_footprint)


class GreedyGeometryTests(unittest.TestCase):
    def test_mst_matches_exhaustive_small_graph(self):
        edges = [(0,1,9.), (0,2,1.), (1,2,2.), (1,3,3.), (2,3,8.), (0,3,7.)]
        selected = kruskal(4, edges)
        costs = []
        for choice in itertools.combinations(range(len(edges)), 3):
            try:
                result = kruskal(4, [edges[i] for i in choice])
            except RuntimeError:
                continue
            costs.append(sum(edges[i][2] for i in choice))
        self.assertEqual(sum(edges[i][2] for i in selected), min(costs))
        self.assertEqual(set(selected), {1,2,3})

    def test_disconnected_graph_is_failure(self):
        with self.assertRaises(RuntimeError):
            kruskal(4, [(0,1,1.), (2,3,1.)])

    def test_one_head_can_own_both_pose_targets(self):
        heads = [np.array([[0.,0.,1.]]), np.array([[10.,0.,1.]]), np.array([[20.,0.,1.]])]
        targets = [np.array([0.,0.,0.]), np.array([0.,0.,2.])]
        self.assertEqual([nearest_order(heads, p)[0][0] for p in targets], [0,0])

    def test_target_polygons_enclose_demands_without_solid_ring(self):
        demands = np.array([[-1.,0.], [0.,-2.], [1.,0.], [0.,2.], [.2,.2]])
        for family in ('hull','minimum_rectangle','axis_box'):
            pads = foot_targets(demands, .1, .01, family)
            self.assertTrue(footprint_coverage(np.concatenate(pads), [0.,0.,0.], demands)['passed'])
            self.assertTrue(all(pad.shape == (4,2) for pad in pads))

    def test_actual_feet_exclude_floating_geometry(self):
        mesh = trimesh.creation.box([2.,2.,1.])
        mesh.apply_translation([0.,0.,.6])
        self.assertEqual(len(actual_footprint(mesh, np.eye(3), np.zeros(3))), 0)
        mesh.apply_translation([0.,0.,-.1])
        self.assertEqual(len(actual_footprint(mesh, np.eye(3), np.zeros(3))), 4)

    def test_original_object_pivot_counts_but_invented_foot_does_not(self):
        feet = np.array([[0.,0.], [0.,1.]])
        demands = np.array([[.5,.25]])
        self.assertTrue(footprint_coverage(feet, [1.,0.,0.], demands)['passed'])
        self.assertFalse(footprint_coverage(feet, [0.,0.,0.], demands)['passed'])


if __name__ == '__main__':
    unittest.main()
