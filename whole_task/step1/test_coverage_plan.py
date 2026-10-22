"""Properties of the full-surface baseline rather than fixture heuristics."""
import unittest
import numpy as np
from .step1_2 import greedy_cover, IncompleteCandidatePool


class CoverageTests(unittest.TestCase):
    def test_weights_are_surface_area_not_triangle_count(self):
        masks = np.array([[1, 0, 0, 0], [0, 1, 1, 1]], dtype=bool)
        selected, _ = greedy_cover(masks, [10, 1, 1, 1])
        self.assertEqual(selected.tolist(), [0, 1])

    def test_keeps_overlap_and_whole_regions(self):
        masks = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
        selected, trace = greedy_cover(masks, [1, 1, 1])
        self.assertEqual(selected.tolist(), [0, 1])
        self.assertEqual(trace[1]['area_m2'], 2)
        self.assertEqual(trace[1]['newly_covered_area_m2'], 1)
        self.assertEqual(trace[1]['face_count'], 2)
        self.assertEqual(trace[-1]['cumulative_coverage_fraction'], 1)

    def test_count_emerges_and_no_tolerance_discards_tiny_faces(self):
        masks = np.eye(4, dtype=bool)
        selected, trace = greedy_cover(masks, [1, .1, .01, 1e-20])
        self.assertEqual(len(selected), 4)
        self.assertEqual(trace[-1]['remaining_face_count'], 0)

    def test_uncovered_pool_fails_with_source_face_diagnostic(self):
        with self.assertRaises(IncompleteCandidatePool) as raised:
            greedy_cover([[1, 0, 0]], [1, 2, 3])
        self.assertEqual(raised.exception.face_ids.tolist(), [1, 2])
        self.assertEqual(raised.exception.area_m2, 5)

    def test_ignored_display_candidate_cannot_leak_into_plan(self):
        masks = [[1, 1, 1], [1, 0, 0], [0, 1, 1]]
        selected, _ = greedy_cover(masks, [1, 1, 1], [2, 1])
        self.assertEqual(selected.tolist(), [2, 1])

    def test_equal_gain_uses_source_index(self):
        selected, _ = greedy_cover(np.eye(2, dtype=bool), [1, 1], [1, 0])
        self.assertEqual(selected.tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
