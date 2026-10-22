"""Enclosure thresholds, true surface holes and projection false positives."""
import unittest

import numpy as np
import trimesh

from .fill_working_holes import angular_surround_deg, fill_surrounded_surface


def disk_with_ring(count=48, extra_back_face=False):
    angle = np.arange(count)*2*np.pi/count
    circle = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(count)])
    vertices = np.vstack([[[0, 0, 0]], .3*circle, circle])
    faces = []
    for i in range(count):
        j = (i+1) % count
        faces.append([0, 1+i, 1+j])
    for i in range(count):
        j = (i+1) % count
        faces.extend([[1+i, 1+count+i, 1+count+j], [1+i, 1+count+j, 1+j]])
    if extra_back_face:
        start = len(vertices)
        vertices = np.vstack([vertices, [[-.05, -.05, -.1], [.05, -.05, -.1], [0, .05, -.1]]])
        faces.append([start, start+1, start+2])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), np.arange(count, 3*count)


class HoleFillingTests(unittest.TestCase):
    def test_strict_270_degree_criterion_and_no_double_counting(self):
        for degrees in [240, 270, 271, 300, 360]:
            angles = np.radians(np.arange(degrees+1))
            points = np.column_stack([np.cos(angles), np.sin(angles)])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            measured = angular_surround_deg([0, 0], np.concatenate([segments, segments]))
            self.assertAlmostEqual(measured, degrees, places=8)
            self.assertEqual(measured > 270+1e-8, degrees > 270)

    def test_enclosed_hole_is_filled_with_original_face_ids(self):
        mesh, selected = disk_with_ring()
        result = fill_surrounded_surface(mesh, selected, [0, 0, 1], np.ones(len(mesh.faces), dtype=bool))
        np.testing.assert_array_equal(result["face_ids"], np.arange(len(mesh.faces)))
        np.testing.assert_array_equal(result["filled_face_ids"], np.arange(48))
        np.testing.assert_allclose(result["filled_enclosure_angles_deg"], 360, atol=1e-8)
        self.assertTrue(result["requires_step2_verification"])

    def test_bounds_and_disconnected_back_surface_are_not_filled(self):
        mesh, selected = disk_with_ring(extra_back_face=True)
        eligible = np.ones(len(mesh.faces), dtype=bool)
        eligible[0] = False
        result = fill_surrounded_surface(mesh, selected, [0, 0, 1], eligible)
        self.assertNotIn(0, result["face_ids"])
        self.assertNotIn(len(mesh.faces)-1, result["face_ids"])
        self.assertEqual(len(result["filled_face_ids"]), 47)


if __name__ == "__main__":
    unittest.main()
