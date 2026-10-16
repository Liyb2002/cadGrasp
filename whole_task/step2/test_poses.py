import unittest

import numpy as np

from .poses import load_inputs, formalize
from .transforms import rigid_transform, inverse, points, directions


class TransformTests(unittest.TestCase):
    def test_frame_direction_translation_and_normals(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        transform = rigid_transform(rotation, [2., 3., 4.])
        np.testing.assert_allclose(points(transform, [[1., 0, 0]]), [[2., 4., 4.]])
        np.testing.assert_allclose(directions(transform, [[1., 0, 0]]), [[0., 1., 0.]])
        np.testing.assert_allclose(points(inverse(transform), [[2., 4., 4.]]), [[1., 0., 0.]])

    def test_rejects_reflection_scaling_and_nonfinite_transforms(self):
        for rotation in (np.diag([-1., 1, 1]), np.eye(3)*2, np.eye(3)*np.nan):
            with self.assertRaises(ValueError):
                rigid_transform(rotation, [0, 0, 0])
        with self.assertRaises(ValueError):
            inverse(np.eye(4)*2)


class PoseExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh, cls.source, cls.arrays, cls.config, _ = load_inputs()
        cls.report, cls.data = formalize(cls.mesh, cls.source, cls.arrays, cls.config)

    def test_selected_source_faces_and_object_alignments_are_preserved(self):
        np.testing.assert_array_equal(self.data["working_area_face_membership"], self.arrays["working_area_face_membership"])
        for pose, source in zip(self.report["poses"], self.source["working_areas"]):
            np.testing.assert_array_equal(np.asarray(pose["world_from_object"])[:3, :3],
                                          source["screening_alignment"]["rotation_world_object"])
            self.assertEqual(pose["working_face_ids"], source["face_ids"])

    def test_filled_faces_are_not_silently_certified(self):
        self.assertFalse(self.report["all_selected_faces_pass_geometry"])
        for pose in self.report["poses"]:
            audit = pose["geometry_audit"]
            self.assertTrue(set(audit["unresolved_face_ids"]) <= set(pose["filled_face_ids"]))
            self.assertTrue(set(pose["raw_face_ids"]) <= set(audit["geometrically_passed_face_ids"]))
            self.assertFalse(pose["executable_working_pose_verified"])

    def test_station_is_identical_and_aimed_landmarks_land_at_its_origin(self):
        station = np.asarray(self.report["fixed_station"]["world_from_station"])
        for pose in self.report["poses"]:
            np.testing.assert_allclose(station @ np.asarray(pose["station_from_object"]), pose["world_from_object"], atol=1e-12)
            if pose["display_target"]:
                target = pose["display_target"]["point_object_m"]
                np.testing.assert_allclose(points(np.asarray(pose["world_from_object"]), target), station[:3, 3], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
