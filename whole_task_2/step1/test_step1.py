"""Check object-frame regions and their reserved robot direction budgets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation
import trimesh

from robots_config.constraints import (load_config, surface_point_allowed,
                                       tool_direction_allowed_at_surface)
from .common import ROOT, load_shape
from .step1_3 import choose_working_areas, requested_region_mask, select_working_areas
from .work_disks import clip_surface_disk


class WorkingAreaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = load_shape(ROOT / "objects/B/mesh.stl")
        cls.config = load_config(ROOT / "robots_config/08_abb_irb1200_5_90.json")

    def test_source_face_ids_and_shape_are_preserved(self):
        source = trimesh.load(ROOT / "objects/B/mesh.stl", force="mesh", process=False)
        np.testing.assert_array_equal(self.mesh.triangles, source.triangles)
        self.assertEqual(len(self.mesh.faces), len(source.faces))

    def test_regions_fit_reserved_cones_and_are_connected(self):
        labels, approaches, areas, budget, _ = choose_working_areas(
            self.mesh, self.config, count=4)
        self.assertEqual(len(areas), 4)
        self.assertAlmostEqual(budget["working_normal_cap_half_angle_deg"], 19)
        axis_world = np.array(self.config["task_constraint"]["working_direction_world"])
        for i, area in enumerate(areas):
            ids = np.array(area["face_ids"])
            np.testing.assert_array_equal(np.flatnonzero(labels == i), ids)
            self.assertIsNone(area["world_pose"])
            pairs = self.mesh.face_adjacency
            pairs = pairs[np.all(labels[pairs] == i, axis=1)]
            positions = np.full(len(labels), -1)
            positions[ids] = np.arange(len(ids))
            graph = coo_matrix((np.ones(len(pairs)),
                                (positions[pairs[:, 0]], positions[pairs[:, 1]])),
                               shape=(len(ids), len(ids)))
            self.assertEqual(connected_components(graph, directed=False)[0], 1)
            # The saved screening alignment verifies the actual fixed window;
            # it is explicitly distinct from the still-unselected working pose.
            alignment = area["screening_alignment"]
            rotation = np.array(alignment["rotation_world_object"])
            translation = np.array(alignment["translation_world_object_m"])
            world_normals = self.mesh.face_normals[ids] @ rotation.T
            world_tool = -approaches[ids] @ rotation.T
            for direction, normal in zip(world_tool, world_normals):
                self.assertTrue(tool_direction_allowed_at_surface(
                    self.config, direction, normal))
            self.assertLessEqual(area["normal_spread_max_deg"], 19+1e-8)
            self.assertLessEqual(area["tool_deviation_max_deg"], 14.25+1e-8)
            self.assertLessEqual(area["incidence_max_deg"], 4.75+1e-8)
            vertices = area["_geometry"]["triangles"].reshape(-1, 3)
            self.assertTrue(all(surface_point_allowed(self.config, point)
                                for point in vertices @ rotation.T+translation))
            self.assertEqual(area["boundary_components"], 1)
            self.assertEqual(area["euler_number"], 1)
            self.assertLessEqual(area["radius_m"], area["usable_window_radius_m"]+1e-12)
            geometry = area["_geometry"]
            source = self.mesh.triangles[geometry["source_face_ids"]]
            reconstructed = np.einsum("nij,njk->nik", geometry["barycentric"], source)
            np.testing.assert_allclose(reconstructed, geometry["triangles"], atol=1e-12)

    def test_five_percent_inset_reduces_the_convex_work_region(self):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=.1)
        full = choose_working_areas(sphere, self.config, count=1, inset_fraction=0)
        inset = choose_working_areas(sphere, self.config, count=1, inset_fraction=.05)
        self.assertLess(inset[2][0]["area_m2"], full[2][0]["area_m2"])

    def test_rotating_the_whole_station_does_not_freeze_an_object_pose(self):
        alternate = copy.deepcopy(self.config)
        rotation = Rotation.from_euler("xyz", [30, 25, 80], degrees=True).as_matrix()
        task = alternate["task_constraint"]
        for key in ["working_direction_world"]:
            task[key] = (rotation @ task[key]).tolist()
        window = task["world_position_constraint"]
        for key in ["outward_normal_world", "center_world_m"]:
            window[key] = (rotation @ window[key]).tolist()
        base = alternate["feasibility"]["base_pose_world"]
        base["position_m"] = (rotation @ base["position_m"]).tolist()
        base["rotation_matrix"] = (rotation @ base["rotation_matrix"]).tolist()
        first = choose_working_areas(self.mesh, self.config, count=2)
        second = choose_working_areas(self.mesh, alternate, count=2)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])

    def test_smaller_metric_window_limits_the_disk_radius_and_area(self):
        smaller = copy.deepcopy(self.config)
        smaller["task_constraint"]["world_position_constraint"]["radius_m"] = .002
        wide = choose_working_areas(self.mesh, self.config, count=1)[2][0]
        small = choose_working_areas(self.mesh, smaller, count=1)[2][0]
        self.assertAlmostEqual(small["radius_m"], .0019)
        self.assertLess(small["area_m2"], wide["area_m2"])
        self.assertAlmostEqual(small["projected_area_m2"]/(np.pi*.0019**2), 1, delta=.00011)

    def test_standoff_and_tool_length_define_the_screening_positions(self):
        config = load_config(ROOT / "robots_config/02_franka_fr3.json")
        area = choose_working_areas(self.mesh, config, count=1)[2][0]
        geometry = area["_geometry"]
        rotation = np.array(area["screening_alignment"]["rotation_world_object"])
        translation = np.array(area["screening_alignment"]["translation_world_object_m"])
        surface = geometry["triangles"].mean(axis=1) @ rotation.T+translation
        tool = geometry["tool_directions_world"]
        np.testing.assert_allclose(surface-geometry["tcp_positions_world_m"], .2*tool, atol=1e-12)
        length = area["tool_length_m"]
        np.testing.assert_allclose(geometry["tcp_positions_world_m"]-geometry["flange_positions_world_m"],
                                   length*tool, atol=1e-12)

    def test_named_sites_share_a_fixed_station_and_have_different_normals(self):
        request = json.loads((ROOT / "whole_task_2/step1/B_working_regions.json").read_text())
        config = load_config(ROOT / "robots_config/04_kuka_lbr_med14_r820.json")
        _, _, areas, _, _ = choose_working_areas(
            self.mesh, config, count=4, region_requests=request["regions"],
            minimum_circle_fill=request["minimum_projected_circle_fill"],
            minimum_normal_separation_deg=request["minimum_reference_normal_separation_deg"])
        self.assertEqual([a["name"] for a in areas], ["ear", "leg", "back", "base"])
        window = config["task_constraint"]["world_position_constraint"]
        for area, site in zip(areas, request["regions"]):
            self.assertTrue(requested_region_mask(self.mesh, site)[area["face_ids"]].all())
            self.assertGreaterEqual(area["projected_circle_fill"], .9-1e-9)
            self.assertEqual(area["processing_center_world_m"], window["center_world_m"])
            self.assertAlmostEqual(area["base_to_processing_center_distance_m"],
                                   window["base_to_center_distance_m"])
            self.assertIsNone(area["world_pose"])
            alignment = area["screening_alignment"]
            rotation = np.array(alignment["rotation_world_object"])
            translation = np.array(alignment["translation_world_object_m"])
            np.testing.assert_allclose(rotation @ area["center_object_m"]+translation,
                                       window["center_world_m"], atol=1e-12)
            vertices = area["_geometry"]["triangles"].reshape(-1, 3) @ rotation.T+translation
            self.assertTrue(all(surface_point_allowed(config, point) for point in vertices))
        normals = np.array([a["mean_surface_normal_object"] for a in areas])
        separation = np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1)))
        self.assertGreaterEqual(separation[np.triu_indices(4, 1)].min(), 30-1e-8)

    def test_a_clipped_boundary_is_allowed_but_a_thin_strip_is_rejected(self):
        def rectangle(left, right):
            return trimesh.Trimesh(
                vertices=[[left, -1, 0], [right, -1, 0], [right, 1, 0], [left, 1, 0]],
                faces=[[0, 1, 2], [0, 2, 3]], process=False)
        clipped = clip_surface_disk(rectangle(-1, .8), np.array([0, 1]),
                                     np.zeros(3), np.array([0, 0, 1]), 1, minimum_fill=.9)
        self.assertIsNotNone(clipped)
        self.assertGreater(clipped["projected_circle_fill"], .9)
        self.assertLess(clipped["projected_circle_fill"], .99)
        self.assertEqual(clipped["boundary_components"], 1)
        self.assertEqual(clipped["euler_number"], 1)
        strip = clip_surface_disk(rectangle(-.15, .15), np.array([0, 1]),
                                  np.zeros(3), np.array([0, 0, 1]), 1, minimum_fill=.9)
        self.assertIsNone(strip)

    def test_changed_robot_snapshot_is_rejected_before_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Path(directory) / "B/step1/step1.1"
            robot.mkdir(parents=True)
            (robot / "config.json").write_text(json.dumps(self.config))
            (robot / "selection.json").write_text('{"config_sha256":"wrong"}')
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                select_working_areas(directory)


if __name__ == "__main__":
    unittest.main()
