"""Check the whole-object, three-dimensional planning interpretation."""
import copy
import unittest

import numpy as np
import trimesh

from robots_config.constraints import load_config
from .common import ROOT, load_shape
from .pose_coverage import (build_candidates, evaluate_alignment, evaluate_alignments,
                            evaluate_target_alignment,
                            representative_indices, sphere_workspace, spatial_workspace_mask,
                            face_components, expand_same_alignment, coverage_angular_budget)
from .step1_3 import _adjacency


class PoseCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT/"robots_config/04_kuka_lbr_med14_r820.json")

    def test_config_angle_controls_space_without_cropping_surface_normals(self):
        config = copy.deepcopy(self.config)
        for angle in (25., 35.):
            config['task_constraint']['cone_half_angle_deg'] = angle
            budget = coverage_angular_budget(config)
            self.assertIsNone(budget['full_normal_cap_half_angle_deg'])
            self.assertFalse(budget['normal_hard_limit_applied'])
            self.assertAlmostEqual(sphere_workspace(config)['spatial_cone']['usable_half_angle_deg'], .95*angle)
        mesh = trimesh.Trimesh(vertices=[[-.002,-.002,0],[.002,-.002,0],[0,.002,0]], faces=[[0,1,2]], process=False)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30), [1,0,0]))
        config['task_constraint']['cone_half_angle_deg'] = 25.
        self.assertIsNotNone(evaluate_alignment(mesh, config, [0,0,1]))
        config['task_constraint']['cone_half_angle_deg'] = 35.
        self.assertIsNotNone(evaluate_alignment(mesh, config, [0,0,1]))

    def test_disconnected_pieces_are_separate_candidates_without_losing_faces(self):
        mesh = trimesh.Trimesh(
            vertices=[[-.03, -.01, -.02], [-.01, -.01, -.02], [-.01, .01, -.02], [-.03, .01, -.02],
                      [.01, -.01, .02], [.03, -.01, .02], [.03, .01, .02], [.01, .01, .02]],
            faces=[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], process=False)
        candidates = evaluate_alignments(mesh, self.config, np.array([0., 0., 1.]))
        self.assertEqual(len(candidates), 2)
        np.testing.assert_array_equal(np.unique(np.concatenate([c["face_ids"] for c in candidates])), [0, 1, 2, 3])
        for candidate in candidates:
            self.assertEqual(len(candidate["face_ids"]), 2)
            self.assertEqual(len(face_components(candidate["face_ids"], _adjacency(mesh))), 1)
        self.assertEqual(len(mesh.split(only_watertight=False)), 2)
        # The two sheets lie 20 mm from their reference plane, exceeding the
        # obsolete +/-5 mm slab, but remain inside the 3D workspace estimate.
        self.assertGreater(np.abs(mesh.vertices[:, 2]).min(), .005)

    def test_workspace_applies_to_flange_with_tool_length_included(self):
        mesh = trimesh.Trimesh(vertices=[[-.002, -.002, 0], [.002, -.002, 0], [0, .002, 0]],
                              faces=[[0, 1, 2]], process=False)
        config = copy.deepcopy(self.config)
        config["robot_specification"]["reach_m"] = .2
        task = config["task_constraint"]
        task["working_direction_world"] = [0, 0, -1]
        task["world_position_constraint"]["center_world_m"] = [0, 0, .1]
        # Surface at 100 mm is inside 190 mm; the 120 mm tool puts the
        # flange at 220 mm. Checking surface distance alone would be wrong.
        self.assertIsNone(evaluate_alignment(mesh, config, [0, 0, 1]))
        config["feasibility"]["tcp_transform"]["translation_flange_m"][2] = .02
        self.assertIsNotNone(evaluate_alignment(mesh, config, [0, 0, 1]))

    def test_aiming_point_lands_on_axis_inside_first_ring_and_keeps_its_component(self):
        mesh = trimesh.Trimesh(
            vertices=[[-.02, -.002, 0], [-.016, -.002, 0], [-.02, .002, 0],
                      [0, -.005, 0], [.01, -.005, 0], [0, .005, 0]],
            faces=[[0, 1, 2], [3, 4, 5]], process=False)
        candidates = evaluate_alignments(mesh, self.config, [0, 0, 1],
            center_object_m=mesh.triangles_center[0])
        self.assertEqual(len(candidates), 2)
        self.assertGreater(candidates[0]["area_m2"], mesh.area_faces[0])
        chosen = evaluate_target_alignment(mesh, self.config,
            dict(site="test_surface", source_face_id=0))
        np.testing.assert_array_equal(chosen["face_ids"], [0])
        point_world = (chosen["rotation_world_object"] @ mesh.triangles_center[0]
                       + chosen["translation_world_object_m"])
        workspace=sphere_workspace(self.config);cone=workspace['spatial_cone']
        delta=point_world-np.asarray(cone['apex_world_m']);axis=np.asarray(cone['axis_world'])
        depth=float(delta@axis)
        np.testing.assert_allclose(delta,depth*axis,atol=1e-12)
        self.assertGreater(depth,0.)
        self.assertLess(depth,cone['maximum_axial_distance_m'])
        self.assertEqual(chosen["display_target"]["source_face_id"], 0)

    def test_configured_local_ball_actually_limits_the_selected_surface(self):
        config = copy.deepcopy(self.config)
        workspace = sphere_workspace(config)
        self.assertEqual(workspace["center_world_m"], [0, -.5, .3])
        self.assertAlmostEqual(workspace["usable_radius_m"], .095)
        self.assertAlmostEqual(workspace["base_reach_estimate"]["usable_radius_m"], .779)
        triangle = (np.array([[[.040, -.001, 0], [.042, -.001, 0], [.040, .001, 0]]])
                    +workspace['center_world_m']-.02*np.asarray(workspace['spatial_cone']['axis_world']))
        self.assertTrue(spatial_workspace_mask(triangle, workspace)[0])
        config["task_constraint"]["world_position_constraint"]["radius_m"] = .04
        self.assertFalse(spatial_workspace_mask(triangle, sphere_workspace(config))[0])

    def test_first_solid_ring_rejects_forward_extension_even_inside_cone_and_ball(self):
        config = copy.deepcopy(self.config)
        config["task_constraint"]["working_direction_world"] = [0, 0, -1]
        config["task_constraint"]["world_position_constraint"]["center_world_m"] = [0, 0, .1]
        workspace = sphere_workspace(config)
        vertices = np.array([[.064, -.001, .1], [.066, -.001, .1], [.065, .001, .1]])
        self.assertLess(np.linalg.norm(vertices-[0, 0, .1], axis=1).max(), .095)
        self.assertFalse(spatial_workspace_mask(vertices[None], workspace)[0])
        vertices[:] = [[.01,-.001,.11],[.012,-.001,.11],[.01,.001,.11]]
        self.assertTrue(spatial_workspace_mask(vertices[None], workspace)[0])
        self.assertAlmostEqual(workspace['spatial_cone']['reference_axial_distance_m'],.12)
        self.assertAlmostEqual(workspace['spatial_cone']['maximum_axial_distance_m'],.114)
        # A face wholly inside the old cone/ball and before the dashed ring
        # must still fail once it crosses the first solid ring's work limit.
        vertices[:,2]=.06
        self.assertLess(np.linalg.norm(vertices-[0,0,.1],axis=1).max(),.095)
        self.assertFalse(spatial_workspace_mask(vertices[None],workspace)[0])
        # All three vertices are tested, including one over the usable cap.
        vertices[:,2]=.11;vertices[2,2]=.105
        self.assertFalse(spatial_workspace_mask(vertices[None],workspace)[0])
        # The cone remains one-sided and is also bounded by the local ball.
        vertices[:] = [[0, 0, .23], [.001, 0, .23], [0, .001, .23]]
        self.assertFalse(spatial_workspace_mask(vertices[None], workspace)[0])
        vertices[:, 2] = -.01
        self.assertFalse(spatial_workspace_mask(vertices[None], workspace)[0])

    def test_45_and_60_degree_limits_do_not_rotate_the_needle(self):
        mesh = trimesh.Trimesh(vertices=[[-.01, -.01, 0], [.01, -.01, 0], [0, .01, 0]],
                              faces=[[0, 1, 2]], process=False)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(40), [1, 0, 0]))
        self.assertIsNotNone(evaluate_alignment(mesh, self.config, [0, 0, 1], normal_half_angle_deg=45))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(10), [1, 0, 0]))
        self.assertIsNone(evaluate_alignment(mesh, self.config, [0, 0, 1], normal_half_angle_deg=45))
        candidate = evaluate_alignment(mesh, self.config, [0, 0, 1], normal_half_angle_deg=60)
        self.assertIsNotNone(candidate)
        outward_world = candidate["outward_tool_object"] @ candidate["rotation_world_object"].T
        np.testing.assert_allclose(-outward_world[0], self.config["task_constraint"]["working_direction_world"], atol=1e-12)
        np.testing.assert_allclose(candidate["outward_tool_object"], [[0, 0, 1]], atol=1e-12)

    def test_bunny_candidates_are_large_and_share_valid_world_constraints(self):
        mesh = load_shape(ROOT/"objects/B/mesh.stl")
        candidates = build_candidates(mesh, self.config, direction_count=64,
                                       normal_half_angle_deg=45)
        self.assertGreater(candidates[0]["area_m2"]/mesh.area, .02)
        # Candidate generation does not consume faces; overlap remains possible.
        self.assertTrue(any(np.any(candidates[0]["mask"] & candidate["mask"])
                            for candidate in candidates[1:]))
        workspace = sphere_workspace(self.config)
        neighbors = _adjacency(mesh)
        for candidate in candidates:
            self.assertEqual(len(face_components(candidate["face_ids"], neighbors)), 1)
        for j in representative_indices(candidates, 4):
            candidate = candidates[j]
            ids = candidate["face_ids"]
            rotation = candidate["rotation_world_object"]
            translation = candidate["translation_world_object_m"]
            outward = candidate["outward_tool_object"] @ rotation.T
            normals = mesh.face_normals[ids] @ rotation.T
            expected = np.tile(self.config["task_constraint"]["working_direction_world"], (len(ids), 1))
            np.testing.assert_allclose(-outward, expected, atol=1e-12)
            surface = mesh.triangles[ids] @ rotation.T+translation
            self.assertTrue(spatial_workspace_mask(surface, workspace).all())
            inward_dot_axis = -normals @ np.asarray(workspace["spatial_cone"]["axis_world"])
            self.assertGreaterEqual(inward_dot_axis.min(), np.cos(np.radians(42.75))-1e-12)
            self.assertLessEqual(np.linalg.norm(surface-[0, -.5, .3], axis=2).max(), .095+1e-9)
            flange = surface+.12*outward[:, None]
            self.assertLessEqual(np.linalg.norm(flange, axis=2).max(), .82*.95+1e-9)
            self.assertAlmostEqual(candidate["area_m2"], mesh.area_faces[ids].sum())
            expanded = expand_same_alignment(mesh, self.config, candidate, 60)
            self.assertTrue(np.isin(ids, expanded).all())
            self.assertEqual(len(face_components(expanded, neighbors)), 1)
            self.assertTrue(spatial_workspace_mask(mesh.triangles[expanded] @ rotation.T+translation, workspace).all())
            expanded_normals = mesh.face_normals[expanded] @ rotation.T
            self.assertGreaterEqual((-expanded_normals @ np.asarray(workspace["spatial_cone"]["axis_world"])).min(), np.cos(np.radians(57))-1e-12)


if __name__ == "__main__":
    unittest.main()
