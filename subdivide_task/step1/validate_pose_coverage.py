"""Audit saved fixed-cone candidates and their rendered composite."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from robots_config.constraints import load_config
from .common import DEFAULT_OUTPUT, ROOT, load_shape, sha256, write_json
from .compose_working_areas import IMAGE_SUFFIXES
from .pose_coverage import POLICY, coverage_angular_budget, face_components, sphere_workspace, spatial_workspace_mask
from .step1_3 import _adjacency, _visible_faces
from .fill_working_holes import fill_surrounded_surface
from .first_layer import first_layer_faces


def _angle(dot):
    return np.degrees(np.arccos(np.clip(dot, -1, 1)))


def validate(output_root=DEFAULT_OUTPUT, object_name="B"):
    stage = Path(output_root)/object_name/"step1"
    output = stage/"step1.3"
    report_path = output/"pose_coverage.json"
    report = json.loads(report_path.read_text())
    if report.get('geometry_model') in ('fixed_length_needle_in_bounded_spherical_sector', 'needle_spherical_sector_surface_intersection'):
        from .validate_needle import validate as validate_needle
        return validate_needle(output_root, object_name)
    target_policy = json.loads(POLICY.read_text()).get("display_target_overrides", {})
    targets = ({target["ordinal"]: target for target in target_policy["targets"]}
               if target_policy.get("object") == object_name else {})
    if targets:
        assert report["shape_sha256"] == target_policy["shape_sha256"]
    config_path = stage/"step1.1/config.json"
    config = load_config(config_path)
    assert sha256(config_path) == report["config_sha256"]
    for field in ("input_sha256", "visualization_source_sha256"):
        for name, digest in report[field].items():
            assert sha256(ROOT/name) == digest, (field, name)
    for name, digest in report["computation_source_sha256"].items():
        assert sha256(Path(__file__).parent/name) == digest, name
    for name, digest in report["artifact_sha256"].items():
        assert sha256(output/name) == digest, name

    mesh = load_shape(ROOT/"objects"/object_name/"mesh.stl")
    assert sha256(ROOT/"objects"/object_name/"mesh.stl") == report["shape_sha256"]
    neighbors = _adjacency(mesh)
    with np.load(output/report["arrays"]) as archive:
        arrays = {name: archive[name] for name in archive.files}
    np.testing.assert_array_equal(arrays["vertices_object"], mesh.vertices)
    np.testing.assert_array_equal(arrays["faces"], mesh.faces)
    np.testing.assert_array_equal(arrays["source_face_ids"], np.arange(len(mesh.faces)))
    np.testing.assert_allclose(arrays["face_area_m2"], mesh.area_faces, rtol=1e-12)
    workspace = sphere_workspace(config, report["workspace"]["inset_fraction"])
    assert workspace == report["workspace"]
    budget = coverage_angular_budget(config, workspace["inset_fraction"])
    assert budget == report["angular_budget"]
    cone = workspace["spatial_cone"]
    assert report['first_solid_ring_axial_limit_applied']
    assert not report['dashed_extension_selectable']
    axis = np.asarray(cone["axis_world"])
    np.testing.assert_allclose(report["fixed_tool_direction_world"], axis, atol=1e-12)
    fixed_rotation = np.asarray(report["fixed_tcp_rotation_world"])
    np.testing.assert_allclose(fixed_rotation[:, 2], axis, atol=1e-12)
    np.testing.assert_allclose(fixed_rotation.T @ fixed_rotation, np.eye(3), atol=1e-12)
    assert not report["tool_orientation_changes_allowed"]
    apex = np.asarray(cone["apex_world_m"])
    reach = workspace["base_reach_estimate"]
    length = float(config["feasibility"]["tcp_transform"]["translation_flange_m"][2])
    gap = float(config["task_constraint"]["tcp_to_surface_distance_m"]["nominal"])
    offsets = arrays["candidate_tool_offsets"]
    maxima = dict(surface_radius_m=0., spatial_cone_angle_deg=0.,
                  local_normal_angle_deg=0., tool_direction_angle_deg=0.,
                  incidence_angle_deg=0., flange_base_radius_m=0.)
    minimum_depth, maximum_depth = float("inf"), float('-inf')
    membership = arrays["candidate_face_membership"]
    assert len(membership) == report["candidate_count"]
    summaries = {}
    selected = set(arrays["representative_candidate_indices"].tolist())
    checked_face_count = 0
    for index, mask in enumerate(membership):
        ids = arrays["candidate_tool_face_ids"][offsets[index]:offsets[index+1]]
        np.testing.assert_array_equal(np.flatnonzero(mask), ids)
        assert len(face_components(ids, neighbors)) == 1, index
        rotation = arrays["candidate_rotation_world_object"][index]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(rotation)-1) < 1e-12
        translation = arrays["candidate_translation_world_object_m"][index]
        surface = mesh.triangles[ids] @ rotation.T+translation
        outward_object = arrays["candidate_outward_tool_object"][offsets[index]:offsets[index+1]]
        np.testing.assert_allclose(np.linalg.norm(outward_object, axis=1), 1, atol=1e-12)
        outward = outward_object @ rotation.T
        np.testing.assert_allclose(-outward, np.tile(axis, (len(ids), 1)), atol=1e-12)
        normals = mesh.face_normals[ids] @ rotation.T
        delta = surface-apex
        depth = delta @ axis
        minimum_depth = min(minimum_depth, float(depth.min()))
        maximum_depth = max(maximum_depth, float(depth.max()))
        assert depth.min() >= -1e-12
        assert depth.max() <= cone['maximum_axial_distance_m']+1e-12
        values = dict(
            surface_radius_m=float(np.linalg.norm(surface-workspace["center_world_m"], axis=2).max()),
            spatial_cone_angle_deg=float(_angle(depth/np.linalg.norm(delta, axis=2)).max()),
            local_normal_angle_deg=float(_angle(-normals @ axis).max()),
            tool_direction_angle_deg=float(np.degrees(np.arctan2(np.linalg.norm(np.cross(-outward, axis), axis=1), -outward @ axis)).max()),
            incidence_angle_deg=float(_angle(np.sum(outward*normals, axis=1)).max()),
            flange_base_radius_m=float(np.linalg.norm(surface+(length+gap)*outward[:, None]-reach["center_world_m"], axis=2).max()))
        limits = dict(surface_radius_m=workspace["usable_radius_m"],
                      spatial_cone_angle_deg=cone["usable_half_angle_deg"],
                      local_normal_angle_deg=budget["working_normal_cap_half_angle_deg"],
                      tool_direction_angle_deg=budget["working_tool_cone_half_angle_deg"],
                      incidence_angle_deg=budget["working_incidence_half_angle_deg"],
                      flange_base_radius_m=reach["usable_radius_m"])
        for name, value in values.items():
            if limits[name] is not None:
                assert value <= limits[name]+1e-7, (index, name, value, limits[name])
            maxima[name] = max(maxima[name], value)
        if index in selected:
            assert first_layer_faces(mesh, arrays['candidate_outward_axis_object'][index])[ids].all()
            checked_face_count += len(ids)
            summaries[index] = dict(candidate_index=index, component_count=1,
                                    area_cm2=float(mesh.area_faces[ids].sum()*1e4),
                                    face_count=len(ids), maxima=values)
    examples = []
    chosen = arrays["working_area_face_membership"]
    filled_masks = arrays["working_area_filled_face_membership"]
    assert report["hole_filling"]["filled_faces_require_step2_verification"]
    assert report["all_colored_faces_geometrically_verified"]
    assert report['local_normal_cap_applied'] is False
    filled_checked_face_count = 0
    for ordinal, (example, index) in enumerate(zip(report["working_areas"], arrays["representative_candidate_indices"])):
        assert example["candidate_index"] == index
        raw_ids = np.flatnonzero(membership[index])
        np.testing.assert_array_equal(example["raw_face_ids"], raw_ids)
        assert abs(example["raw_area_m2"]*1e4-summaries[index]["area_cm2"]) < 1e-10
        final_ids = np.flatnonzero(chosen[ordinal])
        added = np.flatnonzero(filled_masks[ordinal])
        np.testing.assert_array_equal(final_ids, example["face_ids"])
        np.testing.assert_array_equal(added, example["filled_face_ids"])
        np.testing.assert_array_equal(chosen[ordinal], membership[index] | filled_masks[ordinal])
        assert not np.any(membership[index] & filled_masks[ordinal])
        assert len(face_components(final_ids, neighbors)) == 1
        rotation = arrays["candidate_rotation_world_object"][index]
        translation = arrays["candidate_translation_world_object_m"][index]
        surface = mesh.triangles @ rotation.T+translation
        flange = surface-(length+gap)*axis
        eligible = (spatial_workspace_mask(surface, workspace)
                    & (np.linalg.norm(flange-reach["center_world_m"], axis=2).max(axis=1) <= reach["usable_radius_m"]+1e-12))
        assert eligible[final_ids].all()
        object_axis = arrays["candidate_outward_axis_object"][index]
        visible_layer = first_layer_faces(mesh, object_axis)
        eligible &= visible_layer
        assert visible_layer[final_ids].all()
        target = example.get("display_target")
        assert bool(target) == (ordinal in targets)
        if target:
            requested = targets[ordinal]
            assert target["site"] == requested["site"]
            face = requested["source_face_id"]
            assert target["source_face_id"] == face and face in raw_ids
            point = mesh.triangles_center[face]
            np.testing.assert_allclose(target["point_object_m"], point, atol=1e-12)
            np.testing.assert_allclose(rotation @ point+translation, target['point_world_m'], atol=1e-12)
            target_delta = np.asarray(target['point_world_m'])-apex
            target_depth = float(target_delta @ axis)
            np.testing.assert_allclose(target_delta, target_depth*axis, atol=1e-12)
            fractions = workspace['surface_alignment_axial_fractions_of_usable_depth']
            assert np.min(np.abs(target_depth-cone['maximum_axial_distance_m']*np.asarray(fractions))) < 1e-12
            expected_axis = np.array(requested.get("outward_axis_object", mesh.face_normals[face]), dtype=float, copy=True)
            expected_axis /= np.linalg.norm(expected_axis)
            np.testing.assert_allclose(object_axis, expected_axis, atol=1e-12)
            np.testing.assert_allclose(target["outward_axis_object"], object_axis, atol=1e-12)
        replay = fill_surrounded_surface(mesh, raw_ids, object_axis, eligible,
            report["hole_filling"]["surround_threshold_deg_strict"])
        np.testing.assert_array_equal(replay["face_ids"], final_ids)
        np.testing.assert_array_equal(replay["filled_face_ids"], added)
        np.testing.assert_allclose(replay["filled_enclosure_angles_deg"], example["filled_enclosure_angles_deg"], atol=1e-8)
        np.testing.assert_allclose(replay["filled_enclosure_angles_deg"], arrays["working_area_fill_enclosure_deg"][ordinal, added], atol=1e-8)
        assert np.all(replay["filled_enclosure_angles_deg"] > 270+1e-8)
        normal_angle = _angle(mesh.face_normals[added] @ object_axis)
        failed_angle = np.empty(0, dtype=int)
        visible = visible_layer[added]
        np.testing.assert_array_equal(failed_angle, example["normal_limit_failed_filled_face_ids"])
        np.testing.assert_array_equal(added[~visible], example["sampled_visibility_failed_filled_face_ids"])
        filled_checked_face_count += len(added)
        final_area = float(mesh.area_faces[final_ids].sum())
        added_area = float(mesh.area_faces[added].sum())
        assert abs(final_area-example["area_m2"]) < 1e-12
        assert abs(added_area-example["filled_area_m2"]) < 1e-12
        assert abs(final_area/mesh.area-example["area_fraction"]) < 1e-12
        summary = dict(summaries[index])
        final_depth = (surface[final_ids]-apex) @ axis
        summary.update(raw_area_cm2=summary["area_cm2"], area_cm2=final_area*1e4,
                       minimum_forward_depth_m=float(final_depth.min()),
                       maximum_forward_depth_m=float(final_depth.max()),
                       filled_area_cm2=added_area*1e4, filled_face_count=len(added),
                       fill_normal_limit_failure_count=len(failed_angle),
                       fill_visibility_failure_count=int((~visible).sum()),
                       fills_require_step2_verification=True,
                       final_component_count=1)
        examples.append(dict(id=example["id"], **summary))
    union = membership.any(axis=0)
    np.testing.assert_array_equal(np.flatnonzero(~union), report["candidate_pool_uncovered_face_ids"])
    pool_coverage = float(mesh.area_faces[union].sum()/mesh.area)
    assert abs(pool_coverage-report["candidate_pool_surface_coverage_fraction"]) < 1e-12
    selected_coverage = float(mesh.area_faces[chosen.any(axis=0)].sum()/mesh.area)
    assert abs(selected_coverage-report["selected_area_fraction"]) < 1e-12
    raw_chosen = membership[arrays["representative_candidate_indices"]]
    raw_coverage = float(mesh.area_faces[raw_chosen.any(axis=0)].sum()/mesh.area)
    assert abs(raw_coverage-report["geometrically_screened_selected_area_fraction"]) < 1e-12
    overlap = [[float(mesh.area_faces[a & b].sum()*1e4) for b in chosen] for a in chosen]
    assert not report["planning_performed"] and not report["full_covering_plan_selected"]
    assert not report["motion_and_support_verified"]

    assert len(report["visualization"]["scenes"]) == len(report["working_areas"])
    for scene, example in zip(report["visualization"]["scenes"], report["working_areas"]):
        target = example.get("display_target")
        if target:
            assert scene["aimed_site"] == target["site"]
            assert scene["contact_source_face_id"] == target["source_face_id"]
            np.testing.assert_allclose(scene["contact_surface_world_m"], target['point_world_m'], atol=1e-12)
        np.testing.assert_allclose(scene["spatial_cone_apex_world_m"], apex, atol=1e-12)
        np.testing.assert_allclose(scene["spatial_cone_axis_world"], axis, atol=1e-12)
        assert scene["spatial_cone_half_angle_deg"] == cone["half_angle_deg"]
        assert scene["spatial_cone_usable_half_angle_deg"] == cone["usable_half_angle_deg"]
        assert scene['spatial_cone_maximum_work_depth_m'] == cone['maximum_axial_distance_m']
        assert not scene['dashed_extension_selectable']
        assert scene["sphere_radius_m"] == workspace["usable_radius_m"]
        assert scene["tabletop_dimensions_m"] == [.5, .4, .024]
        np.testing.assert_allclose(scene["displayed_tool_direction_world"], axis, atol=1e-6)
        assert scene["surface_normal_half_angle_deg"] == budget["full_normal_cap_half_angle_deg"]
        assert not scene["work_ball_drawn"]
        assert scene["one_color_includes_selected_region_and_initialization_fills"]
        assert scene["tool_direction_fixed_for_entire_region"]
    pictures = sorted(str(p.relative_to(output)) for p in output.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    assert pictures == report["figures"] == ["working_areas_overview.png"]
    with Image.open(output/pictures[0]) as picture:
        picture.load()
        assert picture.size == (4096, 2560)
    binding = dict(input_npz_sha256=sha256(output/report["arrays"]),
                   report_sha256=sha256(report_path), validator_sha256=sha256(__file__))
    validation = dict(status="passed", model=workspace['type'],
                      all_candidates_edge_connected=True, candidate_count=len(membership),
                      source_face_count=len(mesh.faces), source_geometry_unchanged=True,
                      maxima=maxima, minimum_forward_depth_m=minimum_depth,
                      maximum_forward_depth_m=maximum_depth,
                      first_solid_ring_axial_distance_m=cone['reference_axial_distance_m'],
                      maximum_usable_forward_depth_m=cone['maximum_axial_distance_m'],
                      dashed_extension_selectable=False,
                      examples=examples, example_pairwise_overlap_cm2=overlap,
                      representative_first_layer_face_count=checked_face_count,
                      filled_first_layer_face_count=filled_checked_face_count,
                      needle_direction_fixed_for_all_candidates=True,
                      entire_selected_triangles_on_first_visible_layer=True,
                      occluders_outside_work_window_included=True,
                      surface_incidence_is_quality_only=True,
                      requested_aiming_sites_on_cone_axis_inside_first_ring_and_used_for_tool_contact=True,
                      all_fills_surrounded_by_strictly_more_than_270_degrees=True,
                      filled_faces_are_initialization_only=True,
                      geometrically_screened_selected_area_fraction=raw_coverage,
                      pool_surface_coverage_fraction=pool_coverage,
                      selected_surface_coverage_fraction=selected_coverage,
                      full_coverage_plan_verified=False, native_robot_motion_verified=False,
                      covering_plan_scope="separate_step1_2_output", **binding)
    write_json(output/"pose_coverage_validation.json", validation)
    write_json(output/"visualization_validation.json", dict(
        status="passed", one_text_free_composite=True,
        fixed_cone_matches_selection_nominal_boundary=True,
        first_solid_ring_is_axial_limit=True,
        first_solid_ring_distance_m=cone['reference_axial_distance_m'],
        maximum_work_depth_m=cone['maximum_axial_distance_m'],
        dashed_extension_selectable=False,
        needle_direction_identical_in_every_scene=True,
        work_ball_drawn=False, one_color_per_region=True,
        fixed_cone_apex_world_m=apex.tolist(), fixed_cone_axis_world=axis.tolist(),
        picture_sha256=sha256(output/pictures[0]), **binding))
    # These diagnostics described the superseded disconnected, orientation-only
    # model. Their conclusions must not accompany the newly generated arrays.
    for name in ("angular_constraint_diagnostics.json", "initialization_diagnostics.json",
                 "comfort_initialization_diagnostics.json", "expansion_analysis.json"):
        (output/name).unlink(missing_ok=True)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object", default="B")
    args = parser.parse_args()
    result = validate(args.output_root, args.object)
    if result.get('geometry_model') in ('fixed_length_needle_in_bounded_spherical_sector', 'needle_spherical_sector_surface_intersection'):
        print(json.dumps(result, indent=2))
        return
    print(json.dumps(dict(status=result["status"], candidates=result["candidate_count"],
                          maxima=result["maxima"], examples=result["examples"]), indent=2))


if __name__ == '__main__':
    import sys
    from .plan_reach import main as current_main
    sys.argv.insert(1, '--validate-only')
    current_main()
