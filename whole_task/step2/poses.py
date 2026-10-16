"""Formalize the Step1.3 rigid alignments without changing the selected surface."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robots_config.constraints import load_config
from whole_task.step1.common import DEFAULT_OUTPUT, ROOT, load_shape, sha256
from whole_task.step1.pose_coverage import spatial_workspace_mask, face_components
from whole_task.step1.step1_3 import _adjacency, _visible_faces
from .transforms import rigid_transform, inverse, points, directions


def load_inputs(output_root=DEFAULT_OUTPUT, object_name="B"):
    if Path(object_name).name != object_name or object_name in ("", ".", ".."):
        raise ValueError("Expected one object directory name")
    stage = Path(output_root)/object_name/"step1"
    source_path = stage/"step1.3/pose_coverage.json"
    source = json.loads(source_path.read_text())
    config_path = stage/"step1.1/config.json"
    mesh_path = ROOT/"objects"/object_name/"mesh.stl"
    if sha256(config_path) != source["config_sha256"] or sha256(mesh_path) != source["shape_sha256"]:
        raise ValueError("The robot snapshot or shape changed; refresh Step1 first")
    selection = json.loads((stage/"step1.1/selection.json").read_text())
    original_config = Path(selection["source"])
    if original_config.is_file() and sha256(original_config) != selection["source_sha256"]:
        raise ValueError("The source robot configuration changed; refresh Step1 first")
    bound = {str(p.resolve()): sha256(p) for p in (source_path, config_path, mesh_path)}
    for field in ("input_sha256", "visualization_source_sha256"):
        for name, digest in source[field].items():
            path = ROOT/name
            if sha256(path) != digest:
                raise ValueError(f"Step1 input or source changed: {name}")
            bound[str(path.resolve())] = digest
    for name, digest in source["computation_source_sha256"].items():
        path = ROOT/"whole_task/step1"/name
        if sha256(path) != digest:
            raise ValueError(f"Step1 computation source changed: {name}")
        bound[str(path.resolve())] = digest
    for name, digest in source["artifact_sha256"].items():
        path = source_path.parent/name
        if sha256(path) != digest:
            raise ValueError(f"Step1 artifact changed: {name}")
        bound[str(path.resolve())] = digest
    with np.load(source_path.parent/source["arrays"]) as archive:
        arrays = {name: archive[name] for name in archive.files}
    mesh = load_shape(mesh_path)
    np.testing.assert_array_equal(mesh.faces, arrays["faces"])
    np.testing.assert_array_equal(mesh.vertices, arrays["vertices_object"])
    return mesh, source, arrays, load_config(config_path), bound


def fixed_station(source, config):
    axis = np.asarray(source["fixed_tool_direction_world"])
    rotation = np.asarray(source["fixed_tcp_rotation_world"])
    np.testing.assert_allclose(rotation[:, 2], axis, atol=1e-12)
    center = np.asarray(source["workspace"]["center_world_m"])
    gap = float(source["tcp_to_surface_distance_m"]["nominal"])
    station = rigid_transform(rotation, center)
    tcp = rigid_transform(rotation, center-gap*axis)
    tool = config["feasibility"]["tcp_transform"]
    flange_from_tcp = rigid_transform(tool["rotation_flange_tcp"], tool["translation_flange_m"])
    flange = tcp @ inverse(flange_from_tcp)
    base = config["feasibility"]["base_pose_world"]
    return dict(world_from_station=station.tolist(),
                world_from_reference_tcp=tcp.tolist(),
                world_from_reference_flange=flange.tolist(),
                world_from_robot_base=rigid_transform(base["rotation_matrix"], base["position_m"]).tolist(),
                reference_surface_point_world_m=center.tolist(),
                tool_to_surface_axis_world=axis.tolist(),
                reference_tcp_is_station_reference_not_a_complete_work_path=True,
                workspace=source["workspace"], angular_budget=source["angular_budget"],
                world_gravity_m_s2=[0., 0., -9.81])


def audit_region(mesh, transform, face_ids, source, config):
    """Recheck every selected triangle, including the hole-fill exceptions."""
    face_ids = np.asarray(face_ids, dtype=int)
    axis = np.asarray(source["fixed_tool_direction_world"])
    triangles = points(transform, mesh.triangles[face_ids])
    normals = directions(transform, mesh.face_normals[face_ids])
    cosine = np.clip(-normals @ axis, -1, 1)
    angles = np.degrees(np.arccos(cosine))
    angle_limit = source["angular_budget"]["working_normal_cap_half_angle_deg"]
    workspace = source["workspace"]
    reach = workspace["base_reach_estimate"]
    distance = source["tool_length_m"]+source["tcp_to_surface_distance_m"]["nominal"]
    flange = triangles-distance*axis
    outward_object = -axis @ transform[:3, :3]
    checks = dict(
        spatial_workspace=spatial_workspace_mask(triangles, workspace),
        local_normal=angles <= angle_limit+1e-7,
        sampled_visibility=_visible_faces(mesh, face_ids, np.tile(outward_object, (len(face_ids), 1))),
        flange_reach_estimate=np.linalg.norm(flange-reach["center_world_m"], axis=2).max(axis=1)
                              <= reach["usable_radius_m"]+1e-12)
    passed = np.logical_and.reduce(list(checks.values()))
    summary = dict(failed_source_face_ids_by_check={name: face_ids[~ok].tolist() for name, ok in checks.items()},
                   geometrically_passed_face_ids=face_ids[passed].tolist(),
                   unresolved_face_ids=face_ids[~passed].tolist(),
                   geometric_pass_area_m2=float(mesh.area_faces[face_ids[passed]].sum()),
                   unresolved_area_m2=float(mesh.area_faces[face_ids[~passed]].sum()),
                   max_selected_normal_angle_deg=float(angles.max()),
                   all_selected_faces_pass_geometry=bool(passed.all()),
                   component_count=len(face_components(face_ids, _adjacency(mesh))),
                   visibility_rays_per_face=4, visibility_is_sampled=True,
                   native_ik_verified=False, continuous_path_verified=False,
                   complete_tool_arm_fixture_collision_verified=False)
    return summary, passed


def formalize(mesh, source, source_arrays, config):
    station = fixed_station(source, config)
    world_from_station = np.asarray(station["world_from_station"])
    world_from_tcp = np.asarray(station["world_from_reference_tcp"])
    poses, transforms, accepted_masks = [], [], []
    for ordinal, area in enumerate(source["working_areas"]):
        alignment = area["screening_alignment"]
        transform = rigid_transform(alignment["rotation_world_object"], alignment["translation_world_object_m"])
        candidate = area["candidate_index"]
        np.testing.assert_allclose(transform[:3, :3], source_arrays["candidate_rotation_world_object"][candidate], atol=1e-12)
        np.testing.assert_allclose(transform[:3, 3], source_arrays["candidate_translation_world_object_m"][candidate], atol=1e-12)
        ids = np.asarray(area["face_ids"], dtype=int)
        np.testing.assert_array_equal(ids, np.flatnonzero(source_arrays["working_area_face_membership"][ordinal]))
        audit, passed = audit_region(mesh, transform, ids, source, config)
        raw = np.asarray(area["raw_face_ids"], dtype=int)
        if not np.isin(raw, ids[passed]).all():
            raise ValueError(f"{area['id']} lost a previously screened face")
        if not np.isin(audit["unresolved_face_ids"], area["filled_face_ids"]).all():
            raise ValueError("An unresolved surface is not an explicitly recorded fill")
        if audit["component_count"] != 1:
            raise ValueError("A working region became disconnected")
        accepted = np.zeros(len(mesh.faces), dtype=bool)
        accepted[ids[passed]] = True
        accepted_masks.append(accepted)
        transforms.append(transform)
        poses.append(dict(id=area["id"], source_candidate_index=candidate,
            status="initial_pose_with_fill_exceptions" if audit["unresolved_face_ids"] else "geometrically_screened_initial_pose",
            pose_source="step1_screening_alignment_formalized_without_reoptimization",
            world_from_object=transform.tolist(), object_from_world=inverse(transform).tolist(),
            station_from_object=(inverse(world_from_station) @ transform).tolist(),
            reference_tcp_from_object=(inverse(world_from_tcp) @ transform).tolist(),
            translation_world_object_m=transform[:3, 3].tolist(),
            quaternion_world_object_xyzw=Rotation.from_matrix(transform[:3, :3]).as_quat().tolist(),
            gravity_object_m_s2=(transform[:3, :3].T @ np.asarray(station["world_gravity_m_s2"])).tolist(),
            display_target=area.get("display_target"),
            working_face_ids=ids.tolist(), raw_face_ids=raw.tolist(), filled_face_ids=area["filled_face_ids"],
            selected_area_m2=area["area_m2"], selected_surface_fraction=area["area_fraction"],
            geometry_audit=audit,
            fixture_input=dict(object_pose_is_initial=True, working_face_ids_to_keep_accessible=ids.tolist(),
                               support_contacts_selected=False, fixture_verified=False),
            executable_working_pose_verified=False))
    transforms = np.asarray(transforms)
    accepted_masks = np.asarray(accepted_masks)
    data = dict(vertices_object=mesh.vertices, faces=mesh.faces, source_face_ids=np.arange(len(mesh.faces)),
                face_area_m2=mesh.area_faces,
                world_from_object=transforms,
                object_from_world=np.array([inverse(t) for t in transforms]),
                world_from_station=world_from_station,
                world_from_reference_tcp=world_from_tcp,
                station_from_object=np.array([inverse(world_from_station) @ t for t in transforms]),
                vertices_world=np.array([points(t, mesh.vertices) for t in transforms]),
                face_normals_world=np.array([directions(t, mesh.face_normals) for t in transforms]),
                working_area_face_membership=source_arrays["working_area_face_membership"],
                filled_face_membership=source_arrays["working_area_filled_face_membership"],
                geometric_pass_face_membership=accepted_masks)
    report = dict(schema_version=1, object=source["object"],
        robot_config_id=source["robot_config_id"],
        method="formalize_existing_geometric_alignments_in_fixed_station",
        units="m", world_frame="world_z_up", object_frame="unchanged_source_STL_coordinates",
        transform_convention="T_A_B maps column coordinates B to A; p_A=R_A_B p_B+t_A_B",
        numpy_point_convention="Row-stored points: points_A=points_B @ R_A_B.T+t_A_B",
        quaternion_order="xyzw", fixed_station=station, pose_count=len(poses), poses=poses,
        pose_export_complete=True, pose_reoptimized=False,
        regions_unchanged_from_step1=True,
        selected_surface_union_fraction=source["selected_area_fraction"],
        geometric_pass_surface_union_fraction=float(mesh.area_faces[accepted_masks.any(axis=0)].sum()/mesh.area),
        all_selected_faces_pass_geometry=all(p["geometry_audit"]["all_selected_faces_pass_geometry"] for p in poses),
        full_coverage_plan_selected=False, native_robot_motion_verified=False,
        fixture_designed=False, arrays="object_poses.npz")
    return report, data
