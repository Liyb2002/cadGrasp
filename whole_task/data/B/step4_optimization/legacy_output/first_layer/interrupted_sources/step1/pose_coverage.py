"""Fixed-needle candidates and explicit initialization hole fills.

The local work ball bounds selection but is hidden in the cone-only drawing.
Heuristic fills are distinguished from geometrically screened target surfaces.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robots_config.constraints import load_config
from .common import ROOT, DEFAULT_OUTPUT, load_shape, sha256, stage_directory, write_json
from .compose_working_areas import compose_working_areas
from .step1_3 import angular_budget, _adjacency, _seed_faces, _visible_faces
from .fill_working_holes import fill_surrounded_surface
from .first_layer import first_layer_faces, incidence_quality

POLICY = Path(__file__).with_name("planning_workspace.json")


def sphere_workspace(config, inset_fraction=.05):
    policy = json.loads(POLICY.read_text())
    reach = float(config["robot_specification"]["reach_m"])
    window = config["task_constraint"]["world_position_constraint"]
    radius = float(window["radius_m"])
    task = config["task_constraint"]
    axis = np.asarray(task["working_direction_world"], dtype=float)
    axis /= np.linalg.norm(axis)
    distance = (float(config["feasibility"]["tcp_transform"]["translation_flange_m"][2])
                + float(task["tcp_to_surface_distance_m"]["nominal"]))
    if distance <= 0:
        raise ValueError("The fixed cone reference requires a positive tool length plus standoff")
    apex = np.asarray(window["center_world_m"])-distance*axis
    return dict(type=policy["type"], applies_to=policy["applies_to"],
                center_world_m=window["center_world_m"],
                radius_m=radius, usable_radius_m=(1-inset_fraction)*radius,
                inset_fraction=inset_fraction, ik_certified=False,
                provenance="fixed_cone_and_local_sphere_geometric_work_envelope_not_ik",
                legacy_planar_surface_disk_applied=False,
                configured_window_center_and_radius_applied=True,
                spatial_cone=dict(apex_world_m=apex.tolist(), axis_world=axis.tolist(),
                                  half_angle_deg=float(task["cone_half_angle_deg"]),
                                  usable_half_angle_deg=(1-inset_fraction)*float(task["cone_half_angle_deg"]),
                                  reference_axial_distance_m=distance,
                                  apex_is_fixed_across_candidates=True,
                                  apex_is_actual_tool_flange=False,
                                  provenance=policy["spatial_cone"]["apex_provenance"]),
                base_reach_estimate=dict(center_world_m=config["feasibility"]["base_pose_world"]["position_m"],
                                         radius_m=reach, usable_radius_m=(1-inset_fraction)*reach,
                                         applies_to="flange_positions_world", ik_certified=False))


def coverage_angular_budget(config, inset_fraction=.05, normal_half_angle_deg=None):
    policy = json.loads(POLICY.read_text())
    normal_angle = None if normal_half_angle_deg is None else float(normal_half_angle_deg)
    if normal_angle is not None and not 0 < normal_angle < 90:
        raise ValueError("Fixed-tool surface incidence must be between 0 and 90 degrees")
    budget = angular_budget(config, inset_fraction)
    budget["source_incidence_half_angle_deg"] = budget["incidence_half_angle_deg"]
    budget["incidence_half_angle_deg"] = normal_angle
    budget["working_incidence_half_angle_deg"] = None if normal_angle is None else normal_angle*(1-inset_fraction)
    budget["working_tool_cone_half_angle_deg"] = 0.0
    budget["tool_direction_mode"] = policy["tool_direction_mode"]
    budget["full_normal_cap_half_angle_deg"] = normal_angle
    budget["working_normal_cap_half_angle_deg"] = budget['working_incidence_half_angle_deg']
    budget["normal_cap_semantics"] = "inward_surface_normal_against_fixed_needle_axis"
    budget['normal_cap_source'] = 'disabled_first_visible_layer' if normal_angle is None else 'explicit_experiment_override'
    budget['normal_hard_limit_applied'] = normal_angle is not None
    budget['quality_score'] = 'area_weighted_cosine_no_hard_angle_cutoff'
    budget["incidence_override_provenance"] = policy["incidence_override_provenance"]
    return budget


def spatial_workspace_mask(triangles_world, workspace):
    """All triangle vertices must satisfy both convex position constraints."""
    cone = workspace["spatial_cone"]
    delta = np.asarray(triangles_world)-np.asarray(cone["apex_world_m"])
    axis = np.asarray(cone["axis_world"])
    depth = delta @ axis
    lateral = np.linalg.norm(delta-depth[..., None]*axis, axis=-1)
    inside_cone = ((depth >= -1e-12)
                   & (lateral <= depth*np.tan(np.radians(cone["usable_half_angle_deg"]))+1e-12))
    inside_ball = np.linalg.norm(np.asarray(triangles_world)-workspace["center_world_m"], axis=-1) <= workspace["usable_radius_m"]+1e-12
    return np.all(inside_cone & inside_ball, axis=-1)


def face_components(ids, neighbors):
    """Shared-edge components; point contact alone does not connect regions."""
    remaining = set(int(i) for i in ids)
    components = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack, component = [seed], []
        while stack:
            current = stack.pop()
            component.append(current)
            for other in neighbors[current]:
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
        components.append(np.array(sorted(component), dtype=int))
    return components


def evaluate_alignments(mesh, config, outward_axis_object, inset_fraction=.05,
                        visibility_cache=None, neighbors=None, normal_half_angle_deg=None,
                        center_object_m=None):
    """Screen patch-centered translations, keeping each legal component.

    Each first-layer component supplies a translation seed. All resulting
    spatially valid, visible components remain candidates, including small ones
    for later coverage planning. No arbitrary anatomical boundary is imposed.
    """
    budget = coverage_angular_budget(config, inset_fraction, normal_half_angle_deg)
    workspace = sphere_workspace(config, inset_fraction)
    axis = np.asarray(outward_axis_object, dtype=float)
    axis = axis/np.linalg.norm(axis)
    task = config["task_constraint"]
    world_axis = np.asarray(task["working_direction_world"])
    rotation = Rotation.align_vectors([-world_axis], [axis])[0].as_matrix()
    center_world = np.asarray(workspace["center_world_m"])
    base_reach = workspace["base_reach_estimate"]
    base = np.asarray(base_reach["center_world_m"])
    gap = float(task["tcp_to_surface_distance_m"]["nominal"])
    tool_length = float(config["feasibility"]["tcp_transform"]["translation_flange_m"][2])
    visible = first_layer_faces(mesh, axis, visibility_cache)
    limit = budget['working_normal_cap_half_angle_deg']
    if limit is not None:
        visible &= mesh.face_normals @ axis >= np.cos(np.radians(limit))-1e-12
    ids = np.flatnonzero(visible)
    if not len(ids):
        return []
    neighbors = _adjacency(mesh) if neighbors is None else neighbors
    centers = [np.average(mesh.triangles_center[component],
                          weights=mesh.area_faces[component], axis=0)
               for component in face_components(ids, neighbors)]
    if center_object_m is not None:
        center = np.asarray(center_object_m, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("The aiming point must be a finite object-space 3D point")
        centers = [center]
    # All points use precisely the same WORLD tool direction. Object-space
    # axes differ only because the object has a different screening alignment.
    outward = np.tile(axis, (len(ids), 1))
    candidates, seen = [], set()
    for center_object in centers:
        translation = center_world-rotation @ center_object
        world_triangles = mesh.triangles[ids] @ rotation.T+translation
        spatial = spatial_workspace_mask(world_triangles, workspace)
        directions = outward.copy()
        flange = world_triangles+(gap+tool_length)*(directions @ rotation.T)[:, None]
        reach = np.linalg.norm(flange-base, axis=2).max(axis=1) <= base_reach["usable_radius_m"]+1e-12
        valid = spatial & reach
        for component in face_components(ids[valid], neighbors):
            key = tuple(component)
            if key in seen:
                continue
            seen.add(key)
            local = np.searchsorted(ids, component)
            candidates.append(dict(face_ids=component, outward_tool_object=directions[local],
                              rotation_world_object=rotation, translation_world_object_m=translation,
                              reference_center_object_m=center_object,
                              reference_center_world_m=center_world,
                              normal_cap_axis_object=axis,
                              connected_component_count=1,
                              area_m2=float(mesh.area_faces[component].sum()),
                              incidence_quality=incidence_quality(mesh, component, axis)))
    return sorted(candidates, key=lambda candidate: -candidate['incidence_quality']['effective_projected_area_m2'])


def evaluate_alignment(mesh, config, outward_axis_object, inset_fraction=.05,
                       visibility_cache=None, normal_half_angle_deg=None):
    """Largest connected candidate for one orientation (convenience API)."""
    candidates = evaluate_alignments(mesh, config, outward_axis_object, inset_fraction,
                                     visibility_cache, normal_half_angle_deg=normal_half_angle_deg)
    return candidates[0] if candidates else None


def evaluate_target_alignment(mesh, config, target, inset_fraction=.05,
                              normal_half_angle_deg=None):
    """Aim the fixed cone at a surface landmark; do not crop by anatomy."""
    face = int(target["source_face_id"])
    if not 0 <= face < len(mesh.faces):
        raise ValueError("The aiming face is outside the source mesh")
    axis = np.asarray(target.get("outward_axis_object", mesh.face_normals[face]), dtype=float)
    axis = axis/np.linalg.norm(axis)
    point = mesh.triangles_center[face]
    candidates = evaluate_alignments(mesh, config, axis, inset_fraction,
        normal_half_angle_deg=normal_half_angle_deg, center_object_m=point)
    for candidate in candidates:
        if face in candidate["face_ids"]:
            candidate["display_target"] = dict(site=target["site"], source_face_id=face,
                point_object_m=point.tolist(), outward_axis_object=axis.tolist(),
                point_world_m=candidate["reference_center_world_m"].tolist(),
                role="aiming_point_only_no_anatomical_boundary_clipping")
            return candidate
    raise ValueError(f"The {target['site']} aiming face does not pass the geometric checks")


def build_candidates(mesh, config, inset_fraction=.05, direction_count=384,
                     normal_half_angle_deg=None):
    index = np.arange(direction_count)
    z = 1-2*(index+.5)/direction_count
    azimuth = index*np.pi*(3-np.sqrt(5))
    axes = np.column_stack([np.sqrt(1-z*z)*np.cos(azimuth),
                            np.sqrt(1-z*z)*np.sin(azimuth), z])
    seeds = _seed_faces(mesh, np.ones(len(mesh.faces), dtype=bool), limit=192)
    axes = np.concatenate([axes, mesh.face_normals[seeds]])
    cache, candidates, seen = {}, [], set()
    neighbors = _adjacency(mesh)
    for axis in axes:
        for candidate in evaluate_alignments(mesh, config, axis, inset_fraction, cache, neighbors,
                                             normal_half_angle_deg):
            mask = np.zeros(len(mesh.faces), dtype=bool)
            mask[candidate["face_ids"]] = True
            key = np.packbits(mask).tobytes()
            if key in seen:
                continue
            seen.add(key)
            candidate["mask"] = mask
            candidates.append(candidate)
    candidates.sort(key=lambda value: -value['incidence_quality']['effective_projected_area_m2'])
    return candidates


def expand_same_alignment(mesh, config, candidate, normal_half_angle_deg=60,
                          inset_fraction=.05):
    """Larger incidence limit at exactly the same pose, tool axis and station."""
    budget = coverage_angular_budget(config, inset_fraction, normal_half_angle_deg)
    workspace = sphere_workspace(config, inset_fraction)
    axis = candidate["normal_cap_axis_object"]
    rotation = candidate["rotation_world_object"]
    translation = candidate["translation_world_object_m"]
    ids = np.flatnonzero(mesh.face_normals @ axis >= np.cos(np.radians(budget["working_normal_cap_half_angle_deg"]))-1e-12)
    surface = mesh.triangles[ids] @ rotation.T+translation
    outward = np.tile(axis, (len(ids), 1))
    reach = workspace["base_reach_estimate"]
    distance = (float(config["feasibility"]["tcp_transform"]["translation_flange_m"][2])
                + float(config["task_constraint"]["tcp_to_surface_distance_m"]["nominal"]))
    flange = surface+distance*(outward @ rotation.T)[:, None]
    valid = (spatial_workspace_mask(surface, workspace)
             & (np.linalg.norm(flange-reach["center_world_m"], axis=2).max(axis=1) <= reach["usable_radius_m"]+1e-12)
             & _visible_faces(mesh, ids, outward))
    for component in face_components(ids[valid], _adjacency(mesh)):
        if np.isin(candidate["face_ids"], component).all():
            return component
    raise ValueError("The relaxed limit lost the original connected core")


def representative_indices(candidates, count):
    """Large examples in distinct directions; this is not the covering plan."""
    chosen = []
    for i, candidate in enumerate(candidates):
        axis = candidate["normal_cap_axis_object"]
        if any(axis @ candidates[j]["normal_cap_axis_object"] > np.cos(np.radians(35))+1e-12
               for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f"Only {len(chosen)} distinct work candidates were found")
    return chosen


def select_pose_coverage(output_root=DEFAULT_OUTPUT, object_name="B", count=4,
                         inset_fraction=.05, shape_path=None):
    if not isinstance(count, int) or count < 1:
        raise ValueError("Example count must be positive")
    robot_output = Path(output_root)/object_name/"step1/step1.1"
    config_path = robot_output/"config.json"
    selection = json.loads((robot_output/"selection.json").read_text())
    if sha256(config_path) != selection["config_sha256"]:
        raise ValueError("Selected robot snapshot changed; rerun Step1.1 first")
    source = Path(selection["source"])
    if source.is_file() and sha256(source) != selection["source_sha256"]:
        raise ValueError("Source robot configuration changed; rerun Step1.1 first")
    config = load_config(config_path)
    budget = coverage_angular_budget(config, inset_fraction)
    workspace = sphere_workspace(config, inset_fraction)
    shape_path = Path(shape_path) if shape_path else ROOT/"objects"/object_name/"mesh.stl"
    mesh = load_shape(shape_path)
    candidates = build_candidates(mesh, config, inset_fraction)
    if not candidates:
        raise ValueError("No geometrically compatible candidates in the estimated workspace")
    policy = json.loads(POLICY.read_text())
    # Reproduce the reference alignments, then replace explicitly aimed examples.
    seed_candidates = candidates
    overrides = policy.get("display_target_overrides", {})
    targets = {}
    if overrides.get("object") == object_name:
        if sha256(shape_path) != overrides["shape_sha256"]:
            raise ValueError("Surface aiming points belong to a different mesh revision")
        targets = {target["ordinal"]: target for target in overrides["targets"]}
    chosen, fills = [], []
    final_masks = np.zeros((count, len(mesh.faces)), dtype=bool)
    filled_masks = np.zeros_like(final_masks)
    fill_angles = np.zeros(final_masks.shape)
    for ordinal, seed_index in enumerate(representative_indices(seed_candidates, count)):
        candidate = (evaluate_target_alignment(mesh, config, targets[ordinal], inset_fraction)
                     if ordinal in targets else dict(seed_candidates[seed_index]))
        ids = candidate['face_ids']
        axis = candidate["normal_cap_axis_object"]
        candidate.update(face_ids=ids, outward_tool_object=np.tile(axis, (len(ids), 1)),
                         area_m2=float(mesh.area_faces[ids].sum()))
        candidate["mask"] = np.zeros(len(mesh.faces), dtype=bool)
        candidate["mask"][ids] = True
        rotation, translation = candidate["rotation_world_object"], candidate["translation_world_object_m"]
        triangles = mesh.triangles @ rotation.T+translation
        eligible = spatial_workspace_mask(triangles, workspace)
        distance = (float(config["feasibility"]["tcp_transform"]["translation_flange_m"][2])
                    + float(config["task_constraint"]["tcp_to_surface_distance_m"]["nominal"]))
        flange = triangles+distance*(axis @ rotation.T)
        reach = workspace["base_reach_estimate"]
        eligible &= np.linalg.norm(flange-reach["center_world_m"], axis=2).max(axis=1) <= reach["usable_radius_m"]+1e-12
        eligible &= first_layer_faces(mesh, axis)
        fill = fill_surrounded_surface(mesh, ids, axis, eligible,
            policy["hole_filling"]["surround_threshold_deg_strict"])
        added = fill["filled_face_ids"]
        angles = np.degrees(np.arccos(np.clip(mesh.face_normals[added] @ axis, -1, 1)))
        visible = _visible_faces(mesh, added, np.tile(axis, (len(added), 1))) if len(added) else np.empty(0, dtype=bool)
        fill["normal_limit_failed_face_ids"] = []
        fill["sampled_visibility_failed_face_ids"] = added[~visible].tolist()
        fills.append(fill)
        final_masks[ordinal, fill["face_ids"]] = True
        filled_masks[ordinal, added] = True
        fill_angles[ordinal, added] = fill["filled_enclosure_angles_deg"]
        chosen.append(len(candidates))
        candidates.append(candidate)
    masks = np.array([candidate["mask"] for candidate in candidates])
    selected_union = final_masks.any(axis=0)
    pool_union = masks.any(axis=0)
    output = stage_directory(output_root, object_name, "step1.3")
    np.savez_compressed(output/"pose_coverage.npz", vertices_object=mesh.vertices, faces=mesh.faces,
                        source_face_ids=np.arange(len(mesh.faces)), face_area_m2=mesh.area_faces,
                        candidate_face_membership=masks, representative_candidate_indices=chosen,
                        working_area_face_membership=final_masks,
                        working_area_filled_face_membership=filled_masks,
                        working_area_fill_enclosure_deg=fill_angles,
                        candidate_outward_axis_object=np.array([c["normal_cap_axis_object"] for c in candidates]),
                        candidate_rotation_world_object=np.array([c["rotation_world_object"] for c in candidates]),
                        candidate_translation_world_object_m=np.array([c["translation_world_object_m"] for c in candidates]),
                        candidate_tool_offsets=np.r_[0, np.cumsum([len(c["face_ids"]) for c in candidates])],
                        candidate_tool_face_ids=np.concatenate([c["face_ids"] for c in candidates]),
                        candidate_outward_tool_object=np.concatenate([c["outward_tool_object"] for c in candidates]))
    examples = []
    for i, j in enumerate(chosen):
        candidate = candidates[j]
        raw_ids = candidate["face_ids"]
        ids = fills[i]["face_ids"]
        added = fills[i]["filled_face_ids"]
        examples.append(dict(id=f"C{i+1:02d}", candidate_index=j, face_ids=ids.tolist(),
                             display_target=candidate.get("display_target"),
                             raw_face_ids=raw_ids.tolist(),
                             filled_face_ids=added.tolist(),
                             filled_enclosure_angles_deg=fills[i]["filled_enclosure_angles_deg"].tolist(),
                             raw_area_m2=candidate["area_m2"],
                             filled_area_m2=float(mesh.area_faces[added].sum()),
                             normal_limit_failed_filled_face_ids=fills[i]["normal_limit_failed_face_ids"],
                             sampled_visibility_failed_filled_face_ids=fills[i]["sampled_visibility_failed_face_ids"],
                             filled_faces_require_step2_verification=True,
                             face_count=len(ids), area_m2=float(mesh.area_faces[ids].sum()),
                             connected_component_count=1,
                             area_fraction=float(mesh.area_faces[ids].sum()/mesh.area),
                             normal_cap_axis_object=candidate["normal_cap_axis_object"].tolist(),
                             incidence_quality=incidence_quality(mesh, ids, candidate['normal_cap_axis_object']),
                             world_pose=None, pose_status="to_be_solved_in_step2",
                             screening_alignment=dict(rotation_world_object=candidate["rotation_world_object"].tolist(),
                               translation_world_object_m=candidate["translation_world_object_m"].tolist(),
                               role="temporary_geometric_witness_not_final_working_pose")))
    report = dict(schema_version=6, method="fixed_needle_first_visible_layer_with_enclosed_gap_completion",
                  object=object_name, robot_config_id=config["id"], config_sha256=sha256(config_path),
                  shape_sha256=sha256(shape_path), shape_face_count=len(mesh.faces),
                  total_surface_area_m2=float(mesh.area), working_area_count=len(chosen), working_areas=examples,
                  candidate_count=len(candidates), workspace=workspace, angular_budget=budget,
                  fixed_tool_direction_world=config["task_constraint"]["working_direction_world"],
                  fixed_tcp_rotation_world=Rotation.align_vectors(
                    [config["task_constraint"]["working_direction_world"]], [[0., 0., 1.]])[0].as_matrix().tolist(),
                  tool_roll_mode="fixed_reference_roll",
                  tool_orientation_changes_allowed=False,
                  hole_filling=dict(**policy["hole_filling"],
                    method="one_pass_projected_boundary_angular_union_at_face_centroids",
                    only_surface_connected_additions=True,
                    projection_inside_already_selected_surface_is_not_a_hole=True),
                  display_alignment_seed_normal_half_angle_deg=budget['full_normal_cap_half_angle_deg'],
                  geometrically_screened_selected_area_fraction=float(mesh.area_faces[masks[chosen].any(axis=0)].sum()/mesh.area),
                  selected_area_fraction=float(mesh.area_faces[selected_union].sum()/mesh.area),
                  candidate_pool_surface_coverage_fraction=float(mesh.area_faces[pool_union].sum()/mesh.area),
                  candidate_pool_uncovered_face_ids=np.flatnonzero(~pool_union).tolist(),
                  anatomical_bounds_used=False, circular_trimming_used=False, planar_surface_slab_used=False,
                  disconnected_surface_pieces_allowed=False, candidate_overlap_allowed=True,
                  spatial_cone_applied=True, local_normal_cap_applied=False,
                  local_normal_cap_applies_to="disabled_incidence_is_quality_only",
                  all_colored_faces_geometrically_verified=True,
                  first_layer_visibility=policy['visibility'],
                  region_count_target=None, region_count_is_hard_constraint=False,
                  visibility_sample_count_per_face=0, visibility_is_sampled=False,
                  tcp_to_surface_distance_m=config["task_constraint"]["tcp_to_surface_distance_m"],
                  tool_length_m=config["feasibility"]["tcp_transform"]["translation_flange_m"][2],
                  planning_performed=False, full_covering_plan_selected=False,
                  world_pose_selected=False, motion_and_support_verified=False,
                  scope="Representative large candidates, not a complete covering plan. Step1.2 covering selection and Step2 working-pose verification remain pending.",
                  input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [POLICY, Path(__file__).with_name("initialization_targets.json"), shape_path]
                                if p.is_relative_to(ROOT)},
                  computation_source_sha256={name: sha256(Path(__file__).parent/name)
                                            for name in ["pose_coverage.py", "first_layer.py", "fill_working_holes.py", "step1_3.py", "render.py", "common.py", "work_disks.py"]},
                  arrays="pose_coverage.npz", figures=[])
    print(f"Step1.3: {len(candidates)} connected candidates; representative areas "
          f"{[round(example['area_m2']*1e4, 2) for example in examples]} cm2; rendering", flush=True)
    report = compose_working_areas(config, report, output)
    report["artifact_sha256"] = {name: sha256(output/name) for name in [report["arrays"], *report["figures"]]}
    write_json(output/"pose_coverage.json", report)
    return report
