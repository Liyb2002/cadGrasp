"""Step 1.3: whole-object work candidates under the robot constraints.

The default uses pose_coverage.py: a fixed spatial cone intersected with a local
ball, a direct local-normal cap, and sampled visibility. Every candidate is
connected along source-mesh edges; candidates may overlap.
The original compact selector below is retained only as a historical baseline.

Temporary alignments are screening witnesses, not optimized working poses.
Step1.2 remains empty; final poses and robot motions belong to Step2.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robots_config.constraints import load_config
from .common import DEFAULT_OUTPUT, ROOT, load_shape, sha256, stage_directory, write_json
from .render import render_working_areas
from .work_disks import CIRCLE_SIDES, clip_surface_disk, project_surface


def angular_budget(config, inset_fraction=.05):
    if not np.isfinite(inset_fraction) or not 0 <= inset_fraction < 1:
        raise ValueError("Angular inset must lie in [0, 1)")
    task = config["task_constraint"]
    alpha = float(task["cone_half_angle_deg"])
    beta = float(task["surface_incidence_half_angle_deg"])
    scale = 1-inset_fraction
    return dict(
        inset_fraction=float(inset_fraction), inset_kind="angular_radius",
        tool_cone_half_angle_deg=alpha,
        incidence_half_angle_deg=beta,
        working_tool_cone_half_angle_deg=scale*alpha,
        working_incidence_half_angle_deg=scale*beta,
        full_normal_cap_half_angle_deg=alpha+beta,
        working_normal_cap_half_angle_deg=scale*(alpha+beta),
    )


def _adjacency(mesh):
    neighbors = [[] for _ in mesh.faces]
    for first, second in mesh.face_adjacency:
        neighbors[first].append(int(second))
        neighbors[second].append(int(first))
    return neighbors


def _component(seed, allowed, neighbors):
    if not allowed[seed]:
        return np.empty(0, dtype=int)
    visited = np.zeros(len(allowed), dtype=bool)
    stack = [int(seed)]
    visited[seed] = True
    while stack:
        current = stack.pop()
        for other in neighbors[current]:
            if allowed[other] and not visited[other]:
                visited[other] = True
                stack.append(other)
    return np.flatnonzero(visited)


def _seed_faces(mesh, available, limit=96):
    """Spread a bounded seed family across the remaining surface geometry."""
    ids = np.flatnonzero(available)
    if len(ids) <= limit:
        return ids
    centers = mesh.triangles_center[ids]
    chosen = []
    distance = np.full(len(ids), np.inf)
    index = int(np.argmax(mesh.area_faces[ids]))
    for _ in range(limit):
        chosen.append(int(ids[index]))
        distance = np.minimum(distance, np.sum((centers-centers[index])**2, axis=1))
        distance[np.isin(ids, chosen)] = -1
        index = int(np.argmax(distance))
    return np.array(chosen, dtype=int)


def approach_directions(normals, cap_axis, tool_half_angle_deg):
    """Outward surface-to-tool directions nearest to each face normal.

    They lie inside a cap about cap_axis. Their negatives are the tool-to-
    surface directions, once both are rotated by a future object pose.
    """
    axis = np.asarray(cap_axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    normals = np.asarray(normals, dtype=float)
    dots = np.clip(normals @ axis, -1, 1)
    separation = np.arccos(dots)
    angles = np.minimum(separation, np.radians(tool_half_angle_deg))
    tangent = normals-dots[:, None]*axis
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-15)
    result = np.cos(angles)[:, None]*axis+np.sin(angles)[:, None]*tangent
    return result/np.linalg.norm(result, axis=1, keepdims=True)


def _visible_faces(mesh, ids, approach):
    # Center and three interior points near corners. All are on the source face.
    triangles = mesh.triangles[ids]
    centers = mesh.triangles_center[ids]
    points = np.concatenate([centers[:, None, :],
                             .8*triangles+.2*centers[:, None, :]], axis=1)
    epsilon = float(np.linalg.norm(mesh.extents))*1e-6
    origins = points+epsilon*mesh.face_normals[ids, None, :]
    directions = np.repeat(approach[:, None, :], 4, axis=1)
    blocked = mesh.ray.intersects_any(origins.reshape(-1, 3),
                                      directions.reshape(-1, 3)).reshape(-1, 4)
    return ~np.any(blocked, axis=1)


def _clipped_visibility(mesh, geometry, approach):
    triangles = geometry["triangles"]
    centers = triangles.mean(axis=1)
    normals = mesh.face_normals[geometry["source_face_ids"]]
    points = np.concatenate([centers[:, None], .8*triangles+.2*centers[:, None]], axis=1)
    origins = points+np.linalg.norm(mesh.extents)*1e-6*normals[:, None]
    directions = np.repeat(approach[:, None], 4, axis=1)
    return not mesh.ray.intersects_any(origins.reshape(-1, 3), directions.reshape(-1, 3)).any()


def requested_region_mask(mesh, request):
    """Explicit site priors for this shape, applied to source face centers."""
    bounds = np.asarray(request["bounds_object_m"], dtype=float)
    normal_bounds = np.asarray(request["normal_bounds"], dtype=float)
    if (bounds.shape != (2, 3) or normal_bounds.shape != (2, 3)
            or not np.isfinite(bounds).all() or not np.isfinite(normal_bounds).all()
            or np.any(bounds[0] >= bounds[1]) or np.any(normal_bounds[0] > normal_bounds[1])):
        raise ValueError("Invalid anatomical site bounds")
    return (np.all((mesh.triangles_center >= bounds[0]-1e-12)
                   & (mesh.triangles_center <= bounds[1]+1e-12), axis=1)
            & np.all((mesh.face_normals >= normal_bounds[0]-1e-12)
                     & (mesh.face_normals <= normal_bounds[1]+1e-12), axis=1))


def choose_working_areas(mesh, config, count=4, inset_fraction=.05,
                         region_requests=None, minimum_circle_fill=1,
                         minimum_normal_separation_deg=0):
    if not isinstance(count, int) or count < 1:
        raise ValueError("Working area count must be a positive integer")
    if region_requests is not None and len(region_requests) != count:
        raise ValueError("The requested region list must match the area count")
    if not 0 < minimum_circle_fill <= 1:
        raise ValueError("Projected circle fill must lie in (0, 1]")
    budget = angular_budget(config, inset_fraction)
    task = config["task_constraint"]
    window = task.get("world_position_constraint")
    if window is None or window["type"] != "planar_disk":
        raise ValueError("Step 1.3 requires the supplied planar work window")
    world_normal = np.asarray(window["outward_normal_world"], dtype=float)
    world_center = np.asarray(window["center_world_m"], dtype=float)
    world_axis = np.asarray(task["working_direction_world"], dtype=float)
    full_radius = float(window["radius_m"])
    depth_tolerance = float(window["normal_tolerance_m"])
    gap = float(task["tcp_to_surface_distance_m"]["nominal"])
    tool_length = float(config["feasibility"]["tcp_transform"]["translation_flange_m"][2])
    neighbors = _adjacency(mesh)
    normals = mesh.face_normals
    labels = np.full(len(mesh.faces), -1, dtype=np.int32)
    approaches = np.zeros((len(mesh.faces), 3))
    threshold = np.cos(np.radians(budget["working_normal_cap_half_angle_deg"]))
    areas = []
    evaluated = 0
    for region in range(count):
        available = labels < 0
        request = None if region_requests is None else region_requests[region]
        if request is not None:
            available &= requested_region_mask(mesh, request)
        candidates = []
        for seed in _seed_faces(mesh, available):
            center, normal = mesh.triangles_center[seed], normals[seed]
            if any(np.degrees(np.arccos(np.clip(normal @ previous["window_normal_object"], -1, 1)))
                   < minimum_normal_separation_deg for previous in areas):
                continue
            rotation = Rotation.align_vectors([world_normal], [normal])[0].as_matrix()
            cap_axis = -(rotation.T @ world_axis)
            depth = (mesh.vertices-center) @ normal
            eligible = (available & (normals @ cap_axis >= threshold-1e-12)
                        & (np.max(np.abs(depth[mesh.faces]), axis=1) <= depth_tolerance+1e-12))
            ids = _component(seed, eligible, neighbors)
            if not len(ids):
                continue
            _, _, capacity = project_surface(mesh, ids, center, normal, full_radius,
                                             minimum_circle_fill)
            radius = (1-inset_fraction)*min(full_radius, capacity)
            if radius > 1e-8:
                candidates.append((radius, int(seed), ids, rotation, cap_axis))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        best = None
        for upper_radius, seed, ids, rotation, cap_axis in candidates:
            # These radii rank a finite family; no global-optimality claim.
            center, normal = mesh.triangles_center[seed], normals[seed]
            direction = approach_directions(normals[ids], cap_axis,
                                             budget["working_tool_cone_half_angle_deg"])
            visible = _visible_faces(mesh, ids, direction)
            evaluated += 1
            eligible = np.zeros(len(mesh.faces), dtype=bool)
            eligible[ids[visible]] = True
            final_ids = _component(seed, eligible, neighbors)
            if not len(final_ids):
                continue
            _, _, capacity = project_surface(mesh, final_ids, center, normal, full_radius,
                                             minimum_circle_fill)
            radius = (1-inset_fraction)*min(full_radius, capacity)
            if radius <= 1e-8 or (best is not None and radius <= best[0]):
                continue
            geometry = clip_surface_disk(mesh, final_ids, center, normal, radius,
                                          minimum_circle_fill)
            if geometry is None:
                # A fully inscribed envelope is a conservative fallback when
                # clipping a larger envelope would introduce a hole or split.
                _, _, inner = project_surface(mesh, final_ids, center, normal)
                radius = (1-inset_fraction)*min(full_radius, inner)
                if radius <= 1e-8 or (best is not None and radius <= best[0]):
                    continue
                geometry = clip_surface_disk(mesh, final_ids, center, normal, radius)
                if geometry is None:
                    continue
            child_directions = approach_directions(normals[geometry["source_face_ids"]], cap_axis,
                                                    budget["working_tool_cone_half_angle_deg"])
            if not _clipped_visibility(mesh, geometry, child_directions):
                continue
            child_area = np.linalg.norm(np.cross(geometry["triangles"][:, 1]-geometry["triangles"][:, 0],
                                                 geometry["triangles"][:, 2]-geometry["triangles"][:, 0]), axis=1)/2
            mean_normal = np.average(normals[geometry["source_face_ids"]], axis=0,
                                      weights=child_area)
            mean_normal /= np.linalg.norm(mean_normal)
            if any(np.degrees(np.arccos(np.clip(mean_normal @ previous["mean_surface_normal_object"], -1, 1)))
                   < minimum_normal_separation_deg for previous in areas):
                continue
            best = (radius, seed, rotation, cap_axis, capacity, geometry)
        if best is None:
            site = request["name"] if request else str(region+1)
            raise ValueError(f"No valid working region for {site}; selected {len(areas)} of {count}")
        radius, seed, rotation, axis, capacity, geometry = best
        ids = geometry["touched_source_faces"]
        direction = approach_directions(normals[ids], axis,
                                         budget["working_tool_cone_half_angle_deg"])
        labels[ids] = region
        approaches[ids] = direction
        vertices = geometry["triangles"].reshape(-1, 3)
        spread = np.degrees(np.arccos(np.clip(normals[ids] @ axis, -1, 1)))
        deviation = np.degrees(np.arccos(np.clip(direction @ axis, -1, 1)))
        incidence = np.degrees(np.arccos(np.clip(np.sum(direction*normals[ids], axis=1), -1, 1)))
        center = mesh.triangles_center[seed]
        translation = world_center-rotation @ center
        delta_world = vertices @ rotation.T+translation-world_center
        world_depth = delta_world @ world_normal
        world_radius = np.linalg.norm(delta_world-world_depth[:, None]*world_normal, axis=1)
        if (world_radius.max() > (1-inset_fraction)*full_radius+1e-9
                or np.abs(world_depth).max() > depth_tolerance+1e-9):
            raise ValueError("Clipped disk failed the supplied world window")
        child_direction = approach_directions(normals[geometry["source_face_ids"]], axis,
                                               budget["working_tool_cone_half_angle_deg"])
        surface_world = geometry["triangles"].mean(axis=1) @ rotation.T+translation
        tool_world = -child_direction @ rotation.T
        geometry["approach_directions_object"] = child_direction
        geometry["tcp_positions_world_m"] = surface_world-gap*tool_world
        geometry["flange_positions_world_m"] = surface_world-(gap+tool_length)*tool_world
        geometry["tool_directions_world"] = tool_world
        child_areas = np.linalg.norm(np.cross(geometry["triangles"][:, 1]-geometry["triangles"][:, 0],
                                              geometry["triangles"][:, 2]-geometry["triangles"][:, 0]), axis=1)/2
        mean_normal = np.average(normals[geometry["source_face_ids"]], axis=0, weights=child_areas)
        mean_normal /= np.linalg.norm(mean_normal)
        areas.append(dict(
            id=f"A{region+1:02d}", face_ids=ids.tolist(), face_count=len(ids),
            name=request["name"] if request else f"area_{region+1:02d}",
            label_en=request["label_en"] if request else f"Area {region+1}",
            label_zh=request.get("label_zh") if request else None,
            requested_site=request,
            area_m2=geometry["area_m2"], area_fraction=geometry["area_m2"]/float(mesh.area),
            seed_face_id=seed, normal_cap_axis_object=axis.tolist(),
            center_object_m=center.tolist(), window_normal_object=normals[seed].tolist(),
            mean_surface_normal_object=mean_normal.tolist(),
            centroid_object_m=np.average(geometry["triangles"].mean(axis=1), axis=0,
                                         weights=child_areas).tolist(),
            bounds_object_m=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            normal_spread_max_deg=float(spread.max()),
            tool_deviation_max_deg=float(deviation.max()),
            incidence_max_deg=float(incidence.max()),
            connected_components=1, sampled_outgoing_rays_clear=True,
            visibility_sample_count_per_face=4,
            footprint_type="circle_envelope_clipped_by_legal_curved_surface", footprint_polygon_sides=CIRCLE_SIDES,
            minimum_projected_circle_fill=minimum_circle_fill,
            projected_circle_fill=geometry["projected_circle_fill"],
            radius_m=radius, local_capacity_radius_m=min(full_radius, capacity),
            window_radius_m=full_radius, usable_window_radius_m=(1-inset_fraction)*full_radius,
            screening_max_window_radius_m=float(world_radius.max()),
            screening_max_window_depth_m=float(np.abs(world_depth).max()),
            projected_area_m2=geometry["projected_area_m2"],
            boundary_components=geometry["boundary_components"], euler_number=geometry["euler_number"],
            screening_alignment=dict(rotation_world_object=rotation.tolist(),
                                     translation_world_object_m=translation.tolist(),
                                     role="candidate_window_fit_only_not_an_optimized_pose"),
            tcp_to_surface_distance_m=gap, tool_length_m=tool_length,
            processing_center_world_m=world_center.tolist(),
            base_to_processing_center_distance_m=float(np.linalg.norm(
                world_center-np.asarray(config["feasibility"]["base_pose_world"]["position_m"]))),
            world_pose=None, pose_status="to_be_optimized",
            _geometry=geometry,
        ))
    return labels, approaches, areas, budget, evaluated


def select_compact_working_areas(output_root=DEFAULT_OUTPUT, object_name="B", count=4,
                                 inset_fraction=.05, shape_path=None):
    robot_output = Path(output_root) / object_name / "step1/step1.1"
    config_path = robot_output / "config.json"
    selection = json.loads((robot_output / "selection.json").read_text())
    if sha256(config_path) != selection["config_sha256"]:
        raise ValueError("Selected robot snapshot changed; rerun Step 1.1 first")
    source_config = Path(selection["source"])
    if source_config.is_file() and sha256(source_config) != selection["source_sha256"]:
        raise ValueError("Source robot configuration changed; rerun Step 1.1 first")
    config = load_config(config_path)
    shape_path = Path(shape_path) if shape_path else ROOT / "objects" / object_name / "mesh.stl"
    mesh = load_shape(shape_path)
    request_path = Path(__file__).parent / f"{object_name}_working_regions.json"
    requests = json.loads(request_path.read_text()) if request_path.is_file() else None
    if requests is not None and len(requests["regions"]) != count:
        raise ValueError(f"{object_name} currently requests exactly {len(requests['regions'])} named working areas")
    labels, approaches, raw_areas, budget, evaluated = choose_working_areas(
        mesh, config, count, inset_fraction,
        region_requests=None if requests is None else requests["regions"],
        minimum_circle_fill=1 if requests is None else requests["minimum_projected_circle_fill"],
        minimum_normal_separation_deg=0 if requests is None else requests["minimum_reference_normal_separation_deg"])
    geometries = [area["_geometry"] for area in raw_areas]
    areas = [{key: value for key, value in area.items() if not key.startswith("_")}
             for area in raw_areas]
    selected = np.concatenate([geometry["triangles"] for geometry in geometries])
    selected_labels = np.concatenate([np.full(len(g["triangles"]), i, dtype=np.int32)
                                      for i, g in enumerate(geometries)])
    outside = np.concatenate([mesh.triangles[labels < 0],
                              *[g["outside_triangles"] for g in geometries]])
    output = stage_directory(output_root, object_name, "step1.3")
    np.savez_compressed(output / "working_areas.npz",
                        vertices_object=mesh.vertices, faces=mesh.faces,
                        face_normals_object=mesh.face_normals,
                        source_face_ids=np.arange(len(mesh.faces)),
                        face_area_m2=mesh.area_faces, face_region_id=labels,
                        approach_directions_object=approaches,
                        selected_triangles_object=selected,
                        selected_triangle_region_id=selected_labels,
                        selected_triangle_source_face_id=np.concatenate([g["source_face_ids"] for g in geometries]),
                        selected_triangle_barycentric=np.concatenate([g["barycentric"] for g in geometries]),
                        selected_triangle_approach_object=np.concatenate([g["approach_directions_object"] for g in geometries]),
                        unselected_triangles_object=outside,
                        screening_tcp_positions_world_m=np.concatenate([g["tcp_positions_world_m"] for g in geometries]),
                        screening_flange_positions_world_m=np.concatenate([g["flange_positions_world_m"] for g in geometries]),
                        screening_tool_directions_world=np.concatenate([g["tool_directions_world"] for g in geometries]))
    report = dict(
        schema_version=3, stage="step1.3", object=object_name,
        robot_config_id=config["id"], robot_name=config["name"],
        config_snapshot="../step1.1/config.json", config_sha256=sha256(config_path),
        shape_source=str(shape_path.resolve()), shape_sha256=sha256(shape_path),
        shape_units="m", geometry_frame="original_object_coordinates",
        shape_input_only=True, historical_pose_or_work_area_inputs_used=False,
        requested_area_count=count, working_area_count=len(areas),
        working_areas=areas, angular_budget=budget,
        shape_area_m2=float(mesh.area), shape_face_count=len(mesh.faces),
        selected_area_fraction=sum(a["area_fraction"] for a in areas),
        unselected_area_fraction=1-sum(a["area_fraction"] for a in areas),
        selection_method="one compact local footprint per requested site, clipped by normal/depth/visibility constraints in a fixed world station; radius and angle inset 5%",
        region_request_source=None if requests is None else str(request_path.resolve()),
        region_request_sha256=None if requests is None else sha256(request_path),
        minimum_projected_circle_fill=1 if requests is None else requests["minimum_projected_circle_fill"],
        fixed_workstation=True,
        fixed_distance_definition="robot base to processing-window reference center; object COM is not fixed",
        reference_normal_pairwise_deg=np.degrees(np.arccos(np.clip(
            np.array([a["window_normal_object"] for a in areas])
            @ np.array([a["window_normal_object"] for a in areas]).T, -1, 1))).tolist(),
        mean_normal_pairwise_deg=np.degrees(np.arccos(np.clip(
            np.array([a["mean_surface_normal_object"] for a in areas])
            @ np.array([a["mean_surface_normal_object"] for a in areas]).T, -1, 1))).tolist(),
        candidate_seed_limit_per_round=96, visibility_candidates_evaluated=evaluated,
        work_window=config["task_constraint"]["world_position_constraint"],
        standoff=config["task_constraint"]["tcp_to_surface_distance_m"],
        spatial_inset_fraction=inset_fraction,
        computation_source_sha256={name: sha256(Path(__file__).parent / name)
                                    for name in ["step1_3.py", "work_disks.py", "render.py", "common.py"]},
        world_pose_selected=False, planning_performed=False,
        spatial_workspace_applied=True, work_window_is_user_estimate_not_ik_certificate=True,
        scope="Compact candidates at the requested sites fit the same fixed work window under saved screening alignments. Surface areas vary with curvature; optimized poses, IK, full tool collisions, continuous paths and support remain to be solved.",
        arrays="working_areas.npz", figures=["working_areas.png"] + [f"area_{i+1:02d}.png" for i in range(count)],
    )
    display = dict(triangles=np.concatenate([outside, selected]),
                   labels=np.concatenate([np.full(len(outside), -1, dtype=np.int32), selected_labels]),
                   boundaries=[g["boundary_segments"] for g in geometries])
    render_working_areas(mesh, labels, report, output, display=display)
    report["artifact_sha256"] = {name: sha256(output / name)
                                  for name in [report["arrays"], *report["figures"]]}
    write_json(output / "working_areas.json", report)
    return report


def select_working_areas(output_root=DEFAULT_OUTPUT, object_name="B", count=4,
                         inset_fraction=.05, shape_path=None):
    config_path = Path(output_root)/object_name/'step1/step1.1/config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config.get('working_tool', {}).get('model') in ('fixed_length_needle_in_bounded_spherical_sector', 'needle_spherical_sector_surface_intersection'):
            if count != 4 or shape_path is not None:
                raise ValueError('Rigid-needle display uses four examples of the selected object shape')
            from .run_needle import run
            selection = json.loads((config_path.parent/'selection.json').read_text())
            run(output_root, object_name, selection['source'], inset_fraction)
            return json.loads((config_path.parent.parent/'step1.3/pose_coverage.json').read_text())
    from .pose_coverage import select_pose_coverage
    return select_pose_coverage(output_root, object_name, count, inset_fraction, shape_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="B")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--areas", type=int, default=4)
    parser.add_argument("--inset", type=float, default=.05)
    args = parser.parse_args()
    report = select_working_areas(args.output_root, args.object, args.areas, args.inset)
    print(json.dumps(dict(robot=report["robot_config_id"],
                          areas=report["working_area_count"],
                          selected_area_fraction=report["selected_area_fraction"],
                          angular_budget=report["angular_budget"]), indent=2))


if __name__ == '__main__':
    # Public execution uses the current bounded-translation model.
    from .plan_reach import main as current_main
    current_main()
