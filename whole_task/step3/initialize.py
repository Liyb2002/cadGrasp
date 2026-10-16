"""Co-optimize Step3 registration, fitted wrap and complete demand diagnostics."""
import json
from pathlib import Path
import numpy as np
import trimesh

from whole_task.step1.common import ROOT, load_shape, write_json
from whole_task.step2.loads_adapter import load_task
from whole_task.step2.transforms import inverse, points
from whole_task.step2.wrenches import endpoint_demands
from .legacy import kernel as K, unpack_solid
from .work_access import RegisteredWorkVolumes, MODEL, ENDPOINT_SOURCE


def validate_policy(policy):
    implemented = dict(work_access_model=MODEL, head_endpoint_source=ENDPOINT_SOURCE,
        segment_parameter_range=[0., 1.],
        contact_model="frictionless_nonnegative_unbounded_normal_reactions",
        equilibrium_model="legacy_shared_7d_no_uplift", force_diagnostics_gate_initialization=False,
        select_connected_component=False, exit_paths_included=False, construct_base=False,
        finite_tool_radius_included=False)
    for key, value in implemented.items():
        if policy[key] != value:
            raise ValueError(f"Unsupported Step3 policy {key}: {policy[key]}")
    for key in ("wrap_maximum_vertex_displacement_m", "boolean_volume_tolerance_m3", "floor_contact_tolerance_m"):
        if not np.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError(f"Expected positive finite {key}")
    if not np.isfinite(policy["floor_height_world_m"]):
        raise ValueError("Expected a finite world floor height")


def register(output_root, object_name, pose_ids=None):
    folder = Path(output_root)/object_name/"step2/step2.2"
    source = json.loads((folder/"loads.json").read_text())
    available = [row["id"] for row in source["poses"]]
    pose_ids = available if pose_ids is None else list(pose_ids)
    if not pose_ids or len(set(pose_ids)) != len(pose_ids) or not set(pose_ids) <= set(available):
        raise ValueError("Expected unique current Step2 pose IDs")
    mesh = load_shape(ROOT/"objects"/object_name/"mesh.stl")
    states, rows = {}, []
    reference_world = None
    for pose in pose_ids:
        task = load_task(pose, output_root, object_name)
        transform = np.asarray(task.domain.data["frame"]["T_world_mesh"])
        np.testing.assert_array_equal(mesh.faces, task.domain.mesh.faces)
        np.testing.assert_allclose(points(transform, mesh.vertices), task.domain.mesh.vertices, atol=1e-12, rtol=0)
        reference_world = transform.copy() if reference_world is None else reference_world
        alignment = reference_world @ inverse(transform)
        aligned = points(alignment, task.domain.mesh.vertices)
        target = points(reference_world, mesh.vertices)
        np.testing.assert_allclose(aligned, target, atol=1e-12, rtol=0)
        states[pose] = (task, transform, mesh)
        rows.append(dict(pose=pose, T_fixture_to_world=transform.tolist(),
            T_world_to_fixture=inverse(transform).tolist(), object_to_fixture=np.eye(4).tolist(),
            T_native_world_to_reference_world=alignment.tolist(),
            alignment_max_error_m=float(np.max(np.abs(aligned-target))),
            working_face_ids=task.domain.work_ids.tolist(), requires_geometry_review=task.requires_geometry_review))
    return mesh, states, dict(reference_pose=pose_ids[0], states=rows,
                             T_fixture_to_reference_world=reference_world.tolist()), source


def build_seed(mesh, states, policy, source_cone):
    """Same cell wrap/subtraction/contact algorithm as Co-optimize step31."""
    validate_policy(policy)
    thickness = float(policy["wrap_maximum_vertex_displacement_m"])
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("Positive finite wrap thickness required")
    tolerance = float(policy["boolean_volume_tolerance_m3"])
    access = RegisteredWorkVolumes(mesh, states, source_cone)
    allowed = np.setdiff1d(np.arange(len(mesh.faces)), access.ids)
    offsets = K.wrap_offsets(mesh, thickness)
    print(f"Step3: wrapping {len(mesh.faces)} source faces; joining current working faces to the head-side endpoint", flush=True)
    cells = [K.S.solid(K.G.hull_mesh(K.G.head_cell(mesh, triangle, face, offsets)))
             for face, triangle in enumerate(mesh.triangles)]
    body = K.S.solid(mesh)
    seed = K.union(cells)-body
    bounds = np.array([mesh.bounds[0]-thickness, mesh.bounds[1]+thickness])
    print("Step3: subtracting finite surface-to-head volumes at each installation", flush=True)
    forbidden = access.combined_solid()
    remaining = seed-forbidden
    construction_overlap = K.material_volume(remaining ^ forbidden)
    if max(construction_overlap, K.material_volume(remaining ^ body)) > tolerance:
        raise RuntimeError("Constructed wrap intersects work access or object")
    for conditioning in range(4):
        wrapped = unpack_solid(remaining)
        reloaded = K.S.solid(wrapped)
        work_overlap = K.material_volume(reloaded ^ forbidden)
        object_overlap = K.material_volume(reloaded ^ body)
        if max(work_overlap, object_overlap) <= tolerance:
            break
        if conditioning == 3:
            raise RuntimeError("Float64 exported wrap boundary unresolved")
        remaining = reloaded-forbidden-body
    if len(wrapped.faces) and (not wrapped.is_watertight or not wrapped.is_winding_consistent):
        raise RuntimeError("Work-carved wrap is not a closed oriented boundary")
    return wrapped, allowed, access, dict(
        wrap_maximum_vertex_displacement_m=thickness, uncut_material_volume_cm3=K.material_volume(seed)*1e6,
        material_volume_cm3=K.material_volume(reloaded)*1e6,
        work_cut_material_volume_cm3=(K.material_volume(seed)-K.material_volume(reloaded))*1e6,
        continuous_work_overlap_m3=work_overlap, object_overlap_m3=object_overlap,
        construction_work_overlap_m3=construction_overlap, boolean_conditioning_rounds=conditioning,
        finite_work_segment_count=sum(len(f.cells) for f in access.families),
        initial_wrap_bounds_material_m=bounds.tolist(),
        boolean_volume_tolerance_m3=tolerance,
        boundary_component_count=len(reloaded.decompose()),
        positive_boundary_component_count=sum(part.volume() > 0 for part in reloaded.decompose()),
        component_selection_performed=False)


def supply(task, transform, triangles, source_faces, policy):
    """Legacy seven-dimensional columns, with only actual object-floor contact."""
    contact_points = points(transform, triangles.reshape(-1, 3))
    normals = np.repeat(-task.domain.mesh.face_normals[source_faces], 3, axis=0)
    raw = np.c_[normals, np.cross(contact_points-task.domain.com, normals)]
    floor_z = float(policy["floor_height_world_m"])
    tol = float(policy["floor_contact_tolerance_m"])
    heights = task.domain.mesh.vertices[:, 2]-floor_z
    penetrates = bool(heights.min() < -tol)
    floor_points = task.domain.mesh.vertices[np.abs(heights) <= tol] if not penetrates else np.empty((0, 3))
    floor_raw = np.vstack([K.FLOOR.columns(point, task.domain.com) for point in floor_points]) if len(floor_points) else np.empty((0, 6))
    # U.floor(empty) retains only the shared no-uplift slack; it does not
    # invent a ground reaction underneath an airborne workpiece.
    full = K.I.merge_columns(K.U.floor(floor_raw, task.scale), K.U.heads(raw, task.scale))
    return full, dict(minimum_object_world_z_m=float(task.domain.mesh.bounds[0, 2]),
        floor_height_world_m=floor_z, object_floor_penetration=penetrates,
        workpiece_floor_contact_points_m=floor_points.tolist(),
        workpiece_floor_reaction_count=len(floor_raw), no_uplift_slack_retained=True,
        floor_model=K.FLOOR.description() if len(floor_points) else "airborne: no workpiece floor reactions")


def force_diagnostics(states, triangles, sources, policy, out):
    rows = []
    for pose, (task, transform, _) in states.items():
        full, floor = supply(task, transform, triangles, sources, policy)
        endpoints = endpoint_demands(task.domain)["endpoint_need_wrench"]*task.scale
        targets = np.vstack((task.targets, endpoints))
        error = None
        try:
            mask, info = K.J.classify(full, targets)
            status = "pass" if mask.all() else "fail"
        except (RuntimeError, np.linalg.LinAlgError, AssertionError) as exception:
            # No numerical failure is labelled as physical infeasibility.
            mask = np.zeros(len(targets), dtype=bool)
            info, error, status = {}, f"{type(exception).__name__}: {exception}", "unresolved"
        count = len(task.targets)
        folder = out/pose
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/"coverage.npz", mask=mask[:count], endpoint_mask=mask[count:],
                            supply_7d=full, endpoint_targets_7d=K.U.target(endpoints))
        row = dict(pose=pose, status=status, original_sample_count=count, covered_samples=int(mask[:count].sum()),
            endpoint_count=len(endpoints), covered_endpoints=int(mask[count:].sum()),
            continuous_contact_equilibrium_passed=bool(mask[count:].all()) if error is None else None,
            source_geometry_review_sample_count=int((~task.geometry_screened).sum()),
            reaction_generator_count=len(full), floor=floor, classifier=info, error=error,
            full_fixture_accepted=False, diagnostic_is_initialization_gate=False,
            scope="all retained initial contact components; no connected-body, base or exit certificate")
        write_json(folder/"force.json", row)
        rows.append(row)
        print(f"Step3.1 {pose}: {status}; samples {row['covered_samples']}/{count}, "
              f"continuous endpoints {row['covered_endpoints']}/{len(endpoints)}", flush=True)
    return rows
