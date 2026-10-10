"""Construct two immutable B/pose1+pose2 fixtures for the reuse idea videos.

These are geometric illustrations, not a new force-optimization experiment.
Units are meters. Continuous nonconvex translation sweeps use the project's
existing geometry kernel; a relocation path is never carved into a fixture.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import trimesh

IDEA = Path(__file__).resolve().parents[1]
ROOT = IDEA.parents[1]
CO = ROOT / "slides/Co-optimize"
sys.path.insert(0, str(CO / "helper_func"))
from co_common import (D, G, I, S, material_volume, provenance, state, union,
                       wrap_offsets, contact_boundary)
from initial_directions import initialize_close_directions

POSES = ("pose_1", "pose_2")
THICKNESS = .005
SEAT_HEIGHT = .008
SWEEP_LENGTH = .5
CRADLE_HEIGHT = .06


def shell(mesh, work_ids):
    offsets = wrap_offsets(mesh, THICKNESS)
    cells = [S.solid(G.hull_mesh(G.head_cell(mesh, tri, i, offsets)))
             for i, tri in enumerate(mesh.triangles)]
    allowed = np.setdiff1d(np.arange(len(mesh.faces)), work_ids)
    forbidden = union([cells[i] for i in work_ids])
    return union([cells[i] for i in allowed]) - S.solid(mesh) - forbidden, forbidden


def carve(seed, sweeps):
    cut = union(sweeps)
    remaining = seed - cut
    # Resolve coincident Boolean boundaries in the actual stored geometry.
    for sweep in sweeps:
        if material_volume(remaining ^ sweep) > 1e-14:
            remaining = remaining - sweep
    components = remaining.decompose()
    degenerate = [part for part in components if material_volume(part) < 1e-15]
    remaining = union([part for part in components if material_volume(part) >= 1e-15])
    overlaps = [material_volume(remaining ^ sweep) for sweep in sweeps]
    if max(overlaps) > 1e-10:
        raise RuntimeError(f"Continuous exit overlap unresolved: {overlaps}")
    if remaining.is_empty():
        raise RuntimeError("The fixture became empty")
    return remaining, overlaps, [material_volume(part) for part in degenerate]


def export(solid, name):
    mesh = S.unpack(solid)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError(f"Invalid boundary: {name}")
    D.export_exact_obj(mesh, IDEA / "data" / name)
    return mesh


def main():
    began = time.monotonic()
    data = IDEA / "data"
    data.mkdir(parents=True, exist_ok=True)
    states = [state("B", p) for p in POSES]
    tasks = [row[0] for row in states]
    transforms = np.asarray([row[1] for row in states])
    mesh = states[0][2]
    for _, _, other in states[1:]:
        np.testing.assert_allclose(mesh.vertices, other.vertices, atol=1e-14)
    D.export_exact_obj(mesh, data / "object_local.obj")
    normals = transforms[:, 2, :3]
    directions, info = initialize_close_directions(normals)
    if info["common_direction_status"] != "strict_common_direction":
        raise RuntimeError("This illustration needs a common upward exit")
    direction = directions[0]
    work_ids = np.unique(np.concatenate([task.domain.work_ids for task in tasks]))
    print("BUILD pose-following wrap", flush=True)
    seed, _ = shell(mesh, work_ids)
    # One rigid shape is legal above the floor in both endpoint placements.
    for T in transforms:
        seed = seed.trim_by_plane(T[2, :3].tolist(), float(-T[2, 3] / S.SCALE))
    common_sweep = S.solid(S.swept_solid(mesh, direction * SWEEP_LENGTH))
    following, common_overlap, following_degenerate = carve(seed, [common_sweep])
    following_mesh = export(following, "pose_following_wrap.obj")
    if len(following.decompose()) != 1:
        raise RuntimeError("Pose-following wrap must be one connected fixture")

    print("BUILD fixed-seat reuse", flush=True)
    seated = transforms.copy()
    seated[:, 2, 3] += SEAT_HEIGHT
    bodies, local_shells, work_solids, vertical_sweeps = [], [], [], []
    for task, T in zip(tasks, seated):
        body = mesh.copy()
        body.apply_transform(T)
        bodies.append(body)
        wrap, work = shell(mesh, task.domain.work_ids)
        local_shells.append(wrap.transform(np.column_stack([T[:3, :3], T[:3, 3] / S.SCALE])))
        work_solids.append(work.transform(np.column_stack([T[:3, :3], T[:3, 3] / S.SCALE])))
        vertical_sweeps.append(S.solid(S.swept_solid(body, [0., 0., SWEEP_LENGTH])))
    bounds = np.vstack([S.unpack(part).vertices for part in local_shells])
    low, high = bounds.min(0), bounds.max(0)
    low[:2] -= .006
    high[:2] += .006
    # A common low tray provides a real seating plane and joins the shell.
    base = trimesh.creation.box([*(high[:2] - low[:2]), SEAT_HEIGHT])
    base.apply_translation([*((low[:2] + high[:2]) / 2), SEAT_HEIGHT / 2])
    fixed_seed = (union(local_shells) + S.solid(base)) - union(work_solids)
    fixed_seed = fixed_seed.trim_by_plane([0., 0., 1.], 0.)
    # Keep a low shared cradle. Upper-ear wrapping and the high detached cap
    # obscure the two object poses without being needed for this illustration.
    before_pruning, _, _ = carve(fixed_seed, vertical_sweeps)
    fixed_seed = fixed_seed.trim_by_plane([0., 0., -1.], -CRADLE_HEIGHT / S.SCALE)
    fixed, vertical_overlaps, fixed_degenerate = carve(fixed_seed, vertical_sweeps)
    connector_volume = 0.
    components = sorted(fixed.decompose(), key=material_volume, reverse=True)
    if len(components) > 1:
        # Extend detached seating regions down to the common tray. The upward
        # paths are subtracted again, so a connecting rib cannot obstruct them.
        ribs = []
        for part in components[1:]:
            piece = S.unpack(part)
            # Extrude the exact projected silhouette, avoiding near-tangent
            # face prisms in the already triangulated Boolean fragment.
            rib = part.project().extrude(float(piece.bounds[1, 2]) / S.SCALE)
            ribs.append(rib)
        grown = (fixed + union(ribs)) - union(work_solids)
        connected, vertical_overlaps, extra_degenerate = carve(grown, vertical_sweeps)
        connector_volume = material_volume(connected - fixed)
        fixed_degenerate.extend(extra_degenerate)
        fixed = connected
    fixed_mesh = export(fixed, "fixed_seat_reuse.obj")
    if len(fixed.decompose()) != 1:
        raise RuntimeError("Fixed-seat support must be one connected fixture")

    checks = []
    for k, (task, T, body) in enumerate(zip(tasks, transforms, bodies)):
        following_world = following_mesh.copy()
        following_world.apply_transform(T)
        native_body = task.domain.mesh
        exit_world = T[:3, :3] @ direction
        endpoint = native_body.copy()
        endpoint.apply_translation(exit_world * .25)
        final_vertical = body.copy()
        final_vertical.apply_translation([0., 0., .25])
        following_solid = S.solid(following_world)
        allowed = np.setdiff1d(np.arange(len(body.faces)), task.domain.work_ids)
        contacts, _ = contact_boundary(body, fixed_mesh, allowed)
        contact_area = float(trimesh.triangles.area(contacts).sum())
        if contact_area <= 1e-10:
            raise RuntimeError(f"The low shared cradle lost seating contact for {POSES[k]}")
        checks.append(dict(
            pose=POSES[k], following_fixture_world_transform=T.tolist(),
            fixed_object_world_transform=seated[k].tolist(),
            following_exit_world=exit_world.tolist(), fixed_exit_world=[0., 0., 1.],
            following_fixture_min_z_m=float(following_world.vertices[:, 2].min()),
            fixed_object_min_z_m=float(body.vertices[:, 2].min()),
            following_seated_overlap_m3=material_volume(following_solid ^ S.solid(native_body)),
            following_endpoint_overlap_m3=material_volume(following_solid ^ S.solid(endpoint)),
            fixed_seated_overlap_m3=material_volume(fixed ^ S.solid(body)),
            fixed_endpoint_overlap_m3=material_volume(fixed ^ S.solid(final_vertical)),
            fixed_actual_nonworking_contact_area_mm2=contact_area * 1e6,
        ))
    if any(row[metric] > 1e-10 for row in checks for metric in (
        "following_seated_overlap_m3", "following_endpoint_overlap_m3",
        "fixed_seated_overlap_m3", "fixed_endpoint_overlap_m3")):
        raise RuntimeError("Seated/exit endpoint overlap failed")
    if min(row["following_fixture_min_z_m"] for row in checks) < -1e-9:
        raise RuntimeError("An endpoint fixture crosses the floor")
    report = dict(
        complete=True, object="B", poses=list(POSES), units="meters",
        names=dict(pose_following_wrap="一对一复用 / One-to-one Reuse",
                   fixed_seat_reuse="一对多复用 / One-to-many Reuse"),
        demonstration_only=True, force_optimization_run=False,
        force_acceptance_run=False, strength_checked=False,
        wrap_thickness_m=THICKNESS, seat_height_m=SEAT_HEIGHT,
        continuous_exit_length_m=SWEEP_LENGTH,
        following_common_exit_fixture=direction.tolist(),
        following_direction_initialization=info,
        following_continuous_overlap_m3=common_overlap,
        fixed_continuous_overlap_m3=vertical_overlaps,
        discarded_degenerate_components_m3=dict(following=following_degenerate, fixed=fixed_degenerate),
        following_volume_cm3=material_volume(following) * 1e6,
        fixed_volume_cm3=material_volume(fixed) * 1e6,
        fixed_connecting_rib_volume_cm3=connector_volume * 1e6,
        fixed_cradle_maximum_height_m=CRADLE_HEIGHT,
        fixed_material_removed_for_visibility_cm3=material_volume(before_pruning - fixed) * 1e6,
        following_component_count=len(following.decompose()),
        fixed_component_count=len(fixed.decompose()),
        fixed_fixture_transform=np.eye(4).tolist(),
        replacement_policy="Only one object is present at a time; remove it before rotating the empty fixture",
        sweep_policy="Continuous nonconvex sweeps; no convex hull or sampled-frame cavity",
        relative_exit_is_common=True, absolute_exit_is_common=False,
        world_transforms=transforms.tolist(), seated_transforms=seated.tolist(),
        state_checks=checks,
        provenance=provenance([p for task in tasks for p in task.inputs],
                              [Path(__file__), Path(S.__file__).with_name("translation_sweep.py")]),
        seconds=time.monotonic() - began,
    )
    (data / "geometry.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("COMPLETE", {key: report[key] for key in (
        "following_volume_cm3", "fixed_volume_cm3", "following_component_count",
        "fixed_component_count", "seconds")}, flush=True)


if __name__ == "__main__":
    main()
