"""Estimated surface work windows, tool standoff and task direction cones.

All public directions are expressed in the fixed, right-handed, Z-up world.
The working direction points from the tool towards the surface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected a finite three-dimensional vector")
    length = np.linalg.norm(vector)
    if length <= 1e-12:
        raise ValueError("Direction cannot be zero")
    return vector / length


def angle_deg(first, second):
    first, second = unit(first), unit(second)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(first, second)),
                                     np.clip(first @ second, -1, 1))))


def rotate_direction(direction_object, rotation_world_object):
    """Apply the saved task rotation, without translation, to an object direction."""
    rotation = np.asarray(rotation_world_object, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)):
        raise ValueError("Expected a proper 3x3 rotation R_world_object")
    return rotation @ unit(direction_object)


def validate_config(config):
    if config["schema_version"] != 1:
        raise ValueError("Unsupported schema version")
    task = config["task_constraint"]
    axis = np.asarray(task["working_direction_world"], dtype=float)
    unit(axis)
    if not np.isclose(np.linalg.norm(axis), 1, atol=1e-8, rtol=0):
        raise ValueError("working_direction_world must be unit length")
    if task["frame"] != "world_z_up" or task["direction_semantics"] != "tool_to_surface":
        raise ValueError("Expected fixed world tool-to-surface directions")
    for key in ("cone_half_angle_deg", "surface_incidence_half_angle_deg"):
        value = task[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value < 90:
            raise ValueError(f"{key} must lie in [0, 90)")
    if task["provenance"] != "user_editable_estimate_not_ik_certified":
        raise ValueError("Task estimates must be explicitly labeled")
    spec = config["robot_specification"]
    count = spec["dof"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("dof must be a positive integer")
    if not np.isfinite(spec["reach_m"]) or spec["reach_m"] <= 0:
        raise ValueError("reach_m must be positive")
    joints = spec["joint_position_limits"]
    if joints is not None:
        limits = np.asarray(joints["ranges"], dtype=float)
        if (joints["unit"] not in ("deg", "rad") or limits.shape != (count, 2)
                or not np.isfinite(limits).all() or not (limits[:, 0] < limits[:, 1]).all()):
            raise ValueError("Expected one ordered interval per joint, in deg or rad")
    if not config["sources"]:
        raise ValueError("Robot specifications must have sources")
    # Older direction-only configurations remain readable.
    patch = task.get("world_position_constraint")
    if patch is not None:
        if patch["type"] != "planar_disk" or patch["applies_to"] != "surface_points_world":
            raise ValueError("Expected a planar surface work window")
        center = point3(patch["center_world_m"])
        normal = point3(patch["outward_normal_world"])
        if not np.isclose(np.linalg.norm(normal), 1, atol=1e-8, rtol=0):
            raise ValueError("Surface normal must be unit length")
        for key in ("radius_m", "normal_tolerance_m"):
            if not np.isfinite(patch[key]) or patch[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if patch["provenance"] != "estimated_work_window_not_ik_certified":
            raise ValueError("Work window must be labeled as an estimate")
        base = config["feasibility"]["base_pose_world"]
        base_position = point3(base["position_m"])
        rotate_direction([1, 0, 0], base["rotation_matrix"])
        if not np.isclose(np.linalg.norm(center - base_position),
                          patch["base_to_center_distance_m"], atol=1e-8, rtol=0):
            raise ValueError("Base-to-surface distance disagrees with station geometry")
        tcp = config["feasibility"]["tcp_transform"]
        point3(tcp["translation_flange_m"])
        rotate_direction([1, 0, 0], tcp["rotation_flange_tcp"])
        # The simple drawing assumes a straight tool on flange +Z.
        if (not np.allclose(tcp["translation_flange_m"][:2], 0)
                or tcp["translation_flange_m"][2] <= 0
                or not np.allclose(tcp["rotation_flange_tcp"], np.eye(3))):
            raise ValueError("Estimated tool must extend along flange +Z")
        gap = task["tcp_to_surface_distance_m"]
        values = np.array([gap[key] for key in ("min", "nominal", "max")], dtype=float)
        if (not np.isfinite(values).all() or values[0] < 0
                or not values[0] <= values[1] <= values[2]):
            raise ValueError("Expected 0 <= minimum <= nominal <= maximum standoff")
        if gap["measurement"] != "along_tool_axis_from_process_tcp_to_surface":
            raise ValueError("Standoff must be measured from the process TCP along the tool axis")
    return config


def point3(point):
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Expected a finite three-dimensional point")
    return point


def nominal_tool_positions(config):
    """Nominal TCP and flange positions; geometry only, without an IK solution."""
    task = config["task_constraint"]
    center = point3(task["world_position_constraint"]["center_world_m"])
    axis = unit(task["working_direction_world"])
    gap = task["tcp_to_surface_distance_m"]["nominal"]
    length = config["feasibility"]["tcp_transform"]["translation_flange_m"][2]
    tcp = center - gap * axis
    return {"tcp_world_m": tcp, "flange_world_m": tcp - length * axis}


def surface_point_allowed(config, point_world_m):
    """Check the assumed surface disk, not the robot's actual reachable area."""
    patch = config["task_constraint"]["world_position_constraint"]
    delta = point3(point_world_m) - point3(patch["center_world_m"])
    normal = unit(patch["outward_normal_world"])
    depth = float(delta @ normal)
    planar = delta - depth * normal
    return (abs(depth) <= patch["normal_tolerance_m"] + 1e-9
            and np.linalg.norm(planar) <= patch["radius_m"] + 1e-9)


def standoff_allowed(config, distance_m):
    """Check a supplied TCP-to-surface axial distance in metres."""
    gap = config["task_constraint"]["tcp_to_surface_distance_m"]
    return bool(np.isfinite(distance_m) and distance_m >= 0
                and gap["min"] - 1e-9 <= distance_m <= gap["max"] + 1e-9)


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def load_configs(directory=HERE):
    configs = [load_config(path) for path in sorted(Path(directory).glob("[0-9][0-9]_*.json"))]
    ids = [config["id"] for config in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate configuration ids")
    return configs


def direction_allowed(config, tool_direction_world):
    """Only the world direction constraint; this is not a reachability test."""
    task = config["task_constraint"]
    return angle_deg(tool_direction_world, task["working_direction_world"]) <= task["cone_half_angle_deg"] + 1e-8


def surface_normal_compatible(config, outward_normal_world):
    """Whether SOME tool direction satisfies both direction and incidence cones.

    Spherical caps intersect iff the separation of their axes <= alpha + beta.
    This does not require every process direction in the incidence cone to fit.
    """
    task = config["task_constraint"]
    separation = angle_deg(-unit(outward_normal_world), task["working_direction_world"])
    return separation <= task["cone_half_angle_deg"] + task["surface_incidence_half_angle_deg"] + 1e-8


def tool_direction_allowed_at_surface(config, tool_direction_world, outward_normal_world):
    """Check a specified tool direction against both cones."""
    task = config["task_constraint"]
    incidence = angle_deg(tool_direction_world, -unit(outward_normal_world))
    return direction_allowed(config, tool_direction_world) and incidence <= task["surface_incidence_half_angle_deg"] + 1e-8


def process_cone_fully_covered(config, inward_axis_world, process_half_angle_deg):
    """Whether EVERY direction of a prescribed process cone fits the robot cone.

    Supply the original demand half angle explicitly; do not silently replace it
    with the estimated incidence angle in this configuration.
    """
    if not np.isfinite(process_half_angle_deg) or not 0 <= process_half_angle_deg <= 180:
        raise ValueError("Process half angle must lie in [0, 180]")
    task = config["task_constraint"]
    separation = angle_deg(inward_axis_world, task["working_direction_world"])
    return separation + process_half_angle_deg <= task["cone_half_angle_deg"] + 1e-8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    args = parser.parse_args()
    configs = [load_config(args.config)] if args.config else load_configs()
    if not configs:
        parser.error("No robot configurations found")
    for config in configs:
        task = config["task_constraint"]
        print(f'{config["id"]}: {task["working_direction_world"]}, half-angle={task["cone_half_angle_deg"]} deg')
    print(f"Validated {len(configs)} configurations; work windows and standoff are estimates, IK and collisions are not evaluated.")
