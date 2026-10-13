"""Original ground demand points, transformed between fixed task poses.

Only target-world z is tested. There is no lateral bound, material region,
convex hull, object subtraction, or search for a footprint in this check.
"""
from dataclasses import dataclass

import numpy as np

TOL = 1e-9  # metres


def validate_frames(frames):
    frames = np.asarray(frames, float)
    if frames.ndim != 3 or frames.shape[1:] != (4, 4) or not len(frames):
        raise ValueError('Nonempty sequence of 4x4 task frames required')
    for frame in frames:
        rotation = frame[:3, :3]
        if (not np.isfinite(frame).all()
                or not np.allclose(frame[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
                or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-12, rtol=0)
                or abs(np.linalg.det(rotation)-1) > 1e-12):
            raise ValueError('Task frame must be a proper rigid transform')
    return frames


def map_points(points, source_frame, target_frame):
    """Source-world points follow the object into the target pose's world."""
    transform = np.asarray(target_frame)@np.linalg.inv(source_frame)
    return np.asarray(points)@transform[:3, :3].T+transform[:3, 3]


def ground_points(xy):
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('Nonempty finite Nx2 ground demand points required')
    return np.column_stack((xy, np.zeros(len(xy))))


@dataclass
class FloorPointResult:
    frames: np.ndarray
    rows: list
    heights: list
    masks: list


def check_floor_points(clouds_xy, frames, poses):
    frames = validate_frames(frames)
    if not (len(clouds_xy) == len(frames) == len(poses)) or len(set(poses)) != len(poses):
        raise ValueError('Matching clouds and distinct pose IDs required')
    rows, all_heights, masks = [], [], []
    for i, (pose, xy) in enumerate(zip(poses, clouds_xy)):
        points = ground_points(xy)
        mapped = [map_points(points, frames[i], frame) for frame in frames]
        heights = np.column_stack([p[:, 2] for p in mapped])
        if not np.isfinite(heights).all() or np.max(np.abs(heights[:, i])) > TOL:
            raise ValueError('Invalid height mapping to source floor')
        mask = np.all(heights >= -TOL, axis=1)
        pairs = []
        for j, other in enumerate(poses):
            if i == j:
                continue
            worst = int(np.argmin(heights[:, j]))
            pairs.append(dict(other_pose=other,
                violating_sample_count=int(np.count_nonzero(heights[:, j] < -TOL)),
                minimum_height_m=float(heights[worst, j]), worst_sample_index=worst,
                worst_point_source_world_m=points[worst].tolist(),
                worst_point_target_world_m=mapped[j][worst].tolist()))
        rows.append(dict(pose=pose, original_sample_count=len(points),
            violating_sample_count=int(np.count_nonzero(~mask)), passed=bool(mask.all()),
            per_other_pose=pairs))
        all_heights.append(heights)
        masks.append(mask)
    return FloorPointResult(frames, rows, all_heights, masks)


def validate_task_geometry(tasks, frames):
    """Bad task placements are input errors, not support infeasibility proofs."""
    first = tasks[0].domain.mesh
    object_vertices = map_points(first.vertices, frames[0], np.eye(4))
    for i, task in enumerate(tasks):
        vertices = np.asarray(task.domain.mesh.vertices)
        local = map_points(vertices, frames[i], np.eye(4))
        if (local.shape != object_vertices.shape
                or not np.allclose(local, object_vertices, atol=1e-10, rtol=0)
                or not np.array_equal(task.domain.mesh.faces, first.faces)):
            raise ValueError('Tasks must use the same object mesh and correspondence')
        if not np.isfinite(vertices).all() or vertices[:, 2].min() < -TOL:
            raise ValueError('Object itself penetrates a supplied pose floor')
        pivot = np.asarray(task.floor, float)
        if pivot.shape != (3,) or not np.isfinite(pivot).all() or abs(pivot[2]) > TOL:
            raise ValueError('Original object pivot must be on its own floor')
        for frame in frames:
            if map_points(pivot, frames[i], frame)[2] < -TOL:
                raise ValueError('Object pivot crosses another floor; invalid task geometry')
    return object_vertices, np.asarray(first.faces)


def pressure_centers(loads, origin):
    """Required ground wrenches about COM -> ground CoP in world coordinates."""
    loads = np.asarray(loads, float).reshape(-1, 6)
    normal = loads[:, 2]
    if not np.isfinite(loads).all() or np.any(normal <= 0):
        raise ValueError('Finite floor demand requires strictly positive normal reaction')
    moments = loads[:, 3:]+np.cross(origin, loads[:, :3])
    return np.c_[-moments[:, 1], moments[:, 0]]/normal[:, None], normal

