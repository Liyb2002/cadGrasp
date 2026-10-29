"""Continuous pose transformations without preselecting task poses."""
import numpy as np
from scipy.spatial.transform import Rotation

def spin_pose(reference, axis, pivot, angle_deg):
    """Rotate the actual object about a world-space line, including translation."""
    reference = np.asarray(reference, dtype=float)
    axis = np.asarray(axis, dtype=float)
    pivot = np.asarray(pivot, dtype=float)
    axis = axis/np.linalg.norm(axis)
    rotation = Rotation.from_rotvec(axis*np.radians(angle_deg)).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation @ reference[:3, :3]
    transform[:3, 3] = pivot+rotation @ (reference[:3, 3]-pivot)
    return transform


def orbit_bounds(points, axis, pivot, reference_point=(0., 0., 0.)):
    """Exact extrema over all axial angles, for a finite set of world points."""
    points = np.asarray(points)
    axis = np.asarray(axis, dtype=float); axis = axis/np.linalg.norm(axis)
    pivot = np.asarray(pivot)
    delta = points-pivot
    depth = delta @ axis
    radius = np.linalg.norm(delta-depth[..., None]*axis, axis=-1)
    centers = pivot+depth[..., None]*axis
    base = np.asarray(reference_point)
    offset = pivot-base
    base_lateral = np.linalg.norm(offset-(offset @ axis)*axis)
    farthest = np.sqrt(((centers-base) @ axis)**2+(base_lateral+radius)**2)
    minimum_z = centers[..., 2]-radius*np.sqrt(max(0., 1-axis[2]**2))
    return dict(maximum_distance_from_reference_m=float(farthest.max()),
                minimum_world_z_m=float(minimum_z.min()))

