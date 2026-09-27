"""Small mesh helpers for the B/pose2 loading illustration (metres)."""
import numpy as np
import trimesh


def tube(points, radius, closed=False):
    """Round rods with overlapping spherical joints; all dimensions in metres."""
    pieces = []
    for a, b in zip(points, np.roll(points, -1, axis=0) if closed else points[1:]):
        if np.linalg.norm(b-a) > 1e-8:
            pieces.append(trimesh.creation.cylinder(radius=radius, sections=12, segment=[a, b]))
    for p in points:
        joint = trimesh.creation.icosphere(subdivisions=1, radius=radius*1.03)
        joint.apply_translation(p)
        pieces.append(joint)
    return pieces


def resample(polygon, count):
    line = polygon.exterior
    return np.array([line.interpolate(i*line.length/count).coords[0] for i in range(count)])

