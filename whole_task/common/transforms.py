"""Rigid transforms T_A_B map column coordinates from frame B to frame A."""
import numpy as np


def rigid_transform(rotation, translation):
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Expected a 3x3 rotation and a 3-vector translation")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Rigid transform entries must be finite")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-10, rtol=0)):
        raise ValueError("Expected a proper rotation, without scaling or reflection")
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def inverse(transform):
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("Expected a homogeneous 4x4 rigid transform")
    rotation, translation = transform[:3, :3], transform[:3, 3]
    rigid_transform(rotation, translation)
    return rigid_transform(rotation.T, -rotation.T @ translation)


def points(transform, coordinates):
    """Transform row-stored points of shape (..., 3), in meters."""
    return np.asarray(coordinates) @ transform[:3, :3].T+transform[:3, 3]


def directions(transform, vectors):
    """Rotate row-stored directions or normals; translation never applies."""
    return np.asarray(vectors) @ transform[:3, :3].T
