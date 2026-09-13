"""One-time inverse transport from reflected Y-up artifacts to native Z-up.

World positions use (x, y, z) = (old x, old z, old y). This permutation
has determinant -1: polar vectors use P, axial vectors use -P, rotations
use P R P, and mesh faces reverse their winding. All maps are involutions.
"""
import numpy as np

SWAP = np.array([0, 2, 1])
P = np.eye(3)[SWAP]
H = np.eye(4)[[0, 2, 1, 3]]
WRENCH_ORDER = np.array([0, 2, 1, 3, 5, 4])
WRENCH_SIGNS = np.array([1., 1., 1., -1., -1., -1.])


def polar(value):
    a = np.asarray(value)
    if a.shape[-1:] != (3,):
        raise ValueError('A polar vector must end in three coordinates')
    return a[..., SWAP].copy()


def axial(value):
    return -polar(value)


def rotation(value):
    a = np.asarray(value)
    if a.shape[-2:] != (3, 3):
        raise ValueError('A rotation must end in a three-by-three matrix')
    return a[..., SWAP, :][..., :, SWAP].copy()


def transform(value):
    a = np.asarray(value)
    if a.shape[-2:] != (4, 4):
        raise ValueError('A rigid transform must end in a four-by-four matrix')
    indices = [0, 2, 1, 3]
    return a[..., indices, :][..., :, indices].copy()


def quaternion_wxyz(value):
    a = np.asarray(value).copy()
    if a.shape[-1:] != (4,):
        raise ValueError('A quaternion must contain w, x, y, z')
    a[..., 1:] = axial(a[..., 1:])
    return a


def wrench(value):
    """Transport force/moment or translation/angular-velocity; keep auxiliaries."""
    a = np.asarray(value)
    if a.shape[-1] < 6:
        raise ValueError('A wrench needs at least six components')
    out = a.copy()
    out[..., :6] = a[..., WRENCH_ORDER]*WRENCH_SIGNS
    return out


def faces(value):
    a = np.asarray(value)
    if a.shape[-1:] != (3,):
        raise ValueError('Expected triangular face indices')
    return a[..., SWAP].copy()


def triangles(value):
    a = np.asarray(value)
    if a.shape[-2:] != (3, 3):
        raise ValueError('Expected triangles with three three-dimensional vertices')
    return polar(a[..., SWAP, :])


def mesh(value):
    """Preserve topology, outward normals and positive material volume."""
    import trimesh
    return trimesh.Trimesh(vertices=polar(value.vertices), faces=faces(value.faces),
                           process=False, metadata=dict(value.metadata))
