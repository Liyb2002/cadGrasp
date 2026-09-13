"""Exact transport between the existing solver frame and the Y-up world.

World positions use (x, y, z) = (old x, old z, old y). This permutation
has determinant -1: polar vectors use P, axial vectors use -P, rotations
use P R P, and mesh faces reverse their winding. All maps are involutions.
"""
import numpy as np

UP_AXIS = 1
FLOOR_AXES = (0, 2)
WORLD_UP = np.array([0., 1., 0.])
SWAP = np.array([0, 2, 1])
P = np.eye(3)[SWAP]
H = np.eye(4)[[0, 2, 1, 3]]
WRENCH_ORDER = np.array([0, 2, 1, 3, 5, 4])
WRENCH_SIGNS = np.array([1., 1., 1., -1., -1., -1.])


def floor(value):
    """World horizontal coordinates, ordered x, z."""
    return np.asarray(value)[..., [0, 2]]


def lift_floor(value, height=0.):
    a = np.asarray(value)
    result = np.empty(a.shape[:-1]+(3,), dtype=float)
    result[..., [0, 2]] = a
    result[..., 1] = height
    return result


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


def matplotlib_view(ax, elev, azim):
    """Preserve a slide's camera composition while displaying native Y-up data.

    ``azim`` measures the eye in the XZ floor plane from +X toward +Z. The
    reflection of the old world also reverses image parity; reverse projection
    X, rather than changing any physical data or its axis labels.
    """
    # Matplotlib stores box aspect in its current vertical-axis order. Preserve
    # the actual world-axis lengths when changing that order (including zoom).
    world_aspect = ax._roll_to_vertical(ax._box_aspect).copy()
    ax.view_init(elev=elev, azim=90.-azim, vertical_axis='y')
    ax._box_aspect = ax._roll_to_vertical(world_aspect, reverse=True)
    if not hasattr(ax, '_yup_original_projection'):
        ax._yup_original_projection = ax.get_proj
        ax.get_proj = lambda: np.diag([-1., 1., 1., 1.]) @ ax._yup_original_projection()
