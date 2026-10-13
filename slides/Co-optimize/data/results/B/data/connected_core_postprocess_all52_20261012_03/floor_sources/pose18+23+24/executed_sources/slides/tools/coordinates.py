"""Native right-handed Z-up world coordinates; the floor is z=0."""
import numpy as np

UP_AXIS = 2
FLOOR_AXES = (0, 1)
WORLD_UP = np.array([0., 0., 1.])


def floor(value):
    return np.asarray(value)[..., :2]


def lift_floor(value, height=0.):
    a = np.asarray(value)
    result = np.empty(a.shape[:-1]+(3,), dtype=float)
    result[..., :2] = a
    result[..., 2] = height
    return result


def matplotlib_view(ax, elev, azim):
    ax.view_init(elev=elev, azim=azim, vertical_axis='z')
