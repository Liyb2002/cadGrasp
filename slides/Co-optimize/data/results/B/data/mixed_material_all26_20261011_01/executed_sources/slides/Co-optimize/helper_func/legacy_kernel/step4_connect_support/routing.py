"""Tapered convex beams used by the direction-first frame and base connectors."""
from itertools import product
import numpy as np
from step2_local_support import insertion as D

CORNERS = np.array(list(product([-1., 1.], repeat=3)))


def beam(first, second, first_radius, second_radius):
    """Convex tapered bar between two world-aligned cubic joints."""
    return D.engine.hull_mesh(np.vstack([first+CORNERS*first_radius,
                                        second+CORNERS*second_radius]))
