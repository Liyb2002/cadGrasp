"""Translate one seat in fixture coordinates, counter-moving its installation."""
import numpy as np


def translation(layout, owner, delta):
    delta = np.asarray(delta, float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError('Expected a finite XYZ seat displacement')
    q = layout.copy()
    q.material[owner, :3, 3] += delta
    return q
