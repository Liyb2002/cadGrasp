"""Change one task's WORLD extraction direction; the audit rebuilds its channel."""
import numpy as np


def exit_direction_change(layout, owner, direction):
    direction = np.asarray(direction, float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12:
        raise ValueError('Expected a finite nonzero exit vector')
    direction = direction / np.linalg.norm(direction)
    if direction[2] < .05:
        raise ValueError('Extraction must enter the declared upper hemisphere')
    q = layout.copy()
    q.exits[owner] = direction
    return q
