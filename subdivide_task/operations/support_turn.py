"""Turn a seat's installation in SO(3), keeping its working world pose fixed.

This is a design proposal, including a 90 degree turn onto a side face. It
changes the installation of the SAME complete fixture for the selected tasks;
acceptance always rebuilds and checks every use of that fixture.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def support_turn(layout, owners, axis, degrees, pivot):
    owners = [owners] if np.isscalar(owners) else list(owners)
    axis = np.asarray(axis, float)
    pivot = np.asarray(pivot, float)
    if (not owners or len(set(owners)) != len(owners) or not np.isfinite(axis).all()
            or np.linalg.norm(axis) < 1e-12 or pivot.shape != (3,)
            or not np.isfinite(pivot).all() or not np.isfinite(degrees)):
        raise ValueError('Expected distinct owners, a finite pivot and a nonzero rotation axis')
    R = Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.radians(degrees)).as_matrix()
    change = np.eye(4)
    change[:3, :3] = R
    change[:3, 3] = pivot-R@pivot
    q = layout.copy()
    for owner in owners:
        q.material[owner] = change@q.material[owner]
    return q
