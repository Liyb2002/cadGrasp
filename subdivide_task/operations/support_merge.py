"""Propose overlapping two seats, including a whole-fixture yaw installation.

No host label constitutes reuse. The builder cuts BOTH work/exit volumes and
the audit requires both full load domains on the same actual material.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .state import transform


def support_merge(layout, owner, host, head_world, yaw_deg=0., offset=(0., 0., 0.)):
    if owner == host:
        raise ValueError('A merge needs two different tasks')
    q = layout.copy()
    rotation = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
    target = transform(np.asarray(head_world), q.material[host]) + np.asarray(offset)
    q.material[owner, :3, :3] = rotation @ q.material[host, :3, :3]
    q.material[owner, :3, 3] = target - q.material[owner, :3, :3] @ head_world
    return q
