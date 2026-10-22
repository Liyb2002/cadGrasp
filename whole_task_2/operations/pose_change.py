"""Rotate the object about a supplied WORLD axis and pivot, keeping the head fixed.

Needle-axis rotation preserves the spherical sector exactly. A working-area
normal rotation requires the same vertex-wise sector audit as every proposal.
"""
from whole_task_2.common.poses import spin_pose


def pose_change(layout, owner, degrees, axis, pivot):
    q = layout.copy()
    q.poses[owner] = spin_pose(q.poses[owner], axis, pivot, degrees)
    return q
