"""Require the contact membrane itself to meet each installation's ground.

An extension or a distant leg touching the ground does not satisfy this
condition. Floor heights are explicit, so changing coordinate frames cannot
hide a shell/ground gap or an object penetrating the same plane.
"""
import numpy as np

from whole_task_2.operations.state import transform


def check(core, body, layout, ids, floor_heights=0., tolerance=1e-8):
    heights=np.broadcast_to(np.asarray(floor_heights,dtype=float),(len(ids),))
    if not np.isfinite(heights).all():raise ValueError('Ground heights must be finite')
    rows=[]
    for i,name in enumerate(ids):
        shell=transform(core.vertices,np.linalg.inv(layout.material[i]))
        object_world=transform(body.vertices,layout.poses[i])
        index=int(shell[:,2].argmin())
        gap=float(shell[index,2]-heights[i])
        body_gap=float(object_world[:,2].min()-heights[i])
        rows.append(dict(id=name,passed=bool(abs(gap)<=tolerance and body_gap>=-tolerance),
                         floor_world_z_m=float(heights[i]),shell_gap_m=gap,
                         object_clearance_m=body_gap,
                         shell_lowest_world_m=shell[index].tolist(),
                         shell_touches_ground=bool(abs(gap)<=tolerance),
                         object_above_ground=bool(body_gap>=-tolerance)))
    return dict(passed=all(r['passed'] for r in rows),tasks=rows,
                membrane_itself_must_touch=True,remote_feet_do_not_satisfy_direct_contact=True,
                distance_tolerance_m=tolerance)
