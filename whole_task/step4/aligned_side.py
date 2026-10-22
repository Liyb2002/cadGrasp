"""Match two object seats exactly while standing the complete fixture 90°.

A needle-axis spin preserves the original working sector. Solve its angle
analytically, rather than forcing a mismatch between the two object poses.
Only proposals are produced here; the all-task acceptance gate remains final.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from whole_task.operations import pose_change,support_turn,exit_direction_change
from whole_task.operations.state import transform


def needle_angles(layout,axis,owner,host):
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
    z=np.array([0.,0.,1.])
    target=layout.poses[host,:3,:3]@layout.poses[owner,:3,:3].T
    parallel=axis*(axis@z)
    a=float(z@target@parallel)
    b=float(z@target@(z-parallel))
    c=float(z@target@(-np.cross(axis,z)))
    radius=np.hypot(b,c)
    if radius<1e-12 or abs(a)>radius+1e-12:return []
    phase=np.arctan2(c,b)
    delta=np.arccos(np.clip(-a/radius,-1.,1.))
    values=[float((np.degrees(phase+s*delta)+180.)%360.-180.)for s in(-1.,1.)]
    return sorted(set(round(v,11)for v in values),key=abs)


def propose(model,layout,owner,host,angle,offset=(0.,0.,0.)):
    q=pose_change(layout,owner,angle,model.axis,model.head)
    desired=q.material[host]@q.poses[host]@np.linalg.inv(q.poses[owner])
    desired[:3,3]+=np.asarray(offset)
    vector=Rotation.from_matrix(desired[:3,:3]@q.material[owner,:3,:3].T).as_rotvec()
    radians=np.linalg.norm(vector)
    if radians>1e-12:
        pivot=transform(model.source.center_mass,q.material[owner]@q.poses[owner])
        q=support_turn(q,owner,vector/radians,np.degrees(radians),pivot)
    q.material[owner,:3,3]=desired[:3,3]
    return q


def common_exit(layout,owner,host,weight=.5):
    """A shared cavity can retain more contact area with a shared opening."""
    normals=layout.material[:,:3,2]
    direction=(weight*(layout.material[owner,:3,:3]@layout.exits[owner])+
               (1-weight)*(layout.material[host,:3,:3]@layout.exits[host]))
    if np.linalg.norm(direction)<1e-12:return None
    direction/=np.linalg.norm(direction)
    for _ in range(30):
        for n in normals:
            direction+=max(0.,.06-n@direction)*n
        direction/=np.linalg.norm(direction)
    q=layout.copy()
    for k in(owner,host):
        world=q.material[k,:3,:3].T@direction
        world/=np.linalg.norm(world)
        if world[2]<.05+1e-12:return None
        q=exit_direction_change(q,k,world)
    return q


def preserve_seat(model,layout,owner):
    """Stand up a fixture by a needle spin while retaining its existing seat.

    With the 45 degree world needle, a half turn about the needle makes the
    two installation floors perpendicular. The object-to-fixture pose and
    material-space exit stay exactly the same; gravity is still world-fixed.
    """
    q=pose_change(layout,owner,180.,model.axis,model.head)
    desired=layout.material[owner]@layout.poses[owner]@np.linalg.inv(q.poses[owner])
    vector=Rotation.from_matrix(desired[:3,:3]@q.material[owner,:3,:3].T).as_rotvec()
    radians=np.linalg.norm(vector)
    pivot=transform(model.source.center_mass,q.material[owner]@q.poses[owner])
    q=support_turn(q,owner,vector/radians,np.degrees(radians),pivot)
    q.material[owner,:3,3]=desired[:3,3]
    material_exit=layout.material[owner,:3,:3]@layout.exits[owner]
    return exit_direction_change(q,owner,desired[:3,:3].T@material_exit)


def sweep(optimizer,maximum_trials=24):
    """Try side installations before large task cohorts have been assembled."""
    from .optimize import subset,record
    from .quarter_turns import reassigned_groups
    from .spatial_search import floor_repair,separate_free_axis
    from .standing_search import protect_exits
    from whole_task.operations.installations import standing_pairs
    if standing_pairs(optimizer.layout):return False
    m=optimizer.model;base=optimizer.layout.copy();trials=0
    previous_grid=getattr(m,'stock_grid',None)
    m.stock_grid=(.03,.012)
    ranked=sorted((abs(angle),owner,host,angle)
        for owner in range(len(m.ids))for host in range(len(m.ids))if owner!=host
        for angle in needle_angles(base,m.axis,owner,host))
    for _,owner,host,angle in ranked:
        raw=propose(m,base,owner,host,angle)
        groups=[[k for k in g if k not in(owner,host)]for g in optimizer.groups]
        groups=[g for g in groups if g]+[[owner,host]]
        for seed in(raw,protect_exits(raw,margin=.05)):
            q=common_exit(seed,owner,host)
            if q is None:continue
            try:q=floor_repair(m,q,groups,host)
            except(ValueError,RuntimeError,np.linalg.LinAlgError):continue
            if q is None:continue
            local,lq=subset(m,q,[owner,host])
            try:
                candidate=local.build(lq)
                local.audit(candidate,full=False,continuous=False,all_outer=True)
            except(ValueError,RuntimeError,AssertionError,np.linalg.LinAlgError):continue
            packed=separate_free_axis(m,q,groups,host)
            for placed in(q,packed):
                if placed is None:continue
                trials+=1
                if optimizer.try_operation(placed,'support_turn',groups=groups,
                        tasks=[m.ids[owner],m.ids[host]],turn_angle_deg=90.,
                        axial_pose_angle_deg=angle,exact_object_orientation_alignment=True,
                        common_material_exit=True,early_side_installation_search=True):
                    optimizer.requirements=dict(final_standing_installation_required=True,
                        relative_floor_angle_deg=90.,tolerance_deg=1e-5)
                    record(optimizer.folder/'requirements.json',optimizer.requirements)
                    return True
                if trials>=maximum_trials:
                    m.stock_grid=previous_grid
                    return False
    m.stock_grid=previous_grid
    return False
