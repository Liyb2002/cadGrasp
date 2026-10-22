"""Explicit 90 degree side-turn proposals at any task/cohort size."""
from itertools import product
import numpy as np
import warnings
from scipy.spatial.transform import Rotation
from whole_task.operations import support_turn
from whole_task.operations import exit_direction_change
from whole_task.operations.state import transform
from .spatial_search import floor_repair


def propose(model,layout,owners,owner,host,axis,degrees,offset):
    pivot=transform(model.source.center_mass,layout.material[owner]@layout.poses[owner])
    if axis=='closest_side':
        # The intrinsic ZXZ middle angle is the angle between installation
        # floor normals. Replacing it by 90 degrees gives the closest body
        # orientation on a side face; retain the optimal two bearings.
        target=layout.poses[host,:3,:3]@layout.poses[owner,:3,:3].T
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            a,_,c=Rotation.from_matrix(target).as_euler('ZXZ')
        desired_R=layout.material[host,:3,:3]@Rotation.from_euler('ZXZ',[a,np.pi/2,c]).as_matrix()
        vector=Rotation.from_matrix(desired_R@layout.material[owner,:3,:3].T).as_rotvec()
        angle=np.linalg.norm(vector)
        q=layout.copy()if angle<1e-12 else support_turn(layout,owners,vector/angle,np.degrees(angle),pivot)
    else:
        q=support_turn(layout,owners,layout.material[owner,:3,axis],degrees,pivot)
    desired=transform(model.source.center_mass,layout.material[host]@layout.poses[host])+np.asarray(offset)
    delta=desired-transform(model.source.center_mass,q.material[owner]@q.poses[owner])
    for k in owners:q.material[k,:3,3]+=delta
    return q


def reassigned_groups(groups,owners,host):
    owners=set(owners)
    if host in owners:
        raise ValueError('Host must be outside the turned tasks')
    merged=owners|set(next(g for g in groups if host in g))
    result=[[k for k in g if k not in merged]for g in groups]
    return [g for g in result if g]+[sorted(merged)]


def proposals(optimizer,single_tasks=False):
    m=optimizer.model
    q=optimizer.layout
    corners=[]
    for record,pose in zip(m.records,q.poses):
        data=m.world_data(record,pose)
        lo,hi=data['stock_world'].bounds
        corners.append(np.array(list(product(*zip(lo,hi)))))
    def bounds(layout,group):
        points=np.vstack([transform(corners[k],layout.material[k])for k in group])
        return np.array([points.min(0),points.max(0)])
    ranked=[];seen=set()
    bodies=[model['body_world'].vertices for model in
            [m.world_data(r,p)for r,p in zip(m.records,q.poses)]]
    offsets=[(0.,0.,0.),(.025,0.,0.),(-.025,0.,0.),(0.,.025,0.),(0.,-.025,0.),
             (0.,0.,.025),(0.,0.,-.025),(0.,0.,.05),(0.,0.,-.05)]
    owner_options=[[k]for k in range(len(m.ids))]if single_tasks else optimizer.groups
    for ga in owner_options:
        for original_host_group in optimizer.groups:
            gb=[k for k in original_host_group if k not in ga]
            if not gb:continue
            host_box=bounds(q,gb)
            for owner,host,axis,degrees,offset in product(ga,gb,(0,1,'closest_side'),(-90.,90.),offsets):
                if axis=='closest_side' and degrees<0:continue
                p=propose(m,q,ga,owner,host,axis,degrees,offset)
                box=bounds(p,ga)
                overlap=np.maximum(0.,np.minimum(box[1],host_box[1])-np.maximum(box[0],host_box[0]))
                score=float(np.prod(overlap))
                if axis=='closest_side':score*=1.2
                if score<1e-9:continue
                key=np.round(p.material,8).tobytes()
                if key in seen:continue
                seen.add(key)
                groups=reassigned_groups(optimizer.groups,ga,host)
                # Prefer openings that do not pass through either supporting
                # face. A support turn and per-task exit changes form one
                # jointly audited proposal, rather than carving down through
                # the other installation's feet.
                from whole_task.operations.floors import planes
                normals=[n for n,_ in planes(p.material)]
                joint=p.copy()
                for k in range(len(m.ids)):
                    direction=p.material[k,:3,:3]@p.exits[k]
                    if min(n@direction for n in normals)<.05:
                        for _ in range(20):
                            for n in normals:
                                direction+=max(0.,.2-n@direction)*n
                            direction/=np.linalg.norm(direction)
                        world=p.material[k,:3,:3].T@direction
                        if world[2]>=.05:
                            joint=exit_direction_change(joint,k,world)
                changed=not np.allclose(joint.exits,p.exits,atol=1e-10,rtol=0)
                protection=1.
                local_planes=planes(p.material[ga+gb])
                for k in ga+gb:
                    points=transform(bodies[k],p.material[k])
                    if min(float((points@n-o).min())for n,o in local_planes)<.007:
                        protection=.2;break
                if changed:
                    ranked.append((1.01*score*protection,joint,groups,ga,gb,owner,host,axis,degrees,offset,True))
                ranked.append((score*protection,p,groups,ga,gb,owner,host,axis,degrees,offset,False))
    # Do not spend a whole search pass on nearly identical offsets of the
    # single largest casting. Give every task/pair a chance before returning
    # to its next-ranked offset.
    from itertools import zip_longest
    buckets={}
    for row in sorted(ranked,key=lambda x:-x[0]):
        key=row[5]if single_tasks else(tuple(row[3]),tuple(row[4]))
        buckets.setdefault(key,[]).append(row)
    return [row for layer in zip_longest(*buckets.values())for row in layer if row is not None]


def sweep(optimizer,maximum_trials=32,single_tasks=False):
    from .optimize import subset
    m=optimizer.model
    trials=0;changed=False
    done=set()
    for event in reversed(optimizer.events):
        if event['status']=='accepted':break
        if event['status']=='quarter_turn_proposed' and event.get('task_level_turn',False)==single_tasks:
            done.add((tuple(event['tasks']),event['turn_axis_in_owner_world'],event['turn_angle_deg'],
                      tuple(event['offset_m']),event.get('coupled_exit_direction_changes',False)))
    while (single_tasks or len(optimizer.groups)>1) and trials<maximum_trials:
        accepted=False
        ranked=proposals(optimizer,single_tasks)
        optimizer.event('quarter_turn_candidates','support_turn',count=len(ranked),trial=trials,task_level_turn=single_tasks)
        for _,q,groups,ga,gb,owner,host,axis,degrees,offset,joint_exit in ranked:
            axis_name=axis if isinstance(axis,str)else 'xy'[axis]
            key=((m.ids[owner],m.ids[host]),axis_name,degrees,tuple(offset),joint_exit)
            if key in done:continue
            done.add(key)
            q=floor_repair(m,q,groups,host)
            if q is None:continue
            trials+=1
            detail=dict(tasks=[m.ids[owner],m.ids[host]],turn_axis_in_owner_world=axis_name,
                        turn_angle_deg=degrees,offset_m=list(offset),explicit_side_turn=True,
                        coupled_exit_direction_changes=joint_exit,task_level_turn=single_tasks)
            optimizer.event('quarter_turn_proposed','support_turn',trial=trials,**detail)
            local,lq=subset(m,q,ga+gb)
            try:
                candidate=local.build(lq)
                local.audit(candidate,full=False,continuous=False)
            except (ValueError,RuntimeError,AssertionError,np.linalg.LinAlgError) as e:
                optimizer.event('quarter_turn_local_rejected','support_turn',reason=str(e)[:250],**detail)
                if trials>=maximum_trials:break
                continue
            if optimizer.try_operation(q,'support_turn',groups=groups,**detail):
                changed=accepted=True;done.clear()
                break
            if trials>=maximum_trials:break
        if not accepted:break
    return changed
