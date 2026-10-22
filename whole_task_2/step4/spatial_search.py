"""Side-turn proposals during optimization, before seats become large cohorts."""
import math
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.operations import pose_change, support_turn
from whole_task_2.operations.state import transform
from whole_task_2.common.force_cone import tangent_frame


def floor_repair(model, layout, groups, anchor):
    """Translate cohorts to satisfy ALL installation-floor pressure necessities.

    This LP only proposes placements. Actual floor polygons, material geometry,
    contacts and all paired loads are subsequently independently checked.
    """
    owner_group = {k:g for g,group in enumerate(groups)for k in group}
    nvars = 3*len(groups)
    matrices = layout.material
    points = []
    for i,(record,pose) in enumerate(zip(model.records,layout.poses)):
        data = model.world_data(record,pose)
        pressure = G.pressure_points(data['endpoints']['endpoint_need_wrench'],data['domain'].com)
        pressure = pressure[ConvexHull(pressure).vertices]
        p=transform(np.c_[pressure,np.zeros(len(pressure))],matrices[i])
        if owner_group[i] != owner_group[anchor]:
            # Preserve ACTUAL current contacts. Raw casting stocks can cross
            # another floor before carving; requiring those unused raw cells
            # would incorrectly prohibit already valid side installations.
            contacts=getattr(model,'current_contacts',{}).get(record['id'])
            if contacts is not None:
                p=np.vstack([p,transform(contacts,matrices[i])])
        points.append(p)
    A,b = [],[]
    for i in range(len(points)):
        gi = owner_group[i]
        for j,M in enumerate(matrices):
            gj = owner_group[j]
            normal = M[:3,2]
            distance = float(((points[i]-M[:3,3])@normal).min())
            margin = 0. if normal@matrices[i,:3,2]>1.-1e-9 else 2e-4
            if gi==gj:
                if distance < -1e-9:
                    return None
                continue
            row = np.zeros(2*nvars)
            row[3*gi:3*gi+3] -= normal
            row[3*gj:3*gj+3] += normal
            A.append(row)
            b.append(distance-margin)
    # Minimize absolute translation, keeping one cohort as a fixed gauge.
    for k in range(nvars):
        row = np.zeros(2*nvars);row[k]=1;row[nvars+k]=-1
        A.append(row);b.append(0.)
        row = np.zeros(2*nvars);row[k]=-1;row[nvars+k]=-1
        A.append(row);b.append(0.)
    bounds = [(None,None)]*nvars+[(0,None)]*nvars
    for k in range(3*owner_group[anchor],3*owner_group[anchor]+3):
        bounds[k]=(0.,0.)
    result = linprog(np.r_[np.zeros(nvars),np.ones(nvars)],A_ub=np.asarray(A),b_ub=np.asarray(b),bounds=bounds,
                     method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:
        return None
    q=layout.copy()
    for g,group in enumerate(groups):
        for owner in group:
            q.material[owner,:3,3] += result.x[3*g:3*g+3]
    return q


def separate_free_axis(model,layout,groups,anchor):
    """Use a common floor-tangent direction to separate unrelated seats.

    With two independent floor normals there is a direction tangent to both.
    Translations along it leave every installation floor unchanged. This is
    proposal placement, followed by the same all-task physical acceptance.
    """
    normals=layout.material[:,:3,2]
    _,singular,vh=np.linalg.svd(normals,full_matrices=False)
    if np.count_nonzero(singular>1e-8)!=2:
        return separate_inward(model,layout,groups,anchor)
    axis=vh[-1]
    q=layout.copy()
    bounds=[]
    for group in groups:
        clouds=[]
        for k in group:
            data=model.world_data(model.records[k],layout.poses[k])
            clouds.append(transform(data['stock_world'].vertices,layout.material[k]))
            h=transform(model.head,layout.material[k])
            from itertools import product
            clouds.append(h+np.array(list(product((-.13,.13),repeat=3))))
        values=np.vstack(clouds)@axis
        bounds.append((values.min(),values.max()))
    anchor_group=next(g for g,owners in enumerate(groups)if anchor in owners)
    left,right=bounds[anchor_group]
    for g,group in enumerate(groups):
        if g==anchor_group:
            continue
        low,high=bounds[g]
        if g%2:
            shift=left-.055-high
            left=low+shift
        else:
            shift=right+.055-low
            right=high+shift
        for k in group:
            q.material[k,:3,3]+=shift*axis
    return q


def separate_inward(model,layout,groups,anchor):
    """Pack unrelated single-floor cohorts into a common admissible ray.

    Several side faces can leave an unbounded wedge on an upright bottom.
    Moving further into that wedge preserves all floor inequalities; relying
    on a shared null direction alone would miss this placement freedom.
    """
    normals=layout.material[:,:3,2]
    anchor_group=next(g for g,owners in enumerate(groups)if anchor in owners)
    movable,clouds=[],[]
    from itertools import product
    for g,group in enumerate(groups):
        cloud=[]
        for k in group:
            data=model.world_data(model.records[k],layout.poses[k])
            cloud.append(transform(data['stock_world'].vertices,layout.material[k]))
            h=transform(model.head,layout.material[k])
            cloud.append(h+np.array(list(product((-.13,.13),repeat=3))))
        clouds.append(np.vstack(cloud))
        if g!=anchor_group and all(np.linalg.norm(normals[k]-normals[group[0]])<1e-8 for k in group):
            movable.append(g)
    occupied=[c for g,c in enumerate(clouds)if g not in movable]
    q=layout.copy()
    for g in movable:
        normal=normals[groups[g][0]]
        u,v=tangent_frame(normal)
        angles=np.arange(0.,2*np.pi,2*np.pi/360)
        rays=np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v
        projected=normals-(normals@normal)[:,None]*normal
        projected=projected[np.linalg.norm(projected,axis=1)>1e-8]
        if not len(projected):
            occupied.append(clouds[g]);continue
        slack=(rays@projected.T).min(1)
        index=slack.argmax()
        if slack[index]<-1e-10:
            occupied.append(clouds[g]);continue
        ray=rays[index]
        previous=np.vstack(occupied)
        distance=max(0.,float((previous@ray).max()-(clouds[g]@ray).min()+.055))
        shift=distance*ray
        for k in groups[g]:
            q.material[k,:3,3]+=shift
        occupied.append(clouds[g]+shift)
    return q


def proposal(optimizer, ga, gb, owner, host, axial, delta, offset):
    """Rotate a seat toward a shared object orientation, including side faces."""
    q,m=optimizer.layout.copy(),optimizer.model
    q=pose_change(q,owner,axial,m.axis,m.head)
    target_R=q.material[host,:3,:3]@q.poses[host,:3,:3]
    desired_R=delta@target_R@q.poses[owner,:3,:3].T
    center=transform(m.source.center_mass,q.material[host]@q.poses[host])+np.asarray(offset)
    desired=np.eye(4)
    desired[:3,:3]=desired_R
    desired[:3,3]=center-desired_R@transform(m.source.center_mass,q.poses[owner])
    change=desired@np.linalg.inv(q.material[owner])
    rot=Rotation.from_matrix(change[:3,:3])
    vector=rot.as_rotvec()
    angle=np.linalg.norm(vector)
    if angle>1e-12:
        q=support_turn(q,ga,vector/angle,np.degrees(angle),transform(m.source.center_mass,q.material[owner]@q.poses[owner]))
    # The turn and the following translation jointly define a proposal. No
    # intermediate geometry is accepted without the full physical audit.
    shift=desired[:3,3]-q.material[owner,:3,3]
    for k in ga:
        q.material[k,:3,3] += shift
    return q


def sweep(optimizer, maximum_trials=60):
    from .optimize import subset
    m=optimizer.model
    version='spatial-installations-v4'
    attempted=set()
    for event in reversed(optimizer.events):
        if event['status']=='accepted':
            break
        if event['status']=='turn_pair_started' and event.get('search_version')==version:
            owner,host=[m.ids.index(name)for name in event['tasks']]
            ga=next(g for g in optimizer.groups if owner in g)
            gb=next(g for g in optimizer.groups if host in g)
            if ga!=gb:
                attempted.add((tuple(ga),tuple(gb),owner,host))
    trials=0
    offsets=[(0.,0.,0.),(.025,0.,0.),(-.025,0.,0.),(0.,.025,0.),(0.,-.025,0.),
             (0.,0.,.025),(0.,0.,-.025),(.04,0.,.025),(-.04,0.,.025)]
    while len(optimizer.groups)>1 and trials<maximum_trials:
        choices=[]
        for a,ga in enumerate(optimizer.groups):
            for b,gb in enumerate(optimizer.groups):
                if a==b:
                    continue
                for owner in ga:
                    for host in gb:
                        key=(tuple(ga),tuple(gb),owner,host)
                        if key in attempted:
                            continue
                        # Prefer complementary selected surface regions and
                        # similar COM heights; both owner/host roles remain.
                        ia=set(m.records[owner]['working_face_ids']);ib=set(m.records[host]['working_face_ids'])
                        shared=len(ia&ib)/max(1,min(len(ia),len(ib)))
                        score=(1-shared)+.005*(len(ga)+len(gb))
                        choices.append((score,a,b,owner,host,key))
        if not choices:
            break
        _,a,b,owner,host,key=min(choices,key=lambda x:x[0])
        ga,gb=optimizer.groups[a],optimizer.groups[b]
        attempted.add(key);trials+=1
        newgroups=[g for j,g in enumerate(optimizer.groups)if j not in(a,b)]+[sorted(ga+gb)]
        optimizer.event('turn_pair_started','support_turn',tasks=[m.ids[owner],m.ids[host]],trial=trials,search_version=version)
        ranked=[]
        # Exact object alignment leaves one free axial world pose. It changes
        # the fixture floor/working opening relative to the source object.
        for axial in (0.,90.,-90.,180.,45.,-45.,135.,-135.):
            for offset in offsets:
                q=proposal(optimizer,ga,gb,owner,host,axial,np.eye(3),offset)
                repaired=floor_repair(m,q,newgroups,host)
                if repaired is None:
                    continue
                packed=separate_free_axis(m,repaired,newgroups,host)
                # Local construction is a rejection screen only.
                local,lq=subset(m,repaired,ga+gb)
                try:
                    candidate=local.build(lq)
                    local.audit(candidate,full=False,continuous=False)
                except (ValueError,RuntimeError,AssertionError,np.linalg.LinAlgError):
                    continue
                angle=float(np.degrees(np.arccos(np.clip(repaired.material[owner,:3,2]@repaired.material[host,:3,2],-1.,1.))))
                ranked.append((candidate['volume_cm3'],repaired,packed,axial,offset,angle))
                # Avoid building every offset after finding a viable alignment.
                break
            optimizer.event('turn_alignment_screened','support_turn',tasks=[m.ids[owner],m.ids[host]],
                            axial_angle_deg=axial,viable_local_proposals=len(ranked))
        success=False
        for local_volume,q,packed,axial,offset,angle in sorted(ranked,key=lambda row:row[0])[:4]:
            variants=[q]
            if not np.allclose(q.material,packed.material,atol=1e-10,rtol=0):
                variants.append(packed)
            for variant,p in enumerate(variants):
                if optimizer.try_operation(p,'support_turn',groups=newgroups,tasks=[m.ids[owner],m.ids[host]],
                        relative_floor_angle_deg=angle,axial_angle_deg=axial,offset_m=list(offset),
                        placement='3D installation turn, juxtaposition and floor-feasible cohort translations',
                        unrelated_seats_separated=bool(variant),local_proposal_volume_cm3=local_volume):
                    attempted.clear();success=True
                    break
            if success:
                break
