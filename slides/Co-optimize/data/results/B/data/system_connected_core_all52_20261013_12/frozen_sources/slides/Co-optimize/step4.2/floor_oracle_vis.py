"""Zero-thickness floor boundaries and an actual-layout cross-state example.

The pink probe is hypothetical material, not a part of the saved fixture.
Only final installed states are shown; no transfer-motion legality is imposed.
"""
import numpy as np
import trimesh
from pose_oracle import transform_points
from work_access import tangent_frame
from whole_search.common import transform_mesh


def floor_sheet(plane, center, half_lengths):
    """A rectangular boundary a.x+b=0, with no extrusion or closed volume."""
    plane=np.asarray(plane,float);normal=plane[:3]/np.linalg.norm(plane[:3])
    offset=plane[3]/np.linalg.norm(plane[:3]);center=np.asarray(center,float)
    projected=center-(center@normal+offset)*normal
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.asarray(half_lengths)
    vertices=projected+corners@tangent_frame(normal).T
    return trimesh.Trimesh(vertices,[[0,1,2],[0,2,3]],process=False)


def floor_boundaries(constraints,task,center,radius):
    """Pull each final floor back to the selected task's material placement."""
    sheets=[];rows=[];seen={};T=task.T_world_fixture
    for other in constraints.poses:
        state=(np.eye(4) if np.array_equal(other.T_world_fixture,T)
               else other.T_world_fixture@np.linalg.inv(T))
        plane=state[2];key=np.round(plane,11).tobytes()
        if key in seen:
            rows[seen[key]]['source_poses'].append(other.pose);continue
        seen[key]=len(rows)
        sheets.append(floor_sheet(plane,center,[1.15*radius,.85*radius]))
        rows.append(dict(plane_current_world=plane.tolist(),source_poses=[other.pose],
            physical_current_floor=bool(np.allclose(plane,[0,0,1,0],atol=1e-11,rtol=0)),
            illegal_sign='a.x+b < 0',zero_thickness=True))
    return trimesh.util.concatenate(sheets),rows


def cross_state_probe(constraints,fixture,offset=.020,cube_width=.008):
    """Find real other-state feet, then move a hypothetical point past them.

This creates a point above the current ground but below another final ground.
It does not change or mislabel the saved accepted support.
"""
    choices=[]
    for before in constraints.poses:
        for after in constraints.poses:
            if np.allclose(before.T_world_fixture,after.T_world_fixture,atol=1e-12,rtol=0):continue
            a=after.T_world_fixture[2]
            ids=np.flatnonzero(np.max(np.abs(fixture.triangles@a[:3]+a[3]),axis=1)<1e-7)
            if not len(ids):continue
            source=fixture.triangles_center[ids]
            points=source-offset*a[:3]/np.linalg.norm(a[:3])
            b=before.T_world_fixture[2]
            heights=points@b[:3]+b[3]
            for i in np.flatnonzero(heights>.025):
                choices.append((float(heights[i]),before,after,points[i],source[i],int(ids[i])))
    if not choices:raise ValueError('No above-current/below-other floor example in this saved layout')
    _,before,after,point,foot,face=max(choices,key=lambda item:item[0])
    probe=trimesh.creation.box([cube_width]*3);probe.apply_translation(point)
    before_world=transform_points(point,before.T_world_fixture)
    after_world=transform_points(point,after.T_world_fixture)
    before_mesh=transform_mesh(probe,before.T_world_fixture)
    after_mesh=transform_mesh(probe,after.T_world_fixture)
    if before_mesh.vertices[:,2].min()<=0 or after_mesh.vertices[:,2].max()>=0:
        raise ValueError('The full diagnostic marker must switch from above to below ground')
    return before,after,probe,dict(
        before_pose=before.pose,after_pose=after.pose,
        hypothetical_material=True,part_of_saved_final_fixture=False,
        point_fixture_m=point.tolist(),reference_real_foot_fixture_m=foot.tolist(),
        reference_real_foot_face=face,cube_width_m=cube_width,
        before_world_point_m=before_world.tolist(),after_world_point_m=after_world.tolist(),
        before_height_m=float(before_world[2]),after_height_m=float(after_world[2]),
        fixture_origin_z_change_m=float(after.T_world_fixture[2,3]-before.T_world_fixture[2,3]),
        entire_marker_above_before=True,entire_marker_below_after=True,
        marker_geometry_unchanged_between_states=True,
        actual_floor_equation='world z=0',floor_thickness_m=0.,
        intermediate_fixture_rotations_constrained=False)


def demo_scenes(constraints,fixture,scenes):
    before,after,probe,proof=cross_state_probe(constraints,fixture)
    source={task.index:s for task,s in zip(constraints.poses,scenes)}
    result=[]
    for task in (before,after):
        s=dict(source[task.index]);T=task.T_world_fixture
        s['fixture']=transform_mesh(fixture,T);s['probe']=transform_mesh(probe,T)
        marker=transform_points(np.array(proof['point_fixture_m']),T)
        ground=marker.copy();ground[2]=0.
        s['height_guide_world_m']=np.array([marker,ground])
        points=np.vstack([s['body'].vertices,s['fixture'].vertices,s['probe'].vertices,s['yellow'].vertices])
        center=(points.min(0)+points.max(0))/2
        width=max(float(np.ptp(points[:,:2],axis=0).max())*.70,.13)
        s['pink']=floor_sheet([0,0,1,0],center,[width,width*.70])
        result.append(s)
    all_points=np.vstack([mesh.vertices for s in result
        for mesh in (s['body'],s['fixture'],s['probe'],s['yellow'],s['pink'])])
    center=(all_points.min(0)+all_points.max(0))/2
    radius=np.linalg.norm(all_points-center,axis=1).max()
    for s in result:s['center']=center;s['span']=float(2*radius*1.05)
    proof['dashed_height_guide']='graphic from candidate point vertically to world Z=0, not fixture material'
    return result,proof
