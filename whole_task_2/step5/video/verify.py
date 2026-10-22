"""Replay native joints and attachments independently of the movie renderer."""
import argparse,json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from whole_task_2.step1.common import write_json,sha256
from .franka import Franka
from .kuka import Arm


def outer_boundary(vertices, boundary, axis):
    """Find the outer projected boundary independently of the path planner."""
    neighbors={}
    remaining={tuple(sorted(map(int,edge))) for edge in boundary}
    for a,b in remaining:
        neighbors.setdefault(a,[]).append(b);neighbors.setdefault(b,[]).append(a)
    if any(len(row)!=2 for row in neighbors.values()):
        raise ValueError('Selected patch has a branching boundary')
    loops=[]
    while remaining:
        start,current=min(remaining);previous=start;loop=[start]
        remaining.remove(tuple(sorted((start,current))))
        while current!=start:
            loop.append(current)
            following=next(v for v in neighbors[current] if v!=previous)
            remaining.remove(tuple(sorted((current,following))))
            previous,current=current,following
        points=vertices[loop]-vertices[loop[0]]
        area=abs(float(np.cross(points,np.roll(points,-1,axis=0)).sum(axis=0)@axis))/2
        loops.append((area,np.asarray(loop,dtype=int)))
    return max(loops,key=lambda row:row[0])[1],len(loops)


def verify(folder):
    folder=Path(folder);plan=json.loads((folder/'plan.json').read_text())
    # NpzFile access decompresses an entire member on every indexing call.
    # These immutable arrays are used thousands of times during FK replay.
    with np.load(folder/'trajectory.npz') as archive:
        a={name:archive[name] for name in archive.files}
    for p,digest in plan['sources'].items():
        if sha256(p)!=digest:raise ValueError('Changed motion source: '+p)
    panda=Franka();kuka=Arm(plan['work_robot_base_world_m'],np.eye(3),tool=plan['work_tool_length_from_link7_m'])
    assert (a['panda_q']>=panda.lower-1e-10).all() and (a['panda_q']<=panda.upper+1e-10).all()
    assert (a['kuka_q']>=kuka.lower-1e-10).all() and (a['kuka_q']<=kuka.upper+1e-10).all()
    segments=json.loads((folder/'segments.json').read_text())
    with np.load(folder/'scene_meshes.npz') as archive:
        source={name:archive[name] for name in archive.files}
    local=trimesh.Trimesh(source['object_v'],source['object_f'],process=False)
    errors=[];axes=[];attachment=[];active=[];outline_errors={};boundary_loops={}
    work_meshes={}
    # Independently check that the displayed contour follows the full boundary,
    # rather than accepting a small closed curve near the patch centre.
    for k in range(len(plan['object_poses'])):
        faces=source[f'patch_{k}_f']
        edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
        edges=np.sort(edges,axis=1);unique,count=np.unique(edges,axis=0,return_counts=True)
        boundary=unique[count==1]
        local_axis=np.asarray(plan['work_axis_world'])@np.asarray(plan['object_poses'][k])[:3,:3]
        outer,loop_count=outer_boundary(local.vertices,boundary,local_axis)
        boundary_loops[str(k)]=loop_count
        outer_edges=np.c_[outer,np.roll(outer,-1)]
        probes=np.concatenate([local.vertices[outer],local.vertices[outer_edges].mean(1)])
        outline=source[f'path_{k}_outline']
        np.testing.assert_allclose(outline[0],outline[-1],atol=1e-12,rtol=0)
        starts=outline[:-1];delta=np.diff(outline,axis=0);length2=np.sum(delta*delta,axis=1)
        relative=probes[:,None]-starts[None]
        u=np.clip(np.einsum('nsj,sj->ns',relative,delta)/np.maximum(length2,1e-25),0,1)
        distance=np.linalg.norm(relative-u[:,:,None]*delta,axis=2).min(1)
        outline_errors[str(k)]=float(distance.max())
        if distance.max()>.0006:raise ValueError('Work contour omitted part of the patch boundary')
        work=np.flatnonzero(a['work_task']==k);phases=a['work_phase'][work]
        if not (np.any(phases==1) and np.any(phases==2)) or np.any(np.diff(phases)<0):
            raise ValueError('Expected complete outline followed by interior strokes')
        progress=a['paint_progress'][:,k]
        if np.any(np.diff(progress)<0) or progress[-1]!=1:raise ValueError('Paint did not persist to the end')
    for frame,t in enumerate(a['times']):
        H=panda.fk(a['panda_q'][frame],a['gaps'][frame])
        for i,body in enumerate(plan['panda_bodies']):
            b=panda.model.body(body).id
            np.testing.assert_allclose(a['panda_links'][frame,i,:3,:3],panda.data.xmat[b].reshape(3,3),atol=1e-12,rtol=0)
            np.testing.assert_allclose(a['panda_links'][frame,i,:3,3],panda.data.xpos[b],atol=1e-12,rtol=0)
        i=min(np.searchsorted([s['end'] for s in segments],t,side='right'),len(segments)-1);seg=segments[i]
        if seg['moving']:
            expected=H@np.linalg.inv(seg['grasp'])
            actual=a['object_T' if seg['moving']=='object' else 'fixture_T'][frame]
            attachment.append(float(np.max(np.abs(expected-actual))))
        tcp,R,transforms=kuka.forward(a['kuka_q'][frame],geometry=True)
        for i,(r,p) in enumerate(transforms):
            np.testing.assert_allclose(a['kuka_links'][frame,i,:3,:3],r,atol=1e-12,rtol=0)
            np.testing.assert_allclose(a['kuka_links'][frame,i,:3,3],p,atol=1e-12,rtol=0)
        k=int(a['task'][frame])
        if k<0:continue
        active.append(k)
        np.testing.assert_allclose(a['object_T'][frame],plan['object_poses'][k],atol=1e-12,rtol=0)
        np.testing.assert_allclose(a['fixture_T'][frame],plan['fixture_poses'][k],atol=1e-12,rtol=0)
        Q=np.linalg.inv(a['object_T'][frame]);point=Q[:3,:3]@tcp+Q[:3,3]
        face=int(a['work_face'][frame]);tri=local.triangles[face]
        closest=trimesh.triangles.closest_point(tri[None],point[None])[0]
        errors.append(float(np.linalg.norm(point-closest)))
        axes.append(float(np.linalg.norm(R[:,2]-plan['work_axis_world'])))
        patch_faces=source[f'patch_{k}_f']
        if not np.any(np.all(patch_faces==local.faces[face],axis=1)):
            raise ValueError('Tool point outside selected work patch')
        # Check the actual preceding needle segment against the object, rather
        # than using the requested TCP alone as evidence of access.
        # Working frames above are independently checked against the same
        # saved U_k. Preserve that mesh's ray index across its work frames.
        if k not in work_meshes:
            world=local.copy();world.apply_transform(a['object_T'][frame])
            work_meshes[k]=world
        world=work_meshes[k]
        origin=tcp-R[:,2]*.155
        hit,_,_=world.ray.intersects_location([origin],[R[:,2]],multiple_hits=True)
        distances=(hit-origin)@R[:,2];positive=distances[distances>1e-6]
        if not len(positive) or abs(positive.min()-.155)>2e-5:
            raise ValueError('Needle first contact differs from the working TCP')
    if set(active)!=set(range(len(plan['object_poses']))):raise ValueError('Missing work pose')
    if max(attachment)>1e-10 or max(errors)>2e-5 or max(axes)>1e-3:raise ValueError('Motion replay failed')
    result=dict(status='passed',frames=len(a['times']),native_panda_FK_replayed=True,native_kuka_FK_replayed=True,
        moving_body_attachment_max_error=float(max(attachment)),work_TCP_mesh_distance_max_m=float(max(errors)),
        fixed_axis_max_vector_error=float(max(axes)),all_work_points_in_selected_faces=True,
        all_needle_first_contacts_verified=True,work_frames_per_pose={str(k):active.count(k) for k in range(len(plan['object_poses']))},
        complete_outer_boundary_verified=True,boundary_trace_max_inset_m=outline_errors,
        boundary_loop_count_per_pose=boundary_loops,interior_holes_are_not_outer_contours=True,
        work_mesh_acceleration_structure_count=len(work_meshes),
        outline_before_interior_verified=True,completed_pink_paint_persists=True,
        joint_position_limits_passed=True,full_arm_swept_collision_certified=False,
        trajectory_sha256=sha256(folder/'trajectory.npz'),plan_sha256=sha256(folder/'plan.json'))
    write_json(folder/'motion_validation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    print(json.dumps(verify(p.parse_args().folder),indent=2))
