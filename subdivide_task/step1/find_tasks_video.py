"""Ten-second close-up: the needle traces the task's closed 3D surface boundary."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .common import ROOT,load_shape,sha256,write_json
from .arm import Check
from .reach import clearance,tip_contact
from .reach_video import Scene,tool_frame,kuka,GOLD
from .find_tasks_figure import VIEW,cone,tubes,dashed,Opacity,CONE_OPACITY,POSE_INDEX,placement,tabletop,task_outline,OUTLINE_COLOR,OUTLINE_WIDTH_PX,OUTLINE_OFFSET_M,workspace_box,WORKSPACE_COLOR
from .workspace import points_inside_box
from .task_boundary import boundary_loops,sample_loop,candidate_region

HERE=Path(__file__).parent
TRACE_COLOR=(.74,.15,.24)
TRACE_OFFSET_M=2*OUTLINE_OFFSET_M


def contact(triangles,origin,direction):
    """First actual surface point and its source face along one moving needle line."""
    T=triangles;e1=T[:,1]-T[:,0];e2=T[:,2]-T[:,0];P=np.cross(direction,e2)
    det=(e1*P).sum(1);valid=np.abs(det)>1e-12
    inv=np.divide(1,det,out=np.zeros_like(det),where=valid)
    S=origin-T[:,0];u=(S*P).sum(1)*inv;Q=np.cross(S,e1);v=(Q*direction).sum(1)*inv;t=(e2*Q).sum(1)*inv
    # Adjacent triangles share an exact edge; roundoff must not drop both and
    # incorrectly report a far-side surface as the first hit.
    tolerance=1e-9
    t=np.where(valid&(u>=-tolerance)&(v>=-tolerance)&(u+v<=1+tolerance)&(t>1e-9),t,np.inf)
    f=int(t.argmin())
    if not np.isfinite(t[f]):raise ValueError('The moving head misses the object')
    return f,origin+t[f]*direction


def link_matrices(transforms,shift):
    matrices=np.repeat(np.eye(4)[None],len(transforms),axis=0)
    for M,(rotation,point) in zip(matrices,transforms):M[:3,:3]=rotation;M[:3,3]=point-shift
    return matrices


def camera(scene,points,margin):
    scene.fit(points,margin)
    return scene.target.copy(),2/np.linalg.norm(scene.vp[0,:3])


def set_camera(scene,target,scale):
    width,height=scene.size;r=scene.r
    scene.target=target;scene.vp=r.orthographic(scale,scale*height/width)@r.view_matrix(scene.view,target)
    for program in (scene.shader,scene.line_shader):program['view_projection'].write(np.asarray(scene.vp,'f4').T.tobytes())
    scene.shader['eye'].value=tuple(target+scene.view*4)


def render(object_name='B',output=None,fps=24,width=1280,height=960):
    start=time.monotonic();work=ROOT/'subdivide_task/data'/object_name/'step1_reach'
    plan=json.loads((work/'plan.json').read_text());s=plan['setup']
    with np.load(work/'plan.npz') as z:arrays={k:z[k] for k in z.files}
    mesh=load_shape(ROOT/'objects'/object_name/'mesh.stl')
    j=POSE_INDEX
    actual,W,shift=placement(plan,mesh)
    vertices=mesh.vertices@W[:3,:3].T+W[:3,3];triangles=vertices[mesh.faces]
    normals=mesh.face_normals@W[:3,:3].T
    a=np.asarray(s['working_direction_world']);L=s['needle_length_m'];beta=s['max_tilt_deg']
    region=candidate_region(arrays,j)
    loops=boundary_loops(mesh,region)
    if len(loops)!=1:raise ValueError('This video needs one closed task boundary; cannot omit other contours')
    loop=loops[0];times=np.arange(10*fps)/fps
    path=sample_loop(mesh,loop,arrays['reach_axis'][:,j],len(times)-1)
    anchor_faces=loop['face_ids'];first=int(anchor_faces[0])
    if not (arrays['reach'][anchor_faces,j]&region[anchor_faces]).all():
        raise ValueError('Pose1 motion anchors must belong to its displayed certified candidate region')
    anchor_axes=arrays['reach_axis'][anchor_faces,j]@W[:3,:3].T
    targets=path['points']@W[:3,:3].T+W[:3,3]
    directions=path['axes']@W[:3,:3].T
    # Render-only offset follows the same surface contour; physical tips stay on the edge.
    outline_vertices=mesh.vertices+TRACE_OFFSET_M*mesh.vertex_normals
    contour=outline_vertices[loop['vertex_ids']]@W[:3,:3].T+W[:3,3]
    e=path['edge_indices'];u=path['edge_parameters'][:,None]
    trace=(1-u)*contour[e]+u*contour[e+1]
    tips=[];axes=[];roots=[];faces=[];first_hits=[];Qs=[];links=[];gaps=[]
    arm=kuka(s);checker=Check(mesh,s)
    vertex_slot=int(np.flatnonzero(mesh.faces[first]==loop['vertex_ids'][0])[0])
    q=arrays['joints'][first,j,vertex_slot].copy()
    for f in np.unique(anchor_faces):
        if not checker.replay(f,arrays['reach_axis'][f,j],actual,arrays['joints'][f,j]):
            raise ValueError(f'Pose1 motion anchor {f} fails witness replay')
    max_position_error=max_axis_error=0.;previous=None;table_gaps=[]
    eps=plan['reach_definition']['needle_start_m']
    first_hit_error=0.
    for i,(t,target,d,f) in enumerate(zip(times,targets,directions,path['faces'])):
        hit,intersection=contact(triangles,target-.4*d,d)
        error=float(np.linalg.norm(intersection-target));first_hit_error=max(first_hit_error,error)
        if error>1e-7:raise ValueError(f'Frame {i} boundary point is hidden behind another surface')
        tip=target;root=tip-L*d
        if not (region[f] and arrays['reach'][f,j]):
            raise ValueError(f'Frame {i} leaves pose1 retained certified reach')
        if not points_inside_box(tip+shift,s['surface_workspace']['bounds_world_m']):
            raise ValueError(f'Frame {i} surface contact leaves the inward work box')
        cos=-float(normals[f]@d)
        line=float(clearance(triangles,tip[None],d,eps)[0])
        body=float(clearance(triangles,tip[None],d,float(tip_contact(s['needle_radius_m'],cos)))[0])
        head=float(clearance(triangles,tip[None],d,L)[0])
        if cos<=0 or line<=0 or body<s['needle_radius_m']-1e-10 or head<s['head_radius_m']-1e-10:
            raise ValueError(f'Tool hits the object at moving contact {i}')
        radial_z=np.sqrt(max(0.,1-d[2]**2));table_z=s['table_height_m']-shift[2]
        table_gap=min(tip[2]-float(tip_contact(s['needle_radius_m'],cos))*d[2]-s['needle_radius_m']*radial_z,
                      root[2]-s['head_radius_m']*radial_z)-table_z
        if d[2]>=0 or table_gap < -1e-9:raise ValueError(f'Tool hits the table at frame {i}')
        table_gaps.append(table_gap)
        # Every cone is rigidly translated by this head's root. Its nominal axis
        # stays fixed; the actual head/needle moves inside that allowed cone.
        if previous is None or np.linalg.norm(np.r_[tip,d]-previous)>1e-12:
            solved=checker.solve(tip+shift,d,actual,q)
            if solved is None or solved is False:raise ValueError(f'Arm cannot follow the head at frame {i}')
            q=solved
        tcp,axis,_,_=arm.forward(q)
        if not points_inside_box(tcp,s['surface_workspace']['bounds_world_m']):
            raise ValueError(f'Frame {i} actual process TCP leaves the inward work box')
        pe=float(np.linalg.norm(tcp-tip-shift));ae=float(np.linalg.norm(axis-d))
        max_position_error=max(max_position_error,pe);max_axis_error=max(max_axis_error,ae)
        if pe>1e-4 or ae>1e-3:raise ValueError('Arm/tool mismatch')
        previous=np.r_[tip,d];tips.append(tip);axes.append(d);roots.append(root);faces.append(f);first_hits.append(hit);Qs.append(q.copy());gaps.append((line,body,head))
        links.append(link_matrices(arm.forward(q,geometry=True)[2],shift))
        if i%60==0:print('MOVING HEAD CHECK',i,'/',len(times),flush=True)
    tips,axes,roots,links,Qs=map(np.asarray,(tips,axes,roots,links,Qs))
    tilts=np.degrees(np.arccos(np.clip(axes@a,-1,1)))
    if tilts.max()>beta:raise ValueError('Head leaves its allowed cone')
    np.savez_compressed(work/'find_tasks_motion.npz',times=times,axes_world=axes[:,None],tips=tips,
        head_roots=roots,cone_apexes=roots,cone_axes=np.broadcast_to(a,axes.shape),faces=np.asarray(faces),
        joints=Qs,links=links,object_pose=W,object_pose_world=actual,render_shift=shift,
        pose_index=j,anchor_faces=anchor_faces,anchor_axes_world=anchor_axes,
        anchor_joints=arrays['joints'][anchor_faces,j],first_hit_faces=np.asarray(first_hits),
        boundary_vertex_ids=loop['vertex_ids'],boundary_edge_face_ids=loop['face_ids'],
        boundary_edge_indices=path['edge_indices'],boundary_edge_parameters=path['edge_parameters'],
        boundary_points_object=path['points'],trace_points=trace,
        tool_gaps=np.asarray(gaps),table_gaps=np.asarray(table_gaps))
    print('MOVING HEAD all poses checked',flush=True)

    scene=Scene(mesh,s,width,height,arm=True);scene.view=VIEW.copy();r=scene.r
    box,box_points=workspace_box(scene,s,shift)
    table=tabletop(scene,s,shift)+box
    colors=np.tile(r.GRAY,(len(mesh.faces),1));colors[region]=r.PINK
    outline=task_outline(mesh,region,W)
    # One translucent dashed cone travels with the real, opaque head.
    wire=cone(np.zeros(3),a,L,beta,8)
    cone_shape=r.Shape(scene.ctx,scene.shader,*tubes(dashed(wire),.00065),GOLD,metallic=.1,roughness=.5)
    close_points=list(vertices)+list(box_points)
    for tip,d,root in zip(tips,axes,roots):
        close_points.extend(root+p for segment in wire for p in segment)
        B=tool_frame(tip,d)[:3,:3]
        for z in (L,L+.07):
            close_points.extend(tip+B@np.array([x,y,z]) for x in (-s['head_radius_m'],s['head_radius_m']) for y in (-s['head_radius_m'],s['head_radius_m']))
    close_points=np.asarray(close_points);close_target,close_scale=camera(scene,close_points,1.12)
    camera_targets=[];camera_scales=[];compositor=Opacity(scene)
    output=Path(output or ROOT/'subdivide_task/idea/find_tasks.mp4');output.parent.mkdir(parents=True,exist_ok=True)
    temporary=work/'find_tasks_rendering.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r',str(fps),'-i','-','-vf','vflip','-c:v','libx264','-preset','fast','-crf','18','-threads','4',
        '-pix_fmt','yuv420p','-movflags','+faststart','-an',str(temporary)],stdin=subprocess.PIPE)
    try:
        for i,(t,tip,d,root,matrices) in enumerate(zip(times,tips,axes,roots,links)):
            target=close_target;scale=close_scale
            set_camera(scene,target,scale);camera_targets.append(target);camera_scales.append(scale)
            C=np.eye(4);C[:3,3]=root
            shapes=table+[(part,tool_frame(tip,d)) for part in scene.tools[False]]
            segments=list(outline)
            if i:segments.append((np.stack([trace[:i],trace[1:i+1]],axis=1).reshape(-1,3),TRACE_COLOR))
            data=compositor.draw(W,colors,shapes,matrices,[],0.,[(cone_shape,C)],CONE_OPACITY,segments=segments);process.stdin.write(data)
            if i in (0,48,96,144,192,len(times)-1):
                Image.frombytes('RGB',scene.size,data).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(work/f'find_tasks_frame_{i:03d}.png')
            if i%48==0:print('MOVING HEAD FRAME',i,'/',len(times),flush=True)
    finally:
        process.stdin.close();code=process.wait();compositor.close();scene.close()
    if code:raise RuntimeError('Video encoding failed')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_read_frames,r_frame_rate:format=duration','-of','json',str(temporary)]))
    if int(probe['streams'][0]['nb_read_frames'])!=len(times) or abs(float(probe['format']['duration'])-10.)>1/fps:
        raise ValueError('Video duration is not ten seconds')
    temporary.replace(output)
    np.savez_compressed(work/'find_tasks_camera.npz',times=times,targets=np.asarray(camera_targets),scales=np.asarray(camera_scales))
    record=dict(file=str(output),sha256=sha256(output),source_sha256=sha256(Path(__file__)),
        figure_source_sha256=sha256(HERE/'find_tasks_figure.py'),plan_sha256=sha256(work/'plan.json'),
        plan_arrays_sha256=sha256(work/'plan.npz'),task=plan['areas'][j]['id'],pose_number=1,pose_index=j,
        placement=plan['areas'][j]['placement'],table_height_m=s['table_height_m'],
        surface='pose1 candidate region: angle reachable surface intersect 3D work box',surface_workspace=s['surface_workspace'],
        candidate_face_count=int(region.sum()),
        candidate_area_fraction_of_target=float(mesh.area_faces[region].sum()/mesh.area_faces[arrays['target']].sum()),
        downstream_tasks='exclusive face_task partition unchanged',
        workspace_box_color=list(WORKSPACE_COLOR),workspace_box_edges=12,
        all_surface_contact_points_inside_fixed_xyz_box=True,
        task_outline=dict(color_rgb=OUTLINE_COLOR,width_px=OUTLINE_WIDTH_PX,
                          boundary_edges=sum(len(points)//2 for points,_ in outline),
                          vertex_normal_offset_m=OUTLINE_OFFSET_M,visible_surfaces_only=True),
        task_area_fraction_of_target=plan['areas'][j]['area_fraction_of_target'],
        duration_s=10.,fps=fps,frames=len(times),width=width,height=height,zoom_out=False,camera_mode='fixed_close_up',
        description='needle follows every edge of the closed 3D pink-task boundary; fixed depth-tested XYZ box; translated gold direction cone',
        motion_mode='closed_3d_task_boundary',boundary_loops=1,boundary_edges=len(anchor_faces),
        boundary_length_m=loop['length_m'],boundary_closed=bool(np.linalg.norm(tips[0]-tips[-1])<1e-10),
        trace_color=list(TRACE_COLOR),trace_surface_offset_m=TRACE_OFFSET_M,
        all_boundary_edges_visited=bool(set(path['edge_indices'])==set(range(len(anchor_faces)))),
        first_surface_hit_error_m=first_hit_error,
        workspace_rendering='depth-tested 3D edges; rear edges occluded by the object',
        motion_anchor_faces=anchor_faces.tolist(),motion_axis_provenance='continuous blends of neighbouring boundary faces stored axes; every surface contact and blended axis re-certified',
        contacts_in_displayed_certified_candidate_region=True,
        head_opacities=[1.],cone_styles=['dashed'],cone_opacities=[CONE_OPACITY],cone_moves_with_head=True,
        head_translation_span_m=np.ptp(roots,axis=0).tolist(),max_tilts_deg=[float(tilts.max())],
        tool_pose_checks=len(times),arm_pose_checks=len(times),max_tcp_position_error_m=max_position_error,
        max_tcp_axis_error=max_axis_error,max_joint_step_deg=float(np.degrees(np.abs(np.diff(Qs,axis=0))).max()),
        minimum_tool_table_gap_m=float(min(table_gaps)),
        motion_sha256=sha256(work/'find_tasks_motion.npz'),
        camera_sha256=sha256(work/'find_tasks_camera.npz'),elapsed_s=time.monotonic()-start,probe=probe)
    write_json(work/'find_tasks_video.json',record)
    print('MOVING HEAD VIDEO SAVED',output,flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--object',default='B')
    parser.add_argument('--output',type=Path);args=parser.parse_args();render(args.object,args.output)
