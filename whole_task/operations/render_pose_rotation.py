"""Replay the saved pose rotation with the current Step1 tool and its direction axis."""
import json
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont

from whole_task.step1.common import ROOT,sha256,write_json
from whole_task.step1.find_tasks_figure import cone
from whole_task.step1.reach_video import tool_frame,HEAD,STEEL
from whole_task.step3.render_independent import Renderer,BLUE
from whole_task.step5.video import raster as r
from whole_task.common.poses import spin_pose
from .geometry import moved_mesh

HERE=Path(__file__).parent
DATA=ROOT/'whole_task/data/B/operation_videos'
VIEW=np.array([1.1,1.4,.95]);VIEW/=np.linalg.norm(VIEW)


def first_hit(mesh,origin,axis):
    T=mesh.triangles;e1=T[:,1]-T[:,0];e2=T[:,2]-T[:,0];P=np.cross(axis,e2)
    det=(e1*P).sum(1);valid=np.abs(det)>1e-12;inv=np.divide(1,det,out=np.zeros_like(det),where=valid)
    S=origin-T[:,0];u=(S*P).sum(1)*inv;Q=np.cross(S,e1);v=(Q*axis).sum(1)*inv;t=(e2*Q).sum(1)*inv
    t=np.where(valid&(u>=0)&(v>=0)&(u+v<=1)&(t>1e-9),t,np.inf)
    if not np.isfinite(t).any():raise ValueError('Rotation axis misses the object')
    return origin+float(t.min())*axis


def projection(points,bounds,width,height):
    right=np.cross([0,0,1],VIEW);right/=np.linalg.norm(right);up=np.cross(VIEW,right)
    B=np.array([right,up,VIEW]);projected=bounds@B.T;low,high=projected.min(0),projected.max(0)
    scale=max(high[0]-low[0],(high[1]-low[1])*width/height)*1.16
    P=np.asarray(points)@B.T;center=(low+high)/2
    return np.c_[width/2+(P[:,0]-center[0])/scale*width,height/2-(P[:,1]-center[1])/scale*width]


def annotate(image,start,end):
    draw=ImageDraw.Draw(image);start,end=np.asarray(start),np.asarray(end)
    delta=end-start;length=np.linalg.norm(delta);d=delta/length;n=np.array([-d[1],d[0]])
    color=(20,125,154)
    for t in np.arange(0,length-15,15):
        a=start+t*d;b=start+min(t+9,length-15)*d
        draw.line([tuple(a),tuple(b)],fill='white',width=5)
        draw.line([tuple(a),tuple(b)],fill=color,width=2)
    draw.polygon([tuple(end),tuple(end-15*d+6*n),tuple(end-15*d-6*n)],fill=color)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);label='Working direction'
    box=draw.textbbox((0,0),label,font=font);w,h=box[2]-box[0],box[3]-box[1]
    p=start+15*n+5*d;x=float(np.clip(p[0]-w/2,12,image.width-w-12));y=float(np.clip(p[1]-h,12,image.height-h-12))
    draw.rounded_rectangle((x-5,y-3,x+w+5,y+h+8),radius=4,fill='white')
    draw.text((x,y),label,font=font,fill=color)


def main():
    archive=ROOT/'whole_task/data/B/step3_connected_output_legacy/step3.2'
    report=json.loads((archive/'report.json').read_text());manifest=DATA/'videos.json'
    metadata=json.loads(manifest.read_text());base_path=DATA/'pose_shell/000.npz'
    base_meta=json.loads(base_path.with_suffix('.json').read_text())
    if sha256(base_path)!=base_meta['mesh_sha256']:raise ValueError('Saved rotation shell changed')
    if sha256(archive/'report.json')!=base_meta['input_step32_report_sha256']:raise ValueError('Archived source report changed')
    with np.load(archive/'initialization.npz') as z:
        source=trimesh.Trimesh(z['source_vertices_object'],z['source_faces'],process=False)
    with np.load(base_path) as z:
        initial=z['poses'][0];shell=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
        if not np.allclose(z['material'][0],np.eye(4)):raise ValueError('Expected the original world installation')
    records=[]
    for i in range(61):
        path=DATA/f'pose90/{i:03}.json';saved=json.loads(path.read_text())
        if saved['source_shell_sha256']!=sha256(base_path) or not saved['force']['passed']:
            raise ValueError('Saved rotation state is not the original checked state')
        records.append((path,saved))
    task=next(t for t in report['tasks'] if t['id']=='W02');mask=np.zeros(len(source.faces),bool);mask[task['working_face_ids']]=True
    setup_path=ROOT/'whole_task/data/B/step1_reach/plan.json';setup=json.loads(setup_path.read_text())['setup']
    L,R,N=setup['needle_length_m'],setup['head_radius_m'],setup['needle_radius_m']
    axis=np.array(report['source_workspace']['nominal_axis_world']);origin=np.array(report['source_workspace']['reference_head_front_world_m'])
    origin[2]=report['common_tool_head_height_m'];tip=first_hit(moved_mesh(source,initial),origin,axis)
    B=tool_frame(tip,axis);context=[]
    for v,f,color in [(*r.tool_mesh(0,L,N,N),STEEL),(*r.tool_mesh(L,L+.07,R,R),HEAD)]:
        context.append((trimesh.Trimesh(v@B[:3,:3].T+B[:3,3],f,process=False),color,None))
    # Same Step1 direction cone, anchored at the head's needle root; the rotating
    # object and shell are rigidly replayed, without rebuilding their geometry.
    lines=np.asarray(cone(tip-L*axis,axis,L,setup['max_tilt_deg'],8)).reshape(-1,3)
    marker=np.array([tip-(L+.12)*axis,tip+.17*axis])
    states=[];clouds=[lines,marker,*[m.vertices for m,_,_ in context]]
    angles=np.linspace(0.,180.,121);poses=[]
    for angle in angles:
        pose=spin_pose(initial,axis,origin,float(angle));poses.append(pose);delta=pose@np.linalg.inv(initial)
        body=moved_mesh(source,pose);support=moved_mesh(shell,delta)
        if np.linalg.norm(first_hit(body,origin,axis)-tip)>1e-8:raise ValueError('Rotation changed the needle contact')
        states.append((body,support));clouds.extend([body.vertices,support.vertices])
    points=np.vstack(clouds);width=height=720;fps=24;count=192
    marker_pixels=projection(marker,points,width,height)
    renderer=Renderer(width,height);rendered=[]
    try:
        for i,(body,support) in enumerate(states):
            image=renderer.draw([(support,BLUE,None),(body,r.GRAY,mask),*context],points,VIEW,lines)
            annotate(image,*marker_pixels);rendered.append(image)
            if i%15==0:print('POSE ROTATION FRAME',i,'/',len(states),flush=True)
    finally:renderer.close()
    output=HERE/'pose_rotation.mp4';temporary=DATA/'pose_rotation_updated.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE)
    try:
        for frame in range(count):
            alpha=np.clip((frame-24)/(count-49),0,1);index=min(len(rendered)-1,round(alpha*(len(rendered)-1)))
            process.stdin.write(rendered[index].tobytes())
    finally:process.stdin.close();code=process.wait()
    if code:raise RuntimeError('Video encoding failed')
    subprocess.run(['ffmpeg','-v','error','-i',str(temporary),'-f','null','-'],check=True)
    temporary.replace(output)
    np.savez_compressed(DATA/'pose_rotation_motion.npz',angles_deg=angles,poses=np.asarray(poses),axis=axis,pivot=origin)
    rendered[0].save(DATA/'pose_rotation_first.png');rendered[-1].save(DATA/'pose_rotation_last.png')
    old=metadata['pose_rotation.mp4'];metadata['pose_rotation.mp4']=dict(old,video_sha256=sha256(output),
        working_tool='current Step1 black head and steel needle',tool_setup_sha256=sha256(setup_path),
        needle_length_m=L,needle_radius_m=N,head_radius_m=R,head_length_m=.07,
        tool_tip_world_m=tip.tolist(),tool_root_world_m=(tip-L*axis).tolist(),
        direction_axis_world=axis.tolist(),working_direction_axis_annotated=True,
        gold_cone_apex='head_front_needle_root',
        original_rotation_states_preserved=False,original_body_and_shell_preserved=True,
        rotation_extended_from_deg=90.,rotation_deg=180.,states=len(states),width=width,height=height,frames=count,fps=fps,
        annotation_text='Working direction',annotation_language='en',
        new_force_certification_claimed=False,
        annotation_only_text=True,text_overlays=True,source_sha256=sha256(Path(__file__)))
    write_json(manifest,metadata)
    write_json(DATA/'pose_rotation_render.json',dict(**metadata['pose_rotation.mp4'],
        archive_report_sha256=sha256(archive/'report.json'),shell_sha256=sha256(base_path),
        pose_proof_sha256={str(p.relative_to(ROOT)):sha256(p) for p,_ in records},
        tool_contact_on_surface_all_states=True,
        original_90_degree_parent_proofs_retained=True,
        motion_sha256=sha256(DATA/'pose_rotation_motion.npz')))
    print('POSE ROTATION SAVED',output,flush=True)
    return output


if __name__=='__main__':main()
