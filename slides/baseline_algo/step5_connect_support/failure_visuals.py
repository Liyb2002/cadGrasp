"""Save actual failed material, collision witnesses, images and an offline viewer.

Diagnostic motions intentionally include rejected directions. No diagnostic
frame, MP4 or STL is presented as a certified installable support.
"""
import argparse
import html
import json
from pathlib import Path
import sys
import numpy as np
import trimesh
from PIL import Image,ImageDraw
import imageio.v2 as imageio
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step1.needs import ContinuousNeeds,OUTPUTS,sha256
from step1.cases import pose_name,selected_pose
from step3_scheculer import contacts as I
from step5_connect_support import belt_geometry as B, direction_first as X, solids as S, visual_details as V

RED=np.array([208.,49.,56.]);PURPLE='#8b4599'
STAGES=['heads','loose_frame','base','thick_links','whole_path','bearing']
REASONS={
 'no_selected_heads':'Step3 没有选出满足硬约束的第一块头；尚未构造支撑，也未测试轨迹。',
 'no_common_translation_proposed_for_fixed_heads':'固定头没有找到共同平移方向；以下动作展示被拒绝的方向。',
 'fixed_heads_block_all_proposed_sweeps':'头在完整退出过程中碰撞；框架和底座尚未构造。',
 'no_loose_frame_assembly_in_directional_menu':'已尝试连接材料，但未得到通过全部检查的整件。',
 'step3_incomplete':'Step3 数值求解未完成；显示最后已保存的部分接触，尚未执行 Step5 构造。'}


def pack_mesh(mesh):
    return dict(v=(mesh.vertices*1000).tolist(),f=mesh.faces.tolist())


def collision_mesh(scene,part):
    solid=scene.solid ^ B.G.solid64(part,scene.origin,scene.scale)
    data=solid.to_mesh64()
    return trimesh.Trimesh(np.asarray(data.vert_properties[:,:3])*scene.scale+scene.origin,
                           np.asarray(data.tri_verts),process=False)


def find_event(domain,parts,labels,direction,known=None):
    """Actual displaced-solid/floor witnesses; samples never accept a path."""
    scene=B.Scene(domain.mesh);d=np.asarray(direction,float);d=d/max(np.linalg.norm(d),1e-30)
    distances=([known['withdrawal_distance_m']] if known else [])
    distances += [0.]
    distances += (scene.scale*np.array([.001,.003,.01,.025,.05,.1,.2,.4,.8,1.6])).tolist()
    hit=None;last=None
    for distance in distances:
        shifted=[];bad=[];floor=[];volumes=[]
        for i,p in enumerate(parts):
            moved=p.copy();moved.vertices=p.vertices+distance*d
            volume=scene.volume(moved)
            if volume>1e-11*scene.scale**3:bad.append(i);volumes.append((volume,i,moved))
            if moved.vertices[:,2].min() < -1e-10*scene.scale:floor.append(i)
        last=distance
        if bad or floor:
            volume,index,moved=max(volumes,key=lambda row:row[0]) if volumes else (0.,floor[0],None)
            inter=collision_mesh(scene,moved) if moved is not None else None
            focus=inter.bounds.mean(axis=0) if inter is not None and len(inter.vertices) else parts[index].vertices.mean(axis=0)+distance*d
            hit=dict(direction=d.tolist(),translation_m=(distance*d).tolist(),distance_m=distance,
                colliding_parts=sorted(set(bad+floor)),object_collision_parts=bad,floor_collision_parts=floor,
                witness_part=index,witness_label=labels[index],intersection_volume_m3=float(volume),
                witness_kind='actual_object_intersection' if bad else 'actual_floor_penetration',
                focus_m=focus.tolist(),focus_width_m=max(.12*scene.scale,float(np.ptp(parts[index].vertices,axis=0).max())*3),
                accepted_trajectory=False,samples_used_for_acceptance=False)
            return hit,inter
    center=np.vstack([p.vertices for p in parts]).mean(axis=0) if parts else domain.com
    return dict(direction=d.tolist(),translation_m=(d*.02*scene.scale).tolist(),distance_m=.02*scene.scale,
        colliding_parts=[],object_collision_parts=[],floor_collision_parts=[],witness_part=None,witness_label=None,
        intersection_volume_m3=0.,witness_kind='no_sampled_penetration_witness',focus_m=np.asarray(center).tolist(),
        focus_width_m=.4*scene.scale,accepted_trajectory=False,samples_used_for_acceptance=False),None


def clearance_event(domain,parts,labels,direction,gap,index):
    """Exhibit the reserved clearance volume separately from actual material."""
    padded=X.D.engine.hull_mesh((parts[index].vertices[:,None,:]+gap*X.T.CORNERS).reshape(-1,3))
    event,intersection=find_event(domain,[padded],[labels[index]],direction)
    if not event['object_collision_parts']: return None
    event.update(colliding_parts=[index],object_collision_parts=[],floor_collision_parts=[],
        witness_part=index,witness_label=labels[index],witness_kind='reserved_clearance_intersection',
        clearance_overlap_m3=event['intersection_volume_m3'],intersection_volume_m3=0.,
        reserved_clearance_m=gap,clearance_envelope_not_actual_material=True)
    return event,intersection


def diagnostic_directions(domain,contacts,geometry):
    trials=[]
    for i,a in enumerate(geometry.get('attempts',[])):
        trials.append((f'Tested direction {i+1}',np.asarray(a['direction']),a.get('collision_witness')))
    if trials:return trials
    preferred=X.preferred_direction(domain.mesh,domain.work_ids)
    seeds=[('Working-face back',preferred),('Horizontal back',COORD.lift_floor(COORD.floor(preferred)))]
    for c in contacts:
        n=domain.mesh.face_normals[c['source_faces']]
        seeds.append((f'Away from {c["candidate_id"]}',(n*c['triangle_areas_m2'][:,None]).sum(axis=0)))
    seeds += [('World +X',np.array([1.,0,0])),('World -X',np.array([-1.,0,0])),('World +Z',np.array([0.,0,1.]))]
    for label,d in seeds:
        if np.linalg.norm(d)<1e-10:continue
        d=d/np.linalg.norm(d)
        if all(np.linalg.norm(d-t[1])>.02 for t in trials):trials.append((label,d,None))
        if len(trials)>=6:break
    return trials


def overlay_mesh(picture,mesh,view,size,color=RED):
    if mesh is None or not len(mesh.faces):return
    tri=mesh.triangles;layer,ids=V.render((tri,np.tile(color,(len(tri),1)),[],range(len(tri))),view,size)
    picture.paste(layer,mask=Image.fromarray(np.uint8(ids>=0)*255))


def picture(domain,parts,labels,event,intersection,view,size,amount=1.,show_object=True):
    moved=[]
    for p in parts:
        q=p.copy();q.vertices=p.vertices+amount*np.asarray(event['translation_m']);moved.append(q)
    joined=trimesh.util.concatenate(moved)
    data=S.pack_parts(moved,labels,joined)
    image=V.scene(domain,[],[(dict(candidate_id='diagnostic'),data)],size,view=view,
                  xray=True,show_object=show_object,labels=False)
    if amount>=1.-1e-8:
        for index in event['colliding_parts']:overlay_mesh(image,moved[index],view,size,np.array([198.,92.,76.]))
        overlay_mesh(image,intersection,view,size,np.array([231.,151.,35.]) if event.get('clearance_envelope_not_actual_material') else RED)
    center=np.vstack([p.vertices for p in parts]).mean(axis=0)
    focus,width,basis=view;tail=V.R.project(center,focus,basis,width,size)[:2]
    tip=V.R.project(center+.16*float(domain.mesh.extents.max())*np.asarray(event['direction']),focus,basis,width,size)[:2]
    V.arrow(ImageDraw.Draw(image),tail,tip,color=PURPLE,width=4)
    return image


def stage_material(labels):
    if any(l.startswith('connector_') for l in labels):return 'thick_links'
    if any(l.startswith('ground_') for l in labels):return 'base'
    if any(l.startswith(('frame_','neck_')) for l in labels):return 'loose_frame'
    return 'heads'


def render(name,domain,contacts,parts,labels,geometry,status,out,static_only=False,inputs=(),partial=False):
    out.mkdir(parents=True,exist_ok=True)
    for file in out.iterdir():
        if file.is_file() and file.name.startswith(('failure','failed_shape','workpiece_mm')):file.unlink()
    stage=stage_material(labels)
    motion_verified=bool(geometry.get('trajectory',{}).get('passed'))
    note=('这是已保存的部分 Step3 接触，不是完整支撑。' if partial else
          '当前实际材料只有接触头；框架、底座、连接杆尚未构造。' if stage=='heads' else
          '这是实际尝试过的失败材料，可能未连通；不是可安装成品。')
    if motion_verified:
        note='整件轨迹已通过；完整设计未通过承载或上游覆盖验证。装入动作请看 insertion.mp4。'
    elif geometry.get('retained_failure'):
        note+=' 停止原因：'+geometry['retained_failure']['status']
    joined=trimesh.util.concatenate(parts)
    arrays=S.pack_parts(parts,labels,joined);np.savez_compressed(out/'failed_shape.npz',**arrays)
    joined.export(out/'failed_shape.stl',file_type='stl_ascii');mm=joined.copy();mm.apply_scale(1000);mm.export(out/'failed_shape_mm.stl',file_type='stl_ascii')
    obj=domain.mesh.copy();obj.apply_scale(1000);obj.export(out/'workpiece_mm.stl',file_type='stl_ascii')
    trials=diagnostic_directions(domain,contacts,geometry) if not partial and not motion_verified else [('Bearing / upstream coverage' if motion_verified else 'Upstream stopped',np.zeros(3),None)]
    events=[];intersections=[]
    for i,(label,d,known) in enumerate(trials):
        if partial or motion_verified:
            e=dict(direction=[0.,0.,0.],translation_m=[0.,0.,0.],distance_m=0.,colliding_parts=[],object_collision_parts=[],floor_collision_parts=[],witness_part=None,witness_label=None,intersection_volume_m3=0.,witness_kind='trajectory_passed_design_not_certified' if motion_verified else 'upstream_incomplete_no_motion_tested',focus_m=domain.com.tolist(),focus_width_m=float(domain.mesh.extents.max()),accepted_trajectory=False,samples_used_for_acceptance=False);inter=None
        else:
            e,inter=find_event(domain,parts,labels,d,known)
            retained=geometry.get('retained_failure',{})
            if not e['colliding_parts'] and retained.get('clearance_m',0.)>0:
                witness=clearance_event(domain,parts,labels,d,retained['clearance_m'],len(parts)-1)
                if witness is not None:e,inter=witness
        blockers=[]
        for c in contacts:
            value=float(np.min(domain.mesh.face_normals[c['source_faces']]@d))
            if value < -X.NORMAL_TOLERANCE:blockers.append(dict(candidate_id=c['candidate_id'],minimum_normal_dot=value))
        e.update(name=label,normal_constraint_blockers=blockers,direction_was_search_candidate=bool(geometry.get('attempts')),
                 floor_direction_constraint_violated=bool(d[2]<-1e-10),material_stage=stage)
        if inter is not None and len(inter.faces):
            filename=f'failure_collision_{i+1:02d}_mm.stl';copy=inter.copy();copy.apply_scale(1000);copy.export(out/filename,file_type='stl_ascii');e['collision_mesh']=filename
        events.append(e);intersections.append(inter)
    primary=next((i for i,e in enumerate(events) if e['colliding_parts']),0)
    event=events[primary];inter=intersections[primary]
    failed_pose=joined.copy();failed_pose.vertices=(failed_pose.vertices+np.asarray(event['translation_m']))*1000
    failed_pose.export(out/'failure_pose_mm.stl',file_type='stl_ascii')
    # A view roughly perpendicular to the motion exposes the displacement.
    d=np.asarray(event['direction']);h=np.linalg.norm(COORD.floor(d));sight=np.r_[d[1],-d[0],.6*h] if h>1e-5 else np.array([.68,-1.,.65])
    basis=V.R.axes(sight);points=np.vstack([domain.mesh.vertices,joined.vertices,joined.vertices+np.asarray(event['translation_m'])])
    view=V.camera(domain,[],basis=basis,points=points)
    close=V.fit(np.vstack([parts[i].vertices+np.asarray(event['translation_m']) for i in event['colliding_parts']]) if event['colliding_parts'] else joined.vertices,basis,margin=1.9)
    page=Image.new('RGB',(2100,920),V.R.PAPER);ink=ImageDraw.Draw(page)
    ink.text((22,16),f'{name} / {pose_name()} / FAILURE INSPECTION',font=V.R.font(32),fill=V.R.INK)
    subtitles=['Actual attempted material / final pose','Rejected withdrawal / object transparent','Collision close-up / red = intersection']
    if event.get('clearance_envelope_not_actual_material'):
        subtitles=['Actual attempted material / final pose','Reserved clearance fails during withdrawal','Orange: clearance envelope enters object']
    if motion_verified: subtitles=['Constructed material / final pose','Trajectory passed; design not certified','Inspect geometry; see bearing report']
    panels=[picture(domain,parts,labels,event,inter,view,690,amount=0.),picture(domain,parts,labels,event,inter,view,690),picture(domain,parts,labels,event,inter,close,690)]
    for i,panel in enumerate(panels):page.paste(panel,(i*700,105));ink.text((i*700+18,70),subtitles[i],font=V.R.font(20),fill=V.MUTED)
    detail=f"{event['name']} | move {event['distance_m']*1000:.3f} mm | overlap {event['intersection_volume_m3']*1e9:.6g} mm^3 | {event['witness_label'] or 'no collision witness'}"
    if event.get('clearance_envelope_not_actual_material'):
        detail=f"Reserved gap {event['reserved_clearance_m']*1000:.3f} mm fails; orange is clearance envelope, not material collision."
    ink.text((22,806),detail,font=V.R.font(25),fill='#a33336')
    ink.text((22,846),f"Material stage: {stage}. Frame/base not constructed." if stage=='heads' else f'Material stage: {stage}; attempted shape, not a successful design.',font=V.R.font(25),fill=V.R.INK)
    ink.text((22,884),'Trajectory verified; design not certified. See insertion.mp4 and connection.json.' if motion_verified else 'Diagnostic withdrawal only; insertion reverses the same motion. No successful trajectory is claimed.',font=V.R.font(20),fill=V.MUTED)
    page.save(out/'failure.png')
    cols=3;rows=int(np.ceil(len(events)/cols));sheet=Image.new('RGB',(1800,rows*680+75),V.R.PAPER);ink=ImageDraw.Draw(sheet)
    ink.text((20,15),f'{name} / {pose_name()} / '+('DESIGN VERIFICATION INCOMPLETE' if motion_verified else 'REJECTED DIRECTIONS'),font=V.R.font(30),fill=V.R.INK)
    for i,(e,inter) in enumerate(zip(events,intersections)):
        x=(i%cols)*600;y=75+(i//cols)*680
        sheet.paste(picture(domain,parts,labels,e,inter,view,590),(x,y+42))
        ink.text((x+12,y),e['name'],font=V.R.font(23),fill=V.R.INK)
        ink.text((x+12,y+615),f"Move {e['distance_m']*1000:.3f} mm; overlap {e['intersection_volume_m3']*1e9:.5g} mm^3",font=V.R.font(20),fill='#a33336')
        blockers=', '.join(c['candidate_id'] for c in e['normal_constraint_blockers']) or 'none locally'
        ink.text((x+12,y+645),'Opposing heads: '+blockers,font=V.R.font(18),fill=V.MUTED)
    sheet.save(out/'failure_directions.png')
    if not static_only and not partial and not motion_verified:
        frames=[]
        for t in np.linspace(0.,1.,36):
            frame=Image.new('RGB',(1280,760),V.R.PAPER);ink=ImageDraw.Draw(frame)
            ink.text((20,10),f'{name} / {pose_name()} / FAILED withdrawal, not an insertion solution',font=V.R.font(24),fill='#a33336')
            frame.paste(picture(domain,parts,labels,event,intersections[primary],view,635,amount=t),(0,70))
            frame.paste(picture(domain,parts,labels,event,intersections[primary],close,635,amount=t),(640,70))
            ink.text((20,45),'Full object and attempted material',font=V.R.font(20),fill=V.MUTED);ink.text((660,45),'Orange: reserved clearance conflict' if event.get('clearance_envelope_not_actual_material') else 'Close-up at the eventual collision',font=V.R.font(20),fill=V.MUTED)
            ink.text((20,710),f"Withdrawn {event['distance_m']*1000*t:.3f} mm / {event['name']} / material: {stage}",font=V.R.font(22),fill=V.R.INK)
            frames.append(frame)
        frames=[frames[0]]*8+frames+[frames[-1]]*20
        with imageio.get_writer(out/'failure.mp4',fps=12,codec='libx264',quality=None,ffmpeg_params=['-crf','0','-vf','scale=in_range=full:out_range=full','-color_range','pc'],macro_block_size=1) as writer:
            for frame in frames:writer.append_data(np.asarray(frame))
        frames[0].save(out/'failure.gif',save_all=True,append_images=frames[1:],duration=83,loop=0)
    # Self-contained canvas viewer: no network, external library or local server.
    object_mesh=pack_mesh(domain.mesh);palette=np.tile([185,190,189],(len(domain.mesh.faces),1));palette[domain.work_ids]=[149,181,128];object_mesh['colors']=palette.tolist()
    floor=trimesh.Trimesh(V.floor_triangles(points,.1*float(domain.mesh.extents.max())).reshape(-1,3),np.arange(6).reshape(-1,3),process=False)
    viewer_events=[]
    for e,inter in zip(events,intersections):
        description=('真实碰撞：'+str(e['witness_label']) if e['object_collision_parts'] else '穿过地面' if e['floor_collision_parts'] else '未找到采样穿透见证；不能据此接受此方向。')
        if e.get('clearance_envelope_not_actual_material'):
            description=f"预留间隙 {e['reserved_clearance_m']*1000:.3f} mm 不满足；橙色是间隙包络与物体相交，不是实际材料穿透。"
        if partial:description='上游未完成，未测试装入动作。'
        if motion_verified:description='轨迹已通过；失败原因在承载或上游覆盖，详见 connection.json。'
        viewer_events.append(dict(name=e['name'],direction=e['direction'],translation=(np.asarray(e['translation_m'])*1000).tolist(),
            anchor=(joined.vertices.mean(axis=0)*1000).tolist(),colliding_parts=e['colliding_parts'],
            collision=pack_mesh(inter) if inter is not None and len(inter.faces) else None,
            clearance_envelope_not_actual_material=e.get('clearance_envelope_not_actual_material',False),
            description=description+f" 终点见证的相交体积 {e['intersection_volume_m3']*1e9:.6g} mm³。",material_note=note,
            focus=(np.asarray(e['focus_m'])*1000).tolist(),focus_width=e['focus_width_m']*1000))
    payload=dict(coordinate_system='z_up_xy_floor',title=f'{name} / {pose_name()} · Step5 失败形状',stage='实际构造阶段：'+stage+'。'+note,
        reason=REASONS.get(status,status),object=object_mesh,floor=pack_mesh(floor),
        shapes=[[dict(pack_mesh(p),label=l) for p,l in zip(parts,labels)]],events=viewer_events,primary_event=primary,
        center=(points.min(axis=0)+points.max(axis=0)).tolist(),width=float(np.ptp(points,axis=0).max())*1250)
    payload['center']=((points.min(axis=0)+points.max(axis=0))*.5*1000).tolist()
    template=Path(__file__).with_name('failure_viewer.html').read_text()
    (out/'failure_viewer.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/')))
    report=dict(complete=True,object=name,pose=pose_name(),status=status,material_stage=stage,material_note=note,
        successful_design_claimed=False,successful_trajectory_claimed=motion_verified,partial_step3=partial,
        shape_labels=labels,events=events,primary_event=primary,all_parts_translate_together=True,
        source_geometry='actual_saved_material_not_a_fabricated_complete_support',
        stl_units={'failed_shape.stl':'metres','failed_shape_mm.stl':'millimetres','workpiece_mm.stl':'millimetres','failure_pose_mm.stl':'millimetres'},
        provenance=dict(inputs=I.hashes(inputs),code=I.hashes([Path(__file__),Path(__file__).with_name('failure_viewer.html')])),
        artifacts={p.name:sha256(p) for p in out.iterdir() if (p.name.startswith(('failed_shape','failure','workpiece_mm'))) and p.name!='failure.json' and p.is_file()})
    I.save(out/'failure.json',report)
    return report


def no_heads(name,domain,out,inputs=()):
    """Show the actual object and explicit upstream failure, without fake material."""
    out.mkdir(parents=True,exist_ok=True)
    page=Image.new('RGB',(1600,1100),V.R.PAPER);ink=ImageDraw.Draw(page)
    ink.text((25,20),f'{name} / {pose_name()} / NO SELECTED HEADS',font=V.R.font(34),fill=V.R.INK)
    page.paste(V.scene(domain,[],[],950,show_object=True),(325,80))
    ink.text((25,1000),'Step3: no admissible first head. No frame, base or trajectory was constructed.',font=V.R.font(25),fill=V.R.INK)
    page.save(out/'failure.png');page.save(out/'failure_directions.png')
    obj=domain.mesh.copy();obj.apply_scale(1000);obj.export(out/'workpiece_mm.stl',file_type='stl_ascii')
    object_mesh=pack_mesh(domain.mesh)
    colors=np.tile([185,190,189],(len(domain.mesh.faces),1));colors[domain.work_ids]=[149,181,128]
    object_mesh['colors']=colors.tolist()
    center=(domain.com*1000).tolist();width=float(domain.mesh.extents.max())*1250
    event=dict(name='No selected heads',direction=[0,0,0],translation=[0,0,0],anchor=center,
        colliding_parts=[],collision=None,description=REASONS['no_selected_heads'],
        material_note='没有支撑材料；没有可导出的支撑 STL。',focus=center,focus_width=width)
    payload=dict(coordinate_system='z_up_xy_floor',title=f'{name} / {pose_name()} · 未选出支撑头',stage=event['material_note'],
        reason=REASONS['no_selected_heads'],object=object_mesh,floor=None,shapes=[[]],
        events=[event],primary_event=0,center=center,width=width)
    template=Path(__file__).with_name('failure_viewer.html').read_text()
    template=template.replace('<p><a href="failed_shape_mm.stl">', '<p id="materialLinks"><a href="failed_shape_mm.stl">')
    template=template.replace('</style>', '#materialLinks,#motionLabel,#play,#focus{display:none}</style>')
    (out/'failure_viewer.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/')))
    report=dict(complete=True,object=name,pose=pose_name(),status='no_selected_heads',material_stage='none',
        material_note=event['material_note'],shape_labels=[],events=[],trajectory_tested=False,
        successful_design_claimed=False,successful_trajectory_claimed=False,
        provenance=dict(inputs=I.hashes(inputs),code=I.hashes([Path(__file__),Path(__file__).with_name('failure_viewer.html')])),
        artifacts={f:sha256(out/f) for f in ('failure.png','failure_directions.png','failure_viewer.html','workpiece_mm.stl')})
    I.save(out/'failure.json',report)
    return report


def saved_failure(name,static_only=False):
    root=OUTPUTS/name/pose_name();out=root/'step5_connect_support'
    domain=ContinuousNeeds.read(root/'step_1_needs/needs.json')
    state=json.loads((root/'step3_scheculer/status.json').read_text());partial=not state.get('complete')
    if partial:
        choices=sorted((root/'step3.3_optimize_contact').glob('round_*/contacts.npz'))
        if not choices:raise RuntimeError('No saved head geometry is available to inspect')
        contact_path=choices[-1];contacts=I.read_contacts(contact_path)
        directions=json.loads((root/'step2_local_support/insertion_directions.json').read_text())
        parts,labels=X.make_heads(domain.mesh,contacts,directions['normal_depth_m'])
        geometry={};status='step3_incomplete';inputs=[contact_path,root/'step3_scheculer/status.json',root/'step_1_needs/needs.json']
    else:
        report=I.check_report(out/'connection.json');contacts=I.read_contacts(root/'step3_scheculer/final_contacts.npz')
        if not contacts:
            assert report['status']=='no_selected_heads'
            return no_heads(name,domain,out,[out/'connection.json',root/'step_1_needs/needs.json'])
        paths=[out/f for f in ('geometry.npz','failed_attempt.npz','belt.npz','heads.npz') if (out/f).exists()]
        if not paths:raise RuntimeError('No attempted material was saved')
        def material_rank(path):
            data=I.load_npz(path);return (STAGES.index(stage_material(data['part_labels'].tolist())),len(data['part_labels']))
        source=out/'geometry.npz' if (out/'geometry.npz').exists() else max(paths,key=material_rank);a=I.load_npz(source);parts=S.unpack_parts(a);labels=a['part_labels'].tolist()
        geometry=report['geometry'];status=report['status']
        if source.name=='failed_attempt.npz':
            retained=json.loads((out/'failed_attempt.json').read_text())
            if retained['stage']!='heads':
                geometry=dict(geometry,retained_failure=retained,attempts=[dict(direction=retained['withdrawal_direction'])])
        inputs=[out/'connection.json',source,root/'step_1_needs/needs.json']
        if source.name=='failed_attempt.npz': inputs.append(out/'failed_attempt.json')
    return render(name,domain,contacts,parts,labels,geometry,status,out,static_only,inputs,partial)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('--pose',default='pose_1');parser.add_argument('--static-only',action='store_true');args=parser.parse_args()
    with selected_pose(args.pose):saved_failure(args.name,args.static_only)
