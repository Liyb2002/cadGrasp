"""Plan native Panda loading plus fixed-axis KUKA work on the accepted Step5."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import trimesh
from whole_task_2.step1.common import ROOT,sha256,write_json
from whole_task_2.step3.legacy import kernel as K
from whole_task_2.step5.source_model import restore
from .workflow import Workflow
from .kuka import Arm,description,visual_parts
from .work_path import SurfacePath

TABLE_Z=.62
WORKBENCH_ABOVE_WORK_BASE=.50


def matrices(transforms):
    out=np.repeat(np.eye(4)[None],len(transforms),axis=0)
    for T,(R,p) in zip(out,transforms):T[:3,:3]=R;T[:3,3]=p
    return out


def export_robot_meshes(folder,workflow):
    arrays={};parts=[]
    # Original full visual model; local coordinates preserve native link shape.
    model=mujoco.MjModel.from_xml_path(str(ROOT/'slides/Co-optimize/idea/assets/franka_emika_panda/panda.xml'))
    data=mujoco.MjData(model);data.qpos[:7]=workflow.robot.home;mujoco.mj_forward(model,data)
    for g in range(model.ngeom):
        if model.geom_group[g]!=2 or model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        b=model.body(model.geom_bodyid[g]).name
        if not (b.startswith('link') or b in ['hand','left_finger','right_finger']):continue
        mid=model.geom_dataid[g]
        lettering=model.mesh(mid).name=='link5_1'
        a,f=model.mesh_vertadr[mid],model.mesh_faceadr[mid]
        v=model.mesh_vert[a:a+model.mesh_vertnum[mid]].copy();faces=model.mesh_face[f:f+model.mesh_facenum[mid]].copy()
        v=v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        bid=model.geom_bodyid[g];v=(v-data.xpos[bid])@data.xmat[bid].reshape(3,3)
        if lettering:
            center=v.mean(0);_,_,basis=np.linalg.svd(v-center,full_matrices=False)
            local=(v-center)@basis.T;low,high=local.min(0),local.max(0);low-=.0012;high+=.0012
            v=np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])@basis+center
            faces=np.array([[0,4,6],[0,6,2],[1,3,7],[1,7,5],[0,1,5],[0,5,4],[2,6,7],[2,7,3],[0,2,3],[0,3,1],[4,5,7],[4,7,6]])
        name=f'panda_{g}';arrays[name+'_v']=v;arrays[name+'_f']=faces
        color='white' if not lettering and np.mean(model.mat_rgba[model.geom_matid[g],:3])>.6 else 'black'
        parts.append(dict(name=name,robot='panda',body=b,color=color))
    links,_,_=description()
    for i,link in enumerate(links):
        offset=np.fromstring(link.find('visual/origin').get('xyz'),sep=' ')
        for j,(v,f,rgba) in enumerate(visual_parts(i)):
            name=f'kuka_{i}_{j}';arrays[name+'_v']=v+offset;arrays[name+'_f']=f
            color=rgba[:3].tolist()
            if i in (2,4) and np.allclose(color,.2):color=[.6]*3
            parts.append(dict(name=name,robot='kuka',body=i,color=color))
    np.savez_compressed(folder/'robot_meshes.npz',**arrays)
    return parts


def prepare(out,duration=48.,fps=24):
    out=Path(out).resolve();folder=out/'video';folder.mkdir(exist_ok=True)
    report=json.loads((out/'report.json').read_text())
    model,layout=restore(report,out/'layout.npz')
    fixture=trimesh.load(out/'support.obj',force='mesh',process=False)
    S=np.eye(4);S[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();S[2,3]=TABLE_Z
    rows={};config=json.loads((ROOT/'robots_config/04_kuka_lbr_med14_r820.json').read_text())
    for k in layout.active:
        state=model.execution(layout,k);T=S@np.asarray(state['world_from_support']);U=S@np.asarray(state['world_from_object'])
        d=T[:3,:3]@layout.directions[k]
        sv=trimesh.transform_points(fixture.vertices,T);ov=trimesh.transform_points(model.mesh.vertices,U)
        travel=max(.10,float((sv@d).max()-(ov@d).min()+.022))
        ids=model.tasks[k].domain.work_ids
        rows[int(k)]=dict(index=int(k),pose=model.poses[k],host=int(layout.hosts[k]),role='computed_pose',
            fixture=T,object=U,direction=d,travel=travel,work_ids=ids,
            axis=S[:3,:3]@model.tasks[k].domain.axis)
    installations={}
    for k,row in rows.items():
        installations.setdefault(np.round(row['fixture'],11).tobytes(),[]).append(k)
    order=[k for group in installations.values() for k in group]
    data=dict(label='current_bunny',object=model.mesh,fixture=fixture,rows=rows,order=order,
        robot_options=dict(base=[1.05,.04,TABLE_Z],yaw=np.pi),park_xy=[.80,.30],
        initial_fixture_offset=[0,.25,0],joint_space_transfer=True,
        grasp_options=dict(samples=700,axis_subdivisions=1,candidate_limit=1100,expanded_angles=True))
    workflow=Workflow(data);grasp_verification=workflow.prepare(rate=12)
    # Work robot is rigidly transformed together with the certified station;
    # changing this common presentation frame does not change any task pose.
    # A floor-mounted work arm beside a raised bench. This is an explicit
    # presentation installation choice, not a change to Step4 force geometry.
    work_base=S[:3,3]+[0,0,-WORKBENCH_ABOVE_WORK_BASE]
    arm=Arm(work_base,S[:3,:3]@Rotation.from_euler('z',-90,degrees=True).as_matrix(),tool=.155)
    axis=rows[0]['axis'];paths={};retract_q={};patch_arrays={}
    for k,row in rows.items():
        U=row['object'];local_axis=U[:3,:3].T@axis
        path=SurfacePath(model.mesh,row['work_ids'],local_axis);paths[k]=path
        start=trimesh.transform_points(path.outline[:1],U)[0]
        q=arm.solve(start,-axis)
        retract_q[k]=arm.solve(start-axis*.11,-axis,previous=q)
        patch_arrays[f'patch_{k}_v']=model.mesh.vertices.copy()+model.mesh.vertex_normals*2e-5
        patch_arrays[f'patch_{k}_f']=model.mesh.faces[row['work_ids']]
        patch_arrays[f'patch_{k}_ids']=row['work_ids']
        for name in ('outline','outline_s','interior','interior_s'):
            patch_arrays[f'path_{k}_{name}']=getattr(path,name)
        print('SURFACE PATH',row['pose'],path.info(),flush=True)
    np.savez_compressed(folder/'scene_meshes.npz',object_v=model.mesh.vertices,object_f=model.mesh.faces,
        fixture_v=fixture.vertices,fixture_f=fixture.faces,**patch_arrays)
    parts=export_robot_meshes(folder,workflow)
    frames=int(round(duration*fps))
    # Reserve time to see each local work stroke; compress handling phases.
    work_video_seconds=min(6.6,duration/(len(rows)*1.6))
    work_reference=sum(s['end']-s['start'] for s in workflow.segments if s['b']['task'] is not None)
    free_scale=(duration-work_video_seconds*len(rows))/(workflow.duration-work_reference)
    video_edges=[0.];reference_edges=[0.]
    for seg in workflow.segments:
        video_edges.append(video_edges[-1]+(work_video_seconds if seg['b']['task'] is not None else (seg['end']-seg['start'])*free_scale))
        reference_edges.append(seg['end'])
    times=np.interp(np.linspace(0,duration,frames),video_edges,reference_edges)
    p_links=[];k_links=[];objects=[];fixtures=[];tasks=[];tool=[];qwork=[];qpanda=[];gaps=[];hit_ids=[];labels=[]
    panda_bodies=[f'link{i}' for i in range(8)]+['hand','left_finger','right_finger']
    paint_progress=[];work_phases=[];work_tasks=[];progress=np.zeros(len(rows))
    # Follow the entire real patch perimeter before crossing its interior.
    # Every saved working TCP is independently checked against the real mesh.
    previous=retract_q[0].copy();errors=[];deviations=[]
    for frame,t in enumerate(times):
        state=workflow.state(float(t));task=state['task'];label=state['label'];labels.append(label)
        if task is not None:
            task=int(task);row=rows[task]
            seg=next(s for s in workflow.segments if s['start']<=t<=s['end'])
            u=float(np.clip((t-seg['start'])/(seg['end']-seg['start']),0,1))
            local_pt,hit_face,phase=paths[task].at(u)
            pt=trimesh.transform_points(local_pt[None],row['object'])[0]
            approach=np.clip(u/.10,0,1);withdraw=np.clip((u-.90)/.10,0,1)
            ease=lambda v:v*v*(3-2*v)
            gap=.11*(1-ease(approach)+ease(withdraw))
            target=pt-axis*gap
            previous=arm.solve(target,-axis,previous)
            tcp,R,transforms=arm.forward(previous,geometry=True)
            active=task if phase in (1,2) else -1
            progress[task]=float(np.clip((u-.10)/.80,0,1))
            if active>=0:errors.append(np.linalg.norm(tcp-pt));deviations.append(np.degrees(np.arccos(np.clip(R[:,2]@axis,-1,1))))
            hit_ids.append(hit_face if active>=0 else -1)
        else:
            # Keep the work arm retracted while the grasper handles parts.
            # Smooth native joint motion follows the next/last task retreat.
            future=[s for s in workflow.segments if s['start']>=t and s['b']['task'] is not None]
            k=int(future[0]['b']['task']) if future else len(rows)-1
            target_q=retract_q[k]
            previous=previous+np.clip(target_q-previous,-.025,.025)
            tcp,R,transforms=arm.forward(previous,geometry=True);active=-1;phase=-1;hit_ids.append(-1)
        paint_progress.append(progress.copy());work_phases.append(phase);work_tasks.append(-1 if task is None else task)
        k_links.append(matrices(transforms));tool.append(tcp);qwork.append(previous.copy());tasks.append(active)
        qpanda.append(state['joints']);gaps.append(state['gap'])
        workflow.robot.fk(state['joints'],state['gap'])
        native=workflow.robot
        p_links.append(matrices([(native.data.xmat[native.model.body(b).id].reshape(3,3),native.data.xpos[native.model.body(b).id]) for b in panda_bodies]))
        objects.append(state['object']);fixtures.append(state['fixture'])
    np.savez_compressed(folder/'trajectory.npz',times=times,object_T=objects,fixture_T=fixtures,
        panda_links=p_links,kuka_links=k_links,panda_q=qpanda,kuka_q=qwork,gaps=gaps,
        task=tasks,tool_tcp=tool,work_face=hit_ids,paint_progress=paint_progress,work_phase=work_phases,work_task=work_tasks)
    plan=dict(schema='two_robot_native_kinematic_presentation_v2',fps=fps,frames=frames,duration_s=duration,
        reference_duration_s=workflow.duration,playback_speedup=workflow.duration/duration,
        table_height_m=TABLE_Z,work_robot_base_world_m=work_base.tolist(),
        workbench_above_work_robot_base_m=WORKBENCH_ABOVE_WORK_BASE,
        robot_installation_scope='video installation; KUKA base 500mm below fixture floor; saved relative object/fixture poses unchanged',
        presentation_transform=S.tolist(),robot_parts=parts,panda_bodies=panda_bodies,
        object_poses=[r['object'].tolist() for r in rows.values()],fixture_poses=[r['fixture'].tolist() for r in rows.values()],
        execution_order=order,unique_support_installations=len(installations),
        grasp_verification=grasp_verification,work_axis_world=axis.tolist(),work_tool_length_from_link7_m=.155,
        max_work_tcp_error_m=float(max(errors)),max_work_axis_deviation_deg=float(max(deviations)),
        work_frames=sum(x>=0 for x in tasks),labels=labels,
        body_motion_uses_native_gripper_FK=True,grasp_faces_exclude_active_work_area=True,
        work_motion='complete selected-mesh outer boundary, followed by three broad interior strokes; fixed axis; 110mm axial retract',
        work_paths={model.poses[k]:paths[k].info() for k in rows},
        work_area_color='pink',completed_paint_persists=True,
        playback_time_mapping=f'{work_video_seconds}s per work dwell; remaining time allocated to handling',
        full_area_toolpath_or_continuous_arm_collision_certified=False,
        playback_joint_speeds_certified=False,painting_dynamics_certified=False,
        sources={str(p):sha256(p) for p in [out/'support.obj',out/'layout.npz',out/'execution.json',
            ROOT/'robots_config/04_kuka_lbr_med14_r820.json']})
    write_json(folder/'plan.json',plan)
    write_json(folder/'segments.json',[dict(start=s['start'],end=s['end'],label=s['label'],moving=s['moving'],
        task=s['b']['task'],grasp=None if s['grasp'] is None else s['grasp'].tolist()) for s in workflow.segments])
    print('VIDEO PLAN',frames,'frames; work error',plan['max_work_tcp_error_m'],flush=True)
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--duration',type=float,default=48)
    a=p.parse_args();prepare(a.folder,a.duration)
