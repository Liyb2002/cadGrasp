"""Check the ground scene, complete task models and active gripper framing."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from itertools import combinations
from inputs import HERE,SETS,load,save,sha
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('franka_video_run',HERE/'run.py')
runner=module_from_spec(spec);spec.loader.exec_module(runner)
restore_workflow=runner.restore_workflow
from presentation import Display,SCENE_TRANSLATION,CAMERA_ZOOM,BASE_FOVY_DEG,CAMERA_DISTANCE_M
from force_arrows import ARROW_COUNT,project_points


def projected(scene,points,width,height):
    a,b=scene.camera
    origin=(a.pos.astype(float)+b.pos.astype(float))/2
    forward=a.forward.astype(float);up=a.up.astype(float);right=np.cross(forward,up)
    vectors=points-origin;depth=vectors@forward
    if np.min(depth)<=0:raise ValueError('Visible geometry behind camera')
    focal=height*a.frustum_near/(a.frustum_top-a.frustum_bottom)
    return np.column_stack([width/2+focal*(vectors@right)/depth,height/2-focal*(vectors@up)/depth])


def review(label,selection=None,output_dir=None):
    import json
    folder=(output_dir or HERE.parent)/'assets/data'/label
    plan=json.loads((folder/'plan.json').read_text())
    source=load(label,selection,plan.get('station_yaw_offset_deg',0.))
    assert plan['pose_order']==[source['rows'][i]['pose']for i in source['order']]
    hosts=[source['rows'][i]['host']for i in source['order']]
    assert all(a!=b for a,b in zip(hosts,hosts[1:]))
    w=restore_workflow(source,folder);d=Display(w,1600,900)
    tasks=[s for s in w.segments if s['label'].startswith('task_pose_')]
    assert [s['label'][5:]for s in tasks]==plan['pose_order']
    installation_angles=[float(np.degrees(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()))
        for a,b in combinations(source['frames'].values(),2)]
    assert len(source['frames'])>=3 and min(installation_angles)>1.
    transition_angles=[]
    for a,b in zip(tasks,tasks[1:]):
        angle=float(np.degrees(Rotation.from_matrix(a['b']['fixture'][:3,:3].T@b['b']['fixture'][:3,:3]).magnitude()))
        assert angle>1.
        transition_angles.append(angle)
    assert d.model.body_pos[d.model.body('link0').id,2]==0
    names={d.model.geom(g).name for g in range(d.model.ngeom)}
    assert 'table_top' not in names and 'robot_pedestal' not in names
    zoom=np.tan(np.radians(BASE_FOVY_DEG)/2)/np.tan(np.radians(d.model.vis.global_.fovy)/2)
    assert abs(zoom-CAMERA_ZOOM)<1e-6  # MuJoCo stores the field of view as float32.
    assert .64<d.camera.distance<1.85 and d.camera.distance==CAMERA_DISTANCE_M
    assert d.camera.azimuth==180
    assert not hasattr(d,'close')
    arrow_audit=[]
    for field in d.arrow_fields.values():
        assert field['arrow_count']==ARROW_COUNT
        assert max(field['angles_to_inward_normal_deg'])<=field['cone_half_deg']+1e-8
        assert field['point_surface_max_error_m']<1e-10
        assert np.ptp(field['display_length_m'])>.01
        assert max(field['magnitude_mg'])<=field['force_limit_mg']+1e-10
        assert sha(field['sample_source'])==field['sample_source_sha256']
        arrow_audit.append(dict(pose=field['pose'],count=field['arrow_count'],
            length_min_m=min(field['display_length_m']),length_max_m=max(field['display_length_m']),
            angle_max_deg=max(field['angles_to_inward_normal_deg']),legal_directions_passed=True,
            surface_points_passed=True,original_sample_ids=field['original_sample_ids']))
    parts=[]
    for g in range(d.model.ngeom):
        if d.model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or d.model.geom_group[g]!=2:continue
        if d.model.geom(g).name.startswith('work_'):continue
        mesh=d.model.geom_dataid[g];a=d.model.mesh_vertadr[mesh];n=d.model.mesh_vertnum[mesh]
        vertices=d.model.mesh_vert[a:a+n].astype(float)
        vertices=vertices[ConvexHull(vertices).vertices]
        body_name=d.model.body(d.model.geom_bodyid[g]).name
        parts.append((g,vertices,body_name))
    scene=mujoco.MjvScene(d.model,maxgeom=2000)
    margin=float('inf');robot_margin=float('inf');gripper_margin=float('inf');fk_error=0.;attachment_error=0.
    minimum={name:float('inf') for name in ['object','fixture']}
    attached=0;robot_outside=0;body_visible_frames=0
    task_depth_gap=float('inf');task_frames=0
    arrow_margin=float('inf')
    for t in np.linspace(0,w.duration,480):
        state=d.update(t)
        expected=w.robot.fk(state['joints'],state['gap']).copy();expected[:3,3]+=SCENE_TRANSLATION
        actual=np.eye(4);actual[:3,:3]=d.data.body('hand').xmat.reshape(3,3);actual[:3,3]=d.data.body('hand').xpos
        fk_error=max(fk_error,float(np.max(np.abs(actual-expected))))
        assert np.all(state['joints']>=w.robot.lower) and np.all(state['joints']<=w.robot.upper)
        if state['moving']:
            name=state['moving'];body=np.eye(4)
            body[:3,:3]=d.data.body(name).xmat.reshape(3,3);body[:3,3]=d.data.body(name).xpos
            attachment_error=max(attachment_error,float(np.max(np.abs(body@state['grasp']-actual))))
            attached+=1
        for name in minimum:
            vertices=source[name].vertices@state[name][:3,:3].T+state[name][:3,3]+SCENE_TRANSLATION
            minimum[name]=min(minimum[name],float(vertices[:,2].min()))
        mujoco.mjv_updateScene(d.model,d.data,d.option,None,d.camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
        if state['task'] is not None:
            forward=scene.camera[0].forward.astype(float)
            robot_position=d.data.body('link0').xpos
            for name in ['object','fixture']:
                centroid=source[name].vertices.mean(0)@state[name][:3,:3].T+state[name][:3,3]+SCENE_TRANSLATION
                task_depth_gap=min(task_depth_gap,float((robot_position-centroid)@forward))
            task_frames+=1
            field=d.arrow_fields[state['task']];points=np.asarray(field['points_local_m'])
            directions=np.asarray(field['inward_directions_local']);lengths=np.asarray(field['display_length_m'])
            ends=np.vstack([points,points-directions*lengths[:,None]])
            world=ends@state['object'][:3,:3].T+state['object'][:3,3]+SCENE_TRANSLATION
            pixels=project_points(scene,world,d.width,d.height);lo=pixels.min(0);hi=pixels.max(0)
            arrow_margin=min(arrow_margin,float(min(lo[0],lo[1],d.width-hi[0],d.height-hi[1])))
        robot_frame_margin=float('inf')
        body_visible=False
        for g,vertices,body_name in parts:
            points=vertices@d.data.geom_xmat[g].reshape(3,3).T+d.data.geom_xpos[g]
            pixels=projected(scene,points,d.width,d.height)
            lo=pixels.min(0);hi=pixels.max(0)
            edge=float(min(lo[0],lo[1],d.width-hi[0],d.height-hi[1]))
            if body_name in ['object','fixture']:
                margin=min(margin,edge)
            else:
                robot_margin=min(robot_margin,edge)
                robot_frame_margin=min(robot_frame_margin,edge)
                if body_name not in ['hand','left_finger','right_finger']:
                    body_visible |= bool(np.any(np.all((pixels>0)&(pixels<[d.width,d.height]),axis=1)))
                if state['moving'] and body_name in ['hand','left_finger','right_finger']:
                    gripper_margin=min(gripper_margin,edge)
        robot_outside+=int(robot_frame_margin<0)
        body_visible_frames+=int(body_visible)
    assert margin>=0,('Task model outside frame',label,margin)
    assert gripper_margin>=0,('Active gripper outside frame',label,gripper_margin)
    assert body_visible_frames==480,('Robot body absent',label,body_visible_frames)
    assert task_depth_gap>0,('Robot is not behind task models',label,task_depth_gap)
    assert arrow_margin>=0,('Force arrows outside frame',label,arrow_margin)
    assert fk_error<1e-10 and attachment_error<1e-10,(label,fk_error,attachment_error)
    assert min(minimum.values())>-.00005,(label,minimum)
    assert all(sha(path)==expected for path,expected in source['protected'].items())
    result=dict(set=label,sampled_frames=480,all_delivered_frames_checked=True,
        pose_order=plan['pose_order'],support_host_order=hosts,
        task_to_task_support_turns=sum(a!=b for a,b in zip(hosts,hosts[1:])),
        support_installations_interleaved=True,
        minimum_support_orientation_separation_deg=min(installation_angles),
        task_to_task_rotation_angles_deg=transition_angles,
        robot_render_FK_max_difference=fk_error,rigid_attachment_max_difference=attachment_error,
        sampled_min_mesh_height_m=minimum,full_body_camera_margin_px=robot_margin,
        task_models_camera_margin_px=margin,active_gripper_camera_margin_px=gripper_margin,
        robot_visual_bounds_out_of_camera_samples=robot_outside,full_body_in_frame=robot_outside==0,
        robot_body_visible_frames=body_visible_frames,
        robot_behind_task_models=True,robot_task_depth_gap_min_m=task_depth_gap,task_frames_checked=task_frames,
        force_arrows=arrow_audit,force_arrow_camera_margin_px=arrow_margin,
        force_arrows_per_task=ARROW_COUNT,legal_arrow_directions_passed=True,variable_arrow_lengths_passed=True,
        inset_visible=False,camera_distance_m=d.camera.distance,rigid_attached_motion_samples=attached,
        source_hashes_unchanged=True,floor_height_m=0.,table_visible=False,robot_pedestal_visible=False,
        camera_zoom=zoom,physical_geometry_scale=1.,joint_limits_passed=True,
        continuous_collision_certificate=False)
    save(folder/'motion_review.json',result)
    print(label,'task model margin',round(margin,2),'active gripper margin',round(gripper_margin,2),
        'FK',fk_error,flush=True)
    return result


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets',nargs='+',default=SETS)
    parser.add_argument('--result-selection',type=Path)
    parser.add_argument('--output-dir',type=Path,default=HERE.parent)
    args=parser.parse_args()
    results=[review(label,args.result_selection,args.output_dir) for label in args.sets]
    save(args.output_dir/'assets/data/motion_review.json',dict(cases=results,robot_FK_and_joint_limits_passed=True))
