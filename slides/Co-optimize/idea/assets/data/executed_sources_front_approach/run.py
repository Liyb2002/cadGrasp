"""Make isolated native-Franka kinematic workflow movies of accepted pose sets."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,subprocess,time
import numpy as np
import mujoco
from PIL import Image
from inputs import HERE,SETS,load,save,sha
from workflow import Workflow
from presentation import Display,SCENE_TRANSLATION,CAMERA_ZOOM,FOVY_DEG,ARROW_COUNT


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,list):return [serial(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def restore_workflow(data,folder):
    w=Workflow.__new__(Workflow);w.data=data
    from franka import Franka
    w.robot=Franka();segments=json.loads((folder/'segments.json').read_text())
    w.segments=segments['segments'];w.duration=segments['duration'];w.story=segments['story']
    for s in w.segments:
        for key in ['a','b']:
            for name in ['object','fixture','hand']:s[key][name]=np.asarray(s[key][name])
        if s['grasp'] is not None:s['grasp']=np.asarray(s['grasp'])
    with np.load(folder/'trajectory.npz') as a:
        w.times=a['times'];w.joints=a['joints'];w.time_scale=float(a['time_scale'])
    return w


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sets',nargs='+',default=SETS)
    p.add_argument('--preview',action='store_true');p.add_argument('--fps',type=int,default=24)
    p.add_argument('--width',type=int,default=1600);p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--reuse-plan',action='store_true')
    p.add_argument('--result-selection',type=Path,help='Certified Step5 result manifest; otherwise retain stocky baseline')
    p.add_argument('--output-dir',type=Path,help='Independent videos, plans and reviews')
    p.add_argument('--expanded-grasps',action='store_true',help='Denser deterministic parallel-jaw proposals for new layouts')
    p.add_argument('--station-yaw-offset',type=float,default=0.,help='Rotate task scene and exit directions together about gravity for robot access')
    args=p.parse_args()
    if args.seconds<=0:p.error('--seconds must be positive')
    destination=(args.output_dir or HERE).resolve();destination.mkdir(parents=True,exist_ok=True)
    for label in args.sets:
        began=time.perf_counter();folder=destination/'data'/label;folder.mkdir(parents=True,exist_ok=True)
        data=load(label,args.result_selection,args.station_yaw_offset)
        if args.expanded_grasps:
            data['grasp_options']=dict(samples=1000,axis_subdivisions=1,candidate_limit=2000,expanded_angles=True)
        if args.reuse_plan and (folder/'segments.json').exists():
            previous=json.loads((folder/'plan.json').read_text())
            if previous['fixture_source']!=data['fixture_source']or any(sha(f)!=h for f,h in previous['source_hashes'].items()):
                raise ValueError('Saved trajectory belongs to another geometry/layout; plan the new result again')
            w=restore_workflow(data,folder)
            robot=json.loads((folder/'plan.json').read_text())['robot']
        else:
            w=Workflow(data);robot=w.prepare()
            np.savez_compressed(folder/'trajectory.npz',times=w.times,joints=w.joints,time_scale=w.time_scale)
            save(folder/'segments.json',serial(dict(segments=w.segments,duration=w.duration,story=w.story)))
        duration=args.seconds
        display=Display(w,args.width,round(args.width*9/16));width,height=display.width,display.height
        report=dict(set=label,kind='kinematic_workflow',object_pose_count=len(data['rows']),
            support_states=len(data['frames']),pose_order=[data['rows'][i]['pose'] for i in data['order']],
            roles={data['rows'][i]['pose']:data['rows'][i]['role'] for i in data['order']},
            fixture_source=data['fixture_source'],robot=robot,franka_body_visible=True,native_robot_scale=1.,
            grasp_proposal_options=data.get('grasp_options',{}),
            station_yaw_offset_deg=args.station_yaw_offset,
            gripper='native Panda parallel jaws',attached_motion='carried body follows arm forward kinematics with fixed grasp transform',
            dynamics_integrated=False,grasp_force_or_retention_certified=False,continuous_robot_collision_certified=False,
            task_marker='green working area with a field of legal variable-length inward force arrows',scene_floor_height_m=0.,
            task_arrow_count=ARROW_COUNT,task_arrow_lengths='proportional to saved individual force magnitude',
            task_arrow_directions='saved legal inward-cone samples',simultaneous_arrow_load_certified=False,
            table_visible=False,robot_pedestal_visible=False,scene_world_translation_m=SCENE_TRANSLATION,
            inset_visible=False,camera_framing='front workspace view; object and fixture in front of robot',full_body_framing_required=False,
            reference_camera_distances_m=dict(previous_main=1.4),
            camera_zoom=CAMERA_ZOOM,camera_fovy_deg=FOVY_DEG,physical_geometry_scale=1.,
            camera=dict(lookat=display.camera.lookat,distance=display.camera.distance,
                azimuth=display.camera.azimuth,elevation=display.camera.elevation),
            presentation_station_yaw={data['rows'][i]['pose']:data['rows'][i]['yaw'] for i in data['order']},
            object_fixture_relative_transforms_preserved=True,source_hashes=data['protected'],fps=args.fps,
            planned_duration_s=w.duration,video_duration_s=duration,video_playback_factor=w.duration/duration,
            joint_velocity_certified_for_video=False,source_snapshots_unchanged=True,
            rendering_source_hashes={str(HERE/name):sha(HERE/name) for name in
                ['run.py','presentation.py','force_arrows.py','inputs.py','franka.py','workflow.py','grasps.py']})
        save(folder/'force_arrows.json',serial(dict(set=label,fields=display.arrow_fields)))
        save(folder/'plan.json',serial(report))
        with mujoco.Renderer(display.model,height=height,width=width) as renderer:
            stamps=[0,*w.story,w.duration-.01];thumb_w=640;thumb_h=360
            board=Image.new('RGB',(thumb_w*3,thumb_h*int(np.ceil(len(stamps)/3))),'white')
            for i,t in enumerate(stamps):
                frame=Image.fromarray(display.frame(renderer,t));frame.resize((thumb_w,thumb_h)).save(folder/f'keyframe_{i:02d}.png')
                board.paste(frame.resize((thumb_w,thumb_h)),((i%3)*thumb_w,(i//3)*thumb_h))
            board.save(destination/(label+'_storyboard.png'))
            if not args.preview:
                output=destination/(label+'_franka.mp4');pending=output.with_suffix('.pending.mp4')
                command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
                    '-r',str(args.fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(pending)]
                frames=round(duration*args.fps)
                proc=subprocess.Popen(command,stdin=subprocess.PIPE)
                try:
                    for i in range(frames):
                        proc.stdin.write(display.frame(renderer,w.duration*i/max(1,frames-1)).tobytes())
                        if i%(args.fps*5)==0:print('VIDEO',label,round(i/args.fps), '/',round(duration),flush=True)
                finally:proc.stdin.close()
                if proc.wait()!=0:raise RuntimeError('Video encoder failed')
                pending.replace(output);report.update(output=output.name,frames=frames,video_sha256=sha(output))
        changed=[f for f,h in data['protected'].items() if sha(f)!=h]
        changed += [f for f,h in report['rendering_source_hashes'].items() if sha(f)!=h]
        if changed:raise ValueError('Accepted input changed: '+str(changed))
        report.update(complete=True,seconds=time.perf_counter()-began,accepted_inputs_unchanged=True)
        save(folder/'report.json',serial(report));print('COMPLETE',label,flush=True)


if __name__=='__main__':main()
