"""Automatically prepare, render, encode and verify the Step5 two-robot movie."""
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from whole_task_2.step1.common import ROOT,write_json,sha256


def command(args,log):
    with Path(log).open('a') as f:
        environment=os.environ.copy()
        if str(args[0])=='blender':
            environment.update(LIBGL_ALWAYS_SOFTWARE='1',__GLX_VENDOR_LIBRARY_NAME='mesa')
        subprocess.run([str(a) for a in args],stdout=f,stderr=subprocess.STDOUT,check=True,env=environment)


def make_movie(out,model=None,layout=None,duration=48,fps=24,reuse_plan=False,verified_motion=None):
    from .video.prepare import prepare
    out=Path(out).resolve();data=out/'video'
    plan=json.loads((data/'plan.json').read_text()) if reuse_plan and (data/'plan.json').exists() else prepare(out,duration,fps)
    for source,digest in plan['sources'].items():
        path=Path(source)
        if not path.exists() and (out/path.name).exists():path=out/path.name
        if sha256(path)!=digest:raise ValueError('Changed movie geometry input: '+source)
    from .video.verify import verify
    # The same caller may rerender an unchanged, already verified scene with a
    # different visual backend. Bind its proof to the exact saved motion.
    if verified_motion is not None:
        if (verified_motion.get('status')!='passed' or
            verified_motion['trajectory_sha256']!=sha256(data/'trajectory.npz') or
            verified_motion['plan_sha256']!=sha256(data/'plan.json')):
            raise ValueError('Motion proof does not match this rendering')
        motion_validation=verified_motion
    else:motion_validation=verify(data)
    video=out/'video.mp4';poster=out/'video_poster.png'
    from .video.raster import render
    pending=video.with_suffix('.pending.mp4')
    rendering=render(data,pending)
    os.replace(pending,video)
    # These images come from the encoded movie; no unencoded stand-ins.
    trajectory=np.load(data/'trajectory.npz')
    contact=np.flatnonzero(trajectory['task']==2)
    shot=float(contact[len(contact)//2]/plan['fps']) if len(contact) else plan['duration_s']/2
    command(['ffmpeg','-y','-v','error','-ss',shot,'-i',video,'-frames:v','1',poster],data/'encode.log')
    command(['ffmpeg','-y','-v','error','-i',video,'-vf',f'fps=8/{plan["duration_s"]},scale=480:-1,tile=4x2',
        '-frames:v','1',data/'encoded_frames.png'],data/'encode.log')
    command(['ffmpeg','-v','error','-i',video,'-f','null','-'],data/'encode.log')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)],text=True))
    streams=probe['streams'];stream=streams[0]
    if len(streams)!=1 or stream['codec_type']!='video' or int(stream['nb_read_frames'])!=plan['frames']:
        raise ValueError('Encoded video frame/audio verification failed')
    native_sources=[]
    for relative in ['codes/simulation/assets/lbr_med14_r820','slides/Co-optimize/idea/assets/franka_kinematics','slides/Co-optimize/idea/assets/franka_emika_panda']:
        native_sources.extend(p for p in (ROOT/relative).rglob('*') if p.suffix.lower() in ['.xml','.xacro','.yaml','.obj','.stl','.dae'])
    native_sources.extend([ROOT/'codes/simulation/kuka.py',ROOT/'robots_config/arm_visuals.py'])
    from .video.motion_cache import save
    motion_bundle=save(out,native_sources)
    report=dict(motion_bundle=motion_bundle,video=str(video),poster=str(poster),sha256=sha256(video),poster_sha256=sha256(poster),
        frames=int(stream['nb_read_frames']),duration_s=float(probe['format']['duration']),fps=plan['fps'],
        renderer_engine=rendering['engine'],path_tracing=False,rendering=rendering,
        width=stream['width'],height=stream['height'],all_frames_decode=True,no_audio=True,no_text_overlay=True,
        source_video='slides/Co-optimize/idea/pose6+10+13+17+30_franka.mp4',
        style_reference='slides/idea/vis/robot_access',
        motion_plan=str(data/'plan.json'),encoded_contact_sheet=str(data/'encoded_frames.png'),
        motion_validation=motion_validation,
        verification={k:plan[k] for k in ['grasp_verification','max_work_tcp_error_m','max_work_axis_deviation_deg',
            'work_frames','body_motion_uses_native_gripper_FK','grasp_faces_exclude_active_work_area',
            'work_motion','work_paths','work_area_color','completed_paint_persists','full_area_toolpath_or_continuous_arm_collision_certified',
            'playback_joint_speeds_certified','robot_installation_scope']},
        inputs={str(p):sha256(p) for p in [data/'plan.json',data/'trajectory.npz',data/'scene_meshes.npz',data/'robot_meshes.npz',
            *Path(__file__).parent.joinpath('video').glob('*.py'),Path(__file__),*native_sources]})
    write_json(data/'video.json',report)
    saved=json.loads((out/'report.json').read_text());saved['video']=report;write_json(out/'report.json',saved)
    print('VIDEO',str(video),flush=True)
    return report


def rebuild(folder,duration=48,fps=24):
    """Replace only the movie, using the saved accepted fixture and evaluation."""
    from .publish import publish,DATA_FILES
    from .validate import validate
    folder=Path(folder).resolve();report=json.loads((folder/'report.json').read_text())
    output=Path(report['public_output'])
    with tempfile.TemporaryDirectory(prefix='cadgrasp_step5_repaint_') as temporary:
        work=Path(temporary)
        for name in (*DATA_FILES,'stock.obj','report.json'):shutil.copy2(folder/name,work/name)
        shutil.copy2(output/'evaluation.png',work/'evaluation.png')
        from .video.motion_cache import restore
        proof=restore(folder,work)
        if proof is not None:
            cached=json.loads((work/'video/plan.json').read_text())
            if cached['duration_s']!=duration or cached['fps']!=fps:proof=None
        make_movie(work,duration=duration,fps=fps,reuse_plan=proof is not None,verified_motion=proof)
        published,data=publish(work,output)
    validate(data)
    return published['video']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    p.add_argument('--reuse-plan',action='store_true');p.add_argument('--duration',type=float,default=48)
    a=p.parse_args()
    report=json.loads((a.folder/'report.json').read_text())
    if report.get('schema')=='whole_task_step5_compact_v2':rebuild(a.folder,duration=a.duration)
    else:make_movie(a.folder,duration=a.duration,reuse_plan=a.reuse_plan)
