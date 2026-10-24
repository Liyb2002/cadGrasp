"""Audit delivered movies against saved numerical results and source archives."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE.parent)
    args=parser.parse_args();folder=args.output_dir.resolve()
    assets=folder/'assets'
    selection=read(assets/'selection.json');source=read(assets/'data/source_manifest.json')
    archives={}
    for p in (assets/'data').glob('executed_sources*/*.py'):
        archives[p.name,sha(p)]=p
    cases=[];protected={};code={}
    for case in selection['results']:
        label=case['id'];data=assets/'data'/label
        report=read(data/'report.json');review=read(data/'motion_review.json')
        video=folder/report['output']
        if not report['complete']or not report['accepted_inputs_unchanged']:
            raise ValueError('Unfinished movie: '+label)
        hosts=report['support_host_order']
        if (report['pose_order']!=case['pose_order']or len(set(hosts))<3
                or any(a==b for a,b in zip(hosts,hosts[1:]))
                or not review['support_installations_interleaved']
                or review['pose_order']!=report['pose_order']):
            raise ValueError('Support installations are not interleaved: '+label)
        if sha(video)!=report['video_sha256']:
            raise ValueError('Changed movie: '+label)
        for p,h in report['source_hashes'].items():
            if sha(p)!=h:raise ValueError('Changed physical source: '+p)
            protected[p]=h
        for p,h in report['rendering_source_hashes'].items():
            key=Path(p).name,h
            if key not in archives:raise ValueError('Missing actual rendering source archive: '+p)
            code[str(archives[key].relative_to(ROOT))]=h
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
            '-select_streams','v:0','-show_streams','-show_format','-of','json',str(video)]))
        stream=probe['streams'][0];duration=float(probe['format']['duration'])
        frames=int(stream['nb_read_frames']);fps=float(Fraction(stream['avg_frame_rate']))
        if (abs(duration-20)>1e-3 or frames!=480 or fps!=24
                or stream['width']!=1600 or stream['height']!=900):
            raise ValueError('Unexpected output timing/size: '+label)
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        if (not review['all_delivered_frames_checked'] or review['sampled_frames']!=frames
                or review['task_models_camera_margin_px']<0 or review['active_gripper_camera_margin_px']<0
                or not review['joint_limits_passed'] or not review['source_hashes_unchanged']):
            raise ValueError('Motion/frame review failed: '+label)
        cases.append(dict(set=label,video=str(video.relative_to(ROOT)),sha256=sha(video),
            storyboard=str((assets/(label+'_storyboard.png')).relative_to(ROOT)),
            fixture_source=report['fixture_source'],object_poses=report['object_pose_count'],
            support_states=report['support_states'],pose_order=report['pose_order'],
            support_host_order=hosts,task_to_task_support_turns=len(hosts)-1,
            support_installations_interleaved=True,
            duration_s=duration,frames=frames,fps=fps,width=1600,height=900,
            station_yaw_offset_deg=report.get('station_yaw_offset_deg',0.),
            free_transfer_interpolation=report.get('free_transfer_interpolation','Cartesian_IK'),
            full_decode_passed=True,all_delivered_frames_checked=True,
            task_model_margin_px=review['task_models_camera_margin_px'],
            active_gripper_margin_px=review['active_gripper_camera_margin_px'],
            force_arrows_per_task=review['force_arrows_per_task']))
    for p,h in source.get('prior_videos_and_previews',{}).items():
        if sha(p)!=h:raise ValueError('Prior video/preview changed: '+p)
    result=dict(complete=True,passed=True,movies=len(cases),cases=cases,
        new_result_source='step5_show_all52_20261014_02',object_poses=sum(c['object_poses']for c in cases),
        original_numerical_sources_unchanged=True,
        prior_videos_and_previews_removed_by_user=bool(source.get('prior_videos_and_previews_removed_by_user')),
        physical_source_count=len(protected),protected_sources=protected,actual_source_archives=code,
        kinematic_workflow=True,dynamics_integrated=False,continuous_collision_certified=False)
    (assets/'data/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('MOVIES VERIFIED',len(cases),'frames',sum(c['frames']for c in cases),'physical sources',len(protected),flush=True)


if __name__=='__main__':main()
