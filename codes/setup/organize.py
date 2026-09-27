"""Publish per-object videos and lossless slices of the saved continuous trajectory."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from fractions import Fraction
import xml.etree.ElementTree as ET

import numpy as np
from sequence_export import ROOT, digest, write


def attach_video_timing(folder,index):
    video=folder/'video.mp4'
    if not video.exists() or shutil.which('ffprobe') is None:return
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=avg_frame_rate','-of','json',str(video)]))
    fps=float(Fraction(info['streams'][0]['avg_frame_rate']))
    index.update(video='../video.mp4',video_fps=fps)
    for row in index['segments']:
        row['video_start_time_s']=row['first_full_frame']/fps
        row['video_end_time_s']=row['last_full_frame']/fps


def refresh_task_references(folder, record):
    pose_hash=digest(folder/'poses.json')
    for pose in record['poses']:
        task=folder/'tasks'/pose['pose_id']
        with np.load(task/'setup.npz') as z:
            fields=dict(z)
        fields['poses_sha256']=pose_hash
        np.savez_compressed(task/'setup.npz',**fields)
        report=json.loads((task/'setup.json').read_text())
        report['source_snapshot_sha256']=digest(task/'setup.npz')
        if 'incoming_trajectory' in pose:
            report['incoming_trajectory']='../../'+pose['incoming_trajectory']
        write(task/'setup.json',report)


def relocate_simulation_assets(folder):
    """Move backend meshes without changing their bytes or the object's geometry."""
    old=folder/'scene.xml'
    if not old.exists():
        return
    destination=folder.parent/'_simulation_assets'/folder.name
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'scene.xml').exists():
        raise FileExistsError(f'Refusing to replace an existing scene: {destination}')
    root=ET.parse(old).getroot()
    meshdir=root.find('compiler').get('meshdir','.')
    for mesh in root.findall('asset/mesh'):
        source=(folder/meshdir/mesh.get('file')).resolve()
        if source.is_relative_to((folder/'collision').resolve()):
            source=destination/'collision'/source.relative_to((folder/'collision').resolve())
        mesh.set('file',os.path.relpath(source,destination))
    root.find('compiler').set('meshdir','.')
    if (folder/'collision').exists():
        shutil.move(str(folder/'collision'),destination/'collision')
    (destination/'scene.xml').write_text(ET.tostring(root,encoding='unicode')+'\n')
    old.unlink()


def publish_segments(folder, invalidate_video=False):
    record=json.loads((folder/'poses.json').read_text())
    out=folder/'trajectories';out.mkdir(exist_ok=True)
    trajectory_hash=digest(folder/'trajectory.npz')
    rows=[]
    with np.load(folder/'trajectory.npz') as z:
        times=z['time_s']
        # Both adjacent files include the same last stored frame before the
        # nominal boundary. No interpolation, time reset or synthetic pose.
        ends=np.searchsorted(times,z['target_times_s'],side='right')-1
        ends=np.clip(ends,0,len(times)-1)
        assert len(ends)==10 and np.all(np.diff(ends)>0)
        assert ends[-1]==len(times)-1
        first=0
        for i,last in enumerate(ends):
            source='rest' if i==0 else f'pose_{i}'
            target=f'pose_{i+1}'
            filename=f'{source}_to_{target}.npz'
            frames=np.arange(first,last+1)
            robot_mask=(z['robot_frame_indices']>=first)&(z['robot_frame_indices']<=last)
            robot_global_frames=z['robot_frame_indices'][robot_mask]
            requested_start=0. if i==0 else float(z['target_times_s'][i-1])
            requested_end=float(z['target_times_s'][i])
            fields={key:z[key][frames] for key in ('qpos','mocap_pos','mocap_quat','T_world_object')}
            for key in ('ideal_grasp_active','ideal_grasp_eq_data'):
                if key in z:fields[key]=z[key][frames]
            fields.update(time_s=times[frames]-times[first],sequence_time_s=times[frames],
                full_frame_indices=frames,robot_q=z['robot_q'][robot_mask],
                robot_frame_indices=robot_global_frames-first,
                robot_time_s=times[robot_global_frames]-times[first],
                robot_sequence_time_s=times[robot_global_frames],
                source_pose_id=source,destination_pose_id=target,
                source_target_transform=z['rest_transform'] if i==0 else z['target_transforms'][i-1],
                destination_target_transform=z['target_transforms'][i],
                requested_start_time_s=requested_start,requested_end_time_s=requested_end,
                full_trajectory_sha256=trajectory_hash)
            np.savez_compressed(out/filename,**fields)
            path='trajectories/'+filename
            record['poses'][i]['incoming_trajectory']=path
            record['transitions'][i]['trajectory_file']=path
            rows.append(dict(source=source,destination=target,file=filename,
                start_time_s=requested_start,end_time_s=requested_end,
                sampled_start_time_s=float(times[first]),sampled_end_time_s=float(times[last]),
                first_full_frame=int(first),last_full_frame=int(last),
                object_samples=len(frames),robot_joint_samples=len(robot_global_frames)))
            if 'grasp_id' in record['transitions'][i]:
                rows[-1]['grasp_id']=record['transitions'][i]['grasp_id']
                rows[-1]['regrasp_events']=[e for e in record.get('regrasp_events',[])
                    if requested_start<=e['start_time_s']<requested_end]
            first=last
    index=dict(schema='cadgrasp_trajectory_segments_v1',object=folder.name,
        full_trajectory='../trajectory.npz',full_trajectory_sha256=trajectory_hash,
        sampling='Exact slices of the stored 33 ms trajectory; robot_q retains its original sample times. Adjacent files share a boundary frame. The first file includes approach, grasp and pickup from rest. Exact event times can lie between stored frames.',
        fields=dict(time_s='Seconds since this file\'s first stored frame',
            sequence_time_s='Seconds on the original continuous simulation clock',
            T_world_object='Actual object transforms, one per stored frame',
            qpos='Simulated hand free joint (7), finger joints (2), object free joint (7)',
            mocap_pos='Commanded hand driver position',mocap_quat='Commanded hand driver quaternion, wxyz',
            robot_q='KUKA seven joint angles in radians at robot_sequence_time_s; no interpolation',
            source_target_transform='Nominal source target; actual first state is T_world_object[0]',
            destination_target_transform='Nominal destination; actual last state is T_world_object[-1]'),
        segments=rows)
    if not invalidate_video:attach_video_timing(folder,index)
    write(out/'index.json',index)
    record['trajectory_segments']='trajectories/index.json'
    if invalidate_video:
        (folder/'video.mp4').unlink(missing_ok=True)
        for key in ('video','video_sha256','video_trajectory_sha256'):record.pop(key,None)
    elif (folder/'video.mp4').exists():
        record.update(video='video.mp4',video_sha256=digest(folder/'video.mp4'),
                      video_trajectory_sha256=trajectory_hash)
    write(folder/'poses.json',record)
    tasks=json.loads((folder/'tasks.json').read_text())
    tasks['trajectory_segments']='trajectories/index.json'
    tasks['incoming_trajectories']={r['destination']:'trajectories/'+r['file'] for r in rows}
    write(folder/'tasks.json',tasks)
    refresh_task_references(folder,record)


def publish_video(folder,source):
    destination=folder/'video.mp4'
    if Path(source).resolve()!=destination.resolve():shutil.copy2(source,destination)
    record=json.loads((folder/'poses.json').read_text())
    record.update(video='video.mp4',video_sha256=digest(destination),
                  video_trajectory_sha256=digest(folder/'trajectory.npz'))
    write(folder/'poses.json',record)
    refresh_task_references(folder,record)
    index_path=folder/'trajectories/index.json'
    if index_path.exists():
        index=json.loads(index_path.read_text())
        attach_video_timing(folder,index)
        write(index_path,index)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='*')
    parser.add_argument('--import-existing-videos',action='store_true',
        help='Import simulation/videos only when they are known to match the saved trajectories')
    args=parser.parse_args()
    names=args.objects or json.loads((ROOT/'objects/cases.json').read_text())['active_objects']
    for name in names:
        folder=ROOT/'objects'/name
        video=folder/'video.mp4'
        if args.import_existing_videos:
            source=ROOT/'simulation/videos'/f'{name}_grounded_sequence.mp4'
            if not source.exists():raise FileNotFoundError(source)
            shutil.copy2(source,video)
        else:
            if not video.exists():raise FileNotFoundError(f'{video}: run sequence.py {name} --replay')
            record=json.loads((folder/'poses.json').read_text())
            if record.get('video_trajectory_sha256')!=digest(folder/'trajectory.npz'):
                raise ValueError(f'{name}: video provenance is missing or stale; rerender it')
        relocate_simulation_assets(folder)
        publish_segments(folder)
        print(f'{name}: video.mp4 + 10 trajectory segments',flush=True)


if __name__=='__main__':main()
