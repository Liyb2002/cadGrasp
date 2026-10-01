"""Fresh pair inputs and independent head searches for a construction walkthrough."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.cases import selected_pose
from step1.needs import build as build_loads
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step3_scheculer.run_independent import run_batch
from step0_pose_selection.run_floor_points import build_tasks


def prepare(name,poses):
    group=I.OUTPUTS/name/('pose'+'+'.join(p.split('_')[1] for p in poses))
    root=group/'step3_scheculer/independent_poses_floor2mm'
    if root.exists():raise ValueError('Fresh preparation requires an unused group input directory')
    tasks=[]
    for pose in poses:
        folder=root/pose/'step_1_needs'
        with selected_pose(pose):build_loads(name,output_folder=folder)
        tasks.append(read_task(name,pose,folder=folder))
    check=build_tasks(name,tasks,group/'step0_pose_selection')
    if not check['passed']:raise RuntimeError('Requested pose pair failed the original floor-demand check')
    result=run_batch(name,poses,jobs=2,output_root=root,floor_poses=poses)
    if result['passed_count']!=len(poses):raise RuntimeError('Fresh Step3 did not cover all original loads')
    return group


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',default='B');p.add_argument('--poses',nargs='+',default=['pose_1','pose_3'])
    args=p.parse_args();print(prepare(args.object,args.poses),flush=True)
