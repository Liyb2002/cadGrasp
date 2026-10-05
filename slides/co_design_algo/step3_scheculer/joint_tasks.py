"""N-pose paths and immutable Step1 inputs for simultaneous contact search."""
import shutil

from step1.cases import normalize_pose, selected_pose
from step1.needs import OUTPUTS
from step3_scheculer.pair_tasks import task_folder, read_task


def canonical_poses(poses):
    poses = tuple(normalize_pose(p) for p in poses)
    if len(poses) < 2 or len(set(poses)) != len(poses):
        raise ValueError('At least two distinct poses are required')
    return tuple(sorted(poses, key=lambda p: int(p.split('_')[1])))


def task_set_folder(name, poses, stage):
    if stage not in ('step_1_needs', 'step2_local_support', 'step3_scheculer', 'step4_floor_contact'):
        raise ValueError('Joint contact search only owns Step1–4')
    label = 'pose' + '+'.join(p.split('_')[1] for p in canonical_poses(poses))
    return OUTPUTS/name/label/stage


def output_folder(name, poses, stage):
    if stage == 'step_1_needs':
        raise ValueError('Step1 inputs are shared, not specific to a search algorithm')
    return task_set_folder(name, poses, stage)/'joint_weighted'


def prepare_tasks(name, poses):
    problems = []
    for pose in canonical_poses(poses):
        target = task_set_folder(name, poses, 'step_1_needs')/pose
        if not (target/'samples.json').is_file():
            try:
                source = task_folder(name, pose)
            except FileNotFoundError:
                from step1.needs import build
                with selected_pose(pose):
                    build(name, output_folder=target)
            else:
                shutil.copytree(source, target, dirs_exist_ok=True)
        problems.append(read_task(name, pose, folder=target))
    return problems
