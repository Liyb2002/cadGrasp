"""Object and task inputs have one source of truth under objects/."""
import json
from pathlib import Path
import re

OBJECT_ROOT = Path(__file__).resolve().parents[2] / 'objects'


def object_folder(name, root=OBJECT_ROOT):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError(f'Invalid object name: {name!r}')
    folder = Path(root) / name
    if not (folder / 'mesh.stl').is_file():
        raise FileNotFoundError(f'Missing object mesh: {folder / "mesh.stl"}')
    return folder


def active_objects(root=OBJECT_ROOT):
    source = Path(root) / 'cases.json'
    config = json.loads(source.read_text())
    names = config['active_objects']
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError(f'{source}: active_objects must be a nonempty unique list')
    for name in names:
        object_folder(name, root)
    return tuple(names)


def task_poses(name, root=OBJECT_ROOT):
    from codes.precompute_objects.cases import normalize_pose
    folder = object_folder(name, root)
    precomputed = (folder / 'pose_sets.json').is_file()
    manifest = json.loads((folder / ('poses.json' if precomputed else 'tasks.json')).read_text())
    if manifest['object'] != name:
        raise ValueError(f'{name}: task manifest belongs to another object')
    poses = [row['pose_id'] for row in manifest['poses']] if precomputed else manifest['poses']
    if not poses or len(set(poses)) != len(poses):
        raise ValueError(f'{name}: task poses must be nonempty and unique')
    for pose in poses:
        if normalize_pose(pose) != pose:
            raise ValueError(f'{name}: noncanonical task pose {pose}')
        if not (folder / ('poses' if precomputed else 'tasks') / pose / 'setup.npz').is_file():
            raise FileNotFoundError(f'{name}/{pose}: missing task setup.npz')
    return tuple(poses)


def task_snapshot(name, pose, root=OBJECT_ROOT):
    if name not in active_objects(root):
        raise ValueError(f'{name}: not in objects/cases.json active_objects')
    if pose not in task_poses(name, root):
        raise FileNotFoundError(f'{name}/{pose}: no configured target task')
    folder = object_folder(name, root)
    return folder / ('poses' if (folder / 'pose_sets.json').is_file() else 'tasks') / pose / 'setup.npz'


def active_cases(root=OBJECT_ROOT):
    return [(name, pose) for name in active_objects(root) for pose in task_poses(name, root)]
