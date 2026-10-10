"""Explicit task inputs and reproducible unordered two-pose experiment sampling."""
import itertools
import json
from pathlib import Path
import random
import shutil
import numpy as np

from step1.needs import ROOT, OUTPUTS, ContinuousNeeds, sha256, demand
from step1.registry import task_poses, task_snapshot
from step3_scheculer.pair_scoring import TaskProblem
from step3_scheculer.sample_acceptance import SAMPLE_COUNT


def canonical_pair(poses):
    if len(poses) != 2 or len(set(poses)) != 2:
        raise ValueError('A pose pair must contain two distinct poses')
    from step1.cases import normalize_pose
    return tuple(sorted((normalize_pose(p) for p in poses), key=lambda p: int(p.split('_')[1])))


def sample_pairs(name, count=1, seed=None):
    available = list(itertools.combinations(task_poses(name), 2))
    if not 1 <= count <= len(available):
        raise ValueError('Pair count exceeds the available distinct unordered pairs')
    seed = random.SystemRandom().randrange(2**32) if seed is None else int(seed)
    return [canonical_pair(p) for p in random.Random(seed).sample(available, count)], seed


def pair_name(poses):
    first, second = canonical_pair(poses)
    return f'pose{first.split("_")[1]}+{second.split("_")[1]}'


def pair_folder(name, poses, stage):
    if stage not in ('step_1_needs', 'step2_local_support', 'step3_scheculer',
                     'step0_pose_selection', 'step4_connect_support'):
        raise ValueError('Unknown two-pose stage')
    return OUTPUTS/name/pair_name(poses)/stage


def task_folder(name, pose, poses=None):
    """Read explicit pair inputs; unqualified reads require identical saved loads."""
    from step1.cases import normalize_pose
    pose = normalize_pose(pose)
    cached = ROOT/'objects'/name/'poses'/pose
    if poses is None and (cached/'samples.json').is_file():
        return cached
    if poses is not None:
        if pose not in canonical_pair(poses):
            raise ValueError('Task pose is not a member of the pair')
        return pair_folder(name, poses, 'step_1_needs')/pose
    paths = sorted((OUTPUTS/name).glob(f'pose*+*/step_1_needs/{pose}/samples.json'))
    legacy = OUTPUTS/name/pose/'step_1_needs/samples.json'
    if legacy.exists():
        paths.append(legacy)
    if not paths:
        raise FileNotFoundError(f'No saved Step1 inputs for {name}/{pose}')
    signatures = {(sha256(p), sha256(p.with_name('needs.json'))) for p in paths}
    if len(signatures) != 1:
        raise ValueError(f'Conflicting Step1 inputs for {name}/{pose}; specify poses explicitly')
    return paths[0].parent


def prepare_pair_inputs(name, poses):
    """Reuse exact existing task samples when a new pair is first created."""
    for pose in canonical_pair(poses):
        target = task_folder(name, pose, poses)
        if (target/'samples.json').is_file():
            continue
        try:
            source = task_folder(name, pose)
        except FileNotFoundError:
            from step1.cases import selected_pose
            from step1.needs import build
            with selected_pose(pose):
                build(name, output_folder=target)
        else:
            target.mkdir(parents=True, exist_ok=True)
            for filename in ('needs.json','samples.json','setup.npz','setup.json','floor_contact.npz'):
                path=source/filename
                if path.is_file():shutil.copy2(path,target/filename)


def fixed_area_folder(name, poses, stage):
    """Keep fixed-area runs distinct from historical radius-optimized runs."""
    return pair_folder(name, poses, stage)/'fixed_area_1pct'


def completion_folder(name, poses, stage):
    return fixed_area_folder(name, poses, stage)/'terminal_expansion'


def read_task(name, pose, poses=None, *, folder=None):
    snapshot = task_snapshot(name, pose)
    folder = task_folder(name, pose, poses) if folder is None else Path(folder)
    domain_path, samples_path = folder/'needs.json', folder/'samples.json'
    domain = ContinuousNeeds.read(domain_path)
    samples = json.loads(samples_path.read_text())
    if domain.data['object'] != name or domain.data['pose_id'] != pose:
        raise ValueError('Task identity mismatch')
    provenance = domain.data['provenance']
    for path, expected in ((snapshot, provenance['setup_snapshot_sha256']),
                           (ROOT/'objects'/name/'mesh.stl', provenance['mesh_sha256']),
                           (ROOT/'objects'/name/'poses.json', provenance['poses_sha256']),
                           (domain_path, samples['provenance']['physical_domain_sha256'])):
        if sha256(path) != expected:
            raise ValueError(f'Stale pair input: {path}')
    raw_targets = np.asarray(samples['need_wrench'], float)
    np.testing.assert_allclose(raw_targets, demand(samples['pt_m'], samples['force_push_mg'],
        domain.com, domain.gravity), atol=1e-13, rtol=0)
    if len(raw_targets) != samples['count'] or len(raw_targets) != SAMPLE_COUNT:
        raise ValueError(f'Expected exactly {SAMPLE_COUNT} fixed Step1 loads')
    with np.load(snapshot) as data:
        floor = data['floor_contact_m'].copy()
    scale = np.r_[np.ones(3), np.ones(3)/domain.mesh.extents.max()]
    problem = TaskProblem(pose, domain, floor, np.ascontiguousarray(raw_targets*scale), scale)
    problem.random_sample_count = len(raw_targets)
    problem.inputs = [domain_path, samples_path, snapshot]
    return problem


def input_hashes(problems):
    return {str(p.relative_to(ROOT)): sha256(p) for problem in problems for p in problem.inputs}
