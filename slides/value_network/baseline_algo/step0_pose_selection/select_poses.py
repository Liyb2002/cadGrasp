"""Step0: sample distinct n-pose combinations until original floor demands pass."""
import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.cases import selected_pose
from step1.needs import build as build_loads
from step1.registry import task_poses
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task, task_folder, input_hashes
from step3_scheculer.joint_tasks import canonical_poses, task_set_folder
from step0_pose_selection import floor_points as F

SCHEMA = 'random_n_pose_floor_selection_v1'


def shuffled_combinations(available, n, seed):
    """Every unordered set appears at most once; exhaustion always terminates."""
    available = tuple(sorted(available, key=lambda p: int(p.split('_')[1])))
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= len(available):
        raise ValueError(f'n must be between 2 and {len(available)} for this multi-pose baseline')
    if len(set(available)) != len(available):
        raise ValueError('Available poses must be distinct')
    groups = list(itertools.combinations(available, n))
    random.Random(seed).shuffle(groups)
    return groups


def choose(available, n, seed, evaluate, progress=None):
    """Pure selection loop. Never advance downstream on failed floor demands."""
    groups = shuffled_combinations(available, n, seed)
    attempts = []
    for group in groups:
        verdict = evaluate(group)
        attempts.append(dict(poses=list(group), **verdict))
        if progress:
            progress(attempts[-1], len(attempts), len(groups))
        if verdict['passed']:
            return dict(passed=True, status='pose_set_selected', selected_poses=list(group),
                        attempted_count=len(attempts), total_combinations=len(groups), attempts=attempts)
    return dict(passed=False, status='no_floor_compatible_pose_set', selected_poses=None,
                attempted_count=len(attempts), total_combinations=len(groups), attempts=attempts)


class TaskCache:
    """Prepare immutable load inputs without creating rejected pose-group stages."""
    def __init__(self, name):
        self.name = name
        self.root = I.OUTPUTS/name/'step0_pose_selection'
        self.tasks = {}
        self.sources = {}
        self.clouds = {}

    def get(self, pose):
        if pose not in self.tasks:
            # Reuse exact saved samples. Only genuinely missing poses are sampled.
            try:
                source = task_folder(self.name, pose)
            except FileNotFoundError:
                source = self.root/'task_inputs'/pose
                if not (source/'samples.json').is_file():
                    with selected_pose(pose):
                        build_loads(self.name, output_folder=source)
            task = read_task(self.name, pose, folder=source)
            self.tasks[pose], self.sources[pose] = task, source
            self.clouds[pose] = F.pressure_centers(task.targets/task.scale, task.domain.com)[0]
        return self.tasks[pose]

    def evaluate(self, poses):
        tasks = [self.get(p) for p in poses]
        frames = F.validate_frames([t.domain.data['frame']['T_world_mesh'] for t in tasks])
        F.validate_task_geometry(tasks, frames)
        result = F.check_floor_points([self.clouds[p] for p in poses], frames, poses)
        return dict(passed=all(row['passed'] for row in result.rows),
                    violating_counts=[row['violating_sample_count'] for row in result.rows])


def publish_check(name, poses, cache):
    """Publish Step0 before Step1; data come from the exact inputs just checked."""
    from step0_pose_selection.run_floor_points import build_tasks
    output = task_set_folder(name, poses, 'step0_pose_selection')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.step0_build_', dir=output.parent) as tmp:
        staging = Path(tmp)/'step0_pose_selection'
        report = build_tasks(name, [cache.get(p) for p in poses], staging)
        report['load_input_folders'] = {p: str(cache.sources[p].relative_to(I.ROOT)) for p in poses}
        I.save(staging/'report.json', report)
        if output.exists():
            if output.is_symlink():
                raise ValueError('Refusing to replace a symlink')
            shutil.rmtree(output)
        staging.rename(output)
    return output/'report.json'


def materialize_step1(name, poses, check_path):
    """After Step0 passes, populate Step1 using the same source bytes."""
    report = I.check_report(check_path)
    if not report['passed'] or report['object'] != name or report['poses'] != list(poses):
        raise ValueError('Step1 requires an accepted Step0 check for exactly these poses')
    root = task_set_folder(name, poses, 'step_1_needs')
    for pose in poses:
        source = I.ROOT/report['load_input_folders'][pose]
        target = root/pose
        if source.resolve() == target.resolve():
            continue
        for filename in ('needs.json', 'samples.json', 'examples.json'):
            src, dst = source/filename, target/filename
            if not src.is_file():
                continue
            if dst.exists() and I.sha256(dst) != I.sha256(src):
                raise ValueError(f'Step1 differs from Step0 inputs: {dst}')
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    tasks = [read_task(name, p, folder=root/p) for p in poses]
    # Check the consumer inputs against exactly the floor data used at Step0.
    for task in tasks:
        with np.load(Path(check_path).parent/f'data/floor_contact_{task.pose}.npz') as arrays:
            np.testing.assert_array_equal(arrays['load_wrenches'], task.targets/task.scale)
    return tasks


def accepted_tasks(name, poses):
    """Guard direct Step3 invocations as well as the full pipeline entry."""
    check = task_set_folder(name, poses, 'step0_pose_selection')/'report.json'
    if not check.is_file():
        raise ValueError('Run Step0 pose selection before generating candidates or searching heads')
    return materialize_step1(name, poses, check)


@dataclass
class Selection:
    report: dict
    ledger_path: Path
    check_path: Path | None


def select(name, n, seed=20260929):
    available = task_poses(name)
    shuffled_combinations(available, n, seed)  # validate before creating output
    cache = TaskCache(name)
    ledger_path = cache.root/f'selection_n{n}_seed{seed}.json'
    def progress(attempt, done, total):
        print(f'STEP0 {done}/{total}: {"+".join(attempt["poses"])} '
              f'{"PASS" if attempt["passed"] else "REJECT"} {attempt["violating_counts"]}', flush=True)
    result = choose(available, n, seed, cache.evaluate, progress)
    result.update(schema=SCHEMA, complete=True, object=name, n=n, seed=seed,
        available_poses=list(available), sample_count_per_pose=32768,
        repeated_combinations=False, complete_fixture_verified=False,
        scope='Only fixed-pose floor compatibility of original demands; no head or body search',
        provenance=dict(inputs=input_hashes(list(cache.tasks.values())),
                        code=I.hashes([Path(__file__), Path(F.__file__)])))
    check = None
    if result['passed']:
        check = publish_check(name, result['selected_poses'], cache)
        result['selected_check'] = str(check.relative_to(I.ROOT))
    I.save(ledger_path, result)
    return Selection(result, ledger_path, check)


def choose_many(available, n, seed, evaluate, count=2, progress=None):
    """Continue the same random permutation until count distinct sets pass."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Positive integer group count required')
    groups = shuffled_combinations(available, n, seed)
    attempts, selected = [], []
    for group in groups:
        verdict = evaluate(group)
        attempts.append(dict(poses=list(group), **verdict))
        if progress:
            progress(attempts[-1], len(attempts), len(groups))
        if verdict['passed']:
            selected.append(list(group))
            if len(selected) == count:
                break
    return dict(passed=len(selected) == count,
        status='pose_sets_selected' if len(selected) == count else 'insufficient_floor_compatible_pose_sets',
        requested_count=count, selected_groups=selected, attempted_count=len(attempts),
        total_combinations=len(groups), exhausted=len(attempts) == len(groups), attempts=attempts)


def select_many(name, n, seed=20260929, count=2, cache=None):
    available = task_poses(name)
    cache = TaskCache(name) if cache is None else cache
    if cache.name != name:
        raise ValueError('Task cache belongs to a different object')
    def progress(attempt, done, total):
        print(f'STEP0 n={n} {done}/{total}: {"+".join(attempt["poses"])} '
              f'{"PASS" if attempt["passed"] else "REJECT"} {attempt["violating_counts"]}', flush=True)
    result = choose_many(available, n, seed, cache.evaluate, count, progress)
    result.update(schema='random_n_pose_floor_batch_v1', complete=True, object=name,
        n=n, seed=seed, sample_count_per_pose=32768, repeated_combinations=False,
        complete_fixture_verified=False,
        provenance=dict(inputs=input_hashes(list(cache.tasks.values())),
                        code=I.hashes([Path(__file__), Path(F.__file__)])))
    ledger = cache.root/f'selection_n{n}_seed{seed}_groups{count}.json'
    selections = []
    for poses in result['selected_groups']:
        check = publish_check(name, poses, cache)
        selections.append(Selection(dict(passed=True, status='pose_set_selected',
            selected_poses=poses, n=n, seed=seed), ledger, check))
    result['selected_checks'] = [str(s.check_path.relative_to(I.ROOT)) for s in selections]
    I.save(ledger, result)
    return selections, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20260929)
    args = parser.parse_args(argv)
    selection = select(args.object, args.n, args.seed)
    print('STEP0 COMPLETE', selection.report['status'], selection.ledger_path, flush=True)
    return 0 if selection.report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
