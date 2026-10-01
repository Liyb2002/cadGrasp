"""Replay selection from raw push forces and object-local floor halfspaces."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I


def audit(name, seed):
    root = I.OUTPUTS/name/'step0_pose_selection'
    folders = sorted((root/'task_inputs').glob('pose_*'), key=lambda p: int(p.name.split('_')[1]))
    poses = [p.name for p in folders]
    frames, points, inputs = [], [], []
    for folder in folders:
        source = folder/'samples.json'
        samples = json.loads(source.read_text())
        setup_path = I.ROOT/'objects'/name/'tasks'/folder.name/'setup.npz'
        with np.load(setup_path) as setup:
            frame, com = setup['T_world_mesh'].copy(), setup['com_m'].copy()
        push = np.asarray(samples['force_push_mg'])
        applied_at = np.asarray(samples['pt_m'])
        # Balance the complete object + massless fixture about the WORLD origin.
        force = np.array([0., 0., 1.])-push
        moment = np.cross(com, [0., 0., 1.])-np.cross(applied_at, push)
        ground = np.zeros_like(force)
        ground[:, 0] = -moment[:, 1]/force[:, 2]
        ground[:, 1] = moment[:, 0]/force[:, 2]
        np.testing.assert_allclose(np.cross(ground, force)[:, :2], moment[:, :2], atol=1e-12, rtol=0)
        points.append((ground-frame[:3, 3])@frame[:3, :3])
        frames.append(frame); inputs.extend([source, setup_path])
    # Each target floor is a halfspace in the SAME object-local coordinates.
    heights = np.asarray([[p@f[2, :3]+f[2, 3] for f in frames] for p in points])
    failures = heights < -1e-9
    counts = failures.sum(axis=2)
    compatible = (counts == 0) & (counts.T == 0)
    groups = {}
    for n in (2, 3, 4):
        groups[n] = [list(g) for g in itertools.combinations(range(len(poses)), n)
                     if all(compatible[i, j] for i, j in itertools.combinations(g, 2))]
    matched = 0
    for path in sorted(root.glob(f'selection_n*_seed{seed}_groups*.json')):
        ledger = json.loads(path.read_text()); inputs.append(path)
        for attempt in ledger['attempts']:
            ids = [poses.index(p) for p in attempt['poses']]
            actual = [int(failures[i, ids].any(axis=0).sum()) for i in ids]
            if actual != attempt['violating_counts'] or (not any(actual)) != attempt['passed']:
                raise AssertionError(f'Selection mismatch: {attempt["poses"]}')
            matched += 1
    report = dict(object=name, seed=seed, poses=poses, passed=True,
        formulation='Raw push forces -> world-origin equilibrium -> object-local floor halfspaces',
        tolerance_m=1e-9, sample_count_per_pose=32768, checked_attempts=matched,
        directed_violating_counts=counts.tolist(), minimum_heights_m=heights.min(axis=2).tolist(),
        compatible_groups={str(n): [[poses[i] for i in group] for group in rows] for n, rows in groups.items()},
        scope='Exhaustive combinations of these saved task poses and original finite loads only',
        provenance=dict(inputs=I.hashes(inputs), code=I.hashes([Path(__file__)])))
    destination = root/'independent_selection_check.json'
    I.save(destination, report)
    print(json.dumps({k: report[k] for k in ('checked_attempts', 'compatible_groups')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--seed', type=int, default=20260929)
    args = parser.parse_args()
    audit(args.object, args.seed)
