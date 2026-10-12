"""Replay generated pose subsets through the actual baseline Step0 consumer."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

from sequence_export import ROOT, digest, write
sys.path.insert(0, str(ROOT/'slides/baseline_algo'))
from step0_pose_selection.select_poses import TaskCache, publish_check
from step0_pose_selection import floor_points as F
from step3_scheculer import contacts as I
from step1.registry import task_poses


def verify(name):
    poses = task_poses(name)
    folder = ROOT/'objects'/name
    expected = json.loads((folder/'floor_compatibility.json').read_text())
    cache = TaskCache(name)
    for pose in poses:
        task = cache.get(pose)
        # Sample values must agree before comparing feasibility; small rounding
        # differences from mesh reconstruction do not change the physical inputs.
        with np.load(folder/'pose_search'/f'{pose}.npz') as saved:
            np.testing.assert_allclose(task.targets/task.scale, saved['load_wrenches'], atol=1e-12, rtol=0)
    frames = np.asarray([cache.tasks[p].domain.data['frame']['T_world_mesh'] for p in poses])
    F.validate_task_geometry(list(cache.tasks.values()), frames)
    checks = F.check_floor_points([cache.clouds[p] for p in poses], frames, poses)
    counts = np.array([[int(np.count_nonzero(height[:, j] < -F.TOL)) for j in range(len(poses))]
                       for height in checks.heights])
    np.testing.assert_array_equal(counts, expected['directed_violating_counts'])
    reports = {}
    for size in ('3', '4', '5'):
        witness = expected['witnesses'][size]
        result = cache.evaluate(witness)
        assert result['passed'], witness
        path = publish_check(name, witness, cache)
        I.check_report(path)
        reports[size] = dict(poses=witness, passed=True, report=str(path.relative_to(ROOT)))
    report = dict(passed=True, object=name, pose_count=len(poses), sample_count_per_pose=32768,
        source_pose_manifest_sha256=digest(folder/'poses.json'),
        generated_pose_report_sha256=digest(folder/'floor_compatibility.json'),
        all_directed_pair_counts_match=True, compatible_group_counts=expected['compatible_group_counts'],
        witnesses=reports, head_search_run=False, complete_fixture_verified=False,
        code_sha256=digest(__file__))
    write(cache.root/'pose_set_validation.json', report)
    print('BASELINE POSE SET CHECK', json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    verify(parser.parse_args().object)
