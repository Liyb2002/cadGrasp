"""Independent full-solid replay plus genuine installation-sharing checks."""
import argparse
import json
from pathlib import Path
import numpy as np

from .validate_families import validate as validate_physics
from subdivide_task.step1.common import sha256, write_json


def validate(folder):
    folder = Path(folder)
    result = validate_physics(folder)
    report = json.loads((folder/'report.json').read_text())
    execution = json.loads((folder/'execution.json').read_text())['poses']
    groups = []
    for row in execution:
        matrix = np.asarray(row['world_from_support'])
        for group in groups:
            if np.allclose(matrix, group['transform'], atol=1e-10, rtol=0):
                group['poses'].append(row['pose']); break
        else:
            groups.append(dict(transform=matrix, poses=[row['pose']]))
    count = report['requested_support_pose_count']
    assert 3 <= count <= 5 and len(groups) == count
    assert sorted(len(g['poses']) for g in groups) == sorted(report['support_installations']['group_sizes'])
    assert max(len(g['poses']) for g in groups)-min(len(g['poses']) for g in groups) <= 1
    assert sorted(p for g in groups for p in g['poses']) == sorted(report['poses'])
    with np.load(folder/'step4.2/layout.npz') as z:
        hosts = z['hosts'][z['active']]
        assert len(np.unique(hosts)) == count
        # Exactly one full centered transform for each host, and vice versa.
        for host in np.unique(hosts):
            rows = [execution[k] for k in range(len(hosts)) if hosts[k] == host]
            for row in rows:
                np.testing.assert_allclose(row['world_from_support'], rows[0]['world_from_support'], atol=1e-10, rtol=0)
    trace = json.loads((folder/'search_trace.json').read_text())
    initial = [r for r in trace if r['phase'] == 'thirteen-independent-installations']
    merges = [r for r in trace if r['phase'] == 'merge-installations']
    if initial:
        assert initial[0]['support_group_count'] == 13
        assert [r['support_group_count'] for r in merges] == list(range(12, count-1, -1))
        assert len(merges) == 13-count
    result.update(actual_installation_sharing_verified=True, actual_support_pose_count=len(groups),
                  support_groups=[g['poses'] for g in groups], initial_independent_pose_count=13,
                  cold_merge_trace_verified=bool(initial), grouped_validator_sha256=sha256(__file__))
    write_json(folder/'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    result = validate(parser.parse_args().folder)
    print('GROUPED INDEPENDENT VALIDATION PASS', result['support_groups'])
