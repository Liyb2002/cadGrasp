"""Replay full Step5 physics, occupancy and shared movie installation metadata."""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
from whole_task.step1.common import write_json
from .grouped import configure, installation_groups


def validate(folder):
    configure()
    from .validate import validate as full_validate
    folder = Path(folder)
    result = full_validate(folder)
    report = json.loads((folder/'report.json').read_text())
    summary = report['task_summary']
    if (summary['initial_pose_count'], summary['juxtaposition_insertion_count'], summary['support_pose_count']) != (13, 8, 5):
        raise ValueError('Grouped Step5 installation counts differ')
    if sorted(summary['object_pose_counts_per_support_pose']) != [2, 2, 3, 3, 3]:
        raise ValueError('Grouped Step5 membership differs')
    with zipfile.ZipFile(folder/'video_motion.zip') as bundle:
        plan = json.loads(bundle.read('plan.json'))
    groups = installation_groups({i: dict(fixture=T) for i, T in enumerate(plan['fixture_poses'])})
    if len(groups) != 5 or plan['support_installation_groups'] != groups:
        raise ValueError('Movie does not use the five actual installation transforms')
    if plan['execution_order'] != [k for group in groups for k in group]:
        raise ValueError('Movie installation order differs')
    result.update(shared_support_installations_verified=5,
                  same_support_transform_within_group=True,
                  support_group_sizes=[len(group) for group in groups])
    write_json(folder/'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    print(json.dumps(validate(parser.parse_args().folder), indent=2))
