"""Finish the current 13-region Step4 with sparse feet, pruning and a movie."""
import argparse
import json
from pathlib import Path
import shutil

from whole_task_2.step1.common import DEFAULT_OUTPUT, ROOT, write_json


def summarize(work, step4):
    report = json.loads((work/'report.json').read_text())
    metrics = json.loads((work/'evaluation.json').read_text())
    trace = json.loads((step4/'search_trace.json').read_text())
    inserted = sum(row['phase'] == 'inserted' for row in trace)
    initial = sum(row['phase'] == 'single-registered-task' for row in trace)
    if initial != 1 or inserted != len(report['poses'])-1:
        raise ValueError('Expected one initial task and twelve automatic insertions')
    summary = dict(working_pose_count=len(report['poses']),
        initial_pose_count=initial, juxtaposition_insertion_count=inserted,
        juxtaposition_count_definition='number of inserted working regions after the initial region',
        legacy_nonidentity_seat_count=metrics['juxtaposed_pose_count'],
        legacy_count_note='nonidentity seat transforms include the initial task after changing material origin',
        support_pose_count=metrics['support_pose_count'],
        object_pose_counts_per_support_pose=metrics['object_pose_counts_per_support_pose'],
        support_pose_groups=[dict(placement=i+1, poses=group['poses'])
                             for i, group in enumerate(metrics['fixture_states'])],
        execution_envelope=metrics['with_full_support'],
        maximum_single_execution_bbox=max((group['with_full_support'] for group in metrics['fixture_states']),
                                          key=lambda row: row['volume_litres']),
        occupied_xy_bbox_area_cm2=metrics['with_full_support']['dimensions_cm'][0]*metrics['with_full_support']['dimensions_cm'][1],
        material_volume_cm3=metrics['material_step5_cm3'],
        step4_material_volume_cm3=metrics['material_step4_complete_cm3'],
        material_reduction_percent=metrics['material_reduction_vs_step4_complete_percent'],
        full_continuous_force_certificate=True)
    metrics['task_summary'] = summary
    report['task_summary'] = summary
    write_json(work/'evaluation.json', metrics)
    write_json(work/'report.json', report)
    return summary


def run(step4=DEFAULT_OUTPUT/'B/step4/continuous13',
        output=DEFAULT_OUTPUT/'B/step5', duration=96., grids=(3,5)):
    step4, output = Path(step4).resolve(), Path(output).resolve()
    work = ROOT/'whole_task_2/data/continuous13/B/step5_work'
    from .run import solve
    from .validate import validate
    if not (work/'report.json').exists():
        solve(step4, work, grids=grids, video=False)
    else:
        validate(work)
    summarize(work, step4)
    from .movie import make_movie
    if not (work/'report.json').read_text().count('"video":'):
        make_movie(work, duration=duration)
    from .publish import publish
    report, data = publish(work, output)
    validate(data)
    shutil.rmtree(work)
    print('STEP5_13 PASS', json.dumps(report['task_summary']), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step4', type=Path, default=DEFAULT_OUTPUT/'B/step4/continuous13')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT/'B/step5')
    parser.add_argument('--duration', type=float, default=96.)
    args = parser.parse_args()
    run(args.step4, args.output, args.duration)
