"""Finish the accepted 13-region / five-installation fixture and two-arm movie."""
import argparse
import json
from pathlib import Path
import shutil
from whole_task.step1.common import ROOT, DEFAULT_OUTPUT, write_json, sha256
from .grouped import configure


def summarize(work, step4):
    report = json.loads((work/'report.json').read_text())
    metrics = json.loads((work/'evaluation.json').read_text())
    previous = json.loads((step4/'report.json').read_text())
    initial = previous['initial_support_pose_count']
    final = metrics['support_pose_count']
    merges = previous['installation_merge_count']
    if initial != 13 or final != 5 or initial-merges != final:
        raise ValueError('Expected thirteen independent installations merged eight times into five')
    groups = [dict(placement=i+1, poses=group['poses'])
              for i, group in enumerate(metrics['fixture_states'])]
    if sorted(len(g['poses']) for g in groups) != [2, 2, 3, 3, 3]:
        raise ValueError('Each support installation must serve two or three working areas')
    summary = dict(working_pose_count=len(report['poses']), initial_pose_count=initial,
        juxtaposition_insertion_count=merges,
        juxtaposition_count_definition='accepted installation merges; thirteen initial support installations become five',
        support_pose_count=final,
        object_pose_counts_per_support_pose=metrics['object_pose_counts_per_support_pose'],
        support_pose_groups=groups,
        execution_envelope=metrics['with_full_support'],
        maximum_single_execution_bbox=max((g['with_full_support'] for g in metrics['fixture_states']),
                                          key=lambda r: r['volume_litres']),
        occupied_xy_bbox_area_cm2=metrics['with_full_support']['dimensions_cm'][0]*metrics['with_full_support']['dimensions_cm'][1],
        material_volume_cm3=metrics['material_step5_cm3'],
        step4_material_volume_cm3=metrics['material_step4_complete_cm3'],
        material_reduction_percent=metrics['material_reduction_vs_step4_complete_percent'],
        full_continuous_force_certificate=True)
    metrics['task_summary'] = summary
    report['task_summary'] = summary
    from . import grouped
    for source in (Path(__file__), Path(grouped.__file__)):
        report['input_sha256'][str(source.resolve())] = sha256(source)
    write_json(work/'evaluation.json', metrics)
    write_json(work/'report.json', report)
    return summary


def run(step4=DEFAULT_OUTPUT/'B/step4/grouped13_focused', output=DEFAULT_OUTPUT/'B/step5', duration=96., grids=(3, 5)):
    configure()
    step4, output = Path(step4).resolve(), Path(output).resolve()
    work = ROOT/'whole_task/data/continuous13/B/step5_grouped_work'
    from .grouped_prune import solve
    from .validate import validate
    if not (work/'report.json').exists():
        solve(step4, work, grids=grids, video=False)
    else:
        report = json.loads((work/'report.json').read_text())
        if Path(report['step4']) != step4:
            raise ValueError('Checkpoint belongs to a different Step4')
        validate(work)
    summarize(work, step4)
    from .movie import make_movie
    report = json.loads((work/'report.json').read_text())
    if not report.get('video'):
        make_movie(work, duration=duration)
    validate(work)
    from .publish import publish
    report, data = publish(work, output)
    validate(data)
    shutil.rmtree(work)
    print('STEP5_GROUPED PASS', json.dumps(report['task_summary']), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step4', type=Path, default=DEFAULT_OUTPUT/'B/step4/grouped13_focused')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT/'B/step5')
    parser.add_argument('--duration', type=float, default=96.)
    args = parser.parse_args()
    run(args.step4, args.output, args.duration)
