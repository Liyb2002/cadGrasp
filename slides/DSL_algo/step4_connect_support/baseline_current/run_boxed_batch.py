"""Rerun Step4.1/4.2 on every saved Step0 group except pose1+3.

Hash all Step3 source/results and the entire excluded reference before/after.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from step4_connect_support.baseline_current import space_budget as B


def protected_hashes(object_root):
    roots = [p for p in BASE.parent.iterdir() if p.is_dir() and p.name.startswith('step3')]
    roots += list(object_root.glob('pose*+*/step3_scheculer'))
    roots += [object_root/'independent_poses', object_root/'pose1+3']
    paths = {p for root in roots if root.exists() for p in root.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    return {str(p.relative_to(B.ROOT)): B.digest(p) for p in sorted(paths)}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def run(name):
    root = B.OUTPUT/name
    groups = sorted((p.parent.parent for p in root.glob('pose*+*/step0_pose_selection/report.json')
                     if p.parent.parent.name != 'pose1+3'),
                    key=lambda p: (len(p.name.split('+')), [int(i) for i in p.name[4:].split('+')]))
    if not groups:
        raise ValueError('No retained Step0 groups to rerun')
    protected = protected_hashes(root)
    destination = groups[-1]/'step4/data/boxed_support'
    destination.mkdir(parents=True, exist_ok=True)
    save(destination/'protected_inputs.json', protected)
    ledger = dict(complete=False, object=name, excluded_groups=['pose1+3'],
        groups=[g.name for g in groups], saved_step3_exit_directions_preserved=True,
        step3_rerun=False, results=[], protected_file_count=len(protected))
    save(destination/'batch_report.json', ledger)
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for group in groups:
        began = time.monotonic()
        out = group/'step4/data/boxed_support'
        out.mkdir(parents=True, exist_ok=True)
        row = dict(group=group.name)
        with (out/'pipeline.log').open('w', buffering=1) as log:
            for script, options in (
                ('space_budget.py', []),
                ('deterministic_space.py', [])):
                command = [sys.executable, str(BASE/script), '--object', name, '--groups', group.name, *options]
                code = subprocess.run(command, cwd=B.ROOT, env=environment,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
                row[script] = dict(returncode=code)
                if code:
                    row.update(status='pipeline_error', passed=False, constructed=False)
                    break
            else:
                report = json.loads((out/'report.json').read_text())
                row.update(status=report['status'], passed=report['passed'], constructed=report['constructed'],
                    attempt_count=len(report.get('attempts', [])),
                    reasons=[a.get('reason') for a in report.get('attempts', [])],
                    box=report.get('space_budget'), search_seconds=report.get('search_seconds'),
                    box_volume_reduction_percent=report.get('box_volume_reduction_percent'),
                    xy_area_reduction_percent=report.get('xy_area_reduction_percent'),
                    step3_passed=report.get('step3_passed'))
        row['seconds'] = time.monotonic()-began
        ledger['results'].append(row)
        save(destination/'batch_report.json', ledger)
        print('RERUN', group.name, row['status'], flush=True)
    after = protected_hashes(root)
    changes = sorted(p for p in set(protected) | set(after) if protected.get(p) != after.get(p))
    ledger.update(complete=True, step3_and_reference_unchanged=not changes, changed_protected_files=changes,
        constructed_count=sum(r['constructed'] for r in ledger['results']),
        passed_count=sum(r['passed'] for r in ledger['results']))
    save(destination/'batch_report.json', ledger)
    lines = ['# Step4.1 / Step4.2 八组重跑', '',
        f'排除 pose1+3；Step3 未重跑。核对 {len(protected)} 个受保护文件，'
        f'运行前后{"一致" if not changes else "存在变化"}。', '',
        '| 组合 | 几何通过 | XYZ 使用盒（mm） | 使用盒体积减少 | XY 占地减少 | 搜索时间（s） |',
        '|---|---|---|---:|---:|---:|']
    for row in ledger['results']:
        if row.get('box'):
            dimensions = ' × '.join(f'{v:.1f}' for v in row['box']['extents_mm'])
            lines.append(f'| {row["group"]} | {row["passed"]} | {dimensions} | '
                f'{row["box_volume_reduction_percent"]:.1f}% | {row["xy_area_reduction_percent"]:.1f}% | {row["search_seconds"]:.2f} |')
        else:
            lines.append(f'| {row["group"]} | {row["status"]} | — | — | — | — |')
    lines += ['', '以总使用空间的 XYZ 包围盒体积为目标；确定性收紧摆放、小幅扩建、连续退出裁切。',
              'Step4 通过仅指几何，Step3 力学原状态不变；不宣称全局最小。']
    (destination/'batch_report.md').write_text('\n'.join(lines)+'\n')
    if changes:
        raise RuntimeError('Protected Step3 or pose1+3 files changed')
    print('BATCH COMPLETE', len(groups), 'groups;', ledger['passed_count'], 'passed; protected inputs unchanged', flush=True)
    return ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    args = parser.parse_args()
    result = run(args.object)
    raise SystemExit(2 if any(not r['passed'] for r in result['results']) else 0)
