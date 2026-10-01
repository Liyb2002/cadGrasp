"""Rerun the retained pose groups with 2 mm contact clearance and surface heads."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

from step3_scheculer import contacts as I

BASE = Path(__file__).resolve().parent


def command(arguments, log):
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MPLCONFIGDIR='/tmp/cadgrasp-surface-mpl')
    return subprocess.run([sys.executable, *map(str, arguments)], cwd=I.ROOT,
                          env=environment, stdout=log, stderr=subprocess.STDOUT).returncode


def retire_previous_presentation(output):
    """Remove stale public results while keeping the preceding model's evidence."""
    data = output/'data'
    archive = data/'history/before_surface_heads'
    archive.mkdir(parents=True, exist_ok=True)
    for path in [*output.glob('shape*'), output/'all_heads.png', output/'all_heads.html', output/'vis.html']:
        if path.is_file():
            destination = archive/path.name
            if not destination.exists():
                shutil.copyfile(path, destination)
            path.unlink()
    for name in ('report.json', 'codesign_report.json', 'common_object_heads.json',
                 'codesign_visualization.json', 'codesign_replay.json'):
        path = data/name
        if path.exists():
            destination = archive/'data'/name
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                shutil.copyfile(path, destination)
            path.unlink()
    old_body = data/'codesign_body'
    if old_body.exists() and not (archive/'data/codesign_body').exists():
        shutil.copytree(old_body, archive/'data/codesign_body')
    (output/'vis.html').write_text('<!doctype html><meta charset="utf-8"><p>正在重跑：零厚度接触面＋所有 pose 下 2 mm 地面间距。旧模型结果已移出当前展示。</p>')


def run_group(group):
    began = time.monotonic()
    poses = ['pose_'+i for i in group.name.removeprefix('pose').split('+')]
    root = group/'step3_scheculer/independent_poses_floor2mm'
    output = group/'step4'
    data = output/'data'
    data.mkdir(parents=True, exist_ok=True)
    row = dict(group=group.name, poses=poses, complete=False, constructed=False,
               passed=False, step3_root=str(root.relative_to(I.ROOT)))
    record = data/'surface_pipeline.json'
    I.save(record, row)
    with (data/'surface_pipeline.log').open('w', buffering=1) as log:
        code = command([BASE/'step3_scheculer/run_independent.py', group.parent.name,
                        '--poses', *poses, '--floor-poses', *poses,
                        '--floor-clearance-mm', '2', '--output-root', root, '--jobs', '1'], log)
        if code:
            row.update(status='step3_pipeline_error', returncode=code)
        else:
            results = [I.check_report(root/p/'step3_scheculer/schedule.json') for p in poses]
            row['step3'] = [dict(pose=p, passed=r['result']['passed'],
                                covered_count=r['result']['covered_counts'][0],
                                heads=len(r['heads'])) for p, r in zip(poses, results)]
            row['step3_passed'] = all(r['passed'] for r in row['step3'])
            row['status'] = 'constructing'
            I.save(record, row)
            retire_previous_presentation(output)
            # The body entry records incomplete Step3 coverage explicitly;
            # it can still expose a geometric candidate for inspection.
            code = command([BASE/'step4_connect_support/run_codesign.py', output], log)
            if code:
                row.update(status='step4_pipeline_error', returncode=code)
            else:
                physical = I.check_report(data/'report.json')
                row.update(status=physical['status'], constructed=physical['constructed'],
                           passed=physical['passed'], volume_cm3=physical.get('volume_cm3'))
                if physical['constructed']:
                    code = command([BASE/'step4_connect_support/publish_codesign.py', output], log)
                    if not code:
                        code = command([BASE/'step4_connect_support/review_codesign.py', output], log)
                    row['publication_passed'] = code == 0
                    if code:
                        row.update(status='publication_or_replay_error', returncode=code)
    row.update(complete=True, seconds=time.monotonic()-began)
    I.save(record, row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--groups', nargs='+')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    base = I.OUTPUTS/args.object
    groups = sorted((base/g for g in args.groups) if args.groups else
                    (p for p in base.glob('pose*+*') if (p/'step4').is_dir()),
                    key=lambda p: (len(p.name.split('+')), [int(i) for i in p.name.removeprefix('pose').split('+')]))
    if not groups or args.jobs < 1 or any(not p.is_dir() for p in groups):
        parser.error('Existing pose groups and a positive worker count are required')
    path = groups[-1]/'step4/data/surface_batch.json'
    ledger = dict(complete=False, model='zero-thickness contacts with group-wide 2 mm floor margin',
                  groups=[p.name for p in groups], results=[])
    I.save(path, ledger)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_group, group): group for group in groups}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                row = dict(group=futures[future].name, complete=True, status='pipeline_error',
                           constructed=False, passed=False, error=f'{type(error).__name__}: {error}')
            ledger['results'].append(row)
            I.save(path, ledger)
            print('GROUP COMPLETE', json.dumps(row, ensure_ascii=False), flush=True)
    ledger.update(complete=True, constructed_count=sum(r['constructed'] for r in ledger['results']),
                  accepted_count=sum(r['passed'] for r in ledger['results']))
    I.save(path, ledger)
    print('BATCH COMPLETE', ledger['constructed_count'], 'constructed;', ledger['accepted_count'], 'accepted', flush=True)


if __name__ == '__main__':
    main()
