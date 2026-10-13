"""Run every current output/B set and publish images directly for each set."""
from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

COOPT = Path(__file__).resolve().parents[1]
ROOT = COOPT.parents[1]
VISUAL = COOPT / 'output/B'
RESULT = COOPT / 'data/results/B'

from publish_output_images import publish_case_images

_PUBLISHED = set()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def output_sets():
    paths = list(VISUAL.glob('pose*')) + list((VISUAL/'illegal').glob('pose*'))
    rows = []
    for path in paths:
        if not path.is_dir():
            continue
        label = str(path.relative_to(VISUAL))
        source = RESULT/label/'step4/step4.1/data/report.json'
        record = json.loads(source.read_text())
        rows.append(dict(id=label, poses=record['poses'], pose_count=len(record['poses'])))
    return sorted(rows, key=lambda r: (-r['pose_count'], r['id']))


def warm_seeds(rows):
    available = {}
    for path in (RESULT/'data').glob('evaluation_*/summary.json'):
        for record in json.loads(path.read_text()).get('cases', []):
            if 'step4_source' not in record:
                continue
            source = ROOT/record['step4_source']
            layout = source/'layout.npz'
            if not layout.exists():
                continue
            applicable = 4 <= record['pose_count'] <= 8
            mixed = (2 <= record['rotating_reuse_pose_count'] <= 4 and
                     2 <= record['juxtaposed_pose_count'] <= 4)
            rank = (applicable and not mixed,
                    record['with_full_support']['volume_litres'], record['material_full_cm3'])
            available.setdefault(record['id'], []).append((rank, layout))
    seeds = {}
    for row in rows:
        ranked = sorted(available.get(row['id'], []), key=lambda pair: pair[0])
        paths = []
        for rank, path in ranked:
            if path not in paths:
                paths.append(path)
            if len(paths) == 3:
                break
        seeds[row['id']] = [str(p.relative_to(ROOT)) for p in paths]
    return dict(cases=seeds, warm_start=True,
                acceptance='Each seed is evaluated again by the current search')


def publish(experiment, rows, status, complete=False):
    records = []
    for row in rows:
        label = row['id']
        root = VISUAL / label
        built = ((root/'step5/step5.2'/experiment/'all_poses.png').exists() and
                 (root/'evaluation'/experiment/'static_occupancy.png').exists())
        if built and (experiment, label) not in _PUBLISHED:
            publish_case_images(label, experiment, include_evaluation=True)
            _PUBLISHED.add((experiment, label))
        records.append(dict(id=label, pose_count=row['pose_count'],
                            search=status.get(label), pictures_complete=built,
                            images=str((root/'step5.2').relative_to(ROOT)) if built else None))
    write_json(RESULT/'data'/('system_'+experiment)/'image_progress.json',
               dict(complete=complete, experiment=experiment, cases=records))


def refresh_initialization(job):
    label, experiment = job
    sys.path.insert(0, str(COOPT/'helper_func'))
    import _bootstrap
    from co_common import S, trimesh
    from whole_pipeline import load_layout
    from oracle_search import OracleModel
    from whole_step4_render import render_initialization
    import step31_render, step32_render
    import artifact_store as A
    source = RESULT/label
    preview = source/'initialization'/experiment
    s3 = preview/'step3'
    (s3/'step3.1/data').mkdir(parents=True)
    record = json.loads((source/'step3/step3.1/data/report.json').read_text())
    group = dict(id=label, poses=record['poses'])
    for name in ['registered_object.obj', 'wrapped_support.obj']:
        (s3/'step3.1'/name).symlink_to(source/'step3/step3.1'/name)
    shutil.copy2(source/'step3/step3.1/data/report.json', s3/'step3.1/data/report.json')
    step31_render.render('B', group, s3)
    step32_render.render('B', group, s3)
    s41 = preview/'step4/step4.1'
    (s41/'data').mkdir(parents=True)
    (s41/'support.obj').symlink_to(source/'step4/step4.1/support.obj')
    model = OracleModel(group['poses'], 'B', demand_nodes=32, oracle_work_mode='normal')
    model.initialization_report = source/'step3/step3.1/data/report.json'
    model.registered_seed = S.solid(trimesh.load(source/'step3/step3.1/wrapped_support.obj', force='mesh', process=False))
    actual = dict(layout=load_layout(source/'step4/step4.1/layout.npz'),
                  remaining_mesh=trimesh.load(source/'step4/step4.1/support.obj', force='mesh', process=False))
    render_initialization(model, actual, s41)
    A.publish_visuals(s41)
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--initialization-budget', type=int, default=0)
    args = parser.parse_args()
    rows = output_sets(); experiment = args.experiment
    directory = RESULT/'data'/('system_'+experiment)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'requested_sets.json', dict(sets=rows, source='current output/B directories'))
    seeds = directory/'initialization_seeds.json'
    write_json(seeds, warm_seeds(rows))
    status = {}; publish(experiment, rows, status)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    records = []; began = time.monotonic()

    def run(stage, command):
        log_path = Path('/tmp')/(experiment+'_'+stage.replace('.', '_')+'.log')
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush()
                if line.startswith('ORACLE RESULT '):
                    result = json.loads(line.removeprefix('ORACLE RESULT '))
                    status[result['id']] = result
                    publish(experiment, rows, status)
                    print('STEP4', len(status), '/', len(rows), result['id'],
                          'PASS' if result['passed'] else 'FAILED', flush=True)
                elif stage != 'step4' and (' ALL ' in line or 'TOTAL' in line):
                    print(line.rstrip(), flush=True)
            code = process.wait()
        records.append(dict(stage=stage, returncode=code, command=command, log=str(log_path)))
        write_json(directory/'pipeline.json', dict(complete=False, stages=records, requested_count=len(rows)))
        if code:
            print('STAGE FAILED', stage, 'log', log_path, flush=True)
        return code

    run('step4', [sys.executable, str(COOPT/'step4.2/run_material.py'), 'B', '--sets',
                 *[r['id'] for r in rows], '--min-poses', '2', '--max-poses', '10',
                 '--jobs', str(args.jobs), '--iterations', str(args.iterations),
                 '--initialization-budget', str(args.initialization_budget),
                 '--initialization-seeds', str(seeds), '--output-name', experiment])
    batch = json.loads((RESULT/'data'/experiment/'batch.json').read_text())
    status = {r['id']: r for r in batch['results']}
    selected = [r for r in batch['results'] if r['passed']]
    selection = directory/'selection.json'
    write_json(selection, dict(complete=batch['complete'], selected=selected,
                              requested_count=len(rows), failed=[r for r in batch['results'] if not r['passed']]))
    if selected:
        run('step5.1', [sys.executable, str(COOPT/'step5.1/run.py'), 'B', '--experiment', experiment,
                       '--batch-name', 'step51_'+experiment, '--selection-summary', str(selection)])
        run('step5.2', [sys.executable, str(COOPT/'step5.2/run.py'), 'B', '--experiment', experiment,
                       '--batch-name', 'step52_'+experiment, '--selection-summary', str(selection), '--jobs', str(args.jobs)])
        run('evaluation', [sys.executable, str(COOPT/'evaluation/run.py'), 'B', '--experiment', experiment,
                          '--batch-name', 'evaluation_'+experiment, '--selection-summary', str(selection), '--replace'])
    render_errors = []
    with ProcessPoolExecutor(max_workers=min(args.jobs, 8), mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(refresh_initialization, (r['id'], experiment)): r['id'] for r in rows}
        for future in as_completed(futures):
            try:
                print('INITIALIZATION FIGURES', future.result(), flush=True)
            except Exception as error:
                render_errors.append(dict(id=futures[future], error=str(error)))
                print('INITIALIZATION FIGURE ERROR', futures[future], str(error), flush=True)
            publish(experiment, rows, status)
    publish(experiment, rows, status, complete=True)
    if all(record['returncode']==0 for record in records) and not render_errors and len(selected)==len(rows):
        run('images', [sys.executable, str(COOPT/'helper_func/publish_output_images.py'), experiment])
    write_json(directory/'pipeline.json', dict(complete=all(record['returncode']==0 for record in records) and not render_errors,
               experiment=experiment, stages=records,
               requested_count=len(rows), finished_search_count=len(status), passed_search_count=len(selected),
               render_errors=render_errors, warm_start=True, seconds=time.monotonic()-began,
               image_directory=str(VISUAL.relative_to(ROOT))))
    print('OUTPUT B FINISHED', len(status), 'searched;', len(selected), 'passed; per-set images', VISUAL, flush=True)


if __name__ == '__main__':
    main()
