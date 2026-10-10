"""Independent, equal-budget ten-set sampled experiments; no final solids."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, HERE, C
from case_sets import CASES
from fast_search import FastModel, FastReuseSearch
from run_reuse_first import protected


def worker(args):
    numbers = CASES[args.case]
    out = args.out/args.case/args.method
    out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    try:
        model = FastModel([f'pose_{number}' for number in numbers])
        search = FastReuseSearch(model, out, iterations=8, finalists=3, branch_rounds=1)
        search.volume_rounds = 2
        search.screen_budget = 96
        search.search_only = True
        search.capture_process = True
        report = search.joint() if args.method == 'whole' else search.incremental()
        report.update(strategy=args.method, requested_poses=model.poses,
            requested_pose_count=len(numbers), model_prepare_and_search_seconds=time.monotonic()-began,
            independent_initialization=True, final_acceptance_run=False,
            inputs=C.provenance(model.inputs, [])['inputs'])
        C.save(out/'search_report.json', report)
        print('FINISHED', args.case, args.method, report['sampled_force_passed'],
              report['pose_count'], '/', len(numbers), flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        C.save(out/'error.json', dict(error=str(error), seconds=time.monotonic()-began))
        raise


def rows_at(out):
    rows = []
    for case, numbers in CASES.items():
        for method in ['whole', 'incremental']:
            directory = out/case/method
            path = directory/'search_report.json'
            if path.exists():
                report = json.loads(path.read_text())
                full = report['pose_count'] == len(numbers)
                rows.append(dict(case=case, method=method, completed=True,
                    full_requested_set_evaluated=full,
                    sampled_pass=bool(report['sampled_force_passed'] and full),
                    pose_count=report['pose_count'], requested_pose_count=len(numbers),
                    estimated_volume_cm3=report['estimated_volume_cm3'],
                    rotating_reuse=report['rotating_reuse_pose_count'],
                    juxtaposed=report['juxtaposed_pose_count'], counts=report['counts'],
                    search_seconds=report['search_seconds'],
                    sample_evaluations=report['sample_evaluations'],
                    exact_evaluations=report['exact_evaluations_during_search'],
                    report=str(path.relative_to(out))))
            elif (directory/'error.json').exists():
                rows.append(dict(case=case, method=method, completed=True, sampled_pass=False,
                    error=json.loads((directory/'error.json').read_text())))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE/'output/B')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case', choices=list(CASES))
    parser.add_argument('--method', choices=['whole', 'incremental'])
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.worker:
        worker(args)
        return
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    before = protected()
    if not args.resume:
        C.save(args.out/'protected_coopt_sources.json', before)
        runtime = args.out/'runtime_sources'
        hashes = {}
        for source in sorted(set(list((HERE/'code').glob('*.py'))+C.code_sources())):
            relative = source.resolve().relative_to(ROOT)
            destination = runtime/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        C.save(runtime/'manifest.json', dict(sources=hashes))
        C.save(args.out/'experiment.json', dict(cases=CASES, methods=['whole','incremental'],
            independent_initializations=True, final_acceptance_run=False,
            evaluation='sampled_contacts_and_material_occupancy',
            budgets=dict(iterations=8, finalists=3, branch_rounds=1, volume_rounds=2,
                         screen_budget=96, seed=42), jobs=args.jobs))
    tasks = [(case, method) for case in CASES for method in ['whole','incremental']
             if not (args.out/case/method/'search_report.json').exists()]
    pending = list(tasks)
    running = []
    began = time.monotonic()
    while pending or running:
        while pending and len(running) < args.jobs:
            case, method = pending.pop(0)
            directory = args.out/case/method
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory/'search.log').open('a' if args.resume else 'w')
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1')
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--worker','--case',case,'--method',method,'--out',str(args.out)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((case, method, process, log, time.monotonic()))
            print('START', case, method, 'pid', process.pid, flush=True)
        for entry in list(running):
            case, method, process, log, started = entry
            status = process.poll()
            if status is None:
                continue
            log.close()
            running.remove(entry)
            print('DONE', case, method, 'exit', status, 'seconds', round(time.monotonic()-started,1), flush=True)
        rows = rows_at(args.out)
        C.save(args.out/'batch.json', dict(complete=not pending and not running,
            results=rows, sampled_passes=sum(row['sampled_pass'] for row in rows),
            run_count=20, completed_runs=len(rows), final_acceptance_run=False,
            elapsed_seconds=time.monotonic()-began,
            running=[dict(case=case,method=method,pid=process.pid) for case,method,process,_,_ in running]))
        if pending or running:
            time.sleep(2)
    assert before == protected(), 'Production Co-optimize sources changed'
    print('BATCH COMPLETE', len(rows), 'runs;', sum(row['sampled_pass'] for row in rows),
          'sampled passes; no final acceptance', flush=True)


if __name__ == '__main__':
    main()
