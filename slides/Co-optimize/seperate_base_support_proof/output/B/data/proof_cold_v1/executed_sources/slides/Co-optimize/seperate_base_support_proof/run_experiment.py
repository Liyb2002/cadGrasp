"""Fresh all-52 Step4 experiments, followed by unchanged Step5.1 / Step5.2.

Only the entry point's output directory is redirected. Numerical modules,
native tasks, cold starts, budgets and acceptance are the current originals.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
PROOF = Path(__file__).resolve().parent
COOPT = PROOF.parent
ROOT = COOPT.parents[1]
BASE = PROOF / 'output/B'
for path in [ROOT, COOPT/'helper_func', COOPT/'vis_func', COOPT/'step4.2',
             COOPT/'step5.1', COOPT/'step5.2']:
    sys.path.insert(0, str(path))

import numpy as np
import trimesh
import run_stable_gradient as cold
import run_fast_state_seat_gradient as repair
import stage5_common as C
from whole_search.saved_initialization import check_saved_report


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S51 = load_module('proof_original_step51', COOPT/'step5.1/run.py')
S52 = load_module('proof_original_step52', COOPT/'step5.2/run.py')


def save(path, data):
    C.save(path, data)


def worker(job):
    phase, label, options, token = job
    runner = cold if phase == 'cold' else repair
    # Change only this entry point's artifact root, not co_common.HERE or
    # model roots. The shared original run-token ledger remains untouched.
    runner.HERE = PROOF
    result = runner.worker((label, options, token))
    result['experiment'] = options['output_name']
    result['phase'] = phase
    return result


def snapshot(directory):
    code = C.freeze(directory)
    aliases = {str((directory/'executed_sources'/p).relative_to(ROOT)): h
               for p, h in code.items()}
    save(directory/'source_manifest.json', dict(code=code, snapshots=aliases))
    return code


def stage5(result, reader, renderer, code, manifest):
    label = result['id']
    report = json.loads((BASE/label/'step4/step4.2'/result['experiment']/'report.json').read_text())
    case = dict(id=label, pose_count=len(report['poses']), passed=True,
                experiment=result['experiment'], volume_cm3=result['volume_cm3'])
    selection = BASE/label/'data/selection.json'
    save(selection, dict(cases=[case], selection='fresh_attempt_only',
                         previous_incumbents_used=False))
    batch = dict(manifest_path=manifest, executed_code=code)
    row = S51.compute_case('B', case, BASE, reader, renderer, 0., batch)
    inputs = C.Inputs('B', selection)
    inputs.base = BASE
    directory = BASE/'data/step52'/label
    directory.mkdir(parents=True, exist_ok=False)
    original_arguments = C.arguments
    C.arguments = lambda *unused: (SimpleNamespace(object='B'), inputs, [case], directory)
    try:
        S52.main()
    finally:
        C.arguments = original_arguments
    return dict(case=case, step51=row,
                step52_report=str((BASE/label/'step5/step5.2/data/report.json').relative_to(ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--worker-cpus', type=int, default=4)
    args = parser.parse_args()
    if (PROOF/'experiment.json').exists():
        raise RuntimeError('Existing experiment is protected; choose a fresh directory')
    old = COOPT/'output/B'
    paths = list(old.glob('*/step4/step4.1/data/report.json'))
    paths += list(old.glob('illegal/*/step4/step4.1/data/report.json'))
    groups = []
    for path in paths:
        report = check_saved_report(path)
        label = str(path.relative_to(old)).split('/step4/')[0]
        groups.append(dict(id=label, poses=report['poses'], pose_count=len(report['poses'])))
    groups.sort(key=lambda g: (not 4 <= g['pose_count'] <= 6,
                              g['pose_count'], g['id'].startswith('illegal/'), g['id']))
    assert len(groups) == 52
    original_generation = json.loads((old/'data/whole_step4_active_run.json').read_text())
    token = original_generation['run_token']
    # Freeze all pre-existing objects and Co-optimize outputs, including the
    # active-run ledger. The new directory has independent ledgers/galleries.
    protected = {str(p.relative_to(ROOT)): C.digest(p)
                 for folder in [ROOT/'objects', COOPT/'output']
                 for p in folder.rglob('*') if p.is_file()}
    save(PROOF/'protected_existing.json', protected)
    for group in groups:
        for stage in ['step3', 'step4/step4.1']:
            source, destination = old/group['id']/stage, BASE/group['id']/stage
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
    save(BASE/'data/whole_step4_active_run.json', original_generation)
    spec = dict(started_utc=datetime.now(timezone.utc).isoformat(), object='B', groups=groups,
                group_count=len(groups), four_to_six_pose_count=sum(4<=g['pose_count']<=6 for g in groups),
                initialization='byte-identical preserved Step4.1, cold starts',
                old_answers_used_as_start=False, final_selection='fresh_attempts only; old outputs preserved',
                algorithm_changed=False, jobs=args.jobs, worker_cpus=args.worker_cpus,
                cold_iterations=10, repair_iterations=8, demand_nodes=32,
                acceptance='all_original_force_torque_demands_on_search_contacts',
                source_entry=str((COOPT/'step4.2/run.py').relative_to(ROOT)),
                step51_entry=str((COOPT/'step5.1/run.py').relative_to(ROOT)),
                step52_entry=str((COOPT/'step5.2/run.py').relative_to(ROOT)))
    save(PROOF/'experiment.json', spec)
    cold_name, repair_name = 'proof_cold_v1', 'proof_own_repair_v1'
    for name in [cold_name, repair_name, 'step51']:
        directory = BASE/'data'/name
        directory.mkdir(parents=True, exist_ok=False)
        snapshot(directory)
    step51_directory = BASE/'data/step51'
    code = json.loads((step51_directory/'source_manifest.json').read_text())['code']
    manifest = step51_directory/'execution.json'
    save(manifest, dict(executed_code=code, groups=[g['id'] for g in groups],
                        original_demands_reused=True, numerical_algorithms_changed=False))
    reader = S51.NativeInputs('B', trimesh.load(ROOT/'objects/B/mesh.stl', force='mesh', process=False))
    renderer = S51.V.Renderer()
    record = dict(complete=False, step4_attempts=[], step5_results=[], repair_groups=[])
    began = time.monotonic()

    def progress():
        record['seconds'] = time.monotonic()-began
        save(PROOF/'progress.json', record)

    for phase, labels, name, iterations in [
        ('cold', [g['id'] for g in groups], cold_name, 10),
        ('repair', record['repair_groups'], repair_name, 8)]:
        options = dict(output_name=name, iterations=iterations, demand_nodes=32,
                       worker_cpus=args.worker_cpus, resume_experiment=cold_name if phase=='repair' else None)
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(worker, (phase, label, options, token)) for label in labels]
            for future in as_completed(futures):
                result = future.result()
                record['step4_attempts'].append(result)
                if result['passed'] and result.get('mesh_exported'):
                    record['step5_results'].append(stage5(result, reader, renderer, code, manifest))
                elif phase == 'cold':
                    out = BASE/result['id']/'step4/step4.2'/name
                    search_path = out/'search_report.json'
                    if search_path.exists() and (out/'sampled_layout.npz').exists():
                        search = json.loads(search_path.read_text())
                        if search.get('counts') and any(int(n)<32768 for n in search['counts']):
                            record['repair_groups'].append(result['id'])
                progress()
        save(BASE/'data'/name/'batch.json', dict(complete=True, groups=list(labels), options=options,
                 results=[r for r in record['step4_attempts'] if r['phase']==phase]))
    changed = [p for p,h in protected.items() if not (ROOT/p).is_file() or C.digest(ROOT/p)!=h]
    if changed:
        save(PROOF/'protection_failure.json', dict(changed=changed))
        raise RuntimeError(f'Pre-existing files changed: {changed[:5]}')
    selections = [r['case'] for r in record['step5_results']]
    save(PROOF/'selection.json', dict(cases=selections, fresh_attempts_only=True))
    record.update(complete=True, step4_passed_count=len(selections), protected_existing_count=len(protected),
                  protected_existing_unchanged=True, algorithm_changed=False)
    progress()
    save(PROOF/'summary.json', record)
    print('ALL COMPLETE', len(selections), '/', len(groups), 'fresh force passes', flush=True)


if __name__ == '__main__':
    main()
