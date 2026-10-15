"""Trim BBX -> build feet -> finish, always retaining a certified full solid."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

COOPT = Path(__file__).resolve().parents[1]
ROOT = COOPT.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def operation(payload, folder):
    """Native failure in an optional worker cannot erase the saved checkpoint."""
    folder = Path(folder)
    payload_path = folder.with_suffix('.payload.json')
    save(payload_path, dict(payload, output=str(folder)))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    with folder.with_suffix('.log').open('w') as log:
        try:
            result = subprocess.run([sys.executable, str(COOPT/'step5/feasible_geometry.py'),
                                     str(payload_path)], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=payload.get('finishing_seconds', 180)
                                    if payload['operation']=='finish' else None)
        except subprocess.TimeoutExpired:
            return dict(passed=False, error='Optional finishing time budget reached; certified complete support retained',
                        optimization_timeout=True, log=relative(folder.with_suffix('.log')))
    if result.returncode != 0 or not (folder/'report.json').exists():
        failure = read(folder/'failure.json') if (folder/'failure.json').exists() else {}
        return dict(passed=False, worker_exit_code=result.returncode,
                    error=failure.get('error', 'Worker exit '+str(result.returncode)),
                    log=relative(folder.with_suffix('.log')))
    record = read(folder/'report.json')
    if not record.get('passed') or not record.get('complete'):
        raise ValueError('Worker returned an uncertified checkpoint')
    for name, key in [('support.obj', 'support_sha256'), ('core.obj', 'core_sha256')]:
        if digest(folder/name) != record[key]:
            raise ValueError('Certified worker artifact changed: '+str(folder/name))
    return dict(passed=True, output=relative(folder), record=record)


def improve(payload, root, invoke=operation):
    """Commit core trimming only together with a certified complete foot result."""
    baseline = root/'baseline'
    first = invoke(dict(payload, operation='baseline'), baseline)
    if not first['passed']:
        raise ValueError('Saved Step4 full solid failed revalidation: '+first['error'])
    checkpoint = baseline
    stages = [dict(stage='baseline', **first)]
    trim = root/'bbx_trim'
    cropped = invoke(dict(payload, operation='trim'), trim)
    stages.append(dict(stage='bbx_trim', **cropped, committed=False))
    if cropped['passed']:
        feet = root/'feet'
        grown = invoke(dict(payload, operation='legs', checkpoint=str(checkpoint), trim=str(trim)), feet)
        stages.append(dict(stage='feet', **grown, committed=grown['passed']))
        if grown['passed']:
            checkpoint = feet
            stages[-2]['committed'] = True
        else:
            stages[-2]['rollback_reason'] = 'Trim and feet form one complete-support transaction'
    else:
        stages.append(dict(stage='feet', passed=False, skipped=True,
                           error='Retain Step4 feet because BBX trim was rejected'))
    finished = root/'finishing'
    last = invoke(dict(payload, operation='finish', checkpoint=str(checkpoint)), finished)
    stages.append(dict(stage='finishing', **last, committed=last['passed']))
    if last['passed']:
        checkpoint = finished
    return checkpoint, stages


def deliver(case, args, checkpoint, stages, base):
    record = read(checkpoint/'report.json')
    source = base/case['id']/'step4/step4.2'/case['experiment']
    s52 = base/case['id']/'step5/step5.2'/args.experiment
    s53 = base/case['id']/'step5/step5.3'/args.experiment
    for path in [s52, s53]:
        path.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint/'core.obj', s52/'pruned_support.obj')
    if(checkpoint/'core_force_certificate.npz').exists():
        shutil.copy2(checkpoint/'core_force_certificate.npz',s52/'force_certificate.npz')
    shutil.copy2(checkpoint/'support.obj', s53/'support_with_base.obj')
    shutil.copy2(checkpoint/'force_certificate.npz', s53/'force_certificate.npz')
    baseline = read(Path(stages[0]['output']).resolve()/'report.json')
    common = dict(complete=True, passed=True, object=args.object, id=case['id'],
                  pose_count=case['pose_count'], selected_step4_experiment=case['experiment'],
                  selected_step4_source=relative(source),
                  step51_source=relative(base/case['id']/'step5/step5.1'/args.step51_experiment),
                  connected_core_policy=True, step4_rerun=False, step51_rerun=False,
                  layout_changed=False, fixture_weight_ratio=0.,
                  pipeline='bbx_then_feet_then_finish', checkpoint=relative(checkpoint),
                  provenance=dict(inputs=record['protected']))
    save(s52/'data/report.json', dict(common, stage_type='support_pruning',
         support='pruned_support.obj', core_is_paired_with_certified_full_support=True,
         bbx_trim_committed=stages[1].get('committed', False),
         artifacts={'pruned_support.obj': digest(s52/'pruned_support.obj')}))
    if(s52/'force_certificate.npz').exists():
        paired=read(s52/'data/report.json')
        paired.update(intermediate_is_certified_complete_support=True,force_solver_run=True,
            original_demand_count_per_pose=32768,force_counts=record['intermediate_checks']['counts'])
        paired['artifacts']['force_certificate.npz']=digest(s52/'force_certificate.npz');save(s52/'data/report.json',paired)
    checks = record['checks']
    construction = dict(passed=True, connected=True, watertight=True,
         floor_pressure_enclosure_passed=True, serialized_export_verification=checks,
         removed_illegal_nominal_material_cm3=0.,
         added_material_cm3=record['added_to_core_cm3'], material_cm3=checks['material_cm3'],
         retained_legal_step4_material_loss_cm3=record['core_loss_cm3'],
         whole_support_recertified=True, contact_core_preserved=(None if record['core_loss_cm3']is None else record['core_loss_cm3']<=1e-4),
         contact_material_may_be_deleted_only_after_full_demand_recertification=True)
    if 'set_difference_volume_bounds_cm3'in record:
        construction['set_difference_volume_bounds_cm3']=record['set_difference_volume_bounds_cm3']
        construction['comparative_volume_statistics']=record['comparative_volume_statistics']
    final = dict(common, stage_type='final_support_construction',
                 step52_source=relative(s52), support='support_with_base.obj',
                 construction=construction, force_counts=checks['counts'],
                 force_solver_run=True, original_demand_count_per_pose=32768,
                 baseline_bbx_l=baseline['checks']['bbx_l'], final_bbx_l=checks['bbx_l'],
                 optional_operations=stages, feasible_checkpoint_retained=True,
                 saved_step4_fallback_used=not stages[1].get('committed', False),
                 final_is_original_step4_file=digest(s53/'support_with_base.obj') ==
                     digest(source/'support_with_base.obj'),
                 artifacts={name:digest(s53/name) for name in
                            ['support_with_base.obj', 'force_certificate.npz']})
    save(s53/'data/report.json', final)
    return dict(id=case['id'], pose_count=case['pose_count'], experiment=case['experiment'],
                passed=True, status='PASS', output=relative(s53),
                baseline_bbx_l=baseline['checks']['bbx_l'], final_bbx_l=checks['bbx_l'],
                cropped_core_and_feet_committed=stages[1].get('committed', False),
                saved_step4_fallback_used=final['saved_step4_fallback_used'],
                original_step4_file_retained=final['final_is_original_step4_file'],
                rejected_operations=[dict(stage=s['stage'], error=s['error'])
                                     for s in stages if not s['passed']],
                protected=record['protected'])


def run_case(case, args, base):
    root = base/case['id']/'step5/feasible'/args.experiment
    root.mkdir(parents=True, exist_ok=False)
    payload = dict(object=args.object, selection_summary=str(args.selection_summary),
                   step51_experiment=args.step51_experiment, case=case)
    reference = case.get('complete_reference_experiment',getattr(args,'complete_reference_experiment',None))
    if reference:
        payload['complete_reference_experiment'] = reference
    payload['finishing_seconds'] = args.finishing_seconds
    checkpoint, stages = improve(payload, root)
    return deliver(case, args, checkpoint, stages, base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--selection-summary', type=Path)
    parser.add_argument('--step51-experiment', default='material_output_all52_20261012_01')
    parser.add_argument('--sets', nargs='+')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--skip-evaluation', action='store_true')
    parser.add_argument('--skip-images', action='store_true')
    parser.add_argument('--complete-reference-experiment',
        help='Explicit same-layout complete support for a legacy Step4 reference needing boundary repair')
    parser.add_argument('--finishing-seconds', type=float, default=180,
        help='Optional aesthetic worker budget; expiry keeps the certified complete result')
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or args.jobs < 1 or args.finishing_seconds <= 0:
        parser.error('Use a single experiment directory name and positive jobs')
    base = ROOT/'slides/Co-optimize/data/results'/args.object
    args.selection_summary = (args.selection_summary or
        base/'data/material_output_all52_20261012_01/batch.json').resolve()
    selection = read(args.selection_summary)
    if not selection.get('complete'):
        raise ValueError('Select a completed Step4 ledger')
    cases = selection.get('selected', selection.get('cases', selection.get('results', [])))
    if args.sets:
        if set(args.sets)-{c['id'] for c in cases}:
            parser.error('Unknown selected sets')
        cases = [c for c in cases if c['id'] in args.sets]
    if not cases:
        parser.error('No selected sets')
    directory = base/'data'/('step5_feasible_'+args.experiment)
    directory.mkdir(parents=True, exist_ok=False)
    eligible = [c for c in cases if c['passed']]
    for case in eligible:
        for path in [base/case['id']/'step5'/stage/args.experiment
                     for stage in ['feasible', 'step5.2', 'step5.3']]:
            if path.exists():
                raise FileExistsError('Existing result is protected: '+str(path))
    began = time.monotonic()
    code = {relative(Path(__file__)):digest(Path(__file__)),
            relative(COOPT/'step5/run.py'):digest(COOPT/'step5/run.py')}
    results = [dict(id=c['id'], pose_count=c['pose_count'], experiment=c['experiment'],
                    passed=False, status='STEP4_FAIL', error='No certified Step4 support')
               for c in cases if not c['passed']]
    def record():
        save(directory/'summary.json', dict(complete=len(results)==len(cases),
             group_count=len(cases), eligible_groups=len(eligible),
             passed=sum(r['passed'] for r in results), results=results,
             all_eligible_passed=sum(r['passed'] for r in results)==len(eligible),
             stage_order=['bbx_trim', 'feet', 'finishing'], seconds=time.monotonic()-began,
             selection=relative(args.selection_summary), source_sha256=digest(args.selection_summary)))
    record()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run_case, case, args, base): case for case in eligible}
        for future in as_completed(pending):
            case = pending[future]
            try:
                row = future.result()
            except Exception as error:
                row = dict(id=case['id'], passed=False, status='VALIDATION_FAIL', error=str(error))
            results.append(row)
            record()
            print('STEP5 FEASIBLE',len(results),'/',len(cases),row['id'],row['passed'],
                  row.get('error',row.get('rejected_operations',[])),flush=True)
    protected = {p:h for r in results for p,h in r.get('protected', {}).items()}
    changed = [p for p,h in protected.items() if digest(ROOT/p) != h]
    for row in results:
        if row['passed']:
            final = read(ROOT/row['output']/'data/report.json')
            for stage in final['optional_operations']:
                for path, expected in stage.get('record', {}).get('executed_code', {}).items():
                    if path in code and code[path] != expected:
                        raise ValueError('Mixed executed source versions: '+path)
                    code[path] = expected
    source_changed = [path for path,h in code.items() if digest(ROOT/path) != h]
    for path,h in code.items():
        if path not in source_changed:
            destination = directory/'executed_sources'/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/path, destination)
            if digest(destination) != h:
                raise ValueError('Execution snapshot mismatch: '+path)
    save(directory/'verification.json', dict(passed=not changed, protected_inputs=protected,
         changed=changed, original_step4_and_demands_unchanged=not changed,
         executed_code=code, changed_sources=source_changed,
         source_and_execution_snapshot_hashes_match=not source_changed))
    if changed or source_changed or any(not r['passed'] for r in results if r.get('status') != 'STEP4_FAIL'):
        raise SystemExit(1)
    # Independent images and Evaluation never overwrite another campaign's view.
    valid = [dict(c, passed=True) for c in eligible]
    saved_selection = directory/'selection.json'
    save(saved_selection, dict(complete=True, results=valid))
    if not args.skip_evaluation or not args.skip_images:
        cmd = [sys.executable, str(COOPT/'step5/feasible_views.py'), args.object,
               '--experiment', args.experiment, '--selection-summary', str(saved_selection),
               '--step51-experiment', args.step51_experiment]
        if args.skip_images:
            cmd.append('--skip-images')
        if args.skip_evaluation:
            cmd.append('--skip-evaluation')
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
