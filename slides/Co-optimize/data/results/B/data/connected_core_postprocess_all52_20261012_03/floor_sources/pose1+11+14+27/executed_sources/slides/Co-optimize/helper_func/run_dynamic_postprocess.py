"""Run saved-layout pruning and construction in isolated per-case workers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time

COOPT = Path(__file__).resolve().parents[1]
ROOT = COOPT.parents[1]
BASE = COOPT/'data/results/B'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def load_stage(stage):
    folder = COOPT/stage
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('dynamic_'+stage.replace('.', '_'), folder/'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(payload):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    case, options, output = payload
    args = SimpleNamespace(**options)
    result = dict(id=case['id'], pose_count=case['pose_count'], experiment=case['experiment'])
    for stage, key in [('step5.2', 'pruning'), ('step5.3', 'construction')]:
        module = load_stage(stage)
        from stage5_batch import safe_worker
        folder = BASE/case['id']/'step5'/stage/args.experiment
        report_path = folder/'data/report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if not report['passed'] or report['selected_step4_experiment'] != case['experiment']:
                raise ValueError('Existing stage belongs to a different or failed layout')
            for name, expected in report.get('artifacts', {}).items():
                if sha(folder/name) != expected:
                    raise ValueError('Existing stage artifact changed: '+name)
            row = dict(id=case['id'], passed=True, pose_count=case['pose_count'],
                       output=str(folder.relative_to(ROOT)), reused_saved_stage=True)
        else:
            row = safe_worker(module.worker, case, args, BASE, stage)
        result[key] = row
        result.update(passed=row['passed'], status='PASS' if row['passed'] else key.upper()+'_FAIL')
        save(Path(output), result)
        if not row['passed']:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--step51-experiment', required=True)
    parser.add_argument('--selection-summary', type=Path, required=True)
    parser.add_argument('--source-summary', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--candidate-budget', type=int, default=64)
    parser.add_argument('--worker-payload', type=Path)
    args = parser.parse_args()
    if args.worker_payload:
        worker(json.loads(args.worker_payload.read_text()))
        return
    system = BASE/'data'/('system_'+args.experiment)
    directory = BASE/'data'/args.experiment
    directory.mkdir(exist_ok=False)
    selected = json.loads(args.selection_summary.read_text())['results']
    all_cases = json.loads(args.source_summary.read_text())['results']
    sources = []
    for name in ['helper_func','vis_func','step4.2','step5.1','step5.2','step5.3','evaluation']:
        for path in (COOPT/name).rglob('*'):
            if path.is_file() and path.suffix in ['.py','.cpp','.h']:
                relative = path.relative_to(ROOT)
                target = system/'executed_sources'/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                sources.append(dict(path=str(relative), sha256=sha(path), archive=str(target.relative_to(ROOT))))
    options = dict(object='B', experiment=args.experiment,
        selection_summary=str(args.selection_summary.resolve()), step51_experiment=args.step51_experiment,
        reference_stage='step4', reference_experiment=args.experiment,
        source_contact_material='auto', resume_pruning_experiment=None, clean_object_interiors=False,
        candidate_budget=args.candidate_budget, prune_experiment=args.experiment, leg_profile='flat')
    save(system/'execution.json', dict(options=options, executed_sources=sources,
         source_summary=str(args.source_summary), step4_rerun=False, original_demand_count_per_pose=32768))
    # Saved older pictures are already archived by the campaign preparation.
    # Clear the current final-stage pictures so failures cannot display old successes.
    for case in all_cases:
        for stage in ['step5.3','evaluation'] + (['step5.1','step5.2'] if not case['passed'] else []):
            folder = COOPT/'output/B'/case['id']/stage
            for picture in folder.glob('*.png'):
                picture.unlink()
    results = [dict(id=c['id'], pose_count=c['pose_count'], passed=False, status='STEP4_FAIL',
                    experiment=c['experiment'], skipped_reason='No certified Step4 layout')
               for c in all_cases if not c['passed']]
    began = time.monotonic()
    def record():
        save(directory/'batch.json', dict(complete=len(results)==len(all_cases), group_count=len(all_cases),
            finished=len(results), passed=sum(r['passed']for r in results),
            eligible_groups=len(selected), results=results, options=options, seconds=time.monotonic()-began))
    record()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    def one(case):
        label = case['id'].replace('/', '__')
        output = directory/'workers'/(label+'.json')
        payload = directory/'payloads'/(label+'.json')
        save(payload, [case, options, str(output.resolve())])
        log = directory/'logs'/(label+'.log')
        log.parent.mkdir(exist_ok=True)
        with log.open('w') as stream:
            code = subprocess.call([str(ROOT/'.venv/bin/python'), str(Path(__file__).resolve()),
                '--experiment',args.experiment,'--step51-experiment',args.step51_experiment,
                '--selection-summary',str(args.selection_summary),'--source-summary',str(args.source_summary),
                '--worker-payload',str(payload)], env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if output.exists():
            row = json.loads(output.read_text())
            if code and row.get('status')=='PASS':
                row.update(passed=False,status='RUNTIME_FAIL',error='Worker exited '+str(code))
        else:
            row = dict(id=case['id'],pose_count=case['pose_count'],experiment=case['experiment'],
                       passed=False,status='RUNTIME_FAIL',error='Worker exited '+str(code))
        row['execution_log'] = str(log.relative_to(ROOT))
        return row
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(one, case) for case in selected]
        for future in as_completed(pending):
            row = future.result();results.append(row);record()
            print('POSTPROCESS',len(results),'/',len(all_cases),'PASS',sum(r['passed']for r in results),row['id'],row['status'],flush=True)
    protected = json.loads((system/'protected_originals.json').read_text())
    changed = [p for p,h in protected.items()if not (ROOT/p).is_file()or sha(ROOT/p)!=h]
    changed_sources = [r['path']for r in sources if sha(ROOT/r['path'])!=r['sha256']]
    save(system/'execution_verification.json',dict(passed=not changed and not changed_sources,
         protected_files=len(protected), changed_inputs=changed, changed_sources=changed_sources))
    print('POSTPROCESS FINISHED',len(results),'PASS',sum(r['passed']for r in results),flush=True)


if __name__=='__main__':
    main()
