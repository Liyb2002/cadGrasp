"""Keep a complete, truthful ledger even when one frozen-layout case fails."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time
import traceback


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def safe_worker(worker, case, args, base, stage):
    try:
        return worker(case, args)
    except Exception as error:
        result = dict(id=case['id'], passed=False, pose_count=case['pose_count'],
                      error=str(error), error_type=type(error).__name__, protected={})
        save(Path(base)/case['id']/'step5'/stage/args.experiment/'data/failure.json',
             dict(result, complete=False, traceback=traceback.format_exc()))
        return result


def run_cases(cases, args, worker, directory, base, stage, stage_type, result_key):
    began = time.monotonic()
    results = {}

    def record(result):
        results[result['id']] = result
        ordered = [results[case['id']] for case in cases if case['id'] in results]
        passed = sum(row['passed'] for row in ordered)
        finished = len(ordered) == len(cases)
        save(Path(directory)/'summary.json', dict(complete=finished, all_passed=finished and passed==len(cases),
             stage_type=stage_type, group_count=len(cases), finished=len(ordered), passed=passed,
             failed=len(ordered)-passed, **{result_key:ordered}, seconds=time.monotonic()-began))
        protected = {path:sha for row in ordered for path,sha in row.get('protected',{}).items()}
        save(Path(directory)/'verification.json', dict(complete=finished, protected_inputs=protected))
        print(f'{stage.upper()} BATCH {len(ordered)}/{len(cases)}: {result["id"]} '
              f'{"PASS" if result["passed"] else "FAIL: "+result["error"]}', flush=True)

    if args.jobs>1:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = [pool.submit(safe_worker, worker, case, args, base, stage) for case in cases]
            for future in as_completed(pending):
                record(future.result())
    else:
        for case in cases:
            record(safe_worker(worker, case, args, base, stage))
    return [results[case['id']] for case in cases]
