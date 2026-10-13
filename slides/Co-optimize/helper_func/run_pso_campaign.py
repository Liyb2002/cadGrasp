"""Resume the frozen ALL52 PSO/control campaign through Step5 and Evaluation.

Completed cases are reused only from this campaign. Interrupted workers restart
from registered inputs, with their partial files preserved as separate attempts.
The numerical snapshot and all previously accepted artifacts remain immutable.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'slides/Co-optimize/data/results/B'
POLICIES = {'pso': 'pso_all52_20261013_01',
            'differential': 'differential_control_all52_20261013_01'}
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing.json')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(system):
    frozen = read(system / 'frozen_sources.json')['files']
    changed_sources = [r['frozen'] for r in frozen
                       if sha(ROOT / r['frozen']) != r['frozen_sha256']]
    protected = read(system / 'protected_originals.json')
    changed_inputs = [p for p, h in protected.items()
                      if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    record = dict(passed=not changed_sources and not changed_inputs,
                  snapshot_files=len(frozen), protected_files=len(protected),
                  changed_sources=changed_sources, changed_inputs=changed_inputs)
    save(system / 'resume_verification.json', record)
    if not record['passed']:
        raise RuntimeError('Frozen sources or protected artifacts changed')


def archive_partial(system, token, experiment, label, paths):
    entries = []
    for path in paths:
        if path.exists():
            target = system / 'interrupted_attempts' / token / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), target)
            entries.append(dict(original=str(path.relative_to(ROOT)),
                                archive=str(target.relative_to(ROOT))))
    if entries:
        save(system / 'interrupted_attempts' / token / 'records' /
             experiment / (label.replace('/', '__') + '.json'),
             dict(id=label, experiment=experiment, files=entries,
                  restart='registered cold initialization; no checkpoint reuse'))


def execute(script, payload, output, log):
    save(payload[0], payload[1])
    log.parent.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    with log.open('w') as stream:
        code = subprocess.call([str(ROOT / '.venv/bin/python'), str(script),
                                str(payload[0])], cwd=ROOT, env=ENV,
                               stdout=stream, stderr=subprocess.STDOUT)
    row = read(output) if output.exists() else dict(
        id=payload[1].get('id', payload[1].get('case', {}).get('id')),
        passed=False, status='RUNTIME_FAIL', error='Worker exit ' + str(code))
    if code:
        row.update(passed=False, status='RUNTIME_FAIL', error='Worker exit ' + str(code))
    row.update(process_wall_seconds=time.perf_counter() - began,
               worker_exit_code=code, execution_log=str(log.relative_to(ROOT)))
    save(output, row)
    return row


def step4(system, jobs, token):
    scripts = system / 'execution_scripts'
    batches = {p: read(BASE / 'data' / exp / 'batch.json')
               for p, exp in POLICIES.items()}
    began = time.perf_counter()
    old_seconds = {p: b.get('seconds', 0.) for p, b in batches.items()}
    for policy, batch in batches.items():
        save(system / 'interrupted_attempts' / token / (policy + '_batch.json'), batch)
        batch.setdefault('execution_segments', []).append(dict(
            token=token, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            jobs=jobs, retained_completed_cases=len(batch['results']),
            interrupted_compute_excluded_from_completed_case_seconds=True))
        batch.update(jobs=jobs, concurrent_other_policy_jobs=jobs)

    def one(policy, label):
        batch = batches[policy]
        exp = POLICIES[policy]
        folder = BASE / 'data' / exp
        safe = label.replace('/', '__')
        output = folder / 'workers' / (safe + '.json')
        payload = folder / 'payloads' / (safe + '.json')
        log = folder / 'logs' / (safe + '.log')
        archive_partial(system, token, exp, label, [
            BASE / label / 'step4/step4.2' / exp,
            BASE / label / 'step4/step4.1' / exp, output, log, payload])
        row = execute(scripts / 'cadgrasp_pso_step4_worker.py',
                      (payload, dict(id=label, options=batch['options'],
                                     frozen=str(system / 'frozen_sources'), result=str(output))),
                      output, log)
        row.setdefault('experiment', exp)
        if 'pose_count' not in row:
            row['pose_count'] = len(read(BASE / label / 'step4/step4.1/data/report.json')['poses'])
        row['execution_segment'] = token
        return policy, row

    pools = {p: ThreadPoolExecutor(max_workers=jobs) for p in POLICIES}
    try:
        futures = []
        for policy, batch in batches.items():
            done = {r['id'] for r in batch['results']}
            for label in batch['sets']:
                if label not in done:
                    futures.append(pools[policy].submit(one, policy, label))
        for future in as_completed(futures):
            policy, row = future.result()
            batch = batches[policy]
            batch['results'].append(row)
            batch.update(passed=sum(r['passed'] for r in batch['results']),
                         complete=len(batch['results']) == batch['group_count'],
                         seconds=old_seconds[policy] + time.perf_counter() - began,
                         timing_scope='sum of recorded execution segments; see interrupted attempts')
            batch['execution_segments'][-1]['seconds'] = time.perf_counter() - began
            save(BASE / 'data' / POLICIES[policy] / 'batch.json', batch)
            print('STEP4', policy, len(batch['results']), '/52', 'PASS', batch['passed'],
                  row['id'], round(row['process_wall_seconds'], 2), flush=True)
    finally:
        for pool in pools.values():
            pool.shutdown()
    for policy, batch in batches.items():
        if len(batch['results']) != 52 or len({r['id'] for r in batch['results']}) != 52:
            raise RuntimeError('Step4 ledger must contain all 52 unique cases')
    verify(system)


def postprocess(system, jobs, token):
    for policy in ['differential', 'pso']:
        exp = POLICIES[policy]
        post = exp + '_sculpted'
        source = read(BASE / 'data' / exp / 'batch.json')
        if not source['complete']:
            raise RuntimeError('Step4 must finish before this postprocess batch')
        folder = BASE / 'data' / post
        selection = folder / 'selection.json'
        save(selection, dict(complete=True, group_count=52, results=source['results'], passed=source['passed']))
        path = folder / 'batch.json'
        batch = read(path) if path.exists() else dict(
            complete=False, group_count=52, passed=0, results=[], seconds=0.,
            eligible_groups=source['passed'], policy=policy, postprocess_experiment=post,
            source_summary=str((BASE / 'data' / exp / 'batch.json').relative_to(ROOT)))
        done = {r['id'] for r in batch['results']}
        for case in source['results']:
            if case['id'] not in done and not case['passed']:
                batch['results'].append(dict(id=case['id'], pose_count=case['pose_count'],
                    experiment=exp, passed=False, status='STEP4_FAIL',
                    skipped_reason=case.get('error', 'No certified Step4 layout')))
        began = time.perf_counter()
        old_seconds = batch['seconds']

        def record():
            batch.update(complete=len(batch['results']) == 52,
                         passed=sum(r['passed'] for r in batch['results']),
                         seconds=old_seconds + time.perf_counter() - began, jobs=jobs)
            save(path, batch)

        def one(case):
            safe = case['id'].replace('/', '__')
            output = folder / 'workers' / (safe + '.json')
            payload = folder / 'payloads' / (safe + '.json')
            log = folder / 'logs' / (safe + '.log')
            variants = [exp + '_stock', exp + '_rounded', post]
            partial = [BASE / case['id'] / 'step5/step5.1' / (exp + '_floor')]
            partial += [BASE / case['id'] / 'step5' / stage / variant
                        for stage in ['step5.2', 'step5.3'] for variant in variants]
            archive_partial(system, token, post, case['id'], partial + [output, payload, log])
            # Old public images were protected by the campaign's image archive.
            for stage in ['step5.1', 'step5.2', 'step5.3', 'evaluation']:
                for image in (ROOT / 'slides/Co-optimize/output/B' / case['id'] / stage).glob('*.png'):
                    image.unlink()
            row = execute(system / 'execution_scripts/cadgrasp_pso_post_worker.py',
                (payload, dict(frozen=str(system / 'frozen_sources'), case=case,
                    selection=str(selection), initial=exp + '_stock',
                    aesthetic=exp + '_rounded', final=post, floor=exp + '_floor', result=str(output))),
                output, log)
            row.setdefault('pose_count', case['pose_count'])
            row.setdefault('experiment', exp)
            return row

        record()
        done = {r['id'] for r in batch['results']}
        pending = sorted([r for r in source['results'] if r['passed'] and r['id'] not in done],
                         key=lambda r: (-r['pose_count'], r['id']))
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for future in as_completed([pool.submit(one, case) for case in pending]):
                row = future.result()
                batch['results'].append(row)
                record()
                print('STEP5', policy, len(batch['results']), '/52', 'PASS', batch['passed'],
                      row['id'], round(row['process_wall_seconds'], 2), flush=True)
        record()
        for case in batch['results']:
            if not case['passed']:
                for stage in ['step5.1', 'step5.2', 'step5.3', 'evaluation']:
                    for image in (ROOT / 'slides/Co-optimize/output/B' / case['id'] / stage).glob('*.png'):
                        image.unlink()
    verify(system)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', default='pso_comparison_all52_20261013_01')
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--post-jobs', type=int, default=6)
    parser.add_argument('--step4-only', action='store_true')
    args = parser.parse_args()
    system = BASE / 'data' / ('system_' + args.system)
    with (system / 'campaign.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(system)
        token = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        state = read(system / 'work_state.json')
        state.update(active_pid=os.getpid(), resumed_utc=token, complete=False,
                     continuation='preserve completed cases; restart interrupted cases cold')
        save(system / 'work_state.json', state)
        step4(system, args.jobs, token)
        if args.step4_only:
            return
        postprocess(system, args.post_jobs, token)
        for name in ['cadgrasp_pso_evaluate.py', 'cadgrasp_pso_report.py']:
            with (system / (name + '.log')).open('w') as stream:
                subprocess.run([str(ROOT / '.venv/bin/python'),
                                str(system / 'execution_scripts' / name)],
                               cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=True)
        verify(system)
        print('ALL52 PSO AND CONTROL THROUGH STEP5 AND EVALUATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
