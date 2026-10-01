"""Run every B pair, recording failures without stopping the other cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / 'slides/baseline_algo/output/B'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(pair):
    out = OUTPUT / pair / 'step5'
    references = {p.name: digest(p) for p in out.glob('reference_*') if p.is_file()}
    status = dict(pair=pair, started_utc=datetime.now(timezone.utc).isoformat(),
                  status='running', passed=False, references=references,
                  algorithm_sha256=digest(HERE / 'run_loft_growth.py'))
    path = out / 'latest_run.json'
    path.write_text(json.dumps(status, indent=2) + '\n')
    started = time.monotonic()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    commands = [
        ('search', [sys.executable, str(HERE / 'run_loft_growth.py'), '--pair', pair]),
        ('independent_audit', [sys.executable, str(HERE / 'audit_fixture.py'), '--input', str(out)]),
        ('render_png', ['node', str(HERE / 'export_shared_geometry.cjs'), str(out)]),
    ]
    print('START', pair, flush=True)
    try:
        with (out / 'loft_pipeline.log').open('w') as log:
            for stage, command in commands:
                status['stage'] = stage
                path.write_text(json.dumps(status, indent=2) + '\n')
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                log.flush()
                if process.returncode:
                    tail = (out / 'loft_pipeline.log').read_text().splitlines()[-16:]
                    raise RuntimeError(f'{stage} exited {process.returncode}: ' + '\n'.join(tail))
        report = json.loads((out / 'report.json').read_text())
        audit = json.loads((out / 'independent_audit.json').read_text())
        assert report['passed'] and audit['passed']
        assert audit['reviewed_report_sha256'] == digest(out / 'report.json')
        status.update(status='passed', passed=True, volume_cm3=report['volume_cm3'],
                      external_area_cm2=report['external_area_cm2'],
                      comparison=report['comparison'],
                      report_sha256=digest(out / 'report.json'),
                      overview_sha256=digest(out / 'overview.png'))
    except Exception as error:
        status.update(status='failed', passed=False, error=str(error))
    changed = [name for name, sha in references.items() if digest(out / name) != sha]
    if changed:
        status.update(status='failed', passed=False, reference_changes=changed)
    status['elapsed_seconds'] = time.monotonic() - started
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(status, indent=2) + '\n')
    print('DONE', pair, status['status'], round(status['elapsed_seconds'], 1),
          status.get('volume_cm3', status.get('error', '')[-180:]), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs='+')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    pairs = args.pair or sorted(p.name for p in OUTPUT.glob('pose*+*')
                                if (p / 'step5/reference_report.json').exists())
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, pairs))
    print(json.dumps([dict(pair=r['pair'], status=r['status'],
                           volume_cm3=r.get('volume_cm3')) for r in results], indent=2))
