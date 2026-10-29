"""One big task end to end on a station: Step1 (subdivide) -> Step3 -> Step4 keep-out -> Step4 combine ->
Step5 -> evaluation.  The run is the big task's name (objects/<object>/big_tasks/<run>.json); data under
data/<run>/, figures under output/<run>/, one log per step under data/<run>/logs/.

  python -m subdivide_task.run_big big_08
  python -m subdivide_task.run_big big_08 --start step4_combine
"""
import argparse
import os
import subprocess
import sys
import time

from subdivide_task.step1.common import ROOT

CONFIG = 'robots_config/random/07_kuka.json'


def steps(config, workers):
    return [('step1', ['subdivide_task.step1.subdivide', '--config', config]),
            ('step3', ['subdivide_task.step3.run', '--workers', str(min(workers, 12))]),
            ('step4_keepout', ['subdivide_task.step4.keepout', '--workers', str(workers)]),
            ('step4_combine', ['subdivide_task.step4.combine', '--workers', str(workers)]),
            ('step5', ['subdivide_task.step5.combined']),
            ('evaluation', ['subdivide_task.evaluation.combine', '--top', '40'])]


def run(name, start='step1', until='evaluation', config=CONFIG, workers=22):
    logs = ROOT/'subdivide_task/data'/name/'logs'; logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    plan = steps(config, workers); names = [s for s, _ in plan]
    for step, command in plan[names.index(start):names.index(until)+1]:
        began = time.monotonic()
        with (logs/f'{step}.log').open('w') as out:
            done = subprocess.run([sys.executable, '-m', *command, '--object', name], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        print(name, step, 'exit', done.returncode, round(time.monotonic()-began), 's', flush=True)
        if done.returncode: raise SystemExit(f'{name}: {step} failed, see {logs/(step+".log")}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    names = [s for s, _ in steps(CONFIG, 22)]
    p.add_argument('name'); p.add_argument('--start', default='step1', choices=names)
    p.add_argument('--until', default='evaluation', choices=names); p.add_argument('--config', default=CONFIG)
    p.add_argument('--workers', type=int, default=22)
    a = p.parse_args(); run(a.name, a.start, a.until, a.config, a.workers)
