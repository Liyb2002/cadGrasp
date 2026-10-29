"""One station end to end, for the run <object>+<config> (robots_config/random/<config>.json):
Step1 (plan, figures, validation) -> Step3 -> Step4 keep-out -> Step4 combine -> Step5 -> evaluation.
Data under data/<run>/, figures under output/<run>/, one log per step under data/<run>/logs/.

  python -m subdivide_task.run_station B+01_kuka
"""
import argparse
import os
import subprocess
import sys
import time

from subdivide_task.step1.common import ROOT

STEPS = [('step1', ['subdivide_task.step1.run']),
         ('step3', ['subdivide_task.step3.run', '--workers', '6']),
         ('step4_keepout', ['subdivide_task.step4.keepout', '--workers', '6']),
         ('step4_combine', ['subdivide_task.step4.combine', '--workers', '22']),
         ('step5', ['subdivide_task.step5.combined']),
         ('evaluation', ['subdivide_task.evaluation.combine', '--top', '40'])]


def run(name, start='step1'):
    logs = ROOT/'subdivide_task/data'/name/'logs'; logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    for step, command in STEPS[[s for s, _ in STEPS].index(start):]:
        began = time.monotonic()
        with (logs/f'{step}.log').open('w') as out:
            done = subprocess.run([sys.executable, '-m', *command, '--object', name], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        print(name, step, 'exit', done.returncode, round(time.monotonic()-began), 's', flush=True)
        if done.returncode: raise SystemExit(f'{name}: {step} failed, see {logs/(step+".log")}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name'); p.add_argument('--start', default='step1', choices=[s for s, _ in STEPS])
    a = p.parse_args(); run(a.name, a.start)
