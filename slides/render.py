"""Render the current B/pose_2 slide deck using the active Python environment.

Usage: python slides/render.py [--only setup floor area demand heads equations trajectory]
Baseline code/results and setup data are read-only. Historic multi-object studies
are not rerun. Generated media remain local, following the repository .gitignore.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RECIPES = {
    'setup': ['setup/poses/presentation.py'],
    'floor': ['sys_floor/presentation.py'],
    'area': ['obj_supp/area/presentation.py'],
    'demand': ['obj_supp/demand/demand.py'],
    'heads': ['obj_supp/demand/head_total_force.py', 'obj_supp/demand/head_sweep.py'],
    'equations': ['setup/equations/three_equations.py', 'obj_supp/two_equations.py',
                  'obj_supp/demand/demand_equation.py', 'obj_supp/solution/solution.py'],
    'trajectory': ['trajectory/presentation.py'],
}


def protected_files():
    paths = list((HERE/'baseline_algo').rglob('*'))
    paths += list((HERE/'setup/poses').rglob('*.npz'))
    paths += [p for p in (HERE/'setup/poses').rglob('*.json') if p.name != 'target_pose.json']
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in paths if p.is_file() and '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=list(RECIPES))
    args = parser.parse_args()
    before = protected_files()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    try:
        for group in args.only or RECIPES:
            for script in RECIPES[group]:
                start = time.monotonic()
                print(f'Rendering {script}', flush=True)
                subprocess.run([sys.executable, str(HERE/script)], cwd=HERE.parent,
                               env=environment, check=True)
                print(f'Completed in {time.monotonic()-start:.1f}s', flush=True)
    finally:
        after = protected_files()
        changed = [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)]
        if changed:
            raise RuntimeError('Protected baseline/setup files changed: '+', '.join(changed))
    print('All requested figures rendered; baseline and setup data unchanged.', flush=True)


if __name__ == '__main__':
    main()
