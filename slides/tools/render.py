"""Render the current slide deck using the active Python environment.

Usage: python slides/tools/render.py [--only setup floor area demand heads equations trajectory]
Baseline code/results and setup data are read-only. Historic multi-object studies
are not rerun. Generated media remain local, following the repository .gitignore.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
RECIPES = {
    'setup': ['setup/poses/presentation.py', 'setup/working_area.py'],
    'floor': ['sys_floor/presentation.py'],
    'area': ['obj_supp/area/presentation.py'],
    'demand': ['obj_supp/demand/demand.py'],
    'heads': ['obj_supp/demand/head_total_force.py', 'obj_supp/demand/head_sweep.py'],
    'equations': ['setup/equations/three_equations.py', 'obj_supp/two_equations.py',
                  'obj_supp/demand/demand_equation.py', 'obj_supp/solution/solution.py',
                  'tools/combined_equations.py'],
    'trajectory': ['trajectory/presentation.py'],
}


def protected_files():
    paths = list((HERE/'baseline_algo').rglob('*'))
    paths += list((HERE/'setup/poses').rglob('*.npz'))
    paths += list((HERE/'setup/poses').glob('*/*/setup.json'))
    paths += list((HERE/'setup/poses').glob('*/poses.json'))
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in paths if p.is_file() and '__pycache__' not in p.parts}


def remove_old_figures():
    """Remove the retired slide exports, never baseline or setup data."""
    old = [HERE/'obj_supp/demand/pasted-movie.png']
    for name in ('A1-f', 'C5'):
        old.extend(HERE/path for path in (
            f'obj_supp/area/area_{name}.png',
            f'sys_floor/on_the_floor_{name}.png', f'setup/poses/tip_{name}.png'))
    current = {('B', 2), ('B', 3), ('A1-f', 2), ('A1-f', 3), ('C5', 2)}
    for name in ('A1-f', 'B', 'C5'):
        old.append(HERE/f'setup/poses/{name}/poses.png')
        old.extend(HERE/f'setup/poses/{name}/pose_{pose}/pose.png'
                   for pose in range(1, 5) if (name, pose) not in current)
        old.extend((HERE/f'obj_supp/insert_trajectory/output/{name}').glob('*.png'))
    removed = [p for p in old if p.is_file()]
    for path in removed:
        path.unlink()
        print(f'Removed retired figure: {path.relative_to(HERE)}', flush=True)
    print(f'Removed {len(removed)} retired figures.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=list(RECIPES))
    parser.add_argument('--clean-old', action='store_true',
                        help='Delete the known retired slide images after rendering.')
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
        if args.clean_old:
            remove_old_figures()
    finally:
        after = protected_files()
        changed = [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)]
        if changed:
            raise RuntimeError('Protected baseline/setup files changed: '+', '.join(changed))
    print('All requested figures rendered; baseline and setup data unchanged.', flush=True)


if __name__ == '__main__':
    main()
