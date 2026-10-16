"""Run Step 1.1 and Step 1.3; Step 1.2 stays empty, Step 2 runs separately.

From the repository root:
    .venv/bin/python -m whole_task.step1.run
    .venv/bin/python -m whole_task.step1.run --areas 4
    .venv/bin/python -m whole_task.step1.run --config robots_config/01_franka_panda.json
"""
import argparse
from pathlib import Path

from .common import DEFAULT_OUTPUT
from .step1_1 import select_robot
from .step1_3 import select_working_areas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="B")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, help="Override the fixed oblique 25-degree example input")
    parser.add_argument("--areas", type=int, default=4)
    parser.add_argument("--inset", type=float, default=.05, help="Fraction of footprint radius and direction angles reserved as slack")
    args = parser.parse_args()
    robot = select_robot(args.output_root, args.object, args.config)
    print(f'Step 1.1: {robot["selected_config"]} ({robot["mode"]})', flush=True)
    areas = select_working_areas(args.output_root, args.object, args.areas, args.inset)
    print(f'Step 1.3: {areas["working_area_count"]} working areas, '
          f'{100*args.inset:g}% radius and angle inset', flush=True)
    print(Path(args.output_root) / args.object / "step1", flush=True)


if __name__ == "__main__":
    main()
