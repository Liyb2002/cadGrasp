"""Run Step2.1 pose export and Step2.2 current-pose wrench generation."""
import argparse
from pathlib import Path

from whole_task.step1.common import DEFAULT_OUTPUT
from .step2_1 import run as run_poses
from .step2_2 import run as run_loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object", default="B")
    args = parser.parse_args()
    poses = run_poses(args.output_root, args.object)
    loads = run_loads(args.output_root, args.object)
    print(f"Step2.1: {poses['pose_count']} initial object poses")
    print(f"Step2.2: {loads['total_sample_count']} paired force/moment demands")


if __name__ == "__main__":
    main()
