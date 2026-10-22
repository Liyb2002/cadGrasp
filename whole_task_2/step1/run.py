"""Run Step1 with a 100mm ball intersected with the 25-degree forward cone.

Generate the configured tool diagram, four geometric examples and an automatic
full-surface coverage plan. Step2--5 are separate, older-model results.
"""
import argparse
from pathlib import Path

from .common import DEFAULT_OUTPUT
from .run_needle import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="B")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, help="Override the fixed oblique 25-degree example input")
    parser.add_argument("--areas", type=int, choices=[4], default=4, help="Four display examples; automatic coverage count is unrestricted")
    parser.add_argument("--inset", type=float, default=.05, help="Pose-proposal depth margin; does not shrink the selected sector")
    args = parser.parse_args()
    run(args.output_root, args.object, args.config, args.inset)
    print(Path(args.output_root) / args.object / "step1", flush=True)


if __name__ == "__main__":
    main()
