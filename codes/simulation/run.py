"""One video: raise the object, insert the fixture, withdraw arms, test ten loads."""
import sys

sys.dont_write_bytecode = True

import argparse
from pathlib import Path
import shutil
import tempfile

import numpy as np

from scene import build
from workflow import Workflow, render


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'simulation/videos/B_pose2_full_workflow.mp4'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--test-seconds', type=float, default=4.)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if not np.isfinite(args.test_seconds) or args.test_seconds <= 0 or args.fps < 1:
        parser.error('Duration and fps must be positive')
    if args.test_seconds * args.fps < 1:
        parser.error('Each random load must have at least one frame')
    if args.output.suffix.lower() != '.mp4':
        parser.error('Output must be an MP4 file')
    if args.output.exists() and not args.overwrite:
        parser.error('Video already exists; use --overwrite to replace it')
    model, _ = build()
    workflow = Workflow(model, seed=args.seed, test_seconds=args.test_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cadgrasp-video-') as temporary:
        pending = Path(temporary) / args.output.name
        frames = render(workflow, pending, fps=args.fps)
        shutil.move(str(pending), str(args.output))
    print(f'{args.output}: {frames} frames, {workflow.duration:g} seconds', flush=True)


if __name__ == '__main__':
    main()
