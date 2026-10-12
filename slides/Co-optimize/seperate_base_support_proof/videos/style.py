"""Render videos 2 and 3 from saved geometry in video 1's visual style.

Video 2 uses opaque thin ground, clear demand dots and an airborne pose change;
both bases are attached from the first frame. Video 3 uses one camera orbit and
the pose-owned Oracle work cone. No solver is invoked.
"""
from pathlib import Path
import argparse
import json

import render02
import render03

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=HERE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    source = args.source_dir.resolve()
    if source == args.output_dir.resolve():
        raise ValueError('Render to a separate directory before replacing the movies')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected = {path: render02.digest(path) for path in
        [source/'01_hulls_and_support.mp4', source/'02_ground_collision.mp4', source/'04_work_cones.mp4']}
    ground_movie = render02.Movie()
    work_movie = render03.Movie()
    ground_record, work_record = ground_movie.verify(), work_movie.record()
    print(json.dumps(dict(ground=ground_record, work=work_record)), flush=True)
    if args.preview:
        for index in [0, 36, 72, 108, 144, 189]:
            ground_movie.frame(index/render02.FPS).save(args.output_dir/f'02_{index:03d}.png')
            work_movie.frame(index/render03.FPS).save(args.output_dir/f'04_{index:03d}.png')
    else:
        render02.encode(ground_movie, args.output_dir/'02_ground_collision.mp4')
        render03.encode(work_movie, args.output_dir/render03.OUTPUT_NAME)
    ground_movie.model.verify_inputs()
    work_movie.verify_inputs()
    assert all(render02.digest(path) == checksum for path, checksum in protected.items())


if __name__ == '__main__':
    main()
