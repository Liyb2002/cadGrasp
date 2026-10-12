"""Independent forward-projection, playback and preserved-input audit."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import prepare as Q

C, np = Q.C, Q.np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    verification = json.loads((out/'verification.json').read_text())
    for name, expected in verification['protected_inputs'].items():
        assert C.digest(C.ROOT/name) == expected, name
    for name, expected in verification['executed_sources'].items():
        assert C.digest(C.ROOT/name) == expected, name
        assert C.digest(out/'execution/executed_sources'/name) == expected, name
    print('VIDEO INPUTS AND SOURCES MATCH', flush=True)
    inp, data = Q.load('illegal/pose11+19')
    projection = []
    with np.load(out/'geometry/forward_hull_data.npz', allow_pickle=False) as arrays:
        for label in ['original', 'lift_11_16mm', 'lift_11_51mm', 'xyz']:
            for row in data['rows']:
                prefix = f'{label}_{row["pose"]}_'
                object_transform = arrays[prefix+'object_transform']
                delta = object_transform[:3, 3]-row['object_transform'][:3, 3]
                wrench = row['old_wrench_world_origin'].copy()
                wrench[:, 3:] += np.cross(delta, wrench[:, :3])
                expected = np.c_[-wrench[:, 4]/wrench[:, 2], wrench[:, 3]/wrench[:, 2], np.zeros(len(wrench))]
                cloud, hull = arrays[prefix+'cloud'], arrays[prefix+'hull']
                error = float(np.max(np.abs(cloud-expected)))
                assert error < 1e-12 and Q.P.covered(cloud[:, :2], hull[:, :2]).all()
                assert np.array_equal(object_transform[:3, :3], row['object_transform'][:3, :3])
                projection.append(dict(stage=label, pose=row['pose'], all_original_points=len(cloud),
                    largest_forward_projection_error_m=error,
                    hull_area_cm2=float(Q.D.ConvexHull(hull[:, :2]).volume*1e4), hull_vertex_count=len(hull)))
    videos = []
    for path in sorted(out.glob('*.mp4')):
        stream = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams'][0]
        assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
        assert (stream['width'], stream['height']) == (1280, 720) and stream['r_frame_rate'] == '24/1'
        assert int(stream['nb_frames']) == round(float(stream['duration'])*24)
        # Decode the whole file to catch truncated streams, not just container metadata.
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)
        videos.append(dict(file=path.name, duration_seconds=float(stream['duration']),
            frame_count=int(stream['nb_frames']), bytes=path.stat().st_size, full_decode_passed=True,
            sha256=C.digest(path)))
    assert len(videos) == 4
    print('FOUR VIDEOS FULLY DECODED', flush=True)
    protected = json.loads((Q.PROOF/'protected_existing.json').read_text())
    started = time.monotonic()
    def check(item):
        relative, expected = item
        path = C.ROOT/relative
        if not path.exists():
            return relative
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        return relative if digest.hexdigest() != expected else None
    changed = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, mismatch in enumerate(pool.map(check, protected.items()), 1):
            if mismatch:
                changed.append(mismatch)
            if i % 10000 == 0:
                print('PRESERVED FILES', i, '/', len(protected), f'{time.monotonic()-started:.1f}s', flush=True)
    assert not changed, changed[:10]
    # The independent audit source is an additional executed source, not a mutation of the render snapshot.
    own = Path(__file__).resolve()
    shutil.copy2(own, out/'execution/verify_videos.py')
    C.save(out/'independent_audit.json', dict(complete=True, full_video_decode_passed=True,
        videos=videos, forward_projection_checks=projection,
        old_objects_and_coopt_outputs_checked=len(protected), old_files_changed=changed,
        preservation_audit_seconds=time.monotonic()-started,
        audit_source=C.relative(own), audit_source_sha256=C.digest(own),
        source_snapshot_sha256=C.digest(out/'execution/verify_videos.py'),
        scope='Explanatory geometry; changed XYZ layout has no new force/work/exit acceptance.'))
    print('INDEPENDENT AUDIT COMPLETE', len(protected), 'unchanged files', flush=True)


if __name__ == '__main__':
    main()
