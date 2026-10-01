"""Render the saved three-pose fixture animation from its historical Git data."""

from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import time

import numpy as np

import render_cpu as R


ROOT = Path(__file__).resolve().parents[1]
REVISION = '482c35b0113ea343aefdc6ed259d6a358ad29d37'
SOURCE = 'slides/reuse/data.js'


def run(speed):
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('Speed must be positive and finite')
    raw = subprocess.check_output(['git', 'show', f'{REVISION}:{SOURCE}'], cwd=ROOT)
    data, _ = json.JSONDecoder().raw_decode(raw.decode().split('=', 1)[1])
    assert [p['source'] for p in data['poses']] == [
        'B / pose_2', 'B / pose_3', 'B / pose_4']
    output = ROOT / f'reuse_workflow_old_{speed:g}x.mp4'
    if output.exists():
        raise FileExistsError(output)
    temp = output.with_name(output.stem + '.rendering.mp4')
    # The current renderer uses a neutral body with head overlays. This older
    # fixture instead has four fixed rib colors and a grey rear frame.
    adapted = dict(data, fixtureColor='#738894', headOverlays=[])
    scene = R.Scene(adapted)
    vertices = np.asarray(data['fixture']['v']).reshape(-1, 3)
    faces = np.asarray(data['fixture']['f']).reshape(-1, 3)
    centers = vertices[faces].mean(axis=1)
    color_ids = (centers[:, 2] < 0).astype(int) * 2 + (centers[:, 1] > 0).astype(int)
    colors = np.array([R.rgb(data['colors'][i]) for i in color_ids])
    colors[centers[:, 0] > .165] = R.rgb('#738894')
    fixture = R.piece(data['fixture'], np.repeat(colors[:, None, :], 3, axis=1))
    fixture['normals'] = np.asarray(data['fixture']['n']).reshape(-1, 3)[faces]
    scene.fixture = [fixture]
    renderer = R.Renderer()
    width, height, fps = 1440, 900, 24
    duration = data['duration'] / speed
    count = math.ceil(duration * fps)

    # Bound all rendered and saved configurations once, using cached local
    # bounds for each rigid mesh. The table, camera and scale stay fixed.
    local_bounds = {}

    def bounds(parts):
        points = []
        for mesh, rotation, translation in parts:
            key = id(mesh)
            if key not in local_bounds:
                lo = mesh['triangles'].min(axis=(0, 1))
                hi = mesh['triangles'].max(axis=(0, 1))
                local_bounds[key] = np.array([
                    [x, y, z] for x in (lo[0], hi[0])
                    for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            points.append(local_bounds[key] @ rotation.T + translation)
        return np.concatenate(points)

    source_times = np.minimum(np.arange(count) * speed / fps, scene.times[-1])
    framing_times = np.unique(np.r_[scene.times, source_times])
    all_bounds = []
    for t in framing_times:
        all_bounds.append(bounds(scene.parts(scene.sample(t), robot=True, floor=False)))
    all_bounds = np.concatenate(all_bounds)
    camera = R.fit(all_bounds, [.28, 1, .65], width / height, 1.08)
    focus, basis, span = camera
    projected = (all_bounds - focus) @ basis.T
    framing_max_abs = float(max(
        np.abs(projected[:, 0]).max() / (span * width / height / 2),
        np.abs(projected[:, 1]).max() / (span / 2)))
    assert framing_max_abs < 1
    print(f'Recovered three-pose animation: {data["duration"]:.3f}s -> '
          f'{duration:.3f}s at {speed:g}x; {count} frames.', flush=True)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
               '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(temp)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    start = time.perf_counter()
    try:
        for i, t in enumerate(source_times):
            image = renderer.render(scene.parts(scene.sample(t), robot=True),
                                    camera, width, height, 1.25)
            process.stdin.write(image.tobytes())
            if i % 96 == 0 or i == count - 1:
                print(f'Video {i + 1}/{count} frames; '
                      f'{time.perf_counter() - start:.1f}s', flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg encoding failed')
    except BaseException:
        process.kill()
        process.wait()
        raise
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin',
                    '-i', str(temp), '-f', 'null', '-'], check=True)
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=width,height,r_frame_rate,nb_frames',
        '-of', 'json', str(temp)], text=True))
    assert int(info['streams'][0]['nb_frames']) == count
    assert abs(float(info['format']['duration']) - duration) <= 1 / fps + 1e-3
    temp.replace(output)
    report = dict(
        source_commit=REVISION, source_file=SOURCE,
        source_data_sha256=hashlib.sha256(raw).hexdigest(),
        source_poses=[p['source'] for p in data['poses']],
        source_timeline_seconds=data['duration'], speed_relative_to_saved_timeline=speed,
        source_playback_speed_metadata=data.get('playbackSpeed'),
        source_motion_regenerated=False, source_geometry_regenerated=False,
        renderer='CPU triangle z-buffer', video=output.name,
        video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        fps=fps, width=width, height=height, frames=count,
        duration_seconds=float(info['format']['duration']),
        fixed_camera=True, fixed_table=True, cast_shadows=False,
        camera_focus_m=focus.tolist(), camera_axes=basis.tolist(),
        camera_height_m=span, framing_max_abs=framing_max_abs,
        complete_video_decode_passed=True,
        scope='Historical display mesh and saved robot motion replay; no new physical acceptance checks.')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Exported and decoded: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed', type=float, default=3)
    run(parser.parse_args().speed)
