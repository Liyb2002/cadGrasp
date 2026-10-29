"""Show exact screen-left and screen-right endpoints clearly in five seconds."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from subdivide_task.step1.common import ROOT, sha256, write_json
from . import render_refinements as reference


DATA = reference.DATA / 'exit_redirection'
VIDEO = reference.HERE / 'exit_redirection.mp4'
REVIEW_VIDEO = reference.HERE / 'exit_direction_left_to_right.mp4'
FPS = 24
FRAMES = 120
HOLD_FRAMES = 12


def motion_source():
    """Reuse the checked camera-plane geometry and its encoded pictures."""
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / 'screen_plane_motion_source.mp4'
    record = DATA / 'screen_plane_motion_source.json'
    if path.exists() and record.exists():
        metadata = json.loads(record.read_text())
        if (metadata['video_sha256'] == sha256(path)
                and metadata['source_sha256'] == sha256(Path(reference.__file__))
                and all(sha256(ROOT / key) == digest for key, digest in metadata['inputs_sha256'].items())):
            return path, metadata
    current = json.loads((DATA / 'video.json').read_text()) if (DATA / 'video.json').exists() else {}
    if not (VIDEO.exists() and current.get('source_sha256') == sha256(Path(reference.__file__))
            and current.get('directions_in_camera_plane') and current.get('direction_span_deg') == 180.
            and current.get('frames') == 120 and current.get('video_sha256') == sha256(VIDEO)):
        reference.main('exit_redirection')
        current = json.loads((DATA / 'video.json').read_text())
    assert current['full_decode_passed'] and current['frames'] == FRAMES and current['fps'] == FPS
    shutil.copy2(VIDEO, path)
    write_json(record, current)
    return path, current


def arrow_pixels(picture):
    """Measure the yellow direction arrow in an actual decoded video frame."""
    rgb = np.asarray(picture)
    mask = (rgb[:, :, 0] > 195) & (rgb[:, :, 1] > 150) & (rgb[:, :, 2] < 105)
    y, x = np.nonzero(mask)
    assert len(x) > 100, 'The direction arrow must remain visible'
    points = np.c_[x, y].astype(float)
    _, _, axes = np.linalg.svd(points - points.mean(0), full_matrices=False)
    horizontal_error = float(np.degrees(np.arctan2(abs(axes[0, 1]), abs(axes[0, 0]))))
    return dict(horizontal_error_deg=horizontal_error, colored_pixel_count=len(x),
                center_xy=points.mean(0).tolist())


def main():
    source, source_metadata = motion_source()
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(source), '-f', 'rawvideo',
                                    '-pix_fmt', 'rgb24', '-'])
    pictures = np.frombuffer(raw, dtype=np.uint8).reshape(-1, reference.SIZE, reference.SIZE, 3)
    assert len(pictures) == FRAMES
    # Twelve exact horizontal frames at each end; the 180-degree movement
    # occupies the intervening four seconds. No geometry is rotated in 2D.
    indices = np.r_[np.zeros(HOLD_FRAMES, dtype=int),
                    np.rint(np.linspace(1, FRAMES - 2, FRAMES - 2 * HOLD_FRAMES)).astype(int),
                    np.full(HOLD_FRAMES, FRAMES - 1, dtype=int)]
    assert len(indices) == FRAMES and np.all(np.diff(indices) >= 0)
    first, last = arrow_pixels(pictures[0]), arrow_pixels(pictures[-1])
    assert max(first['horizontal_error_deg'], last['horizontal_error_deg']) < 1.
    pending = DATA / 'horizontal_endpoints.pending.mp4'
    encoder = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{reference.SIZE}x{reference.SIZE}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(pending)],
        stdin=subprocess.PIPE)
    try:
        for index in indices:
            encoder.stdin.write(pictures[index].tobytes())
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError('Horizontal endpoint encoding failed')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(pending), '-f', 'null', '-'], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate:format=duration',
        '-of', 'json', str(pending)]))
    assert int(probe['streams'][0]['nb_read_frames']) == FRAMES
    assert probe['streams'][0]['r_frame_rate'] == '24/1'
    assert float(probe['format']['duration']) == 5.
    pending.replace(VIDEO)
    shutil.copy2(VIDEO, REVIEW_VIDEO)
    # Validate the actual final encoding, not just its geometry coordinates.
    encoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(VIDEO),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    encoded = np.frombuffer(encoded, np.uint8).reshape(FRAMES, reference.SIZE, reference.SIZE, 3)
    endpoint_checks = [arrow_pixels(encoded[index]) for index in [0, HOLD_FRAMES - 1, FRAMES - HOLD_FRAMES, FRAMES - 1]]
    assert max(row['horizontal_error_deg'] for row in endpoint_checks) < 1.
    sheet = Image.new('RGB', (reference.SIZE * 3, reference.SIZE), 'white')
    for column, index in enumerate([0, FRAMES // 2, FRAMES - 1]):
        sheet.paste(Image.fromarray(encoded[index]), (column * reference.SIZE, 0))
    sheet.save(DATA / 'left_to_right_endpoints.jpg')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(VIDEO), '-vf',
        'fps=1.6,scale=360:360,tile=4x2', '-frames:v', '1', str(DATA / 'encoded_review.jpg')], check=True)
    states = json.loads((DATA / 'states.json').read_text())['states']
    timeline = [dict(frame=i, time_s=i / FPS, geometry_state=int(index),
                     angle_deg=states[index]['angle_deg'],
                     directions_world=states[index]['directions_world'],
                     endpoint_hold=('left' if i < HOLD_FRAMES else 'right' if i >= FRAMES - HOLD_FRAMES else None))
                for i, index in enumerate(indices)]
    write_json(DATA / 'horizontal_endpoint_timeline.json', timeline)
    snapshot = DATA / 'render_horizontal_endpoint_source.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    metadata = dict(source_metadata, video_sha256=sha256(VIDEO), frames=FRAMES, fps=FPS,
        duration_seconds=5., decoded_frame_count=FRAMES, full_decode_passed=True,
        endpoint_hold_seconds=.5, left_endpoint_screen_direction='horizontal left',
        right_endpoint_screen_direction='horizontal right',
        decoded_endpoint_maximum_horizontal_error_deg=max(row['horizontal_error_deg'] for row in endpoint_checks),
        motion_geometry_source_sha256=source_metadata['source_sha256'],
        motion_video_sha256=sha256(source), geometry_unchanged=True,
        displayed_geometry_state_count=len(set(indices.tolist())),
        timeline=str((DATA / 'horizontal_endpoint_timeline.json').relative_to(ROOT)),
        timeline_sha256=sha256(DATA / 'horizontal_endpoint_timeline.json'),
        source_sha256=sha256(Path(__file__)), source_snapshot=str(snapshot.relative_to(ROOT)),
        alternate_video=str(REVIEW_VIDEO), alternate_video_sha256=sha256(REVIEW_VIDEO))
    write_json(DATA / 'video.json', metadata)
    manifest = reference.DATA / 'videos.json'
    videos = json.loads(manifest.read_text())
    videos['exit_redirection.mp4'] = metadata
    write_json(manifest, videos)
    write_json(DATA / 'horizontal_endpoint_validation.json', dict(status='passed',
        full_decode_passed=True, duration_seconds=5., frames=FRAMES, fps=FPS,
        endpoint_hold_frames=HOLD_FRAMES, decoded_endpoint_checks=endpoint_checks,
        direction_span_deg=180., directions_in_camera_plane=True,
        video_sha256=sha256(VIDEO), alternate_video_sha256=sha256(REVIEW_VIDEO),
        original_geometry_unchanged=True, geometry_state_file_sha256=sha256(DATA / 'states.json'),
        endpoint_preview_sha256=sha256(DATA / 'left_to_right_endpoints.jpg'),
        source_sha256=sha256(Path(__file__))))
    print('HORIZONTAL EXIT SAVED', REVIEW_VIDEO, '5s, horizontal endpoints held 0.5s', flush=True)
    return metadata


if __name__ == '__main__':
    main()
