from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path('/home/yli581/Desktop/cadGrasp')
PROOF = ROOT/'slides/Co-optimize/seperate_base_support_proof'
TMP = Path(__file__).resolve().parent
sys.path.insert(0, str(PROOF))
import diagnose as D
C = D.C

FPS, WIDTH, HEIGHT = 24, 1280, 720


def ease(t, start, end):
    v = float(np.clip((t-start)/(end-start), 0., 1.))
    return v*v*(3-2*v)


def outline(vertices, color, camera, fraction):
    remaining = float(np.linalg.norm(np.roll(vertices, -1, axis=0)-vertices, axis=1).sum()*fraction)
    segments = []
    for first, last in zip(vertices, np.roll(vertices, -1, axis=0)):
        length = float(np.linalg.norm(last-first))
        if remaining <= 1e-14:
            break
        segments.append([first, first+(last-first)*min(1., remaining/length)])
        remaining -= length
    if not segments:
        return None
    segments = np.asarray(segments)
    delta = (segments[:, 1]-segments[:, 0])@camera[1][:2].T
    length = np.linalg.norm(delta, axis=1)
    side = np.c_[-delta[:, 1], delta[:, 0]]/np.maximum(length[:, None], 1e-20)
    offset = side@camera[1][:2]*(1.5*camera[2]/HEIGHT)
    corners = np.stack([segments[:, 0]+offset, segments[:, 1]+offset,
                        segments[:, 1]-offset, segments[:, 0]-offset], axis=1)
    triangles = corners[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3, 3)
    return dict(triangles=triangles, normals=np.broadcast_to(camera[1][2], triangles.shape),
        colors=np.broadcast_to(C.V.CPU.rgb(color), triangles.shape), bias=6e-6, unlit=True)


def fill(vertices, color, opacity):
    faces = [[0, i, i+1] for i in range(1, len(vertices)-1)]
    return C.V.piece(C.trimesh.Trimesh(vertices, faces, process=False), color, opacity)


def video_one():
    inp = C.Inputs('B', D.BASE/'illegal/pose11+19/data/selection.json')
    inp.base = D.BASE
    data = inp.load(inp.cases[0])
    reference = data['rows'][0]['fixture_transform']
    support = C.posed(data['fixture'], reference)
    hulls = []
    for row in data['rows']:
        # A single shared rigid transform follows the true inverse fixture pose.
        common = C.F.transform_points(row['hull'], np.linalg.inv(row['fixture_transform']))
        assert np.allclose(C.F.transform_points(common, row['fixture_transform']), row['hull'], atol=1e-12)
        hulls.append(C.F.transform_points(common, reference))
    points = np.vstack([support.vertices, *hulls])
    center = (points.min(0)+points.max(0))/2
    default_view = np.array([1., -1., 1.])/np.sqrt(3)
    default_right = np.array([1., 1., 0.])/np.sqrt(2)
    def camera(t):
        angle = .65*ease(t, 6.6, 9.5)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
        view, right = rotation@default_view, rotation@default_right
        return center, np.array([right, np.cross(view, right), view]), span
    maximum = 0.
    span = 1.
    for t in np.linspace(0., 10., 65):
        projected = (points-center)@camera(t)[1][:2].T
        maximum = max(maximum, float(np.max(np.abs(projected))))
    span = maximum*2*1.13
    frozen = TMP/'execution'
    frozen.mkdir()
    code = C.freeze(frozen)
    shutil.copy2(__file__, frozen/'revise.py')
    own_hash = C.digest(__file__)
    renderer = C.V.Renderer()
    path = TMP/'staged/01_hulls_and_support.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-profile:v', 'main',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i in range(10*FPS):
            t = i/FPS
            cam = camera(t)
            parts = [C.V.piece(support, C.V.BLUE, .91)]
            for vertices, row, start, end in zip(hulls, data['rows'], [2., 4.], [4., 6.]):
                fraction = ease(t, start, end)
                if fraction > 0:
                    parts.append(fill(vertices, row['color'], .07*fraction))
                    parts.append(outline(vertices, row['color'], cam, fraction))
            image = renderer.render([C.V.CPU.placed(part) for part in parts], cam, HEIGHT)
            frame = Image.new('RGB', (WIDTH, HEIGHT), 'white')
            frame.paste(image, ((WIDTH-HEIGHT)//2, 0))
            encoder.stdin.write(np.asarray(frame).tobytes())
            if i in [24, 72, 120, 228]:
                frame.save(TMP/f'one_{i}.png')
            if i % 48 == 0:
                print('VIDEO 1', i, '/240', flush=True)
    finally:
        encoder.stdin.close()
    assert encoder.wait() == 0
    inp.verify()
    for relative, expected in code.items():
        assert C.digest(ROOT/relative) == expected
        assert C.digest(frozen/'executed_sources'/relative) == expected
    assert C.digest(__file__) == own_hash == C.digest(frozen/'revise.py')
    print('VIDEO 1 COMPLETE: support first, one common 3D frame, zero text', flush=True)


def strip_other_videos():
    # Keep only the known 3D viewports, excluding every burnt-in text region.
    crops = {'02_ground_collision.mp4': 'crop=568:452:132:144',
             '03_lift_and_layout.mp4': 'crop=1208:400:30:150',
             '04_work_cones.mp4': 'crop=568:454:132:140'}
    for name, crop in crops.items():
        filters = crop+',scale=1280:720:force_original_aspect_ratio=decrease:flags=lanczos,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(TMP/'original'/name),
            '-vf', filters, '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(TMP/'staged'/name)], check=True)
        print('TEXT REMOVED', name, flush=True)


def check():
    for path in sorted((TMP/'staged').glob('*.mp4')):
        stream = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams'][0]
        assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
        assert (stream['width'], stream['height']) == (WIDTH, HEIGHT)
        assert stream['r_frame_rate'] == '24/1'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)
        if not path.name.startswith('01'):
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', '15', '-i', str(path),
                '-frames:v', '1', str(TMP/(path.stem+'_clean.png'))], check=True)
        print('DECODE PASSED', path.name, stream['duration'], stream['nb_frames'], flush=True)
    assert len(list((TMP/'staged').glob('*.mp4'))) == 4


if __name__ == '__main__':
    video_one()
    strip_other_videos()
    check()
