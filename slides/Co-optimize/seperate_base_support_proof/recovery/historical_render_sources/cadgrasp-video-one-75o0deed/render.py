from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/home/yli581/Desktop/cadGrasp')
PROOF = ROOT/'slides/Co-optimize/seperate_base_support_proof'
TMP = Path(__file__).resolve().parent
sys.path.insert(0, str(PROOF))
import diagnose as D
C = D.C


def camera(points, sight=None, span=None):
    if sight is None:
        return C.V.camera_for(points)
    sight = np.asarray(sight)/np.linalg.norm(sight)
    right = np.cross([0., 0., 1.], sight); right /= np.linalg.norm(right)
    basis = np.array([right, np.cross(sight, right), sight])
    projected = np.asarray(points)@basis.T
    low, high = projected.min(0), projected.max(0)
    return ((low+high)/2)@basis, basis, span or max(high[:2]-low[:2])*1.20


def polygon(vertices, color):
    mesh = C.trimesh.Trimesh(vertices, [[0, i, i+1] for i in range(1, len(vertices)-1)], process=False)
    return C.V.piece(mesh, color, .11)


def grid(points, view, size, opacity=.09):
    xy = np.asarray(points)[:, :2]
    low, high = xy.min(0)-.008, xy.max(0)+.008
    boundary = np.array([[*low, 0.], [high[0], low[1], 0.], [*high, 0.], [low[0], high[1], 0.]])
    lines = []
    for u in np.linspace(0., 1., 5):
        x, y = low+u*(high-low)
        lines.extend([np.array([[x, low[1], 0.], [x, high[1], 0.], [x, low[1], 0.]]),
                      np.array([[low[0], y, 0.], [high[0], y, 0.], [low[0], y, 0.]])])
    pieces = [C.V.floor_piece(boundary, color='#dfe7ed', opacity=opacity)]
    for line in lines:
        piece = C.H.outline_piece(line, '#adbecb', view, size, half_width_pixels=.30)
        pieces.append(piece)
    return pieces


def main():
    inp = C.Inputs('B', D.BASE/'illegal/pose11+19/data/selection.json')
    inp.base = D.BASE
    data = inp.load(inp.cases[0])
    output = PROOF/'videos'
    other_hashes = {p.name:C.digest(p) for p in output.glob('*.mp4') if not p.name.startswith('01')}
    reference = data['rows'][0]['fixture_transform']
    support = C.posed(data['fixture'], reference)
    hulls, normals = [], []
    for row in data['rows']:
        common = C.F.transform_points(row['hull'], np.linalg.inv(row['fixture_transform']))
        hull = C.F.transform_points(common, reference)
        normal = reference[:3, :3]@row['fixture_transform'][2, :3]
        assert np.ptp(hull@normal) < 1e-12
        assert np.allclose(C.F.transform_points(common, row['fixture_transform']), row['hull'], atol=1e-12)
        hulls.append(hull); normals.append(normal)
    angle = np.degrees(np.arccos(np.clip(normals[0]@normals[1], -1., 1.)))
    assert 26.8 < angle < 27.
    direction = np.cross(normals[0], normals[1]); direction /= np.linalg.norm(direction)
    # Look roughly along the intersection of the two planes; their relative tilt remains visible.
    sight = direction+.55*normals[0]
    common_camera = camera(np.vstack([support.vertices, *hulls]), sight)
    posed_supports = [C.posed(data['fixture'], row['fixture_transform']) for row in data['rows']]
    posed_bodies = [C.posed(data['body'], row['object_transform']) for row in data['rows']]
    cameras = [camera(np.vstack([fixture.vertices, body.vertices, row['hull']]))
               for fixture, body, row in zip(posed_supports, posed_bodies, data['rows'])]
    common_span = max(view[2] for view in cameras)
    cameras = [(focus, basis, common_span) for focus, basis, _ in cameras]
    execution = TMP/'execution'; execution.mkdir()
    frozen = C.freeze(execution)
    shutil.copy2(__file__, execution/'render.py')
    own_hash = C.digest(__file__)
    renderer = C.V.Renderer()
    frames = []
    for stage in range(3):
        parts = [C.V.piece(support, C.V.BLUE, .86)]
        # This grid is the pose11 reference plane. Pose19's polygon remains on its true tilted plane.
        parts += grid(np.vstack(hulls), common_camera, 720, .065)
        for index, (hull, row) in enumerate(zip(hulls, data['rows'])):
            if index < stage:
                parts += [polygon(hull, row['color']), C.H.outline_piece(hull, row['color'], common_camera, 720, 1.6)]
        left = renderer.render([C.V.CPU.placed(p) for p in parts], common_camera, 720)
        frame = Image.new('RGB', (1280, 720), 'white')
        frame.paste(left, (27, 0))
        for index, (row, fixture, body, view) in enumerate(zip(data['rows'], posed_supports, posed_bodies, cameras)):
            side_parts = [C.V.piece(fixture, C.V.BLUE, .83), C.V.piece(body, C.V.GRAY, .20)]
            side_parts += grid(np.vstack([fixture.vertices, body.vertices, row['hull']]), view, 360, .11)
            if index < stage:
                side_parts += [polygon(row['hull'], row['color']), C.H.outline_piece(row['hull'], row['color'], view, 360, 1.4)]
            tile = renderer.render([C.V.CPU.placed(p) for p in side_parts], view, 360)
            frame.paste(tile, (838, index*360))
        draw = ImageDraw.Draw(frame)
        draw.line((785, 24, 785, 696), fill='#e3ebef', width=1)
        draw.line((812, 360, 1253, 360), fill='#e3ebef', width=1)
        frame.save(TMP/f'stage_{stage}.png')
        frames.append(frame)
    print('FRAMES READY', 'tilt', float(angle), 'degrees', flush=True)
    if '--encode' not in sys.argv:
        return
    destination = TMP/'01_hulls_and_support.mp4'
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '24', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '18', '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(destination)], stdin=subprocess.PIPE)
    try:
        for i in range(240):
            stage = 0 if i < 48 else (1 if i < 96 else 2)
            encoder.stdin.write(np.asarray(frames[stage]).tobytes())
    finally:
        encoder.stdin.close()
    assert encoder.wait() == 0
    stream = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(destination)]))['streams'][0]
    assert stream['nb_frames'] == '240' and stream['r_frame_rate'] == '24/1' and stream['pix_fmt'] == 'yuv420p'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', '-i', str(destination), '-f', 'null', '-'], check=True)
    inp.verify()
    for relative, expected in frozen.items():
        assert C.digest(ROOT/relative) == expected == C.digest(execution/'executed_sources'/relative)
    assert C.digest(__file__) == own_hash == C.digest(execution/'render.py')
    for name, expected in other_hashes.items():
        assert C.digest(output/name) == expected
    print('ENCODE AND INPUT CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
