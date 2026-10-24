from pathlib import Path
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

WIDTH, HEIGHT, FPS = 1600, 900, 24
DURATION, TURN_BEGIN, TURN_END = 16., 3., 15.


def disk(vertices, color):
    # A filled minimum convex polygon on its actual plane, never a fitted circle.
    faces = [[0, i, i+1] for i in range(1, len(vertices)-1)]
    piece = C.V.piece(C.trimesh.Trimesh(vertices, faces, process=False), color, .34)
    piece['unlit'] = True
    return piece


def edge(vertices, color, view, size):
    piece = C.H.outline_piece(vertices, color, view, size, half_width_pixels=.9)
    piece['colors'] = np.broadcast_to(C.V.CPU.rgb(color), piece['triangles'].shape)
    piece['opacity'] = .84
    return piece


def main():
    inp = C.Inputs('B', D.BASE/'pose1+11+14+27/data/selection.json')
    inp.base = D.BASE
    data = inp.load(inp.cases[0])
    assert [r['pose'] for r in data['rows']] == ['pose_1', 'pose_11', 'pose_14', 'pose_27']
    output = PROOF/'videos'
    original_hashes = {p.name:C.digest(p) for p in output.glob('*.mp4')}
    reference = data['rows'][0]['fixture_transform']
    support = C.posed(data['fixture'], reference)
    hulls = []
    for row in data['rows']:
        inverse = np.linalg.inv(row['fixture_transform'])
        common = C.F.transform_points(row['hull'], inverse)
        assert np.allclose(C.F.transform_points(common, row['fixture_transform']), row['hull'], atol=1e-12)
        shown = C.F.transform_points(common, reference)
        normal = reference[:3, :3]@row['fixture_transform'][2, :3]
        assert np.ptp(shown@normal) < 1e-12
        hulls.append(shown)
    points = np.vstack([support.vertices, *hulls])
    center = (points.min(0)+points.max(0))/2
    view0 = np.array([1., -1., 1.])/np.sqrt(3)
    right0 = np.array([1., 1., 0.])/np.sqrt(2)
    def view(angle, span):
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
        sight, right = rotation@view0, rotation@right0
        return center, np.array([right, np.cross(sight, right), sight]), span
    span = 0.
    for angle in np.linspace(0., 2*np.pi, 361):
        projected = (points-center)@view(angle, 1.)[1][:2].T
        span = max(span, float(np.max(np.abs(projected)))*2*1.12)
    start_view = view(0., span)
    end_view = view(2*np.pi, span)
    assert np.allclose(start_view[1], end_view[1], atol=1e-12)
    frozen_path = TMP/'execution'; frozen_path.mkdir()
    frozen = C.freeze(frozen_path)
    shutil.copy2(__file__, frozen_path/'render.py')
    own_hash = C.digest(__file__)
    renderer = C.V.Renderer()

    side_models = []
    for row in data['rows']:
        fixture = C.posed(data['fixture'], row['fixture_transform'])
        body = C.posed(data['body'], row['object_transform'])
        cam = C.V.camera_for(np.vstack([fixture.vertices, body.vertices, row['hull']]))
        side_models.append((row, fixture, body, cam))
    side_span = max(v[3][2] for v in side_models)
    right_panels = []
    for show_hulls in [False, True]:
        panel = Image.new('RGB', (700, HEIGHT), 'white')
        for i, (row, fixture, body, old_cam) in enumerate(side_models):
            cam = old_cam[0], old_cam[1], side_span
            parts = [C.V.piece(fixture, C.V.BLUE, .82), C.V.piece(body, C.V.GRAY, .18)]
            if show_hulls:
                parts += [disk(row['hull'], row['color']), edge(row['hull'], row['color'], cam, 340)]
            tile = renderer.render([C.V.CPU.placed(p) for p in parts], cam, 340)
            panel.paste(tile, (7+(i%2)*346, 55+(i//2)*450))
        right_panels.append(panel)

    def frame(angle, show_hulls):
        cam = view(angle, span)
        parts = [C.V.piece(support, C.V.BLUE, .78)]
        if show_hulls:
            for hull, row in zip(hulls, data['rows']):
                parts += [disk(hull, row['color']), edge(hull, row['color'], cam, HEIGHT)]
        left = renderer.render([C.V.CPU.placed(p) for p in parts], cam, HEIGHT)
        canvas = Image.new('RGB', (WIDTH, HEIGHT), 'white')
        canvas.paste(left, (0, 0))
        canvas.paste(right_panels[int(show_hulls)], (900, 0))
        draw = ImageDraw.Draw(canvas)
        draw.line((900, 25, 900, HEIGHT-25), fill='#e6edf1', width=1)
        draw.line((920, 450, WIDTH-20, 450), fill='#e6edf1', width=1)
        return canvas

    initial = frame(0., False)
    stacked = frame(0., True)
    for name, image in [('support_first', initial), ('angle_000', stacked)]:
        image.save(TMP/(name+'.png'))
    for degrees in [90, 180, 270]:
        frame(np.radians(degrees), True).save(TMP/f'angle_{degrees:03}.png')
    print('PREVIEWS READY', flush=True)
    destination = TMP/'01_hulls_and_support.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-profile:v', 'main',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i in range(round(DURATION*FPS)):
            t = i/FPS
            if t < 2.:
                image = initial
            elif t <= TURN_BEGIN or t >= TURN_END:
                image = stacked
            else:
                angle = 2*np.pi*(t-TURN_BEGIN)/(TURN_END-TURN_BEGIN)
                image = frame(angle, True)
            encoder.stdin.write(np.asarray(image).tobytes())
            if i % 48 == 0:
                print('FRAME', i, '/', round(DURATION*FPS), flush=True)
    finally:
        encoder.stdin.close()
    assert encoder.wait() == 0
    stream = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(destination)]))['streams'][0]
    assert stream['nb_frames'] == '384' and stream['r_frame_rate'] == '24/1' and stream['pix_fmt'] == 'yuv420p'
    assert stream['width'] == WIDTH and stream['height'] == HEIGHT and float(stream['duration']) == DURATION
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', '-i', str(destination), '-f', 'null', '-'], check=True)
    inp.verify()
    for relative, expected in frozen.items():
        assert C.digest(ROOT/relative) == expected == C.digest(frozen_path/'executed_sources'/relative)
    assert C.digest(__file__) == own_hash == C.digest(frozen_path/'render.py')
    for name, digest in original_hashes.items():
        assert C.digest(output/name) == digest
    print('CHECKS PASSED: four actual hulls, filled transparent polygons, full 360-degree camera orbit, no text or ground grid', flush=True)


if __name__ == '__main__':
    main()
