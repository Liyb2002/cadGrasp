"""Opaque, depth-buffered Juxtapose animation and one final PNG.

Both panels show alternative uses of the same fixed fixture placement.
The right panel tries native pose 2, retracts, edits material, and retries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont

import geometry as G

COLORS = dict(object='#a4a8ac', support='#319cd7', removed='#ef4938', added='#21c875')


@njit(cache=True)
def raster(triangles, colors, image, depth, left, top, right, bottom):
    for k in range(len(triangles)):
        a, b, c = triangles[k, 0], triangles[k, 1], triangles[k, 2]
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-10:
            continue
        xmin = max(left, int(np.floor(min(a[0], b[0], c[0]))))
        xmax = min(right-1, int(np.ceil(max(a[0], b[0], c[0]))))
        ymin = max(top, int(np.floor(min(a[1], b[1], c[1]))))
        ymax = min(bottom-1, int(np.ceil(max(a[1], b[1], c[1]))))
        for y in range(ymin, ymax+1):
            for x in range(xmin, xmax+1):
                u = ((b[1]-c[1])*(x+.5-c[0]) + (c[0]-b[0])*(y+.5-c[1]))/den
                v = ((c[1]-a[1])*(x+.5-c[0]) + (a[0]-c[0])*(y+.5-c[1]))/den
                if u < 0 or v < 0 or u+v > 1:
                    continue
                z = u*a[2] + v*b[2] + (1-u-v)*c[2]
                if z > depth[y, x]+1e-9:
                    depth[y, x] = z
                    for channel in range(3):
                        image[y, x, channel] = colors[k, channel]


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'+name, size)


class Renderer:
    def __init__(self, meshes, report, width=1280, height=720, include_approach=False):
        self.meshes = meshes
        self.report = report
        self.width, self.height = width, height
        elevation, azimuth = np.radians([24., -55.])
        self.camera = np.array([np.cos(elevation)*np.cos(azimuth),
                               np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
        self.right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
        self.up = np.cross(self.camera, self.right)
        self.light = np.array([1., -1., 1.8])
        self.light /= np.linalg.norm(self.light)
        self.direction = np.asarray(report['guest_direction_world'])
        points = np.vstack([meshes[x].vertices for x in ['initial', 'final', 'host', 'guest']])
        if include_approach:
            # Show the whole approaching rabbit at the actual first blockage,
            # rather than clipping its head above the video viewport.
            points = np.vstack([points, meshes['guest'].vertices+.105*self.direction])
        self.center = (points.min(0)+points.max(0))/2
        horizontal = np.ptp(points @ self.right)
        vertical = np.ptp(points @ self.up)
        self.scale = min((width/2-100)/horizontal, (height-210)/vertical)*.98
        self.base = np.full((height, width, 3), 255, np.uint8)

    def project(self, mesh, shift, panel):
        triangles = mesh.triangles + np.asarray(shift)
        normals = mesh.face_normals
        visible = normals @ self.camera > 0
        triangles, normals = triangles[visible], normals[visible]
        q = triangles-self.center
        x = q @ self.right*self.scale + (panel+.5)*self.width/2
        y = -q @ self.up*self.scale + self.height*.56
        z = triangles @ self.camera
        return np.stack([x, y, z], axis=2), normals

    def frame(self, materials, guest_distance, caption, guest_visible=True):
        pixels = self.base.copy()
        depth = np.full((self.height, self.width), -np.inf)
        for panel in range(2):
            body = self.meshes['host' if panel == 0 else 'guest']
            shift = self.direction*guest_distance if panel == 1 else np.zeros(3)
            layers = [(body, 'object', shift)] if panel == 0 or guest_visible else []
            layers += [(mesh, color, np.zeros(3)) for mesh, color in materials]
            projected, shades = [], []
            for mesh, color, move in layers:
                if not len(mesh.faces):
                    continue
                triangles, normals = self.project(mesh, move, panel)
                rgb = np.asarray(ImageDraw.ImageColor.getrgb(COLORS[color]))
                brightness = .60+.40*np.maximum(normals @ self.light, 0)
                projected.append(triangles)
                shades.append(np.asarray(brightness[:, None]*rgb, dtype=np.uint8))
            if projected:
                raster(np.concatenate(projected), np.concatenate(shades), pixels, depth,
                       int(panel*self.width/2)+5, 124, int((panel+1)*self.width/2)-5, self.height-58)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.text((self.width/2, 27), 'Juxtapose', fill='#273340', font=font(26, True), anchor='mm')
        draw.text((self.width/2, 64), caption, fill='#46505a', font=font(20), anchor='mm')
        draw.text((self.width/4, 104), 'Pose 1  |  original use', fill='#303840', font=font(18), anchor='mm')
        draw.text((3*self.width/4, 104), 'Pose 2  |  same fixture placement', fill='#303840', font=font(18), anchor='mm')
        draw.line([(self.width/2, 92), (self.width/2, self.height-62)], fill='#e8ecef', width=1)
        for i, (label, color) in enumerate([('Object', 'object'), ('Fixture', 'support'),
                                           ('Remove', 'removed'), ('Add', 'added')]):
            x = self.width/2-260+i*145
            y = self.height-30
            draw.rectangle([x, y-6, x+14, y+7], fill=COLORS[color])
            draw.text((x+22, y), label, fill='#46505a', font=font(16), anchor='lm')
        return np.asarray(image)


def load():
    data = G.OUT/'data'
    source = np.load(data/'geometry.npz')
    meshes = {name: G.C.trimesh.Trimesh(source[name+'_vertices'], source[name+'_faces'], process=False)
              for name in ['initial', 'retained', 'removed', 'added', 'final', 'host', 'guest']}
    return meshes, json.loads((data/'geometry.json').read_text())


def collision_stop(initial, guest, direction):
    body = G.C.S.solid(guest)
    def overlap(distance):
        return G.C.material_volume(initial ^ body.translate((direction*distance/G.C.S.SCALE).tolist()))
    samples = np.linspace(.25, 0., 51)
    values = [overlap(t) for t in samples]
    first = next(i for i, v in enumerate(values) if v > G.TOL)
    low, high = samples[first], samples[first-1]
    for _ in range(15):
        middle = (low+high)/2
        if overlap(middle) > G.TOL:
            low = middle
        else:
            high = middle
    stop = max(0., high-.001)
    return stop, overlap(stop)


def smooth(u):
    return u*u*(3-2*u)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--still-only', action='store_true')
    args = parser.parse_args()
    if args.rebuild or not (G.OUT/'data/geometry.npz').exists():
        specimen = G.Specimen()
        specimen.save()
    meshes, report = load()
    renderer = Renderer(meshes, report)
    final = renderer.frame([(meshes['final'], 'support')], 0., 'One fixture placement, two alternative object poses')
    png = G.OUT/'pose1_pose2_final.png'
    Image.fromarray(final).save(png)
    if args.still_only:
        print(png, flush=True)
        return
    renderer = Renderer(meshes, report, include_approach=True)
    solids = {name: G.C.S.solid(meshes[name]) for name in ['initial', 'retained', 'removed', 'added', 'final']}
    stop, collision_volume = collision_stop(solids['initial'], meshes['guest'], renderer.direction)
    fps = 12
    video = G.OUT/'pose1_pose2_juxtapose.mp4'
    encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{renderer.width}x{renderer.height}', '-r', str(fps),
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    frames, timeline = 0, []

    def emit(materials, distance, caption, repeat=1, guest_visible=True, phase=''):
        nonlocal frames
        frame = renderer.frame(materials, distance, caption, guest_visible)
        for _ in range(repeat):
            encoder.stdin.write(frame.tobytes())
        timeline.append(dict(phase=phase, first_frame=frames, repetitions=repeat,
                             guest_exit_offset_m=float(distance), guest_visible=guest_visible))
        frames += repeat

    try:
        original = [(meshes['initial'], 'support')]
        highlighted = [(meshes['retained'], 'support'), (meshes['removed'], 'removed')]
        emit(original, .20, 'Keep the fixture fixed in pose 1', 18, False, 'initial')
        for u in np.linspace(0, 1, 19)[1:]:
            emit(original, .20+(stop-.20)*smooth(u), 'Try inserting pose 2', phase='attempt')
        emit(highlighted, stop, 'Blocked: pose 2 cannot seat', 24, phase='blocked')
        for u in np.linspace(0, 1, 9)[1:]:
            emit(highlighted, stop+(.20-stop)*smooth(u), 'Retract before changing the design', phase='retract')
        bounds = meshes['removed'].bounds
        for u in np.linspace(0, 1, 13)[1:]:
            z = bounds[1, 2]-(bounds[1, 2]-bounds[0, 2]+.001)*u
            visible = solids['removed'] ^ G.box([[-1, -1, -1], [1, 1, max(-.99, z)]])
            emit([(meshes['retained'], 'support'), (G.C.S.unpack(visible), 'removed')],
                 .20, 'Remove material blocking the new object and its path', guest_visible=False, phase='remove')
        emit([(meshes['retained'], 'support')], .20, 'Now there is room, but support must be added', 12, False, 'cut')
        bounds = meshes['added'].bounds
        for u in np.linspace(0, 1, 25)[1:]:
            z = bounds[0, 2]+(bounds[1, 2]-bounds[0, 2]+.001)*smooth(u)
            grown = solids['added'] ^ G.box([[-1, -1, -1], [1, 1, z]])
            emit([(meshes['retained'], 'support'), (G.C.S.unpack(grown), 'added')],
                 .20, 'Grow bottom supports outside every required exit path', guest_visible=False, phase='grow')
        colored = [(meshes['retained'], 'support'), (meshes['added'], 'added')]
        emit(colored, .20, 'Keep useful material; add bottom contacts and connections', 12, False, 'grown')
        for u in np.linspace(0, 1, 25)[1:]:
            emit(colored, .20*(1-smooth(u)), 'Retry pose 2 in the same fixture placement', phase='insert')
        emit(colored, 0., 'Pose 2 seats on the new bottom supports', 24, phase='seated')
        emit([(meshes['final'], 'support')], 0., 'One fixture placement, two alternative object poses', 30, phase='final')
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('ffmpeg failed')
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    record = dict(complete=True, demo_only=True, fps=fps, frame_count=frames,
                  duration_seconds=frames/fps, frame_size_px=[renderer.width, renderer.height],
                  fixture_moves=False, panels_are_alternative_uses=True,
                  blocked_exit_offset_m=float(stop), blocked_body_overlap_cm3=collision_volume*1e6,
                  geometry=report, timeline=timeline,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), Path(G.__file__)]},
                  artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [video, png]})
    (G.OUT/'data/animation.json').write_text(json.dumps(record, indent=2)+'\n')
    print('COMPLETE', video, f'{frames/fps:.1f}s', flush=True)


if __name__ == '__main__':
    main()
