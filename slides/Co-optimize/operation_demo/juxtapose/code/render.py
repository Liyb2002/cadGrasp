"""One guest-only video: remove material only as the real body sweeps into it.

The existing two-panel final PNG remains the work-relieved reference geometry.
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
    def __init__(self, meshes, report, width=1280, height=720, include_approach=False,
                 annotations=True, guest_only=False, approach_distance=.105):
        self.meshes = meshes
        self.report = report
        self.annotations = annotations
        self.guest_only = guest_only
        self.panel_count = 1 if guest_only else 2
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
            points = np.vstack([points, meshes['guest'].vertices+approach_distance*self.direction])
        self.center = (points.min(0)+points.max(0))/2
        horizontal = np.ptp(points @ self.right)
        vertical = np.ptp(points @ self.up)
        margins = (100, 80) if guest_only else (100, 210)
        self.scale = min((width/self.panel_count-margins[0])/horizontal,
                         (height-margins[1])/vertical)*.98
        self.base = np.full((height, width, 3), 255, np.uint8)

    def project(self, mesh, shift, panel):
        triangles = mesh.triangles + np.asarray(shift)
        normals = mesh.face_normals
        visible = normals @ self.camera > 0
        triangles, normals = triangles[visible], normals[visible]
        q = triangles-self.center
        x = q @ self.right*self.scale + (panel+.5)*self.width/self.panel_count
        y = -q @ self.up*self.scale + self.height*(.5 if self.guest_only else .56)
        z = triangles @ self.camera
        return np.stack([x, y, z], axis=2), normals

    def frame(self, materials, guest_distance, caption, guest_visible=True):
        pixels = self.base.copy()
        depth = np.full((self.height, self.width), -np.inf)
        for panel in range(self.panel_count):
            is_guest = self.guest_only or panel == 1
            body = self.meshes['guest' if is_guest else 'host']
            shift = self.direction*guest_distance if is_guest else np.zeros(3)
            # A newly visited red fragment wins coplanar ties against the body.
            # It is still opaque and uses the same depth buffer as everything else.
            layers = [(mesh, color, np.zeros(3)) for mesh, color in materials if color == 'removed']
            layers += [(body, 'object', shift)] if not is_guest or guest_visible else []
            layers += [(mesh, color, np.zeros(3)) for mesh, color in materials if color != 'removed']
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
                       int(panel*self.width/self.panel_count)+5,
                       20 if self.guest_only else 124,
                       int((panel+1)*self.width/self.panel_count)-5,
                       self.height-(20 if self.guest_only else 58))
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        if not self.annotations:
            if not self.guest_only:
                draw.line([(self.width/2, 92), (self.width/2, self.height-62)], fill='#e8ecef', width=1)
            return np.asarray(image)
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
    names = ['initial', 'retained', 'removed', 'added', 'final', 'host', 'guest', 'guest_wrap']
    names += [f'growth_{i}' for i in range(1, G.GROWTH_STEPS+1)]
    meshes = {name: G.C.trimesh.Trimesh(source[name+'_vertices'], source[name+'_faces'], process=False)
              for name in names}
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
    solids = {name: G.C.S.solid(meshes[name]) for name in ['initial', 'retained', 'removed', 'added', 'final']}
    direction = np.asarray(report['guest_direction_world'])
    stop, collision_volume = collision_stop(solids['initial'], meshes['guest'], direction)
    entrance = stop+.035
    renderer = Renderer(meshes, report, width=720, height=720, include_approach=True,
                        annotations=False, guest_only=True, approach_distance=entrance)
    fps = 193/6  # Keep all 193 geometry frames in exactly six seconds.
    video = G.OUT/'pose1_pose2_juxtapose.mp4'
    pending_video = video.with_name('.pose1_pose2_juxtapose.rendering.mp4')
    encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{renderer.width}x{renderer.height}', '-r', str(fps),
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(pending_video)], stdin=subprocess.PIPE)
    frames, timeline, cut_checks = 0, [], []
    body = G.C.S.solid(meshes['guest'])

    def body_at(distance):
        return body.translate((direction*distance/G.C.S.SCALE).tolist())

    def actual_sweep(near, far):
        transform = np.eye(4)
        transform[:3, 3] = direction*near
        moved = G.mesh_in(meshes['guest'], transform)
        return G.C.S.solid(G.C.S.swept_solid(moved, (far-near)*direction, fan_in=8))

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
        emit(original, entrance, '', 12, False, 'initial')
        for u in np.linspace(0, 1, 19)[1:]:
            emit(original, entrance+(stop-entrance)*smooth(u), '', phase='attempt')
        # Highlight ONLY the material actually reached by the object so far.
        swept = actual_sweep(stop, entrance)
        reached = solids['initial'] ^ swept
        remaining = solids['initial'] - swept
        highlighted = [(G.C.S.unpack(remaining), 'support'), (G.C.S.unpack(reached), 'removed')]
        emit(highlighted, stop, '', 18, phase='blocked')
        previous_distance = entrance
        remaining = solids['initial']
        swept = body_at(entrance)
        for index, distance in enumerate(stop*(1-smooth(np.linspace(0, 1, 41)))):
            segment = actual_sweep(float(distance), previous_distance)
            current_body = body_at(float(distance))
            # Pin the actual endpoint body as part of its sweep. Independent
            # Mesh64 remeshes of nearly coplanar boundaries can otherwise lose
            # an endpoint sliver. Neither input is padded or enlarged.
            segment = segment + current_body
            swept = swept + segment
            next_remaining = remaining - segment
            fresh = remaining - next_remaining
            removed = solids['initial'] - next_remaining
            outside = G.C.material_volume(removed - swept)
            overlap = G.C.material_volume(next_remaining ^ current_body)
            restored = G.C.material_volume(next_remaining - remaining)
            assert max(outside, overlap, restored) <= G.TOL, (
                f'Removal must follow the visited body sweep: frame={index}, '
                f'outside={outside}, overlap={overlap}, restored={restored}')
            cut_checks.append(dict(guest_exit_offset_m=float(distance),
                newly_removed_volume_cm3=G.C.material_volume(fresh)*1e6,
                cumulative_removed_volume_cm3=G.C.material_volume(removed)*1e6,
                removed_outside_actual_visited_sweep_m3=outside,
                remaining_current_body_overlap_m3=overlap,
                previously_deleted_material_restored_m3=restored))
            emit([(G.C.S.unpack(next_remaining), 'support'), (G.C.S.unpack(fresh), 'removed')],
                 float(distance), '', phase='sweep_cut')
            remaining = next_remaining
            previous_distance = float(distance)
            if index % 10 == 0:
                print('ACTUAL SWEEP', index+1, '/ 41', flush=True)
        remaining_mesh = G.C.S.unpack(remaining)
        emit([(remaining_mesh, 'support')], 0., '', 12, phase='cut')
        for index in range(1, G.GROWTH_STEPS+1):
            emit([(remaining_mesh, 'support'), (meshes[f'growth_{index}'], 'added')],
                 0., '', repeat=4, phase='grow')
        colored = [(remaining_mesh, 'support'), (meshes['added'], 'added')]
        emit(colored, 0., '', 12, phase='grown')
        emit(colored, 0., '', 18, phase='seated')
        # Never snap to the work-relieved reference: that would silently erase
        # the tiny work-clearance scraps which the physical body never visited.
        # Keep the same two opaque layers; only the added layer's color changes.
        emit([(remaining_mesh, 'support'), (meshes['added'], 'support')],
             0., '', 30, phase='final')
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('ffmpeg failed')
        pending_video.replace(video)
    except BaseException:
        encoder.kill()
        encoder.wait()
        pending_video.unlink(missing_ok=True)
        raise
    extra_work_relief = G.C.material_volume(remaining - solids['retained'])
    reference_missing = G.C.material_volume(solids['retained'] - remaining)
    record = dict(complete=True, demo_only=True, fps=fps, base_animation_fps=12,
                  playback_speed=fps/12, frame_count=frames,
                  duration_seconds=frames/fps, frame_size_px=[renderer.width, renderer.height],
                  fixture_moves=False, panel_count=1, visible_object_pose='pose_2',
                  on_screen_text=False, on_screen_legend=False,
                  material_removal_rule='Initial support minus the continuously visited, exact guest-body insertion sweep; no precomputed future cuts or height wipe',
                  insertion_start_offset_m=float(entrance), cut_checks=cut_checks,
                  canonical_work_relief_not_animated_cm3=extra_work_relief*1e6,
                  canonical_retained_material_missing_from_animation_m3=reference_missing,
                  animation_retained_volume_cm3=G.C.material_volume(remaining)*1e6,
                  animation_added_volume_cm3=report['added_volume_cm3'],
                  animation_final_render_layers=['actual_sweep_retained', 'saved_available_fitted_wrap'],
                  unswept_material_deleted=False,
                  blocked_exit_offset_m=float(stop), blocked_body_overlap_cm3=collision_volume*1e6,
                  reference_still_geometry=report, timeline=timeline,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), Path(G.__file__)]},
                  artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [video, png]})
    (G.OUT/'data/animation.json').write_text(json.dumps(record, indent=2)+'\n')
    print('COMPLETE', video, f'{frames/fps:.1f}s', flush=True)


if __name__ == '__main__':
    main()
