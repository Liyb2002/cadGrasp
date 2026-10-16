"""Saved five-pose demand hulls and illustrative foot growth, in video 01's style.

All task floors are pulled back through their own saved installation transform,
then displayed with ONE fixed absolute fixture transform. The blue hollow rims
and short connecting rods are explanatory geometry, following the requested
reference movie; they are not substituted for any certified numerical result.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent
COOPT = OUTPUT.parents[1]
ROOT = COOPT.parents[1]
sys.path[:0] = [str(ROOT), *[str(COOPT / name) for name in ['helper_func', 'step5.1', 'step5.3', 'vis_func']]]
import _bootstrap
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from deterministic_base import solid, unpack, volume, union, moved
import step51_render as V
import hull_render as H

CASE = 'pose1+2+4+5+6'
POSES = ['pose_1', 'pose_2', 'pose_4', 'pose_5', 'pose_6']
BASE = COOPT / 'data/results/B' / CASE
STEP4 = BASE / 'step4/step4.2/pso_all52_20261013_01'
FLOOR = BASE / 'step5/step5.1/pso_all52_20261013_01_floor'
REFERENCE = COOPT / 'seperate_base_support_proof/videos/01_hulls_and_support.mp4'
MOVIE = OUTPUT / '01_hulls_and_support.mp4'
FPS, SECONDS, MAIN, SMALL = 24, 8, 900, 292
WIDTH, HEIGHT, RADIUS = .004, .005, .003
GROW_START, LINK_END, RIM_END, ORBIT_END = .9, 2.1, 6.1, 7.75


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def posed(mesh, matrix):
    result = mesh.copy()
    result.apply_transform(matrix)
    return result


def placed(piece, matrix):
    return V.CPU.placed(piece, rotation=matrix[:3, :3], point=matrix[:3, 3])


def smooth(value):
    value = float(np.clip(value, 0., 1.))
    return value * value * (3. - 2. * value)


def basis(angle):
    angle += np.deg2rad(75.)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    view = rotation @ (np.array([1., -1., 1.]) / np.sqrt(3))
    right = rotation @ (np.array([1., 1., 0.]) / np.sqrt(2))
    return np.array([right, np.cross(view, right), view])


def prism(xy):
    n = len(xy)
    vertices = np.c_[xy, np.zeros(n)]
    faces = [[0, i + 1, i] for i in range(1, n - 1)]
    faces += [[n, n + i, n + i + 1] for i in range(1, n - 1)]
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, j, n + j], [i, n + j, n + i]])
    mesh = trimesh.Trimesh(np.vstack([vertices, vertices + [0., 0., HEIGHT]]), faces, process=False)
    if mesh.volume < 0.:
        mesh.invert()
    if not mesh.is_watertight:
        raise ValueError('Open illustrative rim cell')
    return solid(mesh)


def rod(start, end):
    vector = end - start
    length = float(np.linalg.norm(vector))
    matrix = trimesh.geometry.align_vectors([0., 0., 1.], vector / length)
    matrix[:3, 3] = (start + end) / 2
    return trimesh.creation.cylinder(radius=RADIUS, height=length, sections=24, transform=matrix)


class Ring:
    def __init__(self, row, support):
        self.inverse = np.linalg.inv(row['T'])
        xy = row['hull_world'][:, :2]
        edges = np.roll(xy, -1, axis=0) - xy
        outward = np.c_[edges[:, 1], -edges[:, 0]] / np.linalg.norm(edges, axis=1)[:, None]
        outer = np.array([np.linalg.solve(
            np.array([outward[i - 1], outward[i]]),
            np.array([outward[i - 1] @ xy[i], outward[i] @ xy[i]]) + WIDTH)
            for i in range(len(xy))])
        inner_polygon, outer_polygon = Polygon(xy), Polygon(outer)
        if not inner_polygon.is_valid or not outer_polygon.covers(inner_polygon):
            raise ValueError('Invalid offset of an actual demand hull')
        band = outer_polygon.difference(inner_polygon)
        if len(band.interiors) != 1:
            raise ValueError('The foot rim must leave the hull interior empty')
        middle = (xy + outer) / 2
        candidates, indices = [], []
        for i in range(len(xy)):
            for fraction in [0., .25, .5, .75]:
                point = middle[i] * (1 - fraction) + middle[(i + 1) % len(xy)] * fraction
                candidates.append([*point, HEIGHT / 2])
                indices.append((i, fraction))
        candidates = transform(candidates, self.inverse)
        roots, distances, _ = trimesh.proximity.closest_point_naive(support, candidates)
        chosen = int(np.argmin(distances))
        i, fraction = indices[chosen]
        self.end = candidates[chosen]
        if distances[chosen] < 1e-8:
            raise ValueError('Degenerate illustrative connecting rod')
        self.start = roots[chosen] - (self.end - roots[chosen]) / distances[chosen] * .002
        self.gap = float(distances[chosen])
        if fraction == 0.:
            self.inner, self.outer = np.roll(xy, -i, axis=0), np.roll(outer, -i, axis=0)
        else:
            order = [(i + 1 + j) % len(xy) for j in range(len(xy))]
            self.inner = np.vstack([xy[i] * (1 - fraction) + xy[(i + 1) % len(xy)] * fraction, xy[order]])
            self.outer = np.vstack([outer[i] * (1 - fraction) + outer[(i + 1) % len(xy)] * fraction, outer[order]])
        centers = (self.inner + self.outer) / 2
        self.cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.roll(centers, -1, axis=0) - centers, axis=1))]
        import manifold3d as md
        self.prefix = [md.Manifold()]
        for i in range(len(self.inner)):
            j = (i + 1) % len(self.inner)
            cell = prism(np.array([self.inner[i], self.inner[j], self.outer[j], self.outer[i]]))
            self.prefix.append(self.prefix[-1] + cell)
        full = self.prefix[-1]
        if not np.isclose(volume(full), band.area * HEIGHT, rtol=1e-8):
            raise ValueError('Unexpected rim volume')
        if volume(full ^ prism(xy)) > 1e-14:
            raise ValueError('The rim filled its demand hull')
        self.solid = moved(full, self.inverse)
        self.mesh = posed(unpack(full), self.inverse)
        self.link = rod(self.start, self.end)
        if volume(solid(self.link) ^ self.solid) <= 1e-14:
            raise ValueError('The illustrative rod does not reach the foot')

    def mesh_at(self, amount):
        if amount >= 1.:
            return self.mesh
        distance = amount * self.cumulative[-1]
        i = min(int(np.searchsorted(self.cumulative, distance, side='right') - 1), len(self.inner) - 1)
        fraction = (distance - self.cumulative[i]) / (self.cumulative[i + 1] - self.cumulative[i])
        j = (i + 1) % len(self.inner)
        value = self.prefix[i]
        if fraction > 1e-8:
            value = value + prism(np.array([
                self.inner[i], (1 - fraction) * self.inner[i] + fraction * self.inner[j],
                (1 - fraction) * self.outer[i] + fraction * self.outer[j], self.outer[i]]))
        return posed(unpack(value), self.inverse)


class Movie:
    def __init__(self):
        self.inputs = {}
        self.sources = {}
        self.rows = []
        self.protect(REFERENCE)
        for path in (COOPT / 'idea').glob('*.mp4'):
            self.protect(path)
        for path in [ROOT / 'objects/B/mesh.stl', STEP4 / 'support.obj', STEP4 / 'layout.npz',
                     STEP4 / 'data/report.json', FLOOR / 'data/report.json']:
            self.protect(path)
        report = json.loads((FLOOR / 'data/report.json').read_text())
        if not report['complete'] or report['poses'] != POSES:
            raise ValueError('Wrong or incomplete saved five-pose input')
        support = trimesh.load(STEP4 / 'support.obj', force='mesh', process=False)
        self.support = support
        self.body = trimesh.load(ROOT / 'objects/B/mesh.stl', force='mesh', process=False)
        arrays, pose_records = {}, []
        for row in report['per_pose']:
            path = FLOOR / row['arrays']
            self.protect(path)
            with np.load(path) as saved:
                cloud = saved['floor_demands_world_m'].copy()
                hull_indices = saved['hull_indices'].copy()
                T, U = saved['T_world_fixture'].copy(), saved['T_world_object'].copy()
                stored_common = saved['floor_demands_fixture_m'].copy()
            if cloud.shape != (32768, 3) or not np.all(cloud[:, 2] == 0.):
                raise ValueError('The hull must come from all original task-floor samples')
            inverse = np.linalg.inv(T)
            common_cloud = transform(cloud, inverse)
            hull = common_cloud[hull_indices]
            error = float(np.max(np.abs(transform(common_cloud, T) - cloud)))
            if error > 1e-12 or np.max(np.abs(common_cloud - stored_common)) > 1e-12:
                raise ValueError('Saved absolute/common frame mismatch')
            ids = np.unique(np.r_[np.linspace(0, len(cloud) - 1, 256, dtype=int), hull_indices])
            entry = dict(pose=row['pose'], color=row['color'], T=T, U=U,
                         hull_world=cloud[hull_indices], hull=hull, dots=common_cloud[ids])
            self.rows.append(entry)
            prefix = row['pose'] + '_'
            arrays.update({prefix + 'T_world_fixture': T, prefix + 'T_world_object': U,
                           prefix + 'hull_world_m': cloud[hull_indices], prefix + 'hull_fixture_m': hull,
                           prefix + 'display_sample_indices': ids, prefix + 'cloud_fixture_m': common_cloud})
            pose_records.append(dict(pose=row['pose'], color=row['color'], original_demand_count=32768,
                                     displayed_sample_count=len(ids), hull_vertex_count=len(hull),
                                     transform_roundtrip_max_error_m=error,
                                     arrays=relative(path), T_world_fixture=T.tolist()))
        self.ref = self.rows[0]['T'].copy()
        arrays['fixed_absolute_fixture_transform'] = self.ref
        np.savez_compressed(HERE / 'data/geometry.npz', **arrays)
        self.rings = [Ring(row, support) for row in self.rows]
        original = solid(support)
        for ring in self.rings:
            if volume(solid(ring.link) ^ original) <= 1e-14:
                raise ValueError('Illustrative connector does not join the saved support')
        result = union([original, *[r.solid for r in self.rings], *[solid(r.link) for r in self.rings]])
        # Decomposition counts material, retaining any real interior cavities.
        positive = [part for part in result.decompose() if float(part.volume()) > 1e-8]
        if len(positive) != 1:
            raise ValueError('Illustrative support is disconnected')
        combined = unpack(result)
        if not combined.is_watertight:
            raise ValueError('Open final illustrative support')
        combined.export(HERE / 'data/illustrative_support.obj', digits=17)
        all_local = np.vstack([combined.vertices, *[r['hull'] for r in self.rows]])
        shown = transform(all_local, self.ref)
        self.focus = (shown.min(0) + shown.max(0)) / 2
        self.span = 2 * max(float(np.abs((shown - self.focus) @ basis(a)[:2].T).max())
                            for a in np.linspace(0., 2 * np.pi, 361)) * 1.12
        self.cameras, self.bodies = [], []
        for row in self.rows:
            body = posed(self.body, row['U'])
            self.bodies.append(V.CPU.placed(V.piece(body, V.GRAY, .18)))
            self.cameras.append(V.camera_for(np.vstack([transform(all_local, row['T']), body.vertices])))
        right_span = max(camera[2] for camera in self.cameras)
        self.cameras = [(f, b, right_span) for f, b, _ in self.cameras]
        self.renderer = V.Renderer()
        self.left_support, self.right_support = V.piece(support, V.BLUE, .78), V.piece(support, V.BLUE, .82)
        self.static_right = {}
        self.record = dict(complete=False, pose_set=CASE, poses=pose_records,
            movie=relative(MOVIE), reference=relative(REFERENCE), duration_seconds=SECONDS,
            fps=FPS, frames=FPS * SECONDS, width=1600, height=900, text=False, grid=False,
            main_view='one fixed absolute fixture pose, camera orbit only',
            fixed_absolute_fixture_transform=self.ref.tolist(),
            hull_transform='T_fixed @ inverse(T_world_fixture_i) @ hull_world_i',
            floor_planes_projected_or_merged=False, orbit_degrees=360,
            original_samples_per_pose=32768, hulls_from_all_original_samples=True,
            graphical_dots_are_fixed_subset_of_original_samples=True,
            growth_starts_seconds=GROW_START, links_finish_seconds=LINK_END, rims_finish_seconds=RIM_END,
            ring_width_m=WIDTH, ring_height_m=HEIGHT, connector_radius_m=RADIUS,
            explanatory_geometry=True, geometry_scope='reference-style hollow hull rims and short rods',
            physical_solver_run=False, oracle_query_run=False, layout_search_run=False,
            certified_results_modified=False, illustrative_connected_components=1,
            illustrative_watertight=True, ring_interiors_empty=True,
            connector_gaps_mm=[r.gap * 1000 for r in self.rings],
            main_camera=dict(focus_m=self.focus.tolist(), span_m=self.span),
            right_panels=dict(pose_order=POSES, positions=[[905, 4], [1251, 4], [905, 304], [1251, 304], [1078, 604]],
                              size=SMALL, all_task_hulls_carried_with_fixture=True))
        save(HERE / 'data/manifest.json', self.record)
        self.freeze_sources()
        print(json.dumps(dict(prepared=True, poses=POSES, connector_gaps_mm=self.record['connector_gaps_mm'],
                              inputs=len(self.inputs), sources=len(self.sources))), flush=True)

    def protect(self, path):
        self.inputs[relative(path)] = digest(path)

    def freeze_sources(self):
        paths = {Path(__file__).resolve(), V.CPU_PATH,
                 V.BASELINE / 'step4_connect_support/translucent_raster.cpp'}
        for module in list(sys.modules.values()):
            file = getattr(module, '__file__', None)
            if file:
                path = Path(file).resolve()
                if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / '.venv') and path.is_file():
                    paths.add(path)
        for path in paths:
            key = relative(path)
            target = HERE / 'data/executed_sources' / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            self.sources[key] = digest(path)
        save(HERE / 'data/inputs.json', self.inputs)
        save(HERE / 'data/sources.json', self.sources)

    def parts(self, matrix, camera, size, t, support_piece, material):
        pieces = [placed(support_piece, matrix), *[placed(p, matrix) for p in material]]
        for row in self.rows:
            hull = transform(row['hull'], matrix)
            fade = 1 - smooth((t - GROW_START) / .5)
            if fade > 1e-5:
                mesh = trimesh.Trimesh(hull, [[0, i, i + 1] for i in range(1, len(hull) - 1)], process=False)
                platform = V.piece(mesh, row['color'], .06 * fade)
                platform['unlit'] = True
                pieces.append(V.CPU.placed(platform))
            outline = H.outline_piece(hull, row['color'], camera, size, half_width_pixels=.7 if size == MAIN else .45)
            outline['colors'] = np.broadcast_to(V.CPU.rgb(row['color']), outline['triangles'].shape)
            outline['opacity'] = .48
            pieces.append(V.CPU.placed(outline))
            points = V.points_piece(transform(row['dots'], matrix), row['color'], camera, size,
                                    half_size_pixels=1.05 if size == MAIN else .60)
            points['opacity'] = .8
            pieces.append(V.CPU.placed(points))
        return pieces

    def frame(self, t):
        camera = self.focus, basis(2 * np.pi * min(t / ORBIT_END, 1.)), self.span
        link_amount = smooth((t - GROW_START) / (LINK_END - GROW_START))
        rim_amount = smooth((t - LINK_END) / (RIM_END - LINK_END))
        material = []
        for ring in self.rings:
            if link_amount > 1e-5:
                material.append(V.piece(rod(ring.start, ring.start + (ring.end - ring.start) * link_amount), V.BLUE, .82))
            if rim_amount > 1e-5:
                material.append(V.piece(ring.mesh_at(rim_amount), V.BLUE, .82))
        parts = self.parts(self.ref, camera, MAIN, t, self.left_support, material)
        main = self.renderer.render(parts, camera, MAIN)
        key = 'initial' if t <= GROW_START else 'final' if t >= RIM_END else None
        if key in self.static_right:
            panel = self.static_right[key]
        else:
            panel = Image.new('RGB', (700, 900), 'white')
            for row, body, cam, (x, y) in zip(self.rows, self.bodies, self.cameras,
                                             self.record['right_panels']['positions']):
                parts = self.parts(row['T'], cam, SMALL, t, self.right_support, material) + [body]
                panel.paste(self.renderer.render(parts, cam, SMALL), (x - 900, y))
            if key is not None:
                self.static_right[key] = panel
        image = Image.new('RGB', (1600, 900), 'white')
        image.paste(main, (0, 0))
        image.paste(panel, (900, 0))
        draw = ImageDraw.Draw(image)
        draw.line((900, 0, 900, 899), fill='#eef0f2', width=1)
        draw.line((901, 300, 1599, 300), fill='#f1f2f4', width=1)
        draw.line((901, 600, 1599, 600), fill='#f1f2f4', width=1)
        return image

    def verify(self):
        for key, expected in self.inputs.items():
            if digest(ROOT / key) != expected:
                raise ValueError('Saved input or existing movie changed: ' + key)
        for key, expected in self.sources.items():
            if digest(ROOT / key) != expected or digest(HERE / 'data/executed_sources' / key) != expected:
                raise ValueError('Rendering source changed: ' + key)
        return dict(passed=True, original_input_and_movie_hashes_unchanged=True,
                    protected_inputs=len(self.inputs), executed_sources=len(self.sources))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encode', action='store_true')
    args = parser.parse_args()
    (HERE / 'data').mkdir(parents=True, exist_ok=True)
    movie = Movie()
    times = [0., 1.6, 3., 4.5, 6.1, 7.75]
    sheet = Image.new('RGB', (1600, 1350), 'white')
    for i, t in enumerate(times):
        frame = movie.frame(t)
        frame.save(HERE / 'data' / f'preview_{i:02d}.png')
        sheet.paste(frame.resize((800, 450), Image.Resampling.LANCZOS), ((i % 2) * 800, (i // 2) * 450))
    sheet.save(HERE / 'storyboard.png')
    save(HERE / 'data/verification.json', movie.verify())
    if not args.encode:
        return
    staging = HERE / 'data/01_hulls_and_support.staging.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1600x900', '-r', str(FPS), '-i', 'pipe:0', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-profile:v', 'main',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(staging)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    began = time.monotonic()
    try:
        for i in range(FPS * SECONDS):
            encoder.stdin.write(movie.frame(i / FPS).tobytes())
            if i % FPS == 0:
                print(json.dumps(dict(frame=i, frames=FPS * SECONDS, seconds=round(time.monotonic() - began, 1))), flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('FFmpeg encoding failed')
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=width,height,r_frame_rate,nb_frames,codec_name', '-of', 'json', str(staging)]))
    stream = probe['streams'][0]
    if (stream['width'], stream['height'], int(stream['nb_frames'])) != (1600, 900, FPS * SECONDS):
        raise ValueError('Wrong encoded dimensions or frame count')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(staging), '-f', 'null', '-'], check=True)
    verification = movie.verify()
    staging.replace(MOVIE)
    verification.update(full_decode_passed=True, encoded_probe=probe, movie_sha256=digest(MOVIE),
                        movie=relative(MOVIE), storyboard_sha256=digest(HERE / 'storyboard.png'))
    save(HERE / 'data/verification.json', verification)
    movie.record.update(complete=True, movie_sha256=digest(MOVIE), encoding_seconds=time.monotonic() - began)
    save(HERE / 'data/manifest.json', movie.record)
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
