"""Video 02: opaque thin ground and an airborne pose change.

Read the historical pose19 / pose11 layout and demands. Both illustrative
open rims are attached from the first frame and stay fixed throughout.
The opaque ground depth-tests every piece, including collision coloring.
"""
from pathlib import Path
import argparse
import json
import hashlib
import subprocess
import sys
import time

import manifold3d as md
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import trimesh

HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
COOPT = PROOF.parent
ROOT = COOPT.parents[1]
sys.path[:0] = [str(ROOT), str(COOPT/'vis_func')]
import step51_render as V

FPS, FRAMES, WIDTH, HEIGHT = 24, 192, 1600, 900


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def points_transform(points, transform):
    return np.asarray(points)@transform[:3, :3].T + transform[:3, 3]


def smooth(value):
    value = float(np.clip(value, 0, 1))
    return value*value*(3-2*value)


def solid(mesh):
    value = md.Manifold(md.Mesh64(np.ascontiguousarray(mesh.vertices*1000, dtype=np.float64),
                                 np.ascontiguousarray(mesh.faces, dtype=np.uint32)))
    if value.status() != md.Error.NoError:
        raise ValueError(str(value.status()))
    return value


def unpack(value):
    data = value.to_mesh64()
    return trimesh.Trimesh(np.asarray(data.vert_properties[:, :3])/1000,
                           np.asarray(data.tri_verts), process=False)


def volume(value):
    if value.status() != md.Error.NoError:
        raise ValueError(str(value.status()))
    return abs(float(value.volume()))*1e-9


def transform_solid(value, transform):
    affine = transform[:3].copy()
    affine[:, 3] *= 1000
    return value.transform(affine.tolist())


def rod(a, b, radius=.003):
    delta = b-a
    length = np.linalg.norm(delta)
    transform = trimesh.geometry.align_vectors([0, 0, 1], delta/length)
    transform[:3, 3] = (a+b)/2
    return trimesh.creation.cylinder(radius=radius, height=length, sections=20, transform=transform)


def offset(xy, width):
    edges = np.roll(xy, -1, axis=0)-xy
    normals = np.c_[edges[:, 1], -edges[:, 0]]/np.linalg.norm(edges, axis=1)[:, None]
    outside = np.array([np.linalg.solve([normals[i-1], normals[i]],
        [normals[i-1]@xy[i]+width, normals[i]@xy[i]+width]) for i in range(len(xy))])
    return outside, normals


class Model:
    def __init__(self):
        directory = PROOF/'output/B/illegal/pose11+19'
        report_path = directory/'step5/step5.1/data/report.json'
        report = json.loads(report_path.read_text())
        selected = ROOT/report['selected_step4_source']
        mesh_path = ROOT/'objects/B/mesh.stl'
        support_path = selected/'support.obj'
        paths = [report_path, mesh_path, support_path, selected/'layout.npz',
                 directory/'data/selection.json', selected/'data/report.json']
        self.body = trimesh.load(mesh_path, force='mesh', process=False)
        self.support = trimesh.load(support_path, force='mesh', process=False)
        self.rows = []
        for row in report['per_pose']:
            path = directory/'step5/step5.1'/row['arrays']
            paths.append(path)
            with np.load(path, allow_pickle=False) as arrays:
                cloud = arrays['floor_demands_world_m'].copy()
                self.rows.append(dict(pose=row['pose'], color=row['color'], cloud=cloud,
                    hull=cloud[arrays['hull_indices']].copy(),
                    fixture_transform=arrays['T_world_fixture'].copy(),
                    object_transform=arrays['T_world_object'].copy()))
            assert len(cloud) == 32768
        self.start = next(r for r in self.rows if r['pose'] == 'pose_19')
        self.end = next(r for r in self.rows if r['pose'] == 'pose_11')
        self.T0, self.T1 = self.start['fixture_transform'], self.end['fixture_transform']
        self.inverse = np.linalg.inv(self.T0)
        certificate_path = PROOF/'certificates/illegal/pose11+19/report.json'
        paths.append(certificate_path)
        certificate = json.loads(certificate_path.read_text())['ground_certificates'][0]
        change = self.T1@self.inverse
        np.testing.assert_allclose(change[2, :2], certificate['inequality_a_xy'], atol=1e-12)
        assert abs(change[2, 3]-certificate['inequality_offset_m']) < 1e-12
        self.xy = self.start['hull'][:, :2]
        assert np.all(self.xy@change[2, :2]+change[2, 3] < -1e-9)
        lengths = np.linalg.norm(np.roll(self.xy, -1, axis=0)-self.xy, axis=1)
        self.gap_edge = int(np.argmax(lengths))
        self.initial_base = self.rim(self.start, .004)
        self.second_base = self.rim(self.end, .004)
        self.bases = self.initial_base+self.second_base
        for row, base in [(self.start, self.initial_base), (self.end, self.second_base)]:
            contacts = points_transform(unpack(base).vertices, row['fixture_transform'])
            contacts = contacts[abs(contacts[:, 2]) < 1e-9, :2]
            hull = ConvexHull(contacts)
            assert np.max(row['hull'][:, :2]@hull.equations[:, :2].T+hull.equations[:, 2]) < 1e-10
        # Retain all saved support pieces and attach both illustrative bases.
        pieces = self.support.split(only_watertight=False)
        accumulated = pieces[0]
        links = []
        for piece in pieces[1:]:
            close, distance, _ = trimesh.proximity.closest_point_naive(accumulated, piece.vertices)
            i = int(np.argmin(distance))
            a, b = close[i], piece.vertices[i]
            direction = (b-a)/np.linalg.norm(b-a)
            links.append(rod(a-direction*.002, b+direction*.002))
            accumulated = trimesh.util.concatenate([accumulated, piece])
        for base in [self.initial_base, self.second_base]:
            candidates = unpack(base).triangles_center
            close, distance, _ = trimesh.proximity.closest_point_naive(self.support, candidates)
            i = int(np.argmin(distance))
            if distance[i] > 1e-9:
                a, b = close[i], candidates[i]
                direction = (b-a)/np.linalg.norm(b-a)
                links.append(rod(a-direction*.002, b+direction*.002))
        self.core = md.Manifold.batch_boolean([solid(self.support), *[solid(p) for p in links]], md.OpType.Add)
        self.core_mesh = unpack(self.core)
        self.fixture_mesh = unpack(self.core+self.bases)
        assert len((self.core+self.bases).decompose()) == 1
        self.protected = {path: digest(path) for path in paths}
        self.certificate = certificate

    def rim(self, row, width):
        xy = row['hull'][:, :2]
        lengths = np.linalg.norm(np.roll(xy, -1, axis=0)-xy, axis=1)
        outer, normals = offset(xy, width)
        band = Polygon(outer).difference(Polygon(xy))
        i = int(np.argmax(lengths))
        j = (i+1) % len(xy)
        middle = (xy[i]+xy[j])/2
        tangent = xy[j]-xy[i]
        tangent /= np.linalg.norm(tangent)
        normal = normals[i]
        half_gap = min(.018, lengths[i]*.6)/2
        # Keep the original open rim's gap.
        cut = Polygon([middle-tangent*half_gap-normal*.005,
                       middle+tangent*half_gap-normal*.005,
                       middle+tangent*half_gap+normal*.080,
                       middle-tangent*half_gap+normal*.080])
        band = band.difference(cut)
        assert band.geom_type == 'Polygon' and not band.interiors
        rim = md.CrossSection([np.array(band.exterior.coords)[:-1]*1000], md.FillRule.NonZero).extrude(5.)
        return transform_solid(rim, np.linalg.inv(row['fixture_transform']))

    def verify_inputs(self):
        assert all(digest(path) == checksum for path, checksum in self.protected.items())


class Renderer(V.Renderer):
    def render(self, parts, camera, size=(WIDTH, HEIGHT)):
        width, height = size
        w, h = width*2, height*2
        focus, basis, span = camera
        light = basis[2]*.72+basis[1]*.5-basis[0]*.22
        light /= np.linalg.norm(light)
        fill = basis[2]*.25+basis[1]*.3+basis[0]*.65
        fill /= np.linalg.norm(fill)
        triangles, colors, alphas = [], [], []
        for mesh, rotation, point in parts:
            if len(mesh['triangles']) == 0:
                continue
            p = (mesh['triangles']@rotation.T+point-focus)@basis.T
            p[..., 0] = w/2+p[..., 0]*h/span
            p[..., 1] = h/2-p[..., 1]*h/span
            p[..., 2] += mesh['bias']
            normals = mesh['normals']@rotation.T
            shade = .64+.27*np.maximum(0, normals@light)+.12*np.maximum(0, normals@fill)
            if mesh.get('unlit'):
                shade[:] = 1.
            color = mesh['colors']*shade[..., None]
            alpha = mesh.get('opacity', 1.)
            if mesh.get('blue_tint'):
                color = color*alpha+255*(1-alpha)
                alpha = 1.  # Pale color with opaque depth: nothing underground leaks through.
            triangles.append(p.reshape(-1, 9))
            colors.append(color.reshape(-1, 9))
            alphas.extend([alpha]*len(p))
        p = np.ascontiguousarray(np.concatenate(triangles), dtype=np.float32)
        c = np.ascontiguousarray(np.concatenate(colors), dtype=np.float32)
        a = np.ascontiguousarray(alphas, dtype=np.float32)
        output = np.empty((h, w, 3), dtype=np.uint8)
        self.raster(p.ctypes.data, c.ctypes.data, len(p), w, h, a.ctypes.data, output.ctypes.data)
        return Image.fromarray(output).resize(size, Image.Resampling.LANCZOS)


def placed(piece, transform):
    return V.CPU.placed(piece, rotation=transform[:3, :3], point=transform[:3, 3])


def blue_piece(mesh):
    return dict(V.piece(mesh, V.BLUE, .78), blue_tint=True)


class Movie:
    def __init__(self):
        self.model = m = Model()
        self.renderer = Renderer()
        minimum = min(points_transform(m.fixture_mesh.vertices, T)[:, 2].min() for T in [m.T0, m.T1])
        self.lift_height = max(.085, -float(minimum)+.020)
        points = []
        for T in [m.T0, m.T1]:
            transformed = points_transform(m.fixture_mesh.vertices, T)
            points.extend([transformed, transformed+[0, 0, self.lift_height]])
        points.append(points_transform(m.body.vertices, m.start['object_transform']))
        points.extend(r['cloud'] for r in m.rows)
        points = np.vstack(points)
        edge = m.xy[(m.gap_edge+1) % len(m.xy)]-m.xy[m.gap_edge]
        angle = np.arctan2(-edge[0], edge[1])
        view = np.array([np.cos(angle), np.sin(angle), .48])
        view /= np.linalg.norm(view)
        right = np.array([-np.sin(angle), np.cos(angle), 0.])
        self.basis = np.array([right, np.cross(view, right), view])
        forward = np.r_[-view[:2]/np.linalg.norm(view[:2]), 0.]
        rotation = np.array([right, forward, [0., 0., 1.]]).T
        theta = np.radians(25.)
        rotation = rotation@np.array([[np.cos(theta), -np.sin(theta), 0.],
                                     [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        underground = points[points[:, 2] < 0]
        floor_hits = underground-underground[:, 2, None]*view/view[2]
        planar = np.vstack([points, floor_hits])@rotation
        low, high = planar[:, :2].min(axis=0)-.015, planar[:, :2].max(axis=0)+.015
        vertices = np.array([[low[0], low[1], 0.], [high[0], low[1], 0.],
                             [high[0], high[1], 0.], [low[0], high[1], 0.]])@rotation.T
        floor = trimesh.Trimesh(vertices, [[0, 1, 2], [0, 2, 3]], process=False)
        assert np.max(np.abs(floor.vertices[:, 2])) < 1e-12
        self.ground = V.CPU.placed(dict(V.piece(floor, '#EEF1F4', 1.), unlit=True))
        all_points = np.vstack([points, floor.vertices])
        projected = all_points@self.basis.T
        lo, hi = projected.min(axis=0), projected.max(axis=0)
        focus = ((lo+hi)/2)@self.basis
        span = max(hi[1]-lo[1], (hi[0]-lo[0])*HEIGHT/WIDTH)*1.08
        self.camera = focus, self.basis, span
        self.core = blue_piece(m.core_mesh)
        self.body = V.piece(m.body, V.GRAY, .18)
        self.marks = []
        for row in [m.start, m.end]:
            hull = row['hull']
            marks = []
            for a, b in zip(hull, np.roll(hull, -1, axis=0)):
                length = np.linalg.norm((b-a)@self.basis[:2].T)*HEIGHT/span
                count = max(1, round(length/3.8))
                marks.extend(a+(b-a)*np.arange(count)[:, None]/count)
            self.marks.append([V.CPU.placed(V.points_piece(row['cloud'][::64], row['color'], self.camera, HEIGHT, 1.)),
                               V.CPU.placed(V.points_piece(marks, row['color'], self.camera, HEIGHT, .70))])
        self.base = blue_piece(unpack(m.bases))
        below = transform_solid(m.initial_base, m.T1).trim_by_plane((0., 0., -1.), 0.)
        self.below_volume_cm3 = volume(below)*1e6
        # Ordinary depth testing hides underground material. Only a real cut
        # at the floor can remain visible.
        cut = V.piece(unpack(below), '#D94C5D', 1.)
        cut['bias'] = 2e-6
        self.cut = V.CPU.placed(cut)

    def state(self, t):
        m = self.model
        if t < 2.3:
            transform = m.T0.copy()
            height = self.lift_height*smooth(t-1.)
        else:
            transform = m.T1.copy()
            height = self.lift_height*(1-smooth((t-2.6)/1.2))
        transform[2, 3] += height
        return transform

    def frame(self, t):
        m = self.model
        transform = self.state(t)
        parts = [self.ground, placed(self.core, transform), placed(self.base, transform)]
        if t < 1.:
            opacity = .18*(1-smooth((t-.8)/.2))
            if opacity > 1e-5:
                parts.append(placed(dict(self.body, opacity=opacity), m.start['object_transform']))
            parts.extend(self.marks[0])
        if t >= 3.8:
            parts.extend(self.marks[1])
            parts.append(self.cut)
        return self.renderer.render(parts, self.camera)

    def verify(self):
        m = self.model
        np.testing.assert_allclose(self.state(0.), m.T0, atol=1e-12)
        np.testing.assert_allclose(self.state(3.8), m.T1, atol=1e-12)
        for t in np.linspace(0, 8, 160):
            transform = self.state(t)
            installed = m.T0 if t < 2.3 else m.T1
            np.testing.assert_allclose(transform[:3, :3], installed[:3, :3], atol=1e-12)
            np.testing.assert_allclose(transform[:2, 3], installed[:2, 3], atol=1e-12)
        for t in [2., 2.29, 2.3, 2.6]:
            assert points_transform(m.fixture_mesh.vertices, self.state(t))[:, 2].min() >= .019
        assert self.below_volume_cm3 > 0
        np.testing.assert_array_equal(self.state(4.3), self.state(7.75))
        m.verify_inputs()
        return dict(duration_seconds=8, fps=FPS, size=[WIDTH, HEIGHT],
            pose_order=[m.start['pose'], m.end['pose']], lift_height_m=self.lift_height,
            stages=['installed pose19', 'vertical pickup', 'instant airborne pose change',
                    'vertical set-down', 'installed pose11'],
            rotation_interpolated=False, camera_fixed=True, ground_opacity=1., ground_thickness_m=0.,
            below_floor_xray=False, collision_color_depth_bias_m=2e-6,
            base_enlargement_m=0., second_pose_workpiece_shown=False,
            base_count=2, both_bases_present_from_first_frame=True,
            scatter_marker_half_size_pixels=1., scatter_points_per_pose=512,
            base_below_ground_cm3=self.below_volume_cm3,
            saved_numeric_inputs_unchanged=True)


def encode(movie, path):
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
        stdin=subprocess.PIPE)
    started = time.monotonic()
    try:
        for i in range(FRAMES):
            process.stdin.write(movie.frame(i/FPS).tobytes())
            if i % 24 == 0:
                print(f'02: {i}/{FRAMES}, {time.monotonic()-started:.1f} s', flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Video encoding failed')
    except BaseException:
        process.kill()
        process.wait()
        raise
    movie.model.verify_inputs()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected = {p: digest(p) for p in [HERE/'01_hulls_and_support.mp4', HERE/'04_work_cones.mp4']}
    movie = Movie()
    record = movie.verify()
    print(json.dumps(record), flush=True)
    if args.preview:
        for t, name in [(0., 'start'), (2.15, 'lift'), (2.45, 'next_pose_airborne'),
                        (3.4, 'drop'), (4.3, 'landed'), (7.75, 'final')]:
            movie.frame(t).save(args.output_dir/f'{name}.png')
    else:
        encode(movie, args.output_dir/'02_ground_collision.mp4')
    assert all(digest(path) == checksum for path, checksum in protected.items())
    (args.output_dir/'render02.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
