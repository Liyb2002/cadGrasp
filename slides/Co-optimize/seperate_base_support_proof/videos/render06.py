"""Continue video 05 in pose20 and show the ground-enclosure Oracle.

Read the SAME saved layout, demand cloud and connected-space witness. Only
display geometry is derived: plane sections, footprints and graphical points.
No force solver, layout search or Oracle query is called.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import triangulate, unary_union

HERE = Path(__file__).resolve().parent
COOPT = HERE.parent.parent
ROOT = COOPT.parents[1]
sys.path[:0] = [str(COOPT/'helper_func'), str(COOPT/'step4.2'), str(COOPT/'vis_func')]
import _bootstrap
import render_floor_transfer as R
from base_oracle import clip, cover, slice_polygon, vertices
from co_common import S
from pose_oracle import transform_planes, transform_points
from whole_search.common import transform_mesh, transform_solid, unpack_solid
import step51_render as V

NAME = '06_pose20_ground_enclosure'
PURPLE, GREEN, RED = '#9751AF', '#148C71', '#D97D9C'


def contours_shape(contours):
    positive, negative = [], []
    for contour in contours:
        ring = np.asarray(contour)*S.SCALE
        if len(ring) < 3:
            continue
        shape = Polygon(ring)
        area = np.sum(ring[:, 0]*np.roll(ring[:, 1], -1)-ring[:, 1]*np.roll(ring[:, 0], -1))
        (positive if area > 0 else negative).append(shape)
    return unary_union(positive).difference(unary_union(negative))


def flat_mesh(shape):
    """Triangulate the actual polygon, preserving holes and disconnected parts."""
    triangles = []
    if shape.is_empty:
        return trimesh.Trimesh()
    polygons = [shape] if shape.geom_type == 'Polygon' else list(shape.geoms)
    for polygon in polygons:
        if polygon.geom_type != 'Polygon':
            continue
        for triangle in triangulate(polygon):
            if polygon.covers(triangle):
                triangles.append(np.c_[np.asarray(triangle.exterior.coords)[:3], np.zeros(3)])
    points = np.asarray(triangles).reshape(-1, 3)
    return trimesh.Trimesh(points, np.arange(len(points)).reshape(-1, 3), process=False)


def ring_segments(ring, fraction=1., dashed=False):
    ring = np.asarray(ring)
    if ring.shape[1] == 2:
        ring = np.c_[ring, np.zeros(len(ring))]
    lengths = np.linalg.norm(np.roll(ring, -1, axis=0)-ring, axis=1)
    remaining = lengths.sum()*np.clip(fraction, 0., 1.)
    segments = []
    for a, b, length in zip(ring, np.roll(ring, -1, axis=0), lengths):
        shown = min(length, remaining)
        if shown <= 0. or length <= 0.:
            break
        end = a+(b-a)*shown/length
        if dashed:
            count = max(1, int(np.ceil(shown/.004)))
            for k in range(count):
                lo, hi = k/count, min(1., (k+.58)/count)
                segments.append((a+(end-a)*lo, a+(end-a)*hi))
        else:
            segments.append((a, end))
        remaining -= shown
    return segments


def line_piece(segments, color, camera, size, pixels=2., opacity=1.):
    """Screen-width graphical lines: never interpreted as material or feet."""
    _, basis, span = camera
    triangles = []
    for a, b in segments:
        projected = (b-a)@basis[:2].T
        length = np.linalg.norm(projected)
        if length < 1e-12:
            continue
        normal = np.array([-projected[1], projected[0]])/length
        offset = normal@basis[:2]*(pixels*span/size/2.)
        triangles.extend([[a-offset, b-offset, b+offset], [a-offset, b+offset, a+offset]])
    if not triangles:
        return None
    triangles = np.asarray(triangles)
    return dict(triangles=triangles, normals=np.broadcast_to(basis[2], triangles.shape),
        colors=np.broadcast_to(V.CPU.rgb(color), triangles.shape), unlit=True,
        bias=2.*span, opacity=opacity)


class Movie:
    def __init__(self, size=900):
        self.size = size
        self.base = R.Movie(size)
        self.row = self.base.rows[1]
        self.protected = dict(self.base.hashes)
        numeric = COOPT/'data/results/B/pose2+20'
        paths = dict(
            previous_video=HERE/'05_pose2_to_pose20_oracle.mp4',
            layout=numeric/'step4/step4.2/base_oracle_v3/layout.npz',
            oracle=numeric/'step4/step4.2/base_oracle_v3/base_oracle.json',
            demands=numeric/'step5/step5.1/connected_base_v1/data/pose_20.npz')
        self.paths = paths
        for path in list(paths.values())+list(HERE.glob('0[124]_*.mp4')):
            self.protected[str(path.relative_to(ROOT))] = R.digest(path)
        saved_oracle = json.loads(paths['oracle'].read_text())
        self.saved_oracle = saved_oracle
        self.saved_row = next(r for r in saved_oracle['per_pose'] if r['pose'] == 'pose_20')
        with np.load(paths['demands']) as data:
            np.testing.assert_allclose(data['T_world_fixture'], self.row['T'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(data['T_world_object'], self.row['U'], atol=1e-12, rtol=0)
            self.cloud = data['floor_demands_world_m'].copy()
            self.hull = self.cloud[data['hull_indices'], :2].copy()
        assert len(self.cloud) == 32768
        np.testing.assert_allclose(self.hull, self.saved_row['demand_hull_xy_m'], atol=1e-12, rtol=0)
        self.required = Polygon(self.hull)
        self.body = transform_mesh(self.base.body, self.row['U'])
        self.fixture = transform_mesh(self.base.fixture, self.row['T'])
        self.carried = self.row['T']@np.linalg.inv(self.base.rows[0]['T'])
        self.other_floor_plane = (self.base.rows[0]['T']@np.linalg.inv(self.row['T']))[2]
        witness = trimesh.load(self.base.legal_space_path, force='mesh', process=False)
        ground = transform_points(witness.vertices, self.row['T'])[witness.faces]
        on_floor = np.max(np.abs(ground[:, :, 2]), axis=1) <= 1e-8
        self.saved_ground = unary_union([Polygon(t[:, :2]) for t in ground[on_floor]
            if Polygon(t[:, :2]).area > 1e-14])
        self.own = np.asarray(self.saved_row['object_ground_xy_m'], float).reshape(-1, 2)
        # Enclosure within a modest local window suffices to show opportunity.
        # The frame rectangle is only a display crop, never a legal obstacle.
        center = np.array(self.required.centroid.coords[0])
        initial = max(float(np.ptp(self.hull, axis=0).max())*.95, .075)
        for factor in (1., 1.5, 2., 3., 5.):
            half = initial*factor
            window = box(*(center-half), *(center+half))
            candidate = self.saved_ground.intersection(window)
            points = vertices(candidate)
            passed, gap, _ = cover(self.hull, points)
            if passed:
                break
        else:
            raise ValueError('The saved connected-space floor slice does not enclose this demand hull')
        self.window = window
        self.window_corners = np.asarray(window.exterior.coords)[:-1]
        # Reuse original exact sweep geometry, capped beyond this finite window.
        with np.load(paths['layout']) as data:
            directions = data['directions'].copy()
        # STL stores duplicate vertices per triangle. Join only EXACT equal
        # positions for the sweep kernel; no coordinates or geometry change.
        joined, indices = np.unique(self.base.body.vertices, axis=0, return_inverse=True)
        sweep_body = trimesh.Trimesh(joined, indices[self.base.body.faces], process=False)
        assert sweep_body.is_watertight and sweep_body.is_winding_consistent
        work_cuts, exit_cuts = [], []
        ground_bounds = np.c_[self.window_corners, np.zeros(4)]
        for k, row in enumerate(self.base.rows):
            world_from_local = self.row['T']@row['Q']
            for family, count in zip(row['work_access'].planes, row['work_access'].counts):
                world_planes = transform_planes(family[:count], np.linalg.inv(world_from_local))
                cut = self.window_corners.copy()
                for plane in world_planes:
                    cut = clip(cut, -plane[:2], -plane[3])
                shape = slice_polygon(cut)
                if not shape.is_empty:
                    work_cuts.append(shape)
            local = transform_points(ground_bounds, np.linalg.inv(world_from_local))
            direction = row['Q'][:3, :3].T@directions[k]
            length = max(.5, float((local@direction).max()-(self.base.body.vertices@direction).min())+.01)
            swept = S.solid(S.swept_solid(sweep_body, length*direction, fan_in=8))
            world_swept = transform_solid(swept, world_from_local)
            exit_cuts.append(contours_shape(world_swept.slice(0.).to_polygons()))
        floor_ring = clip(self.window_corners, self.other_floor_plane[:2], self.other_floor_plane[3])
        floor_legal = slice_polygon(floor_ring)
        self.work_forbidden = unary_union(work_cuts).intersection(window)
        self.exit_forbidden = unary_union(exit_cuts).intersection(window)
        self.floor_forbidden = window.difference(floor_legal)
        self.illegal = unary_union([self.work_forbidden, self.exit_forbidden, self.floor_forbidden])
        self.legal = window.difference(self.illegal)
        # Use only points from the saved common connected-space witness, also
        # check the current displayed normal-ray/exit/floor footprint.
        self.connected_ground = candidate.intersection(self.legal)
        points = vertices(MultiPoint(vertices(self.connected_ground)).convex_hull)
        passed, gap, _ = cover(self.hull, points)
        if not passed:
            raise ValueError('Displayed legal connected ground does not enclose the original hull')
        # Remove redundant graphical foot opportunities without adding points.
        for _ in range(len(points)):
            removed = False
            for i in range(len(points)):
                trial = np.delete(points, i, axis=0)
                if len(trial) >= 3 and cover(self.hull, trial)[0]:
                    points = trial
                    removed = True
                    break
            if not removed:
                break
        self.anchors = points
        self.attainable = vertices(MultiPoint(points).convex_hull)
        self.enclosed, self.gap, _ = cover(self.hull, points)
        assert self.enclosed
        for point in points:
            assert self.legal.distance(MultiPoint([point])) <= 1e-8
            assert self.saved_ground.distance(MultiPoint([point])) <= 1e-8
        self.legal_mesh = flat_mesh(self.legal)
        self.illegal_mesh = flat_mesh(self.illegal)
        self.hull_mesh = flat_mesh(self.required)
        self.envelope_mesh = flat_mesh(Polygon(self.attainable))
        # Check polygon triangulation really retained the holes.
        for shape, mesh in [(self.legal, self.legal_mesh), (self.illegal, self.illegal_mesh)]:
            assert abs(mesh.area-shape.area) <= max(1e-12, shape.area*1e-8)
        self.top_focus = np.r_[center, 0.]
        azimuth = -np.pi/4
        direction = np.array([np.cos(azimuth), np.sin(azimuth), 1/np.sqrt(2)])
        direction /= np.linalg.norm(direction)
        right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
        iso = np.array([right, np.cross(direction, right), direction])
        top = np.array([right, np.cross([0., 0., 1.], right), [0., 0., 1.]])
        self.top_span = float(np.ptp(ground_bounds@top[:2].T, axis=0).max()*1.14)
        self.camera_rotation = Slerp([0., 1.], Rotation.from_matrix([iso.T, top.T]))

    def camera(self, t):
        u = R.smooth((t-3.5)/3.)
        focus = (1-u)*self.base.focus+u*self.top_focus
        span = (1-u)*self.base.span+u*self.top_span
        return focus, self.camera_rotation(u).as_matrix().T, span

    def frame(self, t):
        camera = self.camera(t)
        reveal = R.smooth((t-4.)/2.5)
        hull_alpha = R.smooth((t-1.)/.9)
        envelope = R.smooth((t-8.5)/2.)
        parts = []

        def add(mesh, color, opacity=1., unlit=False, bias=0.):
            if opacity > 1e-6 and len(mesh.faces):
                parts.append(dict(V.piece(mesh, color, opacity), unlit=unlit, bias=bias))

        add(self.base.neutral_floor, '#ECEFF1', .13*(1-reveal), True, -3e-5)
        add(transform_mesh(self.base.first_floor, self.carried), '#EBAAC4', .32*(1-reveal), True, 1e-5)
        add(transform_mesh(self.base.first_outline, self.carried), '#CD6990', .72*(1-reveal), True, 2e-5)
        add(self.base.last_floor, '#F1B9CC', .32*(1-reveal), True, 1e-5)
        add(self.base.last_outline, '#D57C9B', .72*(1-reveal), True, 2e-5)
        add(self.row['cone'], '#F5DF8A', R.CONE_OPACITY*(1-reveal))
        add(self.fixture, V.BLUE, 1-reveal)
        add(self.body, V.GRAY, 1-reveal)
        add(transform_mesh(self.row['area'], self.row['U']), '#DAB652', 1-reveal, True, 1e-5)
        add(self.illegal_mesh, '#F1B7C9', reveal*.70, True, 1e-5)
        add(self.legal_mesh, '#A9DECC', reveal*.60, True, 1e-5)
        add(self.hull_mesh, PURPLE, .12*hull_alpha, True, 2e-5)
        if hull_alpha > 0:
            parts.append(dict(V.points_piece(self.cloud[::96], PURPLE, camera, self.size, .75),
                opacity=.45*hull_alpha, bias=2.*camera[2]))
            line = line_piece(ring_segments(self.hull, R.smooth((t-1.)/2.)), PURPLE,
                camera, self.size, 2.8, hull_alpha)
            if line is not None:
                parts.append(line)
        for point in self.anchors:
            if envelope > 0:
                parts.append(dict(V.points_piece(np.array([[*point, 0.]]), GREEN,
                    camera, self.size, 4.2), opacity=envelope, bias=2.*camera[2]))
        if envelope > 0:
            # No filled foot polygon: enclosure is a virtual convex hull.
            line = line_piece(ring_segments(self.attainable, envelope, dashed=True), GREEN,
                camera, self.size, 2.2, envelope)
            if line is not None:
                parts.append(line)
        return self.base.renderer.render([V.CPU.placed(piece) for piece in parts], camera, self.size)

    def verify_inputs(self):
        for path, checksum in self.protected.items():
            assert R.digest(ROOT/path) == checksum, 'Saved source changed: '+path

    def record(self, seconds, fps):
        self.verify_inputs()
        return dict(video=NAME+'.mp4', pose='pose_20', seconds=seconds, fps=fps, size=[self.size]*2,
            continues='05_pose2_to_pose20_oracle.mp4: final installed pose20 scene',
            source_experiment='base_oracle_v3', demand_sample_count=len(self.cloud),
            demand_hull_xy_m=self.hull.tolist(), graphical_legal_anchor_xy_m=self.anchors.tolist(),
            graphical_attainable_hull_xy_m=self.attainable.tolist(),
            displayed_enclosure_passed=self.enclosed, displayed_enclosure_gap_m=self.gap,
            saved_source_enclosure_passed=self.saved_row['passed'],
            saved_source_connected_passed=self.saved_oracle['connected_passed'],
            green='legal ground area within the display crop; exact normal-ray, complete exit and all final floor exclusions',
            purple='convex hull of ALL 32768 original ground pressure demands; sparse dots are display samples only',
            pink='forbidden ground footprint, including both task exclusions pulled into the pose20 material placement',
            green_dots='possible ground contact points in the SAVED common connected-space witness, not constructed feet',
            green_dashes='virtual convex hull of those possible contact points; not a material rim',
            own_object_ground_points_used=False, virtual_enclosure_interior_may_cross_forbidden_holes=True,
            display_window_xy_m=list(self.window.bounds), display_window_is_physical_boundary=False,
            floor_forbidden_area_m2=self.floor_forbidden.area,
            work_forbidden_area_m2=self.work_forbidden.area,
            exit_forbidden_area_m2=self.exit_forbidden.area,
            legal_ground_area_m2=self.legal.area,
            hull_overlap_with_forbidden_area_m2=self.required.intersection(self.illegal).area,
            full_original_demand_hull_used=True, saved_witness_plane_tolerance_m=1e-8,
            display_work_access_mode='normal', current_task_world_cone_only=True,
            cross_task_ground_footprints_all_included=True, text=False,
            camera_transition='original video05 isometric view to a close ground-plane top view',
            ground_hull_graphical_strokes_drawn_over_scene=True,
            layout_search_run=False, force_solver_run=False, base_oracle_query_run=False,
            common_connection_witness_rebuilt=False, query_origin='saved_final_experiment',
            display_only=True, fixture_modified=False, base_constructed=False,
            provenance=dict(inputs=self.protected, code={str(Path(__file__).relative_to(ROOT)): R.digest(__file__)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--seconds', type=float, default=14.)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--size', type=int, default=900)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    movie = Movie(args.size)
    print(json.dumps(dict(enclosed=movie.enclosed, gap_m=movie.gap, anchors=len(movie.anchors),
        legal_area_m2=movie.legal.area, forbidden_hull_overlap_m2=movie.required.intersection(movie.illegal).area)), flush=True)
    if args.preview:
        for t in (0., 2., 5., 7., 10., 13.):
            movie.frame(t).save(args.output_dir/f'06_preview_{int(t):02d}.png')
    else:
        destination = args.output_dir/(NAME+'.mp4')
        temporary = destination.with_name(NAME+'.rendering.mp4')
        encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{args.size}x{args.size}', '-r', str(args.fps), '-i', 'pipe:0', '-an',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
        start = time.monotonic()
        try:
            count = round(args.seconds*args.fps)
            for k in range(count):
                encoder.stdin.write(movie.frame(k/args.fps).convert('RGB').tobytes())
                if k % args.fps == 0:
                    print('FRAME', k, '/', count, round(time.monotonic()-start, 1), 's', flush=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Video encoding failed')
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(temporary), '-f', 'null', '-'], check=True)
            movie.verify_inputs()
            temporary.replace(destination)
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise
    (args.output_dir/(NAME+'.json')).write_text(json.dumps(movie.record(args.seconds, args.fps), indent=2)+'\n')
    print('DONE', str(args.output_dir/(NAME+'.mp4')), flush=True)


if __name__ == '__main__':
    main()
