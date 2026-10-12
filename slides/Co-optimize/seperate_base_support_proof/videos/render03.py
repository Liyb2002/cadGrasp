"""Video 04: one camera orbit, with the Oracle's normal-swept working area.

The historical pose1+2+4+6 layout, support and demand arrays remain immutable.
Four illustrative open bases stay attached. Only the displayed task's own
outward normal-ray volume is shown; its actual overlap with the bases turns red.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
COOPT = PROOF.parent
ROOT = COOPT.parents[1]
sys.path[:0] = [str(HERE), str(COOPT/'helper_func'), str(COOPT/'step4.2'), str(COOPT/'vis_func')]
import _bootstrap
import render02 as R
import render_pose_oracle as O
from co_common import S, union, material_volume
from pose_oracle import PoseConstraint, transform_planes
from normal_work_access import NormalWorkAccess
from whole_search.common import transform_mesh, unpack_solid
import step51_render as V

FPS, FRAMES, SIZE = 24, 192, 900
OUTPUT_NAME = '04_work_cones.mp4'
CONE_OPACITY = .60


class Movie:
    def __init__(self):
        directory = PROOF/'output/B/pose1+2+4+6'
        report_path = directory/'step5/step5.1/data/report.json'
        report = json.loads(report_path.read_text())
        selected = ROOT/report['selected_step4_source']
        mesh_path = ROOT/'objects/B/mesh.stl'
        support_path = selected/'support.obj'
        layout_path = selected/'layout.npz'
        paths = [report_path, mesh_path, support_path, layout_path, selected/'data/report.json',
                 directory/'data/selection.json', HERE/'01_hulls_and_support.mp4',
                 HERE/'02_ground_collision.mp4', HERE/OUTPUT_NAME]
        self.mesh = trimesh.load(mesh_path, force='mesh', process=False)
        support = trimesh.load(support_path, force='mesh', process=False)
        with np.load(layout_path, allow_pickle=False) as data:
            placements = data['placements'].copy()
            native = data['native_world'].copy()
            hosts = data['hosts'].copy()
        self.rows = []
        for index, row in enumerate(report['per_pose']):
            arrays_path = directory/'step5/step5.1'/row['arrays']
            needs_path = ROOT/'objects/B/poses'/row['pose']/'needs.json'
            paths.extend([arrays_path, needs_path])
            needs = json.loads(needs_path.read_text())
            with np.load(arrays_path, allow_pickle=False) as data:
                cloud = data['floor_demands_world_m'].copy()
                self.rows.append(dict(pose=row['pose'], color=row['color'], cloud=cloud,
                    hull=cloud[data['hull_indices']].copy(),
                    fixture_transform=data['T_world_fixture'].copy(),
                    object_transform=data['T_world_object'].copy(), needs=needs))
            saved = self.rows[-1]
            assert len(cloud) == 32768
            np.testing.assert_allclose(saved['fixture_transform'], native[hosts[index]], atol=1e-12)
            np.testing.assert_allclose(saved['object_transform'], native[hosts[index]]@placements[index], atol=1e-12)
        self.protected = {path: R.digest(path) for path in paths}
        # Use the same open-rim drawing as video 2, pulled into common material
        # coordinates through each rim's own saved installed fixture state.
        bases = [S.solid(R.unpack(R.Model.rim(None, row, .004))) for row in self.rows]
        self.bases = union(bases)
        base_mesh = unpack_solid(self.bases)
        links = []
        for base in bases:
            candidates = unpack_solid(base).triangles_center
            close, distance, _ = trimesh.proximity.closest_point_naive(support, candidates)
            i = int(np.argmin(distance))
            if distance[i] > 1e-9:
                a, b = close[i], candidates[i]
                direction = (b-a)/np.linalg.norm(b-a)
                links.append(R.rod(a-direction*.002, b+direction*.002))
        self.core = union([S.solid(support), *[S.solid(link) for link in links]])
        core_mesh = unpack_solid(self.core)
        model = SimpleNamespace(mesh=self.mesh, extent=float(self.mesh.extents.max()))
        tasks = []
        for index, row in enumerate(self.rows):
            needs = row['needs']
            print('Oracle normal working volume:', row['pose'], flush=True)
            work = NormalWorkAccess(self.mesh, needs['geometry']['work_face_ids'])
            local = tuple(p[:n].copy() for p, n in zip(work.planes, work.counts))
            T, U, Q = row['fixture_transform'], row['object_transform'], placements[index]
            tasks.append(PoseConstraint(index, row['pose'], T, Q, U, local,
                tuple(transform_planes(p, np.linalg.inv(Q)) for p in local),
                tuple(transform_planes(p, np.linalg.inv(U)) for p in local),
                work.face_ids.copy(), work.half_angle_deg, work))
        constraints = SimpleNamespace(poses=tasks)
        choices = []
        for task, row in zip(tasks, self.rows):
            # This is the actual Oracle scene builder: normal-ray volume, spherical
            # display crop, and original work triangles highlighted in gold.
            scene = O.scene(model, constraints, task)
            world_bases = transform_mesh(base_mesh, task.T_world_fixture)
            collision = S.solid(world_bases)^S.solid(scene['yellow'])
            overlap = material_volume(collision)*1e6
            print('Base/cone display overlap:', task.pose, overlap, 'cm3', flush=True)
            choices.append((overlap, task, row, scene, world_bases, collision))
        # Keep the same pose/view when changing the working-volume definition.
        overlap, self.task, self.row, self.scene, world_bases, collision = next(
            item for item in choices if item[1].pose=='pose_4')
        self.overlap_cm3 = overlap
        self.core_world = transform_mesh(core_mesh, self.task.T_world_fixture)
        self.bases_world = world_bases
        self.collision_mesh = unpack_solid(collision)
        base_solid = S.solid(world_bases)
        remaining_solid = base_solid-collision
        self.remaining_mesh = unpack_solid(remaining_solid)
        # Check the actual Boolean pieces directly. A mesh->solid roundtrip
        # can repair coincident normal-prism cuts and obscure this identity.
        assert material_volume(remaining_solid+collision-base_solid) < 1e-13
        assert material_volume(base_solid-(remaining_solid+collision)) < 1e-13
        self.renderer = R.Renderer()
        self.start_azimuth_deg = -45.
        points = np.vstack([mesh.vertices for mesh in
            [self.core_world, self.bases_world, self.scene['body'], self.scene['yellow']]]
            +[self.row['cloud']])
        self.focus = (points.min(axis=0)+points.max(axis=0))/2
        offsets = points-self.focus
        radius = np.linalg.norm(offsets[:, :2], axis=1)
        # Bound every azimuth at the same isometric elevation, so the complete
        # turn keeps its center and scale without clipping or camera zoom.
        extent = max(radius.max(),
            (radius/np.sqrt(3)+np.abs(offsets[:, 2])*np.sqrt(2/3)).max())
        self.span = 2*float(extent)*1.08
        reference_path = PROOF/'recovery/video04_05_camera_reference.json'
        if reference_path.exists():
            reference = json.loads(reference_path.read_text())['04']
            self.focus = np.asarray(reference['focus'],float)
            self.span = float(reference['span'])
            self.start_azimuth_deg = float(reference['start_azimuth_deg'])
            self.protected[reference_path] = R.digest(reference_path)
        self.camera = self.camera_at(-np.pi/4)
        self.prepare_pieces()

    def verify_inputs(self):
        assert all(R.digest(path) == checksum for path, checksum in self.protected.items())

    def camera_at(self, angle):
        direction = np.array([np.cos(angle), np.sin(angle), 1/np.sqrt(2)])
        direction /= np.linalg.norm(direction)
        right = np.array([-np.sin(angle), np.cos(angle), 0.])
        basis = np.array([right, np.cross(direction, right), direction])
        return self.focus, basis, self.span

    def prepare_pieces(self):
        self.core_piece = R.blue_piece(self.core_world)
        self.bases_piece = R.blue_piece(self.bases_world)
        self.remaining_piece = R.blue_piece(self.remaining_mesh)
        self.red_piece = V.piece(self.collision_mesh, '#C94D63', 1.) if len(self.collision_mesh.faces) else None
        self.body_piece = V.piece(self.scene['body'], V.GRAY, .18)
        self.cone_piece = V.piece(self.scene['yellow'], '#F5DF8A', CONE_OPACITY)
        self.area_piece = dict(V.piece(self.scene['highlight'], '#DAB652', 1.), unlit=True, bias=1e-5)
        hull = self.row['hull']
        marks = []
        for a, b in zip(hull, np.roll(hull, -1, axis=0)):
            length = np.linalg.norm(b-a)*SIZE/self.span
            count = max(1, round(length/3.8))
            marks.extend(a+(b-a)*np.arange(count)[:, None]/count)
        self.hull_marks = np.asarray(marks)

    def frame(self, t):
        progress = np.clip(t/((FRAMES-1)/FPS), 0., 1.)
        angle = np.radians(self.start_azimuth_deg)+2*np.pi*progress
        camera = self.camera_at(angle)
        opacity = CONE_OPACITY*R.smooth((t-1.4)/1.4)
        red = t >= 3.2 and self.red_piece is not None
        # Rebuild the screen-facing dots for each camera angle.
        cloud_piece = V.points_piece(self.row['cloud'][::64], self.row['color'], camera, SIZE, .85)
        hull_piece = V.points_piece(self.hull_marks, self.row['color'], camera, SIZE, .65)
        parts = [self.core_piece, self.remaining_piece if red else self.bases_piece,
                 self.body_piece, cloud_piece, hull_piece, self.area_piece]
        if opacity > 1e-6:
            parts.append(dict(self.cone_piece, opacity=opacity))
        if red:
            parts.append(self.red_piece)
        return self.renderer.render([V.CPU.placed(piece) for piece in parts], camera, (SIZE, SIZE))

    def record(self):
        self.verify_inputs()
        return dict(size=[SIZE, SIZE], fps=FPS, frames=FRAMES, duration_seconds=FRAMES/FPS,
            source_set='pose1+2+4+6', shown_pose=self.task.pose, panel_count=1, camera_fixed=False,
            camera_orbit_degrees=360., camera_focus_and_scale_fixed=True,
            camera_start_azimuth_deg=self.start_azimuth_deg,
            camera_focus=self.focus.tolist(), camera_span_m=self.span,
            cone_builder=str(Path(O.__file__).relative_to(ROOT)),
            work_cone='only the displayed pose: original working triangles swept along their own outward normals',
            work_access_mode='normal', work_access_definition=self.task.work_access.definition(),
            cone_color='#F5DF8A', cone_opacity=CONE_OPACITY, work_surface_color='#DAB652',
            cone_half_angle_deg=self.task.half_angle_deg,
            display_spherical_crop_only=True, other_task_world_cones_shown=False, exit_sweeps_shown=False,
            red_geometry='actual illustrative-base intersection with the displayed Oracle cone',
            base_cone_overlap_cm3=self.overlap_cm3, illustrative_base_count=4,
            fixture_geometry_fixed=True, saved_layout_and_support_changed=False, force_solver_run=False,
            saved_numeric_inputs_unchanged=True, base_oracle_query_run=False,
            new_work_volume_is_display_geometry_only=True)


def encode(movie, path):
    encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{SIZE}x{SIZE}', '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    started = time.monotonic()
    try:
        for index in range(FRAMES):
            encoder.stdin.write(movie.frame(index/FPS).tobytes())
            if index % 48 == 0:
                print('03:', index, '/', FRAMES, f'{time.monotonic()-started:.1f} s', flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('Video encoding failed')
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    movie.verify_inputs()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--azimuth-deg', type=float, default=-45.)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    movie = Movie()
    movie.start_azimuth_deg = args.azimuth_deg
    movie.camera = movie.camera_at(np.radians(args.azimuth_deg))
    movie.prepare_pieces()
    record = movie.record()
    record['camera_azimuth_deg'] = args.azimuth_deg
    print(json.dumps(record), flush=True)
    if args.preview:
        for t, name in [(0., 'start'), (2.2, 'cone'), (4., 'collision')]:
            movie.frame(t).save(args.output_dir/(name+'.png'))
    else:
        encode(movie, args.output_dir/OUTPUT_NAME)
    movie.verify_inputs()
    (args.output_dir/'render03.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
