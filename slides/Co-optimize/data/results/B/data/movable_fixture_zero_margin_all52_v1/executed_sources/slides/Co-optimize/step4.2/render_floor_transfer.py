"""Pose2 -> pose20: carry the first floor constraint into the second state.

Render saved meshes/constraints only. The moving pink sheet is a material-frame
constraint, not a physical floor transported by the robot. Transfer frames are
illustrative; legality is required only in the two installed working states.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile

COOPT = Path(__file__).resolve().parents[1]
ROOT = COOPT.parents[1]
sys.path[:0] = [str(COOPT/'helper_func'), str(COOPT/'vis_func')]
import _bootstrap
import artifact_store as A
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from floor_oracle_vis import floor_sheet
from pose_oracle import transform_points
from whole_search.common import transform_mesh, unpack_solid
from co_common import S
import step51_render as V


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(value):
    value = float(np.clip(value, 0., 1.))
    return value*value*(3.-2.*value)


def outline(sheet, width=.0007):
    """Coplanar graphical border; the floor has zero thickness."""
    v = sheet.vertices
    center = v.mean(0)
    vertices, faces = [], []
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        inward = center-(a+b)/2
        inward *= width/np.linalg.norm(inward)
        k = len(vertices)
        vertices.extend([a, b, b+inward, a+inward])
        faces.extend([[k, k+1, k+2], [k, k+2, k+3]])
    return trimesh.Trimesh(vertices, faces, process=False)


class Movie:
    def __init__(self, size):
        self.folder = A.VISUALS/'B/pose2+20/step4/step4.2/oracle'
        self.destination = COOPT/'seperate_base_support_proof/videos/05_pose2_to_pose20_oracle.mp4'
        numeric = A.result_root('B')/'pose2+20'
        self.fixture_path = numeric/'step4/step4.2/base_oracle_v3/support.obj'
        self.legal_space_path = self.fixture_path.with_name('oracle_connected_witness.obj')
        body_path = ROOT/'objects/B/mesh.stl'
        self.inputs = [body_path, self.fixture_path, self.legal_space_path]
        self.body = trimesh.load(body_path, force='mesh', process=False)
        nominal = trimesh.load(self.fixture_path, force='mesh', process=False)
        legal_space = trimesh.load(self.legal_space_path, force='mesh', process=False)
        # Reuse the same retained nominal material as Step5, before adding any
        # base. The Step4 export includes presentation-shell material below
        # installed floors; the saved Oracle component already excludes it.
        nominal_solid = S.solid(nominal)
        legal_nominal = nominal_solid ^ S.solid(legal_space)
        self.fixture = unpack_solid(legal_nominal)
        if not len(self.fixture.faces):
            raise ValueError('Saved nominal support has no material in the legal Oracle component')
        self.removed_nominal_cm3 = float((nominal_solid-legal_nominal).volume())*S.SCALE**3*1e6
        self.rows = []
        for pose in ('pose_2', 'pose_20'):
            npz = self.folder/'work_cones'/f'{pose}.npz'
            obj = npz.with_suffix('.obj')
            self.inputs.extend([npz, obj])
            with np.load(npz) as saved:
                T = saved['T_world_fixture'].copy()
                Q = saved['Q_fixture_object'].copy()
                U = saved['T_world_object'].copy()
                ids = saved['work_face_ids'].copy()
            area = trimesh.Trimesh(self.body.vertices.copy(), self.body.faces[ids].copy(), process=False)
            triangles = area.triangles+.00005*area.face_normals[:, None, :]
            highlight = trimesh.Trimesh(triangles.reshape(-1, 3),
                np.arange(triangles.size//3).reshape(-1, 3), process=False)
            self.rows.append(dict(pose=pose, T=T, Q=Q, U=U,
                area=highlight, cone=trimesh.load(obj, force='mesh', process=False)))
        before, after = self.rows
        # This saved pair admits a genuine rigid object-and-fixture transfer.
        if not np.allclose(before['Q'], after['Q'], atol=1e-12, rtol=0):
            raise ValueError('This animation requires the same object seat in both fixture states')
        self.hashes = {str(p.relative_to(ROOT)): digest(p) for p in self.inputs}
        self.size = size
        self.renderer = V.Renderer()
        self.local_center = self.body.bounds.mean(0)
        self.centers = np.array([transform_points(self.local_center, r['U']) for r in self.rows])
        rotations = Rotation.from_matrix(np.array([r['T'][:3, :3] for r in self.rows]))
        self.rotation = Slerp([0., 1.], rotations)
        self.lift = .09
        installed = np.vstack([transform_points(mesh.vertices, r['T'])
            for r in self.rows for mesh in (self.body, self.fixture)])
        self.sheet_half_lengths = [.145, .115]
        floor_center = self.centers[0].copy()
        self.first_floor = floor_sheet([0., 0., 1., 0.], floor_center, self.sheet_half_lengths)
        self.first_outline = outline(self.first_floor)
        self.last_floor = floor_sheet([0., 0., 1., 0.], self.centers[1], self.sheet_half_lengths)
        self.last_outline = outline(self.last_floor)
        self.neutral_floor = floor_sheet([0., 0., 1., 0.], installed.mean(0), [.21, .175])
        # Frame the complete trajectory once so the camera never changes scale.
        bounds = [self.neutral_floor.vertices]
        for t in np.linspace(0., 1., 81):
            T, _, _, _ = self.state(t)
            bounds.extend([transform_points(mesh.vertices, T) for mesh in (self.body, self.fixture)])
            bounds.append(transform_points(self.first_floor.vertices, T@np.linalg.inv(before['T'])))
        bounds.extend([r['cone'].vertices for r in self.rows])
        points = np.vstack(bounds)
        self.focus = (points.min(0)+points.max(0))/2
        self.span = float(2.*np.linalg.norm(points-self.focus, axis=1).max()*1.07)
        self.verify_geometry()

    def state(self, fraction):
        """First orbit, lift, rotate while airborne, lower, second orbit."""
        before, after = self.rows
        if fraction <= .25:
            return before['T'].copy(), 0, 1., -np.pi/4+2*np.pi*fraction/.25
        if fraction < .35:
            T = before['T'].copy()
            T[2, 3] += self.lift*smooth((fraction-.25)/.10)
            return T, 0, 1.-smooth((fraction-.25)/.10), -np.pi/4
        if fraction < .50:
            u = smooth((fraction-.35)/.15)
            T = np.eye(4)
            T[:3, :3] = self.rotation(u).as_matrix()
            center = (1.-u)*self.centers[0]+u*self.centers[1]
            center[2] += self.lift
            T[:3, 3] = center-T[:3, :3]@self.local_center
            # Clearance is a display trajectory choice, not an Oracle test.
            lowest = min(transform_points(m.vertices, T)[:, 2].min()
                         for m in (self.body, self.fixture))
            T[2, 3] += max(0., .035-lowest)
            return T, 1, 0., -np.pi/4
        if fraction < .60:
            u = smooth((fraction-.50)/.10)
            T = after['T'].copy()
            T[2, 3] += self.lift*(1.-u)
            return T, 1, u, -np.pi/4
        return after['T'].copy(), 1, 1., -np.pi/4+2*np.pi*(fraction-.60)/.40

    def verify_geometry(self):
        before, after = self.rows
        carry = after['T']@np.linalg.inv(before['T'])
        final_sheet = transform_points(self.first_floor.vertices, carry)
        # In pose20, the carried sheet has exactly pose2's pulled-back floor.
        plane = (before['T']@np.linalg.inv(after['T']))[2]
        assert np.max(np.abs(final_sheet@plane[:3]+plane[3])) < 1e-12
        for fraction, row in [(0., before), (1., after)]:
            T, _, _, _ = self.state(fraction)
            assert np.allclose(T, row['T'], atol=1e-12, rtol=0)
            assert np.allclose(T@row['Q'], row['U'], atol=1e-12, rtol=0)
            assert transform_points(self.body.vertices, row['U'])[:, 2].min() >= -1e-8
            assert transform_points(self.fixture.vertices, T)[:, 2].min() >= -1e-8
        # Clipping is a display derivation from the saved legal component.
        # It changes neither the accepted layout nor stored support/force data.

    def frame(self, fraction):
        T, task, cone_alpha, angle = self.state(fraction)
        row = self.rows[task]
        carry = T@np.linalg.inv(self.rows[0]['T'])
        U = T@row['Q']
        # No first-task cone is carried into the second task: only its floor.
        cone_transform = T@np.linalg.inv(row['T'])
        parts = []

        def add(mesh, color, opacity, unlit=False, bias=0.):
            if opacity <= 0.:
                return
            p = V.piece(mesh, color, opacity)
            p['unlit'] = unlit
            p['bias'] = bias
            parts.append(V.CPU.placed(p))

        add(self.neutral_floor, '#ECEFF1', .13, True, -3e-5)
        add(transform_mesh(self.first_floor, carry), '#EBAAC4', .32, True, 1e-5)
        add(transform_mesh(self.first_outline, carry), '#CD6990', .72, True, 2e-5)
        # The second current-world floor appears as the object is installed.
        new_floor_alpha = smooth((fraction-.50)/.10)
        add(self.last_floor, '#F1B9CC', .32*new_floor_alpha, True, 1e-5)
        add(self.last_outline, '#D57C9B', .72*new_floor_alpha, True, 2e-5)
        add(transform_mesh(row['cone'], cone_transform), '#F5DF8A', .24*cone_alpha)
        # An opaque solid avoids the pale doubled silhouette of translucent
        # front/back surfaces, especially around the bottom edge.
        add(transform_mesh(self.fixture, T), V.BLUE, 1.)
        add(transform_mesh(self.body, U), V.GRAY, 1.)
        add(transform_mesh(row['area'], U), '#DAB652', cone_alpha, True, 1e-5)
        direction = np.array([np.cos(angle), np.sin(angle), 1/np.sqrt(2)])
        direction /= np.linalg.norm(direction)
        right = np.array([-np.sin(angle), np.cos(angle), 0.])
        camera = (self.focus, np.array([right, np.cross(direction, right), direction]), self.span)
        return self.renderer.render(parts, camera, self.size)

    def verify_inputs(self):
        for path, sha in self.hashes.items():
            assert digest(ROOT/path) == sha, 'Saved numerical input changed: '+path

    def record(self, seconds, fps):
        before, after = self.rows
        return dict(before_pose=before['pose'], after_pose=after['pose'],
            video=str(self.destination.relative_to(ROOT)), seconds=seconds, fps=fps,
            final_floor_count=2, final_work_cone_count=1,
            carried_pose2_floor_in_pose20_world=(before['T']@np.linalg.inv(after['T']))[2].tolist(),
            pose20_current_floor_world=[0., 0., 1., 0.],
            carried_floor_motion='T(t) inverse(T_pose2) applied to the pose2 world floor',
            floor_thickness_m=0., pink_sheets_are_boundary_constraints=True,
            physical_ground_always_world_z_zero=True, work_cone_is_current_task_only=True,
            text=False, transfer_motion_shown=True, intermediate_transfer_states_constrained=False,
            display_only=True, final_fixture_modified=False, hypothetical_material_marker_shown=False,
            blue_fixture_stage='step4.2', base_shown=False,
            blue_fixture_opacity=1., blue_outline_drawn=False,
            blue_fixture_geometry='saved Step4 nominal support intersected with saved legal Oracle component; no base added',
            saved_legal_component=str(self.legal_space_path.relative_to(ROOT)),
            removed_illegal_nominal_material_cm3=self.removed_nominal_cm3,
            final_floor_display_tolerance_m=1e-8,
            installed_object_min_z_m={r['pose']: float(transform_points(self.body.vertices, r['U'])[:, 2].min())
                                     for r in self.rows},
            installed_support_min_z_m={r['pose']: float(transform_points(self.fixture.vertices, r['T'])[:, 2].min())
                                      for r in self.rows},
            force_solver_run=False, base_oracle_query_run=False, query_origin='saved_final_experiment',
            fixture_source=str(self.fixture_path.relative_to(ROOT)),
            provenance=dict(inputs=self.hashes, code={str(Path(__file__).relative_to(ROOT)): digest(__file__)}))


def encode(movie, path, seconds, fps):
    count = round(seconds*fps)
    encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{movie.size}x{movie.size}', '-r', str(fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    try:
        for k in range(count):
            encoder.stdin.write(movie.frame(k/(count-1)).convert('RGB').tobytes())
            if k % fps == 0:
                print('FRAME', k, '/', count, flush=True)
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoder failed')


def publish(movie, temporary, seconds, fps):
    # Install the complete decoded movie before deleting the superseded videos.
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(temporary), '-f', 'null', '-'],
                            capture_output=True, text=True, check=True)
    if result.stderr:
        raise RuntimeError(result.stderr)
    movie.verify_inputs()
    movie.destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temporary), str(movie.destination))
    record = movie.record(seconds, fps)
    relative = os.path.relpath(movie.destination, movie.folder)
    report_path = movie.folder/'report.json'
    report = json.loads(report_path.read_text())
    report['videos'] = [dict(pose='pose_2 -> pose_20', video=relative,
        frames=round(seconds*fps), fps=fps, seconds=seconds, text=False,
        transfer_motion_shown=True, final_floor_count=2, final_work_cone_count=1)]
    report['floor_example'] = record
    report['video_location'] = str(movie.destination.relative_to(ROOT))
    report['blue_fixture_stage'] = 'step4.2'
    report['base_shown'] = False
    report['original_layout_and_force_results_changed'] = False
    for name, value in [('report.json', report), ('floor_example.json', record)]:
        (movie.folder/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    (movie.folder/'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<style>body{font:16px system-ui;max-width:1000px;margin:28px auto}video{width:100%}p{line-height:1.6}</style>'
        '<h1>pose2 + pose20 · Oracle</h1>'
        '<p>Grey: object. Blue: support within the saved legal space, before adding a base. Yellow: the current task’s working cone. '
        'Pink: zero-thickness floor constraints. The pose2 constraint travels with the fixture; '
        'pose20 shows this carried constraint together with its own horizontal floor.</p>'
        '<p>The robot transfer illustrates a pose change; only installed working states constrain material.</p>'
        f'<video controls loop muted playsinline src="{html.escape(relative)}"></video>\n')
    removed = []
    # Explicit user scope: this pair's oracle video files only.
    for path in movie.folder.glob('*.mp4'):
        removed.append(path.name)
        path.unlink()
    print(json.dumps(dict(video=str(movie.destination), removed_oracle_videos=removed,
                         saved_inputs_unchanged=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--size', type=int, default=900)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps < 1 or args.size < 100 or args.size % 2:
        raise ValueError('Invalid duration, frame rate or dimensions')
    folder = Path(tempfile.mkdtemp(prefix='cadgrasp-carried-floor-'))
    movie = Movie(args.size)
    print('TEMP', folder, flush=True)
    if args.preview:
        for fraction in (0., .18, .31, .425, .56, .63, .78, .94, 1.):
            movie.frame(fraction).save(folder/f'preview_{round(fraction*1000):03d}.png')
        print(json.dumps(movie.record(args.seconds, args.fps)), flush=True)
        movie.verify_inputs()
    else:
        temporary = folder/movie.destination.name
        encode(movie, temporary, args.seconds, args.fps)
        publish(movie, temporary, args.seconds, args.fps)
        folder.rmdir()


if __name__ == '__main__':
    main()
