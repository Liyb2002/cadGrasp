"""A short, natural deletion of real disconnected Step4 fragments.

Reconstruct only the saved layout's finite material domain. Show the actual
discarded components fading in place and retain the saved certified core.
The original force/torque certificates are read and audited, never re-solved.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
COOPT = HERE.parent.parent
ROOT = COOPT.parents[1]
sys.path[:0] = [str(COOPT/'helper_func'), str(COOPT/'step4.2'),
    str(COOPT/'step5.2'), str(COOPT/'step5.3'), str(COOPT/'vis_func')]
import _bootstrap
from co_common import S, material_volume
from whole_search.model import Layout, Model
from whole_search.common import transform_mesh, unpack_solid
from whole_step4_render import NominalBuilder
from prune_support import material_components
import step51_render as V

NAME = '07_remove_disconnected_fragments'
CASE = 'pose1+12+29'
EXPERIMENT = 'connected_core_all52_20261012_05_remaining'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(value):
    value = np.clip(value, 0., 1.)
    return value*value*(3.-2.*value)


class Movie:
    def __init__(self, size=900, seconds=12.):
        self.size = size
        self.seconds = seconds
        self.fade_start = .5*seconds
        self.fade_end = .625*seconds
        case = COOPT/'data/results/B'/CASE
        source = case/'step4/step4.2'/EXPERIMENT
        report_path = source/'data/report.json'
        self.report = json.loads(report_path.read_text())
        report = self.report
        poses = report['poses']
        assert report['passed'] and report['force_passed'] and report['force_material_geometry_certified']
        assert report['force_acceptance_geometry'] == 'support.obj'
        self.component_record = report['connected_contact_core']
        assert self.component_record['connected'] and self.component_record['component_count'] == 1
        assert not self.component_record['connection_material_added']
        support_path, layout_path = source/'support.obj', source/'layout.npz'
        assert digest(support_path) == report['artifacts']['../support.obj']
        paths = [report_path, source/'report.json', support_path, layout_path,
            case/'step3/step3.1/data/report.json', case/'step3/step3.1/wrapped_support.obj']
        self.force_counts = {}
        for pose in poses:
            path = source/(pose+'_force.npz')
            paths.append(path)
            with np.load(path) as data:
                mask = data['mask']
                assert mask.shape == (32768,) and mask.all()
                assert data['supply_7d'].shape[1] == 7
                self.force_counts[pose] = int(mask.sum())
        # Check the saved material-contact certificate and its exact inputs.
        for path, checksum in report['provenance']['inputs'].items():
            resolved = ROOT/path
            assert digest(resolved) == checksum, 'Certified input changed: '+path
            paths.append(resolved)
        paths.extend(sorted(HERE.glob('0[12456]_*.mp4')))
        self.protected = {str(path.relative_to(ROOT)): digest(path) for path in paths}
        with np.load(layout_path) as data:
            layout = Layout.from_arrays(data)
            original_native = data['native_world_original'].copy()
        # Model.__init__ loads geometry/loads but does not evaluate force cones.
        model = Model(poses, 'B', initialization_report=case/'step3/step3.1/data/report.json',
            allow_z_translation=True)
        np.testing.assert_allclose(model.native, original_native, atol=1e-12, rtol=0)
        model.support_floor_clearance_m = 0.
        original = NominalBuilder(model).solid(layout)
        components = material_components(original)
        self.measured_volumes = [material_volume(part)*1e6 for part in components]
        assert len(components) == self.component_record['input_component_count']
        np.testing.assert_allclose(self.measured_volumes,
            self.component_record['component_volumes_cm3'], atol=1e-6, rtol=1e-7)
        core = trimesh.load(support_path, force='mesh', process=False)
        core_solid = S.solid(core)
        # The movie ends at the exact accepted OBJ, not a guessed substitute.
        self.core_extra_m3 = material_volume(core_solid-components[0])
        self.core_missing_m3 = material_volume(components[0]-core_solid)
        assert max(self.core_extra_m3, self.core_missing_m3) <= 1e-10
        self.deleted_cm3 = sum(self.measured_volumes[1:])
        self.before_components = len(components)
        assert core.is_watertight
        assert len(material_components(core_solid)) == 1
        # Use the original mesh and the pose12 placement for context. All
        # materials and objects keep their original coordinates throughout.
        shown = poses.index('pose_12')
        with np.load(layout_path) as data:
            T = data['native_world'][layout.hosts[shown]].copy()
        self.shown_pose = poses[shown]
        self.core = transform_mesh(core, T)
        self.fragments = [transform_mesh(unpack_solid(part), T) for part in components[1:]]
        self.object = transform_mesh(model.mesh, T@layout.placements[shown])
        all_points = np.vstack([self.core.vertices, self.object.vertices,
            *[mesh.vertices for mesh in self.fragments]])
        self.focus = (all_points.min(0)+all_points.max(0))/2
        offsets = all_points-self.focus
        horizontal = np.linalg.norm(offsets[:, :2], axis=1)
        # Bound every azimuth at the same isometric elevation. The camera
        # retains one center and scale throughout both complete turns.
        extent = max(horizontal.max(),
            (horizontal/np.sqrt(3)+np.abs(offsets[:, 2])*np.sqrt(2/3)).max())
        self.span = 2.*float(extent)*1.18
        self.renderer = V.Renderer()
        self.camera_scores = []
        # Choose a real viewpoint where the discarded geometry is visible.
        # No fragment is enlarged, translated, highlighted, or fabricated.
        for degrees in range(-45, 315, 45):
            camera = self.camera_at(np.radians(degrees), all_points)
            before = np.asarray(self.render(1., camera), dtype=float)
            after = np.asarray(self.render(0., camera), dtype=float)
            difference = np.max(np.abs(before-after), axis=2)
            score = int(np.count_nonzero(difference > 20.))
            self.camera_scores.append(dict(azimuth_deg=degrees, changed_pixels=score))
        self.azimuth_deg = max(self.camera_scores, key=lambda row: row['changed_pixels'])['azimuth_deg']
        self.camera = self.camera_at(np.radians(self.azimuth_deg))
        assert max(row['changed_pixels'] for row in self.camera_scores) > 50
        print(json.dumps(dict(case=CASE, components_before=len(components), components_after=1,
            deleted_cm3=self.deleted_cm3, force_counts=self.force_counts,
            view=self.azimuth_deg, visible_change_pixels=max(row['changed_pixels']for row in self.camera_scores))), flush=True)

    def camera_at(self, angle, points=None):
        direction = np.array([np.cos(angle), np.sin(angle), 1/np.sqrt(2)])
        direction /= np.linalg.norm(direction)
        right = np.array([-np.sin(angle), np.cos(angle), 0.])
        basis = np.array([right, np.cross(direction, right), direction])
        return self.focus, basis, self.span

    def render(self, opacity, camera):
        pieces = [V.piece(self.core, V.BLUE), V.piece(self.object, V.GRAY, .28)]
        if opacity > 1e-6:
            pieces.extend(V.piece(fragment, V.BLUE, opacity) for fragment in self.fragments)
        return self.renderer.render([V.CPU.placed(piece)for piece in pieces], camera, self.size)

    def frame(self, seconds):
        opacity = 1.-smooth((seconds-self.fade_start)/(self.fade_end-self.fade_start))
        angle = np.radians(self.azimuth_deg)+4.*np.pi*seconds/self.seconds
        return self.render(opacity, self.camera_at(angle))

    def verify_inputs(self):
        for path, checksum in self.protected.items():
            assert digest(ROOT/path) == checksum, 'Saved input changed: '+path

    def record(self, seconds, fps):
        self.verify_inputs()
        return dict(video=NAME+'.mp4', seconds=seconds, fps=fps, size=[self.size]*2,
            source_case=CASE, source_experiment=EXPERIMENT, shown_pose=self.shown_pose,
            algorithm_stage='Step4.2 connected-contact-core selection',
            material_components_before=self.before_components, material_components_after=1,
            removed_fragment_count=len(self.fragments), component_volumes_cm3=self.measured_volumes,
            removed_material_cm3=self.deleted_cm3,
            animation='one full orbit with all fragments, in-place smooth deletion, then continued orbit of the retained core',
            fragment_fade_start_seconds=self.fade_start, fragment_fade_end_seconds=self.fade_end,
            camera_fixed=False, camera_orbit_degrees=720.,
            camera_focus_and_scale_fixed=True, camera_focus_m=self.focus.tolist(), camera_span_m=self.span,
            camera_azimuth_deg=self.azimuth_deg, viewpoint_visibility=self.camera_scores,
            text=False, overlays=False, artificial_fragments=False, geometry_moved_or_scaled=False,
            retained_core_is_exact_saved_obj=True,
            reconstructed_core_extra_m3=self.core_extra_m3, reconstructed_core_missing_m3=self.core_missing_m3,
            force_torque_verification='existing exact retained-material contact-polygon certificate and full saved masks',
            certified_original_demands_per_pose=self.force_counts,
            force_material_geometry_certified=self.report['force_material_geometry_certified'],
            downward_resultant_passed=self.component_record['downward_resultant_passed'],
            force_solver_run=False, base_oracle_query_run=False, layout_search_run=False,
            display_only=True, saved_inputs_unchanged=True,
            provenance=dict(inputs=self.protected, code={str(Path(__file__).relative_to(ROOT)): digest(__file__)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--size', type=int, default=900)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    movie = Movie(args.size, args.seconds)
    if args.preview:
        for fraction in (0., .125, .25, .375, .5, .5625, .625, .75, .875):
            movie.frame(fraction*args.seconds).save(args.output_dir/f'07_preview_{round(fraction*1000):03d}.png')
    else:
        destination = args.output_dir/(NAME+'.mp4')
        temporary = args.output_dir/(NAME+'.rendering.mp4')
        encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{args.size}x{args.size}', '-r', str(args.fps), '-i', 'pipe:0', '-an',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
        began = time.monotonic()
        try:
            count = round(args.seconds*args.fps)
            for k in range(count):
                encoder.stdin.write(movie.frame(k/args.fps).convert('RGB').tobytes())
                if k % args.fps == 0:
                    print('FRAME', k, '/', count, round(time.monotonic()-began, 1), 's', flush=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Video encoding failed')
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(temporary), '-f', 'null', '-'], check=True)
            movie.verify_inputs()
            temporary.replace(destination)
            movie.frame(0.).save(args.output_dir/(NAME+'.png'))
        except BaseException:
            encoder.kill()
            encoder.wait()
            raise
    (args.output_dir/(NAME+'.json')).write_text(json.dumps(movie.record(args.seconds, args.fps), indent=2)+'\n')
    print('DONE', args.output_dir/(NAME+'.mp4'), flush=True)


if __name__ == '__main__':
    main()
