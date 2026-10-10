"""Render text-free six-second Translation and eight-second Combined videos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

import geometry as H

G, R = H.G, H.R
CO = H.DEMO.parent
DIRECTION_SOURCE = CO/'vis_func/animate_exit_directions.py'


class DirectionWheel:
    """Reuse the Direction video's actual transparent sweep and yellow arrow.

The original wheel supplies the artists and styling. An orthographic matrix
aligns its overlay with the existing depth-buffered scene, retaining the same
solid renderer for the body and support throughout the combined video.
    """
    def __init__(self, renderer, body, support, length):
        wheel = H.module('original_direction_video', DIRECTION_SOURCE)
        self.wheel = wheel.Renderer(body, support, [np.eye(4)], ['pose_2'], length,
                                    columns=1, labels=False)
        self.wheel.fig.set_size_inches(renderer.width/100, renderer.height/100, forward=True)
        self.wheel.fig.patch.set_alpha(0.)
        ax = self.wheel.axes[0]
        ax.patch.set_alpha(0.)
        for line in list(ax.lines):
            line.remove()
        self.wheel.fig.canvas.draw()
        inverse = ax.transData.inverted()
        origin = inverse.transform([renderer.width/2, renderer.height/2])
        unit = inverse.transform([renderer.width/2+1, renderer.height/2+1])-origin
        projection = np.zeros((4, 4))
        for index, basis in enumerate([renderer.right, renderer.up]):
            projection[index, :3] = basis*renderer.scale/1000*unit[index]
            projection[index, 3] = origin[index]-(renderer.center @ basis)*renderer.scale*unit[index]
        projection[2, :3] = -renderer.camera/1000
        projection[2, 3] = renderer.center @ renderer.camera-1
        projection[3, 3] = 1
        ax.get_proj = lambda: projection
        self.renderer = renderer
        self.records = []

    def overlay(self, frame, support, body, direction):
        self.wheel.obj = body
        sweep = G.C.S.solid(G.C.S.swept_solid(body, self.wheel.length*direction, fan_in=8))
        # The source wheel draws the same fitted yellow arrow and full sweep.
        self.wheel.frame(support, np.asarray([direction]), -1, [sweep])
        self.wheel.artists[0].set_visible(False)
        self.wheel.fig.canvas.draw()
        rgba = np.asarray(self.wheel.fig.canvas.buffer_rgba()).copy()
        alpha = rgba[:, :, 3:4].astype(float)/255
        result = np.rint(frame*(1-alpha)+rgba[:, :, :3]*alpha).astype(np.uint8)
        self.records.append(dict(direction_world=np.asarray(direction).tolist(),
                                 full_sweep_length_m=self.wheel.length,
                                 overlay_pixel_count=int(np.count_nonzero(rgba[:, :, 3]))))
        return result

    def close(self):
        self.wheel.plt.close(self.wheel.fig)


class Video:
    def __init__(self, path, renderer, duration):
        self.path = path
        self.pending = path.with_name('.'+path.stem+'.rendering.mp4')
        self.renderer = renderer
        self.fps, self.duration, self.frames = 24, duration, 0
        self.timeline = []
        path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{renderer.width}x{renderer.height}', '-r', str(self.fps),
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.pending)], stdin=subprocess.PIPE)

    def emit(self, materials, body, direction, repeat=1, distance=0., visible=True,
             phase='', state='', direction_wheel=None, sweep_support=None, camera_azimuth=None):
        self.renderer.meshes['guest'] = body
        self.renderer.direction = np.asarray(direction)
        frame = self.renderer.frame(materials, distance, '', guest_visible=visible)
        if direction_wheel is not None:
            frame = direction_wheel.overlay(frame, sweep_support, body, np.asarray(direction))
        for _ in range(repeat):
            self.encoder.stdin.write(frame.tobytes())
        self.timeline.append(dict(phase=phase, state=state, first_frame=self.frames, repetitions=repeat,
                                  guest_exit_offset_m=float(distance), direction_world=list(direction),
                                  guest_visible=visible, direction_overlay=direction_wheel is not None,
                                  camera_azimuth_degrees=camera_azimuth))
        self.frames += repeat

    def finish(self, geometry, **extra):
        assert self.frames == self.fps*self.duration, self.frames
        self.encoder.stdin.close()
        if self.encoder.wait():
            raise RuntimeError('ffmpeg failed')
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries', 'stream=duration,nb_read_frames', '-of', 'json',
            str(self.pending)]))['streams'][0]
        assert int(probe['nb_read_frames']) == self.frames
        assert abs(float(probe['duration'])-self.duration) < 1e-6
        self.pending.replace(self.path)
        record = dict(complete=True, demo_only=True, force_acceptance_run=False,
                      full_fixture_accepted=False, fps=self.fps, frame_count=self.frames,
                      duration_seconds=self.duration, frame_size_px=[self.renderer.width, self.renderer.height],
                      panel_count=1, on_screen_text=False, on_screen_legend=False, fixture_moves=False,
                      visible_object_pose='pose_2', timeline=self.timeline,
                      geometry_record=str(H.OUT/'data/geometry.json'),
                      body_width_m=geometry['body_width_m'],
                      translation_distance_m=geometry['translation_distance_m'],
                      fraction_of_body_width=1/3,
                      direction_step_optimized=False, translation_step_optimized=False,
                      relocation_sweep_carved=False,
                      source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [Path(__file__), Path(H.__file__), Path(G.__file__), Path(R.__file__), DIRECTION_SOURCE]},
                      artifact_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(), **extra)
        (self.path.parent/'animation.json').write_text(json.dumps(record, indent=2)+'\n')
        print('COMPLETE', self.path, f'{self.duration}s / {self.frames} frames', flush=True)

    def abort(self):
        self.encoder.kill()
        self.encoder.wait()
        self.pending.unlink(missing_ok=True)


def renderer(source_meshes, source_report, states, combined=False):
    framing = source_meshes.copy()
    names = ['translation_48']+(['direction_18'] if combined else [])
    meshes = [states[name]['final'] for name in names]
    meshes += [states['translation_48']['guest']]
    if combined:
        direction = np.asarray(source_report['guest_direction_world'])
        meshes.append(H.moved(states['direction_18']['guest'], .07*direction))
    framing['final'] = G.C.trimesh.util.concatenate(meshes)
    return R.Renderer(framing, source_report, width=720, height=720, annotations=False,
                      guest_only=True, include_approach=combined, approach_distance=.13)


def layers(state, red=True):
    result = [(state['kept'], 'support'), (state['added'], 'added')]
    if red:
        result.append((state['fresh'], 'removed'))
    return result


def camera(renderer, elevation, azimuth):
    e, a = np.radians([elevation, azimuth])
    renderer.camera = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    renderer.right = np.array([-np.sin(a), np.cos(a), 0.])
    renderer.up = np.cross(renderer.camera, renderer.right)


def remove_object_and_orbit(video, state, direction, inspection_frames):
    """Withdraw fully, then orbit the camera around the unchanged blue solid."""
    support, guest = state['final'], state['guest']
    support_solid, body = G.C.S.solid(support), G.C.S.solid(guest)
    distance = max(.13, float((support.vertices @ direction).max()
                              -(guest.vertices @ direction).min())+.01)
    # Early positions retrace the already checked first 70mm exit segment.
    early = np.linspace(0, .07, 14)[[1, 3, 5, 7, 10, 13]]
    distances = np.r_[early, np.linspace(.07, distance, 7)[1:]]
    checks = []
    for requested in distances:
        # Resolve numerical boundary ambiguities by selecting a nearby actual
        # playback position, never by changing the support or its tolerance.
        for perturbation in [0., 1e-8, -1e-8, 1e-7, -1e-7, 1e-6, -1e-6]:
            position = float(requested+perturbation)
            current = body.translate((position*direction/G.C.S.SCALE).tolist())
            overlap = G.C.material_volume(support_solid ^ current)
            if overlap <= G.TOL:
                break
        else:
            raise RuntimeError(f'Withdrawal intersection unresolved at {requested}: {overlap}')
        checks.append(dict(offset_m=position, body_overlap_m3=overlap))
        video.emit([(support, 'support')], guest, direction, distance=position, phase='object_exit')
    gap = float((guest.vertices @ direction).min()+checks[-1]['offset_m']
                -(support.vertices @ direction).max())
    assert gap > 0, 'Object must fully detach before it is hidden'
    renderer = video.renderer
    original_center, original_scale = renderer.center.copy(), renderer.scale
    original_elevation = float(np.degrees(np.arcsin(renderer.camera[2])))
    original_azimuth = float(np.degrees(np.arctan2(renderer.camera[1], renderer.camera[0])))
    center = support.bounds.mean(0)
    points = support.vertices-center
    elevation = float(np.degrees(np.arctan(1/np.sqrt(2))))
    spans = []
    for angle in np.linspace(-45, 315, 73):
        camera(renderer, elevation, float(angle))
        spans.append([np.abs(points @ renderer.right).max(), np.abs(points @ renderer.up).max()])
    span = np.max(spans, axis=0)
    scale = min((renderer.width-100)/(2*span[0]), (renderer.height-100)/(2*span[1]))*.96
    for u in np.linspace(0, 1, 7)[1:]:
        alpha = R.smooth(u)
        renderer.center = original_center*(1-alpha)+center*alpha
        renderer.scale = original_scale*(1-alpha)+scale*alpha
        azimuth = original_azimuth*(1-alpha)-45*alpha
        camera(renderer, original_elevation*(1-alpha)+elevation*alpha, azimuth)
        video.emit([(support, 'support')], guest, direction, visible=False,
                   phase='isometric_transition', camera_azimuth=azimuth)
    renderer.center, renderer.scale = center, scale
    for angle in np.linspace(-45, 315, inspection_frames-6):
        camera(renderer, elevation, float(angle))
        video.emit([(support, 'support')], guest, direction, visible=False,
                   phase='support_orbit', camera_azimuth=float(angle))
    return dict(object_withdrawal_checks=checks, final_object_projection_gap_m=gap,
                object_absent_during_inspection=True, support_geometry_changes_during_inspection=False,
                camera_orbit_degrees=360., inspection_elevation_degrees=elevation,
                inspection_frames=inspection_frames)


def translation(states, report, source_meshes, source_report):
    video = Video(H.DEMO/'translation/vis/pose1+2+4+6_translation.mp4',
                  renderer(source_meshes, source_report, states), 6)
    direction = np.asarray(report['direction_initial_world'])
    first, last = states['translation_0'], states['translation_48']
    try:
        video.emit([(first['final'], 'support')], first['guest'], direction, repeat=6,
                   phase='juxtapose_result', state='translation_0')
        for index in range(1, H.TRANSLATION_STEPS+1):
            state = states[f'translation_{index}']
            video.emit(layers(state), state['guest'], direction,
                       phase='translation', state=f'translation_{index}')
        video.emit(layers(last, red=False), last['guest'], direction, repeat=6,
                   phase='translated_wrap', state='translation_48')
        inspection = remove_object_and_orbit(video, last, direction, inspection_frames=72)
        video.finish(report, starts_after_juxtapose=True, direction_changed=False,
                     final_geometry=report['states']['translation_48'], **inspection)
    except BaseException:
        video.abort()
        raise


def juxtapose(video, meshes, report):
    direction = np.asarray(report['guest_direction_world'])
    initial = G.C.S.solid(meshes['initial'])
    body = G.C.S.solid(meshes['guest'])
    stop, _ = R.collision_stop(initial, meshes['guest'], direction)
    entrance = stop+.035

    def sweep(near, far):
        return G.C.S.solid(G.C.S.swept_solid(H.moved(meshes['guest'], near*direction),
                                            (far-near)*direction, fan_in=8))

    video.emit([(meshes['initial'], 'support')], meshes['guest'], direction, repeat=2,
               distance=entrance, visible=False, phase='juxtapose_initial')
    for u in np.linspace(0, 1, 7)[1:]:
        video.emit([(meshes['initial'], 'support')], meshes['guest'], direction,
                   distance=entrance+(stop-entrance)*R.smooth(u), phase='juxtapose_attempt')
    reached = sweep(stop, entrance)
    video.emit([(G.C.S.unpack(initial-reached), 'support'), (G.C.S.unpack(initial ^ reached), 'removed')],
               meshes['guest'], direction, repeat=2, distance=stop, phase='juxtapose_blocked')
    remaining, visited, previous = initial, body.translate((entrance*direction/G.C.S.SCALE).tolist()), entrance
    checks = []
    for distance in stop*(1-R.smooth(np.linspace(0, 1, 21))):
        current = body.translate((distance*direction/G.C.S.SCALE).tolist())
        segment = sweep(float(distance), previous)+current
        visited = visited+segment
        after = remaining-segment
        fresh = remaining-after
        outside = G.C.material_volume((initial-after)-visited)
        overlap = G.C.material_volume(after ^ current)
        restored = G.C.material_volume(after-remaining)
        assert max(outside, overlap, restored) <= G.TOL, (outside, overlap, restored)
        checks.append(dict(guest_exit_offset_m=float(distance),
                           deleted_outside_visited_sweep_m3=outside, current_body_overlap_m3=overlap,
                           previously_deleted_material_restored_m3=restored))
        video.emit([(G.C.S.unpack(after), 'support'), (G.C.S.unpack(fresh), 'removed')],
                   meshes['guest'], direction, distance=float(distance), phase='juxtapose_sweep_cut')
        remaining, previous = after, float(distance)
    retained = G.C.S.unpack(remaining)
    video.emit([(retained, 'support')], meshes['guest'], direction, phase='juxtapose_cut')
    for index in range(1, G.GROWTH_STEPS+1):
        video.emit([(retained, 'support'), (meshes[f'growth_{index}'], 'added')],
                   meshes['guest'], direction, phase='juxtapose_grow')
    video.emit([(retained, 'support'), (meshes['added'], 'added')], meshes['guest'], direction,
               phase='juxtapose_fitted')
    video.emit([(retained, 'support'), (meshes['added'], 'support')], meshes['guest'], direction,
               phase='juxtapose_final')
    assert video.frames == 42
    print('COMBINED Juxtapose finished', flush=True)
    return checks


def combined(states, report, source_meshes, source_report):
    video = Video(H.OUT/'juxtapose_translation_direction.mp4',
                  renderer(source_meshes, source_report, states, combined=True), 8)
    direction = np.asarray(report['direction_initial_world'])
    wheel = None
    try:
        cut_checks = juxtapose(video, source_meshes, source_report)
        first, last = states['translation_0'], states['translation_48']
        video.emit([(first['final'], 'support')], first['guest'], direction, repeat=2,
                   phase='translation_start', state='translation_0')
        for index in np.rint(np.linspace(1, H.TRANSLATION_STEPS, 32)).astype(int):
            state = states[f'translation_{index}']
            video.emit(layers(state), state['guest'], direction,
                       phase='translation', state=f'translation_{index}')
        video.emit(layers(last, red=False), last['guest'], direction,
                   phase='translation_fitted', state='translation_48')
        video.emit([(last['final'], 'support')], last['guest'], direction,
                   phase='translation_final', state='translation_48')
        assert video.frames == 78
        wheel = DirectionWheel(video.renderer, last['guest'], last['final'], report['full_exit_length_m'])
        video.emit([(last['final'], 'support')], last['guest'], direction, repeat=3,
                   phase='direction_start', state='translation_48',
                   direction_wheel=wheel, sweep_support=last['final'])
        reference = G.C.S.solid(last['final'])
        previous = reference
        final = None
        for index in range(1, H.DIRECTION_STEPS+1):
            name = f'direction_{index}'
            state = states[name]
            direction = np.asarray(report['states'][name]['direction_world'])
            final = G.C.S.solid(state['final'])
            kept, added, lost = final ^ reference, final-reference, previous-final
            colored = [(G.C.S.unpack(kept), 'support'), (G.C.S.unpack(added), 'added'),
                       (G.C.S.unpack(lost), 'removed')]
            video.emit(colored, state['guest'], direction, phase='direction', state=name,
                       direction_wheel=wheel, sweep_support=state['final'])
            previous = final
        state = states['direction_18']
        colored = [(G.C.S.unpack(final ^ reference), 'support'), (G.C.S.unpack(final-reference), 'added')]
        video.emit(colored, state['guest'], direction, repeat=6, phase='direction_fitted', state='direction_18',
                   direction_wheel=wheel, sweep_support=state['final'])
        video.emit([(state['final'], 'support')], state['guest'], direction, repeat=9,
                   phase='direction_final', state='direction_18',
                   direction_wheel=wheel, sweep_support=state['final'])
        assert video.frames == 114
        inspection = remove_object_and_orbit(video, state, direction, inspection_frames=66)
        video.finish(report, operation_order=['juxtapose', 'translation', 'direction'],
                     stage_seconds=dict(juxtapose=1.75, translation=1.5, direction=1.5,
                                        object_exit=.5, support_inspection=2.75),
                     direction_change_degrees=8., juxtapose_actual_sweep_checks=cut_checks,
                     direction_renderer_source=str(DIRECTION_SOURCE),
                     direction_arrow_and_sweep_visible=True, direction_overlay_states=wheel.records,
                     final_geometry=report['states']['direction_18'], **inspection)
    except BaseException:
        video.abort()
        raise
    finally:
        if wheel is not None:
            wheel.close()


def main(only='both'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    if args.rebuild or not (H.OUT/'data/states.npz').exists():
        H.Sequence().build()
    states, report = H.load()
    source_meshes, source_report = R.load()
    if only in ['both', 'translation']:
        translation(states, report, source_meshes, source_report)
    if only in ['both', 'combined']:
        combined(states, report, source_meshes, source_report)


if __name__ == '__main__':
    main()
