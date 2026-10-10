"""Render text-free six-second Translation and eight-second Combined videos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

import geometry as H

G, R = H.G, H.R


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

    def emit(self, materials, body, direction, repeat=1, distance=0., visible=True, phase='', state=''):
        self.renderer.meshes['guest'] = body
        self.renderer.direction = np.asarray(direction)
        frame = self.renderer.frame(materials, distance, '', guest_visible=visible)
        for _ in range(repeat):
            self.encoder.stdin.write(frame.tobytes())
        self.timeline.append(dict(phase=phase, state=state, first_frame=self.frames, repetitions=repeat,
                                  guest_exit_offset_m=float(distance), direction_world=list(direction)))
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
                                     for p in [Path(__file__), Path(H.__file__), Path(G.__file__), Path(R.__file__)]},
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


def translation(states, report, source_meshes, source_report):
    video = Video(H.DEMO/'translation/vis/pose1+2+4+6_translation.mp4',
                  renderer(source_meshes, source_report, states), 6)
    direction = np.asarray(report['direction_initial_world'])
    first, last = states['translation_0'], states['translation_48']
    try:
        video.emit([(first['final'], 'support')], first['guest'], direction, repeat=12,
                   phase='juxtapose_result', state='translation_0')
        for index in range(1, H.TRANSLATION_STEPS+1):
            state = states[f'translation_{index}']
            video.emit(layers(state), state['guest'], direction, repeat=2,
                       phase='translation', state=f'translation_{index}')
        video.emit(layers(last, red=False), last['guest'], direction, repeat=24,
                   phase='translated_wrap', state='translation_48')
        video.emit([(last['final'], 'support')], last['guest'], direction, repeat=12,
                   phase='final', state='translation_48')
        video.finish(report, starts_after_juxtapose=True, direction_changed=False,
                     final_geometry=report['states']['translation_48'])
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
               meshes['guest'], direction, repeat=3, distance=stop, phase='juxtapose_blocked')
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
    video.emit([(retained, 'support')], meshes['guest'], direction, repeat=3, phase='juxtapose_cut')
    for index in range(1, G.GROWTH_STEPS+1):
        video.emit([(retained, 'support'), (meshes[f'growth_{index}'], 'added')],
                   meshes['guest'], direction, repeat=2, phase='juxtapose_grow')
    video.emit([(retained, 'support'), (meshes['added'], 'added')], meshes['guest'], direction,
               repeat=5, phase='juxtapose_fitted')
    video.emit([(retained, 'support'), (meshes['added'], 'support')], meshes['guest'], direction,
               repeat=4, phase='juxtapose_final')
    assert video.frames == 60
    print('COMBINED Juxtapose finished', flush=True)
    return checks


def combined(states, report, source_meshes, source_report):
    video = Video(H.OUT/'juxtapose_translation_direction.mp4',
                  renderer(source_meshes, source_report, states, combined=True), 8)
    direction = np.asarray(report['direction_initial_world'])
    try:
        cut_checks = juxtapose(video, source_meshes, source_report)
        first, last = states['translation_0'], states['translation_48']
        video.emit([(first['final'], 'support')], first['guest'], direction, repeat=8,
                   phase='translation_start', state='translation_0')
        for index in range(1, H.TRANSLATION_STEPS+1):
            state = states[f'translation_{index}']
            video.emit(layers(state), state['guest'], direction,
                       phase='translation', state=f'translation_{index}')
        video.emit(layers(last, red=False), last['guest'], direction, repeat=12,
                   phase='translation_fitted', state='translation_48')
        video.emit([(last['final'], 'support')], last['guest'], direction, repeat=4,
                   phase='translation_final', state='translation_48')
        assert video.frames == 132
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
            video.emit(colored, state['guest'], direction, phase='direction', state=name)
            previous = final
        state = states['direction_18']
        colored = [(G.C.S.unpack(final ^ reference), 'support'), (G.C.S.unpack(final-reference), 'added')]
        video.emit(colored, state['guest'], direction, repeat=6, phase='direction_fitted', state='direction_18')
        body = G.C.S.solid(state['guest'])
        exit_checks = []
        for phase, distances in [('exit', np.linspace(0, .07, 14)[1:]),
                                 ('return', np.linspace(.07, 0, 14)[1:])]:
            for distance in distances:
                overlap = G.C.material_volume(final ^ body.translate((distance*direction/G.C.S.SCALE).tolist()))
                assert overlap <= G.TOL, (phase, distance, overlap)
                exit_checks.append(dict(offset_m=float(distance), overlap_m3=overlap))
                video.emit(colored, state['guest'], direction, distance=float(distance),
                           phase='direction_'+phase, state='direction_18')
        video.emit([(state['final'], 'support')], state['guest'], direction, repeat=10,
                   phase='final', state='direction_18')
        video.finish(report, operation_order=['juxtapose', 'translation', 'direction'],
                     stage_seconds=dict(juxtapose=2.5, translation=3., direction=2.5),
                     direction_change_degrees=8., juxtapose_actual_sweep_checks=cut_checks,
                     final_exit_playback_checks=exit_checks, final_geometry=report['states']['direction_18'])
    except BaseException:
        video.abort()
        raise


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
