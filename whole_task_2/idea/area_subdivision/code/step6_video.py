"""Step6 on one area of the Step2 figure, as a film: the support holds the bunny on the table, the work arm (the
station's KUKA) works the part of the area it reaches there, the Franka Panda moves the support (the bunny in it) to the
next installation, the work arm works the next part, and so on -- every installation as Step6 saved it
(data/<run>/step6_robot/<result>_installations.npz: the object's and the support's poses, the faces, their needle
axes and KUKA joints).  The whole area is light pink throughout, deep pink once worked.  No text.

The motion is idea/demo/code/video.py's: the area's installations are the tasks of one seat (Step5's support, the
bunny in its seat), so between two of them the work arm backs off and waits aside while the Panda moves the support
with the bunny in it (franka.py's pick and place); every frame is checked (the work arm, the Panda) and the frames
meeting something are reported.  Nothing in the run's data is computed again or written.

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/area_subdivision/code/step6_video.py
  ... --stills        a few frames as png (the scratch folder) instead of the video
The video goes to idea/area_subdivision/<object>_<area>_step6.mp4 (big_20_W01_step6.mp4); the motion's caches to the
scratch folder.
"""
import argparse
import hashlib
import os
import pickle
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from whole_task_2.step1.common import ROOT, sha256, object_of
from whole_task_2.step1.reach_video import LIGHT
from whole_task_2.step5 import joined_video as J
import sys
sys.path.insert(0, str(ROOT/'whole_task_2/idea/demo/code'))      # its video.py and franka.py (the root has a `video` package)
import video as D

HERE = Path(__file__).resolve().parents[1]
RUN, AREA, RESULT = 'B+big_20', 'W01', 'r00'
SCRATCH = Path(os.environ.get('STEP6_VIDEO_CACHE', '/tmp/claude-1000/-home-yli581-Desktop-cadGrasp/093a55e2-7a8c-4226-abf3-06a98066f6d2/scratchpad/step6_video'))
WORKERS = 6             # processes checking and rendering the film (one GL context each)


class Area:
    """What D.Demo reads as a model: Step5's support with the bunny in the area's seat (video.Model), the area's Step6
    installations as the seat's tasks (their poses, faces, axes and joints as saved), the seat's exit at each."""
    def __init__(self, run=RUN, area=AREA, result=RESULT):
        m = D.Model(run, result); k = next(i for i, r in enumerate(m.records) if r['id'] == area); X = m.X[m.seat_of[k]]
        z = np.load(ROOT/'whole_task_2/data'/run/'step6_robot'/f'{result}_installations.npz'); n = 1; self.records, S = [], []
        while f'{area}_{n}_faces' in z.files:
            name = f'{area}_{n}'; O, W = z[name+'_world_from_object'], z[name+'_world_from_support']
            if not np.allclose(np.linalg.inv(W)@O, X, atol=1e-9): raise ValueError(f'{name}: not the seat of {area} in {result}')
            faces = z[name+'_faces'].astype(int).tolist()
            self.records.append(dict(id=name, pose=O, certified_face_ids=faces, working_face_ids=faces, axes=z[name+'_axes'], joints=z[name+'_joints']))
            S.append(W); n += 1
        # The seat's exit (support frame, from Step5's installation of the area), world at every installation.
        R = (np.asarray(m.records[k]['pose'])@np.linalg.inv(X))[:3, :3]; e = R.T@m.exits[k]
        self.exits = np.asarray([W[:3, :3]@e for W in S]); self.exit = None
        self.source, self.setup, self.support, self.tolerance, self.policy = m.source, m.setup, m.support, m.tolerance, m.policy
        self.X, self.seat_of = [X], [0]*len(S); self.run, self.source_of, self.row = run, m.source_of+f' / step6 {area}', m.row
        self.exit_length = m.row['tasks'][k]['exit_length_m']; self.area = np.unique(np.concatenate([r['working_face_ids'] for r in self.records]))
        # The table top: Step6 stands the support's lowest point on it.  D.Demo reads the layout as Step4's (its lift
        # the table, the support's place world <- support).
        Z = float(min((m.support.vertices@W[:3, :3].T+W[:3, 3])[:, 2].min() for W in S))
        self.layout = SimpleNamespace(poses=[J.lifted(-Z)@r['pose'] for r in self.records], material=[np.linalg.inv(W)@J.lifted(Z) for W in S])
        self.npz = ROOT/'whole_task_2/data'/run/'step6_robot'/f'{result}_installations.npz'


FILM = dict(D.FILM, view=(.34, -.94, .9), fit=.8)     # from above, between the two arms (big_20: the KUKA at 164 deg,
                                                        # the Panda at 57 deg about the bunny), the area facing up
NEAR = .25              # m: the camera holds the needle and head this close to the bunny's centre


class Film(D.Film):
    """video.Film close on the bunny: the camera holds the bunny and the support wherever they go and the needle while
    it works (not the waiting arm); the whole area light pink throughout (deep pink once worked)."""
    def __init__(self, plan, handler, frames, film=FILM):
        super().__init__(plan, handler, frames, film)

    @staticmethod
    def held_points(plan, frames):
        P = [plan.source.vertices[::7]@f['B'][:3, :3].T+f['B'][:3, 3] for f in frames[::6]]
        P += [plan.support.vertices[::40]@f['S'][:3, :3].T+f['S'][:3, 3] for f in frames[::6]]
        for f in frames[::6]:                           # the needle and head near the bunny (NEAR m), not on their way
            tcp, z, _, _ = plan.arm.forward(f['q']); tip = tcp+[0, 0, f['off']]
            if np.linalg.norm(tip-f['B'][:3, :3]@plan.source.centroid-f['B'][:3, 3]) < NEAR: P.append(np.vstack([tip, tip-z*(plan.L+.07)]))
        return np.vstack(P)

    def frame(self, i):
        plan, scene, hd = self.plan, self.scene, self.handler; f = self.frames[i]; src = plan.source; g = scene.r
        support, table, column = self.shapes; w, h = self.size
        colors = np.tile(g.GRAY, (len(src.faces), 1)); colors[plan.model.area] = LIGHT; colors[f['painted']] = g.PINK
        L = D.matrices(plan.arm.forward(f['q'], geometry=True)[2]); L[:, 2, 3] += f['off']
        tcp, z, _, _ = plan.arm.forward(f['q']); tip = tcp+[0, 0, f['off']]
        C = J.lifted(self.bottom); C[2, 2] = self.base[2]+f['off']-self.bottom; C[:2, 3] = self.base[:2]
        H = hd.panda.visual_frames(f['h'], f['open'])
        if scene.colors is None or not np.array_equal(colors, scene.colors): scene.body.paint(colors); scene.colors = colors
        draws = [(scene.body, f['B'])]+[(s, J.tool_frame(tip, z)) for s in self.needle]+[(s, L[k]) for k, s in self.work]
        draws += list(zip(self.panda, H))+[(support, f['S']), (table, np.eye(4)), (column, C)]
        ctx, studio = scene.ctx, scene.studio
        studio.shadow_buffer.use(); ctx.viewport = (0, 0, 1536, 1536); studio.shadow_buffer.clear(depth=1)
        for shape, world in draws: shape.shadow(ctx, scene.shadow_shader, world)
        ctx.enable(scene.gl.DEPTH_TEST)
        studio.gbuffer.use(); ctx.viewport = (0, 0, *studio.size); studio.gbuffer.clear(1, 1, 1, 1, depth=1); studio.shadow.use(3)
        for shape, world in draws: shape.draw(scene.shader, world)
        return np.frombuffer(studio.finish(self.vp, self.view), np.uint8).reshape(h, w, 3)


D.Film = Film                                           # D._segment builds the film's frames with it


def cached(name, inputs, make):
    """make() from the scratch cache when its inputs (and this file, video.py, franka.py, joined_video.py) are unchanged."""
    code = ''.join(Path(m.__file__).read_text() for m in (D, D.F, J))+Path(__file__).read_text()
    key = hashlib.sha256(repr((inputs, code)).encode()).hexdigest(); path = SCRATCH/f'{name}.pkl'
    if path.exists():
        saved = pickle.loads(path.read_bytes())
        if saved['key'] == key: return saved['value']
    value = make(); SCRATCH.mkdir(parents=True, exist_ok=True); path.write_bytes(pickle.dumps(dict(key=key, value=value))); return value


def run(run_name=RUN, area=AREA, result=RESULT, stills=False):
    began = time.monotonic(); model = Area(run_name, area, result); n = len(model.records)
    report = dict(tasks=[dict(exit_length_m=model.exit_length)]*n)
    print('STEP6 VIDEO', run_name, area, n, 'installations,', [len(r['working_face_ids']) for r in model.records], 'faces', flush=True)
    inputs = (sha256(model.npz), model.row['support_sha256'], repr(D.POLICY))

    def motion():
        p = D.Demo(model, model.layout, model.support, report, D.POLICY); p.tasks = p.build(); return p.frames, p.events, p.tasks
    frames, events, tasks = cached(f'{run_name}_{area}_plan', inputs, motion)
    heavy = D.Demo.task_stops; D.Demo.task_stops = J.Plan.task_stops                 # the stops' retreats serve planning only
    try: p = D.Demo(model, model.layout, model.support, report, D.POLICY)
    finally: D.Demo.task_stops = heavy
    p.frames, p.events, p.tasks = frames, events, tasks
    failing = [e for e in events if 'fails' in e]; print('STEP6 VIDEO plan', len(frames), 'frames,', len(failing), 'failing motions', flush=True)

    def handling():
        hd = D.Handler(p); return hd.film(), dict(base=hd.panda.base, yaw=hd.panda.yaw, rest=hd.rest, park=hd.park)
    whole, saved = cached(f'{run_name}_{area}_film', inputs+(repr(D.FRANKA),), handling)
    hd = D.Handler.cached(p, saved)
    start = next(i for i, f in enumerate(whole) if f['task'] is not None)
    pick = np.unique(np.linspace(start, len(whole)-1, int(D.RENDER['seconds']*D.POLICY['fps'])).round().astype(int)); film = [whole[i] for i in pick]
    D._FILM = (p, hd, film, D.POLICY['fps'])
    with ProcessPoolExecutor(WORKERS) as pool: checks = sorted(pool.map(D._check, range(len(film)), chunksize=32))
    bad = [(i, c) for i, c in checks if c]
    print('STEP6 VIDEO check', len(film), 'frames,', len(bad), 'meeting something', sorted({x for _, c in bad for x in c}), flush=True)
    name = f"{run_name.removeprefix(object_of(run_name)+'+')}_{area}_step6"
    if stills:
        from PIL import Image
        f = Film(p, hd, film); out = SCRATCH/'stills'; out.mkdir(parents=True, exist_ok=True)
        try:
            moves = [i for i, x in enumerate(film) if x['grip'] and (i == 0 or film[i-1]['grip'] != x['grip'])]
            for i in sorted({0, len(film)//4, *moves, *[m+40 for m in moves if m+40 < len(film)], len(film)-1}):
                Image.fromarray(f.frame(i)[::-1]).save(out/f'{name}_{i:04d}.png')
        finally: f.close()
        print('STEP6 VIDEO stills', out, flush=True); return out
    video = HERE/f'{name}.mp4'; parts = SCRATCH/f'segments_{name}'; parts.mkdir(parents=True, exist_ok=True)
    cuts = np.linspace(0, len(film), WORKERS+1).astype(int)
    with ProcessPoolExecutor(WORKERS) as pool:
        segments = list(pool.map(D._segment, [(a, b, parts/f'{k:02d}.mp4') for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])) if b > a]))
    (parts/'list.txt').write_text(''.join(f"file '{Path(s).name}'\n" for s in segments))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(parts/'list.txt'), '-c', 'copy', '-movflags', '+faststart', str(video)], check=True)
    for x in parts.iterdir(): x.unlink()
    parts.rmdir()
    print('STEP6 VIDEO', video, round(len(film)/D.POLICY['fps'], 1), 's,', len(failing), 'failing motions,', len(bad), 'frames meeting something,',
          round(time.monotonic()-began), 's', flush=True)
    return video


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run', default=RUN); a.add_argument('--area', default=AREA); a.add_argument('--result', default=RESULT)
    a.add_argument('--stills', action='store_true')
    a = a.parse_args(); run(a.run, a.area, a.result, a.stills)
