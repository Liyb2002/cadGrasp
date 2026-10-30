"""A run's Step5 support at work, every direction area of Step2 in several translations: the needle tip may move only
inside the station's work cube, so it cannot cover a direction area at once; the Franka Panda moves the support (the
bunny in its seat) along the table to bring the next part of the area within reach.  One fixed camera, the work arm on
the right; no text.

The data, as saved (nothing of the run computed again or written): Step5's support (data/<run>/step5_combined/<result>)
with the bunny in its seats, and every area's certified pose (its base, turned as combine turned it), faces and needle
axes, placed as Step6 places an area: the support's lowest point on the table (an area whose bunny would then go
through the table is left out; an area on several bases is split among them as Step6 split it).  The reach: the station's work cube
(robots_config's surface_workspace, EDGE) across the needle, centred on the needle's line through the work point; along
the needle the tip goes as deep as the faces lie.  An area's faces, seen along the needle, are covered with the fewest
such squares (greedy, then every square moved to where it covers the most with the others, from random starts as well),
all but LEFT of its area; each face goes to the nearest square holding it, and a square is one translation: the support
shifted on the table (a shift across the needle is a horizontal one, the needle not being level) so that the square's
middle lies on that line.  The translations of an area follow their squares along the area's long side.  At each
translation the KUKA's joints for every face (the robot-free plan keeps none): its IK from face to nearest face,
tracked.  The work arm and the Panda are idea/demo/code/video.py's: between two translations of an area the Panda moves
the support with the bunny in it; from one area to the next it changes the seat as there (a change it cannot make tried
again with the bunny on another rest spot, away from the work arm).  The Panda stands where video.py puts it, but never between the camera and
the bunny.  The whole area is light pink while it is worked, deep pink once worked.  Every frame is checked (the work
arm, the Panda) and the frames meeting something are reported.

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/demo/code/translations.py B+big_01_v2 --result r01
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/demo/code/translations.py B+whole_shape_v2
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/demo/code/translations.py B+big_08_v2
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/demo/code/translations.py B+big_20_v2
  ... --clip W01      only the support moved from the area's first translation to its second (CLIP_SECONDS)
  ... --stills        a few frames as png (the scratch folder) instead of the video
The video goes to idea/demo/<run>.mp4 (SECONDS long) or idea/demo/<run>_translation.mp4; the motion's caches to the
scratch folder.
"""
import argparse
import hashlib
import inspect
import json
import os
import pickle
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from whole_task_2.step1.common import ROOT
from whole_task_2.step1.reach_video import LIGHT
from whole_task_2.step5 import joined_video as J
sys.path.insert(0, str(ROOT/'whole_task_2/idea/demo/code'))      # its video.py and franka.py (the root has a `video` package)
import video as D

HERE = Path(__file__).resolve().parents[1]
RESULT = 'r00'
SCRATCH = Path(os.environ.get('TRANSLATIONS_CACHE', '/tmp/claude-1000/-home-yli581-Desktop-cadGrasp/093a55e2-7a8c-4226-abf3-06a98066f6d2/scratchpad/translations'))
EDGE = .075             # m: the station's work cube (robots_config/random/07_kuka.json surface_workspace edge_m)
LEFT = .005             # an area's share its squares may leave
STEP, STARTS = .003, 12 # m: the squares' places tried; random starts per count
SECONDS = 22.           # s: the film of a run, evenly in time from its first work to its end
CLIP_SECONDS = 8.       # s: the film of one translation, ...
AROUND = 2.             # s (of the motion): ... from this long before the last stop of the first to after the first of the next
RISE = .5               # the camera's height over its horizontal distance
WORKERS = 8             # processes checking and rendering the film (one GL context each)
NEAR = .25              # m: the camera holds the needle and head this close to the bunny's centre
# The Panda: more grasps kept and tried than video.py's (a seat change here turns the support onto another base far
# from where its grasps were kept); video.py's Handler reads the dict itself.
D.FRANKA.update(keep=dict(S=120, B=200), tries=96, samples=dict(S=6000, B=30000), rest=dict(poses=12, yaws=12))
D.POLICY.update(franka_m=.55)                           # m: the Panda's base nearer the station (video.py: .6)


def unit(v):
    v = np.asarray(v, float); return v/np.linalg.norm(v)


def squares(p, A, edge=EDGE):
    """The fewest squares (edge, axis-aligned in the plane across the needle) holding all but LEFT of the faces' area
    (p: the faces' coordinates there, A: their areas): (the squares' middles, the square of every face or -1)."""
    lo, hi = p.min(0)-edge, p.max(0); X, Y = np.meshgrid(np.arange(lo[0], hi[0]+STEP, STEP), np.arange(lo[1], hi[1]+STEP, STEP))
    c = np.c_[X.ravel(), Y.ravel()]
    M = (p[None, :, 0] >= c[:, None, 0]) & (p[None, :, 0] <= c[:, None, 0]+edge) & (p[None, :, 1] >= c[:, None, 1]) & (p[None, :, 1] <= c[:, None, 1]+edge)
    keep = M.any(1); M, c = M[keep], c[keep]; Mf = M.astype(np.float32); need = (1-LEFT)*A.sum(); rng = np.random.default_rng(0)

    def polish(chosen):                                   # every square to where it covers the most with the others
        while True:
            better = False
            for j in range(len(chosen)):
                others = M[chosen[:j]+chosen[j+1:]].any(0) if len(chosen) > 1 else np.zeros(len(A), bool)
                gain = Mf@(A*~others); b = int(np.argmax(gain))
                if gain[b] > gain[chosen[j]]+1e-12: chosen[j] = b; better = True
            if not better: return chosen
    for k in range(1, 50):
        best = None
        for start in range(STARTS):
            if start == 0:                                # greedy
                chosen, cov = [], np.zeros(len(A), bool)
                for _ in range(k): b = int(np.argmax(Mf@(A*~cov))); chosen.append(b); cov |= M[b]
            else: chosen = [int(i) for i in rng.choice(len(c), k, replace=False)]
            chosen = polish(chosen); got = float(A[M[chosen].any(0)].sum())
            if best is None or got > best[0]: best = (got, chosen)
            if got >= need: break
        if best[0] >= need: break
    middle = c[best[1]]+edge/2; d = np.linalg.norm(p[None]-middle[:, None], axis=2); d[~M[best[1]]] = np.inf
    owner = np.argmin(d, 0); owner[~M[best[1]].any(0)] = -1
    return middle, owner


def ik(arm, tips, axes):
    """The KUKA's joints with the needle tip on every tip along its axis (world): from one solved face to its nearest
    unsolved one, tracked, else solved anew.  Returns joints as Step1 keeps them (n, 4, joints: the centroid's in
    every slot) and which tips it reaches."""
    n = len(tips); Q = np.zeros((n, 4, len(arm.lower))); ok = np.zeros(n, bool); todo = set(range(n)); i = 0; q = None
    while todo:
        todo.discard(i); p = None
        if q is not None:
            try: p = J.track(arm, tips[i], axes[i], q)
            except ValueError: p = None
        if p is None:
            try: p = arm.solve(tips[i], -axes[i])
            except ValueError: p = None
        if p is not None: Q[i] = p; ok[i] = True; q = p
        if todo: rest = np.fromiter(todo, int); i = int(rest[np.argmin(np.linalg.norm(tips[rest]-tips[i], axis=1))])
    return Q, ok


class Translations:
    """What D.Demo reads as a model: Step5's support with the bunny in its seats (video.Model), every area's
    translations as the tasks, in the result's order of the areas."""
    def __init__(self, run, result=RESULT):
        m = D.Model(run, result); s = m.setup; self.path = m.path
        a = unit(s['working_direction_world']); e2 = unit(np.cross([0, 0, 1.], a)); e1 = np.cross(a, e2); h = unit(np.r_[a[:2], 0.])
        line = np.asarray(s['work_point_m'], float); across = lambda x: np.c_[x@e1, x@e2]
        arm = J.work_arm(s); src = m.source; self.records, self.seat_of, S, exits, self.areas = [], [], [], [], []; self.unreached = 0
        bases = Counter(r['id'].split('@')[0] for r in m.records); self.left_out = []
        step6 = ROOT/'whole_task_2/data'/run/'step6_robot'; report6 = json.loads((step6/'report.json').read_text()) if (step6/'report.json').exists() else None
        for k, r in enumerate(m.records):
            # The area as Step6 installs it: its certified pose (its base, turned as combine turned it), raised or lowered
            # so that the support's lowest point stands on the table; its certified faces and their needle axes.
            area, X = r['id'], m.X[m.seat_of[k]]; P = np.asarray(r['pose']); W = P@np.linalg.inv(X)
            O = J.lifted(s['table_height_m']-float((m.support.vertices@W[:3, :3].T+W[:3, 3])[:, 2].min()))@P; W = O@np.linalg.inv(X)
            # The bunny in its seat lower than the support's lowest point would go through the table (a defect of the
            # result): the area is left out.
            if float((src.vertices@O[:3, :3].T+O[:3, 3])[:, 2].min()) < s['table_height_m']-.001: self.left_out.append(area); continue
            faces, axes = np.asarray(r['certified_face_ids'], int), np.asarray(r['axes']); whole = np.asarray(r['working_face_ids'], int)
            if bases[area.split('@')[0]] > 1:
                # An area combine turned in its own seat (several bases, each record all its faces): its faces as Step6
                # shared them out among the bases (its installations of this result).
                if report6 is None or report6['result'] != result: raise ValueError(f'{area}: on several bases, and no Step6 of {result}')
                with np.load(step6/f'{result}_installations.npz') as z6:
                    row6 = next(x for x in report6['areas'] if x['area'] == area); keep = np.isin(faces, np.concatenate([z6[i['name']+'_faces'] for i in row6['installations']]))
                faces, axes, whole = faces[keep], axes[keep], faces[keep]
            C = src.triangles_center[faces]@O[:3, :3].T+O[:3, 3]; middle, owner = squares(across(C), src.area_faces[faces])
            # the translations along the area's long side; the seat's exit (support frame) from Step5's installation
            u = middle-middle.mean(0); order = np.argsort(u@np.linalg.svd(u, full_matrices=False)[2][0]) if len(u) > 1 else [0]
            e = W[:3, :3].T@m.exits[k]
            for n, j in enumerate(order):
                d = across(line)[0]-middle[j]; T = J.shifted(d[0]/(h@e1)*h+d[1]*e2)@O@np.linalg.inv(P)
                mine = np.flatnonzero(owner == j); M = T@P; tips = src.triangles_center[faces[mine]]@M[:3, :3].T+M[:3, 3]; world = axes[mine]@P[:3, :3].T
                Q, ok = ik(arm, tips, world)
                ids = faces[mine][ok].tolist(); self.unreached += int((~ok).sum())
                self.records.append(dict(id=f'{area}.{n+1}', pose=T@P, certified_face_ids=ids, working_face_ids=ids, axes=axes[mine][ok], joints=Q[ok]))
                self.seat_of.append(m.seat_of[k]); S.append(T@P@np.linalg.inv(X)); exits.append(S[-1][:3, :3]@e); self.areas.append(whole)
        lengths = [dict(m.row['tasks'][next(k for k, r in enumerate(m.records) if r['id'] == x['id'].split('.')[0])]) for x in self.records]
        self.exits = np.asarray(exits); self.exit = None; self.X = m.X; self.row = dict(m.row, tasks=lengths)
        self.source, self.setup, self.support, self.tolerance, self.policy = src, s, m.support, m.tolerance, m.policy
        self.run, self.source_of = run, m.source_of+' / step6 translations'
        # The table top: Step6 stands the support's lowest point on it (a shift keeps it there).
        Z = float(min((m.support.vertices@W[:3, :3].T+W[:3, 3])[:, 2].min() for W in S))
        self.layout = SimpleNamespace(poses=[J.lifted(-Z)@r['pose'] for r in self.records], material=[np.linalg.inv(W)@J.lifted(Z) for W in S])
        self.counts = [(r['id'], sum(x['id'].split('.')[0] == r['id'] for x in self.records)) for r in m.records if r['id'] not in self.left_out]


def camera(plan):
    """The camera's direction (to it): square to the line from the bunny's mean place to the work arm's base, the work
    arm on the right, RISE high."""
    arm = np.asarray(plan.model.setup['base_position_world_m'], float)-plan._centre; arm[2] = 0.
    return unit(unit(np.cross(arm, [0, 0, 1.]))+[0, 0, RISE])


def franka_cell(self):
    """video.Demo.franka_cell, the Panda never between the camera and the bunny: of the azimuths (every 30 deg) at least
    90 deg from the work arm's base, those not toward the camera (camera()), the one the exits point to the most."""
    if getattr(self, '_cell', None) is not None: return self._cell
    m, p = self.model, self.policy
    c = np.mean([(S@m.X[m.seat_of[k]])[:3, :3]@self.source.centroid+(S@m.X[m.seat_of[k]])[:3, 3] for k, S in enumerate(self.S)], axis=0); c[2] = self.Z0
    self._centre = c
    arm = np.asarray(m.setup['base_position_world_m'], float)-c; arm[2] = 0.; arm /= np.linalg.norm(arm)
    e = np.array(m.exits, float); e[:, 2] = 0.; e = e.mean(0); view = camera(self); view[2] = 0.
    dirs = [np.array([np.cos(t), np.sin(t), 0.]) for t in np.radians(np.arange(0., 360., 30.))]
    u = max((d for d in dirs if d@arm <= 1e-9 and d@view <= 1e-9), key=lambda d: (round(float(d@e), 6), -float(d@view)))
    base = c+u*p['franka_m']; yaw = float(np.arctan2(*(c-base)[1::-1]))
    foot = np.vstack([(self.support.vertices[::5]@S[:3, :3].T+S[:3, 3])[:, :2] for S in self.S])
    self.reach = float(np.linalg.norm(self.source.vertices-self.source.centroid, axis=1).max())
    sign = -1. if np.cross(c-base, arm)[2] > 0 else 1.                          # turning away from the work arm
    def swung(t): return base+J.Rotation.from_euler('z', sign*t, degrees=True).as_matrix()@(c-base)
    rest = next((swung(t) for t in p['rest_turn'] if np.linalg.norm(foot-swung(t)[:2], axis=1).min() >= self.reach+p['rest_gap']), None)
    if rest is None: raise ValueError('No rest spot for the bunny clear of the support')
    self._cell = (base, yaw, rest); return self._cell


def rest_spots(plan):
    """Where the bunny may rest between seats: franka_cell's spot first, then its rule's other turns about the Panda's
    base (rest_turn, away from the work arm: the table under them stays clear of it), each clear of every footprint of
    the support."""
    base, _, rest = plan.franka_cell(); c = plan._centre; p = plan.policy
    foot = np.vstack([(plan.support.vertices[::5]@S[:3, :3].T+S[:3, 3])[:, :2] for S in plan.S])
    arm = np.asarray(plan.model.setup['base_position_world_m'], float)-c; arm[2] = 0.
    sign = -1. if np.cross(c-base, arm)[2] > 0 else 1.; out = [rest]
    for t in p['rest_turn']:
        x = base+J.Rotation.from_euler('z', sign*t, degrees=True).as_matrix()@(c-base)
        if np.linalg.norm(foot-x[:2], axis=1).min() >= plan.reach+p['rest_gap'] and not any(np.allclose(x, y) for y in out): out.append(x)
    return out


def make_table(self, edge):
    """video.Demo's table, under every rest spot (rest_spots) too."""
    base, _, _ = self.franka_cell(); r = self.reach
    P = np.vstack([*[(self.support.vertices[::5]@S[:3, :3].T+S[:3, 3])[:, :2] for S in self.S],
                   *[x[:2]+[[r, r], [-r, -r]] for x in rest_spots(self)], base[:2]+[[.15, .15], [-.15, -.15]]])
    m = self.policy['table_margin_fraction']*self.D; low, high = P.min(0)-m, P.max(0)+m; t = self.policy['table_thickness_fraction']*self.D
    box = D.trimesh.creation.box(extents=[*(high-low), t]); box.apply_translation([*(low+high)/2, self.Z0-t/2])
    fine = D.trimesh.Trimesh(*D.trimesh.remesh.subdivide_to_size(box.vertices, box.faces, edge, max_iter=40), process=False)
    return box, J.placed(fine, np.eye(4))


def _change(i):
    """video._change, a change the Panda cannot make with the bunny on its rest spot tried again with the bunny resting
    on the other rest spots (rest_spots), in turn."""
    hd, waits = D._HANDLER; began = time.monotonic(); why = []
    for t, rest in enumerate(rest_spots(hd.plan)):
        hd.rest = rest
        try: out = hd.change(waits[i], seed=i+101*t)
        except D.F.Failed as e: why.append(str(e)); continue
        print('TRANSLATIONS FRANKA change', waits[i]['change'], len(out), 'frames, rest spot', t, 'in', round(time.monotonic()-began), 's', flush=True)
        return i, out
    print('TRANSLATIONS FRANKA failed', waits[i]['change'], why[0], flush=True); return i, '; '.join(why[:1])


D.Demo.franka_cell = franka_cell
D.Demo.make_table = make_table
D._change = _change


class Film(D.Film):
    """video.Film from camera(), the whole area of the task light pink (deep pink once worked)."""
    def __init__(self, plan, handler, frames, film=D.FILM):
        super().__init__(plan, handler, frames, dict(film, view=tuple(camera(plan))))

    @staticmethod
    def held_points(plan, frames):
        """The bunny and the support wherever they go, the needle and head while near the bunny (NEAR m)."""
        P = [plan.source.vertices[::7]@f['B'][:3, :3].T+f['B'][:3, 3] for f in frames[::6]]
        P += [plan.support.vertices[::40]@f['S'][:3, :3].T+f['S'][:3, 3] for f in frames[::6]]
        for f in frames[::6]:
            tcp, z, _, _ = plan.arm.forward(f['q']); tip = tcp+[0, 0, f['off']]
            if np.linalg.norm(tip-f['B'][:3, :3]@plan.source.centroid-f['B'][:3, 3]) < NEAR: P.append(np.vstack([tip, tip-z*(plan.L+.07)]))
        return np.vstack(P)

    def frame(self, i):
        plan, scene, hd = self.plan, self.scene, self.handler; f = self.frames[i]; src = plan.source; g = scene.r
        support, table, column = self.shapes; w, h = self.size
        colors = np.tile(g.GRAY, (len(src.faces), 1))
        if f['task'] is not None: colors[plan.model.areas[f['task']]] = LIGHT
        colors[f['painted']] = g.PINK
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
    """make() from the scratch cache when its inputs and the motion's code (the translations here, video.py, franka.py,
    joined_video.py) are unchanged."""
    code = ''.join(Path(m.__file__).read_text() for m in (D, D.F, J))
    code += ''.join(inspect.getsource(f) for f in (squares, ik, Translations, camera, franka_cell, rest_spots, make_table, _change))
    key = hashlib.sha256(repr((inputs, code)).encode()).hexdigest(); path = SCRATCH/f'{name}.pkl'
    if path.exists():
        saved = pickle.loads(path.read_bytes())
        if saved['key'] == key: return saved['value']
    value = make(); SCRATCH.mkdir(parents=True, exist_ok=True); path.write_bytes(pickle.dumps(dict(key=key, value=value))); return value


def run(run_name, result=RESULT, stills=False, clip=None):
    began = time.monotonic(); model = Translations(run_name, result); n = len(model.records)
    report = dict(tasks=model.row['tasks'])
    print('TRANSLATIONS', run_name, result, n, 'translations', model.counts, model.unreached, 'faces the KUKA does not reach; left out (bunny below the table):',
          model.left_out, flush=True)
    inputs = (run_name, result, model.row['support_sha256'], repr(D.POLICY), EDGE, LEFT, RISE)

    def motion():
        p = D.Demo(model, model.layout, model.support, report, D.POLICY); p.tasks = p.build(); return p.frames, p.events, p.tasks
    frames, events, tasks = cached(f'{run_name}_plan', inputs, motion)
    heavy = D.Demo.task_stops; D.Demo.task_stops = J.Plan.task_stops                 # the stops' retreats serve planning only
    try: p = D.Demo(model, model.layout, model.support, report, D.POLICY)
    finally: D.Demo.task_stops = heavy
    p.frames, p.events, p.tasks = frames, events, tasks
    failing = [e for e in events if 'fails' in e]; print('TRANSLATIONS plan', len(frames), 'frames,', len(failing), 'failing motions', failing[:4], flush=True)

    def handling():
        hd = D.Handler(p); return hd.film(), dict(base=hd.panda.base, yaw=hd.panda.yaw, rest=hd.rest, park=hd.park)
    whole, saved = cached(f'{run_name}_film', inputs+(repr(D.FRANKA),), handling)
    hd = D.Handler.cached(p, saved)
    if clip is None:                                    # the run from its first work to its end
        first, last = next(i for i, f in enumerate(whole) if f['task'] is not None), len(whole)-1; seconds = SECONDS
    else:                                               # the support moved from the area's first translation to its second
        ids = [r['id'] for r in model.records]
        if f'{clip}.2' not in ids: raise ValueError(f'{clip} of {run_name} {result} has no second translation')
        k, k2 = ids.index(f'{clip}.1'), ids.index(f'{clip}.2'); around = int(AROUND*D.POLICY['fps'])
        first = max(i for i, f in enumerate(whole) if f['task'] == k and f['contact'])-around
        last = min(i for i, f in enumerate(whole) if f['task'] == k2 and f['contact'])+around; seconds = CLIP_SECONDS
    pick = np.unique(np.linspace(first, last, int(seconds*D.POLICY['fps'])).round().astype(int)); film = [whole[i] for i in pick]
    D._FILM = (p, hd, film, D.POLICY['fps'])
    with ProcessPoolExecutor(WORKERS) as pool: checks = sorted(pool.map(D._check, range(len(film)), chunksize=32))
    bad = [(i, c) for i, c in checks if c]
    print('TRANSLATIONS check', len(film), 'frames,', len(bad), 'meeting something', sorted({x for _, c in bad for x in c}), flush=True)
    name = run_name+('_translation' if clip else '')
    if stills:
        from PIL import Image
        f = Film(p, hd, film); out = SCRATCH/'stills'/name; out.mkdir(parents=True, exist_ok=True)
        try:
            for i in np.linspace(0, len(film)-1, 12).round().astype(int):
                Image.fromarray(f.frame(i)[::-1]).save(out/f'{i:05d}.png')
        finally: f.close()
        print('TRANSLATIONS stills', out, flush=True); return out
    video = HERE/f'{name}.mp4'; parts = SCRATCH/f'segments_{name}'; parts.mkdir(parents=True, exist_ok=True)
    cuts = np.linspace(0, len(film), WORKERS+1).astype(int)
    with ProcessPoolExecutor(WORKERS) as pool:
        segments = list(pool.map(D._segment, [(a, b, parts/f'{k:02d}.mp4') for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])) if b > a]))
    (parts/'list.txt').write_text(''.join(f"file '{Path(s).name}'\n" for s in segments))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(parts/'list.txt'), '-c', 'copy', '-movflags', '+faststart', str(video)], check=True)
    for x in parts.iterdir(): x.unlink()
    parts.rmdir()
    print('TRANSLATIONS VIDEO', video, round(len(film)/D.POLICY['fps'], 1), 's,', len(failing), 'failing motions,', len(bad), 'frames meeting something,',
          round(time.monotonic()-began), 's', flush=True)
    return video


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('run'); a.add_argument('--result', default=RESULT); a.add_argument('--stills', action='store_true')
    a.add_argument('--clip', help='only the support moved from this area\'s (W01, ...) first translation to its second, CLIP_SECONDS long')
    a = a.parse_args(); run(a.run, a.result, a.stills, a.clip)
