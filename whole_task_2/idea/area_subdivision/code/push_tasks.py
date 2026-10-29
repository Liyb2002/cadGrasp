"""A schematic, physics left out, in the manner of idea/easy_task/push_twist_half.mp4: one area of the Step2 figure
(big_01's W01: one needle direction) worked as TASKS tasks of equal area, the Franka Panda pushing the support (the
bunny in it) along the table between them; no text, about half a minute.

Where the support stands decides what the needle reaches: on the table the support at shift D (horizontal) from W01's
Step2 pose (its lowest point on the table) puts a face on the needle's nominal line (the station's direction through
the area's centre) for one D per face.  The faces are cut in TASKS strips of equal area along the main direction of
their D, ordered so that every push goes the same way (towards the work arm); a task's support stands at its strip's
mean D.  For a task the KUKA (the station's, at its base) holds the needle along the station's direction and moves it
across, rows of ROW m, so that its line passes over the strip's faces; the tip stays CLEAR beyond everything, the
needle never touches: the face its line meets first is worked (a gold dashed line from the tip to it; deep pink: the
faces within BRUSH of where it meets, and at the end of a task the rest of its strip).  The tasks' borders are drawn
dark on the light pink area.  Between tasks the needle backs off BACK along itself and the Panda (demo/code/franka.py:
MuJoCo Menagerie's model, its IK) comes down behind the support, its closed fingers level, pushes it with their ends
to the next task's place and leaves.  Every frame is checked: the needle and head, the KUKA's links and the Panda's (its fingers may touch the
support only while pushing) against dense samples of the bunny and the support, the Panda against the table.

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/area_subdivision/code/push_tasks.py
  ... --still 5 12     frames at 5 s and 12 s, as png in the scratch folder
The video goes to idea/area_subdivision/<object>_<area>_push.mp4 (big_01_W01_push.mp4).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation, Slerp

from whole_task_2.step1.common import ROOT, load_shape, object_of
from whole_task_2.step1.reach_video import Scene, tool_frame, kuka, arm_links, GOLD, LIGHT
from whole_task_2.step1.find_tasks_figure import OUTLINE_COLOR, OUTLINE_WIDTH_PX, dashed
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step5.video.prepare import matrices
from whole_task_2.step5.joined_video import track
from whole_task_2.idea.area_subdivision.code.step2_cut import borders, CHUNK
sys.path.insert(0, str(ROOT/'whole_task_2/idea/demo/code'))
import franka as F

HERE = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get('PUSH_TASKS_SCRATCH', '/tmp/claude-1000/-home-yli581-Desktop-cadGrasp/093a55e2-7a8c-4226-abf3-06a98066f6d2/scratchpad/push_tasks'))
RUN, AREA = 'B+big_01', 'W01'
TASKS = 3               # tasks of equal area
ROW = .012              # m: the line's rows across a strip (on the table plane)
BRUSH = .009            # m: the faces this close to where the line meets the bunny are worked
SPEED = .11             # m/s: the line across the table plane while working
CLEAR = .02             # m: the needle tip beyond everything, along the needle
BACK = .05              # m: the needle backs off along itself between tasks
PUSH_SPEED = .05        # m/s: the support pushed
TCP_HEIGHT = .05        # m: the Panda's fingers' pad centre above the table while pushing (its hand +-32 mm about it)
GAP, AHEAD = .004, .011 # m: the closed fingers' gap; their ends this far ahead of the pad centre (and 1 mm)
PRE, ABOVE = .04, .09   # m: the hand stands off the support before and after a push, and comes from above
FRANKA = (.55, .6, .65, .7)          # m: the Panda's base from the support (tried in order) ...
AZIMUTH = (0., 20., -20., 40., -40.) # deg: ... behind the pushes, turned this far (tried in order)
TIMES = dict(start=.6, approach=.45, retreat=.35, come=.9, down=.3, touch=.3, back=.3, up=.3, leave=.9, end=1.2)   # s
FPS = 30
SIZE = (1600, 900)
TURN = 30.              # deg: the camera's azimuth from square to the work arm's reach, towards it
RISE = .8               # the camera's height (before normalising)
SUB = 4                 # line samples per frame
DASH = (.006, .004)     # m: the gold line's dashes and gaps


def unit(v):
    v = np.asarray(v, float); return v/np.linalg.norm(v)


def frame(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p; return T


def shift(v):
    T = np.eye(4); T[:2, 3] = v[:2]; return T


def ease(u):
    return u*u*(3-2*u)


def blend(A, B, u):
    R = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))(u).as_matrix(); return frame(R, (1-u)*A[:3, 3]+u*B[:3, 3])


def steps(seconds):
    n = max(1, round(seconds*FPS)); return np.arange(1, n+1)/n


# ---------------------------------------------------------------------------------------------
# The area, its tasks and their places.

def area(run, name):
    """The station (Step2's setup), the bunny (object frame), the support (object frame), W0 (world <- object: the area's
    Step2 pose, the support's lowest point on the table), the area's faces."""
    plan = json.loads((ROOT/'whole_task_2/data'/run/'step1_reach/plan.json').read_text()); s = plan['setup']
    row = next(a for a in plan['areas'] if a['id'] == name); source = load_shape(ROOT/'objects'/object_of(run)/'mesh.stl')
    P = frame(np.asarray(row['placement']['rotation_world_object']), np.asarray(row['placement']['translation_world_m']))
    with np.load(ROOT/'whole_task_2/data'/run/'step2_seats/supports.npz') as z: V, Fs = z[name+'_vertices'], z[name+'_faces']
    support = trimesh.Trimesh((V-P[:3, 3])@P[:3, :3], Fs, process=False)                 # world at P -> object frame
    P[2, 3] += s['table_height_m']-float((V[:, 2]).min())
    return s, source, support, P, np.asarray(row['face_ids'], int)


def tasks(source, W0, faces, a, toward):
    """Each face's shift D (the support's horizontal move from W0 that puts its centre on the nominal line, through the
    area's centre along a); the strips of equal area along D's main direction (signed towards `toward`, so each push
    goes that way): (label per face, each task's D, the faces' D, the line's point L0, the main directions)."""
    C = source.triangles_center[faces]@W0[:3, :3].T+W0[:3, 3]; A = source.area_faces[faces]
    L0 = (A[:, None]*C).sum(0)/A.sum(); D = (L0+((C[:, 2]-L0[2])/a[2])[:, None]*a-C)[:, :2]
    c = D-(A[:, None]*D).sum(0)/A.sum(); axes = np.linalg.svd(c*np.sqrt(A)[:, None], full_matrices=False)[2]
    if axes[0]@toward < 0: axes[0] *= -1
    u = c@axes[0]; order = np.argsort(u); share = np.cumsum(A[order])/A.sum()
    label = np.empty(len(faces), int); label[order] = np.minimum((share*TASKS).astype(int), TASKS-1)
    places = np.asarray([(A[label == k, None]*D[label == k]).sum(0)/A[label == k].sum() for k in range(TASKS)])
    return label, places, D, L0, axes


def rows(r, axes):
    """The line's path over a strip (residual shifts r of its faces from the task's place): rows along the second main
    direction ROW apart across the first, each over its faces' span, alternating."""
    u, v = r@axes[0], r@axes[1]; out = []
    for i, x in enumerate(np.arange(u.min()+ROW/2, u.max()+ROW/2, ROW)):
        near = np.abs(u-x) <= ROW/2+1e-9
        if not near.any(): continue
        lo, hi = v[near].min(), v[near].max(); ends = [lo, hi] if len(out) % 2 == 0 else [hi, lo]
        out += [x*axes[0]+y*axes[1] for y in ends]
    return np.asarray(out)


# ---------------------------------------------------------------------------------------------
# The film's motion.

class Motion:
    def __init__(self, run=RUN, name=AREA):
        s, source, support, W0, faces = area(run, name); self.s, self.source, self.support, self.faces = s, source, support, faces
        a = self.a = unit(s['working_direction_world']); base = np.asarray(s['base_position_world_m'], float)
        self.label, self.places, D, self.L0, self.axes = tasks(source, W0, faces, a, (base-W0[:3, 3])[:2])
        self.W = [shift(p)@W0 for p in self.places]; self.D = D
        # The tip: CLEAR beyond the bunny and the support at every place, the line moved as far as any row takes it.
        V = np.vstack([source.vertices, support.vertices]); far = max(float(((V@W[:3, :3].T+W[:3, 3]-self.L0)@-a).max()) for W in self.W)
        r = np.linalg.norm(D-self.places[self.label], axis=1).max(); self.tip0 = self.L0-a*(far+r*np.linalg.norm(a[:2])+CLEAR)
        self.arm = kuka(s); self.mesh = trimesh.util.concatenate([source, support]); self.nb = len(source.faces)
        self.L, self.H = s['needle_length_m'], s['head_radius_m']

    def tip(self, r, back=0.):
        return self.tip0-np.r_[r, 0.]-self.a*back

    def meet(self, tip, W):
        """Where the line from tip first meets the bunny (object frame), or None."""
        o = W[:3, :3].T@(tip-W[:3, 3]); v = W[:3, :3].T@self.a
        idx, _, loc = self.mesh.ray.intersects_id([o], [v], multiple_hits=False, return_locations=True)
        return loc[0] if len(idx) and idx[0] < self.nb else None

    def pushes(self):
        """For each push: (direction, contact TCP frame, length): the hand level behind the support, its closed fingers
        pointing along the push, their ends on the support's farthest point there (at TCP_HEIGHT, near the middle)."""
        out = []; z = self.s['table_height_m']+TCP_HEIGHT; V = self.support.vertices
        for k in range(TASKS-1):
            d = np.r_[self.places[k+1]-self.places[k], 0.]; n = float(np.linalg.norm(d)); p = d/n; side = np.cross([0, 0, 1.], p)
            W = self.W[k]; Vw = V@W[:3, :3].T+W[:3, 3]; c = Vw.mean(0)
            band = (np.abs(Vw[:, 2]-z) < .02) & (np.abs((Vw-c)@side) < .025)
            x = float((Vw[band]@p).min()) if band.any() else float((Vw@p).min())
            T = frame(np.c_[np.cross(side, p), side, p], np.r_[(c+(x-c@p)*p)[:2], z]-p*AHEAD)
            out.append((p, T, n))
        return out

    def place_panda(self, pushes):
        """The Panda's base: behind the pushes, the first of FRANKA x AZIMUTH from which its IK reaches every hand pose."""
        back = -unit(np.mean([p for p, _, _ in pushes], axis=0)); c = np.mean([W[:3, 3] for W in self.W], axis=0)
        for dist in FRANKA:
            for az in AZIMUTH:
                u = Rotation.from_euler('z', az, degrees=True).as_matrix()@back; b = np.r_[(c+dist*u)[:2], self.s['table_height_m']]
                P = F.Panda(b, float(np.arctan2(*(c-b)[1::-1])))
                poses = [M for p, T, n in pushes for M in self.hand_path(p, T, n)]
                if all(P.solve(M, [P.ready]) is not None for M in poses): return P
        raise ValueError('No Panda base reaches every push')

    @staticmethod
    def hand_path(p, T, n):
        """Key TCP frames of a push: above, before, touching, pushed, backed off, above again."""
        def at(M, v): M = M.copy(); M[:3, 3] += v; return M
        pushed = at(T, p*n); return [at(T, -p*PRE+[0, 0, ABOVE]), at(T, -p*PRE), T, pushed, at(pushed, -p*PRE), at(pushed, -p*PRE+[0, 0, ABOVE])]

    def build(self):
        """Frames: the support's place W, the tip, the KUKA's joints, the Panda's joints and fingers, whether the line
        works, the fingers pushing."""
        pushes = self.pushes(); P = self.panda = self.place_panda(pushes); park = P.ready.copy(); frames = []
        paths = [rows(self.D[self.label == k]-self.places[k], self.axes) for k in range(TASKS)]
        q = arm_links(self.arm, self.tip(paths[0][0], BACK), self.a)[1]; h = park

        def kuka_at(tip, q):
            try: return track(self.arm, tip, self.a, q)
            except ValueError: return self.arm.solve(tip, -self.a, q)

        def add(W, tip, h, work=False, push=False, task=None, gap=F.PANDA['stroke']):
            nonlocal q; q = kuka_at(tip, q); frames.append(dict(W=W, tip=tip, q=q, h=h, gap=gap, work=work, push=push, task=task))
        for u in steps(TIMES['start']): add(self.W[0], self.tip(paths[0][0], BACK), h)
        for k in range(TASKS):
            path, W = paths[k], self.W[k]
            for u in steps(TIMES['approach']): add(W, self.tip(path[0], BACK*(1-ease(u))), h, task=k)
            for a, b in zip(path[:-1], path[1:]):
                for u in steps(np.linalg.norm(b-a)/SPEED): add(W, self.tip(a+(b-a)*u), h, work=True, task=k)
            frames[-1]['fill'] = k
            for u in steps(TIMES['retreat']): add(W, self.tip(path[-1], BACK*ease(u)), h, task=k)
            if k == TASKS-1: break
            p, T, n = pushes[k]; keys = self.hand_path(p, T, n); nxt = paths[k+1][0]
            above = P.solve(keys[0], [park]); moves = []          # (seconds, hand frames or joints, support moves, fingers)
            moves.append(('come', [(1-ease(u))*park+ease(u)*above for u in steps(TIMES['come'])], None, F.PANDA['stroke']))
            for name, (A, B) in zip(('down', 'touch'), zip(keys[:2], keys[1:3])): moves.append((name, [blend(A, B, ease(u)) for u in steps(TIMES[name])], None, GAP))
            moves.append(('push', [blend(keys[2], keys[3], u) for u in steps(n/PUSH_SPEED)], 'move', GAP))
            for name, (A, B) in zip(('back', 'up'), zip(keys[3:5], keys[4:6])): moves.append((name, [blend(A, B, ease(u)) for u in steps(TIMES[name])], None, GAP))
            moves.append(('leave', None, None, F.PANDA['stroke']))
            total = sum(len(m[1]) if m[1] is not None else len(steps(TIMES['leave'])) for m in moves); i = 0; Wk = W
            for name, seq, moving, gap in moves:
                if seq is None: last = h; seq = [(1-ease(u))*last+ease(u)*park for u in steps(TIMES['leave'])]
                for j, x in enumerate(seq):
                    if x.shape == (4, 4):
                        rest = h.copy(); rest[[2, 4]] *= .5; h2 = P.solve(x, [h], rest=rest)
                        if h2 is None: raise ValueError(f'push {k} {name}: no IK for the hand at {x[:3, 3].round(3)}')
                        h = h2
                    else: h = x
                    if moving: Wk = shift(p[:2]*n*(j+1)/len(seq))@W
                    i += 1; w = i/total                                   # the needle drifts to the next task's start, backed off
                    add(Wk, (1-w)*self.tip(path[-1], BACK)+w*self.tip(nxt, BACK), h, push=name == 'push', task=None, gap=gap)
        for u in steps(TIMES['end']): add(frames[-1]['W'], frames[-1]['tip'], h, task=None)
        self.frames = frames; return frames

    def paint(self):
        """Per frame, the faces worked so far, and where the line meets (object frame, or None)."""
        done = np.zeros(len(self.source.faces), bool); area_ = np.zeros(len(self.source.faces), bool); area_[self.faces] = True
        centres = self.source.triangles_center; out, hits = [], []
        for k, f in enumerate(self.frames):
            prev = self.frames[max(k-1, 0)]
            if f['work']:
                for u in np.arange(1, SUB+1)/SUB:
                    m = self.meet((1-u)*prev['tip']+u*f['tip'], f['W'])
                    if m is not None: done |= area_ & (np.linalg.norm(centres-m, axis=1) < BRUSH)
            if 'fill' in f: done[self.faces[self.label == f['fill']]] = True
            out.append(done.copy()); hits.append(self.meet(f['tip'], f['W']))
        return out, hits

    def check(self):
        """Frames (every third) where the needle or head, the KUKA's links or the Panda's meet the bunny or the support
        (dense samples), the Panda's fingers the support but while pushing, or the Panda the table."""
        from whole_task_2.idea.easy_task.code.slanted import inside
        sb = trimesh.sample.sample_surface_even(self.source, 20000, seed=0)[0]; ss = trimesh.sample.sample_surface_even(self.support, 20000, seed=0)[0]
        hulls = [Delaunay(self.arm.collisions[i]) for i in range(1, 8)]; P = self.panda; bad = dict(tool=[], kuka=[], panda=[], table=[])
        for k in range(0, len(self.frames), 3):
            f = self.frames[k]; W = f['W']; B, S = sb@W[:3, :3].T+W[:3, 3], ss@W[:3, :3].T+W[:3, 3]; both = np.vstack([B, S])
            t = f['tip']; a = self.a
            if inside(both, t, t-self.L*a, self.s['needle_radius_m']).any() or inside(both, t-self.L*a, t-(self.L+.07)*a, self.H).any(): bad['tool'].append(k)
            links = matrices(self.arm.forward(f['q'], geometry=True)[2])
            if any(h.find_simplex((both-links[i+1][:3, 3])@links[i+1][:3, :3]).max() >= 0 for i, h in enumerate(hulls)): bad['kuka'].append(k)
            Fp = P.frames(f['h'], f['gap']); moving = [i for i in range(len(Fp)) if P.level[i] > 0]
            others = [i for i in moving if i not in P.fingers]
            if (F.pierced([P.pieces[i] for i in others], Fp[others], F.Cloud(both)) or
                    F.pierced([P.pieces[i] for i in P.fingers], Fp[P.fingers], F.Cloud(B if f['push'] else both))): bad['panda'].append(k)
            if min((P.hulls[i]@Fp[i][:3, :3].T+Fp[i][:3, 3])[:, 2].min() for i in moving) < self.s['table_height_m']: bad['table'].append(k)
        return bad


# ---------------------------------------------------------------------------------------------

def main(stills=(), run=RUN, name=AREA):
    m = Motion(run, name); frames = m.build(); painted, hits = m.paint(); bad = m.check()
    print('PUSH TASKS', run, name, len(frames), 'frames', round(len(frames)/FPS, 1), 's;', TASKS, 'tasks of',
          [int((m.label == k).sum()) for k in range(TASKS)], 'faces; places', np.round(m.places, 3).tolist(),
          '; worked', round(float(m.source.area_faces[painted[-1]].sum()/m.source.area_faces[m.faces].sum()), 3),
          '; frames meeting something', {k: len(v) for k, v in bad.items()}, {k: v[:6] for k, v in bad.items() if v}, flush=True)
    s, a, P, src = m.s, m.a, m.panda, m.source
    base = np.asarray(s['base_position_world_m'], float); reach = unit((m.L0-base)[:2])
    camera = np.degrees(np.arctan2(-reach[1], -reach[0]))-90+TURN
    view = unit(np.r_[np.cos(np.radians(camera)), np.sin(np.radians(camera)), RISE])
    V = np.vstack([src.vertices[::5], m.support.vertices[::20]])
    hands = [P.tcp_frame(f['h']) for f in frames[::10] if f['push']]                # the fingers and the hand behind them
    fit = np.vstack([*(V@W[:3, :3].T+W[:3, 3] for W in m.W), *(f['tip'] for f in frames[::10]), *(f['tip']-(m.L+.07)*a for f in frames[::10]),
                     *(T[:3, 3] for T in hands), *(T[:3, 3]-.2*T[:3, 2] for T in hands)])
    lo, hi = fit[:, :2].min(0)-.25, fit[:, :2].max(0)+.25; z = s['table_height_m']
    table = trimesh.creation.box(bounds=[[*lo, z-.02], [*hi, z]])
    label = np.full(len(src.faces), -1); label[m.faces] = m.label; edges = borders(src, label)
    scene = Scene(src, s, *SIZE, arm=True)
    try:
        g = scene.r; scene.view = view; scene.fit(fit, 1.08); scene.ctx.line_width = 1.6*OUTLINE_WIDTH_PX*scene.studio.size[0]/SIZE[0]
        table_shape = g.Shape(scene.ctx, scene.shader, table.vertices, table.faces, (.80, .81, .83), smooth=False, roughness=.85)
        support_shape = g.Shape(scene.ctx, scene.shader, m.support.vertices, m.support.faces, BLUE, metallic=.05, roughness=.5)
        panda_shapes = [g.Shape(scene.ctx, scene.shader, v, f, c, metallic=.15, roughness=.45) for v, f, c in P.visuals()]

        def picture(k):
            f = frames[k]; W = f['W']
            colors = np.tile(g.GRAY, (len(src.faces), 1)); colors[m.faces] = LIGHT; colors[painted[k]] = g.PINK
            shapes = [(table_shape, np.eye(4)), (support_shape, W)]+[(part, tool_frame(f['tip'], a)) for part in scene.tools[False]]
            shapes += list(zip(panda_shapes, P.visual_frames(f['h'], f['gap'])))
            e = edges@W[:3, :3].T+W[:3, 3]; segments = [(e[i:i+CHUNK], OUTLINE_COLOR) for i in range(0, len(e), CHUNK)]
            hit = hits[k]; end = W[:3, :3]@hit+W[:3, 3] if hit is not None else f['tip']+.25*a
            segments.append((np.asarray(dashed([(f['tip'], end)], *DASH)).reshape(-1, 3), GOLD))
            links = matrices(m.arm.forward(f['q'], geometry=True)[2])
            return scene.draw(W, colors, [], links=links, shapes=shapes, segments=segments)
        if stills:
            SCRATCH.mkdir(parents=True, exist_ok=True)
            for t in stills:
                k = min(int(round(t*FPS)), len(frames)-1); path = SCRATCH/f'still_{t:05.1f}.png'
                Image.frombytes('RGB', SIZE, picture(k)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(path); print('STILL', path, flush=True)
            return
        path = HERE/f"{run.removeprefix(object_of(run)+'+')}_{name}_push.mp4"
        ffmpeg = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(FPS),
                                   '-i', '-', '-vf', 'vflip', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-an', str(path)],
                                  stdin=subprocess.PIPE)
        try:
            for k in range(len(frames)): ffmpeg.stdin.write(picture(k))
        finally:
            ffmpeg.stdin.close(); ffmpeg.wait()
        print('VIDEO', path, f'{len(frames)/FPS:.1f} s', flush=True)
    finally:
        scene.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--still', type=float, nargs='*', default=())
    main(p.parse_args().still)
