"""idea/easy_task/push_twist_half.mp4's task with the needle free to move: the same bunny, pedestal, fins, station
and needle direction (SLOPE down from the KUKA's reach), the same faces worked -- but the KUKA moves the needle
(its direction fixed) across what the line reaches, so the support stays in one place and is only turned a few times;
no text.  A schematic: the needle never touches -- the face its line meets first is worked (a gold dashed line from the
tip to it; deep pink: the faces within BRUSH of where it meets).

The faces: push_twist_half's, those its motion works (its timeline replayed, nothing rendered).  The turns: the fewest
equally spaced turns of the support (its axis where push_twist's line crosses it at mid height, never moved) at which
the line, moved anywhere across the needle, meets every one of them first (within BRUSH/2 of its centre), the first
the nearest to push_twist's start; each face is worked at the first turn that reaches it.  At a turn the tip moves in
the plane across the needle CLEAR beyond the bunny and the support, in rows ROW apart over that turn's faces; at the
end of the turn its faces are done.  Between turns the needle backs off BACK along itself and the Panda (as in
push_twist: it grips a fin from outside and above) turns the support about its axis, in grips of at most GRIP degrees,
and goes back to rest.  Every frame is checked as push_twist checks it: the needle and head, the KUKA's links and the
Panda's (its fingers only against the bunny) against dense samples of the bunny and the support.

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/area_subdivision/code/moving_needle.py
  ... --still 3 9      frames at 3 s and 9 s, as png in the scratch folder
The video goes to idea/area_subdivision/moving_needle.mp4.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import Delaunay

from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_video import Scene, tool_frame, GOLD, LIGHT
from whole_task_2.step1.find_tasks_figure import OUTLINE_WIDTH_PX, dashed
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step5.video.prepare import matrices
from whole_task_2.step5.joined_video import track
sys.path[:0] = [str(ROOT/'whole_task_2/idea/easy_task/code'), str(ROOT/'whole_task_2/idea/demo/code')]
import push_twist as PT
import franka as F
from slanted import holder, inside

HERE = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get('MOVING_NEEDLE_SCRATCH', '/tmp/claude-1000/-home-yli581-Desktop-cadGrasp/093a55e2-7a8c-4226-abf3-06a98066f6d2/scratchpad/moving_needle'))
BRUSH, CLEAR, FPS, SIZE, DASH, SUB = PT.BRUSH, PT.CLEAR, PT.FPS, PT.SIZE, PT.DASH, PT.SUB
ROW = PT.BRUSH          # m: the line's rows across the needle
SPEED = .06             # m/s: the tip across the needle while working
BACK = .04              # m: the needle backs off along itself while the support turns
GRIP = 120.             # deg: the most the Panda turns the support in one grip
TIMES = dict(PT.TIMES, start=.6, approach=.4, retreat=.3, twist=1.2, end=1.2)   # s (twist: per 90 deg)
EVERY = 15.             # deg: the turns tried


def steps(seconds):
    n = max(1, round(seconds*FPS)); return np.arange(1, n+1)/n


def unit(v):
    v = np.asarray(v, float); return v/np.linalg.norm(v)


class Scene_:
    """push_twist's scene and its worked faces."""
    def __init__(self):
        self.source = load_shape(ROOT/'objects/B/mesh.stl'); self.setup, self.reach, self.d = PT.station()
        self.B, self.bunny, self.support, self.grips, self.rim = PT.bunny_and_support(self.source)
        tip = np.r_[0., 0., self.bunny.bounds[1, 2]+CLEAR]; self.line = PT.Line(self.bunny, self.support, tip, self.d, self.reach)
        self.lo, self.hi = self.line.band(self.bunny.bounds[1, 2]); self.zc = float(np.mean([self.lo, self.hi]))
        mean = self.line.place(0., self.zc)[:3, 3]; side = PT.spin(PT.SIDE)[:2, :2]@self.reach
        self.panda = F.Panda(np.r_[mean[:2]+PT.FRANKA*side, 0.], np.arctan2(-side[1], -side[0])); self.q0 = self.panda.ready.copy()
        self.work = (self.bunny.face_normals[:, 2] > 0) & (self.bunny.triangles_center[:, 2] > self.rim)

    def reference(self):
        """The faces push_twist_half's motion works (its timeline and its painting, replayed)."""
        frames = PT.timeline(self.line, self.lo, self.hi, self.grips, self.panda, self.q0); done = np.zeros(len(self.bunny.faces), bool)
        C = self.bunny.triangles_center
        for k, f in enumerate(frames):
            prev = frames[max(k-1, 0)]; us = np.arange(1, SUB+1)/SUB
            point, _, ok = self.line.meet(prev['psi']+(f['psi']-prev['psi'])*us, prev['z']+(f['z']-prev['z'])*us)
            for p in point[ok]: done |= self.work & (np.linalg.norm(C-p, axis=1) < BRUSH)
        return done

    def place(self, psi):
        return self.line.place(psi, self.zc)

    def reached(self, psi, ids):
        """Which faces (ids) the line along d meets first, within BRUSH/2 of their centre, the support turned psi."""
        S = self.place(psi); ds = S[:3, :3].T@self.d; C = self.bunny.triangles_center[ids]
        idx, ray, loc = self.line.mesh.ray.intersects_id(C-.5*ds, np.tile(ds, (len(ids), 1)), multiple_hits=False, return_locations=True)
        ok = np.zeros(len(ids), bool); good = (idx < self.line.nb) & (np.linalg.norm(loc-C[ray], axis=1) < .5*BRUSH); ok[ray[good]] = True
        return ok

    def meet(self, tip, psi):
        """Where the line from tip first meets the bunny (support frame), or None."""
        S = self.place(psi); o = S[:3, :3].T@(tip-S[:3, 3]); v = S[:3, :3].T@self.d
        idx, _, loc = self.line.mesh.ray.intersects_id([o], [v], multiple_hits=False, return_locations=True)
        return loc[0] if len(idx) and idx[0] < self.line.nb else None


def turns(sc, target):
    """The fewest equally spaced turns that reach every target face, the first nearest push_twist's start (FACE), in
    increasing order; each target face's turn (the first that reaches it)."""
    ids = np.flatnonzero(target); reach = {a: sc.reached(a, ids) for a in np.arange(0., 360., EVERY)}
    for n in range(1, 9):
        starts = [s for s in np.arange(0., 360./n, EVERY) if np.all(np.any([reach[(s+360.*j/n) % 360] for j in range(n)], axis=0))]
        if starts: break
    else: raise ValueError('No equally spaced turns reach every face')
    psis = sorted([(s+360.*j/n) % 360 for s in starts[:1] for j in range(n)], key=lambda a: (a-PT.FACE) % 360)
    first = min(psis, key=lambda a: min((a-PT.FACE) % 360, (PT.FACE-a) % 360)); i = psis.index(first)
    order = psis[i:]+psis[:i]; psis = [order[0]+((a-order[0]) % 360) for a in order]          # increasing from the first
    owner = np.full(len(sc.bunny.faces), -1); left = target.copy()
    for k, a in enumerate(psis):
        hit = np.zeros(len(owner), bool); hit[ids[reach[a % 360]]] = True; owner[hit & left] = k; left &= ~hit
    return psis, owner


def build(sc, psis, owner):
    """Frames: the support's turn psi, the tip, the Panda's TCP (None: a joint-space move `free`), its fingers, the fin
    held, whether the line works; the turn's faces done at its last working frame (`fill`)."""
    d = sc.d; e2 = unit(np.cross([0, 0, 1.], d)); e1 = np.cross(d, e2); p0 = np.r_[sc.place(0.)[:2, 3], sc.zc]
    V = np.vstack([sc.bunny.vertices, sc.support.vertices])
    s0 = max(float(((V@sc.place(a)[:3, :3].T+sc.place(a)[:3, 3]-p0)@-d).max()) for a in psis)+CLEAR
    tip = lambda u, v, back=0.: p0-(s0+back)*d+u*e1+v*e2
    paths = []
    for k, a in enumerate(psis):
        S = sc.place(a); C = sc.bunny.triangles_center[owner == k]@S[:3, :3].T+S[:3, 3]; u, v = (C-p0)@e1, (C-p0)@e2; pts = []
        for x in np.arange(u.min()+ROW/2, u.max()+ROW/2, ROW):
            near = np.abs(u-x) <= ROW/2+1e-9
            if near.any(): ends = [v[near].min(), v[near].max()]; pts += [(x, y) for y in (ends if len(pts) % 4 == 0 else ends[::-1])]
        paths.append(np.asarray(pts))
    az = np.degrees(np.arctan2(*(sc.panda.base[:2]-sc.line.tip[:2])[::-1]))

    def fin(psi, span):
        want = az-span/2; return int(np.argmin([abs((np.degrees(np.arctan2(g[1, 3], g[0, 3]))+psi-want+180) % 360-180) for g in sc.grips]))

    def hand(psi, j, back=0.):
        T = sc.place(psi)@sc.grips[j]; T[:3, 3] -= back*T[:3, 2]; return T
    frames = []; OPEN, SHUT = PT.OPEN, PT.SHUT

    def add(psi, t, T=None, free=None, gap=OPEN, held=None, work=False):
        frames.append(dict(psi=psi, tip=t, T=T, free=free, gap=gap, held=held, work=work))
    n = lambda key: steps(TIMES[key])
    first = tip(*paths[0][0], BACK)
    for u in n('start'): add(psis[0], first, free=(None, 1.))
    for k, a in enumerate(psis):
        path = paths[k]
        for u in n('approach'): add(a, tip(*path[0], BACK*(1-PT.ease(u))), free=(None, 1.))
        for p, q in zip(path[:-1], path[1:]):
            for u in steps(np.linalg.norm(q-p)/SPEED): add(a, tip(*(p+(q-p)*u)), free=(None, 1.), work=True)
        frames[-1]['fill'] = k
        for u in n('retreat'): add(a, tip(*path[-1], BACK*PT.ease(u)), free=(None, 1.))
        if k == len(psis)-1: break
        # the turn to the next: grips of at most GRIP degrees, the needle drifting to the next start, backed off
        span = psis[k+1]-a; g = int(np.ceil(span/GRIP-1e-9)); cuts = a+span*np.arange(g+1)/g; seq = []
        for i in range(g):
            b0, b1 = cuts[i], cuts[i+1]; j = fin(b0, b1-b0); above, grip = hand(b0, j, PT.BACK), hand(b0, j)
            if i == 0: seq += [dict(psi=b0, free=(above, PT.ease(u)), gap=OPEN) for u in n('come')]
            else:
                c = sc.place(b0)[:3, 3]; seq += [dict(psi=b0, T=PT.swing(up, above, c, PT.ease(u)), gap=OPEN) for u in n('move')]
            seq += [dict(psi=b0, T=PT.blend(above, grip, PT.ease(u)), gap=OPEN) for u in n('down')]
            seq += [dict(psi=b0, T=grip, gap=OPEN+(SHUT-OPEN)*u) for u in n('close')]
            seq += [dict(psi=b0+(b1-b0)*PT.ramp(u), T=hand(b0+(b1-b0)*PT.ramp(u), j), gap=SHUT, held=j) for u in steps(TIMES['twist']*(b1-b0)/90.)]
            T = seq[-1]['T']; up = hand(b1, j, PT.BACK)
            seq += [dict(psi=b1, T=T, gap=SHUT+(OPEN-SHUT)*u) for u in n('open')]
            seq += [dict(psi=b1, T=PT.blend(T, up, PT.ease(u)), gap=OPEN) for u in n('up')]
        seq += [dict(psi=psis[k+1], free=(None, PT.ease(u)), gap=OPEN) for u in n('leave')]
        A, Bt = tip(*path[-1], BACK), tip(*paths[k+1][0], BACK)
        for i, f in enumerate(seq):
            w = PT.ease((i+1)/len(seq)); add(f['psi'], (1-w)*A+w*Bt, f.get('T'), f.get('free'), f['gap'], f.get('held'))
    for u in n('end'): add(frames[-1]['psi'], frames[-1]['tip'], free=(None, 1.))
    return frames, e1, e2, p0, s0


def panda_joints(sc, frames):
    """The Panda's joints: a joint-space move from where it is to an approach (its IK nearest the arm's plain posture)
    or back to rest (free = (target or None, share)); on the TCP's path each frame from the last."""
    P, q0 = sc.panda, sc.q0

    def roomy(T):
        found = [p for p in (P.solve(T, [s]) for s in [q0, *np.random.default_rng(0).uniform(P.lower, P.upper, (200, 7))]) if p is not None]
        if not found: raise ValueError(f'no IK for the TCP at {T[:3, 3].round(3)}')
        return min(found, key=lambda p: abs(p[2])+abs(p[4])+abs(p[6]-q0[6]))
    q = q0.copy(); out = []; seg = None
    for k, f in enumerate(frames):
        if f['T'] is None:
            target, u = f['free']; key = (None if target is None else target.tobytes())
            if seg is None or seg[0] != key or (k and frames[k-1]['T'] is not None): seg = (key, q, q0 if target is None else roomy(target))
            p = seg[1]+(seg[2]-seg[1])*u
        else:
            seg = None; rest = q.copy(); rest[[2, 4]] *= .5; p = P.solve(f['T'], [q], rest=rest)
            if p is None: raise ValueError(f'frame {k}: no IK for the TCP at {f["T"][:3, 3].round(3)}')
        out.append(p); q = p
    return out


def main(stills=()):
    sc = Scene_(); target = sc.reference(); psis, owner = turns(sc, target)
    frames, *_ = build(sc, psis, owner); Q = panda_joints(sc, frames)
    d, setup = sc.d, sc.setup; L, H, N = setup['needle_length_m'], setup['head_radius_m'], setup['needle_radius_m']
    tips = np.asarray([f['tip'] for f in frames]); arm, _, q, slide = holder(setup, tips.mean(0), d); K = []
    for t in tips:
        try: q = track(arm, t, d, q)
        except ValueError: q = arm.solve(t, -d, q)
        K.append(q)
    # The work, the line every SUB-th of a frame.
    done = np.zeros(len(sc.bunny.faces), bool); C = sc.bunny.triangles_center; painted, hits = [], []
    for k, f in enumerate(frames):
        prev = frames[max(k-1, 0)]
        if f['work']:
            for u in np.arange(1, SUB+1)/SUB:
                m = sc.meet((1-u)*prev['tip']+u*f['tip'], f['psi'])
                if m is not None: done |= target & (np.linalg.norm(C-m, axis=1) < BRUSH)
        if 'fill' in f: done[owner == f['fill']] = True
        painted.append(done.copy()); hits.append(sc.meet(f['tip'], f['psi']))
    # Checks, as push_twist's.
    sb = trimesh.sample.sample_surface_even(sc.bunny, 30000, seed=0)[0]; ss = trimesh.sample.sample_surface_even(sc.support, 30000, seed=0)[0]
    hulls = [Delaunay(arm.collisions[i]) for i in range(1, 8)]; P = sc.panda; bad = dict(tool=[], kuka=[], panda=[])
    for k in range(0, len(frames), 3):
        f = frames[k]; S = sc.place(f['psi']); pts = np.vstack([sb, ss])@S[:3, :3].T+S[:3, 3]; t = f['tip']
        if inside(pts, t, t-L*d, N).any() or inside(pts, t-L*d, t-(L+.07)*d, H).any(): bad['tool'].append(k)
        links = matrices(arm.forward(K[k], geometry=True)[2])
        if any(h.find_simplex((pts-links[i+1][:3, 3])@links[i+1][:3, :3]).max() >= 0 for i, h in enumerate(hulls)): bad['kuka'].append(k)
        Fp = P.frames(Q[k], f['gap']); moving = [i for i in range(len(Fp)) if P.level[i] > 0]; others = [i for i in moving if i not in P.fingers]
        if (F.pierced([P.pieces[i] for i in others], Fp[others], F.Cloud(pts))
                or F.pierced([P.pieces[i] for i in P.fingers], Fp[P.fingers], F.Cloud(sb@S[:3, :3].T+S[:3, 3]))): bad['panda'].append(k)
    A = sc.bunny.area_faces; grips = sum(1 for k in range(1, len(frames)) if frames[k]['held'] is not None and frames[k-1]['held'] is None)
    print('MOVING NEEDLE', len(frames), 'frames', round(len(frames)/FPS, 1), 's; turns', [round(a % 360) for a in psis], 'grips', grips,
          '; faces per turn', [int((owner == k).sum()) for k in range(len(psis))], '; worked', int(done.sum()), 'of', int(target.sum()),
          'push_twist_half faces, same set', bool(np.array_equal(done, target)), round(float(A[done].sum()/A[sc.work].sum()), 4), 'of the work area;',
          'KUKA slid', slide, 'm; frames meeting something', {k: len(v) for k, v in bad.items()}, {k: v[:6] for k, v in bad.items() if v}, flush=True)

    # The film: push_twist's camera, table and drawing.
    reach_az = np.degrees(np.arctan2(-sc.reach[1], -sc.reach[0])); camera = reach_az-90+PT.TURN
    view = np.r_[np.cos(np.radians(camera)), np.sin(np.radians(camera)), PT.RISE]; view /= np.linalg.norm(view)
    span = [sc.place(f['psi']) for f in frames[::10]]
    fit = np.vstack([*(np.vstack([sc.bunny.vertices[::5], sc.support.vertices[::20]])@S[:3, :3].T+S[:3, 3] for S in span),
                     *(f['tip'] for f in frames[::10]), *(f['tip']-(L+.07)*d+[0, 0, .05] for f in frames[::10]),
                     *(f['T'][:3, 3]+[0, 0, .12] for f in frames[::10] if f['T'] is not None)])
    table = trimesh.creation.box(bounds=[[*(fit[:, :2].min(0)-.25), -.02], [*(fit[:, :2].max(0)+.25), 0.]])
    src = sc.source; scene = Scene(src, setup, *SIZE, arm=True)
    try:
        g = scene.r; scene.view = view; scene.fit(fit, 1.15); scene.ctx.line_width = 1.6*OUTLINE_WIDTH_PX*scene.studio.size[0]/SIZE[0]
        table_shape = g.Shape(scene.ctx, scene.shader, table.vertices, table.faces, (.80, .81, .83), smooth=False, roughness=.85)
        support_shape = g.Shape(scene.ctx, scene.shader, sc.support.vertices, sc.support.faces, BLUE, metallic=.05, roughness=.5)
        panda_shapes = [g.Shape(scene.ctx, scene.shader, v, f, c, metallic=.15, roughness=.45) for v, f, c in P.visuals()]

        def picture(k):
            f = frames[k]; S = sc.place(f['psi']); W = S@sc.B
            colors = np.tile(g.GRAY, (len(src.faces), 1)); colors[target] = LIGHT; colors[painted[k]] = g.PINK
            shapes = [(table_shape, np.eye(4)), (support_shape, S)]+[(part, tool_frame(f['tip'], d)) for part in scene.tools[False]]
            shapes += list(zip(panda_shapes, P.visual_frames(Q[k], f['gap'])))
            hit = hits[k]; end = S[:3, :3]@hit+S[:3, 3] if hit is not None else f['tip']+.3*d
            segments = [(np.asarray(dashed([(f['tip'], end)], *DASH)).reshape(-1, 3), GOLD)]
            return scene.draw(W, colors, [], links=matrices(arm.forward(K[k], geometry=True)[2]), shapes=shapes, segments=segments)
        if stills:
            SCRATCH.mkdir(parents=True, exist_ok=True)
            for t in stills:
                k = min(int(round(t*FPS)), len(frames)-1); path = SCRATCH/f'still_{t:05.1f}.png'
                Image.frombytes('RGB', SIZE, picture(k)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(path); print('STILL', path, flush=True)
            return
        path = HERE/'moving_needle.mp4'
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
