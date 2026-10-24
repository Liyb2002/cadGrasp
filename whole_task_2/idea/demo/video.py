"""Demo video: the whole process of B at its station, one camera close on the bunny, no text.

Run B+03_kuka, Step5 result r00: seat 0 holds +X, +Z and -Z (rotational reuse: the support turns on the
table between them), -X, +Y and -Y have a seat each (juxtaposition).  The motion is step5/joined_video.py's
Plan, every frame checked as there: the bunny enters seat 0 at the loading spot and the support slides to
the station; for each task the KUKA, rising and falling with the object on its column, moves the needle
over the task's certified faces (light pink: what the task still has to paint, deep pink: painted); between
the tasks of a seat the support turns on the table; between seats it slides to the loading spot, the bunny
leaves its seat along that task's certified exit and enters the next seat along the next task's exit.

The camera looks from one direction and follows the bunny (its box on screen, smoothed in time); the zoom
holds the bunny's box at OBJECT_SHARE of the picture, so the needle, the head and the KUKA's wrist show.

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/demo/video.py [--stills]
"""
import argparse
import hashlib
import inspect
import json
import pickle
import subprocess
import time
from pathlib import Path
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d

from whole_task_2.step1.common import ROOT, sha256, write_json
from whole_task_2.step1.reach_video import Scene, LIGHT
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step4.seats import context, read_mesh
from whole_task_2.step4.combine import ExitModel, seats
from whole_task_2.step5.combined import FOLDER, distinct
from whole_task_2.step5.video.prepare import matrices
from whole_task_2.step5 import joined_video as J

HERE = Path(__file__).resolve().parent
RUN, RESULT = 'B+03_kuka', 'r00'
OBJECT_SHARE = .5                                   # the bunny's box on screen, share of the picture
# How the needle leaves a stop: a straight lead back along it (m), then a direction tilted from straight back by
# tilt (deg), toward up turned about the needle by turn (deg); the first the arm and the needle can follow for
# retreat_m (else the one followed farthest), the needle checked from check_m on.
LEADS, TILTS, TURNS = (.012, .006, 0.), (45., 60., 75.), (0., 45., -45., 90., -90., 135., -135., 180.)
POLICY = dict(J.POLICY, retreats=[(l, t, u) for l in LEADS for t in TILTS for u in TURNS], retreat_m=.08, check_m=.004,
              back_off_needles=(.3, .45, .6, 1., 1.6, 2.4), jump_deg=10., width=1920, height=1080, view=(.95, -.55, .6), follow_seconds=.8, light_span=1.2, fit=.92, arm_room=.14)


class Model:
    """Step5's result as joined_video's Plan reads a model: the seats (X), every task's seat and exit."""
    def __init__(self, run=RUN, result=RESULT):
        c = context(run); folder = ROOT/'whole_task_2/data'/run/FOLDER
        report = json.loads((folder/'report.json').read_text())
        self.row = next(r for r in report['results'] if r['name'] == result and r['status'] == 'passed')
        group = {g['file']: g for g in distinct(run)}[self.row['step4_file']]
        self.path = folder/f'{result}.npz'; self.support = read_mesh(self.path)
        if sha256(self.path) != self.row['support_sha256']: raise ValueError('Changed Step5 support')
        base = ExitModel(c, group['slots']); self.layout = base.layout(self.support); rows = seats(group)
        self.X = [X for X, _ in rows]
        self.seat_of = [next(n for n, (_, ks) in enumerate(rows) if k in ks) for k in range(len(base.records))]
        self.source, self.setup, self.records = base.source, base.setup, base.records
        self.tolerance, self.policy = base.tolerance, base.policy
        self.exit = None; self.exits = np.asarray(self.layout.exits, float)        # world, at each task's installation


class Demo(J.Plan):
    def __init__(self, *args):
        super().__init__(*args)
        # Every certified joint set (the face centroid's), with its tip and axis in the arm's own frame.
        self.certified = np.vstack([np.asarray(r['joints'])[:, 3] for r in self.model.records])
        self.reached = np.asarray([self.arm.forward(q)[:2] for q in self.certified])

    def task_stops(self, k):
        """joined_video's stops, each with how the needle leaves it: the first of POLICY['retreats'] the arm can follow
        (in 2 mm steps from the stop's certified joints) for retreat_m, else the one it follows farthest, the needle
        (Step1's rule) and the head clear of the bunny and the support all along.  At some stops the elbow is at its
        limit and the needle cannot back off along its axis at all; at some a bump above blocks the way up."""
        stops = super().task_stops(k); r = self.model.records[k]; joints = np.asarray(r['joints']); P = np.asarray(r['pose'])
        bunny = J.placed(self.source, P)                                          # task k as certified: the arm's frame
        support = J.placed(self.support_fine, P@np.linalg.inv(self.model.X[self.model.seat_of[k]]))
        for i, s in enumerate(stops):
            a = s['axis']; up = np.array([0, 0, 1.])-a[2]*a; up /= np.linalg.norm(up); side = np.cross(a, up)
            best = (0., None, None)
            for lead, tilt, turn in self.policy['retreats']:
                b, c = np.radians(tilt), np.radians(turn); away = -a*np.cos(b)+(np.cos(c)*up+np.sin(c)*side)*np.sin(b)
                q = joints[s['j'], 3]; reach = 0.
                for t in np.arange(.002, self.policy['retreat_m']+1e-9, .002):
                    tip = s['tip']-a*min(t, lead)+away*max(t-lead, 0.); T = J.tool_frame(tip, a)
                    try: q = self.step(tip, a, q)
                    except ValueError: break
                    if t >= self.policy['check_m'] and (J.meets(self.needle(s['allowance']), [T, T, T], bunny) or J.meets(self.tool, [T, T], support)): break
                    reach = t
                if reach > best[0]: best = (reach, lead, away)
                if reach >= self.policy['retreat_m']-1e-9: break
            if best[0] < self.policy['check_m']: raise ValueError(f'No retreat from stop {k}:{i}')
            s.update(reach=best[0], lead=best[1], away=best[2])
        return stops

    def step(self, point, axis, q):
        """Joints with the tip on point (the arm's frame) along axis, near q: tracked, else Arm.solve from q (it keeps
        to the joint limits), a change of at most jump_deg."""
        try: return J.track(self.arm, point, axis, q)
        except ValueError: pass
        p = self.arm.solve(point, -np.asarray(axis), q)
        if np.degrees(np.abs(p-q).max()) > self.policy['jump_deg']: raise ValueError('Arm.solve jumps')
        return p

    def solve(self, tip, axis, off, previous):
        """joined_video's solve, tracking also in task-space steps; a jump starts from the certified joint sets
        nearest the tool pose, then from Arm.solve's seeds."""
        point = np.asarray(tip)-[0, 0, off]; axis = np.asarray(axis, float)
        if previous is not None:
            tcp, z = self.arm.forward(previous)[:2]
            for n in (1, 8, 32):
                try:
                    q = previous
                    for u in np.arange(1, n+1)/n: q = self.step((1-u)*tcp+u*point, J.slerp(z, axis, u), q)
                    return q, False
                except ValueError: pass
            try: return self.arm.solve(point, -axis, previous), False
            except ValueError: pass
        far = np.linalg.norm(self.reached[:, 0]-point, axis=1)+.1*np.linalg.norm(self.reached[:, 1]-axis, axis=1)
        for i in np.argsort(far)[:8]:
            try: return J.track(self.arm, point, axis, self.certified[i]), True
            except ValueError: pass
        return self.arm.solve(point, -axis), True

    def swap(self, k, k2, q_up, q_h, H):
        """joined_video's swap with each task's own exit: the support slides to its loading spot, the bunny
        leaves seat k along task k's exit, enters seat k2 along task k2's, and the support slides back."""
        fps, sec, model = self.fps, self.policy['seconds'], self.model; frames = []
        A = self.S[k if k is not None else k2]; L = self.loading[k if k is not None else k2]
        off = (self.off[k] if k is not None else self.off[k2])+H
        if k is not None:
            X = model.X[model.seat_of[k]]; pivot = self.centres[model.seat_of[k]]
            frames += [self.frame(q_up, off, J.between(A, L, u, pivot), J.between(A, L, u, pivot)@X, None) for u in J.steps(sec['slide'], fps)]
            start = L@X; out = model.exits[k]*self.exit_length[k]; end = J.shifted(out)@start
            frames += [self.frame(q_up, off, L, J.shifted(out*u)@start, None, moving=True) for u in J.steps(sec['exit'], fps)]
        if k2 is None: return frames
        X2 = model.X[model.seat_of[k2]]; seated = L@X2
        entry = (L[:3, :3]@self.layout.material[k2][:3, :3]@model.exits[k2])*self.exit_length[k2]; above = J.shifted(entry)@seated
        if k is None: frames += [self.frame(q_up, off, L, above, None)]*int(sec['hold']*fps)
        else:
            centre = self.source.centroid
            for arc in self.policy['arc_fractions']:
                path = [self.frame(q_up, off, L, J.flight(end, above, u, centre, arc*self.D), None, moving=True) for u in J.steps(sec['flight'], fps)]
                if all(self.overlap(L, f['B']) <= self.tolerance for f in path): break
            frames += path
        frames += [self.frame(q_up, off, L, J.shifted(entry*(1-u))@seated, None, moving=True) for u in J.steps(sec['exit'], fps)]
        S2 = self.S[k2]; pivot = self.centres[model.seat_of[k2]]; off2 = self.off[k2]+H
        for u in J.steps(sec['slide'], fps, self.joint_frames(q_up, q_h)):
            M = J.between(L, S2, u, pivot); frames.append(self.frame((1-u)*q_up+u*q_h, (1-u)*off+u*off2, M, M@X2, None))
        return frames


class Film:
    """One camera from POLICY['view'], following the bunny; the zoom holds its box at OBJECT_SHARE of the picture
    where the bunny stays whole (a tall pose gets less), both smoothed in time."""
    def __init__(self, plan, policy, share=OBJECT_SHARE):
        self.plan = plan; m = plan.model; w, h = self.size = policy['width'], policy['height']
        self.base = np.asarray(m.setup['base_position_world_m'], float)
        self.bottom = min(f['off'] for f in plan.frames)+self.base[2]-.3*plan.D
        self.scene = scene = Scene(plan.source, m.setup, w, h, arm=True); g = scene.r
        r = .9*float(np.abs(plan.arm.collisions[0][:, :2]).max()); c = trimesh.creation.cylinder(radius=r, height=1., sections=48)
        c.apply_translation([0, 0, .5])
        self.shapes = (g.Shape(scene.ctx, scene.shader, plan.support.vertices, plan.support.faces, BLUE, metallic=.05, roughness=.5),
                       g.Shape(scene.ctx, scene.shader, plan.table.vertices, plan.table.faces, J.TABLE, smooth=False, metallic=.02, roughness=.8),
                       g.Shape(scene.ctx, scene.shader, c.vertices, c.faces, J.COLUMN, smooth=False, metallic=.2, roughness=.5))
        self.view = v = np.asarray(policy['view'], float)/np.linalg.norm(policy['view'])
        right = np.cross([0, 0, 1], v); right /= np.linalg.norm(right); up = np.cross(v, right)
        V = plan.source.vertices; box = []
        for f in plan.frames:
            P = V@f['B'][:3, :3].T+f['B'][:3, 3]; x, y = P@right, P@up
            box.append((x.min(), x.max(), y.min(), y.max(), float(P.mean(0)@v)))
        box = np.asarray(box); work = np.array([f['task'] is not None for f in plan.frames])
        bw, bh, cx, cy = box[:, 1]-box[:, 0], box[:, 3]-box[:, 2], (box[:, 0]+box[:, 1])/2, (box[:, 2]+box[:, 3])/2
        # The picture's width: the bunny's box at `share` of it, but whole (at most policy['fit'] of its height and width).
        sigma = policy['follow_seconds']*policy['fps']; whole = np.maximum(bh/policy['fit']*w/h, bw/policy['fit'])
        W = self.widths = np.maximum(gaussian_filter1d(np.maximum(np.sqrt(bw*bh*w/h/share), whole), sigma, mode='nearest'), whole)
        # The bunny moved away from the arm's side (still whole), so the needle, the head and the wrist show.
        flange = np.asarray([(lambda t: t[0]+[0, 0, f['off']]-t[1]*(plan.L+.07))(plan.arm.forward(f['q'])) for f in plan.frames])
        side = np.sign(np.mean((flange@right-cx)[work]))
        tx = gaussian_filter1d(cx+side*np.minimum(policy['arm_room']*W, np.clip((W-bw)/2-.02*W, 0, None)), sigma, mode='nearest')
        ty = gaussian_filter1d(cy, sigma, mode='nearest')
        # Smoothing lags behind a fast slide: the camera never lets the bunny leave the picture.
        mx, my = (1-policy['fit'])/2*W, (1-policy['fit'])/2*W*h/w
        tx = np.clip(tx, box[:, 1]-W/2+mx, box[:, 0]+W/2-mx); ty = np.clip(ty, box[:, 3]-W*h/w/2+my, box[:, 2]+W*h/w/2-my)
        centre = right*cx[:, None]+up*cy[:, None]+v*box[:, 4:5]
        self.target = right*tx[:, None]+up*ty[:, None]+v*gaussian_filter1d(box[:, 4], sigma, mode='nearest')[:, None]
        # The key light's shadow map over the station, not the origin.
        light = g.orthographic(policy['light_span'], policy['light_span'])@g.view_matrix(np.array([-.65, -.9, 1.45])/np.linalg.norm([-.65, -.9, 1.45]), centre[work].mean(0))
        for program in (scene.shader, scene.shadow_shader): program['light_projection'].write(np.asarray(light, dtype='f4').T.tobytes())

    def frame(self, i):
        """RGB rows, bottom up."""
        plan, scene = self.plan, self.scene; f = plan.frames[i]; src = plan.source; g = scene.r; support, table, column = self.shapes
        w, h = self.size; vp = g.orthographic(self.widths[i], self.widths[i]*h/w)@g.view_matrix(self.view, self.target[i])
        scene.shader['view_projection'].write(np.asarray(vp, dtype='f4').T.tobytes()); scene.shader['eye'].value = tuple(self.target[i]+self.view*4)
        colors = np.tile(g.GRAY, (len(src.faces), 1))
        if f['task'] is not None: colors[plan.model.records[f['task']]['working_face_ids']] = LIGHT
        colors[f['painted']] = g.PINK
        L = matrices(plan.arm.forward(f['q'], geometry=True)[2]); L[:, 2, 3] += f['off']
        tcp, z, _, _ = plan.arm.forward(f['q']); tip = tcp+[0, 0, f['off']]
        C = J.lifted(self.bottom); C[2, 2] = self.base[2]+f['off']-self.bottom; C[:2, 3] = self.base[:2]     # the column up to the base
        if scene.colors is None or not np.array_equal(colors, scene.colors): scene.body.paint(colors); scene.colors = colors
        draws = [(scene.body, f['B'])]+[(s, J.tool_frame(tip, z)) for s in scene.tools[False]]+[(s, L[k]) for k, s in scene.arm]
        draws += [(support, f['S']), (table, np.eye(4)), (column, C)]
        ctx, studio = scene.ctx, scene.studio
        studio.shadow_buffer.use(); ctx.viewport = (0, 0, 1536, 1536); studio.shadow_buffer.clear(depth=1)
        for shape, world in draws: shape.shadow(ctx, scene.shadow_shader, world)
        ctx.enable(scene.gl.DEPTH_TEST)
        studio.gbuffer.use(); ctx.viewport = (0, 0, *studio.size); studio.gbuffer.clear(1, 1, 1, 1, depth=1); studio.shadow.use(3)
        for shape, world in draws: shape.draw(scene.shader, world)
        return np.frombuffer(studio.finish(vp, self.view), np.uint8).reshape(h, w, 3)

    def close(self): self.scene.close()


CAMERA = ('width', 'height', 'view', 'follow_seconds', 'light_span', 'fit', 'arm_room')     # not in the motion


def key(model, policy):
    """What the motion depends on: the support, the motion's policy and code (the camera's aside)."""
    code = inspect.getsource(Model)+inspect.getsource(Demo)+(ROOT/'whole_task_2/step5/joined_video.py').read_text()
    return dict(support=model.row['support_sha256'], policy=repr({k: v for k, v in policy.items() if k not in CAMERA}),
                code=hashlib.sha256(code.encode()).hexdigest())


def plan(policy=POLICY, cache=HERE/'B_plan.pkl'):
    """The checked motion, from the cache when its inputs are unchanged."""
    model = Model(); key_ = key(model, policy)
    p = Demo(model, model.layout, model.support, dict(tasks=model.row['tasks']), policy)
    if cache.exists():
        saved = pickle.loads(cache.read_bytes())
        if saved['key'] == key_: p.frames, p.events, p.tasks = saved['frames'], saved['events'], saved['tasks']; return p
    began = time.monotonic(); p.tasks = p.build(); p.seconds = time.monotonic()-began
    cache.write_bytes(pickle.dumps(dict(key=key_, frames=p.frames, events=p.events, tasks=p.tasks)))
    return p


def run(policy=POLICY, stills=False):
    began = time.monotonic(); p = plan(policy); failing = [e for e in p.events if 'fails' in e]
    print('DEMO PLAN', len(p.frames), 'frames,', len(failing), 'failing motions', flush=True)
    film = Film(p, policy); w, h = film.size
    try:
        if stills:
            from PIL import Image
            out = HERE/'stills'; out.mkdir(exist_ok=True)
            starts = [next(i for i, f in enumerate(p.frames) if f['task'] == k) for k in range(len(p.model.records))]
            for i in [0, *starts, *[s+40 for s in starts], len(p.frames)//2, len(p.frames)-1]:
                Image.fromarray(film.frame(i)[::-1]).save(out/f'{i:05d}.png')
            return out
        video = HERE/'B.mp4'
        ffmpeg = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(policy['fps']), '-i', '-',
                                   '-vf', 'vflip', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(video)],
                                  stdin=subprocess.PIPE)
        try:
            for i in range(len(p.frames)):
                ffmpeg.stdin.write(film.frame(i).tobytes())
                if i % 240 == 0: print('DEMO FRAME', i, '/', len(p.frames), flush=True)
        finally:
            ffmpeg.stdin.close(); code = ffmpeg.wait()
        if code: raise RuntimeError('Video encoding failed')
    finally: film.close()
    write_json(HERE/'B.json', dict(run=RUN, result=RESULT, video=str(video.relative_to(ROOT)), video_sha256=sha256(video),
                                   frames=len(p.frames), fps=policy['fps'], duration_s=len(p.frames)/policy['fps'], size=[w, h],
                                   object_share=OBJECT_SHARE, camera_width_m=[float(film.widths.min()), float(film.widths.max())], tasks=p.tasks, motions=p.events, failing=failing,
                                   seconds=time.monotonic()-began))
    print('DEMO VIDEO', video, round(len(p.frames)/policy['fps'], 1), 's,', len(failing), 'failing motions', flush=True)
    return video


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__); a.add_argument('--stills', action='store_true')
    run(stills=a.parse_args().stills)
