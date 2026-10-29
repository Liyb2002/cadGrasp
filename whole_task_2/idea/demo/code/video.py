"""Demo video: a run's Step5 result in use, one fixed camera, no text.

The work arm's motion is step5/joined_video.py's Plan, every frame checked as there: for each task the work arm, rising
and falling with the object on its column, moves the needle over the task's certified faces (light pink: what the task
still has to paint, deep pink: painted).  Between tasks it backs off, rises and turns aside (still in the picture) and
waits while a Franka Panda changes the scene (franka.py's pick and place: grasps, IK, RRT-Connect, straight lines in and
out): for a new seat it takes the bunny out of its seat along that task's certified exit and lays it on a rest spot on
the table, moves the support to the next installation, and puts the bunny into the next seat along that task's exit; for
a turn of the same seat it moves the support with the bunny in it.  Nothing moves by itself.  Every film frame is
checked: the work arm (Plan.fails) and the Panda (Handler.fails) against the support, the bunny, the table and each
other.  One camera, fixed, holds everything the bunny and the support pass through.

The run's station gives the work arm (robots_config's arm module: the KUKA of B+03_kuka, the SO101 of B+9pose, drawn
with its own meshes).

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/demo/code/video.py [--run B+9pose] [--stills]
The video goes to idea/demo/<run>.mp4 (B+03_kuka: B+03.mp4); its record, the caches and stills stay in idea/demo/code/.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import pickle
import subprocess
import time
from pathlib import Path
import numpy as np
import trimesh
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import ConvexHull

from whole_task_2.step1.common import ROOT, sha256, write_json
from whole_task_2.step1.reach_video import Scene, LIGHT
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step4.seats import context, read_mesh
from whole_task_2.step4.combine import ExitModel, seats
from whole_task_2.step5.combined import FOLDER, distinct
from whole_task_2.step5.video.prepare import matrices
from whole_task_2.step5 import joined_video as J
import franka as F

HERE = Path(__file__).resolve().parent
RUN, RESULT = 'B+03_kuka', 'r00'
# How the needle leaves a stop: a straight lead back along it (m), then a direction tilted from straight back by
# tilt (deg), toward up turned about the needle by turn (deg); the first the arm and the needle can follow for
# retreat_m (else the one followed farthest), the needle checked from check_m on.  While the Panda changes the scene,
# the work arm, backed off and raised, waits turned about its base by the first of `wait_yaw` (deg) that takes its links
# and its tool `wait_clear` (m) outside the Panda's work area (Demo.waiting), still in the picture.  The
# bunny leaves a seat by its own size along the exit, plus exit_clear (m): a prefix of its certified exit.  The Panda
# stands `franka_m` from the bunny's place at work (Demo.franka_cell); the bunny rests between seats as far from it,
# swung about its base by the first of `rest_turn` (deg) `rest_gap` (m) clear of the support's footprints.
LEADS, TILTS, TURNS = (.012, .006, 0.), (45., 60., 75.), (0., 45., -45., 90., -90., 135., -135., 180.)
POLICY = dict(J.POLICY, retreats=[(l, t, u) for l in LEADS for t in TILTS for u in TURNS], retreat_m=.08, check_m=.004,
              back_off_needles=(.3, .45, .6, 1., 1.6, 2.4), jump_deg=10., wait_yaw=(25., 45., 65., 85., 105.), wait_clear=.1, exit_clear=.02,
              franka_m=.6, rest_turn=(45., 60., 75., 90.), rest_gap=.04)
# The camera: fixed, the whole way of the bunny and the support at `fit` of the picture.
FILM = dict(width=1920, height=1080, view=(1., .15, .5), fit=.9, light_span=1.6)
# Rendering: `workers` processes (one GL context each) render the film's segments, x264 at `preset`/`crf`.
RENDER = dict(workers=8, preset='veryfast', crf=20, seconds=30.)
# The film: from the first work frame to the end, evenly in time, `seconds` long at the plan's fps.
BENCH = (.58, .60, .63)                                   # the table: gray
TABLE_DEPTH = .05                                         # m, the table drawn this thick (its top the plan's table)
# Joint speed (deg/s) of a work arm whose module gives none: the SO101's Feetech STS3215 servos, about 270 deg/s
# unloaded, two thirds of it.
VELOCITY_DEG = 180.
_kuka = J.work_arm


def work_arm(setup):
    """The station's work arm carrying the head (flange 70 mm behind the needle root): its config's arm module with its
    own joints (not padded to seven, so damped least squares tracks it), the KUKA without one."""
    module = setup.get('arm_module')
    if module is None: return _kuka(setup)
    arm = importlib.import_module(module.removesuffix('.py').replace('/', '.')).Arm(
        np.asarray(setup['base_position_world_m'], float), np.asarray(setup['base_rotation_world'], float), tool=setup['needle_length_m']+.07)
    if not hasattr(arm, 'velocity'): arm.velocity = np.radians(np.full(len(arm.lower), VELOCITY_DEG))
    return arm


J.work_arm = work_arm


def name_of(run):
    """The video's name: the run without its station model (B+03_kuka: B+03)."""
    return run.split('_')[0]


def table_block(plan):
    """The table as drawn: the plan's table top, TABLE_DEPTH thick."""
    lo, hi = plan.table.bounds; b = trimesh.creation.box(extents=[*(hi-lo)[:2], TABLE_DEPTH])
    b.apply_translation([*(lo+hi)[:2]/2, hi[2]-TABLE_DEPTH/2]); return b.vertices, b.faces
# The carrier: a Franka Panda with its hand (franka.py).  Grasps on the support and the bunny: rim pinches of thin walls
# (`rim`: the pads' centre `inset` in from a rim face, among `faces` sampled; the approach into the wall, tilted about the
# closing by `tilts` deg) and antipodal pairs from `samples` surface points, `approaches` directions about the closing
# axis; normals opposite within `cone` deg; the hand clear of the object and what surrounds it, the fingers `margin` (m)
# off it each side; `keep` per set, `apart` (m) or 30 deg from each other, the ones the Panda takes best first; `tries`
# of them per transfer; fewer than `keep` clear: sampled again, at most `rounds` times; on the bunny clear every
# `exit_step` (m) along its exit.  Lifts (m) off the table; the
# bunny rests in the stable pose (of the `poses` likeliest, turned by `yaws`) leaving the most grasps for both seats.
FRANKA = dict(rim=dict(inset=.005, faces=20000, tilts=(0., 25., -25.)), samples=dict(S=6000, B=12000), cone=20., approaches=24,
              stroke=F.PANDA['stroke'], margin=F.PANDA['margin'], keep=dict(S=40, B=60), apart=dict(S=.015, B=.008), tries=24,
              lift=dict(S=.02, B=.03), rest=dict(poses=8, yaws=8), rounds=3, exit_step=.01, workers=20)


class Model:
    """Step5's result as joined_video's Plan reads a model: the seats (X), every task's seat and exit.  Where the run
    has no passed Step5 result, Step4's support of its first (it passed Step4's acceptance: loads, continuous exits,
    tools; Step5 failed it on manufacturing), or Step4's best while Step5 has no report (being rerun); its exit
    lengths from Step4's exit paths."""
    def __init__(self, run=RUN, result=RESULT):
        self.run = run; c = context(run); folder = ROOT/'whole_task_2/data'/run/FOLDER
        report = json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else \
            dict(results=[dict(name='step4', status='none', step4_file=distinct(run)[0]['file'])])
        passed = [r for r in report['results'] if r['status'] == 'passed']
        self.row = next((r for r in passed if r['name'] == result), passed[0] if passed else None)
        if self.row is not None:
            group = {g['file']: g for g in distinct(run)}[self.row['step4_file']]
            self.path = folder/f"{self.row['name']}.npz"; self.support = read_mesh(self.path); self.source_of = 'step5 '+self.row['name']
            if sha256(self.path) != self.row['support_sha256']: raise ValueError('Changed Step5 support')
            base = ExitModel(c, group['slots'])
        else:
            first = report['results'][0]; group = {g['file']: g for g in distinct(run)}[first['step4_file']]
            self.path = ROOT/first['step4_file']; self.support = read_mesh(self.path); self.source_of = 'step4 '+Path(first['step4_file']).name
            base = ExitModel(c, group['slots']); found = dict(layout=base.layout(self.support), support=self.support, construction={})
            found['tasks'] = base.prepare(found['layout']); base.exit_paths(found)
            self.row = dict(name=first['name'], support_sha256=sha256(self.path), tasks=[dict(exit_length_m=float(t['length'])) for t in found['tasks']])
        self.layout = base.layout(self.support); rows = seats(group)
        # The video's way out of a seat: the bunny's own extent along its exit plus a margin (a prefix of the certified exit).
        for k, t in enumerate(self.row['tasks']):
            V = base.source.vertices@np.asarray(base.records[k]['pose'])[:3, :3].T; e = np.asarray(self.layout.exits[k], float); e /= np.linalg.norm(e)
            t['exit_length_m'] = float(min(t['exit_length_m'], np.ptp(V@e)+POLICY['exit_clear']))
        self.X = [X for X, _ in rows]
        self.seat_of = [next(n for n, (_, ks) in enumerate(rows) if k in ks) for k in range(len(base.records))]
        self.source, self.setup, self.records = base.source, base.setup, base.records
        self.tolerance, self.policy = base.tolerance, base.policy
        self.exit = None; self.exits = np.asarray(self.layout.exits, float)        # world, at each task's installation


class Demo(J.Plan):
    def __init__(self, *args):
        super().__init__(*args)
        # Every certified joint set (the face centroid's; the arm's own joints of Step3's seven), with its tip and axis in the arm's own frame.
        self.certified = np.vstack([np.asarray(r['joints'])[:, 3, :len(self.arm.lower)] for r in self.model.records])
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
                q = joints[s['j'], 3, :len(self.arm.lower)]; reach = 0.
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

    def franka_cell(self):
        """Where the Panda stands and where the bunny rests between seats (world).  From c, the bunny's mean place at work
        on the table top: the Panda's base `franka_m` away, facing c, at the azimuth (every 30 deg, at least 90 deg from
        the work arm's base) the bunny's exits point to the most (it comes out toward the Panda; a tie: the side away from
        the camera); the rest spot as far from the Panda as c, swung about its base by the first of `rest_turn` (deg; the
        side away from the work arm) where the bunny there (its farthest point about its centroid) clears every footprint
        of the support by `rest_gap`.  Returns base, yaw, rest spot."""
        if getattr(self, '_cell', None) is not None: return self._cell
        m, p = self.model, self.policy
        c = np.mean([(S@m.X[m.seat_of[k]])[:3, :3]@self.source.centroid+(S@m.X[m.seat_of[k]])[:3, 3] for k, S in enumerate(self.S)], axis=0); c[2] = self.Z0
        self._centre = c
        arm = np.asarray(m.setup['base_position_world_m'], float)-c; arm[2] = 0.; arm /= np.linalg.norm(arm)
        e = np.array(m.exits, float); e[:, 2] = 0.; e = e.mean(0); view = np.asarray(FILM['view'], float); view[2] = 0.
        dirs = [np.array([np.cos(t), np.sin(t), 0.]) for t in np.radians(np.arange(0., 360., 30.))]
        u = max((d for d in dirs if d@arm <= 1e-9), key=lambda d: (round(float(d@e), 6), -float(d@view)))
        base = c+u*p['franka_m']; yaw = float(np.arctan2(*(c-base)[1::-1]))
        foot = np.vstack([(self.support.vertices[::5]@S[:3, :3].T+S[:3, 3])[:, :2] for S in self.S])
        self.reach = float(np.linalg.norm(self.source.vertices-self.source.centroid, axis=1).max())
        sign = -1. if np.cross(c-base, arm)[2] > 0 else 1.                          # turning away from the work arm
        def swung(t): return base+J.Rotation.from_euler('z', sign*t, degrees=True).as_matrix()@(c-base)
        rest = next((swung(t) for t in p['rest_turn'] if np.linalg.norm(foot-swung(t)[:2], axis=1).min() >= self.reach+p['rest_gap']), None)
        if rest is None: raise ValueError('No rest spot for the bunny clear of the support')
        self._cell = (base, yaw, rest); return self._cell

    def make_table(self, edge):
        """A slab under the support at every installation, the rest spot and the Panda's base, its top the table."""
        base, _, rest = self.franka_cell(); r = self.reach
        P = np.vstack([*[(self.support.vertices[::5]@S[:3, :3].T+S[:3, 3])[:, :2] for S in self.S], rest[:2]+[[r, r], [-r, -r]], base[:2]+[[.15, .15], [-.15, -.15]]])
        m = self.policy['table_margin_fraction']*self.D; low, high = P.min(0)-m, P.max(0)+m; t = self.policy['table_thickness_fraction']*self.D
        box = trimesh.creation.box(extents=[*(high-low), t]); box.apply_translation([*(low+high)/2, self.Z0-t/2])
        fine = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(box.vertices, box.faces, edge, max_iter=40), process=False)
        return box, J.placed(fine, np.eye(4))

    def waiting(self, q):
        """The work arm's joints while the Panda works: q turned about its base (J1, within its limits) by the first of
        `wait_yaw` (deg; of the two ways the one farther out) that puts its link origins and its tool `wait_clear` (m)
        outside the Panda's work area (horizontally: the hull of the support's footprints at every installation, the
        bunny at the end of every exit, the bunny on the rest spot and the Panda's base); else the farthest out."""
        base, _, rest = self.franka_cell(); ring = np.c_[np.cos(np.arange(16)*np.pi/8), np.sin(np.arange(16)*np.pi/8)]; m = self.model
        ends = [J.shifted(m.exits[k]*self.exit_length[k])@S@m.X[m.seat_of[k]] for k, S in enumerate(self.S)]
        area = np.vstack([*[(self.support.vertices[::5]@S[:3, :3].T+S[:3, 3])[:, :2] for S in self.S],
                          *[(self.source.vertices[::5]@M[:3, :3].T+M[:3, 3])[:, :2] for M in ends], rest[:2]+self.reach*ring, base[:2]+.15*ring])
        E = ConvexHull(area).equations
        def out(x):                                                     # how far outside the area its nearest point is
            tcp, z, _, _ = self.arm.forward(x); links = self.arm.forward(x, geometry=True)[2]
            P = np.array([o for _, o in links[2:]]+[tcp, tcp-z*(self.L+.07)])[:, :2]; return float((P@E[:, :2].T+E[:, 2]).max(1).min())
        best = None
        for yaw in self.policy['wait_yaw']:
            turned = [x for x in (np.r_[q[0]+np.radians(sg*yaw), q[1:]] for sg in (1., -1.)) if self.arm.lower[0] <= x[0] <= self.arm.upper[0]]
            scored = sorted(((out(x), i, x) for i, x in enumerate(turned)), key=lambda t: -t[0])
            if scored and scored[0][0] >= self.policy['wait_clear']: return scored[0][2]
            if scored and (best is None or scored[0][0] > best[0]): best = scored[0]
        return best[2]

    def changes(self):
        """The changes of the scene, in order: (task before, task after)."""
        return [f['change'] for f in self.frames if 'change' in f]

    def change(self, k, k2, q):
        """The work arm out of the Panda's way: from task k's last stop (None: the start) it backs off along the needle,
        rises and turns aside about its base (waiting); one frame there is marked `change` (k, k2), which the
        Panda's frames replace (it changes the scene from task k's to task k2's); then, in task k2's scene, the column
        goes to task k2's height and the arm turns back over task k2's first stop (None: the end) and comes down onto it.
        Every frame checked, the clearances growing until clear."""
        fps, sec, D = self.fps, self.policy['seconds'], self.D; model = self.model; last = None
        aside = self.waiting
        for back in self.policy['back_off_needles']:
            b = back*self.L; head = []
            try:
                if k is not None:
                    S, off = self.S[k], self.off[k]; B = S@model.X[model.seat_of[k]]
                    head, q_up, _ = self.follow(q, [self.off_contact(k, self.stops[k][-1], b*u) for u in J.steps(sec['back'], fps)], off, S, B, None,
                                                self.stops[k][-1]['allowance'])
                    if back != self.policy['back_off_needles'][-1] and self.first_failure(head) is not None: continue
                if k2 is not None: q_h, _ = self.solve(*self.off_contact(k2, self.stops[k2][0], b), self.off[k2], q_up if k is not None else None)
                wait = aside(q_up if k is not None else q_h)
            except ValueError: continue
            for rise in self.policy['rise_fractions']:
                H = rise*D; frames = list(head)
                if k is not None:
                    frames += [self.frame(q_up, off+H*u, S, B, None) for u in J.steps(sec['rise'], fps)]
                    frames += [self.frame((1-u)*q_up+u*wait, off+H, S, B, None) for u in J.steps(0, fps, self.joint_frames(q_up, wait))]
                    if k2 is not None: frames.append(dict(self.frame(wait, off+H, S, B, None), change=(k, k2)))
                if k2 is not None:
                    S2 = self.S[k2]; B2 = S2@model.X[model.seat_of[k2]]; off2 = self.off[k2]; o = off+H if k is not None else off2+H
                    frames += [self.frame(wait, (1-u)*o+u*(off2+H), S2, B2, None) for u in J.steps(sec['rise'], fps)]
                    frames += [self.frame((1-u)*wait+u*q_h, off2+H, S2, B2, k2) for u in J.steps(0, fps, self.joint_frames(wait, q_h))]
                    frames += [self.frame(q_h, off2+H*(1-u), S2, B2, k2) for u in J.steps(sec['rise'], fps)]
                    try: approach, p, _ = self.follow(q_h, [self.off_contact(k2, self.stops[k2][0], b*(1-u)) for u in J.steps(sec['approach'], fps)[:-1]],
                                                      off2, S2, B2, k2, self.stops[k2][0]['allowance'])
                    except ValueError: break
                    frames += approach
                else: frames += [frames[-1]]*int(sec['end']*fps); p = frames[-1]['q']
                failure = self.first_failure(frames); last = (f'change {k}->{k2} back {back} rise {rise}', frames, failure)
                if failure is None:
                    self.commit(frames); self.events.append(dict(motion=f'change {k}->{k2}', back_off_m=b, rise_m=H, clear=True))
                    return p
        if last is None: raise ValueError(f'No reachable back-off for change {k}->{k2}')
        what, frames, failure = last; self.commit(frames, failure, what); return frames[-1]['q']


def grasp_frame(g, a, c):
    """The hand's TCP frame at pad centre g, approach a, closing c."""
    T = np.eye(4); T[:3, 0] = np.cross(c, a); T[:3, 1] = c; T[:3, 2] = a; T[:3, 3] = g; return T


def antipodal(mesh, count, stroke, cone, seed=0):
    """Antipodal pairs on the mesh (its frame): from `count` surface points inward along the normal to where the ray
    leaves the solid, the normal there opposite within `cone` deg, the width under the stroke less 1 cm.  Returns the
    centres, the closing directions (first contact to second) and the widths."""
    pts, fi = trimesh.sample.sample_surface(mesh, count, seed=seed); n = mesh.face_normals[fi]
    loc, idx, tri = mesh.ray.intersects_location(pts-n*1e-6, -n, multiple_hits=False)
    w = np.linalg.norm(loc-pts[idx], axis=1); ok = (w > 1e-3) & (w < stroke-.01) & ((mesh.face_normals[tri]*-n[idx]).sum(1) > np.cos(np.radians(cone)))
    return (pts[idx[ok]]+loc[ok])/2, -n[idx[ok]], w[ok]


def rims(mesh, p, seed=0):
    """Rim grasps of thin walls (the mesh's frame): from a face, `inset` in along its normal; across that point the
    narrowest crossing among 8 directions square to the normal, within the stroke less 1 cm, its two normals opposite
    within `cone`.  Returns centres, closing directions, widths, approaches (into the wall, from the rim)."""
    r = p['rim']; rng = np.random.default_rng(seed); F_ = rng.choice(len(mesh.faces), min(r['faces'], len(mesh.faces)), replace=False)
    n = mesh.face_normals[F_]; o = mesh.triangles_center[F_]-n*r['inset']
    e = np.cross(n, np.where(np.abs(n[:, :1]) < .9, [[1., 0, 0]], [[0, 1., 0]])); e /= np.linalg.norm(e, axis=1)[:, None]; f = np.cross(n, e)
    best = np.full(len(F_), np.inf); out_c = np.zeros((len(F_), 3)); mid = np.zeros((len(F_), 3)); cos = np.cos(np.radians(p['cone']))
    for t in np.arange(8)*np.pi/8:
        u = np.cos(t)*e+np.sin(t)*f; ends = []
        for d in (u, -u):
            loc, idx, tri = mesh.ray.intersects_location(o, d, multiple_hits=False)
            hit = np.full((len(F_), 3), np.nan); ok = (mesh.face_normals[tri]*d[idx]).sum(1) > cos      # leaving the wall square
            hit[idx[ok]] = loc[ok]; ends.append(hit)
        w = np.linalg.norm(ends[0]-ends[1], axis=1); better = np.isfinite(w) & (w < best)
        best[better] = w[better]; out_c[better] = u[better]; mid[better] = (ends[0][better]+ends[1][better])/2
    ok = np.isfinite(best) & (best > 1e-3) & (best < p['stroke']-.01)
    return mid[ok], out_c[ok], best[ok], -n[ok]


_GRASP = None


def _grasp_context(context):
    global _GRASP
    _GRASP = context


def _grasp(job):
    """The approaches about one pair the Panda's hand clears (its hulls, the fingers `margin` off the pair each side), at
    the grasp and backed off along the approach (thirds of `pre`, the way it comes in): the object itself (its cloud,
    its frame), and at every pose (object frame to scene frame) the scenes' clouds and the table (world: a pose's world
    transform, the table top Z0); a pose flagged `pass` (on the way out) at the grasp only."""
    g, c, w, approaches = job; template, margin, pre, own, poses, Z0 = _GRASP; out = []
    hulls = F.hand_hulls(template, w+2*margin); pieces = F.pieces(hulls)[::-1]; corners = np.vstack(hulls)    # the fingers first
    for a in approaches:
        T = grasp_frame(g, a, c); way = [J.shifted(-a*b)@T for b in np.arange(4)*pre/3]
        if any(F.pierced(pieces, [U]*len(pieces), own) for U in way): continue
        clear = True
        for P, W, scenes, passing in poses:
            for U in (way[:1] if passing else way):
                if (corners@(W@U)[:3, :3].T+(W@U)[:3, 3])[:, 2].min() < Z0+.003 or any(F.pierced(pieces, [P@U]*len(pieces), s) for s in scenes): clear = False; break
            if not clear: break
        if clear: out.append((g, a, c, w))
    return out


def grasps(mesh, own, poses, Z0, template, p, count, seed=0):
    """Every clear grasp (pad centre, approach, closing, width) of an object, its frame (_grasp), in parallel: rim
    grasps of its thin walls and antipodal pairs approached all around their closing axis; where fewer than `keep`
    clear, sampled again (another seed, twice the samples), at most `rounds` times."""
    found = []
    for r in range(p['rounds']):
        found += _grasps(mesh, own, poses, Z0, template, p, count*2**r, seed+1000*r)
        if len(found) >= max(p['keep'].values()): break
    return found


def _grasps(mesh, own, poses, Z0, template, p, count, seed):
    def ring(c, n):
        e = np.cross(c, [1., 0, 0] if abs(c[0]) < .9 else [0, 1., 0]); e /= np.linalg.norm(e); f = np.cross(c, e)
        return [np.cos(t)*e+np.sin(t)*f for t in np.arange(n)*2*np.pi/n]
    def tilted(a, c):
        return [np.cos(np.radians(t))*a+np.sin(np.radians(t))*np.cross(c, a) for t in p['rim']['tilts']]
    jobs = [(g, c, w, tilted(a, c)) for g, c, w, a in zip(*rims(mesh, p, seed))]
    jobs += [(g, c, w, ring(c, p['approaches'])) for g, c, w in zip(*antipodal(mesh, count, p['stroke'], p['cone'], seed))]
    with ProcessPoolExecutor(p['workers'], initializer=_grasp_context, initargs=((template, p['margin'], F.PANDA['pre'], own, poses, Z0),)) as pool:
        return [x for f in pool.map(_grasp, jobs, chunksize=8) for x in f]


def spread(found, score, keep, apart):
    """At most `keep` grasps, best `score` first, each `apart` (m) or 30 deg of approach from those kept."""
    out = []
    for k in np.argsort([-score(x) for x in found]):
        g, a = found[k][:2]
        if all(np.linalg.norm(g-h[0]) > apart or a@h[1] < np.cos(np.radians(30)) for h in out): out.append(found[k])
        if len(out) == keep: break
    return out


_HANDLER = None


def _change(i):
    """One change of the scene, planned in a forked worker."""
    hd, waits = _HANDLER; began = time.monotonic()
    try: out = hd.change(waits[i], seed=i)
    except F.Failed as e: print('DEMO FRANKA failed', e, flush=True); return i, str(e)
    print('DEMO FRANKA change', waits[i]['change'], len(out), 'frames in', round(time.monotonic()-began), 's', flush=True); return i, out


class Handler:
    """The Franka Panda that changes the scene between tasks while the work arm waits aside (franka.py's pick and place):
    for a new seat the bunny out of its seat along its certified exit onto the rest spot on the table, the support to
    the next installation, the bunny from the rest spot into the next seat along that seat's exit; for a turn of the
    same seat the support with the bunny in it.  It starts and ends each change at its park pose.  Its grasps: on the
    support clear of itself and the table (and of the bunny in its seat, for a turn); on the bunny clear of the support
    in the seat it leaves or enters and along that seat's exit; on the rest spot the bunny lies in the stable pose
    leaving the most grasps for both seats."""
    def __init__(self, plan, policy=FRANKA):
        self.plan, self.policy = plan, policy; m = plan.model; base, yaw, self.rest = plan.franka_cell()
        pairs = plan.changes()
        for k in sorted({x for pair in pairs for x in pair}):               # the data: the bunny in its seat above the table
            B = plan.S[k]@m.X[m.seat_of[k]]; low = float((plan.source.vertices@B[:3, :3].T+B[:3, 3])[:, 2].min()-plan.Z0)
            if low < -.001: raise ValueError(f'{m.run} {m.source_of}: in the seat of task {k} the bunny reaches {-1000*low:.1f} mm below the table top')
        self.panda = F.Panda(base, yaw); self.ready(); self.park = self.parking()
        found = grasps(plan.support, self.clouds['S'], [(np.eye(4), plan.S[0], [], False)], plan.Z0, self.template, policy, policy['samples']['S'])
        self.support = {None: self.kept(found, 'S', plan.S[0])}
        for k, k2 in pairs:
            s = m.seat_of[k]
            if s == m.seat_of[k2] and s not in self.support:
                bunny = self.clouds['B'].at(m.X[s]); self.support[s] = self.kept([x for x in found if self.clear_of(x, bunny)], 'S', plan.S[k])
        print('DEMO FRANKA support grasps', {s: len(v) for s, v in self.support.items()}, flush=True)
        self.bunny = {k: self.bunny_grasps(k) for k in sorted({x for k, k2 in pairs if m.seat_of[k] != m.seat_of[k2] for x in (k, k2)})}
        self.stable = plan.source.compute_stable_poses(n_samples=1)

    def ready(self):
        plan = self.plan; self.template = self.panda.hand_template()
        self.work_solids = F.solids([*plan.arm.collisions, J.prism(0., plan.L+.07, plan.model.setup['head_radius_m'], 12)])
        self.body = dict(S=trimesh.sample.sample_surface(plan.support, 2500, seed=1)[0], B=trimesh.sample.sample_surface(plan.source, 1500, seed=2)[0])
        self.clouds = dict(S=F.Cloud.of(plan.support), B=F.Cloud.of(plan.source)); self.placed = {}

    @classmethod
    def cached(cls, plan, saved, policy=FRANKA):
        hd = cls.__new__(cls); hd.plan, hd.policy = plan, policy; hd.rest, hd.park = saved['rest'], saved['park']
        hd.panda = F.Panda(saved['base'], saved['yaw']); hd.ready(); return hd

    def kept(self, found, o, M):
        p = self.policy; return spread(found, lambda x: self.facing(x, M), p['keep'][o], p['apart'][o])

    def facing(self, x, M):
        """How well the Panda takes grasp x of a body at pose M: the approach down and away from its base."""
        a = M[:3, :3]@x[1]; u = M[:3, :3]@x[0]+M[:3, 3]-self.panda.base; u[2] = 0.; return float(-a[2]+a@u/np.linalg.norm(u))

    def hulls(self, x):
        return F.hand_hulls(self.template, x[3]+2*self.policy['margin'])

    def clear_of(self, x, cloud):
        h = self.hulls(x); T = grasp_frame(*x[:3]); return not F.pierced(F.pieces(h), [T]*len(h), cloud)

    def cloud(self, o, M):
        """The support's ('S') or the bunny's ('B') cloud at pose M (world), kept for reuse."""
        key_ = (o, np.round(M, 12).tobytes())
        if key_ not in self.placed:
            if len(self.placed) > 32: self.placed.clear()
            self.placed[key_] = self.clouds[o].at(M)
        return self.placed[key_]

    def on_table(self, x, M):
        """Does grasp x of a body at pose M keep the hand above the table."""
        W = M@grasp_frame(*x[:3]); return float((np.vstack(self.hulls(x))@W[:3, :3].T+W[:3, 3])[:, 2].min()) >= self.plan.Z0+.003

    def bunny_grasps(self, k):
        """Grasps on the bunny (its frame) clear of the support and the table in task k's seat and along its exit there."""
        plan, m, p = self.plan, self.plan.model, self.policy; X = m.X[m.seat_of[k]]; S = plan.S[k]
        e = S[:3, :3].T@m.exits[k]*plan.exit_length[k]
        along = np.linspace(0., 1., max(2, int(np.ceil(plan.exit_length[k]/p['exit_step']))+1))                  # every exit_step along the exit
        poses = [(J.shifted(e*t)@X, S@J.shifted(e*t)@X, [self.clouds['S']], t > 0) for t in along]                 # bunny frame to support frame
        found = grasps(plan.source, self.clouds['B'], poses, plan.Z0, self.template, p, p['samples']['B'], seed=k)
        out = self.kept(found, 'B', S@X); print('DEMO FRANKA bunny grasps for task', k, ':', len(found), 'clear,', len(out), 'kept', flush=True)
        if not out: raise ValueError(f'No grasp on the bunny in the seat of task {k}')
        return out

    def doable(self, grasps, M, world, mover):
        """The grasps (either closing sign) the Panda reaches clear in `world` on the bunny at M, fingers open."""
        out = []
        for x in grasps:
            for sign in (1., -1.):
                if mover.ik(M@grasp_frame(x[0], x[1], sign*x[2]), world, x[3]+2*self.policy['margin'], [self.park]) is not None: out.append(x); break
        return out

    def rest_pose(self, k, k2, out, into):
        """The bunny on the rest spot between tasks k and k2: of its `poses` likeliest stable poses turned by `yaws` about
        the vertical, the one leaving the most grasps above the table both of those it can leave task k's seat with (out)
        and of those it can enter task k2's with (into)."""
        plan, r = self.plan, self.policy['rest']; T, prob = self.stable; mesh = plan.source; best = None
        for i in np.argsort(-prob)[:r['poses']]:
            for yaw in np.arange(r['yaws'])*2*np.pi/r['yaws']:
                M = J.lifted(0.); M[:3, :3] = J.Rotation.from_euler('z', yaw).as_matrix(); M = M@T[i]
                V = mesh.vertices@M[:3, :3].T+M[:3, 3]; M[:3, 3] += [self.rest[0]-V[:, 0].mean(), self.rest[1]-V[:, 1].mean(), plan.Z0-V[:, 2].min()]
                n = min(sum(self.on_table(x, M) for x in out), sum(self.on_table(x, M) for x in into))
                if best is None or n > best[0]: best = (n, M)
        if best[0] == 0: raise ValueError(f'No resting pose of the bunny between tasks {k} and {k2}')
        return best[1]

    def parking(self):
        """The Panda's waiting pose: its ready pose turned about its base, the hand farthest from the station, clear."""
        P = self.panda; world = F.World(P, {}, self.plan.Z0, None); c = np.mean([S[:3, :3]@self.plan.support.centroid+S[:3, 3] for S in self.plan.S], axis=0)
        best = None
        for t in np.linspace(-2.5, 2.5, 11):
            q = P.ready.copy(); q[0] = t
            if not world.clear(q, F.PANDA['stroke']): continue
            d = np.linalg.norm((P.tcp_frame(q)[:3, 3]-c)[:2])
            if best is None or d > best[0]: best = (d, q)
        if best is None: raise F.Failed('No clear park pose for the Panda')
        return best[1]

    def work(self, f):
        """The waiting work arm as solids at frames: its links, its needle and head."""
        plan = self.plan; L = matrices(plan.arm.forward(f['q'], geometry=True)[2]); L[:, 2, 3] += f['off']
        tcp, z, _, _ = plan.arm.forward(f['q']); return self.work_solids, [*L, J.tool_frame(tcp+[0, 0, f['off']], z)]

    def world(self, state, without, work):
        clouds = {o: self.cloud(o, state[o]) for o in ('S', 'B') if o not in (without or '')}
        return F.World(self.panda, clouds, self.plan.Z0, work)

    def change(self, f, seed=0):
        """The Panda's frames for the change marked at wait frame f (the work arm waiting there): from its park pose each
        transfer in turn, back to park.  Per frame the support's and the bunny's poses, the Panda's joints `h`, what it
        holds `grip` and its fingers' opening `open`."""
        plan, m, p = self.plan, self.plan.model, self.policy; k, k2 = f['change']; mover = F.Mover(self.panda, plan.fps, seed)
        X1, X2, S1, S2 = m.X[m.seat_of[k]], m.X[m.seat_of[k2]], plan.S[k], plan.S[k2]; up = lambda h: np.array([0., 0., h])
        if m.seat_of[k] == m.seat_of[k2]:
            moves = [('SB', S1, S2, up(p['lift']['S']), up(p['lift']['S']), self.support[m.seat_of[k]], {'table'}, {'table'})]
        else:
            # the grasps the Panda can take the bunny out of seat k with and put it into seat k2 with, here (the work arm
            # waiting); the rest pose that keeps the most of both
            work = self.work(f); out = self.doable(self.bunny[k], S1@X1, self.world(dict(S=S1, B=S1@X1), None, work), mover)
            into = self.doable(self.bunny[k2], S2@X2, self.world(dict(S=S2), 'B', work), mover); R = self.rest_pose(k, k2, out, into)
            seat = {'S', 'table'}                                         # in a seat the bunny is near the table too
            moves = [('B', S1@X1, R, m.exits[k]*plan.exit_length[k], up(p['lift']['B']), [x for x in out if self.on_table(x, R)], seat, {'table'}),
                     ('S', S1, S2, up(p['lift']['S']), up(p['lift']['S']), self.support[None], {'table'}, {'table'}),
                     ('B', R, S2@X2, up(p['lift']['B']), m.exits[k2]*plan.exit_length[k2], [x for x in into if self.on_table(x, R)], {'table'}, seat)]
        state = dict(S=S1, B=S1@X1); work = self.work(f); q = self.park; out = []
        for n, (o, A, B, leave, enter, cands, skip_out, skip_in) in enumerate(moves):
            after = dict(state, **(dict(B=B) if o == 'B' else dict(S=B) if o == 'S' else dict(S=B, B=B@X1)))
            body = self.body[o] if o != 'SB' else np.vstack([self.body['S'], self.body['B']@X1[:3, :3].T+X1[:3, 3]])
            cands = sorted(cands, key=lambda x: -min(self.facing(x, A), self.facing(x, B)))
            try:
                frames = mover.transfer(body, A, B, leave, enter, cands, self.world(state, None, work), self.world(after, None, work),
                                        self.world(state, o, work), q, self.park if n+1 == len(moves) else None, skip_out, skip_in, p['tries'])
            except F.Failed as e: raise F.Failed(f'change {k}->{k2}, moving {o}: {e}')
            for h, holding, opening, M in frames:
                S_, B_ = (state['S'], M) if o == 'B' else (M, state['B']) if o == 'S' else (M, M@X1)
                out.append(dict(S=S_, B=B_, h=h, grip=o if holding else None, open=opening, moving=o == 'B' and holding))
            state = after; q = frames[-1][0]
        return out

    def film(self):
        """Film frames: the plan's, each wait frame marked `change` replaced by the Panda's change (planned in parallel);
        elsewhere the Panda parked."""
        global _HANDLER
        plan = self.plan; waits = [i for i, f in enumerate(plan.frames) if 'change' in f]; _HANDLER = (self, [plan.frames[i] for i in waits])
        with ProcessPoolExecutor(max(1, min(len(waits), self.policy['workers']))) as pool: done = dict(pool.map(_change, range(len(waits))))
        failed = [x for x in done.values() if isinstance(x, str)]
        if failed: raise F.Failed(f'{len(failed)} of {len(waits)} changes failed: '+'; '.join(failed))
        out = []; stroke = F.PANDA['stroke']
        for i, f in enumerate(plan.frames):
            if 'change' in f: out += [dict({k: v for k, v in f.items() if k != 'change'}, **x) for x in done[waits.index(i)]]
            else: out.append(dict(f, h=self.park, grip=None, open=stroke))
        return out

    def fails(self, f):
        """What the Panda meets at film frame f: its hulls vs the support's and the bunny's dense samples (the fingers not
        vs what they hold) and the table top, vs the work arm with its tool; and the bunny below the table top."""
        P = self.panda; Fm = P.frames(f['h'], f['open']); held = set(f['grip'] or ''); out = []
        moving = [i for i in range(len(Fm)) if P.level[i] > 0]
        for o in ('S', 'B'):
            keep = [i for i in moving if not (o in held and i in P.fingers)]
            if F.pierced([P.pieces[i] for i in keep], Fm[keep], self.cloud(o, f[o])): out.append(f'panda-{o}')
        if min((P.hulls[i]@Fm[i][:3, :3].T+Fm[i][:3, 3])[:, 2].min() for i in moving) < self.plan.Z0: out.append('panda-table')
        if float((self.plan.source.vertices[::5]@f['B'][:3, :3].T+f['B'][:3, 3])[:, 2].min()) < self.plan.Z0-.001: out.append('bunny-table')
        if F.touch(P.solids, Fm, *self.work(f)): out.append('arms')
        return out


class Film:
    """One fixed camera from FILM['view'] holding everything the bunny and the support pass through; the work arm with
    its column, and the Panda on the table, as far as they come into the picture."""
    def __init__(self, plan, handler, frames, film=FILM):
        self.plan, self.handler, self.frames = plan, handler, frames; m = plan.model; w, h = self.size = film['width'], film['height']
        self.base = np.asarray(m.setup['base_position_world_m'], float)
        self.bottom = min(f['off'] for f in plan.frames)+self.base[2]-.6*plan.D
        kuka = len(plan.arm.lower) == 7; self.scene = scene = Scene(plan.source, m.setup, w, h, arm=kuka); g = scene.r
        # the work arm: the KUKA's shapes, else its own meshes (the SO101's); the Panda's visual meshes
        self.work = scene.arm if kuka else [(i, g.Shape(scene.ctx, scene.shader, v, f, tuple(c), metallic=.25, roughness=.4)) for i, v, f, c in plan.arm.visuals()]
        self.panda = [g.Shape(scene.ctx, scene.shader, v, f, c, metallic=.15, roughness=.45) for v, f, c in handler.panda.visuals()]
        r = .9*float(np.abs(plan.arm.collisions[0][:, :2]).max()); c = trimesh.creation.cylinder(radius=r, height=1., sections=48)
        c.apply_translation([0, 0, .5])
        self.shapes = (g.Shape(scene.ctx, scene.shader, plan.support.vertices, plan.support.faces, BLUE, metallic=.05, roughness=.5),
                       g.Shape(scene.ctx, scene.shader, *table_block(plan), BENCH, smooth=False, metallic=.02, roughness=.8),
                       g.Shape(scene.ctx, scene.shader, c.vertices, c.faces, J.COLUMN, smooth=False, metallic=.2, roughness=.5))
        # the tool as the station config has it (and Step1 certified): needle 100 mm of radius 2 mm, head 70 mm of radius 20 mm
        self.needle = scene.tools[False]
        self.view = v = np.asarray(film['view'], float)/np.linalg.norm(film['view'])
        right = np.cross([0, 0, 1], v); right /= np.linalg.norm(right); up = np.cross(v, right)
        P = Film.held_points(plan, frames); x, y = P@right, P@up
        W = max(np.ptp(x)/film['fit'], np.ptp(y)/film['fit']*w/h); self.width = W
        self.target = right*(x.min()+x.max())/2+up*(y.min()+y.max())/2+v*float((P@v).mean())
        self.vp = g.orthographic(W, W*h/w)@g.view_matrix(v, self.target)
        scene.shader['view_projection'].write(np.asarray(self.vp, dtype='f4').T.tobytes()); scene.shader['eye'].value = tuple(self.target+v*4)
        light = g.orthographic(film['light_span'], film['light_span'])@g.view_matrix(np.array([-.65, -.9, 1.45])/np.linalg.norm([-.65, -.9, 1.45]), self.target)
        for program in (scene.shader, scene.shadow_shader): program['light_projection'].write(np.asarray(light, dtype='f4').T.tobytes())

    @staticmethod
    def held_points(plan, frames):
        """Points the camera holds: the bunny and the support wherever they go, the work arm's needle and wrist always."""
        P = [plan.source.vertices[::7]@f['B'][:3, :3].T+f['B'][:3, 3] for f in frames[::6]]
        P += [plan.support.vertices[::40]@f['S'][:3, :3].T+f['S'][:3, 3] for f in frames[::6]]
        for f in frames[::6]:                                                            # the work arm's needle and wrist
            tcp, z, _, _ = plan.arm.forward(f['q']); tip = tcp+[0, 0, f['off']]; P.append(np.vstack([tip, tip-z*(plan.L+.07), tip-z*(plan.L+.2)]))
        return np.vstack(P)

    @staticmethod
    def width_of(plan, frames, film=FILM):
        v = np.asarray(film['view'], float)/np.linalg.norm(film['view']); right = np.cross([0, 0, 1], v); right /= np.linalg.norm(right)
        up = np.cross(v, right); P = Film.held_points(plan, frames)
        return max(np.ptp(P@right)/film['fit'], np.ptp(P@up)/film['fit']*film['width']/film['height'])

    def frame(self, i):
        """RGB rows, bottom up."""
        plan, scene, hd = self.plan, self.scene, self.handler; f = self.frames[i]; src = plan.source; g = scene.r
        support, table, column = self.shapes; w, h = self.size
        colors = np.tile(g.GRAY, (len(src.faces), 1))
        if f['task'] is not None: colors[plan.model.records[f['task']]['working_face_ids']] = LIGHT
        colors[f['painted']] = g.PINK
        L = matrices(plan.arm.forward(f['q'], geometry=True)[2]); L[:, 2, 3] += f['off']
        tcp, z, _, _ = plan.arm.forward(f['q']); tip = tcp+[0, 0, f['off']]
        C = J.lifted(self.bottom); C[2, 2] = self.base[2]+f['off']-self.bottom; C[:2, 3] = self.base[:2]     # the column up to the base
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

    def close(self): self.scene.close()


CAMERA = ('height',)                                    # J.POLICY's picture height: not in the motion


def key(model, policy):
    """What the motion depends on: the support, the motion's policy and code (the camera's aside)."""
    code = inspect.getsource(Model)+inspect.getsource(Demo)+inspect.getsource(work_arm)+(ROOT/'whole_task_2/step5/joined_video.py').read_text()
    return dict(support=model.row['support_sha256'], policy=repr({k: v for k, v in policy.items() if k not in CAMERA}),
                code=hashlib.sha256(code.encode()).hexdigest())


def plan(run=RUN, policy=POLICY):
    """The checked motion, from the cache when its inputs are unchanged."""
    model = Model(run); key_ = key(model, policy); cache = HERE/f'{name_of(run)}_plan.pkl'
    saved = pickle.loads(cache.read_bytes()) if cache.exists() else None
    if saved and saved['key'] == key_:
        heavy = Demo.task_stops; Demo.task_stops = J.Plan.task_stops      # the stops' retreats serve planning only
        try: p = Demo(model, model.layout, model.support, dict(tasks=model.row['tasks']), policy)
        finally: Demo.task_stops = heavy
        p.frames, p.events, p.tasks = saved['frames'], saved['events'], saved['tasks']; p.key = key_; return p
    p = Demo(model, model.layout, model.support, dict(tasks=model.row['tasks']), policy)
    began = time.monotonic(); p.tasks = p.build(); p.seconds = time.monotonic()-began
    cache.write_bytes(pickle.dumps(dict(key=key_, frames=p.frames, events=p.events, tasks=p.tasks))); p.key = key_
    return p


def handled(p):
    """The Panda and the film's frames (the plan's, the Panda's changes), from the cache when the plan, the carrier's
    policy and code are unchanged."""
    code = ''.join(inspect.getsource(f) for f in (Handler, _change, grasps, _grasps, _grasp, antipodal, rims, grasp_frame, spread))+(HERE/'franka.py').read_text()
    key_ = hashlib.sha256(repr((p.key, FRANKA, code)).encode()).hexdigest(); cache = HERE/f'{name_of(p.model.run)}_film.pkl'
    saved = pickle.loads(cache.read_bytes()) if cache.exists() else None
    if saved and saved['key'] == key_: return Handler.cached(p, saved), saved['frames']
    hd = Handler(p); frames = hd.film()
    cache.write_bytes(pickle.dumps(dict(key=key_, frames=frames, base=hd.panda.base, yaw=hd.panda.yaw, rest=hd.rest, park=hd.park)))
    return hd, frames


_FILM = None


def _check(i):
    """What film frame i meets: the work arm (Plan.fails) and the Panda (Handler.fails)."""
    p, hd, frames, _ = _FILM; return i, p.fails(frames[i])+hd.fails(frames[i])


def _segment(job):
    """Frames [a, b) of the film into one mp4 segment, in this process's own GL context."""
    a, b, path = job; p, hd, frames, fps = _FILM; film = Film(p, hd, frames); w, h = film.size
    ffmpeg = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                               '-vf', 'vflip', '-c:v', 'libx264', '-preset', RENDER['preset'], '-crf', str(RENDER['crf']), '-pix_fmt', 'yuv420p', '-an', str(path)],
                              stdin=subprocess.PIPE)
    try:
        for i in range(a, b): ffmpeg.stdin.write(film.frame(i).tobytes())
    finally:
        ffmpeg.stdin.close(); code = ffmpeg.wait(); film.close()
    if code: raise RuntimeError('Video encoding failed')
    return str(path)


def run(run_name=RUN, policy=POLICY, stills=False, at=()):
    began = time.monotonic(); p = plan(run_name, policy); failing = [e for e in p.events if 'fails' in e]
    print('DEMO PLAN', len(p.frames), 'frames,', len(failing), 'failing motions', flush=True)
    hd, whole = handled(p)
    start = next(i for i, f in enumerate(whole) if f['task'] is not None)            # no setting up: from the first work
    pick = np.unique(np.linspace(start, len(whole)-1, int(RENDER['seconds']*policy['fps'])).round().astype(int)); frames = [whole[i] for i in pick]
    print('DEMO CUT', len(frames), 'of', len(whole), 'frames, from', start, flush=True)
    global _FILM
    _FILM = (p, hd, frames, policy['fps'])                     # the workers (forked) read it; no GL context in this process yet
    with ProcessPoolExecutor(RENDER['workers']) as pool: checks = sorted(pool.map(_check, range(len(frames)), chunksize=32))
    bad = [(i, c) for i, c in checks if c]
    print('DEMO CHECK', len(p.changes()), 'changes,', len(frames), 'frames,', len(bad), 'frames meeting something', flush=True)
    from collections import Counter
    print('DEMO CHECK meets', dict(Counter((c, f['grip']) for i, cs in bad for c in cs for f in [frames[i]])),
          flush=True)
    runs = []
    for i, c in bad:
        if runs and i == runs[-1][1]+1: runs[-1][1] = i; runs[-1][2] |= set(c)
        else: runs.append([i, i, set(c), frames[i]['grip'], frames[i]['task']])
    print('DEMO CHECK runs', [(a, b, sorted(c), g, t) for a, b, c, g, t in runs], flush=True)
    name = name_of(run_name); w, h = FILM['width'], FILM['height']
    if stills:
        from PIL import Image
        film = Film(p, hd, frames); out = HERE/'stills'/name; out.mkdir(parents=True, exist_ok=True)
        try:
            starts = [i for i, f in enumerate(frames) if f['grip'] and (i == 0 or frames[i-1]['grip'] != f['grip'])][:4]
            for i in sorted(set(at) or {0, *starts, *[s+30 for s in starts], len(frames)//3, len(frames)//2, len(frames)-1, *[i for i, _ in bad[:3]]}):
                Image.fromarray(film.frame(i)[::-1]).save(out/f'{i:05d}.png')
        finally: film.close()
        return out
    # the film in segments, one GL context per process, joined without re-encoding
    video = HERE.parent/f'{name}.mp4'; parts = HERE/f'segments_{name}'; parts.mkdir(exist_ok=True); n = RENDER['workers']
    cuts = np.linspace(0, len(frames), n+1).astype(int); t0 = time.monotonic()
    with ProcessPoolExecutor(n) as pool:
        segments = list(pool.map(_segment, [(a, b, parts/f'{k:02d}.mp4') for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])) if b > a]))
    (parts/'list.txt').write_text(''.join(f"file '{Path(s).name}'\n" for s in segments))
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(parts/'list.txt'), '-c', 'copy', '-movflags', '+faststart', str(video)], check=True)
    for f in parts.iterdir(): f.unlink()
    parts.rmdir(); print('DEMO RENDER', len(frames), 'frames in', round(time.monotonic()-t0), 's', flush=True)
    write_json(HERE/f'{name}.json', dict(run=run_name, support=p.model.source_of, video=str(video.relative_to(ROOT)), video_sha256=sha256(video),
                                         frames=len(frames), fps=policy['fps'], duration_s=len(frames)/policy['fps'], size=[w, h],
                                         camera=dict(view=FILM['view'], width_m=float(Film.width_of(p, frames)), fixed=True),
                                         carrier=dict(robot='Franka Panda with its hand', base_m=hd.panda.base.tolist(), yaw_rad=hd.panda.yaw, rest_spot_m=hd.rest.tolist(),
                                                      changes=[list(c) for c in p.changes()], policy=FRANKA, planner=F.PANDA),
                                         frames_meeting=[dict(frame=i, meets=c) for i, c in bad],
                                         tasks=p.tasks, motions=p.events, failing=failing, seconds=time.monotonic()-began))
    print('DEMO VIDEO', video, round(len(frames)/policy['fps'], 1), 's,', len(failing), 'failing motions,', len(bad), 'frames meeting something', flush=True)
    return video


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__); a.add_argument('--run', default=RUN); a.add_argument('--stills', action='store_true')
    a.add_argument('--frames', type=lambda t: [int(x) for x in t.split(',')], default=(), help='with --stills: these film frames')
    a = a.parse_args(); run(a.run, stills=a.stills, at=a.frames)
