"""Step5 video: the joined support in use, no text.

The table is fixed; the KUKA stands beside it on a column and rises and falls (its base only moves
vertically).  Tasks run seat by seat in Step4's order.  For a task the support stands at its installation
(turned about the vertical, the seat at the station) and the KUKA, at the task's height, moves the needle
over the task's certified faces: the tip on each stop's face centroid along that face's certified axis,
the joints a continuous IK path from the previous frame.  Light pink is what the task still has to paint,
deep pink is painted and stays.  Between the two tasks of a seat the KUKA backs off along the needle and
rises, the support turns on the table, the KUKA comes down.  Between seats the support slides away from
the station to a loading spot, the bunny leaves its seat along the certified exit, moves to the next seat
and enters it along that seat's exit, and the support slides back.

Every frame is checked, not only the stops: the KUKA's link hulls (step1.arm's exact triangle test)
against the support, the bunny and the table; the needle and the head (circumscribed prisms) against the
support and the table; against the bunny Step1's needle rule (reach.py): the needle line misses it and
the needle body clears it beyond r(1 + tan phi) behind the tip, phi the incidence on the face the tip
leaves or approaches, the stops themselves being Step1's certified contacts; the bunny against the
support (exact Boolean) while it moves between seats.  A motion that fails
is retried with more clearance (hop lift, back-off, rise, arc); anything still failing is reported.  The
joints follow the tool path with the smallest change each frame, and a motion gets as many frames as the
KUKA's joint speed limits need at playback.
"""
import argparse
import json
import subprocess
import time
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation, Slerp

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task_2.step1.arm import clipped
from whole_task_2.step1.reach import tip_contact
from whole_task_2.step1.reach_video import STATION_VIEW, LIGHT, Scene, work_arm, tool_frame, slerp, ease
from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step4.seats import read_mesh
from whole_task_2.step5.video.prepare import matrices
from .joined import FOLDER, model_from_disk

# Lengths are fractions of the object's diagonal D, back-offs of the needle length; each tuple is tried in order.
POLICY = dict(fps=24, close_width=1200, wide_width=800, height=1000, close_view=(.85, .25, .65), cell_fraction=.07, hop_lift_fractions=(.016, .04, .08, .16, .3),
              back_off_needles=(.6, 1., 1.6, 2.4), rise_fractions=(.2, .4, .7, 1., 1.5, 2.), arc_fractions=(.3, .6, 1., 1.6),
              speed_fraction=.9, loading_margin_fraction=.25, table_margin_fraction=.12, table_thickness_fraction=.1, prism_sides=16,
              seconds=dict(hold=.8, back=.5, rise=.5, turn=2., slide=1.5, exit=1., flight=1.6, approach=.5, end=1.2))
TABLE = (.86, .84, .80); COLUMN = (.32, .34, .37)


def pieces(points):
    """Convex pieces as step1.arm's Check keeps them: planes, centre and radius."""
    out = []
    for v in points:
        h = ConvexHull(v); c = v.mean(0)
        out.append((h.equations[:, :3], h.equations[:, 3], c, float(np.linalg.norm(v-c, axis=1).max())))
    return out


def prism(start, end, radius, sides):
    """A circumscribed prism about the tool axis (+z in the tool frame) from start to end."""
    a = np.linspace(0, 2*np.pi, sides, endpoint=False); r = radius/np.cos(np.pi/sides)
    ring = np.c_[np.cos(a)*r, np.sin(a)*r]
    return np.vstack([np.c_[ring, np.full(sides, start)], np.c_[ring, np.full(sides, end)]])


def placed(mesh, M):
    """Triangles of mesh at M, a tree on their centroids and the largest centroid-vertex distance."""
    T = mesh.triangles@M[:3, :3].T+M[:3, 3]; C = T.mean(1)
    return T, cKDTree(C), float(np.linalg.norm(T-C[:, None], axis=2).max())


def meets(convex, frames, scene):
    """Does any convex piece at its frame meet a triangle of the scene?  step1.arm.Check.hits' exact test."""
    T, tree, rho = scene
    for (n, d, c, r), M in zip(convex, frames):
        R, p = M[:3, :3], M[:3, 3]; near = tree.query_ball_point(R@c+p, r+rho)
        if not near: continue
        X = (T[near]-p)@R; s = X@n.T+d
        if (s <= 0).all(2).any(): return True
        for k in np.flatnonzero(~(s > 0).all(1).any(1)):
            if clipped(X[k], n, d): return True
    return False


def lifted(z):
    M = np.eye(4); M[2, 3] = z; return M


def shifted(v):
    M = np.eye(4); M[:3, 3] = v; return M


def heading(M): return float(np.arctan2(M[1, 0], M[0, 0]))


def between(A, B, u, pivot):
    """Support transforms turned about the vertical: the pivot (support frame) moves straight, the heading turns."""
    w = (heading(B)-heading(A)+np.pi) % (2*np.pi)-np.pi
    if abs(abs(w)-np.pi) < 1e-9: w = np.pi
    M = np.eye(4); M[:3, :3] = Rotation.from_euler('z', heading(A)+u*w).as_matrix()
    M[:3, 3] = (1-u)*(A[:3, :3]@pivot+A[:3, 3])+u*(B[:3, :3]@pivot+B[:3, 3])-M[:3, :3]@pivot
    return M


def flight(A, B, u, centre, arc):
    """The bunny from A to B: its centre straight with an arc upward, its rotation by slerp."""
    M = np.eye(4); M[:3, :3] = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))(u).as_matrix()
    M[:3, 3] = (1-u)*(A[:3, :3]@centre+A[:3, 3])+u*(B[:3, :3]@centre+B[:3, 3])+[0, 0, arc*np.sin(np.pi*u)]-M[:3, :3]@centre
    return M


def track(arm, point, axis, q, iterations=80):
    """Damped least squares from q: each step the smallest joint change toward the tip on point along axis,
    so the redundant arm does not wander.  Arm.solve's acceptance: within the joint limits, above its floor."""
    q = np.asarray(q, float).copy()
    for _ in range(iterations):
        tcp, z, Jp, Ja = arm.forward(q); r = np.r_[point-tcp, .18*(axis-z)]
        if np.linalg.norm(point-tcp) < 1e-8 and np.linalg.norm(axis-z) < 1e-8: break
        J = np.vstack([Jp, .18*Ja]); q = q+J.T@np.linalg.solve(J@J.T+1e-8*np.eye(6), r)
    else: raise ValueError('No tracking convergence')
    links = arm.forward(q, geometry=True)[2]
    if (np.any(q < arm.lower) or np.any(q > arm.upper) or
            min((v@R.T+o)[:, 2].min() for v, (R, o) in zip(arm.collisions[1:], links[1:])) < arm.base[2]-.001):
        raise ValueError('Tracking leaves the joint limits or the floor')
    return q


def steps(seconds, fps, at_least=0):
    """Eased parameters of a motion, the start excluded and the end included."""
    n = max(2, int(round(seconds*fps)), at_least)
    return ease(np.linspace(0, 1, n+1)[1:])


class Plan:
    """The whole video as frames, each checked; the table is fixed, the arm moves by `off` vertically."""
    def __init__(self, model, q, support, report, policy):
        self.model, self.layout, self.support, self.policy = model, q, support, policy
        self.source = model.source; self.D = float(np.linalg.norm(self.source.extents)); self.fps = policy['fps']
        self.arm = work_arm(model.setup); self.L = model.setup['needle_length_m']; H = self.L+.07
        self.speed = policy['speed_fraction']*self.arm.velocity/self.fps        # per joint, per frame
        self.links = pieces(self.arm.collisions)
        s = policy['prism_sides']; self.r = model.setup['needle_radius_m']; self.needles = {}
        self.tool = pieces([prism(0., self.L, self.r, s), prism(self.L, H, model.setup['head_radius_m'], s)])
        self.line = pieces([prism(0., self.L, 1e-5, 4)])
        self.cell = policy['cell_fraction']*self.D; self.tolerance = model.tolerance
        n = len(model.records); self.lift = [float((np.asarray(r['pose'])@np.linalg.inv(q.poses[k]))[2, 3]) for k, r in enumerate(model.records)]
        self.Z0 = self.lift[0]                                                     # the table top, world
        self.off = [self.Z0-h for h in self.lift]                                  # where the arm's base stands for task k
        self.S = [lifted(self.Z0)@np.linalg.inv(q.material[k]) for k in range(n)]  # world <- support
        self.exit = model.exit; self.exit_length = [t['exit_length_m'] for t in report['tasks']]
        self.centres = [X[:3, :3]@self.source.centroid+X[:3, 3] for X in model.X]
        setup = model.setup; out = np.asarray(setup['work_window_center_m'], float)-setup['base_position_world_m']; out[2] = 0
        self.outward = out/np.linalg.norm(out)
        edge = model.policy['arm_check_edge_fraction']*self.D
        self.support_fine = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(support.vertices, support.faces, edge, max_iter=20), process=False)
        self.stops = [self.task_stops(k) for k in range(n)]
        self.loading = {k: self.load_spot(self.S[k]) for k in range(n)}
        self.table, self.table_scene = self.make_table(edge)
        self.painted = np.zeros(len(self.source.faces), bool); self.frames = []; self.events = []; self.cache = {}

    # ---- geometry of the scene
    def load_spot(self, S):
        V = self.support.vertices@S[:3, :3].T+S[:3, 3]; t = V@self.outward
        return shifted(self.outward*(t.max()-t.min()+self.policy['loading_margin_fraction']*self.D))@S

    def make_table(self, edge):
        """A slab under every place the support stands or passes, its top the table."""
        seat = self.model.seat_of; n = len(self.S); paths = [(self.loading[0], self.S[0], seat[0])]
        for k in range(n):
            if k+1 < n and seat[k+1] == seat[k]: paths.append((self.S[k], self.S[k+1], seat[k]))
            else:
                paths.append((self.S[k], self.loading[k], seat[k]))
                if k+1 < n: paths.append((self.loading[k], self.S[k+1], seat[k+1]))
        P = [self.support.vertices[::5, :2]@M[:2, :2].T+M[:2, 3] for A, B, s in paths
             for M in [between(A, B, u, self.centres[s]) for u in np.linspace(0, 1, 13)]]
        P = np.vstack(P); m = self.policy['table_margin_fraction']*self.D; low, high = P.min(0)-m, P.max(0)+m
        t = self.policy['table_thickness_fraction']*self.D
        box = trimesh.creation.box(extents=[*(high-low), t]); box.apply_translation([*(low+high)/2, self.Z0-t/2])
        fine = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(box.vertices, box.faces, edge, max_iter=40), process=False)
        return box, placed(fine, np.eye(4))

    def task_stops(self, k):
        """The task's certified faces in cells, visited row by row (alternating); each working face painted at its nearest stop."""
        r = self.model.records[k]; pose = np.asarray(r['pose']); src = self.source
        ids = np.asarray(r['certified_face_ids']); C = src.triangles_center[ids]@pose[:3, :3].T+pose[:3, 3]
        axes = np.asarray(r['axes'])@pose[:3, :3].T; key = np.floor(C/self.cell).astype(int); out = []
        for cell in np.unique(key, axis=0):
            m = np.flatnonzero((key == cell).all(1)); j = int(m[np.argmin(np.linalg.norm(C[m]-C[m].mean(0), axis=1))])
            cos = -float(src.face_normals[ids[j]]@pose[:3, :3].T@axes[j])
            out.append(dict(key=cell, j=j, tip=C[j], axis=axes[j], allowance=float(tip_contact(self.r, cos))))
        rows = sorted({int(s['key'][1]) for s in out}); ordered = []
        for i, row in enumerate(rows):
            line = sorted([s for s in out if s['key'][1] == row], key=lambda s: (s['key'][0], -s['key'][2]))
            ordered += line if i % 2 == 0 else line[::-1]
        work = np.asarray(r['working_face_ids']); W = src.triangles_center[work]@pose[:3, :3].T+pose[:3, 3]
        nearest = cKDTree([s['tip'] for s in ordered]).query(W)[1]
        for i, s in enumerate(ordered): s['faces'] = work[nearest == i]
        return ordered                                                            # tips and axes: arm frame of task k

    # ---- frames and their checks
    def scene(self, name, M):
        key = (name, M.tobytes())
        if key not in self.cache:
            if len(self.cache) > 64: self.cache.clear()
            self.cache[key] = placed(self.support_fine if name == 'support' else self.source, M)
        return self.cache[key]

    def frame(self, q, off, S, B, task, contact=False, moving=False, allowance=0.):
        return dict(q=np.asarray(q, float), off=float(off), S=S, B=B, task=task, painted=self.painted.copy(), contact=contact,
                    moving=moving, allowance=allowance)

    def needle(self, allowance):
        """The needle line, the needle body from `allowance` behind the tip, and the head."""
        key = round(allowance, 9)
        if key not in self.needles:
            self.needles[key] = self.line+pieces([prism(min(allowance, self.L), self.L, self.r, self.policy['prism_sides'])])+self.tool[1:]
        return self.needles[key]

    def fails(self, f):
        L = matrices(self.arm.forward(f['q'], geometry=True)[2]); L[:, 2, 3] += f['off']
        tcp, z, _, _ = self.arm.forward(f['q']); T = tool_frame(tcp+[0, 0, f['off']], z)
        support, bunny = self.scene('support', f['S']), self.scene('bunny', f['B']); out = []
        if any(meets(self.links, L, s) for s in (support, bunny, self.table_scene)): out.append('arm')
        if any(meets(self.tool, [T, T], s) for s in (support, self.table_scene)): out.append('tool-support')
        if not f['contact'] and meets(self.needle(f['allowance']), [T, T, T], bunny): out.append('tool-bunny')
        if f['moving'] and self.overlap(f['S'], f['B']) > self.tolerance: out.append('bunny-support')
        return out

    def first_failure(self, frames):
        for i, f in enumerate(frames):
            bad = self.fails(f)
            if bad: return i, bad
        return None

    def solve(self, tip, axis, off, previous):
        """Joints with the tip (world) on its point along axis, the arm's base at off: the smallest joint change
        from the previous frame (damped least squares), else Arm.solve from it, else from its seeds (a jump)."""
        point = np.asarray(tip)-[0, 0, off]; axis = np.asarray(axis)
        if previous is not None:
            try: return track(self.arm, point, axis, previous), False
            except ValueError: pass
            try: return self.arm.solve(point, -axis, previous), False
            except ValueError: pass
        return self.arm.solve(point, -axis), True

    def commit(self, frames, failure=None, what=''):
        self.frames += frames
        if failure is not None: self.events.append(dict(motion=what, frame=len(self.frames)-len(frames)+failure[0], fails=failure[1]))

    # ---- motions
    def world_stop(self, k, s):
        return s['tip']+[0, 0, self.off[k]], s['axis']

    def segment(self, q, a, b, off, depth=0):
        """Joints from tool pose a to b (tip, axis; world): tracked, cut into equal task-space sub-steps, again
        where needed, until no joint passes its speed limit in a frame.  Returns the joints and the restarts."""
        p, jump = self.solve(b[0], b[1], off, q); m = int(np.ceil((np.abs(p-q)/self.speed).max()))
        if m <= 1 or jump or depth == 4: return [p], int(jump)
        out, restarts, c = [], 0, a
        for u in np.arange(1, m+1)/m:
            d = ((1-u)*a[0]+u*b[0], slerp(a[1], b[1], u)); Q, n = self.segment(q, c, d, off, depth+1)
            out += Q; restarts += n; q, c = Q[-1], d
        return out, restarts

    def follow(self, q, points, off, S, B, task, allowance=0.):
        """Frames along (tip, axis) waypoints (world) from joints q, within the joint speed limits.
        Returns the frames, the joints and the restarts."""
        tcp, z, _, _ = self.arm.forward(q); a = (tcp+[0, 0, off], z); frames = []; restarts = 0
        for tip, axis in points:
            b = (np.asarray(tip, float), np.asarray(axis, float)); Q, n = self.segment(q, a, b, off)
            frames += [self.frame(p, off, S, B, task, allowance=allowance) for p in Q]; restarts += n; q, a = Q[-1], b
        return frames, q, restarts

    def work(self, k, q):
        """Every stop of task k; between stops the tip lifts off along the needle, more until the hop is clear."""
        S = self.S[k]; B = S@self.model.X[self.model.seat_of[k]]; stops = self.stops[k]; restarts = 0
        for i, s in enumerate(stops):
            tip, axis = self.world_stop(k, s); painted = self.painted.copy(); painted[s['faces']] = True
            for lift in self.policy['hop_lift_fractions'] if i else (0.,):
                via = []
                if i:
                    a_tip, a_axis = self.world_stop(k, stops[i-1]); h = lift*self.D; d = slerp(a_axis, axis, .5)
                    via = ([((a_tip+tip)/2-d*h, d)] if np.linalg.norm(tip-a_tip) <= 1.5*self.cell else
                           [(a_tip-a_axis*h, a_axis), (tip-axis*h, axis)])
                allowance = max(s['allowance'], stops[i-1]['allowance'] if i else 0.)
                frames, p, n = self.follow(q, via+[(tip, axis)], self.off[k], S, B, k, allowance)
                frames[-1].update(contact=True, painted=painted)
                failure = self.first_failure(frames)
                if failure is None: break
            restarts += n; self.painted = painted; self.commit(frames, failure, f'stop {k}:{i}'); q = p
        return q, dict(task=self.model.records[k]['id'], stops=len(stops), ik_restarts=int(restarts))

    def change(self, k, k2, q):
        """From the last stop of task k (None: the start) to the first stop of task k2 (None: the end), checked;
        the clearances grow until every frame is clear.  Returns the joints over the first stop's hover point."""
        fps, sec, D = self.fps, self.policy['seconds'], self.D; model = self.model
        seat = None if k is None else model.seat_of[k]; seat2 = None if k2 is None else model.seat_of[k2]
        swap = seat != seat2; last = None
        for back in self.policy['back_off_needles']:
            b = back*self.L; head = []
            if k is not None:
                S, off = self.S[k], self.off[k]; B = S@model.X[seat]; tip, axis = self.world_stop(k, self.stops[k][-1])
                head, p, _ = self.follow(q, [(tip-axis*b*u, axis) for u in steps(sec['back'], fps)], off, S, B, None,
                                         self.stops[k][-1]['allowance'])
                if back != self.policy['back_off_needles'][-1] and self.first_failure(head) is not None: continue
                q_up = p
            if k2 is not None:
                tip2, axis2 = self.world_stop(k2, self.stops[k2][0])
                q_h, _ = self.solve(tip2-axis2*b, axis2, self.off[k2], q_up if k is not None else None)
                if k is None: q_up = q_h
            for rise in self.policy['rise_fractions']:
                H = rise*D; frames = list(head)
                if k is not None:
                    frames += [self.frame(q_up, off+H*u, S, B, None) for u in steps(sec['rise'], fps)]
                if swap: frames += self.swap(k, k2, q_up, q_h if k2 is not None else q_up, H)
                else:
                    pivot = self.centres[seat]; S2 = self.S[k2]
                    for u in steps(sec['turn'], fps, self.joint_frames(q_up, q_h)):
                        M = between(S, S2, u, pivot)
                        frames.append(self.frame((1-u)*q_up+u*q_h, (1-u)*(off+H)+u*(self.off[k2]+H), M, M@model.X[seat], None))
                if k2 is not None:
                    S2 = self.S[k2]; B2 = S2@model.X[seat2]; off2 = self.off[k2]
                    frames += [self.frame(q_h, off2+H*(1-u), S2, B2, k2) for u in steps(sec['rise'], fps)]
                    approach, p, _ = self.follow(q_h, [(tip2-axis2*b*(1-u), axis2) for u in steps(sec['approach'], fps)[:-1]], off2, S2, B2, k2,
                                                 self.stops[k2][0]['allowance'])
                    frames += approach
                else: frames += [frames[-1]]*int(sec['end']*fps); p = frames[-1]['q']
                failure = self.first_failure(frames); last = (f'change {k}->{k2} back {back} rise {rise}', frames, failure)
                if failure is None:
                    self.commit(frames); self.events.append(dict(motion=f'change {k}->{k2}', back_off_m=b, rise_m=H, clear=True))
                    return p
        what, frames, failure = last; self.commit(frames, failure, what); return frames[-1]['q']

    def swap(self, k, k2, q_up, q_h, H):
        """The support slides to its loading spot, the bunny leaves seat k along its exit, enters seat k2's along
        that seat's exit, and the support slides back to task k2's installation.  The arm waits raised."""
        fps, sec, model = self.fps, self.policy['seconds'], self.model; frames = []
        A = self.S[k if k is not None else k2]; L = self.loading[k if k is not None else k2]
        off = (self.off[k] if k is not None else self.off[k2])+H
        if k is not None:
            X = model.X[model.seat_of[k]]
            frames += [self.frame(q_up, off, between(A, L, u, self.centres[model.seat_of[k]]), between(A, L, u, self.centres[model.seat_of[k]])@X, None)
                       for u in steps(sec['slide'], fps)]
            start = L@X; end = shifted(self.exit*self.exit_length[k])@start
            frames += [self.frame(q_up, off, L, shifted(self.exit*self.exit_length[k]*u)@start, None, moving=True) for u in steps(sec['exit'], fps)]
        if k2 is None: return frames
        X2 = model.X[model.seat_of[k2]]; seated = L@X2
        entry = (L[:3, :3]@self.layout.material[k2][:3, :3]@self.exit)*self.exit_length[k2]; above = shifted(entry)@seated
        if k is None: frames += [self.frame(q_up, off, L, above, None)]*int(sec['hold']*self.fps)
        else:
            centre = self.source.centroid
            for arc in self.policy['arc_fractions']:
                path = [self.frame(q_up, off, L, flight(end, above, u, centre, arc*self.D), None, moving=True) for u in steps(sec['flight'], fps)]
                if all(self.overlap(L, f['B']) <= self.tolerance for f in path): break
            frames += path
        frames += [self.frame(q_up, off, L, shifted(entry*(1-u))@seated, None, moving=True) for u in steps(sec['exit'], fps)]
        S2 = self.S[k2]; pivot = self.centres[model.seat_of[k2]]; off2 = self.off[k2]+H
        for u in steps(sec['slide'], fps, self.joint_frames(q_up, q_h)):
            M = between(L, S2, u, pivot); frames.append(self.frame((1-u)*q_up+u*q_h, (1-u)*off+u*off2, M, M@X2, None))
        return frames

    def joint_frames(self, a, b):
        """Frames an eased joint-space move from a to b needs within the speed limits (peak rate 1.5 times the mean)."""
        return int(np.ceil(1.5*(np.abs(np.asarray(b)-a)/self.speed).max()))

    def overlap(self, S, B):
        """Exact volume shared by the support at S and the bunny at B."""
        key = ('solid', S.tobytes())
        if key not in self.cache: self.cache[key] = G.solid(moved_mesh(self.support, S))
        return G.volume(self.cache[key] ^ G.solid(moved_mesh(self.source, B)))

    def build(self):
        n = len(self.model.records); q = self.change(None, 0, None); tasks = []
        for k in range(n):
            q, row = self.work(k, q); tasks.append(row)
            q = self.change(k, k+1 if k+1 < n else None, q)
        return tasks


def extent(plan, frames, arm):
    """Points a camera must hold: bunny and support, and the arm's links or only the tool."""
    out = []
    for f in frames:
        out += [plan.source.vertices[::9]@f['B'][:3, :3].T+f['B'][:3, 3], plan.support.vertices[::9]@f['S'][:3, :3].T+f['S'][:3, 3]]
        if arm:
            L = matrices(plan.arm.forward(f['q'], geometry=True)[2]); L[:, 2, 3] += f['off']
            out.append(np.vstack([v[::40]@M[:3, :3].T+M[:3, 3] for v, M in zip(plan.arm.collisions, L)]))
        else:
            tcp, z, _, _ = plan.arm.forward(f['q']); tip = tcp+[0, 0, f['off']]; out.append(np.vstack([tip, tip-z*(plan.L+.07)]))
    return np.vstack(out)


class Film:
    """Two fixed cameras side by side, one GL context: close, from where the needle's faces show, on the
    station (the swaps leave it); wide, the whole station from the side: table, support, column and KUKA."""
    def __init__(self, plan, policy):
        self.plan = plan; h = policy['height']; m = plan.model
        self.base = np.asarray(m.setup['base_position_world_m'], float)
        self.bottom = min(f['off'] for f in plan.frames)+self.base[2]-.3*plan.D
        station = [f for f in plan.frames if f['task'] is not None][::3]
        wide = np.vstack([extent(plan, plan.frames[::4], True), plan.table.vertices, self.base+[0, 0, self.bottom-self.base[2]]])
        self.scene = scene = Scene(plan.source, m.setup, policy['close_width'], h, arm=True); g = scene.r
        r = .9*float(np.abs(plan.arm.collisions[0][:, :2]).max()); c = trimesh.creation.cylinder(radius=r, height=1., sections=48)
        c.apply_translation([0, 0, .5])
        self.shapes = (g.Shape(scene.ctx, scene.shader, plan.support.vertices, plan.support.faces, BLUE, metallic=.05, roughness=.5),
                       g.Shape(scene.ctx, scene.shader, plan.table.vertices, plan.table.faces, TABLE, smooth=False, metallic=.02, roughness=.8),
                       g.Shape(scene.ctx, scene.shader, c.vertices, c.faces, COLUMN, smooth=False, metallic=.2, roughness=.5))
        self.cameras = [self.camera(scene.studio, (policy['close_width'], h), policy['close_view'], extent(plan, station, False), 1.1),
                        self.camera(g.Studio(scene.ctx, policy['wide_width'], h), (policy['wide_width'], h), STATION_VIEW, wide, 1.04)]
        self.size = (policy['close_width']+policy['wide_width'], h)

    def camera(self, studio, size, view, points, margin):
        scene = self.scene; scene.studio, scene.size = studio, size
        scene.view = np.asarray(view, float)/np.linalg.norm(view); scene.fit(points, margin)
        return studio, size, scene.view, scene.vp, scene.target

    def aim(self, camera):
        scene = self.scene; scene.studio, scene.size, scene.view, scene.vp, scene.target = camera
        for program in (scene.shader, scene.line_shader): program['view_projection'].write(np.asarray(scene.vp, dtype='f4').T.tobytes())
        scene.shader['eye'].value = tuple(scene.target+scene.view*4)

    def frame(self, f):
        """RGB rows, bottom up, of both cameras."""
        plan = self.plan; src = plan.source; g = self.scene.r; support, table, column = self.shapes
        colors = np.tile(g.GRAY, (len(src.faces), 1))
        if f['task'] is not None: colors[plan.model.records[f['task']]['working_face_ids']] = LIGHT
        colors[f['painted']] = g.PINK
        L = matrices(plan.arm.forward(f['q'], geometry=True)[2]); L[:, 2, 3] += f['off']
        tcp, z, _, _ = plan.arm.forward(f['q']); tool = [(tcp+[0, 0, f['off']], z, False)]
        C = lifted(self.bottom); C[2, 2] = self.base[2]+f['off']-self.bottom; C[:2, 3] = self.base[:2]     # the column up to the base
        out = []
        for camera in self.cameras:
            self.aim(camera); w, h = camera[1]
            image = self.scene.draw(f['B'], colors, tool, L, (), [(support, f['S']), (table, np.eye(4)), (column, C)])
            out.append(np.frombuffer(image, np.uint8).reshape(h, w, 3))
        return np.hstack(out)

    def close(self): self.scene.close()


def render(plan, path, policy):
    """Encode every frame; returns the frame count."""
    film = Film(plan, policy); w, h = film.size
    try:
        ffmpeg = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(policy['fps']), '-i', '-',
                                   '-vf', 'vflip', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(path)],
                                  stdin=subprocess.PIPE)
        try:
            for i, f in enumerate(plan.frames):
                ffmpeg.stdin.write(film.frame(f).tobytes())
                if i % 240 == 0: print('STEP5 VIDEO FRAME', i, '/', len(plan.frames), flush=True)
        finally:
            ffmpeg.stdin.close(); code = ffmpeg.wait()
        if code: raise RuntimeError('Video encoding failed')
    finally: film.close()
    return len(plan.frames)


def run(object_name='B', output_root=DEFAULT_OUTPUT, policy=POLICY):
    began = time.monotonic(); c, model, seats, q, report = model_from_disk(object_name)
    folder = ROOT/'whole_task_2/data'/object_name/FOLDER; step5 = json.loads((folder/'report.json').read_text())
    support = read_mesh(folder/'support.npz')
    if sha256(folder/'support.npz') != step5['support_sha256']: raise ValueError('Changed Step5 support')
    plan = Plan(model, q, support, step5, policy); tasks = plan.build(); planned = time.monotonic()-began
    print('STEP5 VIDEO PLAN', len(plan.frames), 'frames,', sum(1 for e in plan.events if 'fails' in e), 'failing motions', flush=True)
    out = output_root/object_name/'step5'; out.mkdir(parents=True, exist_ok=True); video = out/'joined.mp4'
    frames = render(plan, video, policy)
    Q = np.asarray([f['q'] for f in plan.frames]); step = np.degrees(np.abs(np.diff(Q, axis=0)).max(1))
    result = dict(status='passed' if not any('fails' in e for e in plan.events) else 'frames fail checks', video=str(video.relative_to(ROOT)),
                  video_sha256=sha256(video), frames=frames, fps=policy['fps'], duration_s=frames/policy['fps'],
                  size=[policy['close_width']+policy['wide_width'], policy['height']], table_top_world_m=plan.Z0, arm_base_offsets_m=plan.off, tasks=tasks,
                  motions=plan.events, failing=[e for e in plan.events if 'fails' in e],
                  largest_joint_step_deg=float(step.max()), joint_steps_over_10_deg=int((step > 10).sum()),
                  checks=dict(arm='KUKA link collision hulls vs support, bunny and table (exact triangle test), every frame',
                              tool='needle and head prisms vs support and table every frame; vs bunny except tip-on-face stops',
                              bunny='exact Boolean bunny vs support while the bunny moves'),
                  policy=policy, input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                                               (folder/'report.json', folder/'support.npz', ROOT/'whole_task_2/step5/joined_video.py')},
                  plan_seconds=planned, seconds=time.monotonic()-began)
    write_json(folder/'video.json', result)
    print('STEP5 VIDEO', result['status'], video, round(result['duration_s'], 1), 's', flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--object', default='B')
    run(p.parse_args().object)
