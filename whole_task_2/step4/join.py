"""Step4: choose the seats, and the model of tasks on the slots of one support.

Every feasible member set from step4/seats.py is a seat that stands on the table in each of
its down directions.  A choice of seats covers every task once, with as few seats as possible.
Each seat is turned so that its down directions fall on down directions already used, as many as
possible; the support's poses are the distinct down directions that remain.  A pose is a face
of the support resting on the table; turning on the table about the vertical is not a pose.
step4/merge.py then merges the chosen seats into one support with Step4's operations.
"""
import itertools
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation

from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.operations.state import Layout
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.membranes import MembraneModel
from .seats import SeatModel

SAME = 1e-6


def frame(d1, d2):
    """Orthonormal frame from a direction and a second, non-parallel one."""
    x = d1/np.linalg.norm(d1); y = d2-(d2@x)*x; y /= np.linalg.norm(y)
    return np.c_[x, y, np.cross(x, y)]


def mapping(d1, d2, e1, e2):
    """The rotation taking d1 to e1 and d2 to e2 (angles equal)."""
    return frame(e1, e2)@frame(d1, d2).T


def align(down_sets):
    """Rotations turning each seat into the support frame and the support's poses.

    Seats with more down directions go first; each next seat takes the rotation that adds the
    fewest new down directions: matching two of its directions to two used ones with the same
    angle between them, or one to one (the turn about it is then free)."""
    order = sorted(range(len(down_sets)), key=lambda s: -len(down_sets[s]))
    rotations = [None]*len(down_sets); used = []
    for s in order:
        D = [np.asarray(d, float) for d in down_sets[s]]; options = [np.eye(3)]
        for d, e in itertools.product(D, used):
            options.append(Rotation.align_vectors([e], [d])[0].as_matrix())
        for (d1, d2), (e1, e2) in itertools.product(itertools.permutations(D, 2), itertools.permutations(used, 2)):
            if abs(d1@d2-e1@e2) < SAME: options.append(mapping(d1, d2, e1, e2))
        def new(R): return [R@d for d in D if not any(1-(R@d)@e < SAME for e in used)]
        R = min(options, key=lambda R: len(new(R)))
        rotations[s] = R; used += new(R)
    return rotations, used


def covers(sets, n, limit=None):
    """Every way to split tasks 0..n-1 into feasible member sets, at most `limit` of them."""
    def rec(left, budget):
        if not left: yield []; return
        if budget == 0: return
        first = min(left)
        for T in sets:
            if first in T and set(T) <= left:
                for rest in rec(left-set(T), budget-1): yield [T]+rest
    yield from rec(set(range(n)), n if limit is None else limit)


def fewest_seats(sets, n):
    """The fewest feasible member sets covering every task once."""
    return next(k for k in range(1, n+1) if next(covers(sets, n, k), None) is not None)


def rank(sets_json, n, top=12, max_poses=3, rho=.6):
    """Covers with the fewest seats; at most max_poses poses first, then less membrane, then fewer
    poses.  A shared seat counts only if it is a merge the rule accepts:
    V(shared) <= rho * (sum of its members' own membranes)."""
    data = json.loads(sets_json.read_text()); feasible = {}; alone = {}
    for r in data['sets']:
        if len(r['members']) == 1 and r.get('status') == 'built': alone[r['members'][0]] = r['volume_cm3']
    for r in data['sets']:
        grounds = [g for g in r.get('grounds', []) if g['passed'] and g.get('constructed')]
        if not (r.get('feasible') and grounds): continue
        if len(r['members']) > 1 and r['volume_cm3'] > rho*sum(alone[i] for i in r['members']): continue
        feasible[tuple(r['members'])] = dict(r, grounds=grounds)
    rows = []
    for cover in covers(list(feasible), n, fewest_seats(list(feasible), n)):
        for choice in itertools.product(*[feasible[T]['grounds'] for T in cover]):
            rotations, poses = align([g['downs'] for g in choice])
            rows.append(dict(seats=[list(T) for T in cover], angles=[g['angles'] for g in choice], poses=len(poses), files=[g['file'] for g in choice],
                             standing_cm3=sum(g['standing_volume_cm3'] for g in choice),
                             volume_cm3=sum(feasible[T]['volume_cm3'] for T in cover),
                             seat_footprints_cm2=[max(g['footprints_cm2']) for g in choice],
                             rotations=[R.tolist() for R in rotations], downs=[d.tolist() for d in poses]))
    rows.sort(key=lambda r: (r['poses'] > max_poses, len(r['seats']), r['volume_cm3'], r['poses']))
    return rows[:top], feasible


# ---------------------------------------------------------------------------------------------
# The joined model: every task on its slot of one support, one face of the support always down.


def turn_about_vertical(R):
    """Is R a turn about the world vertical (the same face of the support down)?"""
    return np.allclose(R[:, 2], [0, 0, 1], atol=1e-9) and np.allclose(R[2], [0, 0, 1], atol=1e-9)


class JoinedModel(SeatModel):
    """Every task on its seat of one support.  Seat s sits in the support at X[s] (support <- object);
    task k is installed at world_from_support = P_k X_s^-1, a turn about the vertical with the table
    at z = 0, its object at the work point (the arm rises and falls, the support slides and turns)."""
    def __init__(self, c, seats, angles, placements):
        members = [i for T in seats for i in T]; turns = [a for A in angles for a in A]
        rows = [c.record(i, a) for i, a in zip(members, turns)]
        super().__init__(c.base, [r for r, _ in rows], [k for _, k in rows], [c.sweeps[i] for i in members])
        self.seats = [list(T) for T in seats]; self.seat_of = [s for s, T in enumerate(seats) for _ in T]
        self.X = [np.asarray(X, float) for X in placements]
        self.placed = {}

    def solids(self, k):
        """Task k's body, keep-out and exit sweep in the support frame."""
        if k not in self.placed:
            X = self.X[self.seat_of[k]]
            self.placed[k] = (moved_mesh(self.source, X), G.solid(moved_mesh(self.keepouts[k], X)),
                              self.sweeps[k].transform(np.c_[X[:3, :3], X[:3, 3]/G.SCALE]))
        return self.placed[k]

    def layout(self):
        poses, material = [], []
        for k, r in enumerate(self.records):
            X = self.X[self.seat_of[k]]; pose = np.asarray(r['pose'])
            W = np.eye(4); W[:3, :3] = pose[:3, :3]@X[:3, :3].T
            if not turn_about_vertical(W[:3, :3]): raise ValueError(r['id']+': its seat does not rest on the same face')
            W[:2, 3] = pose[:2, 3]-(W[:3, :3]@X[:3, 3])[:2]
            poses.append(W@X); material.append(np.linalg.inv(W))
        return Layout(np.asarray(poses), np.asarray(material), np.tile(self.exit, (len(poses), 1)))

    def prepare(self, layout):
        rows = []
        for k, r in enumerate(self.records):
            P, M, d = layout.poses[k], layout.material[k], layout.exits[k]
            lift = P@np.linalg.inv(r['pose'])
            if not (np.allclose(lift[:3, :3], np.eye(3), atol=1e-9) and np.allclose(lift[:2, 3], 0, atol=1e-9)):
                raise ValueError(r['id']+': pose is not the certified one moved vertically')
            if not np.allclose(M@P, self.X[self.seat_of[k]], atol=1e-9): raise ValueError(r['id']+': not on its seat')
            if not turn_about_vertical(M[:3, :3]) or abs(M[2, 3]) > 1e-12: raise ValueError(r['id']+': table is not z = 0')
            if not np.allclose(d, self.exit, atol=1e-12): raise ValueError('The exit is against the needle')
            body, work, sweep = self.solids(k)
            rows.append(dict(self.world_data(r, P), body=body, work=work, sweep=sweep, direction=M[:3, :3]@d,
                             length=self.length, material=M, pose=P, index=k, id=r['id'],
                             working_face_ids=np.asarray(r['working_face_ids'])))
        return rows

    def audit(self, candidate, *, arm=True, **options):
        result = MembraneModel.audit(self, candidate, **options); began = time.monotonic()
        for t, row in zip(candidate['tasks'], result['tasks']):
            r = self.records[t['index']]; X = self.X[self.seat_of[t['index']]]
            hits = self.arm_hits(moved_mesh(candidate['support'], np.asarray(r['pose'])@np.linalg.inv(X)), r)
            row['arm'] = dict(configurations=4*len(r['certified_face_ids']), faces_hit=hits)
            if hits and arm: raise ValueError(t['id']+': KUKA links meet the support at faces '+str(hits[:10]))
        result['arm_seconds'] = time.monotonic()-began
        return result
