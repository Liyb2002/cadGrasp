"""Step4: combine the independent Step3 supports into one support by random pairwise merges.

Every task starts alone on its Step3 membrane in the object frame (a seat X = I), its exit against the
needle; Step4's keep-out (needle, head and KUKA, step4/keepout.py) is cut from it once.  A sample
holds a set of groups (supports with seats; several tasks may share a seat).  Each step a sample draws
two of its groups A, B and one of Step4's two operations at random:

  Rotational reuse  no material is added.  B's tasks go into A's seats one at a time: the object sits
                    where A already holds one, the support turned to the task's pose, and only that
                    task's keep-out and exit sweep are cut from A to expose its work area.  Each takes
                    the seat, then the exit direction, that keeps the most contact skin.
  Juxtaposition     A and B stand on the table (each on its first pose), centre on centre in x and y,
                    B turned about the vertical by the best of a few coarse turns; the union minus every
                    task's body, keep-out and exit sweep, the largest piece kept.

The candidate is audited in full at once: loads, continuous exits, body and keep-out, the KUKA at every
stored joint set; with one pose the standing support (post and plate) built and audited, with several
the foot-region oracle.  It must also pass the gate bbx(AB) <= gate (bbx(A) + bbx(B)), bbx the
minimum-volume oriented box of the support.  When the audit finds a task's loads unsupported, the
candidate is fine-tuned for the contact skin of every task found so (its parent's material within a
fifth of the membrane thickness of its object) and audited again: Rotational reuse re-chooses each
added task's seat and exit; Juxtaposition slides B on the table, turns it about the vertical and changes
exits ((1+lambda) local search).  Candidates with a required pressure point below another pose's resting
plane are not audited.

Samples are independent and run until one support holds every task or their proposals run out.  The best
support: fewest poses, then smallest bbx, then least material.
"""
import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import FIRST_COMPLETED, wait
from functools import lru_cache
from pathlib import Path
import numpy as np
import trimesh
import manifold3d as md
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from whole_task.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task.operations.geometry import moved_mesh
from whole_task.operations.state import Layout, transform
from whole_task.operations.sweep_cache import sweep_mesh
from whole_task.step3 import independent_geometry as G
from whole_task.step3.exit_geometry import whole_object_sweep
from whole_task.step3.membranes import positive_components
from whole_task.step3.tool_membranes import stem
from .seats import context, construction_length, read_mesh, settled
from .merge import PlacedModel, dilated, move, flat, downs, footprint, turn

FOLDER = 'step4_combine'
POLICY = dict(samples=32,               # independent merge sequences
              gate=.7,                  # bbx(AB) <= gate (bbx(A) + bbx(B)), minimum-volume oriented boxes
              proposals=20,             # merge proposals per sample at most
              seed=11,
              neighbours=8, iterations=3,         # Juxtaposition fine-tuning: lambda, rounds
              turn_step_deg=45.,        # Juxtaposition: coarse turns about the vertical tried centre on centre
              step=.04, step_deg=8.,    # first slide (object diagonals) and turn about the vertical (deg)
              exit_rings_deg=(15., 30.), exit_azimuths=8, exit_step_deg=20.,
              connected=.9,             # the kept piece holds at least this share of the material
              audits=4)                 # full audits per proposal at most


def unit(v):
    return np.asarray(v, float)/np.linalg.norm(v)


def obb(points):
    """Volume (m^3) of the minimum-volume oriented box of points."""
    return float(np.prod(trimesh.bounds.oriented_bounds(np.asarray(points))[1]))


@lru_cache(maxsize=None)
def exit_set(object_name, i, rings=POLICY['exit_rings_deg'], azimuths=POLICY['exit_azimuths']):
    """Task i's exit directions, object frame: against the needle first, then rings about it and
    straight up (world).  Every one rises."""
    c = context(object_name); R = np.asarray(c.base.records[i]['pose'])[:3, :3]; d = -unit(c.base.axis)
    u = unit(np.cross(d, [1., 0., 0.])); v = np.cross(d, u); rows = [d]
    for t in np.radians(rings):
        rows += [np.cos(t)*d+np.sin(t)*(np.cos(p)*u+np.sin(p)*v) for p in 2*np.pi*np.arange(azimuths)/azimuths]
    rows.append(np.array([0., 0., 1.]))
    world = np.asarray([unit(w) for w in rows])
    if (world[:, 2] < np.sin(np.radians(10.))).any(): raise ValueError('exit directions must rise')
    return world@R


_sweeps = {}


def sweep(object_name, i, k):
    """Task i's exit sweep along its direction k, object frame (cached on disk)."""
    if (i, k) not in _sweeps:
        c = context(object_name); v = exit_set(object_name, i)[k]; length = construction_length(c.base)
        _sweeps[i, k] = G.solid(sweep_mesh(ROOT/'whole_task/data'/object_name/FOLDER/'sweeps', c.base.source, v, length,
                                           lambda: G.mesh(whole_object_sweep(c.base.source, v, length)[0]), 'object_frame_v1'))
    return _sweeps[i, k]


class ExitModel(PlacedModel):
    """PlacedModel with each task's own exit (object frame), cut and checked at its slot."""
    def __init__(self, c, slots):
        super().__init__(c, *flat(dict(slots=slots)))
        self.slots = slots; self.e = [exit_set(c.object_name, s['members'][0])[s['exit']] for s in slots]
        self.sweeps = [sweep(c.object_name, s['members'][0], s['exit']) for s in slots]
        self.seats = [[k] for k in range(len(slots))]; self.seat_of = list(range(len(slots)))

    def layout(self, support):
        q = super().layout(support)
        return Layout(q.poses, q.material, np.asarray([P[:3, :3]@e for P, e in zip(q.poses, self.e)]))

    def table_layout(self):
        """One pose, the support in its table frame (table at z = 0): each task turned on the table about
        the vertical and slid so that its object is at the station."""
        poses, material = [], []
        for k, r in enumerate(self.records):
            X = self.X[k]; P = np.asarray(r['pose']); W = np.eye(4); W[:3, :3] = P[:3, :3]@X[:3, :3].T
            if abs(W[2, 2]-1) > 1e-9: raise ValueError(r['id']+': its slot does not rest on the table face')
            W[:2, 3] = P[:2, 3]-(W[:3, :3]@X[:3, 3])[:2]
            poses.append(W@X); material.append(np.linalg.inv(W))
        poses = np.asarray(poses)
        return Layout(poses, np.asarray(material), np.asarray([P[:3, :3]@e for P, e in zip(poses, self.e)]))


# ---------------------------------------------------------------------------------------------
# Per process: each group's support, hull, contact skins and forbidden region, in its own frame.

_groups = {}


def solids(object_name, group):
    key = group['file']
    if key not in _groups:
        c = context(object_name); mesh = read_mesh(ROOT/group['file']); s = G.solid(mesh)
        X = [np.asarray(sl['X'], float) for sl in group['slots']]; D = float(np.linalg.norm(c.base.source.extents))
        _groups[key] = dict(s=s, hull=mesh.vertices[ConvexHull(mesh.vertices).vertices], bounds=mesh.bounds,
                            window=G.box(mesh.bounds[0]-D, mesh.bounds[1]+D),
                            skins=[s ^ move(x, dilated(object_name)) for x in X], pieces={})
    return _groups[key]


@lru_cache(maxsize=None)
def body_and_keepout(object_name, i):
    c = context(object_name); return G.union([G.solid(c.base.source), G.solid(c.record(i, 0.)[1])])


def boxed(value):
    """(value, its axis-aligned box in metres), the box empty for an empty value."""
    b = np.asarray(value.bounding_box(), float).reshape(2, 3)*G.SCALE
    return value, b


def corners(box):
    return np.array([[x, y, z] for x in box[:, 0] for y in box[:, 1] for z in box[:, 2]])


def moved_box(box, C):
    p = transform(corners(box), C); return np.array([p.min(0), p.max(0)])


def meets(a, b):
    """Two boxes overlap (an empty box, with infinite corners, meets nothing)."""
    return bool(np.isfinite(a).all() and np.isfinite(b).all() and (a[0] <= b[1]).all() and (b[0] <= a[1]).all())


def pieces(object_name, group, exits, windowed=True):
    """Every slot's forbidden pieces in the group's frame, each with its box: its body and keep-out, and its
    exit sweep along direction exits[k]; within the group's window (its support's box grown by an object
    diagonal) unless windowed is False."""
    g = solids(object_name, group); out = []
    for k, (sl, e) in enumerate(zip(group['slots'], exits)):
        i = sl['members'][0]; X = np.asarray(sl['X'], float)
        for key, solid in ((('body', k), lambda: body_and_keepout(object_name, i)), (('sweep', k, int(e)), lambda: sweep(object_name, i, int(e)))):
            key = key+(windowed,)
            if key not in g['pieces']:
                value = move(X, solid()); g['pieces'][key] = boxed(value ^ g['window'] if windowed else value)
            out.append(g['pieces'][key])
    return out


def own(object_name, group, exits):
    """The group's support without the sweeps of its changed exits (it already excludes its other regions)."""
    value = solids(object_name, group)['s']
    for sl, e in zip(group['slots'], exits):
        if int(e) != int(sl['exit']): value = value-move(np.asarray(sl['X'], float), sweep(object_name, sl['members'][0], int(e)))
    return value


def construct(object_name, A, B, C, eA, eB):
    """(kept piece, whole volume m^3) of (A u C B) minus every slot's forbidden region.  Each support already
    excludes its own slots' regions (but a changed exit's sweep), so each is cut only by the other group's
    pieces, and only by those whose box meets it: the same solid as subtracting the union of all of them."""
    a, b = solids(object_name, A), solids(object_name, B); D = float(np.linalg.norm(context(object_name).base.source.extents))
    Ci = np.linalg.inv(C); boxA, boxB = a['bounds'], moved_box(b['bounds'], C)
    # Each window must hold the other group's support, or this placement uses the whole regions.
    windowed = ((boxB[0] >= boxA[0]-D).all() and (boxB[1] <= boxA[1]+D).all() and
                (moved_box(boxA, Ci)[0] >= b['bounds'][0]-D).all() and (moved_box(boxA, Ci)[1] <= b['bounds'][1]+D).all())
    sa, sb = own(object_name, A, eA), move(C, own(object_name, B, eB))
    for p, box in pieces(object_name, B, eB, windowed):
        if meets(moved_box(box, C), boxA): sa = sa-move(C, p)
    for p, box in pieces(object_name, A, eA, windowed):
        if meets(box, boxB): sb = sb-p
    parts = positive_components(sa+sb)
    if not parts: return None, 0.
    return max(parts, key=lambda p: float(p.volume())), sum(G.volume(p) for p in parts)


@lru_cache(maxsize=None)
def wrenches(object_name, i):
    """Task i's required wrenches about the centre of mass, in the world orientation of its pose (they do
    not change when the pose moves), its rotation world <- object and the centre of mass, object frame."""
    c = context(object_name); r = c.base.records[i]; P = np.asarray(r['pose'])
    return (c.base.world_data(r, P)['endpoints']['endpoint_need_wrench'], P[:3, :3], np.asarray(c.base.source.center_mass))


def floor_gap(object_name, slots, points):
    """Lowest height (m) of a task's required pressure points above another pose's resting plane, the support
    (points) resting on its lowest point in every pose: ground_oracle.pressure_floor_gap without a layout.
    0 when every task rests on one face."""
    rows = []
    for s in slots:
        W, R, com = wrenches(object_name, s['members'][0]); X = np.asarray(s['X'], float); Q = R@X[:3, :3].T
        rows.append((W, Q, X[:3, :3]@com+X[:3, 3], float((points@Q.T)[:, 2].min())))
    gap = 0.
    for W, Q, centre, low in rows:
        world = Q@centre; world[2] -= low
        p = G.pressure_points(W, world)                        # on the table, world xy
        ground = np.c_[p, np.full(len(p), low)]@Q             # the same points, support frame
        for _, Qj, _, lj in rows:
            if Qj[:, 2]@Q[:, 2] < 1-1e-9: gap = min(gap, float((ground@Qj.T)[:, 2].min()-lj))
    return gap


def evaluate(object_name, A, B, C, eA, eB, axis, limit):
    """Exact construction of one placement and its scores."""
    keep, whole = construct(object_name, A, B, C, eA, eB)
    if keep is None: return None
    a, b = solids(object_name, A), solids(object_name, B); kept = G.volume(keep)
    lost = [G.volume(S-keep) for S in a['skins']]+[G.volume(move(C, S)-keep) for S in b['skins']]
    data = keep.to_mesh64(); points = np.asarray(data.vert_properties[:, :3])*G.SCALE
    slots = combined(A, B, np.asarray(C), eA, eB)
    return dict(C=np.asarray(C), eA=list(eA), eB=list(eB), axis=axis, volume_cm3=kept*1e6, share=kept/max(whole, 1e-30), slots=slots,
                bbx_m3=obb(points), skin_lost_m3=lost, floor_gap_m=floor_gap(object_name, slots, points), limit_m3=limit)


def score(r, policy, focus=()):
    """Lower is better: within the gate, one piece, no pressure point below another pose's resting plane
    (how far below), then the contact skin lost -- by the slots whose loads an
    audit found unsupported (focus), else by every slot -- the most any of them loses, then bbx."""
    lost = max(r['skin_lost_m3'][k] for k in focus) if focus else max(r['skin_lost_m3'])
    return (r['bbx_m3'] > r['limit_m3'], r['share'] < policy['connected'], max(0., -r['floor_gap_m']), lost, r['bbx_m3'])


def usable(r, policy):
    return r['bbx_m3'] <= r['limit_m3'] and r['share'] >= policy['connected'] and r['floor_gap_m'] >= -1e-8


def table_frame(c, group):
    """support -> table for the group's first pose: that face down, its box centred on the origin in x and y,
    its lowest point at z = 0."""
    s = group['slots'][0]; X = np.asarray(s['X'], float); hull = solids(c.object_name, group)['hull']
    R = np.asarray(c.record(s['members'][0], s['angles'][0])[0]['pose'])[:3, :3]@X[:3, :3].T
    p = hull@R.T; T = np.eye(4); T[:3, :3] = R
    T[:3, 3] = [-(p[:, 0].min()+p[:, 0].max())/2, -(p[:, 1].min()+p[:, 1].max())/2, -p[:, 2].min()]
    return T


def juxtaposed(TA, TB, table):
    """C (B's support -> A's): both standing on the table, B slid by (dx, dy) and turned about the vertical (deg)."""
    dx, dy, deg = table; M = np.eye(4); M[:3, :3] = turn(np.array([0., 0., 1.]), deg); M[:2, 3] = dx, dy
    return np.linalg.inv(TA)@M@TB


def juxtapose_eval(object_name, A, B, TA, TB, table, eA, eB, limit):
    r = evaluate(object_name, A, B, juxtaposed(TA, TB, table), eA, eB, None, limit)
    if r is not None: r.update(kind='juxtaposition', table=[float(x) for x in table])
    return r


def nearby_exit(object_name, i, now, rng, policy):
    E = exit_set(object_name, i)
    near = [j for j in range(len(E)) if j != now and np.degrees(np.arccos(np.clip(E[j]@E[now], -1, 1))) <= policy['exit_step_deg']]
    return int(rng.choice(near))


def jitter(object_name, A, B, TA, TB, r, rng, step, deg, policy):
    """A small change of a juxtaposition: slide on the table, turn about the vertical, or one task's exit."""
    D = float(np.linalg.norm(context(object_name).base.source.extents)); (dx, dy, th), eA, eB = r['table'], list(r['eA']), list(r['eB'])
    kind = rng.choice(['translate', 'rotate', 'exit'])
    if kind == 'translate': v = unit(rng.normal(size=2))*step*D*rng.uniform(.25, 1.); dx, dy = dx+v[0], dy+v[1]
    elif kind == 'rotate': th = th+rng.normal()*deg
    else:
        slots = A['slots']+B['slots']; k = int(rng.integers(len(slots)))
        j = nearby_exit(object_name, slots[k]['members'][0], (eA+eB)[k], rng, policy)
        if k < len(eA): eA[k] = j
        else: eB[k-len(eA)] = j
    return juxtapose_eval(object_name, A, B, TA, TB, (dx, dy, th), eA, eB, r['limit_m3'])


def tune(start, move, rng, policy, focus=()):
    """(1+lambda) local search from start by move(r, rng, step, deg); the step halves when no neighbour is
    better.  focus: the slots whose skin counts (audits found their loads unsupported)."""
    current = start; step, deg = policy['step'], policy['step_deg']; seen = []; key = lambda t: score(t, policy, focus)
    for _ in range(policy['iterations']):
        trial = [t for t in (move(current, rng, step, deg) for _ in range(policy['neighbours'])) if t is not None]; seen += trial
        best = min(trial, key=key) if trial else None
        if best is not None and key(best) < key(current): current = best
        else: step, deg = step/2, deg/2
    return current, seen


def seats_of(group):
    """The group's seats: its distinct object placements X (several tasks may share one), each with its first slot."""
    out = []
    for k, s in enumerate(group['slots']):
        X = np.asarray(s['X'], float)
        if not any(np.allclose(X, Y, atol=1e-9) for Y, _ in out): out.append((X, k))
    return out


_added = {}


def added(object_name, A, i, X, e):
    """Task i's keep-out and exit sweep at seat X of A, within A's window (its body is the seat's own)."""
    key = (A['file'], i, np.asarray(X).round(9).tobytes(), int(e))
    if key not in _added:
        g = solids(object_name, A)
        _added[key] = [move(X, body_and_keepout(object_name, i)) ^ g['window'], move(X, sweep(object_name, i, int(e))) ^ g['window']]
    return _added[key]


def carve(object_name, A, eA, adds):
    """A's support without the sweeps of its changed exits and without every added task's keep-out and exit
    sweep at its seat; nothing is added.  adds: [(task, seat X, exit)].  (kept piece, whole volume m^3)."""
    value = own(object_name, A, eA)
    for i, X, e in adds:
        for p in added(object_name, A, i, X, e): value = value-p
    parts = positive_components(value)
    if not parts: return None, 0.
    return max(parts, key=lambda p: float(p.volume())), sum(G.volume(p) for p in parts)


def reuse_eval(object_name, A, B, eA, choice, limit):
    """Rotational reuse with choice {B slot: (A seat, exit)}: carved and scored like a placement; an added
    task rests on its seat's skin."""
    seats = seats_of(A); adds = [(B['slots'][t]['members'][0], seats[k][0], e) for t, (k, e) in sorted(choice.items())]
    keep, whole = carve(object_name, A, eA, adds)
    if keep is None: return None
    a = solids(object_name, A); kept = G.volume(keep); lost = [G.volume(S-keep) for S in a['skins']]
    lost += [lost[seats[k][1]] for t, (k, e) in sorted(choice.items())]
    slots = ([dict(sl, exit=int(e)) for sl, e in zip(A['slots'], eA)] +
             [dict(members=[i], angles=[0.], X=np.asarray(X).tolist(), exit=int(e)) for i, X, e in adds])
    data = keep.to_mesh64(); points = np.asarray(data.vert_properties[:, :3])*G.SCALE
    return dict(kind='rotational_reuse', choice=dict(choice), eA=list(eA), slots=slots, volume_cm3=kept*1e6,
                share=kept/max(whole, 1e-30), bbx_m3=obb(points), skin_lost_m3=lost, limit_m3=limit,
                floor_gap_m=floor_gap(object_name, slots, points))


def reuse(object_name, A, B, rng, policy, limit, focus=(), start=None):
    """B's tasks into A's seats, one at a time (in random order; again in turn from start when repairing):
    each takes the seat, then the exit (its current one or a neighbour), that keeps the most skin.
    Returns (best, every complete evaluation)."""
    eA = [s['exit'] for s in A['slots']]; seats = seats_of(A); key = lambda r: score(r, policy, focus)
    choice = dict(start['choice']) if start else {}; best = start; seen = []
    for t in (rng.permutation(len(B['slots'])) if start is None else range(len(B['slots']))):
        t = int(t); i = B['slots'][t]['members'][0]; e = choice.get(t, (None, B['slots'][t]['exit']))[1]
        rows = [reuse_eval(object_name, A, B, eA, {**choice, t: (k, e)}, limit) for k in range(len(seats))]
        rows = [r for r in rows if r is not None]; seen += rows
        if not rows: return None, seen
        k = min(rows, key=key)['choice'][t][0]
        E = exit_set(object_name, i)
        rows += [reuse_eval(object_name, A, B, eA, {**choice, t: (k, j)}, limit) for j in range(len(E))
                 if j != e and np.degrees(np.arccos(np.clip(E[j]@E[e], -1, 1))) <= policy['exit_step_deg']]
        rows = [r for r in rows if r is not None]; seen += rows[len(seats):]
        best = min(rows, key=key); choice = dict(best['choice'])
    # Only placements of every one of B's tasks are candidates; the earlier ones were steps of the greedy.
    return best, [r for r in seen if len(r['choice']) == len(B['slots'])]


def tag(r):
    return (r['kind'], np.asarray(r.get('C', np.eye(4))).round(9).tobytes(), tuple(r['eA']), tuple(r.get('eB', ())),
            tuple(sorted(r.get('choice', {}).items())))


def combined(A, B, C, eA, eB):
    """The merged group's slots: A's, then B's moved by C, with their exits."""
    return ([dict(s, exit=int(e)) for s, e in zip(A['slots'], eA)] +
            [dict(s, X=(C@np.asarray(s['X'], float)).tolist(), exit=int(e)) for s, e in zip(B['slots'], eB)])


def stand(c, slots, membrane):
    """step4/merge.stand with each task's own exit: the membrane resting at plate height in the table
    frame T, an upright post under its lowest part up to a needle radius below the lowest object, and a
    plate under the post and every task's pressure hull, widened by the edge margin.  Settled.
    Returns (standing support, slots in the table frame, the model, its layout)."""
    from .ground_oracle import pressure_rows
    from whole_task.step3.tool_keepout import disc
    p = c.base.policy; h = p['plate_thickness_m']; t = p['mold_outward_displacement_m']; clear = 1.1*c.base.setup['needle_radius_m']
    s0 = slots[0]; X0 = np.asarray(s0['X'], float)
    R = np.asarray(c.record(s0['members'][0], s0['angles'][0])[0]['pose'])[:3, :3]@X0[:3, :3].T
    T = np.eye(4); T[:3, :3] = R; T[2, 3] = h-float((membrane.vertices@R.T)[:, 2].min())
    table = [dict(s, X=(T@np.asarray(s['X'], float)).tolist()) for s in slots]
    placed = moved_mesh(membrane, T)
    low = min(float(moved_mesh(c.base.source, np.asarray(s['X'])).vertices[:, 2].min()) for s in table)
    top = min(h+t, low-clear); V = placed.vertices[placed.vertices[:, 2] <= top]
    if top <= h or len(V) < 3: raise ValueError('no membrane below the objects to stand on')
    model = ExitModel(c, table); q = model.table_layout()
    rows = pressure_rows(model, q, model.prepare(q))
    ring = disc(np.array([0., 0., 1.]), p['plate_edge_margin_m'], 16)[:, :2]
    F = (np.vstack([V[:, :2], *[r['ground_material'][:, :2] for r in rows]])[:, None]+ring[None]).reshape(-1, 2)
    post = md.Manifold.hull_points(np.vstack([np.c_[V[:, :2], np.zeros(len(V))], np.c_[V[:, :2], np.full(len(V), top)]])/G.SCALE)
    plate = md.Manifold.hull_points(np.vstack([np.c_[F, np.zeros(len(F))], np.c_[F, np.full(len(F), h)]])/G.SCALE)
    support = settled(G.mesh(G.union([G.solid(placed), post, plate])))
    if support is None: raise ValueError('standing support does not settle under manifold')
    return support, table, model, q


def certify(c, slots, membrane):
    """One pose: the standing support (membrane, post, plate) built, every task audited in full on it, its
    ground faces enclosing every pressure hull -- the post and plate stay below the objects, so its contacts
    are the membrane's.  Several poses: no pressure hull below another resting plane (checked first, it is
    cheap), every task audited in full on the membrane, and the foot-region oracle.  Returns facts."""
    from .ground_oracle import pressure_rows, pressure_floor_gap, ground_enclosure, foot_regions
    out = dict(poses=len(downs(c, dict(slots=slots))), membrane_cm3=float(membrane.volume*1e6), bbx_m3=obb(membrane.vertices))
    if out['poses'] == 1:
        support, table, joined, q = stand(c, slots, membrane)
        standing = dict(layout=q, support=support, tasks=joined.prepare(q), construction={})
        joined.audit(standing)
        rows = ground_enclosure(support, pressure_rows(joined, q, standing['tasks']))
        if not all(r['passed'] for r in rows): raise ValueError('a pressure hull leaves the plate')
        T = np.asarray(table[0]['X'])@np.linalg.inv(np.asarray(slots[0]['X']))
        out.update(standing=support, table_from_support=T, standing_cm3=float(support.volume*1e6), footprint_cm2=1e4*footprint(support))
    else:
        model = ExitModel(c, slots); candidate = dict(layout=model.layout(membrane), support=membrane, construction={})
        candidate['tasks'] = model.prepare(candidate['layout'])
        if pressure_floor_gap(model, candidate['layout']) < -1e-8: raise ValueError('Required pressure hull crosses another resting plane')
        model.audit(candidate)
        found = next(foot_regions(model, candidate), None)
        if found is None: raise ValueError('no foot region')
        out.update(foot_padding_m=found[0])
    return out


def accept(object_name, A, B, r, file):
    """One candidate in full: build, settle, refine exits, gate, audit, stand.  Saves the group's file."""
    c = context(object_name); began = time.monotonic(); slots = r['slots']
    out = dict(kind=r['kind'], eA=r['eA'], screened_bbx_m3=r['bbx_m3'], skin_lost_m3=r['skin_lost_m3'],
               **({'C': np.asarray(r['C']).tolist(), 'eB': r['eB'], 'table': r['table']} if r['kind'] == 'juxtaposition' else
                  {'choice': {str(t): list(v) for t, v in r['choice'].items()}}))
    try:
        if sorted(i for x in slots for i in x['members']) != sorted(i for x in A['slots']+B['slots'] for i in x['members']):
            raise ValueError('the candidate does not hold every task of both groups')
        if r['kind'] == 'juxtaposition': keep, _ = construct(object_name, A, B, np.asarray(r['C']), r['eA'], r['eB'])
        else:
            seats = seats_of(A)
            keep, _ = carve(object_name, A, r['eA'], [(B['slots'][t]['members'][0], seats[k][0], e) for t, (k, e) in sorted(r['choice'].items())])
        membrane = settled(G.mesh(keep.simplify(0.)))
        if membrane is None: raise ValueError('merged membrane does not settle under manifold')
        model = ExitModel(c, slots)
        candidate = dict(layout=model.layout(membrane), support=membrane, construction={})
        candidate['tasks'] = model.prepare(candidate['layout']); model.refine_exits(candidate)
        membrane = settled(candidate['support'])
        if membrane is None: raise ValueError('refined membrane does not settle under manifold')
        if obb(membrane.vertices) > r['limit_m3']: raise ValueError('the merge does not pass the bbx gate')
        facts = certify(c, slots, membrane)
        arrays = dict(vertices=membrane.vertices, faces=membrane.faces)
        if 'standing' in facts:
            arrays.update(standing_vertices=facts['standing'].vertices, standing_faces=facts['standing'].faces,
                          table_from_support=facts['table_from_support'])
        file.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(file, **arrays)
        out.update({k: v for k, v in facts.items() if k not in ('standing', 'table_from_support')},
                   passed=True, file=str(file.relative_to(ROOT)), slots=slots)
    except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as ex:
        out.update(passed=False, reason=str(ex)[:300])
    out['seconds'] = time.monotonic()-began
    return out


def propose(args):
    """One merge proposal of a sample.  Rotational reuse: B's tasks into A's seats, nothing added.
    Juxtaposition: both standing on the table, centre on centre.  The candidate is audited in full at once;
    when an audit fails, the candidate is fine-tuned for the tasks found unsupported and audited again."""
    object_name, A, B, op, seed, file, policy = args; began = time.monotonic(); c = context(object_name)
    rng = np.random.default_rng(seed); limit = policy['gate']*(A['bbx_m3']+B['bbx_m3'])
    eA, eB = [s['exit'] for s in A['slots']], [s['exit'] for s in B['slots']]
    if op == 'rotational_reuse':
        first, built = reuse(object_name, A, B, rng, policy, limit)
        improve = lambda r, focus: reuse(object_name, A, B, rng, policy, limit, focus, r)
    else:
        TA, TB = table_frame(c, A), table_frame(c, B)
        move = lambda r, rng_, step, deg: jitter(object_name, A, B, TA, TB, r, rng_, step, deg, policy)
        # Centre on centre, B turned about the vertical in coarse steps; the best is the start.
        built = [r for r in (juxtapose_eval(object_name, A, B, TA, TB, (0., 0., float(t)), eA, eB, limit)
                             for t in np.arange(0., 360., policy['turn_step_deg'])) if r is not None]
        first = min(built, key=lambda r: score(r, policy)) if built else None
        if first is not None and not usable(first, policy):
            first, seen = tune(first, move, rng, policy); built += seen
        improve = lambda r, focus: tune(r, move, rng, policy, focus)
    ids = [c.base.ids[s['members'][0]] for s in A['slots']+B['slots']]; done = set(); results = []; repairs = []; focus = []
    r = first if first is not None and usable(first, policy) else None
    while r is not None and len(results) < policy['audits']:
        done.add(tag(r)); result = accept(object_name, A, B, r, file.with_name(f'{file.stem}_{len(results)}.npz')); results.append(result)
        if result['passed']: break
        name = result['reason'].split(':')[0]
        if 'force/moment certificate failed' in result['reason'] and name in ids:
            # Every slot found unsupported so far counts, so the search does not trade one for another.
            focus = sorted(set(focus) | {ids.index(name)})
        better, seen = improve(r, tuple(focus)); built += seen
        lost = lambda x: max(x['skin_lost_m3'][k] for k in focus) if focus else max(x['skin_lost_m3'])
        repairs.append(dict(tasks=[ids[k] for k in focus], skin_lost_m3=[lost(r), lost(better)] if better else None))
        rest = [x for x in built if x is not None and usable(x, policy) and tag(x) not in done]
        r = min(rest, key=lambda x: score(x, policy, tuple(focus))) if rest else None
    stats = dict(operation=op, parents=[A['name'], B['name']], limit_m3=limit, built=len(built),
                 within_gate=sum(usable(x, policy) for x in built if x is not None), first_usable=bool(first and usable(first, policy)),
                 repairs=repairs, audits=[{k: v for k, v in x.items() if k != 'slots'} for x in results], seconds=time.monotonic()-began)
    passed = next((x for x in results if x['passed']), None)
    return dict(stats, passed=passed is not None, result=passed)


# ---------------------------------------------------------------------------------------------

def initial_one(args):
    """Task i alone: its Step3 membrane in the object frame, Step4's keep-out cut once, settled and audited."""
    object_name, i, folder = args; c = context(object_name); r = c.base.records[i]; P = np.asarray(r['pose'])
    source = ROOT/'whole_task/data'/object_name/'step3_tool_membranes'/(stem(r['id'])+'.npz')
    with np.load(source) as z: world = trimesh.Trimesh(z['vertices'], z['faces'], process=False)
    mesh = moved_mesh(world, np.linalg.inv(P)); value = G.solid(mesh)-body_and_keepout(object_name, i)
    parts = positive_components(value); keep = max(parts, key=lambda p: float(p.volume()))
    membrane = settled(G.mesh(keep.simplify(0.)))
    slots = [dict(members=[i], angles=[0.], X=np.eye(4).tolist(), exit=0)]
    facts = certify(c, slots, membrane)
    file = Path(folder)/'groups'/(stem(r['id'])+'.npz'); file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file, vertices=membrane.vertices, faces=membrane.faces)
    print('START', r['id'], round(facts['membrane_cm3'], 1), 'cm3, bbx', round(facts['bbx_m3']*1e6), 'cm3', flush=True)
    return dict(name=stem(r['id']), slots=slots, file=str(file.relative_to(ROOT)), volume_cm3=facts['membrane_cm3'],
                bbx_m3=facts['bbx_m3'], poses=1, step3_cm3=float(world.volume*1e6), chain=[])


def as_group(result, A, B, name):
    """An accepted merge as a group; `chain` lists every merge that made it, gate ratio included."""
    step = dict(name=name, parents=[A['name'], B['name']], operation=result['operation'],
                bbx_ratio=result['bbx_m3']/(A['bbx_m3']+B['bbx_m3']), bbx_m3=result['bbx_m3'],
                parent_bbx_m3=[A['bbx_m3'], B['bbx_m3']], volume_cm3=result['membrane_cm3'],
                parent_cm3=[A['volume_cm3'], B['volume_cm3']], poses=result['poses'],
                exits_changed=int(sum(a['exit'] != b['exit'] for a, b in zip(result['slots'], A['slots']+B['slots']))),
                file=result['file'], slots=result['slots'], parent_files=[A['file'], B['file']],
                parent_slots=[A['slots'], B['slots']], footprint_cm2=result.get('footprint_cm2'))
    if result['operation'] == 'juxtaposition':
        step.update(table_move=dict(dx_m=result['table'][0], dy_m=result['table'][1], turn_deg=result['table'][2]))
    else:
        # Which of A's seats each of B's tasks went to (A's slot index of that seat).
        seats = seats_of(A); step.update(seats={B['slots'][int(t)]['members'][0]: seats[k][1] for t, (k, e) in result['choice'].items()})
    return dict(name=name, slots=result['slots'], file=result['file'], volume_cm3=result['membrane_cm3'],
                bbx_m3=result['bbx_m3'], poses=result['poses'], standing_cm3=result.get('standing_cm3'),
                footprint_cm2=result.get('footprint_cm2'), chain=A['chain']+B['chain']+[step])


def plain(x):
    """x with numpy scalars and arrays as Python values, for JSON."""
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.generic): return x.item()
    return x


def best_first(groups):
    return sorted(groups, key=lambda g: (g['poses'], g['bbx_m3'], g['volume_cm3']))


def pin(cpus):
    """One CPU per worker: manifold spreads each Boolean over every CPU it may use, so a pool of unpinned
    workers oversubscribes the machine (a construction took 2-5 s instead of 0.8 s)."""
    os.sched_setaffinity(0, {cpus.get()})


def pinned_pool(workers):
    context_ = multiprocessing.get_context('spawn'); cpus = sorted(os.sched_getaffinity(0)); queue = context_.Queue()
    for k in range(workers): queue.put(cpus[k % len(cpus)])
    return ProcessPoolExecutor(max_workers=workers, mp_context=context_, initializer=pin, initargs=(queue,))


def run(object_name='B', workers=22, policy=POLICY, resume=False):
    """resume: continue the samples saved in search.json (every group there was accepted in full) with the
    current proposals; the log keeps the earlier ones."""
    began = time.monotonic(); c = context(object_name); folder = ROOT/'whole_task/data'/object_name/FOLDER
    folder.mkdir(parents=True, exist_ok=True); n = len(c.base.ids); log = []; resumed = []
    with pinned_pool(workers) as executor:
        if resume:
            state = json.loads((folder/'search.json').read_text()); start, log = state['start'], state['log']
            samples = [dict(s, merges=list(s['merges'])) for s in state['samples']]
            resumed = state.get('resumed', [])+[dict(proposals=len(log), seconds=state['seconds'])]; began -= state['seconds']
        else:
            start = list(executor.map(initial_one, [(object_name, i, folder) for i in range(n)]))
            samples = [dict(id=k, groups=list(start), proposals=0, merges=[]) for k in range(policy['samples'])]
        pending = {}

        def submit(s):
            rng = np.random.default_rng([policy['seed'], s['id'], s['proposals']])
            a, b = rng.choice(len(s['groups']), 2, replace=False); op = ['juxtaposition', 'rotational_reuse'][int(rng.integers(2))]
            file = folder/'groups'/f"s{s['id']:02d}_p{s['proposals']:02d}.npz"
            job = (object_name, s['groups'][a], s['groups'][b], op, [policy['seed'], s['id'], s['proposals'], 1], file, policy)
            pending[executor.submit(propose, job)] = (s, int(a), int(b)); s['proposals'] += 1

        for s in samples:
            if len(s['groups']) > 1 and s['proposals'] < policy['proposals']: submit(s)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                s, a, b = pending.pop(future); r = future.result(); A, B = s['groups'][a], s['groups'][b]
                log.append(dict(sample=s['id'], proposal=s['proposals']-1, **{k: v for k, v in r.items() if k != 'result'}))
                if r['passed']:
                    g = as_group(dict(r['result'], operation=r['operation']), A, B, A['name']+'+'+B['name'])
                    s['groups'] = [x for k, x in enumerate(s['groups']) if k not in (a, b)]+[g]
                    s['merges'].append(g['chain'][-1]['name'])
                print('SAMPLE', s['id'], 'proposal', s['proposals']-1, r['operation'], A['name'], '+', B['name'],
                      'ACCEPT' if r['passed'] else 'reject', *([round(g['chain'][-1]['bbx_ratio'], 3), 'bbx ratio', g['poses'], 'poses']
                                                             if r['passed'] else []),
                      'groups', len(s['groups']), round(r['seconds']), 's', flush=True)
                if len(s['groups']) > 1 and s['proposals'] < policy['proposals']: submit(s)
            write_json(folder/'search.json', plain(dict(policy=policy, start=start, log=log, seconds=time.monotonic()-began, resumed=resumed,
                                                  samples=[dict(id=s['id'], proposals=s['proposals'], merges=s['merges'],
                                                                groups=s['groups']) for s in samples])))
    finals = [dict(s['groups'][0], sample=s['id']) for s in samples if len(s['groups']) == 1]
    report = dict(object=object_name, policy=policy, start=start, samples_done=len(finals), samples=len(samples), resumed=resumed,
                  finals=best_first(finals), log=log, seconds=time.monotonic()-began)
    if finals:
        best = best_first(finals)[0]; report.update(status='combined', group=best)
        print('STEP4 COMBINED sample', best['sample'], 'poses', best['poses'], 'bbx', round(best['bbx_m3']*1e6), 'cm3, membrane',
              round(best['volume_cm3'], 1), 'cm3', flush=True)
    else: report.update(status='not_combined', group=None)
    write_json(folder/'report.json', plain(report))
    if finals: validate(object_name)
    return report


def validate(object_name='B'):
    """Disk replay of the chosen support: its chain's gates; the membrane and its standing support audited
    again, the standing support rebuilt from the membrane and equal to the saved one."""
    c = context(object_name); folder = ROOT/'whole_task/data'/object_name/FOLDER
    report = json.loads((folder/'report.json').read_text()); g = report['group']; policy = report['policy']
    for step in g['chain']:
        if step['bbx_ratio'] > policy['gate']+1e-12: raise ValueError(step['name']+': gate not met')
        if step['operation'] not in ('juxtaposition', 'rotational_reuse'): raise ValueError('unknown operation')
    if sorted(i for s in g['slots'] for i in s['members']) != list(range(len(c.base.ids))): raise ValueError('not every task once')
    membrane = read_mesh(ROOT/g['file']); facts = certify(c, g['slots'], membrane)
    if 'standing' in facts:
        with np.load(ROOT/g['file']) as z: saved = trimesh.Trimesh(z['standing_vertices'], z['standing_faces'], process=False)
        d = G.volume(G.solid(facts['standing'])-G.solid(saved))+G.volume(G.solid(saved)-G.solid(facts['standing']))
        if d > c.base.tolerance: raise ValueError(f'standing support differs from the saved one: {d} m^3')
    result = dict(status='passed', report_sha256=sha256(folder/'report.json'), membrane_sha256=sha256(ROOT/g['file']),
                  poses=facts['poses'], bbx_m3=facts['bbx_m3'], membrane_cm3=facts['membrane_cm3'],
                  standing_cm3=facts.get('standing_cm3'), merges=len(g['chain']))
    write_json(folder/'validation.json', plain(result)); print('STEP4 COMBINE DISK REPLAY PASS', flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', default='B'); p.add_argument('--workers', type=int, default=22)
    p.add_argument('--samples', type=int, default=POLICY['samples']); p.add_argument('--validate-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    validate(a.object) if a.validate_only else run(a.object, a.workers, dict(POLICY, samples=a.samples), a.resume)
