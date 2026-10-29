"""Step4: combine the Step3 supports into one, one task at a time, in random samples.

Every task starts alone on its Step3 membrane in the object frame (seat X = I), its exit against the
needle; Step4's keep-out (needle, head and KUKA, step4/keepout.py) is cut from it once.  A group is one
support with seats; a seat holds the object at X (support <- object) for the tasks that use it.

A sample starts from a task drawn at random; step by step it draws one more task T at random and joins it
to its group G with one of Step4's two operations:

  Rotational reuse  adds no material.  T goes into one of G's seats: its object exactly where that seat's
                    object is (it is the same object, so it fits); G is opened for T -- T's keep-out and
                    exit sweep are cut from it -- and the support is turned to install T.  T first leaves
                    through the seat's existing exit channel (a seat task's exit, if it rises for T), which
                    cuts nothing more.
  Juxtaposition     opens a new seat.  T's own membrane keeps the installation of a seat's task (the same
                    support face down), placed with its centre of mass on that task's (the most overlap),
                    half-way, or its work area's centre on that task's, at a few turns; the two supports are
                    joined, minus every task's body, keep-out and exit sweep.

Each step a sample draws its operation: Rotational reuse with a share falling from 80% at the first step to
50% at the last (it fails more often, so it is tried more), else Juxtaposition.  It tries only that operation,
on at most policy['seats'] seats, those used by the fewest tasks first.  When Step1 split a big task into
seat groups (step1/subdivide.py: tasks of one group keep the same face up and leave its hold zone alone), the
draws follow them: a task whose group already has a seat in the support is drawn policy['same_group_odds']
times as often and joins by Rotational reuse with share policy['same_group_reuse'], into its group's seats
first; a task of a group with no seat yet joins by Rotational reuse with share policy['other_group_reuse'].  Every join passes the gate
bbx(G+T) <= gate (bbx(G)+bbx(T)), bbx the minimum-volume oriented box, so that a join shares material
instead of laying T beside G.

Every task works at its own certified pose: the placing arm installs the support for each task so that its
object faces the work arm (the KUKA, its needle direction fixed) exactly as Step1 certified it.  So the
KUKA keeps its certified joints and every keep-out stays fixed in the object frame.

When the paired forces and moments do not hold, Step4's basic operations restore them by gradient descent
(finite differences and backtracking, step4/gradient_repair.block_step, plus the secant step to zero loss) on
the paired-load residual: every task's fixed set of paired demands classified on the actual contacts of the
exact construction, the residual their mean normalized projection loss.  It is the only loss; the rest are
binary: a construction outside the gate, or with a task's required pressure point below another installation's
resting plane (its feet would cross the table there), is infeasible to the step, and skin narrower than
skin.WIDTH (20 mm) holds nothing (its contacts do not count; the oracle cuts it once).  In Rotational reuse the
seat is fixed, so only T's exit direction moves; in Juxtaposition T's seat turns about T's needle direction
(Pose rotation: the placing arm then turns the support about the object, which may rest the support on another
face) and translates.  A demand the full audit finds unsupported joins the set and the descent goes on.

A join is accepted when -- after what is not joined to its largest piece and its skin narrower than
skin.WIDTH along the object's surface are cut away, once (step4/skin.py's pruned; nothing is added, and narrow
skin the cut leaves is Step5's to make thick) -- every task passes the full audit
(32,768 paired samples and the continuous outer targets, the continuous exit of every source face, body and
keep-out, the KUKA at every stored joint set) and the support stands: with one pose the standing support
(post and plate) is built and audited, with several the foot-region oracle.  These checks are binary, so when
one fails the join tries its next start that holds (at most policy['audits'] audits).

The samples run on their own, none waiting for another.  The next task is drawn with odds 1 + the number of
its joins that have failed so far, so the tasks that are hard to join come early, while the group is small (most
constrained first).  A sample whose task joins nowhere continues from a random state that has already succeeded
at that step (else one at its own step), the odds halving with every seat it has more than the fewest (the most
Rotational reuse), and draws again.  After policy['retries'] failures in a row while no state has reached the
next step it goes back to a state one step earlier than its last time back (one, then two, ...), at most
policy['backtracks'] times; then it stops.  The search stops at policy['enough'] distinct supports holding every
task.  When every sample has stopped with none, the gate grows by policy['gate_step'] (at most to
policy['gate_max']) and they go on from where they stopped: a join that passed stands, one that failed is tried
again under the new gate.  The best support: the fewest seats, then the
smallest bbx, then the least material.
"""
import argparse
import collections
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
from pathlib import Path
import numpy as np
import trimesh
import manifold3d as md
from scipy.spatial.transform import Rotation

from subdivide_task.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from subdivide_task.common.force_cone import tangent_frame
from subdivide_task.operations.geometry import moved_mesh, contact_surface
from subdivide_task.operations.state import Layout, transform
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step3.independent_physics import certificate, contacts
from subdivide_task.step3.legacy import kernel as K
from subdivide_task.step3.membranes import positive_components
from subdivide_task.step3.tool_membranes import stem
from .gradient_repair import block_step
from .seats import context, construction_length, read_mesh, settled, same_surface
from .merge import PlacedModel, move, flat, downs, footprint, turn
from .skin import WIDTH, pruned, piece, thin_cells, skin as skin_of
from .fast_contacts import contacts as guide_contacts

FOLDER = 'step4_combine'
DOWN = np.array([0., 0., -1.])
POLICY = dict(samples=32, seed=11,
              reuse_share=(.8, .5),     # the share of Rotational reuse draws at the first and the last step
              gate=.7,                  # bbx(G+T) <= gate (bbx(G) + bbx(T)), minimum-volume oriented boxes
              gate_step=.1, gate_max=1.,  # every sample stopped with no support: the gate grows by gate_step, to gate_max
              demands=24,               # guidance demands per task, of the samples and of the continuous targets
              rounds=4,                 # gradient rounds (every basic operation once per round)
              feedback=2,               # times an audit's unsupported demands join the guidance
              audits=3,                 # full audits per seat: after a binary failure the next start that holds is tried
              exit_rise_deg=10.,        # every exit rises at least this much in its installation
              exit_deg=(.5, 5.),        # Exit redirection: finite difference, first step
              pose_deg=(.5, 5.),        # Pose rotation: finite difference, first step
              pose_starts=(0., 60., 120., 180., -120., -60.),   # Juxtaposition starts from the best of these turns
              exit_starts=((15., 30.), 8),                      # Rotational reuse starts from the best exit on these rings
              scales=3,                 # while a measured gradient is zero, the finite difference grows x4, this many times
              translation=(.004, .03),  # Translation, object diagonals: finite difference, first step
              retries=6,                # a sample goes back after this many failed joins in a row, none yet past them
              backtracks=3,             # ... at most this many times, one step further each time; then it stops
              seats=3,                  # a join tries at most this many seats (those used by the fewest tasks first)
              enough=3,                 # the search stops at this many distinct supports holding every task
              grouped=True,             # Step1's seat groups (subdivide.py) steer the draws, when the plan has them:
              same_group_odds=8.,       # ... a task whose group already has a seat in the support is drawn this much more often,
              same_group_reuse=.9,      # ... and joins it by Rotational reuse with this share (into its group's seats first);
              other_group_reuse=.2)     # ... a task of a group with no seat yet, with this share


def unit(v):
    return np.asarray(v, float)/np.linalg.norm(v)


def obb(points):
    """Volume (m^3) of the minimum-volume oriented box of points."""
    return float(np.prod(trimesh.bounds.oriented_bounds(np.asarray(points))[1]))


def about(Q, p):
    """The rotation Q about the point p, 4 x 4."""
    M = np.eye(4); M[:3, :3] = Q; M[:3, 3] = p-Q@p; return M


# ---------------------------------------------------------------------------------------------
# A task: its pose turned about the needle, its exit, its keep-out, its work area's centre.

def default_exit(object_name, i):
    """Task i's exit against the needle, object frame (the same at every turn, which is about the needle)."""
    return -needle(object_name, i)


def pose(object_name, i):
    """World <- object: task i's certified pose."""
    return np.asarray(context(object_name).base.records[i]['pose'], float)


def needle(object_name, i):
    """Task i's nominal needle direction, object frame."""
    c = context(object_name); return unit(pose(object_name, i)[:3, :3].T@c.base.axis)


def rises(object_name, i, e, policy=POLICY):
    """Does exit e (object frame) rise at least exit_rise_deg at task i's pose?"""
    return bool((pose(object_name, i)[:3, :3]@e)[2] >= np.sin(np.radians(policy['exit_rise_deg']))-1e-12)


def direction_key(e):
    return tuple(float(x) for x in np.round(unit(e), 12))


def exact_sweep(body, direction, length):
    """body and every leading face's prism along length direction, step3/exit_geometry.whole_object_sweep exactly,
    but each wall two neighbouring prisms share cut along the same diagonal on both sides (from the bottom corner of
    the lower vertex index): the union glues identical triangles.  Step3's prisms cut each wall along their own
    diagonal, and the union of thousands of such coplanar, differently cut walls dropped whole prisms (B+8pose S2:
    80-380 mm^3, where Step5 then added material).  The sweep of a connected body is connected: its largest piece,
    without the zero-volume shells the union leaves.  Built in memory (about 1 s): a round trip through Mesh64
    re-cuts its long thin walls and drops prisms again (B+03_kuka +X: 0.5 cm^3)."""
    d = np.asarray(direction, float); T = body.triangles; F = body.faces; parts = [G.solid(body)]
    for f in np.flatnonzero(np.cross(T[:, 1]-T[:, 0], T[:, 2]-T[:, 0])@d > 0):
        faces = [[2, 1, 0], [3, 4, 5]]
        for p, q in ((0, 1), (1, 2), (2, 0)):
            faces += [[p, q, q+3], [p, q+3, p+3]] if F[f, p] < F[f, q] else [[p, q, p+3], [q, q+3, p+3]]
        points = np.vstack([T[f], T[f]+length*d])/G.SCALE
        parts.append(md.Manifold(md.Mesh64(np.ascontiguousarray(points), np.asarray(faces, np.uint64))))
    while len(parts) > 1: parts = [G.union(parts[i:i+8]) for i in range(0, len(parts), 8)]
    return max(parts[0].decompose(), key=lambda p: p.volume())


@lru_cache(maxsize=64)
def _sweep(object_name, key):
    c = context(object_name); return exact_sweep(c.base.source, np.asarray(key), construction_length(c.base))


def sweep(object_name, e):
    """The exit sweep along direction e, object frame (the latest ones kept in memory)."""
    return _sweep(object_name, direction_key(e))


@lru_cache(maxsize=None)
def body_and_keepout(object_name, i):
    """Task i's body and keep-out (needle, head and KUKA at its certified joints), object frame."""
    c = context(object_name); return G.union([G.solid(c.base.source), G.solid(c.record(i, 0.)[1])])


@lru_cache(maxsize=None)
def membrane(object_name, i):
    """Task i's Step3 membrane, object frame (before Step4's keep-out)."""
    c = context(object_name); r = c.base.records[i]
    with np.load(ROOT/'subdivide_task/data'/object_name/'step3_tool_membranes'/(stem(r['id'])+'.npz')) as z:
        world = trimesh.Trimesh(z['vertices'], z['faces'], process=False)
    return G.solid(moved_mesh(world, np.linalg.inv(np.asarray(r['pose']))))


@lru_cache(maxsize=None)
def centre(object_name, i):
    """The area-weighted centre of task i's work area, object frame."""
    s = context(object_name).base.source; ids = np.asarray(context(object_name).base.records[i]['working_face_ids'])
    w = s.area_faces[ids]; return (s.triangles_center[ids]*w[:, None]).sum(0)/w.sum()


def slot(i, X, e):
    """Task i at its certified pose (no turn: the placing arm installs the support for it), seat X, exit e."""
    return dict(members=[int(i)], angles=[0.], X=np.asarray(X, float).tolist(), exit=list(direction_key(e)))


# ---------------------------------------------------------------------------------------------
# A group: its support, its forbidden region and its seats, in its own frame.

class ExitModel(PlacedModel):
    """PlacedModel with each task's own exit (object frame), cut and checked at its slot."""
    def __init__(self, c, slots):
        super().__init__(c, *flat(dict(slots=slots)))
        self.slots = slots; self.e = [unit(s['exit']) for s in slots]
        self.sweeps = [sweep(c.object_name, e) for e in self.e]
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


def seats(group):
    """The group's seats: [(X, [its slots])], in order of first use."""
    out = []
    for k, s in enumerate(group['slots']):
        X = np.asarray(s['X'], float)
        for Y, ks in out:
            if np.allclose(X, Y, atol=1e-9): ks.append(k); break
        else: out.append((X, [k]))
    return out


_groups = {}
PAD = .002                      # m: a cutter clipped to a box this much wider than a solid cuts it as the whole cutter does


def bounds(value):
    """A solid's bounding box (low, high), m."""
    b = np.asarray(value.bounding_box(), float)*G.SCALE; return b[:3], b[3:]


def clipped(value, box):
    """value within box (low, high) widened by PAD."""
    return value ^ G.box(np.asarray(box[0])-PAD, np.asarray(box[1])+PAD)


def forbidden(object_name, slots, box=None):
    """The union of every slot's body, keep-out and exit sweep, support frame.  Given box (low, high), each part
    clipped to it first: a solid inside the box is cut by it exactly as by the whole, and the parts far from the
    box -- the KUKA's links, an exit sweep's length: most of the triangles -- cost the Booleans nothing (at 8 poses
    the union of one task's keep-out and exit sweep took 17 s, the whole region 107 s)."""
    F = []
    for s in slots:
        X = np.asarray(s['X'], float)
        F += [move(X, body_and_keepout(object_name, s['members'][0])), move(X, sweep(object_name, s['exit']))]
    return G.union(F if box is None else [clipped(f, box) for f in F])


def group_solids(object_name, group):
    """The group's solids, its frame, kept for the latest few groups: its support 's', and what solids() and
    nearby() add."""
    key = group['file']
    if key not in _groups:
        if len(_groups) >= 4: _groups.pop(next(iter(_groups)))
        _groups[key] = dict(s=G.solid(read_mesh(ROOT/group['file'])))
    return _groups[key]


def solids(object_name, group):
    """The group's support and the union of every slot's body, keep-out and exit sweep, its frame."""
    g = group_solids(object_name, group)
    if 'F' not in g: g['F'] = forbidden(object_name, group['slots'])
    return g


def nearby(object_name, group, value):
    """The group's forbidden region for cutting value: clipped (once per group) to the box of its support widened by
    the object's diagonal when value lies in that box, else to value's own box."""
    g = group_solids(object_name, group); low, high = bounds(value)
    if 'near' not in g:
        D = float(np.linalg.norm(context(object_name).base.source.extents)); lo, hi = bounds(g['s'])
        g['near'] = (lo-D, hi+D, forbidden(object_name, group['slots'], (lo-D, hi+D)))
    lo, hi, F = g['near']
    return F if (low >= lo).all() and (high <= hi).all() else forbidden(object_name, group['slots'], (low, high))


def largest(value):
    """(the largest piece, its share of the material), or (None, 0)."""
    parts = positive_components(value)
    if not parts: return None, 0.
    keep = max(parts, key=lambda p: float(p.volume()))
    return keep, G.volume(keep)/max(sum(G.volume(p) for p in parts), 1e-30)


def reuse(object_name, group, i, X, e):
    """Rotational reuse: task i in the seat X, the group's support opened by its keep-out and exit sweep along e."""
    s = group_solids(object_name, group)['s']; return largest(s-forbidden(object_name, [slot(i, X, e)], bounds(s)))


def juxtaposed(object_name, group, k, i, angle, t):
    """Juxtaposition's seat for task i beside seat k's first task j: at angle 0, i's installation is j's
    (the same support face down) and its work area's centre is on j's; Pose rotation turns the seat by angle
    (deg) about i's needle direction through that centre, Translation moves it by t (support frame)."""
    X, ks = seats(group)[k]; j = group['slots'][ks[0]]['members'][0]; c_i = centre(object_name, i)
    Q = X[:3, :3]@pose(object_name, j)[:3, :3].T@pose(object_name, i)[:3, :3]
    out = np.eye(4); out[:3, :3] = Q; out[:3, 3] = transform(centre(object_name, j), X)+np.asarray(t, float)-Q@c_i
    return out@about(turn(needle(object_name, i), angle), c_i)


def juxtapose(object_name, group, i, X, e):
    """Juxtaposition: task i's Step3 membrane at X joined to the group's support, each cut by the other's
    forbidden region (each already excludes its own)."""
    s = group_solids(object_name, group)['s']; placed = move(X, alone(object_name, i, direction_key(e)))
    return largest((s-forbidden(object_name, [slot(i, X, e)], bounds(s)))+(placed-nearby(object_name, group, placed)))


@lru_cache(maxsize=None)
def alone(object_name, i, e):
    """Task i's Step3 membrane less its own body, keep-out and exit sweep along e, object frame."""
    m = membrane(object_name, i); return m-forbidden(object_name, [slot(i, np.eye(4), e)], bounds(m))


# ---------------------------------------------------------------------------------------------
# The paired-load residual on the actual contacts.

def picks(n, count):
    return np.unique(np.linspace(0, n-1, count).astype(int))


_judged = {}


def signature(mesh):
    """The area, first and second area moments of mesh's triangles, to 10 significant digits: the same for the
    same surface however a Boolean triangulates it (a triangle's second moment is a/12 (sum v v^T + s s^T), s the
    sum of its vertices)."""
    T = mesh.triangles; a = mesh.area_faces; v = T.sum(1)
    second = np.einsum('t,tij->ij', a/12, np.einsum('tki,tkj->tij', T, T)+np.einsum('ti,tj->tij', v, v))
    return tuple(float(f'{x:.10g}') for x in np.r_[a.sum(), a@v/3, second[np.triu_indices(3)]])


def residual(object_name, slots, support, extra, thin=None, policy=POLICY):
    """(loss, covered, rows): every task's guidance demands -- `demands` samples and continuous targets
    spread over their index, and the ones audits found unsupported (extra[task] = (samples, targets)) --
    classified on the actual contacts of support.  A task's loss is its demands' mean normalized projection
    loss onto its contact cone (0 when all hold); the loss is the mean over tasks.  Contacts come only from
    the support's triangles near the task's body (contact_surface), less those over its seat's skin narrower
    than WIDTH (thin[slot], skin cells: they hold nothing), so a task whose nearby surface is one already judged
    (signature), with the same demands, keeps that judgement: a change elsewhere is not recomputed.  The contact
    polygons are the kernel's (step4/fast_contacts.py)."""
    from continuous_support.projection import project_demands
    c = context(object_name); losses = []; rows = []
    for k, s in enumerate(slots):
        i = s['members'][0]; r = c.base.records[i]; P = pose(object_name, i)
        data = c.base.world_data(r, P); body = data['body_world']; domain = data['domain']
        world = moved_mesh(support, P@np.linalg.inv(np.asarray(s['X'], float)))
        S, E = data['samples']['need_wrench'], data['endpoints']['endpoint_need_wrench']
        more = extra.get(i, ((), ()))
        si = np.union1d(picks(len(S), policy['demands']), np.asarray(more[0], int))
        ei = np.union1d(picks(len(E), policy['demands']), np.asarray(more[1], int))
        near = contact_surface(body, world)
        if thin is not None and thin[k].any():
            cell = skin_of(c)[0].tree.query(transform(near.triangles_center, np.linalg.inv(P)))[1]
            near = trimesh.Trimesh(near.vertices, near.faces[~thin[k][cell]], process=False)
        key = (i, signature(near), si.tobytes(), ei.tobytes())
        if key not in _judged:
            if len(_judged) > 256: _judged.pop(next(iter(_judged)))
            try:
                _, _, rays, scale = guide_contacts(body, near, domain.work_ids, domain.com)
                targets = np.vstack([S[si], E[ei]])*scale; held, _ = K.J.classify(rays, targets)
                loss = 0. if held.all() else float(np.mean(project_demands(rays, K.J.C.U.target(targets, rays.shape[1]))['normalized_losses']))
                _judged[key] = (loss, int(held.sum()), len(targets))
            except RuntimeError: _judged[key] = (1., 0, len(si)+len(ei))
        loss, passed, count = _judged[key]
        losses.append(loss); rows.append(dict(task=r['id'], loss=loss, passed=passed, demands=count))
    return float(np.mean(losses)), all(r['passed'] == r['demands'] for r in rows), rows


def unsupported(model, candidate, count=16):
    """{task: (sample ids, target ids)} of the paired demands the actual contacts of candidate do not hold,
    at most count of each, spread over their index."""
    out = {}
    for t, (i, s) in zip(candidate['tasks'], [(s['members'][0], s) for s in model.slots]):
        world = moved_mesh(candidate['support'], np.linalg.inv(t['material']))
        try: proof, data = certificate(t['body_world'], contact_surface(t['body_world'], world), t['domain'], t['endpoints'], t['samples'])
        except RuntimeError: continue
        if proof['passed']: continue
        bad_s, bad_e = np.flatnonzero(~data['sample_passed']), np.flatnonzero(~data['endpoint_passed'])
        out[i] = (bad_s[picks(len(bad_s), count)].tolist() if len(bad_s) else [], bad_e[picks(len(bad_e), count)].tolist() if len(bad_e) else [])
    return out


# ---------------------------------------------------------------------------------------------
# One join: build, restore the loads with the basic operations, accept.

def steady(mesh, rounds=6):
    """settled(mesh), or -- when the no-op Booleans alternate between two triangulations (coplanar prism
    seams) -- the later of the two, a Boolean's own output; it is audited in full after, as always.  (Moving
    vertices to break the cycle loses contacts: they are exact, to 1e-10 m.)  None if neither within rounds."""
    support = settled(mesh, rounds)
    if support is not None: return support
    far = G.box(mesh.bounds[1]+.05, mesh.bounds[1]+.06); seen = [mesh]
    for _ in range(rounds):
        seen.append(G.mesh((G.solid(seen[-1])+far)-far))
        if len(seen) > 2 and same_surface(seen[-1], seen[-3]): return seen[-1]
    return None


def stand(c, slots, membrane):
    """step4/merge.stand with each task's own exit: the membrane resting at plate height in the table
    frame T, an upright post under its lowest part up to a needle radius below the lowest object, and a
    plate under the post and every task's pressure hull, widened by the edge margin.  Settled.
    Returns (standing support, slots in the table frame, the model, its layout)."""
    from .ground_oracle import pressure_rows
    from subdivide_task.step3.tool_keepout import disc
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
    support = steady(G.mesh(G.union([G.solid(placed), post, plate])))
    if support is None: raise ValueError('standing support does not settle under manifold')
    return support, table, model, q


def certify(c, slots, membrane):
    """One pose: the standing support (membrane, post, plate) built, every task audited in full on it, its
    ground faces enclosing every pressure hull -- the post and plate stay below the objects, so its contacts
    are the membrane's.  Several poses: no pressure hull below another resting plane (checked first, it is
    cheap), the foot-region oracle (then: it costs less than the audit and fails more often -- at 8 poses most
    failed audits were this), every task audited in full on the membrane.  Returns facts."""
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
        model.exit_paths(candidate)             # every task's exit travel on this support, as the audit sets it
        found = next(foot_regions(model, candidate), None)
        if found is None: raise ValueError('no foot region')
        model.audit(candidate)
        out.update(foot_padding_m=found[0])
    return out


def accept(object_name, slots, keep, limit, file):
    """One construction in full: settle, refine exits, gate, audit, stand.  Saves the group's file.  On a load
    failure, also the demands the actual contacts do not hold."""
    c = context(object_name); began = time.monotonic(); out = dict(file=None)
    try:
        support = steady(G.mesh(keep.simplify(0.)))
        if support is None: raise ValueError('merged membrane does not settle under manifold')
        model = ExitModel(c, slots)
        # What is not joined to the largest piece goes, and the skin narrower than WIDTH, once; then the oracle
        # judges what is left.
        support, out['skin_pruning'] = pruned(c, slots, support, forbidden(object_name, slots, support.bounds))
        candidate = dict(layout=model.layout(support), support=support, construction={})
        candidate['tasks'] = model.prepare(candidate['layout']); model.refine_exits(candidate)
        support = steady(candidate['support'])
        if support is None:
            # Coplanar prism seams can alternate between two triangulations.
            # Intersect them, remove only the existing piece() seam crumbs,
            # and still require the stability check (steady) and full audit.
            raw = candidate['support']; native = G.solid(raw)
            far = G.box(raw.bounds[1]+.05, raw.bounds[1]+.06)
            alternate = G.mesh((native+far)-far)
            cleaned = piece(native ^ G.solid(alternate), 'unstable membrane splits the support')
            support = steady(cleaned)
            out['mesh_conditioning'] = dict(method='intersect_boolean_triangulations_then_remove_seam_crumbs',
                                            before_cm3=float(raw.volume*1e6),
                                            after_cm3=None if support is None else float(support.volume*1e6))
        if support is None: raise ValueError('refined membrane does not settle under manifold')
        if obb(support.vertices) > limit: raise ValueError('the join does not pass the bbx gate')
        try: facts = certify(c, slots, support)
        except ValueError as ex:
            if 'force/moment' in str(ex):
                candidate.update(support=support, layout=model.layout(support)); candidate['tasks'] = model.prepare(candidate['layout'])
                out['unsupported'] = unsupported(model, candidate)
            raise
        arrays = dict(vertices=support.vertices, faces=support.faces)
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


def scaled_step(evaluate, current, perturb, dimension, difference, step, policy):
    """block_step, its finite difference grown x4 while the measured gradient is zero: contacts change by whole
    triangles, so the residual is flat under small changes and a small difference sees no direction.  Its steps
    are a fixed length halved four times; near a solution the step that reaches zero loss along the measured
    gradient (secant: loss / |gradient|) is far shorter, and every halved step overshoots.  Then that step, doubled
    and then itself, is tried too (the gradient's probes are the evaluator's, cached)."""
    for k in range(policy['scales']):
        d = difference*4**k; first = max(step, 2*d); new, row = block_step(evaluate, current, perturb, dimension, d, first)
        norm = float(np.linalg.norm(row['gradient']))
        if not row['accepted'] and norm > 1e-12 and 2*current['loss']/norm < first/8:
            new, near = block_step(evaluate, current, perturb, dimension, d, 2*current['loss']/norm, backtracks=2)
            row = dict(near, line_search=row['line_search']+near['line_search'])
        if row['accepted'] or norm > 1e-12: break
    return new, row


def descend(evaluate, current, blocks, policy):
    """Gradient rounds over the basic operations until the guidance demands hold; a construction that breaks a
    constraint is infeasible to the step (evaluate gives None).  blocks: (name, perturb (base) -> (vector ->
    params), dimension, difference, step).  Returns (current, record)."""
    record = []
    for _ in range(policy['rounds']):
        changed = False
        for name, perturb, dimension, difference, step in blocks:
            if current['covered']: return current, record
            current, row = scaled_step(evaluate, current, perturb(current['params']), dimension, difference, step, policy)
            record.append(dict(operation=name, accepted=row['accepted'], loss=current['loss'], gradient=row['gradient'],
                               difference=row['difference']))
            changed |= row['accepted']
        if not changed: break
    return current, record


def evaluator(build):
    """build(params) -> (keep, slots, rows...) wrapped for block_step, cached by params; None where it is
    infeasible (a broken constraint, or a Boolean the kernel cannot resolve), the reasons counted in .why."""
    cache = {}; why = collections.Counter()
    def evaluate(params):
        key = tuple(np.round(np.r_[params[0], np.ravel(params[1])], 12)) if isinstance(params, tuple) else tuple(np.round(params, 12))
        if key not in cache:
            try: cache[key] = build(params)
            except (ValueError, RuntimeError) as ex: cache[key] = None; why[str(ex)[:100]] += 1
            if len(cache) > 32: cache.pop(next(iter(cache)))
        return cache[key]
    evaluate.why = why
    return evaluate


@lru_cache(maxsize=None)
def wrenches(object_name, i):
    """Task i's required wrenches about its centre of mass, in the world orientation of its certified pose."""
    c = context(object_name); return c.base.world_data(c.base.records[i], pose(object_name, i))['endpoints']['endpoint_need_wrench']


def floor_gap(object_name, slots, points):
    """The oracle's first condition without a layout (ground_oracle.pressure_floor_gap): the lowest height (m) of a
    task's required pressure points above another installation's resting plane, the support (points) resting on its
    lowest point in every installation; 0 when every task rests on one part of the support."""
    com = context(object_name).base.source.center_mass; rows = []
    for s in slots:
        i = s['members'][0]; X = np.asarray(s['X'], float); Q = pose(object_name, i)[:3, :3]@X[:3, :3].T   # world <- support
        rows.append((wrenches(object_name, i), Q, X[:3, :3]@com+X[:3, 3], float((points@Q.T)[:, 2].min())))
    gap = 0.
    for W, Q, centre, low in rows:
        world = Q@centre; world[2] -= low
        ground = np.c_[G.pressure_points(W, world), np.full(len(W), low)]@Q          # on the table, support frame
        for _, Qj, _, lj in rows:
            if Qj[:, 2]@Q[:, 2] < 1-1e-9: gap = min(gap, float((ground@Qj.T)[:, 2].min()-lj))
    return gap


def scored(object_name, slots, keep, limit, extra, params, policy):
    """A construction for the gradient: its loss is the paired-load residual alone, covered when the guidance
    demands hold.  The rest are binary constraints, checked first since they are cheap: outside the gate, or with a
    task's required pressure point below another installation's resting plane (the oracle's first condition for
    standing: its feet would cross the table there), it is infeasible (ValueError).  Skin narrower than WIDTH
    (step4/skin.py) holds nothing: contacts on it do not count, however wide the rest is; the oracle cuts it once."""
    c = context(object_name); mesh = G.mesh(keep); bbx = obb(mesh.vertices)
    if bbx > limit: raise ValueError('outside the bbx gate')
    gap = floor_gap(object_name, slots, mesh.vertices)
    if gap < -1e-8: raise ValueError('a required pressure point below another resting plane')
    loss, held, rows = residual(object_name, slots, mesh, extra, thin_cells(c, slots, mesh), policy)
    return dict(params=params, keep=keep, slots=slots, loss=loss, covered=held, volume_cm3=float(mesh.volume*1e6),
                bbx_m3=bbx, floor_gap_m=gap, rows=rows)


def try_reuse(object_name, group, i, k, limit, file, policy):
    """Rotational reuse into seat k; the loads restored by Exit redirection of task i."""
    c = context(object_name); X = seats(group)[k][0]; extra = {}
    row = dict(operation='rotational_reuse', seat=k)
    # Every exit sweep only removes more than task i's keep-out does, and removed material (or skin it narrows)
    # holds nothing more, so if the group's support opened by the keep-out alone cannot hold every task's loads,
    # no exit direction will.  Only the loads bound it: the gate and the floors can still come right as the exit
    # removes more, so they are judged on each exit.
    s = group_solids(object_name, group)['s']; opened, _ = largest(s-clipped(move(X, body_and_keepout(object_name, i)), bounds(s)))
    if opened is None: return dict(row, passed=False, reason='the keep-out removes the whole seat')
    try:
        mesh = G.mesh(opened); opened_slots = group['slots']+[slot(i, X, default_exit(object_name, i))]
        bound, held, _ = residual(object_name, opened_slots, mesh, extra, thin_cells(c, opened_slots, mesh), policy)
    except (ValueError, RuntimeError) as ex: return dict(row, passed=False, reason='opened by the keep-out alone: '+str(ex)[:120])
    row['bound_loss'] = bound
    if not held: return dict(row, passed=False, reason='opened by the keep-out alone, the seat cannot hold every task')

    def build(e):
        e = unit(e)
        if not rises(object_name, i, e, policy): return None
        keep, _ = reuse(object_name, group, i, X, e)
        if keep is None: return None
        return scored(object_name, group['slots']+[slot(i, X, e)], keep, limit, extra, e, policy)

    def exit_redirection(e):
        axes = tangent_frame(e)
        return lambda v: unit(Rotation.from_rotvec(np.cross(e, v[0]*axes[0]+v[1]*axes[1])).apply(e))
    blocks = [('exit_redirection', exit_redirection, 2, *np.radians(policy['exit_deg']))]
    # Leaving through the seat's existing exit channel cuts nothing more, so the exits of the seat's tasks come
    # first; then the default and rings about it (every one rising), in this order for restore.  The residual is
    # flat in pieces under exit changes, so a start that already holds is worth more than a descent.
    d = default_exit(object_name, i); u, v = tangent_frame(d); rings, azimuths = policy['exit_starts']
    starts = [unit(group['slots'][j]['exit']) for j in seats(group)[k][1]]+[d]+[
        unit(np.cos(t)*d+np.sin(t)*(np.cos(f)*u+np.sin(f)*v)) for t in np.radians(rings) for f in 2*np.pi*np.arange(azimuths)/azimuths]
    return restore(object_name, build, [e for e in starts if rises(object_name, i, e, policy)], blocks, extra, limit, file, row, policy)


def try_juxtapose(object_name, group, i, k, limit, file, policy):
    """Juxtaposition beside seat k; the loads restored by Pose rotation and Translation of task i."""
    c = context(object_name); D = float(np.linalg.norm(c.base.source.extents)); e = default_exit(object_name, i); extra = {}
    row = dict(operation='juxtaposition', seat=k)

    def build(params):
        angle, t = float(params[0]), np.asarray(params[1], float)
        X = juxtaposed(object_name, group, k, i, angle, t)
        keep, _ = juxtapose(object_name, group, i, X, e)
        if keep is None: return None
        return scored(object_name, group['slots']+[slot(i, X, e)], keep, limit, extra, (angle, t), policy)

    def pose_rotation(p):
        return lambda v: (p[0]+np.degrees(v[0]), p[1])
    def translation(p):
        return lambda v: (p[0], np.asarray(p[1])+D*np.asarray(v))
    blocks = [('pose_rotation', pose_rotation, 1, *np.radians(policy['pose_deg'])),
              ('translation', translation, 3, *policy['translation'])]
    # Every turn, its work-area centre on the seat task's (t = 0), its centre of mass on that task's (the most
    # overlap, the smallest bbx) and half-way, in this order for restore.
    X, ks = seats(group)[k]; com = c.base.source.center_mass; starts = []
    for a in policy['pose_starts']:
        to_mass = transform(com, X)-transform(com, juxtaposed(object_name, group, k, i, a, np.zeros(3)))
        starts += [(float(a), f*to_mass) for f in (1., .5, 0.)]
    return restore(object_name, build, starts, blocks, extra, limit, file, row, policy)


def restore(object_name, build, starts, blocks, extra, limit, file, row, policy):
    """The starts in order: each feasible start whose guidance demands hold goes to the full audit at once.  The
    oracle's checks are binary, so when one fails (the support cannot stand, an exit, the skin), the next start is
    tried, at most policy['audits'] audits; when no start holds, the descent starts from the best feasible one.  On
    a load failure, the audit's unsupported demands join the guidance and the descent goes on.  Guidance demands
    are some of the audit's on the same contacts, so a construction whose guidance does not hold is not audited."""
    began = time.monotonic(); evaluate = evaluator(build); row.update(gradient=[], audits=[]); built = []

    def attempt(current):
        nonlocal evaluate
        row.update(start_loss=current['loss'], start=current['params'], starts_tried=len(built))
        for turn_ in range(policy['feedback']+1):
            current, record = descend(evaluate, current, blocks, policy); row['gradient'].append(record)
            if not current['covered']:
                row['audits'].append(dict(passed=False, reason=f"the gradient leaves the guidance unresolved (loss {current['loss']:.3g})")); return None
            result = accept(object_name, current['slots'], current['keep'], limit, file)
            row['audits'].append(dict({k: v for k, v in result.items() if k not in ('slots', 'unsupported')}, guided_loss=current['loss']))
            if result['passed']:
                return dict(row, passed=True, result=result, params=current['params'], end_loss=current['loss'], seconds=time.monotonic()-began)
            if not result.get('unsupported'): return None
            for i, (s, t) in result['unsupported'].items():
                old = extra.get(i, ([], [])); extra[i] = (sorted(set(old[0]) | set(s)), sorted(set(old[1]) | set(t)))
            evaluate = evaluator(build); current = evaluate(current['params'])
            if current is None: return None
        return None

    for start in (starts if isinstance(starts, list) else [starts]):
        x = evaluate(start)
        if x is None: continue
        built.append(x)
        if x['covered']:
            out = attempt(x)
            if out is not None: return out
            if len(row['audits']) >= policy['audits']: break
    else:
        if built and not row['audits']:
            out = attempt(min(built, key=lambda x: x['loss']))
            if out is not None: return out
    if not built:
        why = evaluate.why.most_common(1)
        return dict(row, passed=False, reason='no feasible start'+(': '+why[0][0] if why else ''), seconds=time.monotonic()-began)
    return dict(row, passed=False, reason=row['audits'][-1]['reason'] if row['audits'] else 'unresolved', seconds=time.monotonic()-began)


def task_groups(object_name, ids):
    """Each task's seat group from Step1's subdivision (data/<run>/step1_reach/plan.json), or None."""
    path = ROOT/'subdivide_task/data'/object_name/'step1_reach/plan.json'
    if not path.exists(): return None
    group = {t['id']: t.get('group') for t in json.loads(path.read_text())['tasks']}
    out = [group.get(i) for i in ids]
    return None if any(g is None for g in out) else out


def seated(group, groups):
    """The Step1 groups with a seat in the support."""
    return {groups[m] for s in group['slots'] for m in s['members']}


def join(args):
    """Task i joins the group by the drawn operation, on at most policy['seats'] seats, those of its own Step1
    group first (when the plan has groups), then those used by the fewest tasks.  The first accepted is the join."""
    object_name, group, single, i, operation, file, seed, policy = args; began = time.monotonic(); rng = np.random.default_rng(seed)
    limit = policy['gate']*(group['bbx_m3']+single['bbx_m3']); order = seats(group); tg = policy.get('task_groups')
    own = lambda k: tg is not None and any(tg[m] == tg[i] for j in order[k][1] for m in group['slots'][j]['members'])
    ranked = sorted(range(len(order)), key=lambda k: (not own(k), len(order[k][1]), rng.random()))[:policy['seats']]
    attempts = []
    for attempt in [dict(rotational_reuse=try_reuse, juxtaposition=try_juxtapose)[operation]]:
        for k in ranked:
            r = attempt(object_name, group, i, k, limit, file.with_name(f'{file.stem}_{len(attempts)}.npz'), policy)
            attempts.append({key: v for key, v in r.items() if key != 'result'})
            print('  JOIN', group['name'], '+', single['name'], operation, 'seat', k, 'PASS' if r['passed'] else 'fail: '+r.get('reason', '')[:90],
                  round(r.get('seconds', 0)), 's', flush=True)
            if r['passed']:
                return dict(passed=True, task=i, operation=operation, seat=k, params=r['params'], result=r['result'], attempts=attempts,
                            limit_m3=limit, gate=policy['gate'], seconds=time.monotonic()-began)
    return dict(passed=False, task=i, operation=operation, attempts=attempts, limit_m3=limit, gate=policy['gate'],
                seconds=time.monotonic()-began)


# ---------------------------------------------------------------------------------------------
# Samples.

def initial_one(args):
    """Task i alone, conditioned and audited by the same construction as a join."""
    object_name, i, folder = args; c = context(object_name); r = c.base.records[i]
    keep, _ = largest(membrane(object_name, i)-body_and_keepout(object_name, i))
    slots = [slot(i, np.eye(4), default_exit(object_name, i))]
    file = Path(folder)/'groups'/(stem(r['id'])+'.npz'); file.parent.mkdir(parents=True, exist_ok=True)
    if keep is None: raise ValueError(r['id']+': the keep-out removes the initial membrane')
    facts = accept(object_name, slots, keep, np.inf, file)
    if not facts['passed']: raise ValueError(r['id']+': initial membrane: '+facts['reason'])
    print('START', r['id'], round(facts['membrane_cm3'], 1), 'cm3, bbx', round(facts['bbx_m3']*1e6), 'cm3', flush=True)
    return dict(name=stem(r['id']), slots=slots, file=str(file.relative_to(ROOT)), volume_cm3=facts['membrane_cm3'],
                bbx_m3=facts['bbx_m3'], poses=1, chain=[], skin_pruning=facts['skin_pruning'],
                mesh_conditioning=facts.get('mesh_conditioning'))


def as_group(r, group, single):
    """An accepted join as a group; `chain` lists every join that made it, gate ratio included."""
    result = r['result']; name = group['name']+'+'+single['name']; X = np.asarray(result['slots'][-1]['X'])
    step = dict(name=name, parents=[group['name'], single['name']], operation=r['operation'], seat=r['seat'], gate=r['gate'],
                seat_tasks=len(seats(dict(slots=result['slots']))[r['seat']][1]) if r['operation'] == 'rotational_reuse' else 1,
                bbx_ratio=result['bbx_m3']/(group['bbx_m3']+single['bbx_m3']), bbx_m3=result['bbx_m3'],
                parent_bbx_m3=[group['bbx_m3'], single['bbx_m3']], volume_cm3=result['membrane_cm3'],
                parent_cm3=[group['volume_cm3'], single['volume_cm3']], poses=result['poses'],
                seat_turn_deg=float(r['params'][0]) if r['operation'] == 'juxtaposition' else 0.,
                exit_from_default_deg=float(np.degrees(np.arccos(np.clip(np.asarray(result['slots'][-1]['exit'])@default_exit_cached(single), -1, 1)))),
                rotation_deg=float(np.degrees(Rotation.from_matrix(X[:3, :3]).magnitude())), translation_m=X[:3, 3].tolist(),
                attempts=len(r['attempts']), file=result['file'], slots=result['slots'], parent_files=[group['file'], single['file']],
                parent_slots=[group['slots'], single['slots']], footprint_cm2=result.get('footprint_cm2'))
    return dict(name=name, slots=result['slots'], file=result['file'], volume_cm3=result['membrane_cm3'], bbx_m3=result['bbx_m3'],
                poses=result['poses'], standing_cm3=result.get('standing_cm3'), footprint_cm2=result.get('footprint_cm2'),
                chain=group['chain']+[step])


def default_exit_cached(single):
    return np.asarray(single['slots'][0]['exit'])


def plain(x):
    """x with numpy scalars and arrays as Python values, for JSON."""
    if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.generic): return x.item()
    return x


def best_first(groups):
    """The fewest seats (the most Rotational reuse) first, then the smallest bbx, then the least material."""
    return sorted(groups, key=lambda g: (len(seats(g)), g['bbx_m3'], g['volume_cm3']))


def pin(cpus):
    """One CPU per worker: manifold spreads each Boolean over every CPU it may use, so a pool of unpinned
    workers oversubscribes the machine (a construction took 2-5 s instead of 0.8 s)."""
    os.sched_setaffinity(0, {cpus.get()})


def idle_first(cpus, seconds=.5):
    """cpus, the least busy first (idle share over a short window, /proc/stat), so pinned workers do not
    share a CPU with other pinned work."""
    def times():
        rows = {}
        for line in Path('/proc/stat').read_text().splitlines():
            name, *values = line.split()
            if name.startswith('cpu') and name != 'cpu': rows[int(name[3:])] = np.array(values[:8], float)
        return rows
    try:
        before = times(); time.sleep(seconds); after = times()
    except (OSError, ValueError): return sorted(cpus)
    def idle(k):
        d = after[k]-before[k]; return (d[3]+d[4])/max(d.sum(), 1.)
    return sorted(cpus, key=lambda k: (-idle(k), k))


def pinned_pool(workers):
    context_ = multiprocessing.get_context('spawn'); cpus = idle_first(os.sched_getaffinity(0)); queue = context_.Queue()
    for k in range(workers): queue.put(cpus[k % len(cpus)])
    return ProcessPoolExecutor(max_workers=workers, mp_context=context_, initializer=pin, initargs=(queue,))


def run(object_name='B', workers=22, policy=POLICY):
    """Every sample starts from a random task and joins one task at a time, on its own: no sample waits for
    another.  A task is drawn with odds 1 + its failed joins so far.  A join that succeeds goes on; one that fails
    continues from a state that has already succeeded at that step (else one at the sample's own step), the odds
    halving with every seat more than the fewest, and draws again.  After policy['retries'] failures in a row with
    no state yet at the next step a sample goes back one step further than its last time back, at most
    policy['backtracks'] times, then stops.  Samples copy each other's states, so
    they often draw a join already made: a join is the same for the same group, task and operation, so it runs
    once and every later draw of it takes its result; a draw of a join still running draws again (only when every
    join left is running does it wait for one).  When every sample has stopped with no support holding every task,
    the gate grows by policy['gate_step'] (to at most policy['gate_max']) and the samples go on from where they
    stopped; a join that failed under a smaller gate is made again.  The search stops at policy['enough'] distinct supports holding
    every task; the joins still running then are dropped."""
    began = time.monotonic(); c = context(object_name); folder = ROOT/'subdivide_task/data'/object_name/FOLDER
    folder.mkdir(parents=True, exist_ok=True); n = len(c.base.ids); rng = np.random.default_rng(policy['seed']); log = []
    executor = pinned_pool(workers); done = {}                     # done: the distinct supports holding every task
    try:
        start = list(executor.map(initial_one, [(object_name, i, folder) for i in range(n)]))
        samples, reached, running = [], {0: []}, {}               # reached[step]: (group, left, sample) that got there
        made, waiting, ready = {}, {}, []                          # (group file, task, operation): its result; samples waiting
        failed = collections.Counter()                             # failed[task]: its distinct joins that failed
        for k in range(policy['samples']):
            first = int(rng.integers(n)); samples.append(dict(id=k, group=start[first], left=[i for i in range(n) if i != first], step=0, attempts=0, fails=0, back=0))
            reached[0].append((samples[-1]['group'], samples[-1]['left'], k))

        groups = task_groups(object_name, c.base.ids) if policy.get('grouped') else None
        if groups: print('GROUPS', dict(zip(c.base.ids, groups)), flush=True)

        def draw(left, group):
            w = np.array([1.+failed[i] for i in left])
            if groups:
                have = seated(group, groups); w = w*np.array([policy['same_group_odds'] if groups[i] in have else 1. for i in left])
            return int(rng.choice(left, p=w/w.sum()))

        def carry(at):
            states = reached[at]; k = np.array([len(seats(g)) for g, _, _ in states], float); w = .5**(k-k.min())
            return states[int(rng.choice(len(states), p=w/w.sum()))]

        gate = policy['gate']

        def submit(s):
            first, last = policy['reuse_share']; share = first+(last-first)*s['step']/max(1, n-2)
            for _ in range(2*len(s['left'])):       # a join another sample is running: draw another (none waits)
                i = draw(s['left'], s['group'])
                if groups: share = policy['same_group_reuse'] if groups[i] in seated(s['group'], groups) else policy['other_group_reuse']
                o = 'rotational_reuse' if rng.random() < share else 'juxtaposition'
                key = (s['group']['file'], i, o)
                if key not in waiting: break
            if key in made and (made[key]['passed'] or made[key]['gate'] == gate): ready.append((s, i, made[key], True)); return
            if key in waiting: waiting[key].append((s, i)); return
            s['attempts'] += 1; file = folder/'groups'/f"s{s['id']:02d}_step{s['step']+1}_a{s['attempts']}.npz"; waiting[key] = []
            running[executor.submit(join, (object_name, s['group'], start[i], i, o, file, [policy['seed'], s['id'], s['step']+1, s['attempts']],
                                           dict(policy, gate=gate, task_groups=groups)))] = (s, i, key)
        for s in samples: submit(s)
        while True:
            while (running or ready) and len(done) < policy['enough']:
                if not ready:
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    for job in finished:
                        s, i, key = running.pop(job); made[key] = job.result(); failed[i] += not made[key]['passed']
                        ready += [(s, i, made[key], False)]+[(w, j, made[key], True) for w, j in waiting.pop(key)]
                while ready:
                    s, i, r, again = ready.pop(0); step = s['step']+1
                    log.append(dict({key: v for key, v in r.items() if key != 'result'}, sample=s['id'], step=step, again=again,
                                    group=s['group']['name'], task=start[i]['name'], seconds_since_start=time.monotonic()-began,
                                    **(dict(seconds=0.) if again else {})))
                    print('SAMPLE', s['id'], 'step', step, s['group']['name'], '+', start[i]['name'], r['operation'],
                          'ACCEPT' if r['passed'] else 'fails', '(made before)' if again else '', round(time.monotonic()-began), 's', flush=True)
                    if r['passed']:
                        s.update(group=as_group(r, s['group'], start[i]), left=[x for x in s['left'] if x != i], step=step, fails=0); s.pop('copied_from', None)
                        reached.setdefault(step, []).append((s['group'], list(s['left']), s['id']))
                    else:
                        # A state that already succeeded at this step carries the sample on (progress: its failures
                        # start over); with none yet it draws again from a state at its own step.  After
                        # policy['retries'] such failures in a row it goes back further than last time, at most
                        # policy['backtracks'] times, then stops.
                        at = step if reached.get(step) else s['step']; s['fails'] = 0 if at == step else s['fails']+1
                        if s['fails'] > policy['retries']:
                            if s['back'] >= policy['backtracks'] or s['step'] == 0: s['stopped'] = True; continue
                            s['back'] += 1; at = max(0, s['step']-s['back']); s['fails'] = 0
                            print('SAMPLE', s['id'], 'back to step', at, flush=True)
                        g, left, source = carry(at)
                        s.update(group=g, left=list(left), step=at, copied_from=source)
                    if s['left']: submit(s)
                    else: done.setdefault(s['group']['file'], s['id'])
                write_json(folder/'search.json', plain(dict(policy=policy, start=start, log=log, seconds=time.monotonic()-began,
                           samples=[dict(id=s['id'], group=s['group'], left=s['left'], step=s['step'], copied_from=s.get('copied_from'),
                                         back=s['back'], stopped=s.get('stopped', False)) for s in samples], gate=gate)))
            # Every sample stopped with no support holding every task: the gate grows and they go on from where
            # they stopped (a join that failed is made again under it).
            if done or gate >= policy['gate_max']-1e-9: break
            gate = round(gate+policy['gate_step'], 6)
            print('GATE', gate, round(time.monotonic()-began), 's', flush=True)
            for s in samples: s.update(stopped=False, fails=0, back=0); submit(s)
    finally:                                                       # enough: the joins still running are dropped
        for process in list(getattr(executor, '_processes', {}).values()): process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
    finals = [dict(s['group'], sample=s['id']) for s in samples if not s['left']]
    report = dict(object=object_name, policy=policy, start=start, samples=len(samples), samples_done=len(finals),
                  finals=best_first(finals), log=log, gate=gate, seconds=time.monotonic()-began)
    if finals:
        best = best_first(finals)[0]; report.update(status='combined', group=best)
        print('STEP4 COMBINED sample', best['sample'], 'bbx', round(best['bbx_m3']*1e6), 'cm3, membrane',
              round(best['volume_cm3'], 1), 'cm3', flush=True)
    else: report.update(status='not_combined', group=None)
    write_json(folder/'report.json', plain(report))
    if finals: validate(object_name)
    return report


def validate(object_name='B'):
    """Disk replay of the chosen support: every join's operation and gate; the membrane and its standing
    support audited again, the standing support rebuilt from the membrane and equal to the saved one."""
    c = context(object_name); folder = ROOT/'subdivide_task/data'/object_name/FOLDER
    report = json.loads((folder/'report.json').read_text()); g = report['group']; policy = report['policy']
    for step in g['chain']:
        if step['bbx_ratio'] > step.get('gate', policy['gate'])+1e-12: raise ValueError(step['name']+': gate not met')
        if step['operation'] not in ('juxtaposition', 'rotational_reuse'): raise ValueError('unknown operation')
        if step['operation'] == 'rotational_reuse':
            X = np.asarray(step['slots'][-1]['X']); before = [np.asarray(s['X']) for s in step['parent_slots'][0]]
            if not any(np.allclose(X, Y, atol=1e-9) for Y in before): raise ValueError(step['name']+': reuse outside an existing seat')
    if sorted(i for s in g['slots'] for i in s['members']) != list(range(len(c.base.ids))): raise ValueError('not every task once')
    support = read_mesh(ROOT/g['file']); facts = certify(c, g['slots'], support)
    if 'standing' in facts:
        with np.load(ROOT/g['file']) as z: saved = trimesh.Trimesh(z['standing_vertices'], z['standing_faces'], process=False)
        # The same triangles are the same support; Booleans between two coincident surfaces leave slivers (B+03:
        # 2.9e-9 m^3 between identical meshes), so the volume test is only for surfaces that differ.
        if not same_surface(facts['standing'], saved):
            d = G.volume(G.solid(facts['standing'])-G.solid(saved))+G.volume(G.solid(saved)-G.solid(facts['standing']))
            if d > c.base.tolerance: raise ValueError(f'standing support differs from the saved one: {d} m^3')
    result = dict(status='passed', report_sha256=sha256(folder/'report.json'), membrane_sha256=sha256(ROOT/g['file']),
                  poses=facts['poses'], bbx_m3=facts['bbx_m3'], membrane_cm3=facts['membrane_cm3'],
                  standing_cm3=facts.get('standing_cm3'), joins=len(g['chain']),
                  seats=[len(ks) for _, ks in seats(g)])
    write_json(folder/'validation.json', plain(result)); print('STEP4 COMBINE DISK REPLAY PASS', flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', default='B'); p.add_argument('--workers', type=int, default=22)
    p.add_argument('--samples', type=int, default=POLICY['samples']); p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    validate(a.object) if a.validate_only else run(a.object, a.workers, dict(POLICY, samples=a.samples))
