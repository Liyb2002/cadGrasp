"""Step4: which tasks share a seat, and how each task turns about the needle axis.

A seat holds the object in one place in the support; the support frame is the object frame.
Tasks sharing a seat use one 5 mm membrane: the Step3 stock minus every member's keep-out
(needle, head and KUKA, step4/keepout.py) and continuous exit.  So the shared membrane never
costs more than the cheaper member alone; the merge rule V(AB) <= rho (V(A)+V(B)) holds by
itself for rho >= 1/2.  What decides a merge is whether the membrane still holds the object
under every member's loads, keeps its KUKA clear, lets it leave, and stands on the table.

Poses.  Task i may turn by psi about the nominal direction a through the work point (the arm
rises and falls with the object, so heights are free).  The turn moves gravity, seen from the
object, around a cone of half angle angle(a, down) about the object-frame needle direction a_i.
Two tasks feel gravity along the same object direction g where their cones meet; a seat that
holds both then rests on the same face of the support for both, turned on the table about the
vertical.  So each task's turns are psi = 0 (Step1's pose) and the turns that meet another
task's cone.  A turn is used only if the KUKA holds every certified face there.

The search is bottom-up over member sets: a set is tried only when all its subsets hold (more
members only cut more membrane away), each member is audited at each of its turns on the set's
membrane, and the set takes the turns with the fewest support faces down that also stand on
the table.  The set's poses are its distinct down directions.
"""
import argparse
import itertools
import json
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import trimesh

from subdivide_task.step1.common import ROOT, write_json
from subdivide_task.operations.geometry import moved_mesh
from subdivide_task.operations.state import Layout
from subdivide_task.operations.sweep_cache import sweep_mesh
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step3.exit_geometry import whole_object_sweep
from subdivide_task.step3.membranes import MembraneModel, positive_components
from subdivide_task.step3.tool_membranes import ToolMembraneModel, load_model, stem, task_partition
from . import fast_checks  # noqa: F401  exact exit checks, installed on import
from .keepout import folders, load as load_keepouts, turned_keepout, signature

DOWN = np.array([0., 0., -1.])
SAME_DOWN = 1e-6                  # down directions closer than this (1 - cos) are one support face
FOLDER = 'step4_seats'


def unit(v):
    return np.asarray(v, float)/np.linalg.norm(v)


def turn_angle(R0, g, a):
    """psi (deg) with (Rot_a(psi) R0)^T DOWN = g, for g on the task's gravity cone."""
    w = R0@g; u = w-(w@a)*a; v = DOWN-(DOWN@a)*a
    psi = np.degrees(np.arctan2(a@np.cross(u, v), u@v))
    R = rotation(a, psi)@R0
    if np.linalg.norm(R.T@DOWN-g) > 1e-9: raise ValueError('turn does not reach g')
    return float(psi)


def rotation(a, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(unit(a)*np.radians(deg)).as_matrix()


def meeting_turns(model):
    """{(i, j): [(psi_i, psi_j, g)]}: turns after which tasks i and j feel gravity along the same g."""
    a = unit(model.axis); c = float(a@DOWN); out = {}
    R0 = [np.asarray(r['pose'])[:3, :3] for r in model.records]
    for i, j in itertools.combinations(range(len(R0)), 2):
        ai, aj = R0[i].T@a, R0[j].T@a; m = float(ai@aj); n = np.cross(ai, aj); rows = []
        if m > -1+1e-12 and np.linalg.norm(n) > 1e-12:
            alpha = c/(1+m); beta2 = 1-2*alpha**2*(1+m)
            # Two cones of the same half angle: they cross twice, touch once (beta2 = 0) or miss.
            for beta in ([0.] if abs(beta2) <= 1e-9 else [np.sqrt(beta2), -np.sqrt(beta2)] if beta2 > 0 else []):
                g = alpha*(ai+aj)+beta*unit(n)
                rows.append((turn_angle(R0[i], g, a), turn_angle(R0[j], g, a), g))
        out[(i, j)] = rows
    return out


def turn_options(model):
    """Per task: 0 and every turn meeting another task's cone, deg, sorted and deduplicated."""
    options = [{0.} for _ in model.records]
    def wrapped(x): return round(180.-(180.-x) % 360., 9)+0.     # (-180, 180]; -180 is 180, -0 is 0
    for (i, j), rows in meeting_turns(model).items():
        for pi, pj, _ in rows: options[i].add(wrapped(pi)); options[j].add(wrapped(pj))
    return [sorted(o, key=lambda x: (x != 0, x)) for o in options]


def certify(args):
    """(task index, turn) -> passed record summary; the cached KUKA keep-out at that turn."""
    object_name, i, angle = args
    if angle == 0: return i, angle, dict(status='passed')
    result = turned_keepout(object_name, i, angle)
    return i, angle, (dict(status='failed') if result is None else dict(status='passed', keepout_cm3=result[3]['keepout_cm3']))


def pool(workers):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))


def certify_turns(object_name, workers):
    source, _ = folders(object_name); model, _, _ = load_model(source); options = turn_options(model)
    jobs = [(object_name, i, a) for i, o in enumerate(options) for a in o]
    with pool(workers) as p: results = list(p.map(certify, jobs))
    passed = [[a for (i, a, r) in results if i == k and r['status'] == 'passed'] for k in range(len(options))]
    return options, passed, results


class SeatModel(ToolMembraneModel):
    """Tasks on one seat.  The support frame is the object frame (material = pose^-1).  A pose is the
    certified one moved vertically (the arm rises and falls with it), so the table is world z = 0 in
    every installation; the KUKA is checked where it was certified, at the record's pose."""
    def __init__(self, base, records, keepouts, sweeps):
        super().__init__(base.source, base.outer, records, base.setup, base.physical, base.policy, base.name)
        self.keepouts = list(keepouts); self.sweeps = list(sweeps)
        self.stock = G.solid(base.outer); self.works = [G.solid(k) for k in self.keepouts]
        self.exit = -unit(self.axis); self.length = construction_length(base)

    def tool(self, i):
        return self.keepouts[i]

    def prepare(self, layout):
        rows = []
        for i, r in enumerate(self.records):
            P, M, d = layout.poses[i], layout.material[i], layout.exits[i]
            lift = P@np.linalg.inv(r['pose'])
            if not (np.allclose(lift[:3, :3], np.eye(3), atol=1e-12) and np.allclose(lift[:2, 3], 0, atol=1e-12)):
                raise ValueError(r['id']+': pose is not the certified one moved vertically')
            if not np.allclose(M@P, np.eye(4), atol=1e-9): raise ValueError('The support frame is the object frame')
            if not np.allclose(d, self.exit, atol=1e-12): raise ValueError('The exit is against the needle')
            rows.append(dict(self.world_data(r, P), body=self.source, stock=self.stock, work=self.works[i],
                             sweep=self.sweeps[i], direction=M[:3, :3]@d, length=self.length, material=M, pose=P,
                             index=i, id=r['id'], working_face_ids=np.asarray(r['working_face_ids'])))
        return rows

    def audit(self, candidate, *, arm=True, **options):
        result = MembraneModel.audit(self, candidate, **options); began = time.monotonic()
        for t, row in zip(candidate['tasks'], result['tasks']):
            r = self.records[t['index']]
            hits = self.arm_hits(moved_mesh(candidate['support'], r['pose']), r)
            row['arm'] = dict(configurations=4*len(r['certified_face_ids']), faces_hit=hits)
            if hits and arm: raise ValueError(t['id']+': KUKA links meet the support at faces '+str(hits[:10]))
        result['arm_seconds'] = time.monotonic()-began
        return result


def construction_length(base):
    return float(np.ceil(4*np.linalg.norm(base.source.extents)/.1)*.1)


class Context:
    """Per process: Step3's model, its keep-outs at every certified turn, and the object-frame exits."""
    def __init__(self, object_name):
        source, keep = folders(object_name); self.object_name = object_name; self.folder = keep.parent/FOLDER
        self.base, _, _ = load_model(source); self.zero = load_keepouts(object_name)
        a = unit(self.base.axis); self.sweeps = []; self.records = {}; length = construction_length(self.base)
        for r in self.base.records:
            v = -(np.asarray(r['pose'])[:3, :3].T@a)                # the exit against the needle, object frame
            self.sweeps.append(G.solid(sweep_mesh(self.folder/'sweeps', self.base.source, v, length,
                lambda v=v: G.mesh(whole_object_sweep(self.base.source, v, length)[0]), 'object_frame_v1')))

    @property
    def passed(self):
        """Every task's certified turns (step4/seats.run), read when the seat search needs them."""
        turns = json.loads((self.folder/'turns.json').read_text())
        if turns.get('signature') != signature(self.object_name)[1]:
            raise ValueError('Step4 seat tasks changed; rerun subdivide_task.step4.seats')
        return turns['passed']

    def record(self, i, angle):
        """Task i turned by angle: its record (pose and joints certified there) and keep-out."""
        if (i, angle) not in self.records:
            r = dict(self.base.records[i])
            if angle == 0: self.records[i, angle] = r, self.zero[r['id']][2]
            else:
                pose, joints, keepout, _ = turned_keepout(self.object_name, i, angle)
                r.update(pose=pose, joints=joints, angle_deg=angle); self.records[i, angle] = r, keepout
        return self.records[i, angle]

    def model(self, members, angles):
        rows = [self.record(i, a) for i, a in zip(members, angles)]
        return SeatModel(self.base, [r for r, _ in rows], [k for _, k in rows], [self.sweeps[i] for i in members])

    def down(self, i, angle):
        return np.asarray(self.record(i, angle)[0]['pose'])[:3, :3].T@DOWN

    def name(self, members):
        return '_'.join(stem(self.base.ids[i]) for i in members)


@lru_cache(maxsize=1)
def context(object_name):
    return Context(object_name)


def grounded(model, support):
    """Layout with every pose lowered onto the table: the support's lowest point at world z = 0."""
    poses = []
    for r in model.records:
        P = np.asarray(r['pose']).copy(); P[2, 3] -= float((support.vertices@P[:3, :3].T+P[:3, 3])[:, 2].min()); poses.append(P)
    poses = np.asarray(poses)
    return Layout(poses, np.linalg.inv(poses), np.tile(model.exit, (len(poses), 1)))


def membrane(args):
    """The membrane of a member set: stock minus every member's keep-out at every certified turn and
    its exit, largest piece, exits refined.  A keep-out enters only where it meets the stock's box."""
    object_name, members = args; began = time.monotonic(); c = context(object_name)
    low, high = c.base.outer.bounds; window = G.box(low-.01, high+.01)
    keepouts = [G.solid(c.record(i, a)[1]) ^ window for i in members for a in c.passed[i]]
    value = G.solid(c.base.outer)-G.union([c.sweeps[i] for i in members]+keepouts); parts = positive_components(value)
    if not parts: return dict(members=list(members), status='failed', reason='No membrane material remains')
    keep = max(parts, key=lambda p: float(p.volume()))
    for p in parts:
        if p is not keep: value = value-p
    support = G.mesh(value.simplify(0.)); forbidden = [G.solid(c.base.source), *keepouts]
    for _ in range(3):
        native = G.solid(support)
        if max(G.volume(native ^ f) for f in forbidden) <= c.base.tolerance: break
        support = G.mesh((native-G.union(forbidden)).simplify(0.))
    model = c.model(members, [0.]*len(members))
    candidate = dict(layout=grounded(model, support), support=support, construction={})
    candidate['tasks'] = model.prepare(candidate['layout'])
    try: model.refine_exits(candidate)
    except ValueError as ex: return dict(members=list(members), status='failed', reason=str(ex)[:300])
    # The mesh is audited as manifold leaves it after any Boolean: a Boolean merges short and collinear
    # edges over the whole mesh, so contacts certified on an unsettled mesh could vanish when joined.
    support = settled(candidate['support'])
    if support is None: return dict(members=list(members), status='failed', reason='membrane mesh does not settle under manifold')
    path = c.folder/'membranes'/(c.name(members)+'.npz')
    path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, vertices=support.vertices, faces=support.faces)
    return dict(members=list(members), status='built', volume_cm3=float(support.volume*1e6), file=str(path),
                seconds=time.monotonic()-began)


def settled(mesh, rounds=6):
    """The mesh after no-op Booleans (union and removal of a far block) until one more changes nothing."""
    far = G.box(mesh.bounds[1]+.05, mesh.bounds[1]+.06)
    for _ in range(rounds):
        nxt = G.mesh((G.solid(mesh)+far)-far)
        if same_surface(nxt, mesh): return nxt
        mesh = nxt
    return None


def same_surface(a, b, digits=12):
    """The same set of triangles (vertex order within and between triangles may differ)."""
    def rows(m): return np.unique(np.sort(np.round(m.triangles, digits).reshape(-1, 3, 3).tolist(), axis=1).reshape(-1, 9), axis=0)
    A, B = rows(a), rows(b)
    return A.shape == B.shape and np.array_equal(A, B)


def read_mesh(path):
    with np.load(path) as z: return trimesh.Trimesh(z['vertices'], z['faces'], process=False)


def member_audit(args):
    """Member i at one turn on a set's membrane: loads, exit, KUKA, body and keep-out overlap."""
    object_name, members, path, i, angle = args; began = time.monotonic(); c = context(object_name)
    model = c.model([i], [angle]); support = read_mesh(path)
    candidate = dict(layout=grounded(model, support), support=support, construction={})
    candidate['tasks'] = model.prepare(candidate['layout'])
    try:
        row = model.audit(candidate)['tasks'][0]
        return dict(members=list(members), task=i, angle=angle, passed=True, seconds=time.monotonic()-began,
                    exit_length_m=row['exit_length_m'], contact_area_cm2=row['force']['contact_area_m2']*1e4)
    except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as ex:
        return dict(members=list(members), task=i, angle=angle, passed=False, reason=str(ex)[:300], seconds=time.monotonic()-began)


def downs(c, members, angles):
    """Distinct down directions (object frame) of a turn assignment: the support faces it rests on."""
    out = []
    for i, a in zip(members, angles):
        g = c.down(i, a)
        if not any(1-g@h < SAME_DOWN for h in out): out.append(g)
    return out


def shadow(points, g):
    """Area of the convex shadow of points on a table with down direction g (m^2)."""
    from scipy.spatial import ConvexHull
    from subdivide_task.common.force_cone import tangent_frame
    u, v = tangent_frame(g); return float(ConvexHull(np.c_[points@u, points@v]).volume)


def ground(args):
    """A turn assignment of a set stands on the table: Step5's foot region exists (ground_oracle.foot_regions)."""
    from .ground_oracle import foot_regions, pressure_rows
    object_name, members, path, angles = args; began = time.monotonic(); c = context(object_name)
    model = c.model(members, angles); support = read_mesh(path)
    candidate = dict(layout=grounded(model, support), support=support, construction={})
    candidate['tasks'] = model.prepare(candidate['layout'])
    out = dict(members=list(members), angles=list(angles))
    try: found = next(foot_regions(model, candidate), None)
    except (ValueError, RuntimeError) as ex: return dict(out, passed=False, reason=str(ex)[:300], seconds=time.monotonic()-began)
    if found is None: return dict(out, passed=False, reason='no foot region', seconds=time.monotonic()-began)
    rows = pressure_rows(model, candidate['layout'], candidate['tasks']); D = downs(c, members, angles); prints = []
    for g in D:
        hulls = [r['ground_material'] for r, i, a in zip(rows, members, angles) if 1-c.down(i, a)@g < SAME_DOWN]
        prints.append(1e4*shadow(np.vstack([support.vertices, *hulls]), g))
    return dict(out, passed=True, foot_padding_m=found[0], downs=[g.tolist() for g in D], footprints_cm2=prints,
                seconds=time.monotonic()-began)


def grounded_seat(c, members, angles, membrane):
    """The seat built standing on the table, for turns with one down direction (table frame T, plate bottom
    at z = 0): the membrane resting at plate height, an upright post over the footprint of its lowest part up
    to a needle radius below the object, and a plate under the post and every member's pressure hull, widened
    by the edge margin -- Step3's joined foot, for one seat.  Settled, since any later Boolean would merge
    the same slivers.  Returns (seat in the table frame, T)."""
    from .join import JoinedModel
    from .ground_oracle import pressure_rows
    from subdivide_task.step3.tool_keepout import disc
    import manifold3d as md
    p = c.base.policy; h = p['plate_thickness_m']; t = p['mold_outward_displacement_m']; clear = 1.1*c.base.setup['needle_radius_m']
    R = np.asarray(c.record(members[0], angles[0])[0]['pose'])[:3, :3]
    T = np.eye(4); T[:3, :3] = R; T[2, 3] = h-float((membrane.vertices@R.T)[:, 2].min())
    placed = moved_mesh(membrane, T); low = float(moved_mesh(c.base.source, T).vertices[:, 2].min())
    top = min(h+t, low-clear); V = placed.vertices[placed.vertices[:, 2] <= top]
    if top <= h or len(V) < 3: raise ValueError('no membrane below the object to stand on')
    model = JoinedModel(c, [list(members)], [list(angles)], [T]); q = model.layout()
    rows = pressure_rows(model, q, model.prepare(q))
    ring = disc(np.array([0., 0., 1.]), p['plate_edge_margin_m'], 16)[:, :2]
    F = (np.vstack([V[:, :2], *[r['ground_material'][:, :2] for r in rows]])[:, None]+ring[None]).reshape(-1, 2)
    post = md.Manifold.hull_points(np.vstack([np.c_[V[:, :2], np.zeros(len(V))], np.c_[V[:, :2], np.full(len(V), top)]])/G.SCALE)
    plate = md.Manifold.hull_points(np.vstack([np.c_[F, np.zeros(len(F))], np.c_[F, np.full(len(F), h)]])/G.SCALE)
    seat = settled(G.mesh(G.union([G.solid(placed), post, plate])))
    if seat is None: raise ValueError('standing seat does not settle under manifold')
    return seat, T


def seat_file(c, members, angles):
    return c.folder/'seats'/(c.name(members)+'__'+'_'.join(f'{a:+.3f}' for a in angles)+'.npz')


def stand(args):
    """A turn assignment of a set stands on the table.  One down direction: the seat is built standing
    (grounded_seat) and every member is audited on it in full, with the plate's ground faces enclosing
    every member's pressure hull.  Several down directions: the foot-region oracle only (ground)."""
    object_name, members, path, angles = args; began = time.monotonic(); c = context(object_name)
    D = downs(c, members, angles)
    if len(D) > 1: return dict(ground(args), constructed=False)
    from .join import JoinedModel
    from .ground_oracle import pressure_rows, ground_enclosure
    out = dict(members=list(members), angles=list(angles), constructed=True, downs=[g.tolist() for g in D])
    try:
        seat, T = grounded_seat(c, members, angles, read_mesh(path))
        model = JoinedModel(c, [list(members)], [list(angles)], [T]); q = model.layout()
        candidate = dict(layout=q, support=seat, tasks=model.prepare(q), construction={})
        model.audit(candidate)
        rows = ground_enclosure(seat, pressure_rows(model, q, candidate['tasks']))
        if not all(r['passed'] for r in rows): raise ValueError('a pressure hull leaves the plate')
    except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as ex:
        return dict(out, passed=False, reason=str(ex)[:300], seconds=time.monotonic()-began)
    file = seat_file(c, members, angles); file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file, vertices=seat.vertices, faces=seat.faces, table_from_object=T)
    from scipy.spatial import ConvexHull
    bottom = seat.vertices[np.abs(seat.vertices[:, 2]) <= 1e-9][:, :2]
    return dict(out, passed=True, file=str(file), standing_volume_cm3=float(seat.volume*1e6),
                footprints_cm2=[1e4*float(ConvexHull(bottom).volume)], seconds=time.monotonic()-began)


def assignments(c, members, passing, limit):
    """Turn assignments whose every member passes, fewest support faces first, then fewest turns."""
    rows = [(len(downs(c, members, combo)), sum(a != 0 for a in combo), combo) for combo in itertools.product(*passing)]
    return [list(combo) for *_, combo in sorted(rows, key=lambda r: r[:2])][:limit]


def run(object_name='B', workers=20, ground_tries=4):
    began = time.monotonic(); folder = folders(object_name)[1].parent/FOLDER; folder.mkdir(parents=True, exist_ok=True)
    options, passed, results = certify_turns(object_name, workers)
    key = signature(object_name)[1]
    write_json(folder/'turns.json', dict(signature=key, options=options, passed=passed, results=[list(r) for r in results]))
    print('TURNS', passed, flush=True)
    c = context(object_name); n = len(c.base.ids); sets = {}; level = [(i,) for i in range(n)]
    with pool(workers) as p:
        while level:
            built = {tuple(r['members']): r for r in p.map(membrane, [(object_name, T) for T in level])}
            jobs = [(object_name, T, r['file'], i, a) for T, r in built.items() if r['status'] == 'built'
                    for i in T for a in passed[i]]
            audits = list(p.map(member_audit, jobs)); trials = []
            for T, r in built.items():
                rows = [x for x in audits if tuple(x['members']) == T]
                passing = [[x['angle'] for x in rows if x['task'] == i and x['passed']] for i in T]
                r['audits'] = rows; r['passing_turns'] = passing
                if r['status'] == 'built' and all(passing):
                    trials += [(object_name, T, r['file'], combo) for combo in assignments(c, T, passing, ground_tries)]
            grounds = list(p.map(stand, trials))
            for T, r in built.items():
                r['grounds'] = [g for g in grounds if tuple(g['members']) == T]
                r['feasible'] = any(g['passed'] for g in r['grounds']); sets[T] = r
                print('SET', c.name(T), r['status'], round(r.get('volume_cm3', 0), 1), 'turns', r['passing_turns'],
                      'feasible', r['feasible'], flush=True)
            write_json(folder/'sets.json', dict(signature=key, task_partition=task_partition(c.base.records, c.base.source),
                                                sets=[dict(v, name=c.name(k)) for k, v in sets.items()],
                                                seconds=time.monotonic()-began))
            good = [T for T, r in sets.items() if r['feasible'] and len(T) == len(level[0])]
            level = sorted({tuple(sorted(set(A) | set(B))) for A in good for B in good
                            if len(set(A) | set(B)) == len(A)+1})
            level = [T for T in level if all(tuple(x for x in T if x != k) in sets and
                                             sets[tuple(x for x in T if x != k)]['feasible'] for k in T)]
    return sets


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', default='B'); p.add_argument('--workers', type=int, default=20)
    a = p.parse_args()
    run(a.object, a.workers)
