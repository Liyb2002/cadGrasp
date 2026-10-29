"""Step4 for a seat plan (step3/seat_init.py), fast: combine's joins in one guided pass, with Working-area change.

Step3 gives the start: shells round the object, each opened against the needle (its exit e, the needle reversed: the
opening the object leaves by is the one the tool works through), its tasks turns about the needle (each turn the
support resting on another face, a base) and a first partition of the faces.  Here the tasks join one support one
at a time, shell by shell (Step3's seats) and base by base within a shell (the bases by area): the first task of a
shell starts it -- alone, or by Rotational reuse into a shell already made (its own exit, redirected), or by
Juxtaposition beside one -- and the rest join by Rotational reuse, each kept only when everything still holds.  Before
a shell opens, the windows of the shells already in (Step3: their tasks for later shells' faces, the support turned
so their exit still rises) take what they can of the faces still to be worked, joining their own shell by
Rotational reuse -- turning the support instead of juxtaposing another shell; the later tasks keep what is left.  Each join by combine's operations:

  Rotational reuse  into a seat that holds a task of T's Step3 seat, then into any other: the seat's exit channels
                    first (they cut nothing more), then T's own exit and Exit redirection (rings about it, combine's
                    descent);
  Juxtaposition     a new seat beside a seat of the group (fewest tasks first): Pose rotation and Translation
                    (combine's starts and descent), T's own membrane;
  Working-area change  when loads fail -- T's or a group task's -- the failing faces go to the tasks that reach them
                    (Step1's exact checks over the whole object, computed once per task) and face them most squarely,
                    among the tasks whose loads hold.  A receiver's keep-out grows by those faces' needle, head and KUKA
                    pieces, cut from the support; a donor keeps its cut (conservative: the support only loses).  A task
                    left with no certified face is dropped.  T that joins nowhere gives all its faces away.

So which tasks share a seat, how many seats and every task's area are what the joins make, not Step3's.  A join is
judged by combine's guidance residual and binary constraints (the bbx gate, the floor gap), then by every task's
exact paired loads (32,768 samples and the continuous outer targets on its actual contacts less the skin narrower
than WIDTH), in parallel, a task whose nearby support is unchanged keeping its judgement.  The rest of the audit
runs once at the end (combine.accept: settled, skin pruned, exits refined, every exit, body and keep-out, the KUKA at
every stored joint, the stance); a failure that names tasks (or the stance of several poses: the tasks off the main
one) gives their faces away and the audit runs again.  T that joins nowhere under the gate is tried again under a
no gate after every other task (the support may grow), or gives its faces to the tasks that joined since.

Every join, start and Working-area change passes Step5's feet test: the feet must enclose every
demand's centre of pressure on the table (the weight and the needle's push), so no load tips the support.

Task versions (A3r1, ...) keep every cache keyed by a task index true.  Writes data/<run>/step4_seats/: reach.npz
(the tasks' whole-object reach, kept across runs of the same plan), records.npz/.json (the versions, read back by
SeatContext), keepout/, groups/, report.json ('finals' as combine's, for Step5).

  python -m whole_task_2.step4.seat_combine whole_shape [--workers 16]
"""
import argparse
import collections
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from robots_config.constraints import load_config
from whole_task_2.step1.common import ROOT, load_shape, sha256, write_json
from whole_task_2.step1 import plan_reach as P1
from whole_task_2.common.force_cone import tangent_frame
from whole_task_2.common.wrenches import resolve_policy
from whole_task_2.operations.geometry import moved_mesh, contact_surface
from whole_task_2.operations.state import transform
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.independent_physics import certificate
from whole_task_2.step3.tool_keepout import keepout as tool_keepout
from whole_task_2.step3.tool_membranes import ToolMembraneModel, POLICIES, stem
from whole_task_2.step3.big_init import big_task
from . import combine as C
from .keepout import arm_solid
from .seats import Context, SAME_DOWN

FOLDER = 'step4_seats'
POLICY = dict(C.POLICY,
              offsets=(1., .5, 0.),      # Juxtaposition: at every pose_starts turn its centre of mass on the seat's (the
                                         #     most overlap), half-way, its work-area centre on the seat's (combine's)
              moves=2,                   # Working-area changes per start
              final_audits=3,            # full audits at the end, a Working-area change after each failure
              guidance=16,               # failing samples and outer targets per task kept for the guidance
              floor_gap=False,           # no hard floor gap: with several resting faces the stance is Step5's feet test
              exit_redirection=True,     # Rotational reuse: the shell's exit first, then Exit redirection (rings, descent)
              window_min=.003)           # a window task takes at least this share of the area


def data(run):
    return ROOT/'whole_task_2/data'/run


def is_seat_plan(run):
    return (data(run)/'step3_seats/report.json').exists()


@lru_cache(maxsize=4)
def step3(run):
    folder = data(run)/'step3_seats'; report = json.loads((folder/'report.json').read_text())
    with np.load(folder/'plan.npz') as z: arrays = {k: z[k] for k in z.files}
    return report, arrays


def step3_records(run):
    """Step3's tasks as records: the working area its certified faces and those it absorbed (loads), the keep-out the
    certified faces'; with its Step3 seat and the seat's exit (object frame)."""
    report, z = step3(run); out = []
    for seat in report['seats']:
        for t in seat['tasks']:
            n = t['id']; own = z[n+'_face_ids'].astype(int); extra = z[n+'_absorbed_ids'].astype(int)
            out.append(dict(id=n, seat=seat['id'], exit_object=np.asarray(seat['exit_object'], float),
                            working_face_ids=np.r_[own, extra].tolist(), certified_face_ids=own.tolist(),
                            axes=np.asarray(z[n+'_axes'], float), joints=np.asarray(z[n+'_joints'], float),
                            pose=np.asarray(z[n+'_pose'], float), angle_deg=0.))
    return out


def saved_records(run):
    """Step4's task versions, in the order made, if saved for this Step3 plan."""
    folder = data(run)/FOLDER
    if not ((folder/'records.json').exists() and (folder/'records.npz').exists()): return []
    meta = json.loads((folder/'records.json').read_text())
    if meta['step3_plan_sha256'] != sha256(data(run)/'step3_seats/plan.npz'): return []
    with np.load(folder/'records.npz') as z: z = {k: z[k] for k in z.files}
    return [dict(id=m['id'], seat=m['seat'], exit_object=np.asarray(m['exit_object'], float),
                 working_face_ids=z[m['id']+'_working'].astype(int).tolist(), certified_face_ids=z[m['id']+'_certified'].astype(int).tolist(),
                 axes=z[m['id']+'_axes'], joints=z[m['id']+'_joints'], pose=z[m['id']+'_pose'], angle_deg=0.) for m in meta['records']]


def model_of(run, records):
    report, _ = step3(run); obj = report['object']; physical = json.loads((ROOT/'objects'/obj/'meta.json').read_text()); policy = {}
    for p in POLICIES: policy.update(json.loads(p.read_text()))
    policy = resolve_policy(policy, load_config(ROOT/report['station'])); policy['task_count'] = len(records)
    with np.load(data(obj)/'step3_mold_plate/fitted_outer.npz') as o: outer = trimesh.Trimesh(o['vertices'], o['faces'], process=False)
    return ToolMembraneModel(load_shape(ROOT/'objects'/obj/'mesh.stl'), outer, records, report['setup'], physical, policy, run)


def keepout_file(run, name):
    return data(run)/FOLDER/'keepout'/(stem(name)+'.npz')


def base(name):
    return name.split('r')[0]


class _Sweeps:
    """A record's exit sweep (combine.sweep, exact, in memory)."""
    def __init__(self, c): self.c = c
    def __getitem__(self, i): return C.sweep(self.c.object_name, self.c.base.records[i]['exit_object'])


class SeatContext(Context):
    """seats.Context over a seat plan: Step3's tasks, then Step4's versions; keep-outs from step4_seats/keepout/."""
    def __init__(self, run):
        self.object_name = run; self.folder = data(run)/FOLDER
        self.base = model_of(run, step3_records(run)+saved_records(run)); self.records = {}; self.sweeps = _Sweeps(self)

    @property
    def passed(self):
        raise ValueError('seat tasks are not turned')

    def record(self, i, angle):
        if angle != 0: raise ValueError('seat tasks are not turned')
        if i not in self.records:
            r = self.base.records[i]
            with np.load(keepout_file(self.object_name, r['id'])) as z:
                self.records[i] = r, trimesh.Trimesh(z['keepout_vertices'], z['keepout_faces'], process=False)
        return self.records[i]

    def add(self, r):
        self.base.records.append(r); self.base.ids.append(r['id']); return len(self.base.records)-1


# ---------------------------------------------------------------------------------------------
# Workers: keep-outs and exact loads.

_W = {}


def _init_worker(run, cpus):
    os.sched_setaffinity(0, {cpus.get()}); _W.update(run=run, model=model_of(run, []))


def workers_pool(run, n):
    context_ = multiprocessing.get_context('spawn'); cpus = sorted(os.sched_getaffinity(0)); queue = context_.Queue()
    for k in range(n): queue.put(cpus[k % len(cpus)])
    return ProcessPoolExecutor(n, mp_context=context_, initializer=_init_worker, initargs=(run, queue))


def pieces(r):
    """Needle, head and KUKA keep-out of record r's certified faces (Step3's tool keep-out, step4/keepout.arm_solid)."""
    m = _W['model']; tool = tool_keepout(m.source, r['certified_face_ids'], r['axes'], m.setup, m.reach, m.policy['tool_polygon_sides'])
    return G.union([tool, arm_solid(m, r)[0]])


def save_keepout(name, value):
    mesh = G.mesh(value); np.savez_compressed(keepout_file(_W['run'], name), keepout_vertices=mesh.vertices, keepout_faces=mesh.faces)


def build_keepout(r):
    """Record r's keep-out, saved."""
    began = time.monotonic(); value = pieces(r); save_keepout(r['id'], value)
    return dict(id=r['id'], keepout_cm3=G.volume(value)*1e6, seconds=time.monotonic()-began)


def grow_keepout(args):
    """Record r's keep-out: the old one (of record `old`) and the pieces of the faces `new` (with their axes and
    joints), saved; the pieces (object frame), to cut from the support."""
    r, old, new = args
    with np.load(keepout_file(_W['run'], old)) as z: before = G.solid(trimesh.Trimesh(z['keepout_vertices'], z['keepout_faces'], process=False))
    more = pieces(dict(r, certified_face_ids=new['faces'], axes=new['axes'], joints=new['joints']))
    save_keepout(r['id'], G.union([before, more])); m = G.mesh(more)
    return m.vertices, m.faces


def picks(ids, count):
    return ids[np.unique(np.linspace(0, len(ids)-1, count).astype(int))].tolist() if len(ids) else []


def loads(args):
    """Record r's paired loads on the support surface near its body (world frame at its pose): whether they hold,
    the failing faces (a sample's face; an outer target's work-area vertex, its faces; its own weight, all) and some
    failing samples and outer targets for the guidance."""
    r, vertices, faces, count = args; m = _W['model']; d = m.world_data(r, np.asarray(r['pose'], float)); W = np.asarray(d['domain'].work_ids)
    try:
        proof, cd = certificate(d['body_world'], trimesh.Trimesh(vertices, faces, process=False), d['domain'], d['endpoints'], d['samples'])
    except RuntimeError: return dict(passed=False, faces=W.tolist(), samples=[], targets=[])
    if proof['passed']: return dict(passed=True, faces=[], samples=[], targets=[])
    bad_s, bad_e = np.flatnonzero(~cd['sample_passed']), np.flatnonzero(~cd['endpoint_passed'])
    bad = W[np.asarray(d['samples']['work_face_index'])[bad_s]]; v = np.asarray(d['endpoints']['source_vertex_ids'])[bad_e]
    if (v < 0).any(): bad = W
    elif len(v): bad = np.r_[bad, W[np.isin(m.source.faces[W], v).any(1)]]
    return dict(passed=False, faces=np.unique(bad).tolist(), samples=picks(bad_s, count), targets=picks(bad_e, count))


# ---------------------------------------------------------------------------------------------

def squareness(mesh, faces, r, a):
    """The angle between each face's normal and record r's needle (Step3's rule)."""
    u = r['pose'][:3, :3].T@(a/np.linalg.norm(a)); return np.arccos(np.clip(-mesh.face_normals[faces]@u, -1., 1.))


class Combine:
    def __init__(self, run, workers, policy=POLICY):
        self.began = time.monotonic(); self.run = run; self.policy = policy; folder = data(run)/FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        for p in folder.iterdir():                                 # a rerun overwrites the last one (the reach stays)
            if p.name != 'reach.npz': shutil.rmtree(p) if p.is_dir() else p.unlink()
        for sub in ('keepout', 'groups'): (folder/sub).mkdir()
        self.c = C.context(run); self.first = len(self.c.base.records)
        self.mesh = self.c.base.source; self.area = self.mesh.area_faces
        self.workers = workers_pool(run, workers); self.versions = {}; self.judged = {}; self.files = 0; self.history = []; self.why = collections.Counter()
        self.base_group = None; self.F = {}; self.parents = {}; self.replaced = []; self.tried = set(); self.windows_in = []
        report, _ = step3(run); self.s = report['setup']; self.a = np.asarray(self.s['working_direction_world'], float)
        _, self.shape, _, _, self.target = big_task(run); self.total = float(self.area[self.target].sum())
        rows = list(self.workers.map(build_keepout, self.c.base.records))
        self.log('KEEPOUTS', len(rows), 'tasks, %.0f s each at most' % max(r['seconds'] for r in rows))
        self.reach = self.whole_reach(workers)

    def log(self, *args):
        print(*args, round(time.monotonic()-self.began), 's', flush=True)

    def whole_reach(self, workers):
        """Every Step3 task's reach over the whole big task: Step1's exact checks (needle, head, KUKA) at its
        placement.  {base id: (faces, axes, joints)}, saved for this plan."""
        path = data(self.run)/FOLDER/'reach.npz'; key = sha256(data(self.run)/'step3_seats/plan.npz'); recs = self.c.base.records[:self.first]
        if path.exists():
            with np.load(path) as z:
                if str(z['plan_sha256']) == key:
                    return {r['id']: (z[r['id']+'_faces'], z[r['id']+'_axes'], z[r['id']+'_joints']) for r in recs}
        began = time.monotonic(); rp = json.loads((ROOT/'whole_task_2/step1/reach_policy.json').read_text())
        axes, reach, _ = P1.screen(self.mesh, self.shape, self.s, rp, data(step3(self.run)[0]['object'])/'step1_reach')
        eps = rp['needle_start_fraction_of_length']*self.s['needle_length_m']; faces = np.flatnonzero(self.target)
        placements = [dict(rotation_world_object=r['pose'][:3, :3].tolist(), translation_world_m=r['pose'][:3, 3].tolist()) for r in recs]
        directions = np.array([r['pose'][:3, :3].T@self.a for r in recs])
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=P1._init_mesh,
                                 initargs=(str(self.shape), self.s)) as pool:
            tilt, witness, _, joints, _, _ = P1.certify_directions(pool, self.mesh, faces, axes, reach, directions,
                                                                   self.s['max_tilt_deg'], self.s, eps, placements)
        out = {}; arrays = dict(plan_sha256=key)
        for j, r in enumerate(recs):
            f = np.flatnonzero(np.isfinite(tilt[:, j])); out[r['id']] = (f, witness[f, j], joints[f, j])
            arrays.update({r['id']+'_faces': f, r['id']+'_axes': witness[f, j], r['id']+'_joints': joints[f, j]})
        np.savez_compressed(path, **arrays)
        self.log('REACH', len(recs), 'tasks over the whole object: %.2f-%.2f of it each, %.0f s' % (
                 min(self.area[v[0]].sum() for v in out.values())/self.total, max(self.area[v[0]].sum() for v in out.values())/self.total,
                 time.monotonic()-began))
        return out

    # -- records ---------------------------------------------------------------------------------

    def version(self, i, working, certified, axes, joints):
        r = self.c.base.records[i]; b = base(r['id']); self.versions[b] = self.versions.get(b, 0)+1
        return self.c.add(dict(r, id=f'{b}r{self.versions[b]}', working_face_ids=[int(f) for f in working],
                               certified_face_ids=[int(f) for f in certified], axes=np.asarray(axes, float).reshape(-1, 3),
                               joints=np.asarray(joints, float).reshape(-1, 4, 7)))

    def save(self, keep, slots, chain, name):
        """A group: its support saved, its slots and chain."""
        self.files += 1; file = data(self.run)/FOLDER/'groups'/f'{self.files:03d}_{name}.npz'; mesh = G.mesh(keep)
        np.savez_compressed(file, vertices=mesh.vertices, faces=mesh.faces)
        return dict(file=str(file.relative_to(ROOT)), slots=slots, chain=chain, name=name, volume_cm3=float(mesh.volume*1e6),
                    bbx_m3=C.obb(mesh.vertices))

    # -- the judge -------------------------------------------------------------------------------

    def exact(self, slots, keep):
        """Every task's exact paired loads on keep (support frame) less its skin narrower than WIDTH, in parallel; a
        task whose nearby surface is one already judged keeps that judgement.  {task index: result} of the failing."""
        mesh = G.mesh(keep); thin = C.thin_cells(self.c, slots, mesh); jobs = []; out = {}
        for k, s in enumerate(slots):
            i = s['members'][0]; r = self.c.base.records[i]; P = np.asarray(r['pose'], float); d = self.c.base.world_data(r, P)
            near = contact_surface(d['body_world'], moved_mesh(mesh, P@np.linalg.inv(np.asarray(s['X'], float))))
            if thin[k] is not None and thin[k].any():
                cell = C.skin_of(self.c)[0].tree.query(transform(near.triangles_center, np.linalg.inv(P)))[1]
                near = trimesh.Trimesh(near.vertices, near.faces[~thin[k][cell]], process=False)
            key = (r['id'], C.signature(near))
            if key in self.judged: out[i] = self.judged[key]
            else: jobs.append((i, key, (r, near.vertices, near.faces, self.policy['guidance'])))
        for (i, key, _), res in zip(jobs, self.workers.map(loads, [j[2] for j in jobs])):
            self.judged[key] = out[i] = res
        return {i: res for i, res in out.items() if not res['passed']}

    def failing_area(self, failing):
        faces = np.unique(np.concatenate([np.asarray(v['faces'], int) for v in failing.values()])) if failing else np.zeros(0, int)
        return float(self.area[faces].sum()/self.total)

    # -- Working-area change ----------------------------------------------------------------------

    def give(self, slots, keep, failing, keep_tasks=()):
        """Working-area change (module docstring): failing {task index: faces} to the tasks of slots that reach them and
        face them most squarely, among those not failing.  (slots, keep, faces nobody took) or None if none moved."""
        donors = {}
        for i, f in failing.items():
            for x in f: donors.setdefault(int(x), i)
        faces = np.asarray(sorted(donors), int)
        takers = [k for k, s in enumerate(slots) if s['members'][0] not in failing]
        if not len(faces) or not takers: return None
        cost = np.full((len(faces), len(takers)), np.inf); where = {}
        for n, k in enumerate(takers):
            r = self.c.base.records[slots[k]['members'][0]]; f, A, Q = self.reach[base(r['id'])]
            hit = np.isin(faces, f)
            if hit.any():
                cost[hit, n] = squareness(self.mesh, faces[hit], r, self.a); where[n] = dict(zip(f.tolist(), range(len(f))))
        best = np.argmin(cost, 1); ok = np.isfinite(cost[np.arange(len(faces)), best])
        if not ok.any(): return None
        slots = [dict(s, members=list(s['members'])) for s in slots]; growth = []; jobs = []
        for n in set(best[ok].tolist()):                          # receivers: more faces, the keep-out grown
            k = takers[n]; i = slots[k]['members'][0]; r = self.c.base.records[i]; f, A, Q = self.reach[base(r['id'])]
            new = faces[ok & (best == n)]; fresh = new[~np.isin(new, r['certified_face_ids'])]; rows = [where[n][x] for x in fresh]
            j = self.version(i, np.union1d(r['working_face_ids'], new), np.r_[r['certified_face_ids'], fresh],
                             np.r_[np.asarray(r['axes']).reshape(-1, 3), A[rows]], np.r_[np.asarray(r['joints']).reshape(-1, 4, 7), Q[rows]])
            slots[k]['members'][0] = j
            if len(fresh): jobs.append((k, (self.c.base.records[j], r['id'], dict(faces=fresh.tolist(), axes=A[rows], joints=Q[rows]))))
            else: shutil.copy(keepout_file(self.run, r['id']), keepout_file(self.run, self.c.base.records[j]['id']))
        for (k, _), (V, F) in zip(jobs, self.workers.map(grow_keepout, [j[1] for j in jobs])):
            growth.append(C.move(np.asarray(slots[k]['X'], float), G.solid(trimesh.Trimesh(V, F, process=False))))
        drop = []
        for i in failing:                                         # donors: fewer faces, the cut kept
            k = next(k for k, s in enumerate(slots) if s['members'][0] == i); r = self.c.base.records[i]
            out = faces[ok & np.isin(faces, failing[i])]; Cf = np.asarray(r['certified_face_ids'], int); stay = ~np.isin(Cf, out)
            if not len(out): continue
            if not stay.any(): drop.append(k); continue
            j = self.version(i, np.setdiff1d(r['working_face_ids'], out), Cf[stay], np.asarray(r['axes'])[stay], np.asarray(r['joints'])[stay])
            shutil.copy(keepout_file(self.run, r['id']), keepout_file(self.run, self.c.base.records[j]['id'])); slots[k]['members'][0] = j
        for k in drop:                                            # a dropped task's uncertified faces to the nearest task
            r = self.c.base.records[slots[k]['members'][0]]; rest = np.setdiff1d(r['working_face_ids'], faces[ok])
            rest = rest[~np.isin(rest, r['certified_face_ids'])]
            if len(rest): slots = self.absorb(slots, [x for x in range(len(slots)) if x not in drop], rest)
        slots = [s for k, s in enumerate(slots) if k not in drop]
        if growth:
            box = C.bounds(keep); keep, _ = C.largest(keep-G.union([C.clipped(g, box) for g in growth]))
            if keep is None: return None
        self.log('    AREA %d faces moved (%.4f of the object), %d tasks dropped, %d reached by nobody' % (
                 int(ok.sum()), float(self.area[faces[ok]].sum()/self.total), len(drop), int((~ok).sum())))
        return slots, keep, faces[~ok]

    def absorb(self, slots, among, faces):
        """faces (uncertified) to the nearest task of slots `among` by face centre."""
        centres = self.mesh.triangles_center; owners = [(k, f) for k in among for f in self.c.base.records[slots[k]['members'][0]]['certified_face_ids']]
        _, near = cKDTree(centres[[f for _, f in owners]]).query(centres[faces]); slots = [dict(s, members=list(s['members'])) for s in slots]
        for k in {owners[n][0] for n in near}:
            i = slots[k]['members'][0]; r = self.c.base.records[i]; add = faces[[owners[n][0] == k for n in near]]
            j = self.version(i, np.union1d(r['working_face_ids'], add), r['certified_face_ids'], r['axes'], r['joints'])
            shutil.copy(keepout_file(self.run, r['id']), keepout_file(self.run, self.c.base.records[j]['id'])); slots[k]['members'][0] = j
        return slots

    # -- one join --------------------------------------------------------------------------------

    def stance(self, slots, keep):
        """Step5's feet test (step5/combined.grounds, legs aside), not the stricter foot-region oracle: in every
        installation a ring of legal feet -- tips on Step5's candidate rings round the task's pressure hull, in the legal
        space (above every installation's floor, out of every body, keep-out and exit sweep) -- encloses the hull, or the
        support's own ground faces do.  The hull itself need not be free.  None, or why not."""
        import manifold3d as md
        from shapely.geometry import Polygon
        from shapely.geometry.polygon import orient
        from whole_task_2.step5.combined import FOOT, section, resample, enclosing
        from .skin import legal
        from .ground_oracle import pressure_rows, ground_enclosure
        mesh = G.mesh(keep); model = C.ExitModel(self.c, slots); q = model.layout(mesh); rows = pressure_rows(model, q)
        points = np.vstack([mesh.vertices, *[r['ground_material'] for r in rows]]); pad = max(FOOT['outward'])+FOOT['padding']
        space = legal(points, pad, q.material, self.forbidden(slots))
        disc = md.CrossSection.circle(FOOT['tip']/G.SCALE, 32)
        for r in rows:
            hull = np.asarray(r['hull']); free = section(r, space)
            sites = np.vstack([resample(np.asarray(g.exterior.coords)[:-1], FOOT['spacing']) for d in FOOT['outward']
                               for ring in [orient(Polygon(hull).buffer(d, join_style=2), sign=1.)] for g in getattr(ring, 'geoms', [ring]) if not g.is_empty])
            sites = sites[np.array([(disc.translate(tuple(x/G.SCALE))-free).area() <= 1e-6*disc.area() for x in sites], bool)]
            if len(sites) >= 3 and enclosing(sites, np.ones(len(sites)), hull) is not None: continue
            if ground_enclosure(mesh, [r])[0]['passed']: continue
            return 'no feet for '+r['id']
        return None

    def forbidden(self, slots):
        """Every slot's body, keep-out and exit sweep (combine.forbidden), on the joining group's: each group's is its
        parent's and its new slots' (the latest few kept)."""
        return self.grown(self.base_group, slots) if self.base_group is not None else C.forbidden(self.run, slots)

    def grown(self, group, slots):
        key = lambda s: (s['members'][0], tuple(np.round(np.ravel(s['X']), 9)))
        if group['file'] not in self.F:
            parent = self.parents.get(group['file'])
            self.F[group['file']] = self.grown(parent, group['slots']) if parent is not None else C.forbidden(self.run, group['slots'])
            while len(self.F) > 3: self.F.pop(next(iter(self.F)))
        known = {key(s) for s in group['slots']}; new = [s for s in slots if key(s) not in known]
        return G.union([self.F[group['file']], C.forbidden(self.run, new)]) if new else self.F[group['file']]

    def settle(self, x):
        """A construction whose guidance holds: its exact loads, Working-area change while they fail (at most moves);
        when they hold, Step5's feet test (every installation: the needle's push may tip even one resting face).
        (slots, keep) when every task holds and the support stands, else None."""
        slots, keep = x['slots'], x['keep']
        for move in range(self.policy['moves']+1):
            failing = self.exact(slots, keep)
            if not failing:
                why = self.stance(slots, keep)
                if why is None: return slots, keep
                self.why['stance: '+why] += 1; return None
            self.log('    LOADS fail on %d tasks, %.4f of the object' % (len(failing), self.failing_area(failing)))
            if move == self.policy['moves']: self.why['loads'] += 1; return None
            moved = self.give(slots, keep, {i: v['faces'] for i, v in failing.items()})
            if moved is None: self.why['loads, no task takes the faces'] += 1; return None
            slots, keep, _ = moved
            if C.obb(G.mesh(keep).vertices) > x.get('limit', np.inf): return None
        return None

    def attempt(self, build, starts, blocks, limit, bases, descent=True):
        """combine.restore, judged by settle(): each start whose guidance holds, at once; when none does, the descent
        from the best feasible start (policy['rounds'] rounds).  When the exact loads fail, their failing samples and
        outer targets join the guidance and the descent goes on (combine's feedback, at most policy['feedback']
        times), then settle() (its Working-area change); a binary failure (the feet, the gate) goes on to the next
        start.  build(params, extra).  (slots, keep, params) or None; why not in self.why."""
        extra = {}; evaluate = C.evaluator(lambda p: build(p, extra)); self.why = evaluate.why; built = []; audits = 0

        def guided(x):
            nonlocal evaluate
            for _ in range(self.policy['feedback'] if descent else 0):
                failing = self.exact(x['slots'], x['keep'])
                if not failing: break
                for i, v in failing.items():
                    old = extra.get(i, ((), ())); extra[i] = (sorted(set(old[0]) | set(v['samples'])), sorted(set(old[1]) | set(v['targets'])))
                evaluate = C.evaluator(lambda p: build(p, extra)); y = evaluate(x['params'])
                if y is None: break
                y, _ = C.descend(evaluate, y, blocks, self.policy)
                if not y['covered']: break
                x = y
            return self.settle(dict(x, limit=limit, bases=bases)), x

        for start in starts:
            x = evaluate(start)
            if x is None: continue
            built.append(x)
            if x['covered']:
                done, x = guided(x); audits += 1
                if done: return (*done, x['params'])
                if audits >= self.policy['audits']: return None
        if not built or audits: return None
        if not descent or not blocks:                   # nothing to descend: the exact loads judge the best start
            done, x = guided(min(built, key=lambda x: x['loss']))
            return (*done, x['params']) if done else None
        x, _ = C.descend(evaluate, min(built, key=lambda x: x['loss']), blocks, self.policy)
        if not x['covered']: self.why['guidance unresolved'] += 1; return None
        done, x = guided(x)
        return (*done, x['params']) if done else None

    def alone(self, t):
        """Task t's own membrane: the outer layer minus its body, keep-out and exit sweep, largest piece."""
        outer = G.solid(self.c.base.outer); e = self.c.base.records[t]['exit_object']
        keep, _ = C.largest(outer-C.forbidden(self.run, [C.slot(t, np.eye(4), e)], C.bounds(outer)))
        return keep

    def reuse(self, group, t, k, limit, bases, descent=True):
        """Rotational reuse of task t into seat k: the seat's exit channel, then rings about t's exit."""
        X, ks = C.seats(group)[k]; r = self.c.base.records[t]; e = C.unit(r['exit_object']); rise = np.sin(np.radians(self.policy['exit_rise_deg']))
        Q = r['pose'][:3, :3]; u, v = tangent_frame(e); rings, azimuths = self.policy['exit_starts']
        starts = [C.unit(group['slots'][j]['exit']) for j in ks]+[e]+[C.unit(np.cos(a)*e+np.sin(a)*(np.cos(f)*u+np.sin(f)*v))
                  for a in np.radians(rings) for f in 2*np.pi*np.arange(azimuths)/azimuths]
        starts = [x for x in starts if (Q@x)[2] >= rise-1e-12]
        if not self.policy['exit_redirection']: starts = starts[:len(ks)]       # the shell's shared exit only

        def build(e, extra):
            e = C.unit(e)
            if (Q@e)[2] < rise-1e-12: return None
            keep, _ = C.reuse(self.run, group, t, X, e)
            if keep is None: return None
            return C.scored(self.run, group['slots']+[C.slot(t, X, e)], keep, limit, extra, e, self.policy)

        def exit_redirection(e):
            axes = tangent_frame(e)
            return lambda w: C.unit(C.Rotation.from_rotvec(np.cross(e, w[0]*axes[0]+w[1]*axes[1])).apply(e))
        blocks = [('exit_redirection', exit_redirection, 2, *np.radians(self.policy['exit_deg']))] if self.policy['exit_redirection'] else []
        return self.attempt(build, starts, blocks, limit, bases, descent and bool(blocks))

    def juxtapose(self, group, t, k, mine, limit, bases, descent=True):
        """Juxtaposition of task t beside seat k: its own membrane at combine's seat X, combine's starts (offsets at
        every pose_starts turn), Pose rotation and Translation restoring the loads (attempt)."""
        D = float(np.linalg.norm(self.mesh.extents)); e = self.c.base.records[t]['exit_object']

        def build(p, extra):
            angle, d = float(p[0]), np.asarray(p[1], float); X = C.juxtaposed(self.run, group, k, t, angle, d)
            s = C.group_solids(self.run, group)['s']; placed = C.move(X, mine)
            keep, _ = C.largest((s-C.forbidden(self.run, [C.slot(t, X, e)], C.bounds(s)))+(placed-C.nearby(self.run, group, placed)))
            if keep is None: return None
            return C.scored(self.run, group['slots']+[C.slot(t, X, e)], keep, limit, extra, (angle, d), self.policy)
        blocks = [('pose_rotation', lambda p: (lambda w: (p[0]+np.degrees(w[0]), p[1])), 1, *np.radians(self.policy['pose_deg'])),
                  ('translation', lambda p: (lambda w: (p[0], np.asarray(p[1])+D*np.asarray(w))), 3, *self.policy['translation'])]
        X, _ = C.seats(group)[k]; com = self.mesh.center_mass; starts = []
        for a in self.policy['pose_starts']:
            to_mass = transform(com, X)-transform(com, C.juxtaposed(self.run, group, k, t, a, np.zeros(3)))
            starts += [(float(a), f*to_mass) for f in self.policy['offsets']]
        return self.attempt(build, starts, blocks, limit, bases, descent)

    def join(self, group, t, gate, operations=('rotational_reuse', 'juxtaposition'), descent=True):
        """Task t joins group by operations (module docstring); the new group, or None."""
        r = self.c.base.records[t]; mine = self.alone(t)
        if mine is None: return None
        limit = gate*(group['bbx_m3']+C.obb(G.mesh(mine).vertices)); seats = C.seats(group)
        homes = [k for k, (_, ks) in enumerate(seats) if any(self.c.base.records[group['slots'][j]['members'][0]]['seat'] == r['seat'] for j in ks)]
        # Rotational reuse into its own shell first, then into every other (its exit its own, redirected), then beside one.
        tries = ([('rotational_reuse', k) for k in homes]+[('rotational_reuse', k) for k in range(len(seats)) if k not in homes]+
                 [('juxtaposition', k) for k in sorted(range(len(seats)), key=lambda k: len(seats[k][1]))[:self.policy['seats']]])
        tries = [(o, k) for o, k in tries if o in operations]
        for operation, k in tries:
            began = time.monotonic()
            bases = len(C.downs(self.c, group)); self.why = collections.Counter(); self.base_group = group
            got = self.reuse(group, t, k, limit, bases, descent) if operation == 'rotational_reuse' else self.juxtapose(group, t, k, mine, limit, bases, descent)
            self.log('  %s %s seat %d gate %.1f: %s' % (r['id'], operation, k, gate, 'PASS' if got else 'fail'), '(%.0f s)' % (time.monotonic()-began),
                     '' if got else '; '.join('%s x%d' % (w[:60], n) for w, n in self.why.most_common(2)))
            if got:
                slots, keep, params = got; name = '%s+%s' % (group['name'], base(r['id']))
                new = self.save(keep, slots, [], name); bbx = new['bbx_m3']; self.parents[new['file']] = group
                new['chain'] = group['chain']+[dict(name=name, parents=[group['name'], base(r['id'])], operation=operation, seat=k, gate=float(gate) if np.isfinite(gate) else None,
                                                    bbx_ratio=bbx/(group['bbx_m3']+C.obb(G.mesh(mine).vertices)), bbx_m3=bbx, volume_cm3=new['volume_cm3'])]
                return new
        return None

    def given_away(self, group, t):
        """Task t that joins nowhere gives all its faces to the group's tasks: the new group, or None."""
        r = self.c.base.records[t]; X = np.eye(4); slots = group['slots']+[C.slot(t, X, r['exit_object'])]
        keep = C.group_solids(self.run, group)['s']
        moved = self.give(slots, keep, {t: r['working_face_ids']})
        if moved is None or len(moved[2]): return None
        slots, keep, _ = moved; slots = [s for s in slots if s['members'][0] != t]
        self.base_group = group
        if any(s['members'][0] == t for s in slots) or self.exact(slots, keep) or self.stance(slots, keep): return None
        return self.save(keep, slots, group['chain']+[dict(name=group['name']+'-'+base(r['id']), parents=[group['name'], base(r['id'])],
                                                          operation='working_area_change', seat=-1, gate=0., bbx_ratio=0.)], group['name'])

    # -- the search ------------------------------------------------------------------------------

    def search(self):
        # Shell by shell, base by base (module docstring); a base is added when one of its tasks joins, judged (Step5's
        # feet test, every installation) against every base already in.
        recs = self.c.base.records; report, _ = step3(self.run); area = lambda i: self.area[recs[i]['working_face_ids']].sum()
        group = None; waiting = []; n = 0
        seats = report['seats']; pose = {t['id']: t['pose'] for seat in seats for t in seat['tasks']}
        for k, seat in enumerate(seats):
            ids = [i for i in range(self.first) if recs[i]['seat'] == seat['id']]
            if group is not None:                                 # the open shells' windows first, then what is left
                group = self.absorb(group, [i for later in seats[k:] for i in range(self.first) if recs[i]['seat'] == later['id']])
                ids = [j for j in (self.trimmed(i, group) for i in ids) if j is not None]
            bases = sorted([sorted([i for i in ids if pose[base(recs[i]['id'])] == p], key=lambda i: -area(i))
                            for p in {pose[base(recs[i]['id'])] for i in ids}], key=lambda ks: -sum(area(i) for i in ks))
            for ks in bases:
                n += 1; before = len(C.downs(self.c, group)) if group else 0
                self.log('BASE', n, 'seat', seat['id'], '%d tasks, %.3f of the object' % (len(ks), sum(area(i) for i in ks)/self.total))
                for t in ks:
                    if group is None:
                        keep = self.alone(t); slots = [C.slot(t, np.eye(4), recs[t]['exit_object'])]; self.base_group = None
                        if keep is None or self.exact(slots, keep) or self.stance(slots, keep):
                            self.log('START', recs[t]['id'], 'does not hold or stand alone'); waiting.append(t); continue
                        group = self.save(keep, slots, [], base(recs[t]['id'])); self.log('START', recs[t]['id'], '%.1f cm3' % group['volume_cm3']); continue
                    new = self.join(group, t, self.policy['gate']) or self.given_away(group, t)
                    if new is None: waiting.append(t); self.log('WAIT', recs[t]['id'])
                    else: group = new; self.log('GROUP', group['name'], '%.1f cm3, bbx %.0f cm3, %d seats' % (group['volume_cm3'], group['bbx_m3']*1e6, len(C.seats(group))))
                if group: self.log('BASE', n, 'added' if len(C.downs(self.c, group)) > before else 'not added', '(%d resting faces)' % len(C.downs(self.c, group)))
        left = []
        for t in waiting:                                         # once more after every other task, no gate: its shell, or beside one,
            new = self.join(group, t, np.inf) or self.given_away(group, t)
            if new is None: left.append(t); self.log('LEFT', recs[t]['id'])   # or its faces away
            else: group = new
        return self.audited(group), left

    def windows(self):
        """Step3's window tasks: their shell, exit, pose and certified faces (with axes and joints)."""
        report, z = step3(self.run)
        return [dict(id=x['id'], seat=x['seat'], exit_object=np.asarray(x['exit_object'], float), up=np.asarray(x['up_object'], float),
                     pose=np.asarray(z[x['id']+'_pose'], float),
                     reach=(z[x['id']+'_reach_ids'].astype(int), np.asarray(z[x['id']+'_reach_axes'], float), np.asarray(z[x['id']+'_reach_joints'], float)))
                for x in report.get('windows', [])]

    def worked(self, group):
        """The target faces the group's tasks work."""
        out = np.zeros(len(self.area), bool)
        for s_ in group['slots']: out[self.c.base.records[s_['members'][0]]['working_face_ids']] = True
        return out & self.target

    def absorb(self, group, ids):
        """Before a shell opens: the windows of the open shells, base by base (a window base: its up; Step3's order, the
        least turned first), each a record working (and certified on) the faces of tasks ids it reaches that nothing works yet, its
        keep-out built, joining its shell by Rotational reuse; each tried once.  The group."""
        recs = self.c.base.records; aim = np.zeros(len(self.area), bool)
        for i in ids: aim[recs[i]['working_face_ids']] = True
        open_ = {recs[s_['members'][0]]['seat'] for s_ in group['slots']}; bases = {}
        for w in self.windows():
            if w['seat'] in open_ and w['id'] not in self.tried: bases.setdefault((w['seat'], tuple(np.round(w['up'], 6))), []).append(w)
        gain = lambda w, left: float(self.area[w['reach'][0][left[w['reach'][0]]]].sum())
        for key in list(bases):                                    # Step3's order: the least turned first
            left = aim & ~self.worked(group); ws = bases.pop(key)
            if sum(gain(w, left) for w in ws) < self.policy['window_min']*self.total: continue
            before = len(C.downs(self.c, group))
            self.log('WINDOWS of shell', key[0], '%d tasks; %.3f of the object to take' % (len(ws), self.area[left].sum()/self.total))
            for w in sorted(ws, key=lambda w: -gain(w, left)):
                left = aim & ~self.worked(group); f, A, Q = w['reach']; take = f[left[f]]
                if self.area[take].sum() < self.policy['window_min']*self.total: continue
                self.tried.add(w['id']); where = dict(zip(f.tolist(), range(len(f)))); rows = [where[x] for x in take]
                t = self.c.add(dict(id=w['id'], seat=w['seat'], exit_object=w['exit_object'], pose=w['pose'], angle_deg=0.,
                                    working_face_ids=take.tolist(), certified_face_ids=take.tolist(), axes=A[rows], joints=Q[rows]))
                list(self.workers.map(build_keepout, [recs[t]])); self.reach[w['id']] = w['reach']
                new = self.join(group, t, np.inf, ('rotational_reuse',))
                if new is None: self.log('WINDOW', w['id'], 'not joined')
                else: group = new; self.windows_in.append(w['id']); self.log('WINDOW', w['id'], 'takes %.3f of the object' % (self.area[take].sum()/self.total))
            self.log('WINDOWS of shell', key[0], 'a new resting face' if len(C.downs(self.c, group)) > before else 'no new resting face',
                     '(%d resting faces)' % len(C.downs(self.c, group)))
        return group

    def trimmed(self, i, group):
        """Task i less the faces the group already works (windows took them): i, a version of it (its cut kept), or
        None when no certified face of it is left."""
        r = self.c.base.records[i]; worked = self.worked(group); W = np.asarray(r['working_face_ids'], int); stay = ~worked[W]
        if stay.all(): return i
        Cf = np.asarray(r['certified_face_ids'], int); keep = ~worked[Cf]
        if not keep.any(): self.replaced.append(r['id']); self.log('REPLACED', r['id'], 'by windows'); return None
        j = self.version(i, W[stay], Cf[keep], np.asarray(r['axes'])[keep], np.asarray(r['joints'])[keep])
        shutil.copy(keepout_file(self.run, r['id']), keepout_file(self.run, self.c.base.records[j]['id'])); return j

    def blame(self, slots, reason):
        """The tasks a failure of the full audit names (the KUKA, an interference), or for the stance of several poses
        the tasks off the main pose (the one with the most area): {task index: working faces}."""
        recs = {s['members'][0]: self.c.base.records[s['members'][0]] for s in slots}
        named = [i for i, r in recs.items() if reason.startswith(r['id']+':')]
        if named: return {i: recs[i]['working_face_ids'] for i in named}
        if not any(k in reason for k in ('foot region', 'resting plane', 'pressure hull')): return {}
        poses = []
        for s in slots:
            i = s['members'][0]; g = np.asarray(s['X'])[:3, :3]@(recs[i]['pose'][:3, :3].T@C.DOWN)
            for h, ks in poses:
                if 1-g@h < SAME_DOWN: ks.append(i); break
            else: poses.append((g, [i]))
        if len(poses) < 2: return {}
        main = max(poses, key=lambda p: sum(self.area[recs[i]['working_face_ids']].sum() for i in p[1]))[1]
        return {i: recs[i]['working_face_ids'] for _, ks in poses if ks is not main for i in ks}

    def audited(self, group):
        """The full audit at the end (combine.accept); a failure that names tasks gives their faces away, again."""
        for n in range(self.policy['final_audits']):
            began = time.monotonic(); file = data(self.run)/FOLDER/'groups'/f'final_{n}.npz'
            # The audit without combine's foot-region oracle, then Step5's feet test on the support it accepted
            # (recorded; Step5 builds the feet and audits the stance in every installation).
            result = C.accept(self.run, group['slots'], C.group_solids(self.run, group)['s'], np.inf, file, stance=False)
            self.log('AUDIT', group['name'], 'PASSED' if result['passed'] else 'failed: '+str(result.get('reason'))[:200], '(%.0f s)' % (time.monotonic()-began))
            if result['passed']:
                out = dict(group, file=result['file'], slots=result['slots'], volume_cm3=result['membrane_cm3'], bbx_m3=result['bbx_m3'],
                           poses=result['poses'], audit={k: v for k, v in result.items() if k not in ('slots', 'unsupported', 'file')})
                self.base_group = None
                why = self.stance(result['slots'], C.group_solids(self.run, out)['s']); self.log('STANCE', 'Step5 feet test:', why or 'PASSED')
                if why: out['stance'] = why
                return out
            reason = result.get('reason') or ''
            if any(k in reason for k in ('foot region', 'resting plane', 'pressure hull')):
                # The oracle only predicts Step5's feet; Step5 builds them on every resting face and audits the stance in
                # every installation, so the support goes on and Step5 judges.
                return dict(group, stance=reason)
            failing = self.blame(group['slots'], reason)
            if not failing and result.get('unsupported'):
                failing = {i: self.c.base.records[i]['working_face_ids'] for i in result['unsupported']}
            moved = self.give(group['slots'], C.group_solids(self.run, group)['s'], failing) if failing else None
            if moved is None: return dict(group, failed=result.get('reason'))
            slots, keep, _ = moved; group = self.save(keep, slots, group['chain'], group['name'])
        return dict(group, failed='full audit failed %d times' % self.policy['final_audits'])

    def write(self, group, left):
        folder = data(self.run)/FOLDER; extra = self.c.base.records[self.first:]; arrays = {}
        for r in extra:
            arrays.update({r['id']+'_working': np.asarray(r['working_face_ids'], int), r['id']+'_certified': np.asarray(r['certified_face_ids'], int),
                           r['id']+'_axes': np.asarray(r['axes']), r['id']+'_joints': np.asarray(r['joints']), r['id']+'_pose': np.asarray(r['pose'])})
        np.savez_compressed(folder/'records.npz', **arrays)
        write_json(folder/'records.json', dict(step3_plan_sha256=sha256(data(self.run)/'step3_seats/plan.npz'),
                                              records=[dict(id=r['id'], seat=r['seat'], exit_object=np.asarray(r['exit_object']).tolist()) for r in extra]))
        recs = [self.c.base.records[s['members'][0]] for s in group['slots']]; worked = np.zeros(len(self.area), bool)
        for r in recs: worked[r['working_face_ids']] = True
        passed = 'failed' not in group and 'stance' not in group and not left
        # Rotational reuse about any axis: a seat's installations rest on as many faces as its tasks' downs differ.
        seats = []
        for X, ks in C.seats(group):
            downs = []
            for j in ks:
                g = np.asarray(X)[:3, :3]@(recs[j]['pose'][:3, :3].T@C.DOWN)
                if not any(1-g@h < SAME_DOWN for h in downs): downs.append(g)
            seats.append(dict(tasks=[recs[j]['id'] for j in ks], resting_faces=len(downs)))
        bases = []                                                # every resting face: its tasks and their area
        for X, ks in C.seats(group):
            for j in ks:
                g = np.asarray(X)[:3, :3]@(recs[j]['pose'][:3, :3].T@C.DOWN)
                b = next((b for b in bases if 1-g@np.asarray(b['down_support']) < SAME_DOWN), None)
                if b is None: b = dict(down_support=g.tolist(), tasks=[], area=np.zeros(len(self.area), bool)); bases.append(b)
                b['tasks'].append(recs[j]['id']); b['area'][recs[j]['working_face_ids']] = True
        for b in bases: b['area_fraction_of_target'] = float(self.area[b.pop('area') & self.target].sum()/self.total)
        reuse = [s for s in group['chain'] if s['operation'] == 'rotational_reuse']
        report = dict(status='passed' if passed else 'failed' if 'failed' in group else 'partial', run=self.run, policy=self.policy,
                      reason=group.get('failed'), stance_left_to_step5=group.get('stance'), group=group,
                      task_count=len(recs), seat_count=len(C.seats(group)), seats=[len(ks) for _, ks in C.seats(group)],
                      seat_installations=seats, rotational_reuse=len(reuse), resting_faces=len(C.downs(self.c, group)), bases=bases,
                      juxtaposition=sum(s['operation'] == 'juxtaposition' for s in group['chain']),
                      covered_fraction_of_target=float(self.area[worked & self.target].sum()/self.total),
                      left=[self.c.base.records[t]['id'] for t in left], windows=self.windows_in, replaced_by_windows=self.replaced,
                      tasks=[dict(id=r['id'], step3_seat=r['seat'], seat=k, area_fraction_of_target=float(self.area[r['working_face_ids']].sum()/self.total))
                             for k, (_, ks) in enumerate(C.seats(group)) for r in [recs[j] for j in ks]],
                      chain=group['chain'], finals=[] if 'failed' in group else [dict(group, sample=0)],
                      input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [data(self.run)/'step3_seats/report.json',
                                    data(self.run)/'step3_seats/plan.npz', ROOT/'whole_task_2/step4/seat_combine.py']},
                      seconds=time.monotonic()-self.began)
        write_json(folder/'report.json', C.plain(report))
        self.log('STEP4', self.run, report['status'].upper(), '%d tasks in %d seats %s, %d resting faces (per seat %s); %d rotational reuse, %d juxtaposition, %d windows; '
                 '%.1f cm3, bbx %.0f cm3, covered %.3f' % (report['task_count'], report['seat_count'], report['seats'], report['resting_faces'],
                 [s['resting_faces'] for s in seats], report['rotational_reuse'], report['juxtaposition'], len(self.windows_in), group['volume_cm3'],
                 group['bbx_m3']*1e6, report['covered_fraction_of_target']), ('left: '+' '.join(report['left'])) if left else '')
        self.log('BASES', '; '.join('%d tasks, %.3f of the object' % (len(b['tasks']), b['area_fraction_of_target']) for b in bases))
        return report


def run(run_name, workers=16):
    s = Combine(run_name, workers)
    try:
        group, left = s.search(); return s.write(group, left)
    finally: s.workers.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run'); p.add_argument('--workers', type=int, default=16)
    a = p.parse_args(); run(a.run, a.workers)
