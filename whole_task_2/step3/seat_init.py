"""Step3 for a big task: at most two seats, split by a parting direction (no candidate pool, no seat count).

A seat is an exit e (object frame).  It opens the object's front F(e), the faces turned to e (n.e > 0), and its
support is the back half-shell; every task of the seat leaves through e and works only faces of F(e), so the tool
stays on the opened side and does not cut the seat's back.  One exit opens a half; the two exits d and -d open the
two halves that together are the whole shape, so a big task needs at most two seats:
  - one, if a single opening holds all of the big task (all but `uncovered`) that the screen reaches with e rising;
  - else two, the big task split by a parting direction d: the faces with n.d > 0 go to the seat of exit d, the rest
    to the seat of -d.  Where the line between the halves runs is Step4's to optimize; here d is only the start:
    of near-uniform directions, the one whose two halves take the fewest tasks (screened tasks, Step1's screen
    and the work cube, no exact checks), ties to what is left, then to the even split.
Each seat's half is tiled greedily by tasks.  A task is a pose (the support's face down) and a turn about the
vertical, the work cube at its one best translation (for the faces not yet worked); its reach the faces of the half
in the cube that pass Step1's exact needle, head and KUKA checks, worked ones too.  Turns of the seat's poses come
first; when none adds min_gain of the big task (faces not yet worked), a new pose: e upright, or aimed at a face not
yet worked (a needle axis that reaches it, a turn about the needle at which e rises).  The seat is done when what it
leaves of its half is at most `uncovered` of it, or no candidate adds min_gain.
Then the work areas, Step4's work-area rule with every task's weight at zero: each face of a half goes to the task of
its seat that faces it most squarely among those that reach it -- the smallest angle between the face normal and the
task's needle, ties to the earlier task.  A task left with no face is dropped (and a pose left with no task).  What no
task reaches -- a deep pit the needle never reaches, a sliver between two cubes -- goes to the nearest task of its
seat (by face centre), uncertified: the plan is an approximation and Step4 moves the line and the tasks anyway.  Work
areas never overlap.

Every seat's support, the same for all its tasks (supports()): the 5 mm outer layer minus the object, the exit sweep
along e and every task's tool keep-out (needle and head), its largest piece, object frame.  Geometry only: Step4
audits loads, the KUKA and the feet.

Writes data/<run>/step3_seats/ (report.json, plan.npz: per task <id>_face_ids, _axes, _joints, _pose, _exit_object as
big_init, _absorbed_ids, and its reach _reach_ids, _reach_axes, _reach_joints for Step4's rule; supports.npz,
supports.json) and output/<run>/step3/seats.png, seats_connected.png.

  python -m whole_task_2.step3.seat_init whole_shape [--workers 16] [--draw: supports and figures of the saved plan]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, load_shape, sha256, write_json
from whole_task_2.step1 import plan_reach as P, reach as R
from whole_task_2.step1.workspace import onto
from . import independent_geometry as G
from .big_init import CONFIG, big_task, station, placement, certify, task_record, pinned
from .exit_geometry import whole_object_sweep
from .tool_keepout import keepout

POLICY = dict(parting_spacing_deg=10.,     # candidate parting directions (near-uniform; d and -d are one)
              turns=24,                    # turns about the vertical tried per pose
              aims=6,                      # aimed new poses tried at a time, besides e upright
              certify=4,                   # best screened candidates certified exactly at a time
              min_gain=.005,               # a task adds at least this share of the big task (as Step4's combine)
              uncovered=.01,               # a seat is done when it leaves at most this share of its half
              exit_rise_deg=10.,           # the exit rises at least this much (as Step4's combine)
              position_step=1/6,           # work-cube corner grid, share of the cube edge
              seed=20261026)
UP = np.array([0., 0., 1.])


def rising_axes(axes, a, e, rise):
    """Needle axes (object frame) along which some turn about the needle lets exit e rise: with the axis taken to
    world a, e keeps its angle to a, and its highest height is cos * a_z + sin * |a_xy|."""
    c = axes@e
    return c*a[2]+np.sqrt(np.clip(1-c*c, 0, 1))*np.linalg.norm(a[:2]) >= rise-1e-12


def cut_one(args):
    """One piece the seats remove, object frame: a task's tool keep-out, or a seat's exit sweep."""
    kind, vertices, faces, value, s, reach, sides = args
    source = trimesh.Trimesh(vertices, faces, process=False)
    if kind == 'sweep': return G.mesh(whole_object_sweep(source, value[0], value[1])[0])
    return G.mesh(keepout(source, value[0], value[1], s, reach, sides))


def supports(run, workers=16):
    """Every seat's support (module docstring), saved to supports.npz; volumes to supports.json."""
    began = time.monotonic(); folder = ROOT/'whole_task_2/data'/run/'step3_seats'; report = json.loads((folder/'report.json').read_text())
    source = load_shape(ROOT/'objects'/report['object']/'mesh.stl'); s = report['setup']
    with np.load(ROOT/'whole_task_2/data'/report['object']/'step3_mold_plate/fitted_outer.npz') as o: outer = trimesh.Trimesh(o['vertices'], o['faces'], process=False)
    tool = {}
    for name in ('independent_policy.json', 'membrane_policy.json', 'tool_policy.json'): tool.update(json.loads((Path(__file__).parent/name).read_text()))
    reach = float(np.linalg.norm(source.extents))+2*tool['mold_outward_displacement_m']; sides = tool['tool_polygon_sides']
    length = float(np.ceil(4*np.linalg.norm(source.extents)/.1)*.1)
    with np.load(folder/'plan.npz') as z: z = {k: z[k] for k in z.files}
    jobs = [('sweep', source.vertices, source.faces, (np.asarray(seat['exit_object']), length), s, reach, sides) for seat in report['seats']]
    jobs += [('tool', source.vertices, source.faces, (z[t['id']+'_face_ids'], z[t['id']+'_axes']), s, reach, sides)
             for seat in report['seats'] for t in seat['tasks']]
    with pinned(workers) as pool: cuts = list(pool.map(cut_one, jobs))
    sweeps = cuts[:len(report['seats'])]; tools = iter(cuts[len(report['seats']):]); arrays = {}; rows = []
    for seat, sweep in zip(report['seats'], sweeps):
        cut = [G.solid(sweep)]+[G.solid(next(tools)) for _ in seat['tasks']]
        value = G.solid(outer)-G.solid(source)-G.union(cut); parts = value.decompose()
        keep = max(parts, key=lambda p: p.volume()); mesh = G.mesh(keep); total = sum(G.volume(p) for p in parts)
        arrays.update({seat['id']+'_vertices': mesh.vertices, seat['id']+'_faces': mesh.faces})
        rows.append(dict(id=seat['id'], volume_cm3=G.volume(keep)*1e6, largest_share=G.volume(keep)/max(total, 1e-30), pieces=len(parts)))
        print('SUPPORT', seat['id'], '%.1f cm3' % rows[-1]['volume_cm3'], 'largest piece %.3f of %d' % (rows[-1]['largest_share'], len(parts)), flush=True)
    np.savez_compressed(folder/'supports.npz', **arrays)
    write_json(folder/'supports.json', dict(rule='the 5 mm outer layer minus the object, the exit sweep along the seat exit and every '
                                                 'task tool keep-out (needle and head), largest piece, object frame; geometry only',
                                            seats=rows, plan_sha256=sha256(folder/'plan.npz'), seconds=time.monotonic()-began))
    return rows


class Tiler:
    """The station's view of the object: the screen, the work cube and the tiling of one seat's half."""
    def __init__(self, mesh, s, axes, reach, target, policy):
        self.mesh, self.s, self.axes, self.reach, self.target, self.policy = mesh, s, axes, reach, target, policy
        self.a = np.asarray(s['working_direction_world'], float)/np.linalg.norm(s['working_direction_world'])
        self.cosb = np.cos(np.radians(s['max_tilt_deg'])); self.rise = np.sin(np.radians(policy['exit_rise_deg']))
        self.area = mesh.area_faces; self.N = mesh.face_normals; self.gain = policy['min_gain']*float(self.area[target].sum())
        self.points = R.test_points(mesh)

    def inside(self, M, t):
        """Faces wholly inside the work cube, the object turned by M and shifted by t (as placement)."""
        edge = self.s['surface_workspace']['edge_m']; lo = np.asarray(self.s['surface_workspace']['bounds_world_m'][0])
        X = self.points@M.T+t; return ((X >= lo+1e-7) & (X <= lo+edge-1e-7)).all((1, 2))

    def opening(self, e):
        """The faces of the big task that exit e opens and the screen reaches along an axis with e rising."""
        return (self.N@e > 0) & self.target & self.reach[rising_axes(self.axes, self.a, e, self.rise)].any(0)

    def turns(self, M0, e, pose, origin, half, left):
        """The pose's turns about the vertical, each at its best cube translation for the faces not yet worked: its
        mask every face of the half the screen reaches in that cube, worked ones too (the work-area rule assigns
        them); its value the faces not yet worked, screened (an upper bound)."""
        if (M0@e)[2] < self.rise-1e-12: return []
        out = []
        for psi in np.arange(self.policy['turns'])*2*np.pi/self.policy['turns']:
            M = Rotation.from_rotvec(psi*UP).as_matrix()@M0; u = M.T@self.a
            f = (self.N@e > 0) & half & self.target & self.reach[self.axes@u >= self.cosb-1e-12].any(0)
            put = placement(self.mesh, M, f & left, self.s, self.policy)
            if put is None: continue
            t = put[0]; mask = f & self.inside(M, t)
            out.append(dict(origin=origin, d=e, rotation=M, translation=t, mask=mask, left=left, pose=pose, M0=M0,
                            turn_deg=float(np.degrees(psi)), screened=float(self.area[mask & left].sum())))
        return out

    def new_poses(self, e, left, rng):
        """e upright, and poses aimed at faces of `left` the opening holds: a reaching axis with e rising, to world a,
        a turn about the needle at which e rises."""
        out = [(onto(e, UP), 'upright')]; ok = rising_axes(self.axes, self.a, e, self.rise)
        w = np.where(self.opening(e) & left, self.area, 0.)
        for f in (rng.choice(len(w), size=self.policy['aims'], p=w/w.sum()) if w.sum() > 0 else []):
            k = np.flatnonzero(self.reach[:, f] & ok)
            if not len(k): continue
            M = onto(self.axes[rng.choice(k)], self.a)
            spins = [p for p in np.arange(36)*np.pi/18 if (Rotation.from_rotvec(p*self.a).as_matrix()@M@e)[2] >= self.rise]
            if spins: out.append((Rotation.from_rotvec(rng.choice(spins)*self.a).as_matrix()@M, 'aimed'))
        return out

    def screened(self, cands):
        """The screen's judge: the candidate screened largest, its screened faces taken as certified."""
        c = max(cands, key=lambda c: c['screened'], default=None)
        if c is None or c['screened'] < self.gain: return None
        c['certified'] = c['mask']; return c

    def seat(self, e, half, judge, rng, log=None):
        """One seat's tasks over its half: turns of its poses first, then a new pose (module docstring)."""
        left = half & self.target; goal = self.policy['uncovered']*float(self.area[left].sum()); poses = []; tasks = []
        while self.area[left].sum() > goal:
            c = judge([x for i, M0 in enumerate(poses) for x in self.turns(M0, e, i, 'turn', half, left)])
            if c is None:
                c = judge([x for M0, origin in self.new_poses(e, left, rng) for x in self.turns(M0, e, len(poses), origin, half, left)])
                if c is None: break
                poses.append(c['M0'])
            c['gain'] = float(self.area[c['certified'] & left].sum()); tasks.append(c); left = left & ~c['certified']
            if log: log(c, left)
        return tasks, poses, left


def exact_judge(pool, T, axes, reach, eps):
    """Tiler.seat's exact judge: certify the best screened candidates (Step1's checks); the one adding the most faces
    not yet worked, if it adds min_gain."""
    area = T.area
    def exact(cands):
        cands = sorted((c for c in cands if c['screened'] >= T.gain), key=lambda c: -c['screened'])[:T.policy['certify']]
        if not cands: return None
        certify(pool, T.mesh, axes, reach, cands, T.s, eps)
        new = lambda c: area[c['certified'] & c['left']].sum()
        c = max(cands, key=new)
        return c if new(c) >= T.gain else None
    return exact


def assign(seat, mesh, s, letter='', total=1.):
    """The work areas of one seat's tasks: each face to the task that faces it most squarely among those that reach it
    (the smallest angle between its normal and the needle; Step4's rule, every weight at zero), ties to the earlier
    task.  A task left with no face is dropped, and a pose left with no task; what no task reaches (seat['left']) goes
    to the nearest task (by face centre), uncertified (seat['absorbed'], one mask per task)."""
    area = mesh.area_faces; N = mesh.face_normals; C = mesh.triangles_center
    a = np.asarray(s['working_direction_world'], float); a = a/np.linalg.norm(a)
    tasks = seat['tasks']; seat['absorbed'] = []; seat['dropped'] = 0
    if not tasks: return
    cost = np.array([np.where(c['certified'], np.arccos(np.clip(-N@(c['rotation'].T@a), -1., 1.)), np.inf) for c in tasks])
    best = np.argmin(cost, 0); reached = np.isfinite(cost).any(0)
    for k, c in enumerate(tasks): c['owned'] = reached & (best == k)
    seat['tasks'] = tasks = [c for c in tasks if c['owned'].any()]; seat['dropped'] = len(cost)-len(tasks)
    used = sorted({c['pose'] for c in tasks}); seat['poses'] = [seat['poses'][i] for i in used]
    for c in tasks: c['pose'] = used.index(c['pose'])
    seat['absorbed'] = [np.zeros(len(area), bool) for _ in tasks]; rest = np.flatnonzero(seat['left'])
    if len(rest):
        faces = [np.flatnonzero(c['owned']) for c in tasks]; owner = np.concatenate([np.full(len(f), k) for k, f in enumerate(faces)])
        _, near = cKDTree(C[np.concatenate(faces)]).query(C[rest])
        for k in range(len(tasks)): seat['absorbed'][k][rest[owner[near] == k]] = True
    seat['left'] = np.zeros(len(area), bool)
    print('ASSIGN', letter, '%d tasks (%d dropped), %d poses:' % (len(tasks), seat['dropped'], len(used)),
          ' '.join('%.3f' % (area[c['owned'] | m].sum()/total) for c, m in zip(tasks, seat['absorbed'])), flush=True)


_TILER = []


def _init_screen(state):
    _TILER.append(Tiler(*state))


def screen_parting(args):
    """Screened task count of the two halves of parting direction d, and what they leave."""
    d, seed = args; T = _TILER[0]; rng = np.random.default_rng(seed); n = 0; left = 0.
    for e, half in ((d, T.N@d > 0), (-d, T.N@d <= 0)):
        tasks, _, rest = T.seat(e, half, T.screened, rng); n += len(tasks); left += float(T.area[rest].sum())
    return n, left


def plan(run, config_path, workers=16, policy=POLICY, output_root=DEFAULT_OUTPUT):
    began = time.monotonic(); config_path = Path(config_path).resolve(); rng = np.random.default_rng(policy['seed'])
    obj, shape, mesh, big, target = big_task(run); s = station(obj, config_path, mesh); area = mesh.area_faces; total = float(area[target].sum())
    reach_policy = json.loads((ROOT/'whole_task_2/step1/reach_policy.json').read_text())
    cache = ROOT/'whole_task_2/data'/obj/'step1_reach'; cache.mkdir(parents=True, exist_ok=True)
    axes, reach, _ = P.screen(mesh, shape, s, reach_policy, cache)
    eps = reach_policy['needle_start_fraction_of_length']*s['needle_length_m']; T = Tiler(mesh, s, axes, reach, target, policy)
    directions = R.sphere_axes(policy['parting_spacing_deg']); N = mesh.face_normals
    # One seat if a single opening holds the big task, else a parting direction: the fewest screened tasks.
    held = np.array([area[T.opening(e)].sum() for e in directions]); k = int(np.argmax(held)); parting = None
    if held[k] >= (1-policy['uncovered'])*total:
        halves = [(directions[k], target)]; print('ONE SEAT, exit', np.round(directions[k], 3), 'holds %.3f' % (held[k]/total), flush=True)
    else:
        half = directions[(directions[:, 2] > 1e-9) | ((abs(directions[:, 2]) <= 1e-9) & (directions[:, 1] > 0))]
        with ProcessPoolExecutor(workers, initializer=_init_screen, initargs=((mesh, s, axes, reach, target, policy),)) as pool:
            scored = list(pool.map(screen_parting, [(d, policy['seed']+i) for i, d in enumerate(half)]))
        n = np.array([x[0] for x in scored]); rest = np.array([x[1] for x in scored])/total
        split = np.array([area[target & (N@d > 0)].sum()/total for d in half])
        k = min(range(len(half)), key=lambda i: (n[i], round(rest[i], 3), abs(split[i]-.5))); d = half[k]
        if split[k] < .5: d = -d                                           # seat A takes the larger half
        halves = [(d, N@d > 0), (-d, N@d <= 0)]
        parting = dict(direction_object=d.tolist(), screened_tasks=int(n[k]), screened_left_fraction_of_target=float(rest[k]),
                       split=[float(area[target & h].sum()/total) for _, h in halves], candidates=len(half),
                       screened_tasks_over_candidates=dict(min=int(n.min()), median=float(np.median(n)), max=int(n.max())))
        print('PARTING', np.round(d, 3), 'screened %d tasks (candidates %d..%d), split %.3f/%.3f' % (
              n[k], n.min(), n.max(), *parting['split']), round(time.monotonic()-began), 's', flush=True)
    seats = []
    with ProcessPoolExecutor(workers, initializer=P._init_mesh, initargs=(str(shape), s)) as pool:
        exact = exact_judge(pool, T, axes, reach, eps)
        for j, (e, half) in enumerate(halves):
            letter = chr(65+j)
            print('SEAT', letter, 'exit', np.round(e, 3), 'half %.3f of the big task' % (area[target & half].sum()/total), flush=True)
            def log(c, left, letter=letter, counter=[0]):
                counter[0] += 1
                print('  TASK %s%d pose %d turn %3.0f: adds %.3f of the big task (screened %.3f), reaches %.3f; left of the half %.3f' % (
                      letter, counter[0], c['pose'], c['turn_deg'], c['gain']/total, c['screened']/total,
                      area[c['certified']].sum()/total, area[left].sum()/total), round(time.monotonic()-began), 's', flush=True)
            tasks, poses, left = T.seat(e, half, exact, rng, log)
            seats.append(dict(exit=e, half=float(area[target & half].sum()/total), poses=poses, tasks=tasks, left=left))
    for j, seat in enumerate(seats): assign(seat, mesh, s, chr(65+j), total)
    left = np.any([seat['left'] for seat in seats], axis=0) & target
    absorbed = np.any([m for seat in seats for m in seat['absorbed']], axis=0) & target
    # Save the plan, the tasks named by seat letter and order.
    folder = ROOT/'whole_task_2/data'/run/'step3_seats'; folder.mkdir(parents=True, exist_ok=True); arrays = {}; rows = []
    for j, seat in enumerate(seats):
        letter = chr(65+j); tasks = []
        for n_, c in enumerate(seat['tasks']):
            rec = task_record(dict(c, certified=c['owned']), f'{letter}{n_+1}'); name = rec['id']; faces = np.asarray(rec['working_face_ids'])
            extra = np.flatnonzero(seat['absorbed'][n_]); reached = np.flatnonzero(c['certified'])
            arrays.update({f'{name}_face_ids': faces, f'{name}_axes': rec['axes'], f'{name}_joints': rec['joints'],
                           f'{name}_pose': rec['pose'], f'{name}_exit_object': rec['exit_object'], f'{name}_absorbed_ids': extra,
                           f'{name}_reach_ids': reached, f'{name}_reach_axes': c['axis'][reached], f'{name}_reach_joints': c['joints'][reached]})
            tasks.append(dict(id=name, pose=c['pose'], pose_origin=c['origin'], turn_deg=c['turn_deg'], face_count=len(faces)+len(extra),
                              area_fraction_of_target=float((area[faces].sum()+area[extra].sum())/total),
                              certified_fraction_of_target=float(area[faces].sum()/total),
                              absorbed_fraction_of_target=float(area[extra].sum()/total),
                              reach_fraction_of_target=float(area[reached].sum()/total), tiling_gain_fraction_of_target=c['gain']/total,
                              screened_fraction_of_target=c['screened']/total, certification=c['counts']))
        rows.append(dict(id=letter, exit_object=seat['exit'].tolist(), half_fraction_of_target=seat['half'],
                         pose_count=len(seat['poses']), down_object=[(M0.T@-UP).tolist() for M0 in seat['poses']], dropped_tasks=seat['dropped'],
                         area_fraction_of_target=float(sum(t['area_fraction_of_target'] for t in tasks)),
                         absorbed_fraction_of_target=float(sum(t['absorbed_fraction_of_target'] for t in tasks)),
                         left_fraction_of_target=float(area[seat['left']].sum()/total), tasks=tasks))
    np.savez_compressed(folder/'plan.npz', **arrays, target=target, left=left, absorbed=absorbed)
    report = dict(status='passed', run=run, object=obj, big_task=str(big.relative_to(ROOT)), station=str(config_path.relative_to(ROOT)),
        setup=s, policy=dict(policy), parting=parting, seat_count=len(rows), pose_count=sum(r['pose_count'] for r in rows),
        task_count=sum(len(r['tasks']) for r in rows), seats=rows,
        covered_fraction_of_target=float(area[target & ~left].sum()/total), left_fraction_of_target=float(area[left].sum()/total),
        certified_fraction_of_target=float(area[target & ~left & ~absorbed].sum()/total),
        absorbed_fraction_of_target=float(area[absorbed].sum()/total),
        left_face_ids=np.flatnonzero(left).tolist(),
        rule='at most two seats: one exit if one opening holds the big task, else the halves of a parting direction d '
             '(exits d and -d), d the fewest screened tasks; a seat leaves through its exit and works only its half; a '
             'task: a pose and a turn about the vertical, one cube translation, reaching the half faces in the cube that '
             'pass Step1 exactly, added while it reaches faces not yet worked; each face to the task of its seat that faces '
             'it most squarely among those that reach it (Step4 work-area rule, every weight zero), a task with no face '
             'dropped; what no task reaches goes to the nearest task of its seat, uncertified; the line between the halves '
             'and the work areas are optimized in Step4',
        input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [shape, big, config_path, Path(__file__)]},
        seconds=time.monotonic()-began)
    write_json(folder/'report.json', report)
    supports(run, workers)
    print('STEP3 SEATS', run, report['seat_count'], 'seats,', report['pose_count'], 'poses,', report['task_count'], 'tasks;',
          'covered %.3f (certified %.3f, absorbed %.3f), left %.3f' % (report['covered_fraction_of_target'],
          report['certified_fraction_of_target'], report['absorbed_fraction_of_target'], report['left_fraction_of_target']),
          round(time.monotonic()-began), 's', flush=True)
    from .render_seat_init import render
    render(run, Path(output_root)/run/'step3')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run'); p.add_argument('--config', default=None); p.add_argument('--workers', type=int, default=16)
    p.add_argument('--draw', action='store_true')
    a = p.parse_args(); run = a.run
    if a.draw:
        from .render_seat_init import render
        supports(run, a.workers); render(run, Path(DEFAULT_OUTPUT)/run/'step3'); return
    config = a.config or (f"robots_config/random/{run.split('+', 1)[1]}.json" if '+' in run else CONFIG)
    plan(run, ROOT/config, a.workers)


if __name__ == '__main__':
    main()
