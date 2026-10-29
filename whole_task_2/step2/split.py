"""Step2: the tasks of a big working area, split top-down until every task's support can be built.

The area is not divided first: it starts as one task.  A task is a set of faces worked from one nominal needle
direction d (object frame), against the faces' area-weighted mean normal; the object leaves against the needle.  It is
accepted by Step3's own check (step3/tool_membranes.py), without the work arm:
  - every face is reached by the needle within the tilt cone about d, needle and head clear of the object
    (step1/reach.certify, the smallest tilt first);
  - the largest legal support -- the 5 mm outer layer less the object, every face's needle and head and the exit
    sweep -- is one piece, lets the object leave and holds every paired force and moment of the task (32,768 samples
    and the continuous outer targets), at one of Step3's turns about the needle (the first that does).
Removing material only weakens a support, so when the largest legal support of a task fails, every support of it does.

A task that fails splits in two by its faces' normals (area-weighted 2-means; equal normals: by position): opposite
faces are what fight -- one's support stands on the other's work side.  Each half has a subset of the demands and a
smaller keep-out, so it is easier; the split ends at single faces at worst, and a face that fails alone is reported.
Faces no needle axis reaches at all are reported first and left out.

The work arm is not asked while splitting (a small one only means more turns of a task, later).  At the end each task
is placed as Step1 places a pose -- its area's centre over the work point, the object on the table, the needle along
the station's direction -- and the KUKA is solved for every face (step1/arm.Check); faces it cannot reach are left out
and reported.  The tasks are written as Step1's plan (data/<run>/step1_reach: plan.json, plan.npz, validation.json),
so Step3, Step4 and Step5 run unchanged.  Step2's own record: data/<run>/step2_split/ (report.json, supports.npz),
its figure output/<run>/step2/split.png.  step2/random_tasks.py (the baseline) shares all but the split.

  python -m whole_task_2.step2.split B+2task --task objects/B/big_tasks/2task.json --config robots_config/random/07_kuka.json
  python -m whole_task_2.run_station B+2task --start step3
"""
import argparse
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from robots_config.constraints import load_config
from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, load_shape, object_of, sha256, write_json
from whole_task_2.step1 import plan_reach as P1, reach_video as V
from whole_task_2.step1.arm import OK, order
from whole_task_2.common.wrenches import resolve_policy
from whole_task_2.operations.state import Layout
from whole_task_2.step3 import tool_membranes as TM

FOLDER = 'step2_split'
SPARE_M = 10.                   # the work cube Step2 does not use: a box this far round the work point holds every tip
PALETTE = [(.90, .42, .55), (.30, .62, .85), (.95, .70, .25), (.45, .75, .45), (.65, .50, .85), (.95, .55, .35),
           (.35, .80, .80), (.80, .80, .35), (.60, .40, .30), (.55, .55, .90)]


def data(run):
    return ROOT/'whole_task_2/data'/run


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def unit(v):
    v = np.asarray(v, float); return v/np.linalg.norm(v)


# ---------------------------------------------------------------------------------------------
# The station's tool, Step3's model without the work arm.

def station(run, task, config):
    """Step1's setup for the run's object and the station's tool, the work point and the table as Step1 places them;
    the work cube is not used (a box SPARE_M round the work point)."""
    mesh, shape, target, s = P1.setup(object_of(run), config, task)
    point, forward = V.work_point(s, .5*float(mesh.extents.max()))
    table = float(point[2]-.5*mesh.extents.max())
    s.update(work_point_m=point.tolist(), work_point_forward_of_window_m=forward,
             work_point_rule="Step1's: the first point forward of the window centre where the KUKA reaches every tip within half the object size at any tilt in the cone",
             table_height_m=table, table_bounds_world_m=[[point[0]-.25, point[1]-.25, table-.02], [point[0]+.25, point[1]+.25, table]],
             surface_workspace=dict(shape='none', bounds_world_m=[(point-SPARE_M).tolist(), (point+SPARE_M).tolist()],
                                    note='Step2 does not use the work cube: the arm is only solved for the tasks it made'),
             config_file=rel(config),
             pose_rule="Step2: the task's needle direction along the station's, turned about it, its area's centre over the work point, the object on the table")
    return mesh, shape, target, s


def step3_inputs(run, config):
    """Step3's policy, the object's 5 mm outer layer and its physical properties (tool_membranes.inputs_and_policy
    without a plan)."""
    policy = {}
    for p in TM.POLICIES: policy.update(json.loads(p.read_text()))
    policy = resolve_policy(policy, load_config(config)); policy['task_count'] = 1
    with np.load(ROOT/'whole_task_2/data'/object_of(run)/'step3_mold_plate/fitted_outer.npz') as z:
        outer = trimesh.Trimesh(z['vertices'], z['faces'], process=False)
    return policy, outer, json.loads((ROOT/'objects'/object_of(run)/'meta.json').read_text())


class Free(TM.ToolMembraneModel):
    """Step3's model without the work arm: Step2 splits by the support alone."""
    def arm_hits(self, world, record):
        return []


_W = {}


def _init(cpus, run, shape, setup, config):
    """A worker: one CPU (manifold spreads a Boolean over every CPU it may use), Step1's mesh and arm, Step3's inputs."""
    os.sched_setaffinity(0, {cpus.get()})
    P1._init_mesh(shape, setup); policy, outer, physical = step3_inputs(run, config)
    _W.update(source=P1._mesh, outer=outer, physical=physical, policy=policy, setup=setup, run=run,
              a=unit(setup['working_direction_world']))


def workers_pool(run, shape, s, config, workers):
    """Workers pinned one per CPU, the least busy CPUs first (step4/combine.idle_first), so another run's pinned
    workers are not shared."""
    from whole_task_2.step4.combine import idle_first
    context = multiprocessing.get_context('spawn'); cpus = context.Queue(); free = idle_first(os.sched_getaffinity(0))
    for k in range(workers): cpus.put(free[k % len(free)])
    return ProcessPoolExecutor(workers, mp_context=context, initializer=_init, initargs=(cpus, run, str(shape), s, str(config)))


def support(job):
    """Step3's construction and audit of one task at one turn, without the work arm: the largest legal support."""
    name, faces, axes, P, angle = job; began = time.monotonic()
    faces = [int(f) for f in faces]
    record = dict(id=name, direction_index=0, working_face_ids=faces, certified_face_ids=faces, pose=P, axes=np.asarray(axes),
                  joints=np.zeros((len(faces), 4, 7)), angle_deg=angle)
    model = Free(_W['source'], _W['outer'], [record], _W['setup'], _W['physical'], _W['policy'], _W['run'])
    q = Layout(P[None], np.eye(4)[None], -_W['a'][None])
    try:
        candidate = model.build(q, facewise=True); model.audit(candidate)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as ex:
        return dict(passed=False, reason=str(ex)[:300], seconds=time.monotonic()-began)
    out = candidate['support']
    return dict(passed=True, volume_cm3=float(candidate['volume_cm3']), vertices=out.vertices, faces=out.faces,
                seconds=time.monotonic()-began)


# ---------------------------------------------------------------------------------------------
# A task: its needle direction, its place, its halves.

def nominal(mesh, faces):
    """(the needle direction against the faces' area-weighted mean normal, object frame; the area's centre)."""
    w = mesh.area_faces[faces]; n = w@mesh.face_normals[faces]
    if np.linalg.norm(n) < 1e-9*w.sum(): n = mesh.face_normals[faces[np.argmax(w)]]     # the normals cancel
    return -unit(n), (w[:, None]*mesh.triangles_center[faces]).sum(0)/w.sum()


def toward(d, a):
    """The least rotation taking unit d to unit a."""
    v = np.cross(d, a); s = float(np.linalg.norm(v)); c = float(d@a)
    if s > 1e-12: return Rotation.from_rotvec(v/s*np.arctan2(s, c)).as_matrix()
    if c > 0: return np.eye(3)
    u = np.cross(d, [1., 0., 0.]); u = u if np.linalg.norm(u) > .1 else np.cross(d, [0., 1., 0.])
    return Rotation.from_rotvec(np.pi*unit(u)).as_matrix()


def placement(mesh, d, centre, angle, s):
    """World <- object: d along the station's needle, turned by angle (deg) about it, the area's centre over the
    work point, the object on the table."""
    a = unit(s['working_direction_world']); R = Rotation.from_rotvec(a*np.radians(angle)).as_matrix()@toward(d, a)
    P = np.eye(4); P[:3, :3] = R; P[:3, 3] = np.asarray(s['work_point_m'])-R@centre
    P[2, 3] = s['table_height_m']-float((mesh.vertices@R.T)[:, 2].min())
    return P


def halves(mesh, faces):
    """Two halves by the faces' normals (area-weighted 2-means, from the normal farthest from the mean and the one
    farthest from it); when the normals are all equal, by position along the area's longest spread."""
    n = mesh.face_normals[faces]; w = mesh.area_faces[faces]

    def mean(m):
        v = w[m]@n[m]; return v/max(np.linalg.norm(v), 1e-300)
    first = n[np.argmin(n@mean(np.ones(len(faces), bool)))]; centres = [first, n[np.argmin(n@first)]]; side = None
    for _ in range(100):
        new = n@centres[1] > n@centres[0]
        if side is not None and np.array_equal(new, side): break
        side = new
        if side.all() or not side.any(): break
        centres = [mean(~side), mean(side)]
    if side.all() or not side.any():
        x = mesh.triangles_center[faces]; x = x-(w[:, None]*x).sum(0)/w.sum()
        t = x@np.linalg.svd(x*np.sqrt(w)[:, None], full_matrices=False)[2][0]; side = t > np.median(t)
        if side.all() or not side.any(): side = np.arange(len(faces)) >= len(faces)//2
    return faces[~side], faces[side]


def verdict(futures):
    """The first turn (in Step3's order) whose support holds, -1 when none does, None while an earlier one runs."""
    for k, f in enumerate(futures):
        if not f.done(): return None
        if f.result()['passed']: return k
    return -1


# ---------------------------------------------------------------------------------------------
# The split, the arm, the plan.

def split(pool, mesh, s, axes, reach, root, angles, eps):
    """Every task from the root area: (tasks, the tree's rows, faces that fail alone)."""
    cosb = np.cos(np.radians(s['max_tilt_deg'])); L, H, N, incidence = s['needle_length_m'], s['head_radius_m'], s['needle_radius_m'], s['max_incidence_deg']
    total = float(mesh.area_faces[root].sum()); pending = [dict(id='T', faces=np.asarray(root))]; tasks, rows, alone = [], [], []
    while pending:
        # The needle: for every face of every pending task, the least tilt about the task's direction that clears.
        jobs = []
        for node in pending:
            node['d'], node['centre'] = nominal(mesh, node['faces']); C = axes@node['d']
            for f in node['faces']:
                k = np.flatnonzero((C >= cosb-1e-12) & reach[:, f]); k = k[np.argsort(-C[k])]
                jobs.append((int(f), [(0, int(i), axes[i]) for i in k], L, H, N, eps, incidence))
        found = {face: (None if i is None else axes[i]) for face, _, i, _, _, _ in pool.map(P1._certify, jobs, chunksize=16)}
        # The support: Step3's turns about the needle, for each task its needle reaches whole; the first that holds.
        for node in pending:
            node['missing'] = [int(f) for f in node['faces'] if found[int(f)] is None]
            if node['missing']: continue
            node['axes'] = np.asarray([found[int(f)] for f in node['faces']])
            node['futures'] = [pool.submit(support, (node['id'], node['faces'], node['axes'], placement(mesh, node['d'], node['centre'], a, s), a))
                               for a in angles]
        live = [n for n in pending if 'futures' in n]
        while live:
            running = [f for n in live for f in n['futures'] if not f.done()]
            if running: wait(running, return_when=FIRST_COMPLETED)
            for node in list(live):
                k = verdict(node['futures'])
                if k is None: continue
                node['verdict'] = k; live.remove(node)
                for f in node['futures'][k+1:] if k >= 0 else []: f.cancel()
        following = []
        for node in pending:
            faces = node['faces']; k = node.get('verdict', -1)
            row = dict(id=node['id'], faces=len(faces), area_fraction_of_target=float(mesh.area_faces[faces].sum()/total),
                       needle_object=node['d'].tolist(), needle_misses=len(node['missing']))
            done = [(a, f.result()) for a, f in zip(angles, node.get('futures', [])) if f.done() and not f.cancelled()]
            row['turns'] = [dict(angle_deg=a, passed=r['passed'], reason=r.get('reason'), seconds=r['seconds']) for a, r in done]
            if k >= 0:
                a, r = done[k]
                tasks.append(dict(id=node['id'], faces=faces, d=node['d'], centre=node['centre'], axes=node['axes'], angle=a,
                                  pose=placement(mesh, node['d'], node['centre'], a, s), volume_cm3=r['volume_cm3'],
                                  vertices=r['vertices'], triangles=r['faces']))
                row.update(verdict='task', turn_deg=a, support_cm3=r['volume_cm3'])
            elif len(faces) == 1:
                alone.append(dict(face=int(faces[0]), reason='needle misses' if node['missing'] else done[-1][1]['reason']))
                row['verdict'] = 'fails alone'
            else:
                low, high = halves(mesh, faces); following += [dict(id=node['id']+'0', faces=low), dict(id=node['id']+'1', faces=high)]
                row['verdict'] = 'split'
            rows.append(row)
            print('STEP2', row['id'], row['faces'], 'faces', round(100*row['area_fraction_of_target'], 2), '%:', row['verdict'],
                  f"(needle misses {row['needle_misses']})" if row['needle_misses'] else
                  f"turn {row.get('turn_deg')}" if row['verdict'] == 'task' else (row['turns'][-1]['reason'] or '')[:100] if row['turns'] else '', flush=True)
        pending = following
    return tasks, rows, alone


def arm(pool, mesh, tasks):
    """{(face, task): (code, joints)}: the KUKA for every face of every task at its place, in Step1's walking order."""
    chunks = []
    for j, t in enumerate(tasks):
        W = t['pose']; axis = dict(zip(map(int, t['faces']), t['axes'])); f = t['faces'][order(mesh.triangles_center[t['faces']]@W[:3, :3].T)]
        chunks += [(j, W, [(int(g), axis[int(g)]) for g in f[i:i+60]]) for i in range(0, len(f), 60)]
    return {(face, j): (code, Q) for rows in pool.map(P1._arm_pairs, chunks) for face, j, code, Q in rows}


def write_plan(run, mesh, target, s, tasks, solved, inputs, eps, folder, source):
    """Step1's plan for Step3: one area per task with a face the KUKA reaches (its faces, its place), plan.npz's
    face_task, reach, reach_axis and joints; the faces the KUKA misses are left out of their task; a task's 'merged'
    faces (assumed reachable) are its own, uncertified.  folder: the Step2 record (data/<run>/<folder>), source: the
    module that made the tasks."""
    work = data(run)/'step1_reach'; F = len(mesh.faces); k = len(tasks); area = mesh.area_faces; total = float(area[target].sum())
    face_task = np.full(F, -1); reach = np.zeros((F, k), bool); reach_axis = np.zeros((F, k, 3)); joints = np.zeros((F, k, 4, 7))
    code = np.full((F, k), -1, np.int8); areas = []
    for j, t in enumerate(tasks):
        for f, x in zip(map(int, t['faces']), t['axes']):
            c, Q = solved[(f, j)]; code[f, j] = c
            if c == OK: reach[f, j] = True; reach_axis[f, j] = x; joints[f, j] = Q
        kept = np.flatnonzero(reach[:, j]); face_task[kept] = j; t['kept'] = kept
        # Faces assumed reachable (step2/seats.py: hard to reach, joined to the nearest task): the task's, uncertified.
        extra = np.setdiff1d(np.asarray(t.get('merged', []), int), kept); face_task[extra] = j; ids = np.union1d(kept, extra)
        areas.append(dict(id=t['id'], face_ids=ids.tolist(), face_count=len(ids), area_m2=float(area[ids].sum()),
                          area_fraction_of_target=float(area[ids].sum()/total), axis_object=t['d'].tolist(), turn_deg=t['angle'],
                          uncertified_face_ids=extra.tolist(),
                          placement=dict(rotation_world_object=t['pose'][:3, :3].tolist(), translation_world_m=t['pose'][:3, 3].tolist()),
                          arm_missed_face_ids=sorted(set(map(int, t['faces']))-set(kept.tolist())), step2_support_cm3=t['volume_cm3'],
                          seat=t.get('seat'), robot=t.get('robot', True),
                          **({} if t.get('exit') is None else dict(exit_object=unit(t['exit']).tolist()))))
    np.savez_compressed(work/'plan.npz', face_task=face_task, reach=reach, reach_axis=reach_axis, joints=joints, arm_code=code, target=target)
    plan = dict(status='passed', model=folder, object=object_of(run), run=run, setup=s, areas=areas,
                robot_free=not any(a['robot'] for a in areas),
                task_count=len(areas), target_face_count=int(target.sum()), target_area_m2=total,
                covered_fraction_of_target=float(area[face_task >= 0].sum()/total),
                task_definition=f"Step2's tasks ({rel(source)}); face_task = the faces of a task the KUKA reaches at its place",
                needle_start_m=eps, input_sha256=inputs, arrays_sha256=sha256(work/'plan.npz'), step2=rel(data(run)/folder/'report.json'))
    write_json(work/'plan.json', plan)
    return plan


def validate(run, pool):
    """Replay of the plan from disk: the partition (disjoint, in the big task), every place (the needle along the
    station's direction, the object on the table), every (face, axis) by Step1's exact needle and head test within
    the tilt cone of its task, every KUKA joint set by forward kinematics.  Written where Step3 reads it."""
    work = data(run)/'step1_reach'; plan = json.loads((work/'plan.json').read_text()); s = plan['setup']
    mesh = load_shape(ROOT/'objects'/object_of(run)/'mesh.stl'); a = unit(s['working_direction_world']); eps = plan['needle_start_m']
    with np.load(work/'plan.npz') as z: arrays = {k: z[k] for k in z.files}
    face_task, reach = arrays['face_task'], arrays['reach']; cosb = np.cos(np.radians(s['max_tilt_deg']))
    places = tilts = rise = True; listed = True; jobs = []; arm_jobs = []
    for j, row in enumerate(plan['areas']):
        R = np.asarray(row['placement']['rotation_world_object']); t = np.asarray(row['placement']['translation_world_m'])
        W = np.eye(4); W[:3, :3] = R; W[:3, 3] = t; d = np.asarray(row['axis_object']); f = np.flatnonzero(reach[:, j])
        if 'exit_object' in row: rise &= bool((R@unit(row['exit_object']))[2] >= .05)     # Step3's upper-hemisphere exit
        places &= bool(np.allclose(R.T@R, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1, atol=1e-12) and np.allclose(R@d, a, atol=1e-12)
                       and abs((mesh.vertices@R.T+t)[:, 2].min()-s['table_height_m']) < 1e-12)
        ids = np.flatnonzero(face_task == j)
        listed &= bool(ids.tolist() == row['face_ids'] and np.setdiff1d(ids, f).tolist() == row.get('uncertified_face_ids', []))
        if row.get('robot', True): tilts &= bool(np.all(arrays['reach_axis'][f, j]@d >= cosb-1e-12))
        else:       # robot-free (step2/seats.py): a face's needle is one of its base's turns about the vertical, so its
                    # axis lies within the tilt of the cone the station's needle sweeps about the object's down
            fall = np.arccos(np.clip(-a[2], -1, 1)); t = np.arccos(np.clip(-(arrays['reach_axis'][f, j]@R[2]), -1, 1))
            tilts &= bool(np.all(np.abs(t-fall) <= np.radians(s['max_tilt_deg'])+1e-9))
        jobs += [(int(g), [(j, 0, arrays['reach_axis'][g, j])], s['needle_length_m'], s['head_radius_m'], s['needle_radius_m'], eps, s['max_incidence_deg']) for g in f]
        if row.get('robot', True):        # a robot-free area has no joints: Step6 places the robot
            arm_jobs += [(W, [(int(g), arrays['reach_axis'][g, j], arrays['joints'][g, j]) for g in f[i:i+60]]) for i in range(0, len(f), 60)]
    passed = sum(r[1] is not None for r in pool.map(P1._certify, jobs, chunksize=16)); held = sum(pool.map(P1._arm_replay, arm_jobs))
    checks = dict(arrays=sha256(work/'plan.npz') == plan['arrays_sha256'], one_task_per_face=bool((reach.sum(1) <= 1).all()),
                  tasks_in_big_task=bool(arrays['target'][face_task >= 0].all()), face_lists_match=listed,
                  needle_along_station_object_on_table=places, axes_within_tilt_cone=tilts, exits_rise=rise,
                  exact_needle_and_head=passed == len(jobs), kuka_replay=held == sum(len(job[1]) for job in arm_jobs))
    result = dict(status='passed' if all(checks.values()) else 'failed', checks=checks, pairs=len(jobs), pairs_passed=passed,
                  arm_pairs_replayed=held, validator='whole_task_2/step2/split.py', plan_sha256=sha256(work/'plan.json'))
    write_json(work/'validation.json', result)
    if result['status'] != 'passed': raise ValueError('Step2 plan validation failed: '+json.dumps(checks))
    print('STEP2 PLAN VALIDATION PASSED', passed, 'pairs', flush=True)
    return result


def prepare(run_name, task, config):
    """A Step2 run's inputs: the station, the screen of tool axes, Step3's turns, the needle start; the big task's faces
    some needle axis reaches (root) and those none does."""
    task = Path(task).resolve(); config = Path(config).resolve()
    mesh, shape, target, s = station(run_name, task, config)
    work = data(run_name)/'step1_reach'; work.mkdir(parents=True, exist_ok=True)
    reach_policy = json.loads(P1.POLICY.read_text()); eps = reach_policy['needle_start_fraction_of_length']*s['needle_length_m']
    # The screen of tool axes depends on the object and the tool only: another run's of the object is a valid cache (key checked).
    if not (work/'screen.npz').exists():
        for other in sorted(ROOT.glob(f'whole_task_2/data/{object_of(run_name)}*/step1_reach/screen.npz')): shutil.copy2(other, work/'screen.npz'); break
    axes, reach, _ = P1.screen(mesh, shape, s, reach_policy, work); reach = np.asarray(reach)
    angles = [float(x) for x in step3_inputs(run_name, config)[0]['initial_axial_angles_deg']]
    ids = np.flatnonzero(target); unreachable = ids[~reach[:, ids].any(0)]
    print('STEP2', run_name, len(ids), 'faces;', len(unreachable), 'no needle axis reaches', flush=True)
    return dict(run=run_name, task=task, config=config, mesh=mesh, shape=shape, target=target, s=s, axes=axes, reach=reach,
                angles=angles, eps=eps, unreachable=unreachable, root=np.setdiff1d(ids, unreachable))


def finish(x, pool, tasks, folder, source, robot=True):
    """The KUKA for every task at its place (none for a robot-free plan: Step6 places the robot), the plan written
    where Step3 reads it, and its replay."""
    solved = arm(pool, x['mesh'], tasks) if robot else {(int(f), j): (OK, np.zeros((4, 7))) for j, t in enumerate(tasks) for f in t['faces']}
    inputs = {rel(p): sha256(p) for p in [x['shape'], x['task'], x['config'], Path(source), Path(__file__), *TM.POLICIES, P1.POLICY,
                                          ROOT/'whole_task_2/step1/needle_workspace.json']}
    plan = write_plan(x['run'], x['mesh'], x['target'], x['s'], tasks, solved, inputs, x['eps'], folder, source)
    return plan, validate(x['run'], pool)


def record(x, tasks, plan, validation, folder, rule, began, output_root, figure=None, **facts):
    """data/<run>/<folder>/: supports.npz (every task's Step2 support and place) and report.json; the figure (render,
    or the caller's figure(run, output_root, folder))."""
    area = x['mesh'].area_faces; total = float(area[x['target']].sum()); out = data(x['run'])/folder; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'supports.npz', **{k: v for t in tasks for k, v in
                                              ((t['id']+'_vertices', t['vertices']), (t['id']+'_faces', t['triangles']), (t['id']+'_pose', t['pose']))})
    report = dict(status='passed', run=x['run'], object=object_of(x['run']), big_task=rel(x['task']), station=rel(x['config']), rule=rule,
                  task_count=len(tasks), tasks=[dict(id=t['id'], faces=len(t['faces']), area_fraction_of_target=float(area[t['faces']].sum()/total),
                                                     needle_object=t['d'].tolist(), turn_deg=t['angle'], support_cm3=t['volume_cm3'], seat=t.get('seat'),
                                                     kuka_reaches=len(t['kept']), kuka_misses=len(t['faces'])-len(t['kept'])) for t in tasks],
                  no_needle_face_ids=x['unreachable'].tolist(), **facts,
                  task_fraction_of_target=float(sum(area[t['faces']].sum() for t in tasks)/total),
                  plan_fraction_of_target=plan['covered_fraction_of_target'], plan_validation=validation['status'],
                  seconds=time.monotonic()-began)
    write_json(out/'report.json', report)
    (figure or render)(x['run'], output_root, folder)
    print('STEP2 DONE', len(tasks), 'tasks; tasks cover', round(report['task_fraction_of_target'], 4),
          '; plan (KUKA) covers', round(plan['covered_fraction_of_target'], 4), round(report['seconds']), 's', flush=True)
    return report


def run(run_name, task, config, workers=22, output_root=DEFAULT_OUTPUT):
    began = time.monotonic(); x = prepare(run_name, task, config)
    pool = workers_pool(run_name, x['shape'], x['s'], x['config'], workers)
    try:
        tasks, rows, alone = split(pool, x['mesh'], x['s'], x['axes'], x['reach'], x['root'], x['angles'], x['eps'])
        plan, validation = finish(x, pool, tasks, FOLDER, __file__)
    finally:
        pool.shutdown(cancel_futures=True)
    return record(x, tasks, plan, validation, FOLDER, __doc__.split('\n\n')[1:4], began, output_root,
                  tree=rows, nodes=len(rows), failing_alone=alone)


# ---------------------------------------------------------------------------------------------
# The figure.

def render(run, output_root=DEFAULT_OUTPUT, folder=FOLDER, size=620, gap=40, columns=6):
    """output/<run>/step2/<split|random>.png, rows of `columns` panels: first the big task coloured by task (two
    opposite views, the object +Y up; dark: no task), then every task at its place: its Step2 support (blue) and the
    object, the task's faces pink."""
    from whole_task_2.step1.reach_video import Scene
    from whole_task_2.step3.render_tool_membranes import camera, image
    from whole_task_2.step3.render_independent import BLUE
    work = data(run)/'step1_reach'; plan = json.loads((work/'plan.json').read_text()); report = json.loads((data(run)/folder/'report.json').read_text())
    s = plan['setup']; a = unit(s['working_direction_world']); mesh = load_shape(ROOT/'objects'/object_of(run)/'mesh.stl')
    with np.load(work/'plan.npz') as z: target = z['target']
    with np.load(data(run)/folder/'supports.npz') as z: saved = {k: z[k] for k in z.files}
    panels = []
    # The big task, the object +Y up, from the side its area faces (or +Z, for an area all round) and from the other.
    U = np.eye(4); U[:3, :3] = Rotation.from_rotvec([np.pi/2, 0., 0.]).as_matrix()
    upright = trimesh.Trimesh(mesh.vertices@U[:3, :3].T, mesh.faces, process=False)
    scene = Scene(upright, s, size, size)
    try:
        colors = np.tile(scene.r.GRAY, (len(mesh.faces), 1)); colors[target] = (.25, .25, .25)
        for k, row in enumerate(plan['areas']): colors[row['face_ids']] = PALETTE[k % len(PALETTE)]
        n = mesh.area_faces[target]@mesh.face_normals[target]
        out = U[:3, :3]@(unit(n) if np.linalg.norm(n) > .05*mesh.area_faces[target].sum() else np.array([0., 0., 1.]))
        for v in (out, -out):
            view = np.array([v[0], v[1], 0.])+np.array([0., 0., .45]); scene.view = view/np.linalg.norm(view)
            scene.fit(upright.vertices, 1.08); panels.append(image(scene, colors, [], None, []))
    finally: scene.close()
    for k, row in enumerate(report['tasks']):
        P = saved[row['id']+'_pose']; support = trimesh.Trimesh(saved[row['id']+'_vertices'], saved[row['id']+'_faces'], process=False)
        scene = Scene(support, s, size, size)
        try:
            g = scene.r; bunny = g.Shape(scene.ctx, scene.shader, mesh.vertices, mesh.faces, g.GRAY, dynamic=True, metallic=.03, roughness=.63)
            colors = np.tile(g.GRAY, (len(mesh.faces), 1)); colors[plan['areas'][k]['face_ids']] = g.PINK; bunny.paint(colors)
            v = camera(P[:3, :3], a); scene.view = v/np.linalg.norm(v)
            scene.fit(np.vstack([support.vertices, mesh.vertices@P[:3, :3].T+P[:3, 3]]), 1.06)
            panels.append(image(scene, np.tile(BLUE, (len(support.faces), 1)), [], None, [(bunny, P)]))
        finally: scene.close()
    from PIL import Image
    wide = min(columns, len(panels)); rows = -(-len(panels)//wide)
    sheet = Image.new('RGB', (wide*(size+gap)+gap, rows*(size+gap)+gap), 'white')
    for k, panel in enumerate(panels): sheet.paste(panel, (gap+(k % wide)*(size+gap), gap+(k//wide)*(size+gap)))
    path = Path(output_root)/run/'step2'/(folder.removeprefix('step2_')+'.png'); path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)
    print('STEP2 FIGURE', path, flush=True)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run'); p.add_argument('--task', required=True); p.add_argument('--config', required=True)
    p.add_argument('--workers', type=int, default=22); p.add_argument('--render-only', action='store_true')
    a = p.parse_args()
    render(a.run) if a.render_only else run(a.run, a.task, a.config, a.workers)
