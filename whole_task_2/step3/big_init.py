"""Step3 for a big task, with Step1 folded in: a pool of candidate tasks, spread evenly over the big task.

Input: a big task (objects/<object>/big_tasks/<name>.json, its face_ids; the run is <name> on station 07 or
<name>+<station>) and a station (robots_config: the KUKA, its needle direction a, the tilt cone, the work cube).

Every task goes the same way: the support is put down with the object in it and stays where it is, the needle
works what it can, the object comes out, the support goes and the next task's comes.  A task is one placement of
the object: the object leaves against the needle (world -a), the way the needle came, so its exit d (object frame)
opens the support over the object's front, the faces turned to d (n.d > 0); the task works what of that front
the needle reaches from where the support stands: the big task's faces of the front wholly inside the work cube
that Step1's exact needle and head test passes (along an axis within max_tilt of the nominal) and that the KUKA
holds (Step1's arm test).  Its support -- the 5 mm outer layer minus the object, the tool keep-out over the work
area and the exit sweep along d -- is built and audited when Step4 draws the task (support() here).

The pool (BIG_TASK.md), the faces held by the tasks so far counted as it grows:
  1. `start` tasks of uniformly random orientations, each moved to the placement where the work cube holds the
     most of its front;
  2. then, `per_round` at a time until `pool`: a face drawn with odds area / (1 + times held)^2, the exit along a
     needle axis that reaches it (Step1's screen), a random turn about the needle, the cube placed to hold that
     face and the most of the little-held ones (weight 1 / (1 + times held));
  every task certified exactly at its placement (step1.plan_reach.certify_directions); one working less than
  min_area of the big task is dropped.  A face no axis of the screen reaches, or aimed at give_up times and still
  held by none, is hard: no task aims at it again; at the end it goes to the nearest task's work area, uncertified
  (Step4's final assignment).

Writes data/<run>/step3_init/ (report.json, initialization.npz) and output/<run>/step3/pool.png: one panel per
task, the object at its placement, its work area pink, the needle on its middle face.

  python -m whole_task_2.step3.big_init whole_shape [--workers 16]
"""
import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from robots_config.constraints import load_config
from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, load_shape, sha256, write_json
from whole_task_2.step1 import plan_reach as P, reach as R
from whole_task_2.step1.reach_video import work_point
from whole_task_2.step1.workspace import onto
from whole_task_2.operations.state import Layout
from .tool_membranes import ToolMembraneModel, turned

CONFIG = 'robots_config/random/07_kuka.json'
POLICY = dict(pool=50,                    # tasks in the pool
              start=20,                   # the first ones of uniformly random orientations
              per_round=2,                # then this many aimed tasks at a time
              give_up=3,                  # a face aimed at this often and still held by none is hard
              min_area=.02,               # a task working less than this share of the big task is dropped
              position_step=1/6,          # work-cube corner grid, share of the cube edge
              rounds=200,                 # at most this many aimed rounds
              seed=20261025)
LETTER = 'P'


def big_task(run):
    """The object, its mesh path and mesh, the big task's json and its target faces."""
    name = run.split('+', 1)[0]; found = sorted(ROOT.glob(f'objects/*/big_tasks/{name}.json'))
    if len(found) != 1: raise ValueError(f'{name}: expected one objects/*/big_tasks/{name}.json')
    path = found[0]; obj = path.parent.parent.name; shape = ROOT/'objects'/obj/'mesh.stl'; mesh = load_shape(shape)
    area = json.loads(path.read_text())
    if area['mesh_sha256'] != sha256(shape): raise ValueError(run+': the big task belongs to another mesh')
    target = np.zeros(len(mesh.faces), bool); target[area['face_ids']] = True
    return obj, shape, mesh, path, target


def station(obj, config_path, mesh):
    """Step1's setup, the work cube centred on the work point and the table one object size below the cube."""
    _, _, _, s = P.setup(obj, config_path)
    edge = float(load_config(config_path)['task_constraint']['surface_workspace']['edge_m'])
    point, forward = work_point(s, edge/2); point = np.asarray(point, float); lo, hi = point-edge/2, point+edge/2
    table = float(lo[2]-mesh.extents.max())
    s.update(work_point_m=point.tolist(), work_point_forward_of_window_m=forward,
             surface_workspace=dict(frame='world', shape='cube', edge_m=edge, bounds_world_m=[lo.tolist(), hi.tolist()],
                                    workspace_center_world_m=point.tolist(), rule='the configured cube centred on the work point',
                                    robot=s['robot']),
             table_height_m=table, table_bounds_world_m=[[point[0]-.25, point[1]-.25, table-.02], [point[0]+.25, point[1]+.25, table]],
             pose_rule='each task: an orientation, its exit against the needle, one translation; the support stays where it is put')
    return s


def front(mesh, axes, reach, d, beta, target):
    """Faces of the target turned to exit d that Step1's screen reaches along an axis within beta of -d."""
    return (mesh.face_normals@d > 0) & reach[axes@-d >= np.cos(np.radians(beta))-1e-12].any(0) & target


def placement(mesh, M, weight, s, policy, must=None):
    """The work-cube placement (corner on a grid) holding the most weight of faces wholly inside it, for the
    object turned by M, among those holding face `must` if given: (translation, mask), or None."""
    edge = s['surface_workspace']['edge_m']; lo = np.asarray(s['surface_workspace']['bounds_world_m'][0])
    step = policy['position_step']*edge; tight = 1e-7; value = mesh.area_faces*weight; ids = np.flatnonzero(value > 0)
    if not len(ids): return None
    X = R.test_points(mesh)[ids]@M.T; fmin, fmax = X.min(1), X.max(1)
    origin = fmin.min(0)-edge; n = np.ceil((fmax.max(0)-origin)/step).astype(int)+2
    # corner c = origin + i step holds face f wholly when fmax - edge <= c <= fmin (with a tight margin)
    first = np.maximum(np.ceil((fmax+tight-edge-origin)/step-1e-9).astype(int), 0)
    last = np.minimum(np.floor((fmin-tight-origin)/step+1e-9).astype(int), n-1)
    ok = (first <= last).all(1); f0, l1, v = first[ok], last[ok]+1, value[ids][ok]
    if not ok.any(): return None
    D = np.zeros(n+1)
    for corner in range(8):
        bits = [(corner >> i) & 1 for i in range(3)]
        index = tuple(np.where(bits[i], l1[:, i], f0[:, i]) for i in range(3))
        np.add.at(D, index, v*(-1)**sum(bits))
    score = D.cumsum(0).cumsum(1).cumsum(2)[:n[0], :n[1], :n[2]]
    if must is not None:                                      # only the corners that hold face `must` wholly
        k = np.flatnonzero(ids == must)
        if not len(k) or not ok[k[0]]: return None
        box = np.full(score.shape, -np.inf); a, b = first[k[0]], last[k[0]]+1
        box[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = 0.; score = score+box
    c = origin+np.array(np.unravel_index(int(np.argmax(score)), score.shape))*step
    mask = np.zeros(len(mesh.faces), bool); mask[ids[((fmin >= c+tight) & (fmax <= c+edge-tight)).all(1)]] = True
    return lo-c, mask


def certify(pool, mesh, axes, reach, cands, s, eps):
    """Exact certificates of candidates at their placements (Step1's needle, head and KUKA checks)."""
    ids = np.flatnonzero(np.any([c['mask'] for c in cands], axis=0))
    placements = [dict(rotation_world_object=c['rotation'].tolist(), translation_world_m=c['translation'].tolist()) for c in cands]
    directions = np.array([c['rotation'].T@np.asarray(s['working_direction_world']) for c in cands])
    tilt, witness, _, joints, _, _ = P.certify_directions(pool, mesh, ids, axes, reach, directions, s['max_tilt_deg'], s, eps, placements)
    for j, c in enumerate(cands):
        ok = np.isfinite(tilt[:, j]) & c['mask']
        c.update(certified=ok, axis=witness[:, j], joints=joints[:, j], counts=dict(screened=int(c['mask'].sum()), certified=int(ok.sum())))


def task_record(c, name):
    """Step3's task record of a certified candidate."""
    faces = np.flatnonzero(c['certified']); W = np.eye(4); W[:3, :3] = c['rotation']; W[:3, 3] = c['translation']
    return dict(id=name, origin=c['origin'], exit_object=c['d'], working_face_ids=faces.tolist(), certified_face_ids=faces.tolist(),
                axes=c['axis'][faces], joints=c['joints'][faces], pose=W, angle_deg=0., counts=c['counts'])


def support(args):
    """One task's support (for Step4): built and audited, turned about the needle while it fails."""
    vertices, faces, ov, of, record, s, physical, policy, obj, spins = args
    source = trimesh.Trimesh(vertices, faces, process=False); outer = trimesh.Trimesh(ov, of, process=False)
    exit_world = -np.asarray(s['working_direction_world'], float); trace = []
    for angle in spins:
        r = record if angle == 0 else turned(record, source, s, angle)
        if r is None: trace.append(dict(turn_deg=angle, status='rejected', reason='KUKA cannot hold every certified face')); continue
        model = ToolMembraneModel(source, outer, [r], s, physical, policy, obj)
        q = Layout(np.asarray(r['pose'])[None], np.eye(4)[None], exit_world[None])
        try:
            candidate = model.build(q, facewise=True); audit = model.audit(candidate)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as ex:
            trace.append(dict(turn_deg=angle, status='rejected', reason=str(ex)[:300])); continue
        task = audit['tasks'][0]; mesh = candidate['support']
        return dict(record=r, vertices=mesh.vertices, faces=mesh.faces, volume_cm3=candidate['volume_cm3'],
                    bbx_m3=float(mesh.bounding_box_oriented.volume),
                    audit=dict(exit_length_m=task.get('exit_length_m'), arm=task.get('arm')), trace=trace)
    return dict(record=record, failed=True, trace=trace)


def _pin(cpus):
    os.sched_setaffinity(0, {cpus.get()})


def pinned(workers):
    """A spawn pool, one CPU per worker: manifold spreads each Boolean over every CPU a process may use, so
    unpinned workers oversubscribe the machine."""
    context = multiprocessing.get_context('spawn'); cpus = sorted(os.sched_getaffinity(0)); queue = context.Queue()
    for k in range(workers): queue.put(cpus[k % len(cpus)])
    return ProcessPoolExecutor(workers, mp_context=context, initializer=_pin, initargs=(queue,))


def plan(run, config_path, workers=16, policy=POLICY, output_root=DEFAULT_OUTPUT):
    began = time.monotonic(); config_path = Path(config_path).resolve(); rng = np.random.default_rng(policy['seed'])
    obj, shape, mesh, big, target = big_task(run); s = station(obj, config_path, mesh); area = mesh.area_faces; total = float(area[target].sum())
    reach_policy = json.loads((ROOT/'whole_task_2/step1/reach_policy.json').read_text())
    cache = ROOT/'whole_task_2/data'/obj/'step1_reach'; cache.mkdir(parents=True, exist_ok=True)
    axes, reach, _ = P.screen(mesh, shape, s, reach_policy, cache)
    eps = reach_policy['needle_start_fraction_of_length']*s['needle_length_m']; a = np.asarray(s['working_direction_world'], float)
    beta = s['max_tilt_deg']; F = len(area)
    tasks = []; held = np.zeros(F, int); aimed = np.zeros(F, int); hard = target & ~reach.any(0); dropped = [0]

    def draw(M, origin, weight=None, must=None):
        d = M.T@-a; f = front(mesh, axes, reach, d, beta, target)
        put = placement(mesh, M, f if weight is None else f*weight, s, policy, must)
        if put is None: return None
        t, mask = put; return dict(origin=origin, d=d, rotation=M, translation=t, mask=mask & f)

    def aim(f):
        """A task aimed at face f: the exit along a needle axis that reaches it, a random turn about the needle, the
        cube holding f and the most of the little-held faces."""
        d = -axes[rng.choice(np.flatnonzero(reach[:, f]))]
        M = Rotation.from_rotvec(rng.uniform(0, 2*np.pi)*a).as_matrix()@onto(d, -a)
        c = draw(M, 'aimed', 1./(1.+held), f)
        if c is not None: c['aim'] = int(f)
        return c

    with ProcessPoolExecutor(workers, initializer=P._init_mesh, initargs=(str(shape), s)) as pool:
        def add(cands):
            if not cands: return
            certify(pool, mesh, axes, reach, cands, s, eps)
            for c in cands:
                if area[c['certified']].sum() < policy['min_area']*total: dropped[0] += 1
                else: tasks.append(c); held[c['certified']] += 1
        add([c for c in (draw(M, 'random') for M in Rotation.random(policy['start'], random_state=rng).as_matrix()) if c is not None])
        print('START', len(tasks), 'random tasks', round(time.monotonic()-began), 's', flush=True)
        for _ in range(policy['rounds']):
            if len(tasks) >= policy['pool']: break
            w = np.where(target & ~hard, area/(1.+held)**2, 0.)
            if w.sum() <= 0: break
            faces = rng.choice(F, size=policy['per_round'], p=w/w.sum())
            for f in faces: aimed[f] += 1
            add([c for c in (aim(f) for f in faces) if c is not None])
            hard |= target & (held == 0) & (aimed >= policy['give_up'])
    del tasks[policy['pool']:]
    # Name the pool by work area, save it.
    tasks.sort(key=lambda c: -area[c['certified']].sum())
    held = np.sum([c['certified'] for c in tasks], axis=0); hard &= held == 0
    folder = ROOT/'whole_task_2/data'/run/'step3_init'; folder.mkdir(parents=True, exist_ok=True); arrays = {}; rows = []
    for n, c in enumerate(tasks):
        rec = task_record(c, f'{LETTER}{n+1:02d}'); name = rec['id']; faces = np.asarray(rec['working_face_ids'])
        arrays.update({f'{name}_face_ids': faces, f'{name}_axes': rec['axes'], f'{name}_joints': rec['joints'], f'{name}_pose': rec['pose'],
                       f'{name}_exit_object': rec['exit_object']})
        rows.append(dict(id=name, origin=c['origin'], aim_face=c.get('aim'), exit_object=np.asarray(c['d']).tolist(),
                         face_count=len(faces), area_fraction_of_target=float(area[faces].sum()/total), certification=c['counts']))
    np.savez_compressed(folder/'initialization.npz', **arrays, target=target, held=held, hard=hard)
    times = {str(k): float(area[target & (held == k)].sum()/total) for k in range(6)}
    times['6+'] = float(area[target & (held >= 6)].sum()/total)
    report = dict(status='passed', run=run, object=obj, big_task=str(big.relative_to(ROOT)), station=str(config_path.relative_to(ROOT)),
        setup=s, policy=dict(policy), task_count=len(rows), tasks=rows, dropped_small=dropped[0],
        held_area_fraction_by_times=times, hard_face_ids=np.flatnonzero(hard).tolist(),
        hard_area_fraction_of_target=float(area[hard].sum()/total),
        hard_rule='no axis of the screen reaches the face, or it was aimed at give_up times and no task certifies it: it goes '
                  'to the nearest task, uncertified (Step4)',
        rule='a task: one placement of the object, its exit against the needle, the support staying where it is put; its '
             'work area the front the exit opens (n.d > 0) inside the work cube that the needle, head and KUKA reach; the '
             'pool: random tasks, then tasks aimed at the least held faces; its support built when Step4 draws it',
        input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [shape, big, config_path, Path(__file__)]},
        seconds=time.monotonic()-began)
    write_json(folder/'report.json', report)
    print('STEP3 POOL', run, len(rows), 'tasks; held by times', {k: round(v, 3) for k, v in times.items()},
          'hard %.4f,' % report['hard_area_fraction_of_target'], 'dropped', dropped[0], round(report['seconds']), 's', flush=True)
    from .render_big_init import render
    render(run, Path(output_root)/run/'step3')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run'); p.add_argument('--config', default=None); p.add_argument('--workers', type=int, default=16)
    a = p.parse_args(); run = a.run
    config = a.config or (f"robots_config/random/{run.split('+', 1)[1]}.json" if '+' in run else CONFIG)
    plan(run, ROOT/config, a.workers)


if __name__ == '__main__':
    main()
