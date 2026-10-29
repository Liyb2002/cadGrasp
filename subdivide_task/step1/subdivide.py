"""Step1 for a big task: split a large work area into tasks that are easy to hold.

Input: a big task (objects/<object>/big_tasks/<name>.json for the run <name> or <name>+<tag>, its face_ids; whole_shape is every face) and a
station (robots_config: the KUKA, its needle direction a, the tilt cone, the work cube).  A task is one
placement of the object, a rotation and a translation, and the faces the needle paints there, each by
Step1's exact checks (needle, head, KUKA inverse kinematics and links, table) with its tip inside the work
cube.  The KUKA rises and falls with the object, so a task's height is free like the rest of its translation.

What makes the tasks easy to hold (Step3 and Step4):
  1. Seat groups.  A group is one seat, an axis e of the object.  Its tasks keep an up direction u within
     host_deg of e and differ by u, a turn about the vertical and a translation.  So in the object their gravity
     stays within host_deg of -e, every needle comes from above (a, within the tilt, from the down direction),
     and one seat holding the object from below, a cup on its hold zone (the faces lowest along e,
     hold_fraction of its height), can carry all of them by Rotational reuse (step4/combine.py), the support
     tipped by up to host_deg between them.  A group paints only faces above its hold zone and a margin, so
     none of its tasks cuts the cup.  The fewest groups whose paintable faces (any hosted up, any turn, any
     translation) cover the area: a set cover over sampled e.  With host_deg 0 a group is one up direction:
     its gravity is one direction, so its seat is a bowl against it, and groups come out far apart, often
     upside down, which no seat can share.
  2. Tasks.  Candidates in every group: hosted ups times turns (turn_step_deg) times work-cube placements (the faces wholly
     inside the cube, its corner on a grid of position_step of its edge; per turn the best positions_per_turn
     by painted area, overlapping less than `overlap`).  The fewest candidates covering `coverage` of the area
     (a partial set cover, setcover.py), each certified exactly; a face its candidate fails leaves the
     candidate's mask and the cover is solved again.  When the certified cover falls short, new candidates
     aimed at the faces still uncovered join the pool (at most `rounds` times).  Tasks of every group compete
     for every face, so a crevice one group reaches badly goes to a task of another.
  3. Faces to tasks.  A Potts cut (partition.py): each face to a task that certified it, its cost the needle
     tilt plus how low the face stands in the task's group, with the shortest boundaries.  Target faces no
     chosen task certified go to the nearest task, uncertified (no tool keep-out; listed).

Writes data/<run>/step1_reach/plan.json, plan.npz and validation.json in Step1's schema (Step3 reads them
unchanged) and the figures under output/<run>/step1/.  plan.json's `subdivision` (also
output/<run>/step1/subdivision.json) records the groups, the task count and the coverage.

  python -m subdivide_task.step1.subdivide --object big_08 --config robots_config/random/07_kuka.json
"""
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robots_config.constraints import load_config
from .common import ROOT, DEFAULT_OUTPUT, load_shape, object_of, sha256, write_json
from . import plan_reach as P, reach as R, arm as A
from .areas import Surface, nearest_area
from .partition import Potts, BIG, UNIT
from .reach_video import work_point
from .setcover import cover, covered
from .workspace import points_inside_box

HERE = Path(__file__).parent
POLICY = dict(up_directions=240,          # sampled up directions u and seat axes e (Fibonacci sphere)
              host_deg=30.,               # a seat's tasks keep their up within this of its axis (0: one up per group)
              hold_fraction=.25,          # a group's hold zone: the lowest share of the object's height along e
              hold_margin_fraction=.04,   # painted faces stand at least this much (of the height) above it
              turn_step_deg=10.,          # turns about the vertical
              position_step=1/6,          # work-cube corner grid, share of the cube edge
              positions_per_turn=12,      # candidate placements per turn, by painted area
              overlap=.8,                 # ... sharing at most this share of the smaller one's area
              coverage=.99,               # certified share of the area the tasks must cover
              rounds=6,                   # times new candidates are aimed at the faces still uncovered
              kappa=.1,                   # Potts: boundary length against tilt and height
              height_weight=1.,           # Potts: a face's height cost (1 - height) against its tilt / max tilt
              group_seconds=30., task_seconds=60.)
SOURCES = ['subdivide.py', 'setcover.py', 'reach.py', 'arm.py', 'plan_reach.py', 'partition.py', 'areas.py',
           'workspace.py', 'reach_video.py', 'reach_render.py', 'reach_policy.json', 'needle_workspace.json', 'common.py']
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def folder(run):
    return ROOT/'subdivide_task/data'/run/'step1_reach'


def big_task(run):
    """The object, its mesh and the big task's target faces."""
    obj = object_of(run); shape = ROOT/'objects'/obj/'mesh.stl'; mesh = load_shape(shape)
    path = ROOT/'objects'/obj/'big_tasks'/f"{run.split('+', 1)[0]}.json"; area = json.loads(path.read_text())
    if area['mesh_sha256'] != sha256(shape): raise ValueError(run+': the big task belongs to another mesh')
    target = np.zeros(len(mesh.faces), bool); target[area['face_ids']] = True
    return obj, shape, mesh, path, target


def station(run, config_path, mesh, path):
    """Step1's setup for the station, with the work cube centred on the work point and the table below it."""
    _, _, _, s = P.setup(object_of(run), config_path)
    config = load_config(config_path); edge = float(config['task_constraint']['surface_workspace']['edge_m'])
    point, forward = work_point(s, edge/2); point = np.asarray(point, float)
    lo, hi = point-edge/2, point+edge/2
    table = float(lo[2]-mesh.extents.max()); half = float(config['feasibility'].get('table_half_size_m', .25))
    s.update(target='big_task:'+str(path.relative_to(ROOT)), work_point_m=point.tolist(), work_point_forward_of_window_m=forward,
             work_point_rule='first point forward of the window centre, along base->window, where the KUKA reaches every tip '
                             'within the work cube at any tilt in the cone',
             surface_workspace=dict(frame='world', shape='cube', applies_to='surface_process_point', edge_m=edge,
                                    workspace_size_world_m=[edge]*3, workspace_center_world_m=point.tolist(),
                                    bounds_world_m=[lo.tolist(), hi.tolist()], rule='the configured cube centred on the work point',
                                    robot=s['robot']),
             table_height_m=table, table_bounds_world_m=[[point[0]-half, point[1]-half, table-.02], [point[0]+half, point[1]+half, table]],
             pose_rule='seat groups: the object turned up an up u within host_deg of its seat axis; its tasks turn it about the '
                       'vertical and translate it (the KUKA rises and falls with it); the table one object size below the cube')
    return s


def fibonacci(n):
    k = np.arange(n)+.5; z = 1-2*k/n; r = np.sqrt(1-z*z); phi = k*np.pi*(3-5**.5)
    return np.c_[r*np.cos(phi), r*np.sin(phi), z]


def lift(u):
    """The least rotation turning the object's u up (world +z)."""
    z = np.array([0., 0., 1.]); u = np.asarray(u, float)/np.linalg.norm(u); c = float(u@z); v = np.cross(u, z); n = np.linalg.norm(v)
    if n < 1e-12: return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    return Rotation.from_rotvec(v/n*np.arctan2(n, c)).as_matrix()


def rotation(u, turn_deg):
    """World <- object: u up, then turned about the vertical."""
    return Rotation.from_rotvec([0., 0., np.radians(turn_deg)]).as_matrix()@lift(u)


def heights(mesh, U):
    """height[k, f]: face f's lowest test point along u_k, 0 at the object's bottom and 1 at its top."""
    T = R.test_points(mesh); out = np.empty((len(U), len(mesh.faces)))
    for k, u in enumerate(U):
        z = mesh.vertices@u; out[k] = ((T@u).min(1)-z.min())/(z.max()-z.min())
    return out


def hosted(U, policy):
    """host[k, j]: a seat with axis u_k takes tasks with u_j up (within host_deg; only u_k itself at 0)."""
    return U@U.T >= np.cos(np.radians(policy['host_deg']))-1e-9


def paintable(axes, reach, a, beta, U, height, target, policy):
    """capable[k, f]: a seat with axis u_k paints face f with some hosted up at some turn (screened reach), and f
    stands above the seat's hold zone."""
    down = float(np.degrees(np.arccos(np.clip(-a[2], -1, 1))))         # the needle's angle from straight down
    floor = policy['hold_fraction']+policy['hold_margin_fraction']; up = np.zeros((len(U), reach.shape[1]), bool)
    for k, u in enumerate(U):                                          # with u_k up, some turn paints
        angle = np.degrees(np.arccos(np.clip(axes@-u, -1, 1)))
        up[k] = reach[(angle >= down-beta-1e-9) & (angle <= down+beta+1e-9)].any(0)
    return (hosted(U, policy).astype(np.float32)@up.astype(np.float32) > 0) & (height >= floor) & target


def candidates(mesh, axes, reach, a, beta, group, capable, weight, s, policy):
    """Placements of one group: every hosted up and turn, the work cube's corner on a grid; per up and turn the
    best (by weighted painted area) few that overlap little.  Each: up, turn, rotation, translation and the
    screened mask."""
    edge = s['surface_workspace']['edge_m']; lo = np.asarray(s['surface_workspace']['bounds_world_m'][0])
    step = policy['position_step']*edge; tight = 1e-7; T = R.test_points(mesh); value = mesh.area_faces*weight
    out = []; cos = np.cos(np.radians(beta))
    for u in group['ups']:
        for turn in np.arange(0., 360., policy['turn_step_deg']):
            M = rotation(u, turn); d = M.T@a
            ids = np.flatnonzero(reach[axes@d >= cos-1e-12].any(0) & capable & (value > 0))
            if not len(ids): continue
            X = T[ids]@M.T; fmin, fmax = X.min(1), X.max(1)
            origin = fmin.min(0)-edge; n = np.ceil((fmax.max(0)-origin)/step).astype(int)+2
            # corner c = origin + i step holds face f wholly when fmax - edge <= c <= fmin (with a tight margin)
            first = np.maximum(np.ceil((fmax+tight-edge-origin)/step-1e-9).astype(int), 0)
            last = np.minimum(np.floor((fmin-tight-origin)/step+1e-9).astype(int), n-1)
            ok = (first <= last).all(1); f0, l1, v = first[ok], last[ok]+1, value[ids][ok]
            D = np.zeros(n+1)
            for corner in range(8):
                bits = [(corner >> i) & 1 for i in range(3)]
                index = tuple(np.where(bits[i], l1[:, i], f0[:, i]) for i in range(3))
                np.add.at(D, index, v*(-1)**sum(bits))
            score = D.cumsum(0).cumsum(1).cumsum(2)[:n[0], :n[1], :n[2]]
            taken = []
            for flat in np.argsort(-score, axis=None):
                if score.flat[flat] <= 1e-15 or len(taken) == policy['positions_per_turn']: break
                c = origin+np.array(np.unravel_index(flat, score.shape))*step
                inside = ids[((fmin >= c+tight) & (fmax <= c+edge-tight)).all(1)]
                if not len(inside): continue
                worth = value[inside].sum()
                if any(value[np.intersect1d(inside, other, assume_unique=True)].sum() > policy['overlap']*min(worth, value[other].sum())
                       for other in taken): continue
                taken.append(inside)
                mask = np.zeros(len(mesh.faces), bool); mask[inside] = True
                out.append(dict(group=group['id'], u=u, turn_deg=float(turn), rotation=M, translation=lo-c, mask=mask))
    return out


def certify(pool, mesh, axes, reach, beta, s, eps, cands):
    """Exact certificates of candidates (Step1's needle, head and KUKA checks) on their screened faces."""
    ids = np.flatnonzero(np.any([c['mask'] for c in cands], axis=0))
    placements = [dict(rotation_world_object=c['rotation'].tolist(), translation_world_m=c['translation'].tolist()) for c in cands]
    directions = np.array([c['rotation'].T@np.asarray(s['working_direction_world']) for c in cands])
    tilt, witness, gaps, joints, code, record = P.certify_directions(pool, mesh, ids, axes, reach, directions, beta, s, eps, placements)
    for j, c in enumerate(cands):
        ok = np.isfinite(tilt[:, j]) & c['mask']
        c.update(certified=ok, tilt=np.where(ok, tilt[:, j], np.inf), witness=witness[:, j], gaps=gaps[:, j],
                 joints=joints[:, j], code=code[:, j], direction=directions[j])
    return record


def subdivide(mesh, shape, target, s, axes, reach, workers, policy=POLICY, log=print):
    """Groups, certified tasks (candidate dicts) and the face assignment."""
    area = mesh.area_faces; a = np.asarray(s['working_direction_world'], float); beta = float(s['max_tilt_deg'])
    eps = json.loads((HERE/'reach_policy.json').read_text())['needle_start_fraction_of_length']*s['needle_length_m']
    U = fibonacci(policy['up_directions']); height = heights(mesh, U)
    capable = paintable(axes, reach, a, beta, U, height, target, policy)
    paintable_faces = target & capable.any(0)
    chosen = cover(capable, area, paintable_faces, 1., policy['group_seconds'])
    chosen = chosen[np.argsort(-(capable[chosen]@area), kind='stable')]
    host = hosted(U, policy); others = lambda k: host[k] & (np.arange(len(U)) != k)
    groups = [dict(id=LETTERS[g], k=int(k), u=U[k], ups=np.vstack([U[k], U[others(k)]]), capable=capable[k], height=height[k])
              for g, k in enumerate(chosen)]
    log('GROUPS', len(groups), [np.round(g['u'], 3).tolist() for g in groups], 'hosting', [len(g['ups']) for g in groups], 'ups,',
        'paintable %.4f of the area' % (area[paintable_faces].sum()/area[target].sum()))
    pool_ = []
    def generate(weight):
        before = len(pool_)
        for g in groups: pool_.extend(candidates(mesh, axes, reach, a, beta, g, g['capable'], weight, s, policy))
        return len(pool_)-before
    generate(paintable_faces.astype(float))
    stats = dict(candidates_screened=0, candidates_certified=0, rounds=0, certificate_records=[])
    with ProcessPoolExecutor(workers, initializer=P._init_mesh, initargs=(str(shape), s)) as pool:
        while True:
            masks = np.array([c['certified'] if 'certified' in c else c['mask'] for c in pool_])
            chosen = cover(masks, area, paintable_faces, policy['coverage'], policy['task_seconds'])
            new = [pool_[i] for i in chosen if 'certified' not in pool_[i]]
            share = covered(masks, chosen, area, paintable_faces)
            log('COVER', len(chosen), 'tasks of', len(pool_), 'candidates,', len(new), 'to certify, covers %.4f' % share, flush=True)
            if new:
                record = certify(pool, mesh, axes, reach, beta, s, eps, new)
                stats['candidates_certified'] += len(new); stats['certificate_records'].append(record)
                continue
            if share >= policy['coverage']-1e-9 or stats['rounds'] >= policy['rounds']: break
            stats['rounds'] += 1; left = paintable_faces & ~masks[chosen].any(0)
            if not generate(left.astype(float)): break
            log('ROUND', stats['rounds'], 'aimed at %.4f uncovered' % (area[left].sum()/area[paintable_faces].sum()), flush=True)
    stats['candidates_screened'] = len(pool_)
    tasks = [pool_[i] for i in chosen]
    # Faces to tasks: tilt and height, shortest boundaries.
    ids = np.flatnonzero(paintable_faces & np.any([t['certified'] for t in tasks], axis=0))
    by_group = {g['id']: g for g in groups}
    cost = np.column_stack([np.where(t['certified'], t['tilt']/np.radians(beta)+policy['height_weight']*(1-by_group[t['group']]['height']), np.inf)
                            for t in tasks])
    model = Potts(mesh, ids, beta, policy['kappa'])
    D = np.where(np.isfinite(cost[ids]), np.round(policy['kappa']*model.perimeter[:, None]*np.nan_to_num(cost[ids], posinf=0.)/UNIT), BIG).astype(np.int64)
    cut, _ = model.solve(D, D.argmin(1)); label = np.full(len(mesh.faces), -1); label[ids] = cut
    keep = [j for j in range(len(tasks)) if (label == j).any()]
    order = sorted(keep, key=lambda j: (tasks[j]['group'], -area[label == j].sum()))
    remap = np.full(len(tasks)+1, -1); remap[order] = np.arange(len(order)); label = remap[label]
    tasks = [tasks[j] for j in order]
    counter = {}
    for t in tasks:
        counter[t['group']] = counter.get(t['group'], 0)+1; t['id'] = f"{t['group']}{counter[t['group']]}"
    certified_label = label.copy()
    nearest_area(Surface(mesh), np.flatnonzero(target), label)
    return dict(groups=groups, tasks=tasks, label=label, certified_label=certified_label, paintable=paintable_faces,
                stats=stats, U=U)


def plan(run, config_path, output_root=DEFAULT_OUTPUT, workers=20, render=True):
    began = time.monotonic(); config_path = Path(config_path).resolve()
    reach_policy = json.loads((HERE/'reach_policy.json').read_text())
    obj, shape, mesh, big, target = big_task(run); s = station(run, config_path, mesh, big)
    work = folder(run); work.mkdir(parents=True, exist_ok=True)
    cache = ROOT/'subdivide_task/data'/obj/'step1_reach/screen.npz'
    if cache.exists() and not (work/'screen.npz').exists(): shutil.copy2(cache, work/'screen.npz')
    axes, reach, pixel = P.screen(mesh, shape, s, reach_policy, work)
    out = subdivide(mesh, shape, target, s, axes, reach, workers)
    groups, tasks, label = out['groups'], out['tasks'], out['label']
    F, K = len(mesh.faces), len(tasks); area = mesh.area_faces; total = float(area[target].sum())
    reach_ = np.column_stack([t['certified'] for t in tasks]); witness = np.stack([t['witness'] for t in tasks], 1)
    joints = np.stack([t['joints'] for t in tasks], 1); gaps = np.stack([t['gaps'] for t in tasks], 1)
    code = np.stack([t['code'] for t in tasks], 1); tilt = np.column_stack([t['tilt'] for t in tasks])
    assigned = label >= 0; hit = assigned & reach_[np.arange(F), np.maximum(label, 0)]; nearest = assigned & ~hit
    face_axis = np.zeros((F, 3)); face_tilt = np.zeros(F); face_gaps = np.zeros((F, 2))
    face_axis[hit] = witness[hit, label[hit]]; face_tilt[hit] = np.degrees(tilt[hit, label[hit]]); face_gaps[hit] = gaps[hit, label[hit]]
    directions = np.array([t['direction'] for t in tasks]); face_axis[nearest] = directions[label[nearest]]
    surface = Surface(mesh); rows = []
    for j, t in enumerate(tasks):
        f = np.flatnonzero(label == j); can = reach_[:, j]
        rows.append(dict(id=t['id'], group=t['group'], up_object=t['u'].tolist(), turn_deg=t['turn_deg'],
            axis_object=directions[j].tolist(), view_object=t['u'].tolist(),
            placement=dict(rotation_world_object=t['rotation'].tolist(), translation_world_m=t['translation'].tolist()),
            reach_area_fraction_of_target=float(area[can & target].sum()/total), reach_face_count=int(can.sum()),
            task_area_fraction_of_target=float(area[can & target].sum()/total), enclosed_face_count=0, removed_face_count=0,
            removed_area_fraction_of_target=0., face_ids=f.tolist(), face_count=len(f), area_m2=float(area[f].sum()),
            area_fraction_of_target=float(area[f].sum()/total), pieces=len(surface.pieces(label == j)),
            mean_tilt_deg=float(np.average(face_tilt[f], weights=area[f])) if len(f) else 0.,
            max_tilt_deg=float(face_tilt[f].max()) if len(f) else 0., uncertified_face_count=int((nearest & (label == j)).sum())))
    uncovered = np.flatnonzero(target & ~out['paintable'])
    certified_share = float(area[hit & target].sum()/total); assigned_share = float(area[assigned & target].sum()/total)
    group_rows = [dict(id=g['id'], up_object=g['u'].tolist(), task_ids=[r['id'] for r in rows if r['group'] == g['id']],
                       task_count=sum(r['group'] == g['id'] for r in rows),
                       area_fraction_of_target=float(sum(r['area_fraction_of_target'] for r in rows if r['group'] == g['id'])),
                       paintable_area_fraction_of_target=float(area[g['capable']].sum()/total)) for g in groups]
    subdivision = dict(task_count=K, group_count=len(groups), groups=group_rows,
        task_ids=[r['id'] for r in rows], task_area_fractions=[r['area_fraction_of_target'] for r in rows],
        target_face_count=int(target.sum()), target_area_fraction_of_object=float(total/area.sum()),
        paintable_fraction_of_target=float(area[out['paintable']].sum()/total),
        certified_fraction_of_target=certified_share, assigned_fraction_of_target=assigned_share,
        uncertified_fraction_of_target=float(area[nearest].sum()/total), uncertified_face_count=int(nearest.sum()),
        policy=POLICY, candidates_screened=out['stats']['candidates_screened'],
        candidates_certified=out['stats']['candidates_certified'], aimed_rounds=out['stats']['rounds'],
        rule='seat groups (up directions, hold zone below) by set cover; tasks (turn x work-cube placement) by a partial set '
             'cover of certified candidates; faces to tasks by a Potts cut; uncovered target faces to the nearest task, uncertified')
    report = dict(status='passed', model='subdivided_big_task_seat_groups', object=run, source_object=object_of(run), setup=s,
        allow_partial_analysis=False, task_count_mode='subdivided', task_count_optimized=True, strict_surface_intersection=False,
        required_coverage=POLICY['coverage'], possible_coverage_screened=subdivision['paintable_fraction_of_target'],
        coverage_goal=POLICY['coverage'], requirement_met=bool(certified_share >= POLICY['coverage']*subdivision['paintable_fraction_of_target']-1e-9),
        covered_fraction_of_target=assigned_share, certified_fraction_before_completion=certified_share,
        coverage_definition='faces assigned to a task; certified: the task paints the face by the exact checks; the rest of the '
                            'target goes to the nearest task uncertified',
        direction_count=K, area_count=K, areas=rows, task_count=K, tasks=[dict(r, direction_index=j, direction_id=r['id']) for j, r in enumerate(rows)],
        active_task_count=K, empty_task_ids=[], subdivision=subdivision,
        task_definition='subdivided big task: one placement per task, its exclusive faces (face_task == index)',
        task_surface_rule='face_task == task index; reach == the task certifies the face (needle, head, KUKA, table, work cube)',
        target_face_count=int(target.sum()), target_area_m2=total,
        unreachable_face_ids=uncovered.tolist(), unreachable_area_fraction_of_target=float(area[uncovered].sum()/total),
        unassigned_face_ids=np.flatnonzero(target & ~assigned).tolist(), unassigned_area_fraction_of_target=1-assigned_share,
        uncertified_face_ids=np.flatnonzero(nearest).tolist(), uncertified='target faces no chosen task certified go to the nearest task',
        reach_definition=dict(needle='line from the tip back to infinity misses the object beyond needle_start_m',
            needle_body='needle radius clears the object beyond the tip contact r(1+tan(incidence)), incidence capped at 80 deg',
            head='no object point within head_radius_m of the needle line at needle_length_m and beyond',
            facing='face normal faces the tool', face_test_points='three vertices and the centroid',
            arm_motion='fixed base (robots_config); needle within max_tilt_deg of the slanted nominal direction; the arm rises and falls with the object',
            arm='KUKA inverse kinematics at every test point along the certified axis (joint limits, above the floor), convex hulls of all '
                'link collision meshes clear of the object and tabletop; joints kept for replay',
            surface_position='surface process point (needle tip) inside the work cube centred on the work point',
            placement=s['pose_rule'], table='needle body and head above the table; arm collision hulls clear of the finite tabletop',
            needle_start_m=reach_policy['needle_start_fraction_of_length']*s['needle_length_m']),
        grids=dict(tool_axes=len(axes), tool_axis_spacing_deg=reach_policy['tool_axis_spacing_deg'], screen_pixel_m=pixel),
        certification=dict(exact=True, method='clipped-triangle distance to the needle line (reach.clearance)',
            faces_assigned=int(hit.sum()), records=out['stats']['certificate_records'],
            min_needle_body_gap_minus_radius_m=float(face_gaps[hit, 0].min()-s['needle_radius_m']) if hit.any() else None,
            min_head_gap_minus_radius_m=float(face_gaps[hit, 1].min()-s['head_radius_m']) if hit.any() else None),
        seconds=dict(total=time.monotonic()-began),
        input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [shape, big, *[HERE/n for n in SOURCES], config_path]})
    surface_in_box = np.column_stack([points_inside_box(R.test_points(mesh)@t['rotation'].T+t['translation'],
                                                        s['surface_workspace']['bounds_world_m']).all(axis=1) for t in tasks])
    np.savez_compressed(work/'plan.npz', face_area=label, face_task=label, face_certified=hit, face_axis=face_axis, face_tilt_deg=face_tilt,
        gaps=face_gaps, surface_inside_box=surface_in_box, area_axes=directions, reach=reach_, task_reach=reach_,
        task_enclosed=np.zeros_like(reach_), task_removed=np.zeros_like(reach_), reach_axis=witness, target=target, arm_code=code,
        joints=joints, group_up_object=np.array([g['u'] for g in groups]))
    report['arrays_sha256'] = sha256(work/'plan.npz'); write_json(work/'plan.json', report)
    print('STEP1 SUBDIVIDED', run, 'groups', len(groups), 'tasks', K, 'certified %.4f assigned %.4f' % (certified_share, assigned_share),
          round(time.monotonic()-began), 's', flush=True)
    validate(run, workers)
    if render:
        from .reach_render import publish
        report['figures'] = publish(mesh, report, Path(output_root)/run/'step1')
        report['figures']['groups'] = str(groups_figure(mesh, report, Path(output_root)/run/'step1'/'step1.2'/'groups.png').relative_to(ROOT))
        write_json(work/'plan.json', report); validate(run, workers, replay=False)
        write_json(Path(output_root)/run/'step1'/'subdivision.json', dict(object=run, **subdivision))
    return report


def groups_figure(mesh, report, path):
    """Six views of the target, every face in its task's group colour (uncovered faces red)."""
    from .reach_video import Scene
    from .reach_render import picture, tiles
    from subdivide_task.step5.video import raster as r
    with np.load(folder(report['object'])/'plan.npz') as z: label, target = z['face_task'], z['target']
    palette = np.array([[.93, .42, .62], [.25, .55, .85], [.38, .74, .48], [.95, .66, .22], [.62, .48, .82], [.40, .78, .85]])
    group_of = np.array([LETTERS.index(t['group']) for t in report['tasks']])
    colors = np.tile(r.GRAY, (len(mesh.faces), 1)); colors[label >= 0] = palette[group_of[label[label >= 0]] % len(palette)]
    colors[target & (label < 0)] = (.94, .12, .16)
    rotation_ = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]]); W = np.eye(4); W[:3, :3] = rotation_; W[:3, 3] = -rotation_@mesh.bounds.mean(0)
    views = np.array([[1, .15, .2], [.15, 1, .1], [.15, .15, 1], [-1, .15, -.2], [.15, -1, -.1], [-.15, .15, -1]], float)
    scene = Scene(mesh, report['setup'], 560, 560); images = []
    try:
        for v in views/np.linalg.norm(views, axis=1)[:, None]:
            scene.view = rotation_@v; scene.fit(scene.points(W, []), 1.1); images.append(picture(scene, W, colors, [], fit=False))
    finally:
        scene.close()
    tiles(images, 3).save(path); return path


def _replay(job):
    W, pairs = job; return sum(P._arm.replay(face, axis, W, Q) for face, axis, Q in pairs)


def validate(run, workers=20, replay=True):
    """Disk replay: every (face, task) certificate (needle, head, KUKA at the stored joints), the exclusive
    assignment, the cube, the groups (u up, faces above the hold zone) and the recorded counts."""
    work = folder(run); report = json.loads((work/'plan.json').read_text())
    if sha256(work/'plan.npz') != report['arrays_sha256']: raise ValueError('plan.npz changed')
    with np.load(work/'plan.npz') as z: a = {k: z[k] for k in z.files}
    s = report['setup']; mesh = load_shape(ROOT/'objects'/object_of(run)/'mesh.stl'); area = mesh.area_faces
    eps = report['reach_definition']['needle_start_m']; tasks = report['tasks']; K = len(tasks)
    checks = {}
    pairs = np.argwhere(a['reach'])
    if replay:
        jobs = [(int(f), [(int(j), 0, a['reach_axis'][f, j])], s['needle_length_m'], s['head_radius_m'], s['needle_radius_m'], eps,
                 s['max_incidence_deg']) for f, j in pairs]
        arm_jobs = []
        for j, t in enumerate(tasks):
            W = np.eye(4); W[:3, :3] = t['placement']['rotation_world_object']; W[:3, 3] = t['placement']['translation_world_m']
            f = np.flatnonzero(a['reach'][:, j])
            arm_jobs += [(W, [(int(g), a['reach_axis'][g, j], a['joints'][g, j]) for g in f[i:i+60]]) for i in range(0, len(f), 60)]
        with ProcessPoolExecutor(workers, initializer=P._init_mesh, initargs=(str(ROOT/'objects'/object_of(run)/'mesh.stl'), s)) as pool:
            passed = sum(r[1] is not None for r in pool.map(P._certify, jobs, chunksize=32))
            held = sum(pool.map(_replay, arm_jobs))
        checks.update(exact_reach=passed == len(jobs), arm_replay=held == len(jobs))
    tilt = np.degrees(np.arccos(np.clip(np.einsum('fjk,jk->fj', a['reach_axis'], a['area_axes'])[a['reach']], -1, 1)))
    label = a['face_task']; target = a['target']; hit = a['face_certified']
    bounds = np.asarray(s['surface_workspace']['bounds_world_m']); T = R.test_points(mesh)
    up = a['group_up_object']; letters = [g['id'] for g in report['subdivision']['groups']]
    U = np.array([g['up_object'] for g in report['subdivision']['groups']]); hold = report['subdivision']['policy']['hold_fraction']
    margin = report['subdivision']['policy']['hold_margin_fraction']; inside = True; upright = True; above = True; listed = True
    host = np.cos(np.radians(report['subdivision']['policy'].get('host_deg', 0.)))-1e-9
    for j, t in enumerate(tasks):
        M = np.asarray(t['placement']['rotation_world_object']); shift = np.asarray(t['placement']['translation_world_m'])
        f = np.flatnonzero(a['reach'][:, j]); inside &= bool(points_inside_box(T[f]@M.T+shift, bounds).all())
        e = U[letters.index(t['group'])]; u = np.asarray(t['up_object'])
        upright &= bool(np.allclose(M@u, [0, 0, 1], atol=1e-9) and u@e >= host)
        z = mesh.vertices@e; h = ((T[f]@e).min(1)-z.min())/(z.max()-z.min()); above &= bool((h >= hold+margin-1e-9).all())
        listed &= np.flatnonzero(label == j).tolist() == t['face_ids']
        listed &= bool((mesh.vertices@M.T+shift)[:, 2].min() >= s['table_height_m']-1e-9)
    total = float(area[target].sum())
    checks.update(tilt_within_limit=bool(tilt.max() <= s['max_tilt_deg']+1e-9) if len(tilt) else True,
        certified_means_own_task_paints=bool(np.array_equal(hit, (label >= 0) & a['reach'][np.arange(len(label)), np.maximum(label, 0)])),
        every_certified_surface_point_inside_work_cube=inside, tasks_up_within_their_seat=upright,
        painted_faces_above_the_hold_zone=above, task_faces_listed_and_above_table=listed,
        group_ups_match=bool(np.allclose(up, U)), assignment_only_on_target=bool(np.all(target[label >= 0])),
        every_target_face_assigned=bool(np.all(label[target] >= 0)),
        task_count_recorded=bool(report['task_count'] == K == report['subdivision']['task_count'] == len(set(t['id'] for t in tasks))),
        coverage_recorded=bool(abs(float(area[hit & target].sum()/total)-report['subdivision']['certified_fraction_of_target']) < 1e-12),
        uncertified_listed=bool(np.flatnonzero((label >= 0) & ~hit).tolist() == report['uncertified_face_ids']))
    result = dict(status='passed' if all(checks.values()) else 'failed', checks=checks, pairs=len(pairs),
                  task_count=K, group_count=report['subdivision']['group_count'], max_tilt_deg=float(tilt.max()) if len(tilt) else 0.,
                  certified_fraction_of_target=report['subdivision']['certified_fraction_of_target'], plan_sha256=sha256(work/'plan.json'))
    if not replay:
        old = json.loads((work/'validation.json').read_text()); result['checks'] = dict(old['checks'], **checks)
        result['status'] = 'passed' if old['status'] == 'passed' and all(checks.values()) else 'failed'
    write_json(work/'validation.json', result)
    if result['status'] != 'passed': raise ValueError('Step1 subdivision validation failed: '+json.dumps(checks))
    print('STEP1 VALIDATION PASSED', len(pairs), 'pairs' if replay else '(plan rewritten, replay kept)', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', required=True); p.add_argument('--config', default='robots_config/random/07_kuka.json')
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT); p.add_argument('--workers', type=int, default=20)
    p.add_argument('--no-render', action='store_true'); p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    if a.validate_only: validate(a.object, a.workers)
    else: plan(a.object, ROOT/a.config, a.output_root, a.workers, not a.no_render)


if __name__ == '__main__':
    main()
