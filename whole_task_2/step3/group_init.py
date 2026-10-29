"""Step3: shells round the object, each opened against the needle -- a start for Step4.

The support starts as a shell wrapping the object.  The object leaves the way the needle comes in: a shell's exit e
is the needle's direction reversed (object frame), so the opening it needs to come out is the opening the tool needs
to work -- the faces the needle sees along -e (within its tilt) are the shell's work area, and its push always goes
into the shell behind them.  Turning the support (and the object in it) about the needle keeps the needle, the exit
and the opening as they are and rests the support on another face: Rotational reuse, a base per turn.

  1. Exits.  The fewest exits whose openings show the whole area (set cover over sampled e; big_init.front: the faces
     turned to e that Step1's screen reaches along an axis within the tilt of -e).  Each is a seat (a shell).
  2. Tasks.  Candidates per exit: turns about the needle times work-cube placements (the best few per turn by worked
     area, overlapping little).  The fewest candidates covering `coverage` of the area (setcover.py: greedy, then an
     integer program within a time limit), each certified by Step1's needle, head and KUKA checks; a face its
     candidate fails leaves its mask and the cover is solved again; when it falls short, candidates aimed at the
     faces still uncovered join (at most `rounds` times).  Whether a task holds is Step4's (force, moment) pairs.
  3. Faces to tasks.  A Potts cut: each face to a task that certified it, its cost the needle tilt plus how far it is
     from facing the exit (1 - n.e), the shortest boundaries.  Faces no task certified go to the nearest task,
     uncertified.
  4. Windows.  For every shell but the last, candidate tasks for the later shells' faces with the support turned so
     that this shell's exit still rises (combine's exit_rise_deg): per up (window_ups spread over those) the turn and
     placement that works the most of them, certified, no faces of their own, the least turned first.  Before Step4 opens a
     later shell it lets them take its faces -- Rotational reuse through this shell's opening, the support turned
     about another axis, instead of a juxtaposed shell -- as far as everything still holds.

The station, screen and exact checks are seat_init's (big_init.station, Step1's).  Fewer ups, turns, placements and
a shorter integer program than subdivide (25 min there for whole_shape): the start only needs to be good.  Writes
data/<run>/step3_seats/ in seat_init's format (report.json, plan.npz, supports.npz) and its figures.

  python -m whole_task_2.step3.group_init whole_shape [--workers 16]
  python -m whole_task_2.step3.group_init 2task          (objects/B/big_tasks/2task.json: two far work areas)
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task_2.step1 import plan_reach as P, reach as R
from whole_task_2.step1.areas import Surface, nearest_area
from whole_task_2.step1.partition import Potts, BIG, UNIT
from .big_init import CONFIG, big_task, station, task_record
from .seat_init import supports
from .setcover import cover, covered

POLICY = dict(up_directions=240,          # sampled up directions u and seat axes e (Fibonacci sphere)
              turn_step_deg=15.,          # turns about the needle
              position_step=1/6,          # work-cube corner grid, share of the cube edge
              positions_per_turn=6,       # candidate placements per turn, by worked area
              overlap=.8,                 # ... sharing at most this share of the smaller one's area
              coverage=.99,               # certified share of the area the tasks must cover
              rounds=4,                   # times new candidates are aimed at the faces still uncovered
              cover_passes=8,             # covers re-solved after certification, at most (then the certified only), ...
              certify_batch=16,           # ... each pass certifying at least this many candidates (the chosen first)
              exit_rise_deg=10.,          # windows: a shell's exit rises at least this (combine's) in every pose, ...
              window_ups=12,              # ... the ups (spread over those) its windows turn the support to, one window each
              kappa=.1,                   # Potts: boundary length against tilt and height
              stand_weight=1.,            # Potts: how far a face is from facing the exit (1 - n.e) against tilt / max tilt
              group_seconds=10., task_seconds=10.)
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def fibonacci(n):
    k = np.arange(n)+.5; z = 1-2*k/n; r = np.sqrt(1-z*z); phi = k*np.pi*(3-5**.5)
    return np.c_[r*np.cos(phi), r*np.sin(phi), z]


def needle_pose(e, a, turn_deg):
    """World <- object: exit e against the needle a (the object leaves the way the needle comes in: e -> -a), turned
    turn_deg about the needle."""
    a = np.asarray(a, float)/np.linalg.norm(a)
    return (Rotation.from_rotvec(a*np.radians(turn_deg))*Rotation.align_vectors([-a], [np.asarray(e, float)])[0]).as_matrix()


def turned_poses(e, policy):
    """World <- object rotations keeping exit e rising at least exit_rise_deg: window_ups ups spread over those (the
    farthest from those taken first, from the one nearest e), each turned about the vertical: (up, turn, rotation)."""
    U = fibonacci(policy['up_directions']); rest = list(np.flatnonzero(U@e > np.sin(np.radians(policy['exit_rise_deg']))+1e-6))
    taken = [max(rest, key=lambda j: U[j]@e)]; rest.remove(taken[0])
    while rest and len(taken) < policy['window_ups']:
        j = max(rest, key=lambda j: min(1-U[j]@U[i] for i in taken)); taken.append(j); rest.remove(j)
    out = []
    for u in U[taken]:
        z = np.array([0., 0., 1.]); v = np.cross(u, z); n = np.linalg.norm(v)
        L = Rotation.from_rotvec(v/n*np.arctan2(n, u@z)) if n > 1e-12 else Rotation.identity()
        out += [(u, float(t), (Rotation.from_rotvec([0., 0., np.radians(t)])*L).as_matrix()) for t in np.arange(0., 360., policy['turn_step_deg'])]
    return out


def exit_groups(mesh, axes, reach, a, beta, target, policy, log=print):
    """The fewest exits whose openings show the target (module docstring, 1): per exit its poses (turns about the
    needle), the faces it shows and how squarely each faces it."""
    from .big_init import front
    area = mesh.area_faces; U = fibonacci(policy['up_directions']); capable = np.array([front(mesh, axes, reach, e, beta, target) for e in U])
    goal = target & capable.any(0); chosen = cover(capable, area, goal, 1., policy['group_seconds'])
    chosen = chosen[np.argsort(-(capable[chosen]@area), kind='stable')]; turns = np.arange(0., 360., policy['turn_step_deg'])
    groups = []
    for g, k in enumerate(chosen):
        poses = [(M.T@np.array([0., 0., 1.]), float(t), M) for t in turns for M in [needle_pose(U[k], a, t)]]
        groups.append(dict(id=LETTERS[g], k=int(k), u=U[k], exit=U[k], poses=poses, capable=capable[k],
                           stand=np.clip(mesh.face_normals@U[k], 0., 1.)))
    log('EXITS', len(groups), [np.round(g['u'], 3).tolist() for g in groups], 'show %.4f of the area' % (area[goal].sum()/area[target].sum()), flush=True)
    return groups, goal


def candidates(mesh, axes, reach, a, beta, group, capable, weight, s, policy):
    """Placements of one group: its poses (up, turn, rotation), the work cube's corner on a grid; per pose the best (by
    weighted worked area) few that overlap little.  Each: up, turn, rotation, translation and the screened mask."""
    edge = s['surface_workspace']['edge_m']; lo = np.asarray(s['surface_workspace']['bounds_world_m'][0])
    step = policy['position_step']*edge; tight = 1e-7; T = R.test_points(mesh); value = mesh.area_faces*weight
    out = []; cos = np.cos(np.radians(beta))
    for u, turn, M in group['poses']:
        d = M.T@a
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
            out.append(dict(group=group['id'], u=u, exit=group['exit'], turn_deg=float(turn), rotation=M, translation=lo-c, mask=mask))
    return out


def certify(pool, mesh, axes, reach, beta, s, eps, cands):
    """Exact certificates of candidates on their screened faces: Step1's needle, head and KUKA checks."""
    ids = np.flatnonzero(np.any([c['mask'] for c in cands], axis=0))
    placements = [dict(rotation_world_object=c['rotation'].tolist(), translation_world_m=c['translation'].tolist()) for c in cands]
    directions = np.array([c['rotation'].T@np.asarray(s['working_direction_world']) for c in cands])
    tilt, witness, _, joints, _, _ = P.certify_directions(pool, mesh, ids, axes, reach, directions, beta, s, eps, placements)
    for j, c in enumerate(cands):
        ok = np.isfinite(tilt[:, j]) & c['mask']
        c.update(certified=ok, tilt=np.where(ok, tilt[:, j], np.inf), axis=witness[:, j], joints=joints[:, j],
                 counts=dict(screened=int(c['mask'].sum()), certified=int(ok.sum())))


def group_tasks(pool, mesh, axes, reach, a, beta, s, eps, groups, goal, policy, log=print):
    """Certified tasks (candidate dicts) of groups covering `coverage` of goal (module docstring, 2); the rounds; every
    candidate made."""
    area = mesh.area_faces; pool_ = []
    def generate(weight):
        before = len(pool_)
        for g in groups: pool_.extend(candidates(mesh, axes, reach, a, beta, g, g['capable'] & goal, weight, s, policy))
        return len(pool_)-before
    generate(goal.astype(float)); rounds = 0; passes = 0; began = time.monotonic()
    while True:
        masks = np.array([c['certified'] if 'certified' in c else c['mask'] for c in pool_])
        chosen = cover(masks, area, goal, policy['coverage'], policy['task_seconds'])
        new = [pool_[i] for i in chosen if 'certified' not in pool_[i]]; share = covered(masks, chosen, area, goal)
        log('COVER', len(chosen), 'tasks of', len(pool_), 'candidates,', len(new), 'to certify, covers %.4f' % share,
            round(time.monotonic()-began), 's', flush=True)
        if new:
            # The chosen and, to fill a batch, the uncertified candidates that would cover the most of what is left.
            left = goal & ~masks[[i for i in chosen if 'certified' in pool_[i]]].any(0) if any('certified' in pool_[i] for i in chosen) else goal
            rest = sorted((c for c in pool_ if 'certified' not in c and not any(c is n for n in new)), key=lambda c: -area[c['mask'] & left].sum())
            certify(pool, mesh, axes, reach, beta, s, eps, new+rest[:max(0, policy['certify_batch']-len(new))]); passes += 1
            if passes < policy['cover_passes']: continue
            masks = np.array([c['certified'] if 'certified' in c else np.zeros_like(c['mask']) for c in pool_])   # the certified only
            chosen = cover(masks, area, goal, policy['coverage'], policy['task_seconds'])
            log('COVER', len(chosen), 'certified tasks, covers %.4f (passes spent)' % covered(masks, chosen, area, goal), flush=True); break
        if share >= policy['coverage']-1e-9 or rounds >= policy['rounds']: break
        rounds += 1; left = goal & ~masks[chosen].any(0)
        if not generate(left.astype(float)): break
        log('ROUND', rounds, 'aimed at %.4f uncovered' % (area[left].sum()/area[goal].sum()), flush=True)
    return [pool_[i] for i in chosen], rounds, pool_


def subdivide(mesh, shape, target, s, axes, reach, eps, workers, policy=POLICY, log=print):
    """The exits (seat groups), their certified tasks (candidate dicts) and the face assignment (module docstring)."""
    area = mesh.area_faces; a = np.asarray(s['working_direction_world'], float); beta = float(s['max_tilt_deg'])
    groups, goal = exit_groups(mesh, axes, reach, a, beta, target, policy, log=log)
    with ProcessPoolExecutor(workers, initializer=P._init_mesh, initargs=(str(shape), s)) as pool:
        tasks, rounds, _ = group_tasks(pool, mesh, axes, reach, a, beta, s, eps, groups, goal, policy, log)
        label, tasks = assigned(mesh, target, goal, beta, groups, tasks, policy)
        # Windows: every shell but the last, over the later shells' faces, the support turned so its exit still rises.
        seat_of = np.array([tasks[j]['group'] if j >= 0 else '' for j in label]); windows = []; cos = np.cos(np.radians(beta))
        for n, g in enumerate(groups[:-1]):
            later = target & np.isin(seat_of, [h['id'] for h in groups[n+1:]]); poses = turned_poses(g['u'], policy)
            able = later & np.any([reach[axes@(M.T@a) >= cos-1e-12].any(0) for _, _, M in poses], axis=0)
            if not able.any(): continue
            # per up the candidate that works the most of them: options for Step4, the least turned first
            best = {}
            for c in candidates(mesh, axes, reach, a, beta, dict(g, poses=poses), able, able.astype(float), s, policy):
                key = tuple(np.round(c['u'], 6))
                if key not in best or area[c['mask']].sum() > area[best[key]['mask']].sum(): best[key] = c
            found = sorted(best.values(), key=lambda c: -(c['u']@g['u'])); certify(pool, mesh, axes, reach, beta, s, eps, found)
            found = [c for c in found if (c['certified'] & able).any()]; windows += found
            log('WINDOWS', g['id'], 'for %.4f of the area: %d ups, %.4f reached' % (area[able].sum()/area[target].sum(), len(found),
                area[able & np.any([c['certified'] for c in found], axis=0)].sum()/area[target].sum() if found else 0.), flush=True)
    return groups, tasks, label, rounds, windows


def assigned(mesh, target, goal, beta, groups, tasks, policy):
    """Faces to tasks (module docstring, 3): (label, the tasks that kept faces, in order)."""
    area = mesh.area_faces
    ids = np.flatnonzero(goal & np.any([t['certified'] for t in tasks], axis=0)); by_group = {g['id']: g for g in groups}
    cost = np.column_stack([np.where(t['certified'], t['tilt']/np.radians(beta)+policy['stand_weight']*(1-by_group[t['group']]['stand']), np.inf)
                            for t in tasks])
    model = Potts(mesh, ids, beta, policy['kappa'])
    D = np.where(np.isfinite(cost[ids]), np.round(policy['kappa']*model.perimeter[:, None]*np.nan_to_num(cost[ids], posinf=0.)/UNIT), BIG).astype(np.int64)
    cut, _ = model.solve(D, D.argmin(1)); label = np.full(len(mesh.faces), -1); label[ids] = cut
    keep = [j for j in range(len(tasks)) if (label == j).any()]
    order = sorted(keep, key=lambda j: (tasks[j]['group'], -area[label == j].sum()))
    remap = np.full(len(tasks)+1, -1); remap[order] = np.arange(len(order)); label = remap[label]; tasks = [tasks[j] for j in order]
    nearest_area(Surface(mesh), np.flatnonzero(target), label)
    return label, tasks


def plan(run, config_path, workers=16, policy=POLICY, output_root=DEFAULT_OUTPUT):
    began = time.monotonic(); config_path = Path(config_path).resolve()
    obj, shape, mesh, big, target = big_task(run); s = station(obj, config_path, mesh); area = mesh.area_faces; total = float(area[target].sum())
    rp = json.loads((ROOT/'whole_task_2/step1/reach_policy.json').read_text())
    cache = ROOT/'whole_task_2/data'/obj/'step1_reach'; cache.mkdir(parents=True, exist_ok=True)
    axes, reach, _ = P.screen(mesh, shape, s, rp, cache); eps = rp['needle_start_fraction_of_length']*s['needle_length_m']
    groups, tasks, label, rounds, window_tasks = subdivide(mesh, shape, target, s, axes, reach, eps, workers, policy)
    # Seat_init's format: a seat per group, its exit the group's axis; per task its certified faces, those it absorbed.
    folder = ROOT/'whole_task_2/data'/run/'step3_seats'; folder.mkdir(parents=True, exist_ok=True); arrays = {}; rows = []
    absorbed = np.zeros(len(area), bool)
    for g in groups:
        mine = [(j, t) for j, t in enumerate(tasks) if t['group'] == g['id']]
        if not mine: continue
        ups = []; out = []
        for n, (j, t) in enumerate(mine):
            own = (label == j) & t['certified']; extra = (label == j) & ~t['certified']; absorbed |= extra
            pose = next((m for m, u in enumerate(ups) if u@t['u'] > 1-1e-9), None)
            if pose is None: ups.append(t['u']); pose = len(ups)-1
            rec = task_record(dict(t, certified=own, origin='group', d=g['u']), f"{g['id']}{n+1}"); name = rec['id']
            reached = np.flatnonzero(t['certified'])
            arrays.update({f'{name}_face_ids': np.flatnonzero(own), f'{name}_axes': rec['axes'], f'{name}_joints': rec['joints'],
                           f'{name}_pose': rec['pose'], f'{name}_exit_object': g['u'], f'{name}_absorbed_ids': np.flatnonzero(extra),
                           f'{name}_reach_ids': reached, f'{name}_reach_axes': t['axis'][reached], f'{name}_reach_joints': t['joints'][reached]})
            out.append(dict(id=name, pose=pose, up_object=t['u'].tolist(), turn_deg=t['turn_deg'], face_count=int((label == j).sum()),
                            area_fraction_of_target=float(area[label == j].sum()/total), certified_fraction_of_target=float(area[own].sum()/total),
                            absorbed_fraction_of_target=float(area[extra].sum()/total), reach_fraction_of_target=float(area[t['certified']].sum()/total),
                            certification=t['counts']))
        rows.append(dict(id=g['id'], exit_object=g['u'].tolist(), pose_count=len(ups), up_object=[u.tolist() for u in ups],
                         workable_fraction_of_target=float(area[g['capable']].sum()/total),
                         area_fraction_of_target=float(sum(t['area_fraction_of_target'] for t in out)),
                         absorbed_fraction_of_target=float(sum(t['absorbed_fraction_of_target'] for t in out)), tasks=out))
    # Windows: per shell its window tasks' pose and certified faces, no faces of their own; Step4 lets them take the
    # faces of a later shell before it opens that shell.
    windows = []
    for g in groups:
        for n, t in enumerate([t for t in window_tasks if t['group'] == g['id']]):
            name = f"{g['id']}w{n+1}"; f = np.flatnonzero(t['certified']); W = np.eye(4); W[:3, :3] = t['rotation']; W[:3, 3] = t['translation']
            arrays.update({f'{name}_reach_ids': f, f'{name}_reach_axes': t['axis'][f], f'{name}_reach_joints': t['joints'][f],
                           f'{name}_pose': W, f'{name}_exit_object': g['u']})
            windows.append(dict(id=name, seat=g['id'], exit_object=g['u'].tolist(), up_object=t['u'].tolist(), turn_deg=t['turn_deg'],
                                reach_fraction_of_target=float(area[f].sum()/total), certification=t['counts']))
    left = target & (label < 0)
    np.savez_compressed(folder/'plan.npz', **arrays, target=target, left=left, absorbed=absorbed & target)
    report = dict(status='passed', run=run, object=obj, big_task=str(big.relative_to(ROOT)), station=str(config_path.relative_to(ROOT)),
        setup=s, policy=dict(policy), parting=None, seat_count=len(rows), pose_count=sum(r['pose_count'] for r in rows),
        task_count=sum(len(r['tasks']) for r in rows), aimed_rounds=rounds, seats=rows, windows=windows,
        covered_fraction_of_target=float(area[target & ~left].sum()/total), left_fraction_of_target=float(area[left].sum()/total),
        certified_fraction_of_target=float(area[target & ~left & ~absorbed].sum()/total), absorbed_fraction_of_target=float(area[absorbed & target].sum()/total),
        left_face_ids=np.flatnonzero(left).tolist(),
        rule='shells opened against the needle: exits (the needle reversed) by set cover of the faces they show; tasks (turn about '
             'the needle x work-cube placement) by a partial set cover of '
             'certified candidates; faces to tasks by a Potts cut (tilt and standing); the rest to the nearest task, '
             'uncertified; Step4 optimizes it all',
        input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [shape, big, config_path, Path(__file__), Path(__file__).parent/'setcover.py']},
        seconds=time.monotonic()-began)
    write_json(folder/'report.json', report)
    supports(run, workers)
    print('STEP3 GROUPS', run, report['seat_count'], 'seats,', report['pose_count'], 'poses,', report['task_count'], 'tasks',
          [len(r['tasks']) for r in rows], '+ %d window tasks' % len(windows),
          '; covered %.3f (certified %.3f, absorbed %.3f)' % (report['covered_fraction_of_target'],
          report['certified_fraction_of_target'], report['absorbed_fraction_of_target']), round(time.monotonic()-began), 's', flush=True)
    from .render_seat_init import render
    render(run, Path(output_root)/run/'step3')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run'); p.add_argument('--config', default=None); p.add_argument('--workers', type=int, default=16)
    a = p.parse_args(); config = a.config or (f"robots_config/random/{a.run.split('+', 1)[1]}.json" if '+' in a.run else CONFIG)
    plan(a.run, ROOT/config, a.workers)


if __name__ == '__main__':
    main()
