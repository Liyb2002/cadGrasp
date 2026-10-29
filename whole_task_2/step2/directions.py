"""Step2 by work direction: areas the needle works from one direction each -- no seats, no bases, no exits.

User, 2026-10-29: "area 改成按工作方向 a_k 分组，不再按'底座 + z 转角'分组 ... Step4 从一个包住整个物体的支撑开始，R 优先，
失败了才 J".  Step2 only says from where the needle works each part of the area; how the support turns between them
(about each work direction: combine's Pose rotation), where the object leaves and which areas share a seat are
combine's (Rotational reuse first, Juxtaposition only when that fails).

  1. Work directions a (Fibonacci, object frame): a reaches the faces whose screened needle axes lie within the
     station's tilt of a.  Few directions that cover all but policy['assumed'] (0.1%) of the area: one more at a time,
     the one covering the most of what is left, every chosen one then swapped for a better while the cover grows
     (fewest).  Covering every face let a few scattered faces call for a direction each
     (B+big_01: 8 directions by a greedy cover of every face, the last four for 1.7% of the area; 4 cover 99.4%).
     A target above a share of the object's surface gets at least so many (policy['least']: >60% 4, >75% 5, >90% 6),
     for the figure's sake and a problem worth combining.  The faces they cover go to them in about equal, connected blocks (seats.blocks: a block grows along
     the surface from the face that faces against its direction most squarely); the rest are assumed reachable.
  2. A block is one area: the needle along a (turned 0 about it; split.placement: its faces' centre over the work
     point, the object on the table) and Step1's exact needle and head test there (seats.exact: a face that fails tries
     the other areas' directions, the squarest first, and joins that area).  No exit: Step3 builds an area's membrane
     leaving against its needle, as combine's default.
  3. Every area alone passes Step3's audit as Step3 builds it, stands and holds (seats.shell, seats.hold); a failing
     area's faces are cut again without its direction, and the faces the exact test misses again (once).  Faces
     assumed reachable -- no needle axis reaches them, no chosen direction covers them, a single face that fails, a
     block too small -- stay within policy['assumed'] (0.1%) of the target area in all, and join the nearest area,
     uncertified (Step6 still installs what it reaches of them).

The plan (Step1's format, step2/split.py) marks every area robot-free, as seats.py does; Step6 splits every area into
the work arm's installations on the finished support.  Area ids are W01, W02, ... the largest first.

  python -m whole_task_2.step2.directions B+big_01 --task objects/B/big_tasks/big_01.json --config robots_config/random/07_kuka.json
  python -m whole_task_2.step2.directions B+big_01 --render-only     (the figure again, step2/directions_figure.py)
"""
import argparse
import time
import numpy as np
from scipy.spatial import cKDTree

from whole_task_2.step1.common import DEFAULT_OUTPUT
from . import split as S
from . import seats as SE
from .directions_figure import render

FOLDER = SE.FOLDER                      # Step6 reads the run's Step2 report there
MODEL = SE.MODEL
POLICY = dict(SE.POLICY, directions=240,   # candidate work directions a (Fibonacci)
              assumed=.001,             # the target area left to no direction, at most (user, 2026-10-29: "扔0.1%吧")
              least=((.6, 4), (.75, 5), (.9, 6)))   # (share of the object's surface, fewest areas) for a target above it


def directions(x, policy):
    """(candidate work directions V, what each reaches: the faces whose screened needle axes lie within the tilt of it)."""
    V = SE.fibonacci(policy['directions']); cosb = np.cos(np.radians(x['s']['max_tilt_deg']))
    return V, np.asarray([x['reach'][x['axes']@v >= cosb-1e-12].any(0) for v in V])


def fewest(sets, faces, area, budget, banned=(), least=1):
    """Few directions (rows of sets, not banned) that cover faces but at most budget of their area, and at least least
    of them: one more at a time, the one that covers the most of what is left (of equals, the one that covers the
    most: it takes the most from the others), then every chosen one swapped for the direction that covers the most
    with the others while that covers more, until they cover enough.  [] when budget takes them all."""
    allowed = np.setdiff1d(np.arange(len(sets)), sorted(banned)); a = area[faces]; need = float(a.sum())-budget
    if need <= 0 or not len(allowed): return []
    R = sets[np.ix_(allowed, faces)]; need = min(need, float(a[R.any(0)].sum()))*(1-1e-9)   # no direction: assumed anyway
    chosen = []; covered = lambda c: float(a[R[c].any(0)].sum()) if c else 0.; reach = R@a
    while covered(chosen) < need or len(chosen) < min(least, len(allowed)):
        gain = (R & ~R[chosen].any(0))@a; gain[chosen] = -1.
        chosen.append(int(np.lexsort((-reach, -gain))[0]))
        better = True
        while better:
            better = False
            for j in range(len(chosen)):
                with_others = (R | R[chosen[:j]+chosen[j+1:]].any(0))@a; c = int(np.argmax(with_others))
                if with_others[c] > covered(chosen)*(1+1e-12): chosen[j] = c; better = True
    return [int(allowed[c]) for c in chosen]


def plan(pool, x, policy):
    """The areas: (passed areas, every attempt's row, faces that fail alone, faces assumed reachable, every block as
    cut: its direction, faces and seed, in cutting order)."""
    mesh = x['mesh']; area = mesh.area_faces; total = float(area[x['target']].sum())
    V, sets = directions(x, policy)
    merged = [int(f) for f in x['unreachable']]; budget = policy['assumed']*total-float(area[merged].sum())
    problems = [(np.asarray(x['root']), set())]; out, rows, alone, missed_once, cuts = [], [], [], set(), []
    surface = total/float(area.sum()); first = True
    while problems:
        offers = []
        for faces, banned in problems:
            # The first cut: at least policy['least'] areas for a target that large a share of the surface (user,
            # 2026-10-29: "纯为了好看 ... 也让问题有意思一些"); a cut again: as few as cover it.
            least = max([k for share, k in policy['least'] if first and surface > share], default=1)
            chosen = fewest(sets, faces, area, max(budget, 0.), banned, least)
            on = sets[chosen][:, faces].any(0) if chosen else np.zeros(len(faces), bool)
            merged += [int(f) for f in faces[~on]]; budget -= float(area[faces[~on]].sum()); covered = faces[on]
            if not chosen: continue
            # About equal, connected blocks, each seeded on the face that faces against its direction most squarely.
            label = SE.blocks(x, covered, sets[chosen], -V[chosen], policy)
            offers += [(c, covered[label == j], set(banned) | {c}) for j, c in enumerate(chosen) if (label == j).any()]
            for c in chosen:                    # the blocks as cut, each with its seed as seats.blocks picks it
                mine = covered[sets[c][covered]]
                cuts.append(dict(direction=V[c].tolist(), faces=[int(f) for f in covered[label == chosen.index(c)]],
                                 seed=int(mine[np.argmax(mesh.face_normals[mine]@-V[c])])))
            print('STEP2 COVER', len(faces), 'faces:', len(chosen), 'directions (at least', str(least)+') cover',
                  round(100*float(area[covered].sum()/area[faces].sum()), 2), '% of their area', flush=True)
        first = False
        problems, pending = [], []
        for c, f, banned in sorted(offers, key=lambda o: area[o[1]].sum()):
            if area[f].sum() <= budget: merged += [int(g) for g in f]; budget -= float(area[f].sum()); continue
            pending.append(dict(v=V[c], faces=f, banned=banned, needles=[V[c]], option=np.array([c]),
                                order=np.zeros((len(f), 1), int), reachable=sets[c][f][:, None]))
        if not pending: break
        SE.exact(pool, x, pending)
        members = []
        for d in pending:
            back = [int(f) for f in d['missed'] if int(f) not in missed_once]; missed_once |= set(back)
            if back: problems.append((np.asarray(back), set()))
            alone += [dict(face=int(f), reason='no needle reaches it') for f in d['missed'] if int(f) not in back]
            if not d['reached']: continue
            _, faces, axes = d['reached'][0]; P = S.placement(mesh, d['v'], S.nominal(mesh, faces)[1], 0., x['s'])
            d['member'] = (f'w{len(out)+len(members)}', faces, axes, d['v'], 0., P); members.append(d)
        own = [pool.submit(SE.shell, [d['member']], -d['v'], True) for d in members]
        own = [f.result() for f in own]
        held = [pool.submit(SE.hold, (d['member'], r['mesh'], r['mesh'])) if r['mesh'] is not None else None for d, r in zip(members, own)]
        for d, r, h in zip(members, own, held):
            faces = d['member'][1]; why = r['reason'] if h is None else (None if h.result().get('passed') else h.result()['reason'])
            row = dict(direction=d['v'].tolist(), faces=len(faces), area_fraction_of_target=float(area[faces].sum()/total),
                       support_cm3=r['volume_cm3'], reason=why)
            if why is None: d['result'] = r; out.append(d); row['verdict'] = 'area'
            elif len(faces) == 1: alone.append(dict(face=int(faces[0]), reason=why)); row['verdict'] = 'fails alone'
            else: problems.append((faces, d['banned'])); row['verdict'] = 'again'
            rows.append(row)
            print('STEP2 DIRECTION', np.round(d['v'], 3).tolist(), round(100*row['area_fraction_of_target'], 1), '%:', row['verdict'],
                  why or '', flush=True)
    return out, rows, alone, merged+[a['face'] for a in alone], cuts


def run(run_name, task, config, workers=22, output_root=DEFAULT_OUTPUT, policy=POLICY):
    began = time.monotonic(); x = S.prepare(run_name, task, config); mesh = x['mesh']
    pool = S.workers_pool(run_name, x['shape'], x['s'], x['config'], workers)
    try:
        found, rows, alone, merged, cuts = plan(pool, x, policy)
        found.sort(key=lambda d: -mesh.area_faces[d['member'][1]].sum()); tasks = []
        for k, d in enumerate(found):
            _, faces, axes, v, a, P = d['member']; V, F = d['result']['mesh']
            tasks.append(dict(id=f'W{k+1:02d}', faces=faces, d=v, centre=S.nominal(mesh, faces)[1], axes=axes, angle=a, pose=P,
                              robot=False, volume_cm3=d['result']['volume_cm3'], vertices=V@P[:3, :3].T+P[:3, 3], triangles=F, merged=[]))
        # Faces assumed reachable join the area with the nearest face (uncertified: Step3 counts their loads, no keep-out).
        owner = np.concatenate([np.full(len(t['faces']), j) for j, t in enumerate(tasks)])
        near = cKDTree(np.concatenate([mesh.triangles_center[t['faces']] for t in tasks]))
        for f in merged: tasks[owner[near.query(mesh.triangles_center[f])[1]]]['merged'].append(int(f))
        plan_, validation = S.finish(x, pool, tasks, FOLDER, __file__, robot=False)
    finally:
        pool.shutdown(cancel_futures=True)
    total = float(mesh.area_faces[x['target']].sum()); assumed = float(mesh.area_faces[merged].sum()/total) if merged else 0.
    shares = [round(100*float(mesh.area_faces[t['faces']].sum()/total), 1) for t in tasks]
    print('STEP2 DIRECTIONS', len(tasks), 'areas; shares', shares, '; assumed reachable', round(100*assumed, 2), '%', flush=True)
    return S.record(x, tasks, plan_, validation, FOLDER, __doc__.split('\n\n')[1:4], began, output_root, figure=render,
                    policy=policy, load_model=MODEL, cut='work directions', attempts=rows, failing_alone=alone, blocks=cuts,
                    assumed_reachable=dict(face_count=len(merged), fraction_of_target=assumed,
                                           areas={t['id']: len(t['merged']) for t in tasks if t['merged']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run'); p.add_argument('--task'); p.add_argument('--config')
    p.add_argument('--workers', type=int, default=22); p.add_argument('--render-only', action='store_true')
    a = p.parse_args()
    if a.render_only: render(a.run, folder=FOLDER)
    elif not (a.task and a.config): p.error('--task and --config are required (but for --render-only)')
    else: run(a.run, a.task, a.config, a.workers)
