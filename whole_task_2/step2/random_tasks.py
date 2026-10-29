"""Step2 baseline: random tasks.

A task is a random needle direction d (uniform on the sphere, object frame) and a random turn about it (uniform).  It
works the big task's faces no earlier task works that the needle reaches within the tilt cone about d, needle and head
clear of the object (step1/reach.certify, the smallest tilt first).  It is kept only when Step3's check without the work
arm passes at its turn (step2/split.py: the largest legal support -- the 5 mm outer layer less the object, every face's
needle and head and the exit sweep -- is one piece, lets the object leave and holds every paired force and moment);
else it is drawn again.  Draws go on until every face some needle axis reaches is worked, or `patience` draws in a row
keep nothing (the rest is reported).

Draws are made `batch` at a time against the same unworked faces, judged in parallel and kept in the order drawn; a
task some of whose faces an earlier one of the batch took is judged again on what is left.  The rest -- the KUKA, the
plan for Step3, its replay, the record and the figure -- is step2/split.py's.

  python -m whole_task_2.step2.random_tasks B+2task_random --task objects/B/big_tasks/2task.json --config robots_config/random/07_kuka.json
  python -m whole_task_2.run_station B+2task_random --start step3
"""
import argparse
import time
import numpy as np

from whole_task_2.step1.common import DEFAULT_OUTPUT
from whole_task_2.step1 import plan_reach as P1
from . import split as S

FOLDER = 'step2_random'
POLICY = dict(seed=20261028,    # the draws' random generator
              batch=22,         # draws judged together
              patience=300)     # draws in a row that keep nothing: the rest is left


def draw(rng):
    """A needle direction uniform on the sphere (object frame) and a turn about it uniform in [0, 360) deg."""
    d = rng.normal(size=3); return d/np.linalg.norm(d), float(rng.uniform(0., 360.))


def needles(pool, x, draws, open_):
    """For every draw, {face: axis}: the open faces its needle reaches within the tilt cone, the least tilt that clears."""
    s = x['s']; axes, reach = x['axes'], x['reach']; cosb = np.cos(np.radians(s['max_tilt_deg'])); jobs = []
    for j, (d, _) in enumerate(draws):
        C = axes@d; near = C >= cosb-1e-12
        for f in open_[reach[np.ix_(near, open_)].any(0)]:
            k = np.flatnonzero(near & reach[:, f]); k = k[np.argsort(-C[k])]
            jobs.append((int(f), [(j, int(i), axes[i]) for i in k], s['needle_length_m'], s['head_radius_m'], s['needle_radius_m'],
                         x['eps'], s['max_incidence_deg']))
    out = [{} for _ in draws]
    for face, j, i, *_ in pool.map(P1._certify, jobs, chunksize=16):
        if j is not None: out[j][face] = axes[i]
    return out


def judge(pool, x, items):
    """Step3's check without the arm for every item (name, faces, axes, d, angle): its support row, judged together."""
    futures = []
    for name, faces, axes, d, angle in items:
        P = S.placement(x['mesh'], d, S.nominal(x['mesh'], np.asarray(faces))[1], angle, x['s'])
        futures.append(pool.submit(S.support, (name, np.asarray(faces), np.asarray(axes), P, angle)))
    return [f.result() for f in futures]


def sample(pool, x, policy):
    """The random tasks: (tasks, every draw's row, the faces left)."""
    rng = np.random.default_rng(policy['seed']); mesh = x['mesh']; total = float(mesh.area_faces[x['target']].sum())
    open_ = x['root'].copy(); tasks, rows = [], []; idle = 0; made = 0
    while len(open_) and idle < policy['patience']:
        batch = [draw(rng) for _ in range(policy['batch'])]; found = needles(pool, x, batch, open_)
        items = [(f'D{made+k}', sorted(found[k]), [found[k][f] for f in sorted(found[k])], *batch[k]) for k in range(len(batch)) if found[k]]
        results = judge(pool, x, items); taken = set(); again = []; kept = 0

        def keep(item, r):
            name, faces, axes, d, angle = item; centre = S.nominal(mesh, np.asarray(faces))[1]
            tasks.append(dict(id=f'R{len(tasks)+1:02d}', draw=name, faces=np.asarray(faces), d=d, centre=centre, axes=np.asarray(axes),
                              angle=angle, pose=S.placement(mesh, d, centre, angle, x['s']), volume_cm3=r['volume_cm3'],
                              vertices=r['vertices'], triangles=r['faces']))
            print('STEP2 KEEP', tasks[-1]['id'], name, len(faces), 'faces', round(100*mesh.area_faces[faces].sum()/total, 2), '%, support',
                  round(r['volume_cm3'], 1), 'cm3', flush=True)
        for item, r in zip(items, results):
            name, faces, axes, d, angle = item
            rows.append(dict(draw=name, needle_object=d.tolist(), turn_deg=angle, faces=len(faces), passed=r['passed'], reason=r.get('reason'),
                             seconds=r['seconds']))
            if not r['passed']: continue
            left = [k for k, f in enumerate(faces) if f not in taken]
            if len(left) == len(faces): keep(item, r); kept += 1
            elif left: again.append((name, [faces[k] for k in left], [axes[k] for k in left], d, angle))
            taken |= {faces[k] for k in left}
        # What an earlier draw of the batch took is not this one's: judged again on what is left.
        for item, r in zip(again, judge(pool, x, again)):
            rows.append(dict(draw=item[0], again=True, faces=len(item[1]), passed=r['passed'], reason=r.get('reason'), seconds=r['seconds']))
            if r['passed']: keep(item, r); kept += 1
        made += len(batch); worked = np.concatenate([t['faces'] for t in tasks]) if tasks else np.zeros(0, int)
        open_ = np.setdiff1d(open_, worked); idle = 0 if kept else idle+len(batch)
        print('STEP2 DRAWS', made, '| tasks', len(tasks), '| open', round(100*mesh.area_faces[open_].sum()/total, 2), '%', flush=True)
    return tasks, rows, open_


def run(run_name, task, config, workers=22, output_root=DEFAULT_OUTPUT, policy=POLICY):
    began = time.monotonic(); x = S.prepare(run_name, task, config)
    pool = S.workers_pool(run_name, x['shape'], x['s'], x['config'], workers)
    try:
        tasks, rows, left = sample(pool, x, policy)
        plan, validation = S.finish(x, pool, tasks, FOLDER, __file__)
    finally:
        pool.shutdown(cancel_futures=True)
    return S.record(x, tasks, plan, validation, FOLDER, __doc__.split('\n\n')[1:3], began, output_root,
                    policy=policy, draws=rows, draw_count=len([r for r in rows if not r.get('again')]), left_face_ids=left.tolist())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run'); p.add_argument('--task', required=True); p.add_argument('--config', required=True)
    p.add_argument('--workers', type=int, default=22); p.add_argument('--seed', type=int, default=POLICY['seed'])
    p.add_argument('--render-only', action='store_true')
    a = p.parse_args()
    S.render(a.run, folder=FOLDER) if a.render_only else run(a.run, a.task, a.config, a.workers, policy=dict(POLICY, seed=a.seed))
