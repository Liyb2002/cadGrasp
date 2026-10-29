"""Step4's view of a big task's candidate pool (step3/big_init.py, BIG_TASK.md).

PoolContext gives combine.py and Step5 what seats.Context gives them for Step3's tasks: Step3's model over the
pool's tasks, every task's keep-out (needle, head and KUKA), the exit sweeps and the records.  A task's work area
is its whole certified area: conservative, since the fewer faces a task finally works the fewer its demands and the
smaller its keep-out, so a support that holds the whole area holds the part the task keeps.

prepare() builds, for every pool task, its keep-out (step4/keepout.arm_solid with Step3's tool keep-out) and its
Step3 membrane (the 5 mm outer layer minus the object, the tool keep-out and the exit sweep against the needle,
largest piece, exits refined: geometry only, combine audits every start), saved where combine and Step5 read
them (data/<run>/step4_keepout/, data/<run>/step3_tool_membranes/).  A task whose membrane cannot be built is
listed as failed; combine leaves it out.

  python -m whole_task_2.step4.pool whole_shape [--workers 16]
"""
import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import trimesh

from robots_config.constraints import load_config
from whole_task_2.step1.common import ROOT, load_shape, sha256, write_json
from whole_task_2.common.wrenches import resolve_policy
from whole_task_2.operations.state import Layout
from whole_task_2.operations.sweep_cache import sweep_mesh
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.exit_geometry import whole_object_sweep
from whole_task_2.step3.tool_membranes import ToolMembraneModel, POLICIES, stem
from .keepout import arm_solid
from .seats import Context, FOLDER, construction_length, unit


def data(run):
    return ROOT/'whole_task_2/data'/run


def is_pool(run):
    return (data(run)/'step3_init/report.json').exists()


@lru_cache(maxsize=4)
def pool(run):
    """The pool's report and arrays (step3/big_init.py)."""
    folder = data(run)/'step3_init'; report = json.loads((folder/'report.json').read_text())
    with np.load(folder/'initialization.npz') as z: arrays = {k: z[k] for k in z.files}
    return report, arrays


def base_model(run):
    """Step3's tool membrane model over every pool task, its work area its whole certified area."""
    report, z = pool(run); obj = report['object']; source = load_shape(ROOT/'objects'/obj/'mesh.stl')
    physical = json.loads((ROOT/'objects'/obj/'meta.json').read_text()); policy = {}
    for p in POLICIES: policy.update(json.loads(p.read_text()))
    policy = resolve_policy(policy, load_config(ROOT/report['station'])); policy['task_count'] = len(report['tasks'])
    with np.load(ROOT/'whole_task_2/data'/obj/'step3_mold_plate/fitted_outer.npz') as o: outer = trimesh.Trimesh(o['vertices'], o['faces'], process=False)
    records = [dict(id=t['id'], working_face_ids=z[t['id']+'_face_ids'].tolist(), certified_face_ids=z[t['id']+'_face_ids'].tolist(),
                    axes=z[t['id']+'_axes'], joints=z[t['id']+'_joints'], pose=z[t['id']+'_pose'], angle_deg=0.) for t in report['tasks']]
    return ToolMembraneModel(source, outer, records, report['setup'], physical, policy, run)


def keepouts(run, ids):
    """{task id: (tool, arm, keepout) trimeshes, object frame} as prepare() saved them."""
    out = {}
    for i in ids:
        with np.load(data(run)/'step4_keepout'/(stem(i)+'.npz')) as z:
            out[i] = tuple(trimesh.Trimesh(z[n+'_vertices'], z[n+'_faces'], process=False) for n in ('tool', 'arm', 'keepout'))
    return out


class PoolContext(Context):
    """seats.Context over a pool: the same model, keep-outs, exits and records; pool tasks are not turned."""
    def __init__(self, run):
        self.object_name = run; self.folder = data(run)/FOLDER; self.base = base_model(run)
        self.zero = keepouts(run, self.base.ids); a = unit(self.base.axis); self.sweeps = []; self.records = {}
        length = construction_length(self.base)
        for r in self.base.records:
            v = -(np.asarray(r['pose'])[:3, :3].T@a)                # the exit against the needle, object frame
            self.sweeps.append(G.solid(sweep_mesh(self.folder/'sweeps', self.base.source, v, length,
                lambda v=v: G.mesh(whole_object_sweep(self.base.source, v, length)[0]), 'object_frame_v1')))

    @property
    def passed(self):
        raise ValueError('a pool has no seat turns')

    def record(self, i, angle):
        if angle != 0: raise ValueError('pool tasks are not turned')
        if (i, 0.) not in self.records: r = dict(self.base.records[i]); self.records[i, 0.] = r, self.zero[r['id']][2]
        return self.records[i, 0.]


def build_one(args):
    """Task i's keep-out (tool, KUKA, both) and its Step3 membrane, saved; its record."""
    run, i = args; began = time.monotonic(); model = base_model(run); r = model.records[i]; name = stem(r['id'])
    arm, record = arm_solid(model, r); tool = G.solid(model.tool(i)); both = G.union([tool, arm])
    meshes = {n: G.mesh(v) for n, v in (('tool', tool), ('arm', arm), ('keepout', both))}
    np.savez_compressed(data(run)/'step4_keepout'/(name+'.npz'), **{f'{n}_{p}': getattr(m, p) for n, m in meshes.items() for p in ('vertices', 'faces')})
    record.update(id=r['id'], tool_cm3=G.volume(tool)*1e6, arm_cm3=G.volume(arm)*1e6, keepout_cm3=G.volume(both)*1e6)
    one = model.subset([i]); q = Layout(np.asarray(r['pose'])[None], np.eye(4)[None], (-unit(model.axis))[None])
    try:
        candidate = one.build(q, facewise=True); support = candidate['support']
        np.savez_compressed(data(run)/'step3_tool_membranes'/(name+'.npz'), vertices=support.vertices, faces=support.faces,
                            poses=q.poses, exits=q.exits)
        record.update(status='passed', membrane_cm3=candidate['volume_cm3'])
    except (ValueError, RuntimeError) as ex: record.update(status='failed', reason=str(ex)[:300])
    record['seconds'] = time.monotonic()-began
    print(r['id'], record['status'].upper(), round(record['keepout_cm3'], 1), 'cm3 keep-out', round(record['seconds'], 1), 's', flush=True)
    return record


def _pin(cpus):
    os.sched_setaffinity(0, {cpus.get()})


def prepare(run, workers=16):
    began = time.monotonic(); report, _ = pool(run)
    for sub in ('step4_keepout', 'step3_tool_membranes'): (data(run)/sub).mkdir(parents=True, exist_ok=True)
    context_ = multiprocessing.get_context('spawn'); cpus = sorted(os.sched_getaffinity(0)); queue = context_.Queue()
    n = len(report['tasks'])
    for k in range(min(workers, n)): queue.put(cpus[k % len(cpus)])
    with ProcessPoolExecutor(min(workers, n), mp_context=context_, initializer=_pin, initargs=(queue,)) as executor:
        records = list(executor.map(build_one, [(run, i) for i in range(n)]))
    inputs = {str(p.relative_to(ROOT)): sha256(p) for p in (data(run)/'step3_init/report.json', data(run)/'step3_init/initialization.npz')}
    for sub, rule in (('step4_keepout', 'needle and head: Step3 tool keep-out; KUKA: step4/keepout.arm_solid at every stored joint; object frame'),
                      ('step3_tool_membranes', 'the 5 mm outer layer minus the object, the tool keep-out and the exit sweep against '
                                               'the needle, largest piece, exits refined; world frame at the task pose')):
        write_json(data(run)/sub/'report.json', dict(status='passed', run=run, source='pool', input_sha256=inputs, rule=rule, tasks=records,
                                                     seconds=time.monotonic()-began))
    print('POOL PREPARED', run, sum(r['status'] == 'passed' for r in records), 'of', n, 'membranes', round(time.monotonic()-began), 's', flush=True)
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run'); p.add_argument('--workers', type=int, default=16)
    a = p.parse_args(); prepare(a.run, a.workers)
