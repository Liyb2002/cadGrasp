"""Step4's keep-out per task: needle, head and KUKA, as one solid in the object frame.

Needle and head are Step3's tool keep-out, unchanged (step3/tool_keepout.py: along every
certified face's own axis, the needle within r over [0, L], the head within R beyond L).
The KUKA comes from the joints Step1 certified and Step3 kept: for every certified face,
the arm at the face's three vertices and its centroid.  The arm rises and falls with the
object, so where it stands relative to the object does not depend on the height, and its
keep-out lives in the object frame like the tool's.

Per link, the convex hull of its collision hull (the hull step1.arm tests) over every stored
configuration whose link sits in one cell: configurations are grouped by the centre of the link's
hull on a grid of cell_fraction times the object diagonal.  Grouping by where the link is keeps
each piece tight even when the redundant arm changes posture between neighbouring faces (Step1
solves each face on its own; consecutive configurations jump by up to 125 degrees).  Grouping by
face position instead gave pieces about 27 cm across and unions of up to minutes.  Each piece is
the hull of the configurations it covers, so every stored link hull lies in one piece; that is
checked plane by plane.

The keep-out covers the arm where Step1 certified it, painting each face; moves between
painting points are not certified there and not covered here.  Final acceptance keeps the
exact test of step1.arm at the stored joints; the keep-out lets construction, the ground
oracle and feet avoid the arm without it.

A later step reads the saved keep-outs (load) if the saved files of the earlier steps they were built
from are unchanged: each step reruns from the files of the one before it, whatever code built them.  The
keep-out's own cache and its disk replay (validate) also bind its code.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import trimesh
import manifold3d as md
from scipy.spatial import ConvexHull

from whole_task_2.step1.common import ROOT, sha256, write_json
from whole_task_2.step1.reach_video import kuka as work_arm
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.tool_membranes import load_model, stem, task_partition

HERE = Path(__file__).parent
POLICY = dict(cell_fraction=.12, containment_tolerance_m=2e-6)
TOLERANCE = 1e-10                 # m^3, the Boolean volume tolerance of Step3
SOURCES = [Path(__file__), ROOT/'whole_task_2/step3/tool_keepout.py', ROOT/'whole_task_2/step3/tool_membranes.py',
           ROOT/'whole_task_2/step1/arm.py', ROOT/'whole_task_2/step5/video/kuka.py',
           ROOT/'whole_task_2/step5/video/so101.py', ROOT/'whole_task_2/step1/reach_video.py']


def folders(object_name):
    data = ROOT/'whole_task_2/data'/object_name
    return data/'step3_tool_membranes', data/'step4_keepout'


def data_inputs(object_name):
    """The saved files of earlier steps the keep-out is built from: Step3's tasks, poses and joints, Step1's plan."""
    source, _ = folders(object_name)
    files = [source/'report.json', source/'initialization.npz', ROOT/'whole_task_2/data'/object_name/'step1_reach/plan.npz']
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def signature(object_name):
    """Every input the keep-out depends on: the data inputs, the code, the policy."""
    inputs = dict(data_inputs(object_name), **{str(p.relative_to(ROOT)): sha256(p) for p in SOURCES})
    key = hashlib.sha256(json.dumps(dict(inputs=inputs, policy=POLICY), sort_keys=True).encode()).hexdigest()
    return inputs, key


def link_points(model, r):
    """Per link, its collision hull's vertices at every stored configuration of record r: (faces, 4, n, 3), object frame."""
    arm = work_arm(model.setup); inverse = np.linalg.inv(r['pose'])
    hulls = [v[ConvexHull(v).vertices] for v in arm.collisions]
    out = [np.empty((len(r['joints']), 4, len(h), 3)) for h in hulls]
    for f, Q in enumerate(r['joints']):
        for c, q in enumerate(Q):
            for k, (h, (R, o)) in enumerate(zip(hulls, arm.forward(q, geometry=True)[2])):
                out[k][f, c] = (h@R.T+o)@inverse[:3, :3].T+inverse[:3, 3]
    return out


def arm_pieces(model, r, points):
    """Convex pieces (their hull vertices) covering every stored link hull, and the containment record."""
    cell = POLICY['cell_fraction']*float(np.linalg.norm(model.source.extents))
    pieces = []; counts = []; spreads = []; residual = 0.
    for X in points:
        Y = X.reshape(-1, X.shape[2], 3); centres = Y.mean(1)
        spreads.append(float(np.ptp(centres, 0).max()))
        _, group = np.unique(np.floor(centres/cell).astype(np.int64), axis=0, return_inverse=True); group = group.ravel()
        counts.append(int(group.max()+1))
        for g in range(group.max()+1):
            # A link that barely moves repeats its vertices thousands of times; near-duplicates make
            # Qhull return ~10^5 facets, so they are merged on a 1 um grid first (the residual covers it).
            Z = Y[group == g].reshape(-1, 3); hull = ConvexHull(np.unique(np.round(Z, 6), axis=0))
            # Every stored link hull in this cell lies inside this piece: all its vertices meet every plane.
            for chunk in np.array_split(Z, max(1, len(Z)*len(hull.equations)//10**7)):
                residual = max(residual, float((chunk@hull.equations[:, :3].T+hull.equations[:, 3]).max()))
            pieces.append(hull.points[hull.vertices])
    if residual > POLICY['containment_tolerance_m']: raise ValueError(r['id']+': a stored link hull leaves its piece')
    return pieces, dict(cell_size_m=cell, pieces=len(pieces), pieces_per_link=counts, link_spread_m=spreads,
                        containment_residual_m=residual)


def arm_solid(model, r):
    """The KUKA keep-out of record r (object frame) and its record."""
    pieces, record = arm_pieces(model, r, link_points(model, r))
    return G.union([md.Manifold.hull_points(p/G.SCALE) for p in pieces]), record


def build_one(args):
    """Task i's tool, KUKA and combined keep-out meshes, and its record."""
    object_name, i = args; began = time.monotonic()
    source, _ = folders(object_name); model, _, _ = load_model(source); r = model.records[i]
    arm, record = arm_solid(model, r); seconds_arm = time.monotonic()-began
    tool = G.solid(model.tool(i)); both = G.union([tool, arm])
    body = G.solid(model.source); outer = G.solid(model.outer)
    record.update(id=r['id'], working_face_count=len(r['working_face_ids']),
                  certified_faces=len(r['certified_face_ids']), stored_configurations=4*len(r['joints']),
                  tool_cm3=G.volume(tool)*1e6, arm_cm3=G.volume(arm)*1e6, keepout_cm3=G.volume(both)*1e6,
                  arm_meets_object_cm3=G.volume(arm ^ body)*1e6, arm_meets_outer_layer_cm3=G.volume(arm ^ (outer-body))*1e6,
                  arm_seconds=seconds_arm, seconds=time.monotonic()-began)
    meshes = {name: G.mesh(value) for name, value in (('tool', tool), ('arm', arm), ('keepout', both))}
    print(r['id'], 'KEEPOUT', round(record['keepout_cm3'], 1), 'cm3,', record['pieces'], 'KUKA pieces,',
          round(record['seconds'], 1), 's', flush=True)
    return record, {f'{name}_{part}': getattr(m, part) for name, m in meshes.items() for part in ('vertices', 'faces')}


def turned_keepout(object_name, i, angle):
    """Task i turned by angle (deg) about the nominal direction through the work point.

    The base is fixed, so the KUKA is solved again for every certified face (step3's turned); the
    needle and head keep-out does not change in the object frame.  Returns (pose, joints, keep-out
    trimesh, record), or None when the KUKA cannot hold some face there.  Cached per (task, angle)."""
    from whole_task_2.step3.tool_membranes import turned
    source, folder = folders(object_name); key = signature(object_name)[1]
    model, _, _ = load_model(source); r = model.records[i]
    path = folder/'turns'/f'{stem(r["id"])}_{float(angle):+.9f}.npz'; meta = path.with_suffix('.json')
    if meta.exists():
        saved = json.loads(meta.read_text())
        if saved.get('signature') == key:
            if saved['status'] != 'passed': return None
            with np.load(path) as z:
                return z['pose'], z['joints'], trimesh.Trimesh(z['keepout_vertices'], z['keepout_faces'], process=False), saved
    began = time.monotonic(); path.parent.mkdir(parents=True, exist_ok=True)
    t = turned(r, model.source, model.setup, float(angle))
    if t is None:
        write_json(meta, dict(status='failed', id=r['id'], angle_deg=float(angle), signature=key,
                              reason='KUKA cannot hold every certified face', seconds=time.monotonic()-began)); return None
    arm, record = arm_solid(model, t); both = G.mesh(G.union([G.solid(model.tool(i)), arm]))
    record.update(status='passed', id=r['id'], angle_deg=float(angle), signature=key, keepout_cm3=both.volume*1e6,
                  seconds=time.monotonic()-began)
    np.savez_compressed(path, pose=t['pose'], joints=t['joints'], keepout_vertices=both.vertices, keepout_faces=both.faces)
    record['sha256'] = sha256(path); write_json(meta, record)
    return t['pose'], t['joints'], both, record


def compute(object_name, workers):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    source, _ = folders(object_name); model, _, _ = load_model(source)
    with ProcessPoolExecutor(max_workers=min(workers, len(model.ids)), mp_context=multiprocessing.get_context('spawn')) as pool:
        return model, list(pool.map(build_one, [(object_name, i) for i in range(len(model.ids))]))


def run(object_name='B', workers=6):
    began = time.monotonic(); inputs, key = signature(object_name)
    _, folder = folders(object_name); folder.mkdir(parents=True, exist_ok=True)
    model, results = compute(object_name, workers); tasks = []
    for i, (record, arrays) in enumerate(results):
        path = folder/(stem(model.ids[i])+'.npz'); np.savez_compressed(path, **arrays)
        tasks.append(dict(record, file=path.name, sha256=sha256(path)))
    report = dict(status='passed', object=object_name, signature=key, input_sha256=inputs, policy=POLICY,
                  definition='needle and head: Step3 tool keep-out; KUKA: per link, hulls of its collision hull over the '
                             'stored configurations whose link centre falls in one grid cell; one solid per task, object frame',
                  height_rule='the KUKA rises and falls with the object, so the keep-out is fixed in the object frame',
                  tasks=tasks, task_partition=task_partition(model.records, model.source), seconds=time.monotonic()-began)
    write_json(folder/'report.json', report)
    validate(object_name, workers)
    return report


def load(object_name='B'):
    """{task id: (tool, arm, keepout) trimeshes, object frame}, refusing a keep-out built from other data."""
    _, folder = folders(object_name); report = json.loads((folder/'report.json').read_text())
    if any(report['input_sha256'].get(p) != d for p, d in data_inputs(object_name).items()):
        raise ValueError('Keep-out data inputs changed; rerun whole_task_2.step4.keepout')
    out = {}
    for t in report['tasks']:
        path = folder/t['file']
        if sha256(path) != t['sha256']: raise ValueError(t['id']+': changed keep-out file')
        with np.load(path) as z:
            out[t['id']] = tuple(trimesh.Trimesh(z[name+'_vertices'], z[name+'_faces'], process=False) for name in ('tool', 'arm', 'keepout'))
    return out


def validate(object_name='B', workers=6):
    """Disk replay: same inputs, every keep-out rebuilt from Step3's saved joints and the same solid as the
    saved one (symmetric difference within the Boolean tolerance; the union's vertex order may differ)."""
    _, folder = folders(object_name); report = json.loads((folder/'report.json').read_text())
    inputs, key = signature(object_name)
    if report['signature'] != key or report['input_sha256'] != inputs: raise ValueError('Keep-out inputs changed since the build')
    model, results = compute(object_name, workers); rows = []
    partition = task_partition(model.records, model.source)
    if report['task_partition'] != partition:raise ValueError('Step4 task partition differs from Step3')
    if [task['id'] for task in report['tasks']] != model.ids:raise ValueError('Step4 task IDs differ from Step3')
    for saved, (record, arrays) in zip(report['tasks'], results):
        difference = {}
        with np.load(folder/saved['file']) as z:
            for name in ('tool', 'arm', 'keepout'):
                A = G.solid(trimesh.Trimesh(z[name+'_vertices'], z[name+'_faces'], process=False))
                B = G.solid(trimesh.Trimesh(arrays[name+'_vertices'], arrays[name+'_faces'], process=False))
                difference[name] = G.volume(A-B)+G.volume(B-A)
        if max(difference.values()) > TOLERANCE: raise ValueError(saved['id']+': rebuilt keep-out differs '+str(difference))
        if record['containment_residual_m'] > POLICY['containment_tolerance_m']: raise ValueError(saved['id']+': containment')
        rows.append(dict(id=saved['id'], symmetric_difference_m3=difference, containment_residual_m=record['containment_residual_m'],
                         arm_meets_outer_layer_cm3=record['arm_meets_outer_layer_cm3']))
    result = dict(status='passed', report_sha256=sha256(folder/'report.json'), tasks=rows, task_partition=partition)
    write_json(folder/'validation.json', result); print('KEEPOUT DISK REPLAY PASS', flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', default='B'); p.add_argument('--workers', type=int, default=6)
    p.add_argument('--validate-only', action='store_true'); p.add_argument('--no-figure', action='store_true')
    a = p.parse_args()
    validate(a.object, a.workers) if a.validate_only else run(a.object, a.workers)
    if not a.no_figure:
        from .render_keepout import render
        render(a.object)
