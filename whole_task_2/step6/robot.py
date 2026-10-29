"""Step6: the work arm on the finished support -- every robot-free area of Step2 split into the installations the
KUKA needs.

Step2 cut the seats without the robot (step2/seats.py): an area is a seat's faces on one base, loaded about each
face's own needle.  Step5 finished one support for all of them.  Here every area gets its installations: the support
on the same base (the same face down), turned about the vertical and moved on the table, so that the station's needle
and the KUKA reach its faces; turning and moving cost no material, only another installation.  For every area:

  1. Turns: the station's needle at every turn of policy['turns'] about the vertical, in the object frame; a turn
     reaches the faces whose screened needle axes lie within the tilt cone of it.  A greedy cover picks the turns.
  2. An installation: a turn, its faces' centre over the work point (step2/split.placement), the support's lowest
     point (its feet, legs as long as combine made them) on the table.  Every face: Step1's
     exact needle and head test, the KUKA at its four configurations (step1/arm), then on the finished support the
     KUKA's links do not meet it (step3/tool_membranes.arm_hits) and the face's needle and head do not either.  The
     faces that pass are the installation's; the others go to the next installation (the cover again, over what is
     left, so a face may come back at another turn or, centred anew, at the same one).
  3. Its loads on the finished support: the station's needle cone (world_needle_direction_cone), more than 99% of the
     paired samples (step3/independent_physics.HELD).  An installation that fails is tried again as two, its faces
     halved by their normals (step2/split.halves), at the same turn: fewer faces, fewer demands.
  An area combine turned in its own seat (several bases) is split among them first (shares): each base installs its
  part, and the faces one base leaves the others try.
  The faces no installation takes are reported.

  python -m whole_task_2.step6.robot --object B+big_01_free
"""
import argparse
import json
import time
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, write_json
from whole_task_2.step1 import plan_reach as P1
from whole_task_2.step1.arm import OK
from whole_task_2.step2 import split as S
from whole_task_2.step2.seats import cover
from whole_task_2.operations.geometry import moved_mesh, contact_surface
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.independent_physics import certificate
from whole_task_2.step4.seats import context, read_mesh
from whole_task_2.step5.combined import distinct, FOLDER as STEP5

FOLDER = 'step6_robot'
POLICY = dict(turns=24,             # the needle's turns about the vertical tried on an area's base
              rounds=6)             # covers over the faces left (each may add installations)


def data(run):
    return ROOT/'whole_task_2/data'/run


# ---------------------------------------------------------------------------------------------
# The workers' side (step2/split.workers_pool: Step1's mesh and KUKA, Step3's inputs).

def model_of(faces, axes, P, joints):
    """Step3's model of one installation, the KUKA included, the station's needle cone as the loads."""
    W = S._W; faces = [int(f) for f in faces]
    r = dict(id='i', direction_index=0, working_face_ids=faces, certified_face_ids=faces, pose=P, axes=np.asarray(axes),
             joints=np.asarray(joints), angle_deg=0.)
    policy = dict(W['policy'], force_direction_model='world_needle_direction_cone')
    return S.TM.ToolMembraneModel(W['source'], W['outer'], [r], W['setup'], W['physical'], policy, W['run']), r


def clear(job):
    """The faces of an installation whose KUKA and tool keep off the finished support (support frame, at X)."""
    faces, axes, joints, P, X, support = job; support = trimesh.Trimesh(*support, process=False)
    model, r = model_of(faces, axes, P, joints)
    hit = set(model.arm_hits(moved_mesh(support, P@np.linalg.inv(X)), r))
    solid = G.solid(moved_mesh(support, np.linalg.inv(X)))                       # object frame
    if G.volume(G.solid(model.tool(0)) ^ solid) <= model.tolerance: return [k for k, f in enumerate(faces) if int(f) not in hit]
    for k, f in enumerate(faces):                                               # some tool meets it: which
        if int(f) in hit: continue
        one, _ = model_of([f], axes[k:k+1], P, joints[k:k+1])
        if G.volume(G.solid(one.tool(0)) ^ solid) > one.tolerance: hit.add(int(f))
    return [k for k, f in enumerate(faces) if int(f) not in hit]


def loaded(job):
    """Does the finished support hold the installation's loads (more than HELD of its paired samples)?"""
    faces, axes, joints, P, X, support = job; support = trimesh.Trimesh(*support, process=False)
    model, r = model_of(faces, axes, P, joints); d = model.world_data(r, P); world = moved_mesh(support, P@np.linalg.inv(X))
    try: proof, _ = certificate(d['body_world'], contact_surface(d['body_world'], world), d['domain'], d['endpoints'], d['samples'])
    except RuntimeError as ex: return dict(passed=False, reason=str(ex)[:120])
    return dict(passed=bool(proof['passed']), held_sample_share=proof['held_sample_share'])


# ---------------------------------------------------------------------------------------------
# One area.

def turn_of(R, a):
    """(needle in the object frame, the turn about the needle for split.placement, deg) of rotation R (world <- object)."""
    v = R.T@a; return v, float(np.degrees(Rotation.from_matrix(R@S.toward(v, a).T).as_rotvec()@a))


def install(pool, x, faces, R, X, support, policy):
    """Installations of one area (rotation R at turn 0, seat X) over faces: [installation], faces left."""
    s = x['s']; a = S.unit(s['working_direction_world']); cosb = np.cos(np.radians(s['max_tilt_deg']))
    axes, reach = x['axes'], x['reach']; left = np.asarray(sorted(map(int, faces))); out = []
    for _ in range(policy['rounds']):
        if not len(left): break
        turns = [turn_of(Rotation.from_rotvec([0., 0., phi]).as_matrix()@R, a) for phi in 2*np.pi*np.arange(policy['turns'])/policy['turns']]
        full = np.asarray([reach[axes@v >= cosb-1e-12].any(0) for v, _ in turns])        # what each turn reaches, every face
        picked = cover(full[:, left], np.arange(len(left)), x['mesh'].area_faces[left])
        if not picked: break
        made = 0; queue = [(k, None) for k in picked]
        while queue:
            # (turn, faces): None takes what the turn reaches of the faces left now; a turn whose loads fail comes back
            # as two smaller installations, its faces halved by their normals (step2/split.halves).
            k, mine = queue.pop(0); mine = left[full[k][left]] if mine is None else np.intersect1d(mine, left)
            if not len(mine): continue
            v, psi = turns[k]; P = S.placement(x['mesh'], v, S.nominal(x['mesh'], mine)[1], psi, s); C = axes@v
            # The support, not the object, stands on the table: its lowest point (a foot) there, the object above it.
            W = P@np.linalg.inv(X); P[2, 3] += s['table_height_m']-float((support[0]@W[:3, :3].T+W[:3, 3])[:, 2].min())
            jobs = []
            for f in mine:
                b = np.flatnonzero((C >= cosb-1e-12) & reach[:, f]); b = b[np.argsort(-C[b])]
                jobs.append((int(f), [(0, int(i), axes[i]) for i in b], s['needle_length_m'], s['head_radius_m'], s['needle_radius_m'],
                             x['eps'], s['max_incidence_deg']))
            got = {face: axes[i] for face, n, i, *_ in pool.map(P1._certify, jobs, chunksize=16) if n is not None}
            pairs = [(f, got[f]) for f in mine if f in got]
            if not pairs: continue
            arm = {face: Q for rows in pool.map(P1._arm_pairs, [(0, P, pairs[i:i+60]) for i in range(0, len(pairs), 60)])
                   for face, _, code, Q in rows if code == OK}
            ok = [f for f, _ in pairs if f in arm]
            if not ok: continue
            ax = np.asarray([got[f] for f in ok]); Q = np.asarray([arm[f] for f in ok])
            keep = pool.submit(clear, (ok, ax, Q, P, X, support)).result()
            ok, ax, Q = [ok[i] for i in keep], ax[keep], Q[keep]
            if not ok: continue
            loads = pool.submit(loaded, (ok, ax, Q, P, X, support)).result()
            print('STEP6   turn', round(360*k/policy['turns']), 'deg:', len(mine), 'faces aimed,', len(ok), 'clear,',
                  'loads', 'hold' if loads['passed'] else 'fail', loads.get('held_sample_share', loads.get('reason')), flush=True)
            if not loads['passed']:
                if len(ok) > 1: queue += [(k, h) for h in S.halves(x['mesh'], np.asarray(ok))]
                continue
            out.append(dict(turn_deg=360.*k/policy['turns'], world_from_object=P, faces=np.asarray(ok), axes=ax, joints=Q,
                            held_sample_share=loads['held_sample_share']))
            left = np.setdiff1d(left, ok); made += 1
        if not made: break
    return out, left


def shares(x, faces, bases, policy):
    """An area on several bases (combine turned it in its own seat: Rotational reuse): its faces in one part per base,
    the largest part halved by the normals (step2/split.halves) until there are enough, each part to the base whose
    turns reach the most of its area (the largest overlaps first)."""
    s = x['s']; a = S.unit(s['working_direction_world']); cosb = np.cos(np.radians(s['max_tilt_deg'])); area = x['mesh'].area_faces
    parts = [np.asarray(faces, int)]
    while len(parts) < len(bases):
        big = max(range(len(parts)), key=lambda k: area[parts[k]].sum()); parts += list(S.halves(x['mesh'], parts.pop(big)))
    reached = [np.any([x['reach'][x['axes']@turn_of(Rotation.from_rotvec([0., 0., phi]).as_matrix()@R, a)[0] >= cosb-1e-12].any(0)
                       for phi in 2*np.pi*np.arange(policy['turns'])/policy['turns']], axis=0) for R in bases]
    score = np.array([[area[q[reached[b][q]]].sum() for b in range(len(bases))] for q in parts])
    out = [None]*len(bases)
    for q, b in sorted(np.ndindex(*score.shape), key=lambda k: -score[k]):
        if out[b] is None and not any(o is parts[q] for o in out if o is not None): out[b] = parts[q]
    return [o if o is not None else np.zeros(0, int) for o in out]


# ---------------------------------------------------------------------------------------------
# The figure: Step5's, its lower rows the installations as the station sees them.

def panel(model, support, X, P, faces, axes, joints, size=620):
    """One installation at the station, as step5/render_joined.use draws a task: the support at P X^-1, the object at
    P (pink its faces), the KUKA at the joints of its central face and the needle along that face's axis."""
    from whole_task_2.step1.reach_render import contact
    from whole_task_2.step1.reach_video import STATION_VIEW, Scene, kuka as work_arm
    from whole_task_2.step3.render_tool_membranes import image
    from whole_task_2.step3.render_independent import BLUE
    from whole_task_2.step5.video.prepare import matrices
    world = moved_mesh(support, P@np.linalg.inv(X)); faces = np.asarray(faces)
    f = contact(model.source, faces, P); j = int(np.flatnonzero(faces == f)[0])
    tip = model.source.triangles_center[f]@P[:3, :3].T+P[:3, 3]; axis = P[:3, :3]@axes[j]
    links = matrices(work_arm(model.setup).forward(joints[j][3], geometry=True)[2])
    scene = Scene(world, model.setup, size, size, arm=True)
    try:
        g = scene.r
        bunny = g.Shape(scene.ctx, scene.shader, model.source.vertices, model.source.faces, g.GRAY, dynamic=True, metallic=.03, roughness=.63)
        colors = np.tile(g.GRAY, (len(model.source.faces), 1)); colors[faces] = g.PINK; bunny.paint(colors)
        # From the side: the station's view turned 90 deg about the vertical, the KUKA on the right and the needle coming
        # down from it at its fixed fall (user, 2026-10-28).
        view = np.array([-STATION_VIEW[1], STATION_VIEW[0], STATION_VIEW[2]])
        blue = np.tile(BLUE, (len(world.faces), 1)); scene.view = view/np.linalg.norm(view)
        top = tip-axis*(model.setup['needle_length_m']+.07)
        scene.fit(np.vstack([world.vertices, moved_mesh(model.source, P).vertices, tip, top]), 1.12)
        return image(scene, blue, [(tip, axis, False)], links, [(bunny, P)])
    finally: scene.close()


def figure(object_name, result=None, size=620, gap=40, columns=8):
    """Step5's figure of the result again (output/<run>/step5/<result>.png): on top the seats and the support alone
    (step5/combined.overview), below every installation of Step6 at the station, area by area -- the needle always the
    station's, the KUKA at its solved joints.  Drawn from the saved data alone."""
    from PIL import Image
    from whole_task_2.step1.common import DEFAULT_OUTPUT
    from whole_task_2.step4.combine import ExitModel
    from whole_task_2.step5.combined import overview
    c = context(object_name); records = c.base.records
    report = json.loads((data(object_name)/FOLDER/'report.json').read_text()); result = result or report['result']
    report5 = json.loads((data(object_name)/STEP5/'report.json').read_text())
    row = next(r for r in report5['results'] if r['name'] == result)
    group = next(g for g in distinct(object_name) if g['file'] == row['step4_file'])
    support = read_mesh(data(object_name)/STEP5/f'{result}.npz'); model = ExitModel(c, group['slots'])
    seat_of = {records[s_['members'][0]]['id']: np.asarray(s_['X'], float) for s_ in group['slots']}
    seat_of = lambda name, seats_=seat_of: seats_[name.split('@')[0]]          # an area's later bases: <id>@<turn>
    with np.load(data(object_name)/FOLDER/f'{result}_installations.npz') as z:
        panels = [panel(c.base, support, seat_of(a['area']), z[f"{i['name']}_world_from_object"], z[f"{i['name']}_faces"],
                        z[f"{i['name']}_axes"], z[f"{i['name']}_joints"], size) for a in report['areas'] for i in a['installations']]
    rows = [overview(model, group, support, model.layout(support), size)]+[panels[k:k+columns] for k in range(0, len(panels), columns)]
    width = max(len(r_) for r_ in rows)
    sheet = Image.new('RGB', (width*(size+gap)+gap, len(rows)*(size+gap)+gap), 'white')
    for r_, line in enumerate(rows):
        for k, image in enumerate(line): sheet.paste(image, (gap+k*(size+gap), gap+r_*(size+gap)))
    path = DEFAULT_OUTPUT/object_name/'step5'/f'{result}.png'; path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)
    print('STEP6 FIGURE', path, len(panels), 'installations', flush=True)
    return path


def run(object_name, result=None, workers=22, policy=POLICY):
    began = time.monotonic(); c = context(object_name); records = c.base.records
    plan = json.loads((data(object_name)/'step1_reach/plan.json').read_text())
    if not plan.get('robot_free'): print('STEP6 SKIPPED: the plan already places the robot', flush=True); return None
    report5 = json.loads((data(object_name)/STEP5/'report.json').read_text())
    # The best of Step5's results that passed (its results are best first), unless one is named.
    row = next(r for r in report5['results'] if r['status'] == 'passed' and result in (None, r['name'])); result = row['name']
    group = next(g for g in distinct(object_name) if g['file'] == row['step4_file'])
    support = read_mesh(data(object_name)/STEP5/f'{result}.npz'); mesh_ = (support.vertices, support.faces)
    step2 = json.loads((data(object_name)/'step2_seats/report.json').read_text())
    x = S.prepare(object_name, ROOT/step2['big_task'], ROOT/step2['station']); area = x['mesh'].area_faces
    pool = S.workers_pool(object_name, x['shape'], x['s'], x['config'], workers); rows, store = [], {}
    try:
        by_task = {}
        for slot in group['slots']: by_task.setdefault(slot['members'][0], []).append(slot)
        for i, slots in by_task.items():
            r = records[i]; X = np.asarray(slots[0]['X'], float); faces = np.asarray(r['working_face_ids'])
            names = [r['id'] if n == 0 else f"{r['id']}@{s_['angles'][0]:g}" for n, s_ in enumerate(slots)]
            print('STEP6 AREA', r['id'], len(faces), 'faces on', len(slots), 'base(s)', flush=True)
            # The area's bases: its certified pose, turned about the needle when combine's Rotational reuse turned it.
            bases = [np.asarray(c.record(i, float(s_['angles'][0]))[0]['pose'] if s_['angles'][0] else r['pose'])[:3, :3] for s_ in slots]
            parts = shares(x, faces, bases, policy) if len(slots) > 1 else [faces]
            made, todo = [], []
            for R, part in zip(bases, parts):
                m, rest = install(pool, x, part, R, X, mesh_, policy); made.append(m); todo.append(rest)
            left = np.concatenate(todo).astype(int)
            if len(slots) > 1:                              # what one base left, the others try
                for b, R in enumerate(bases):
                    if not len(left): break
                    m, left = install(pool, x, left, R, X, mesh_, policy); made[b] += m
            done = float(area[np.setdiff1d(faces, left)].sum()/area[faces].sum())
            for b, (name_, m_) in enumerate(zip(names, made)):
                for n, m in enumerate(m_):
                    name = f"{name_}_{n+1}"
                    store.update({f'{name}_faces': m['faces'], f'{name}_axes': m['axes'], f'{name}_joints': m['joints'],
                                  f'{name}_world_from_object': m['world_from_object'],
                                  f'{name}_world_from_support': m['world_from_object']@np.linalg.inv(X)})
                rows.append(dict(area=name_, base_turn_deg=float(slots[b]['angles'][0]), faces=len(parts[b]),
                                 installations=[dict(name=f"{name_}_{n+1}", turn_deg=m['turn_deg'], faces=len(m['faces']),
                                                     held_sample_share=m['held_sample_share']) for n, m in enumerate(m_)],
                                 done_fraction=done, faces_left=left.tolist() if b == len(slots)-1 else []))
                print('STEP6 AREA', name_, len(m_), 'installations,', round(100*done, 1), '% of its faces', flush=True)
    finally:
        pool.shutdown(cancel_futures=True)
    out = data(object_name)/FOLDER; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/f'{result}_installations.npz', **store)
    target = float(area[x['target']].sum()); left = np.concatenate([r['faces_left'] for r in rows]).astype(int) if rows else np.zeros(0, int)
    report = dict(status='passed', object=object_name, result=result, rule=__doc__.split('\n\n')[1:3], policy=policy, areas=rows,
                  installation_count=sum(len(r['installations']) for r in rows), area_count=len(rows),
                  done_fraction_of_target=1-float(area[left].sum()/target), seconds=time.monotonic()-began)
    write_json(out/'report.json', report); figure(object_name, result)
    print('STEP6 DONE', report['area_count'], 'areas,', report['installation_count'], 'installations,',
          round(100*report['done_fraction_of_target'], 2), '% of the target done,', round(report['seconds']), 's', flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', required=True); p.add_argument('--result', default=None); p.add_argument('--workers', type=int, default=22)
    p.add_argument('--figure-only', action='store_true', help="draw Step5's figure again from the saved installations")
    a = p.parse_args(); figure(a.object, a.result) if a.figure_only else run(a.object, a.result, a.workers)
