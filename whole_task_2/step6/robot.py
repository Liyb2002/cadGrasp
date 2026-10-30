"""Step6: every direction task split into the installations the work arm needs -- at once, nothing checked again.

Step4 certified every direction task (a Step2 area) at its pose, whole: its needle, its loads, its keep-out and the
support standing; Step5 finished the support.  But the work arm moves the needle only so far from where an installation
puts it: policy['reach_m'] (50 mm, user 2026-10-29) about the installation's centre.  So a direction task is done in
installations, the support moved on the table between them:

  1. An area on several bases (combine turned it in its own seat: Rotational reuse) is split among them first: its
     faces in one part per base, each part to the base whose turns reach the most of it (shares).
  2. On a base, the faces in windows: the centre is the face whose ball of radius policy['reach_m'] holds the most of
     the faces left, the faces within it one installation, the rest the next (window).  An installation keeps the
     base and the turn Step4 certified and moves the support on the table, its centre over the work point
     (step2/split.placement), the support's lowest point (its feet) on the table.

Nothing is checked here (user, 2026-10-29: "不需要验证。验证是step4做过的了"): the needle, the loads and the support
are Step4's and Step5's.  The figure: a direction task per row, its installations along the row.

  python -m whole_task_2.step6.robot --object B+big_01_v2
"""
import argparse
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, write_json
from whole_task_2.step2 import split as S
from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.step4.seats import context, read_mesh
from whole_task_2.step5.combined import distinct, FOLDER as STEP5

FOLDER = 'step6_robot'
POLICY = dict(turns=24,             # the needle's turns about the vertical, for splitting an area among its bases
              reach_m=.05)          # the work arm moves the needle at most this far from an installation's centre (m)


def data(run):
    return ROOT/'whole_task_2/data'/run


def turn_of(R, a):
    """(needle in the object frame, the turn about the needle for split.placement, deg) of rotation R (world <- object)."""
    v = R.T@a; return v, float(np.degrees(Rotation.from_matrix(R@S.toward(v, a).T).as_rotvec()@a))


def window(mesh, faces, r):
    """(the centre, the faces within r of it, the rest): the centre the face whose ball of radius r holds the most of
    the faces' area."""
    p = mesh.triangles_center[faces]; w = mesh.area_faces[faces]; balls = cKDTree(p).query_ball_point(p, r)
    best = max(range(len(faces)), key=lambda j: float(w[balls[j]].sum())); near = np.zeros(len(faces), bool); near[balls[best]] = True
    return p[best], faces[near], faces[~near]


def install(x, faces, R, X, support, policy):
    """The installations of an area's part on one base (rotation R, world <- object; seat X): [dict(centre (object
    frame), faces, world_from_object)], window by window."""
    s = x['s']; v, psi = turn_of(R, S.unit(s['working_direction_world'])); left = np.asarray(sorted(map(int, faces)), int); out = []
    while len(left):
        centre, mine, left = window(x['mesh'], left, policy['reach_m'])
        P = S.placement(x['mesh'], v, centre, psi, s)
        # The support, not the object, stands on the table: its lowest point (a foot) there, the object above it.
        W = P@np.linalg.inv(X); P[2, 3] += s['table_height_m']-float((support.vertices@W[:3, :3].T+W[:3, 3])[:, 2].min())
        out.append(dict(centre=centre, faces=mine, world_from_object=P))
    return out


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
# The figure: a direction task per row, its installations along the row.

def panel(model, support, X, P, faces, centre, size=620):
    """One installation at the station: the support at P X^-1, the object at P (pink its faces), the station's needle
    at the installation's centre."""
    from whole_task_2.step1.reach_video import STATION_VIEW, Scene
    from whole_task_2.step3.render_tool_membranes import image
    from whole_task_2.step3.render_independent import BLUE
    world = moved_mesh(support, P@np.linalg.inv(X)); a = S.unit(model.setup['working_direction_world'])
    tip = P[:3, :3]@np.asarray(centre)+P[:3, 3]; top = tip-a*(model.setup['needle_length_m']+.07)
    scene = Scene(world, model.setup, size, size)
    try:
        g = scene.r
        bunny = g.Shape(scene.ctx, scene.shader, model.source.vertices, model.source.faces, g.GRAY, dynamic=True, metallic=.03, roughness=.63)
        colors = np.tile(g.GRAY, (len(model.source.faces), 1)); colors[np.asarray(faces, int)] = g.PINK; bunny.paint(colors)
        # From the side: the station's view turned 90 deg about the vertical, the needle coming down from the right.
        view = np.array([-STATION_VIEW[1], STATION_VIEW[0], STATION_VIEW[2]]); scene.view = view/np.linalg.norm(view)
        scene.fit(np.vstack([world.vertices, moved_mesh(model.source, P).vertices, tip, top]), 1.12)
        return image(scene, np.tile(BLUE, (len(world.faces), 1)), [(tip, a, False)], None, [(bunny, P)])
    finally: scene.close()


def figure(object_name, result=None, size=620, gap=40, label=260):
    """Step6's figure (output/<run>/step6/<result>.png): a direction task x its real tasks -- one row per area on its
    base, along the row every installation (user, 2026-10-29: "在这个pose上，我们可能要挪动支撑几次才能完成").  Drawn
    from the saved data alone."""
    from PIL import Image, ImageDraw, ImageFont
    from whole_task_2.step1.common import DEFAULT_OUTPUT
    c = context(object_name); records = c.base.records
    report = json.loads((data(object_name)/FOLDER/'report.json').read_text()); result = result or report['result']
    report5 = json.loads((data(object_name)/STEP5/'report.json').read_text())
    row = next(r for r in report5['results'] if r['name'] == result)
    group = next(g for g in distinct(object_name) if g['file'] == row['step4_file'])
    support = read_mesh(data(object_name)/STEP5/f'{result}.npz')
    seat_of = {records[s_['members'][0]]['id']: np.asarray(s_['X'], float) for s_ in group['slots']}
    seat_of = lambda name, seats_=seat_of: seats_[name.split('@')[0]]          # an area's later bases: <id>@<turn>
    with np.load(data(object_name)/FOLDER/f'{result}_installations.npz') as z:
        rows = [(a['area'], [panel(c.base, support, seat_of(a['area']), z[f"{i['name']}_world_from_object"], z[f"{i['name']}_faces"],
                                   z[f"{i['name']}_centre"], size) for i in a['installations']]) for a in report['areas']]
    width = max([len(p) for _, p in rows]+[1])
    sheet = Image.new('RGB', (label+width*(size+gap)+gap, len(rows)*(size+gap)+gap), 'white'); ink = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 56)
    for r_, (name, line) in enumerate(rows):
        y = gap+r_*(size+gap); ink.text((gap, y+size//2), name, fill=(60, 60, 60), font=font, anchor='lm')
        for k, image in enumerate(line): sheet.paste(image, (label+k*(size+gap), y))
    path = DEFAULT_OUTPUT/object_name/'step6'/f'{result}.png'; path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)
    print('STEP6 FIGURE', path, len(rows), 'direction tasks,', sum(len(p) for _, p in rows), 'installations', flush=True)
    return path


def run(object_name, result=None, policy=POLICY):
    began = time.monotonic(); c = context(object_name); records = c.base.records
    plan = json.loads((data(object_name)/'step1_reach/plan.json').read_text())
    if not plan.get('robot_free'): print('STEP6 SKIPPED: the plan already places the robot', flush=True); return None
    report5 = json.loads((data(object_name)/STEP5/'report.json').read_text())
    # The best of Step5's results that passed (its results are best first), unless one is named.
    row = next(r for r in report5['results'] if r['status'] == 'passed' and result in (None, r['name'])); result = row['name']
    group = next(g for g in distinct(object_name) if g['file'] == row['step4_file'])
    support = read_mesh(data(object_name)/STEP5/f'{result}.npz')
    step2 = json.loads((data(object_name)/'step2_seats/report.json').read_text())
    x = S.prepare(object_name, ROOT/step2['big_task'], ROOT/step2['station']); area = x['mesh'].area_faces
    rows, store, by_task = [], {}, {}
    for slot in group['slots']: by_task.setdefault(slot['members'][0], []).append(slot)
    for i, slots in by_task.items():
        # The area's faces and its bases: its certified pose, turned about the needle when combine's Rotational reuse
        # turned it.
        r = records[i]; X = np.asarray(slots[0]['X'], float); faces = np.asarray(r['working_face_ids'], int)
        names = [r['id'] if n == 0 else f"{r['id']}@{s_['angles'][0]:g}" for n, s_ in enumerate(slots)]
        bases = [np.asarray(c.record(i, float(s_['angles'][0]))[0]['pose'] if s_['angles'][0] else r['pose'])[:3, :3] for s_ in slots]
        parts = shares(x, faces, bases, policy) if len(slots) > 1 else [faces]
        for b, (name_, R, part) in enumerate(zip(names, bases, parts)):
            made = install(x, part, R, X, support, policy)
            for n, m in enumerate(made):
                name = f"{name_}_{n+1}"
                store.update({f'{name}_faces': m['faces'], f'{name}_centre': m['centre'], f'{name}_world_from_object': m['world_from_object'],
                              f'{name}_world_from_support': m['world_from_object']@np.linalg.inv(X)})
            rows.append(dict(area=name_, base_turn_deg=float(slots[b]['angles'][0]), faces=len(part),
                             installations=[dict(name=f"{name_}_{n+1}", faces=len(m['faces'])) for n, m in enumerate(made)]))
            print('STEP6 AREA', name_, len(part), 'faces,', len(made), 'installations', flush=True)
    out = data(object_name)/FOLDER; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/f'{result}_installations.npz', **store)
    done = np.concatenate([store[k] for k in store if k.endswith('_faces')]) if store else np.zeros(0, int)
    report = dict(status='passed', object=object_name, result=result, rule=__doc__.split('\n\n')[1:4], policy=policy, areas=rows,
                  installation_count=sum(len(r_['installations']) for r_ in rows), area_count=len(rows),
                  done_fraction_of_target=float(area[np.intersect1d(done, np.flatnonzero(x['target']))].sum()/area[x['target']].sum()),
                  seconds=time.monotonic()-began)
    write_json(out/'report.json', report); figure(object_name, result)
    print('STEP6 DONE', report['area_count'], 'areas,', report['installation_count'], 'installations,',
          round(100*report['done_fraction_of_target'], 2), '% of the target in installations,', round(report['seconds']), 's', flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', required=True); p.add_argument('--result', default=None); p.add_argument('--workers', type=int, default=22)
    p.add_argument('--figure-only', action='store_true', help="draw Step6's figure again from the saved installations")
    a = p.parse_args(); figure(a.object, a.result) if a.figure_only else run(a.object, a.result)
