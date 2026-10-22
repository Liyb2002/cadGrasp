"""Step5: every distinct support of step4/combine.py with its feet, certified for every task, and its figure.

The placing arm installs the support for each task so that the task faces the work arm at its certified
pose, and the support rests on the table.  Feet wrap every task's pressure hull (as Co-optimize's Step5):

  for every task, in its installation, a band FOOT['width'] wide and FOOT['height'] high on the table
  around its pressure hull, its outer edge FOOT['outward'][0] outside the hull's, kept only in the legal
  space -- above every installation's floor, out of every task's body, keep-out and exit sweep (Step4's own
  forbidden region), so a foot never blocks another task;
  every legal piece of the band joins the support: directly, or by a round link FOOT['link'] in radius from
  the nearest point of the support, clipped to the legal space;
  where another task's keep-out crosses the band (a seat's KUKA reaching past another installation's
  floor), the band's legal part may not enclose the hull: the band then moves out, through the offsets
  FOOT['outward'], to the first whose legal part, joined, makes the ground faces enclose the hull.

The whole membrane and every accepted piece stay; nothing is pruned.  Every task is then audited in full on
the support with its feet (32,768 paired samples and the continuous outer targets, the continuous exit of
every source face, body and keep-out, the KUKA at every stored joint set), and in every installation the
actual ground faces must enclose the task's pressure hull.  Validation rebuilds every support from Step4's
membrane and audits it again.

One figure per support, output/<object>/step5/<name>.png, no text: the top row, the support with the bunny
in each seat (pink: the areas of the seat's tasks), then the support alone; the bottom row, every task at
the station, the support as the placing arm installs it and the KUKA holding the needle on the task's
central certified face.
"""
import argparse
import json
import time
import numpy as np
import trimesh
import manifold3d as md
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task_2.operations.floors import clip, planes
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.membranes import positive_components
from whole_task_2.step4.ground_oracle import pressure_rows, ground_enclosure
from whole_task_2.step4.seats import context, read_mesh
from whole_task_2.step4.merge import move
from whole_task_2.step4.combine import FOLDER as COMBINE, ExitModel, largest, plain, seats, solids, obb

FOLDER = 'step5_combined'
FOOT = dict(width=.008, height=.005,                 # Co-optimize's band: 8 mm wide, 5 mm high,
            outward=(.004, .008, .012, .016, .024, .032, .048, .064),   # its outer edge this far outside the hull
            link=.008,                               # round links, radius
            links=12,                                # links tried per piece, from spread anchors
            spread=.01,                              # the anchors of tried links at least this far apart
            fragment=2e-8,                           # legal pieces below this volume (m^3) are not grown
            padding=.03)                             # the legal space: a box this much around the widest bands


def distinct(object_name):
    """Step4's final supports, one per distinct support, best first, with the samples that ended on it.
    Samples that copied one parent and then made the same joins hold the same support up to rounding (the
    multithreaded Booleans differ at ~1e-10 m^3 across processes): one row."""
    report = json.loads((ROOT/'whole_task_2/data'/object_name/COMBINE/'report.json').read_text()); out = {}
    for g in report['finals']:
        key = (g['chain'][0]['parents'][0] if g['chain'] else None,
               tuple((s['parents'][1], s['operation'], s['seat']) for s in g['chain']),
               round(g['volume_cm3'], 1), round(g['bbx_m3']*1e6))
        row = out.setdefault(key, dict(g, samples=[], files=[]))
        row['samples'].append(g['sample'])
        if g['file'] not in row['files']: row['files'].append(g['file'])
    return list(out.values())


def count(value):
    return len(positive_components(value))


def rim(row, outward, foot=FOOT):
    """The band around a task's pressure hull on its table, its outer edge outward outside the hull's,
    support frame."""
    hull = Polygon(row['hull'])
    outer = orient(hull.buffer(outward, join_style=2), sign=1.)
    inner = orient(hull.buffer(outward-foot['width'], join_style=2), sign=1.)
    contours = [np.asarray(outer.exterior.coords)[:-1]/G.SCALE]
    if not inner.is_empty: contours.append(np.asarray(inner.exterior.coords)[:-1][::-1]/G.SCALE)
    return move(row['material'], md.CrossSection(contours, md.FillRule.NonZero).extrude(foot['height']/G.SCALE))


def capsule(a, b, radius):
    ball = md.Manifold.sphere(radius/G.SCALE, 24)
    return G.union([ball.translate(tuple(a/G.SCALE)), ball.translate(tuple(b/G.SCALE))]).hull()


def anchors(value):
    m = G.mesh(value); return np.vstack([m.vertices, m.triangles_center])


def link(stock, piece, legal, foot=FOOT):
    """stock + piece joined by a round link through the legal space, or None.  The links start from piece
    points spread at least foot['spread'] apart, nearest to the stock first."""
    a = anchors(piece); b = anchors(stock); d, j = cKDTree(b).query(a); tried = []
    for k in np.argsort(d):
        if len(tried) == foot['links']: break
        if any(np.linalg.norm(a[k]-p) < foot['spread'] for p in tried): continue
        tried.append(a[k])
        parts = [p for p in positive_components(capsule(a[k], b[j[k]], foot['link']) ^ legal)
                 if count(stock+p) == 1 and count(piece+p) == 1]
        if not parts: continue
        joined = stock+piece+G.union(parts)
        if count(joined) == 1: return joined, dict(length_m=float(d[k]), path_m=[a[k].tolist(), b[j[k]].tolist()])
    return None


def grow(stock, band, legal, tolerance, foot=FOOT):
    """stock with every legal piece of band joined to it, and the record of the pieces."""
    record = []
    for piece in positive_components(band ^ legal):
        v = G.volume(piece)
        if v < foot['fragment'] or G.volume(piece-stock) <= tolerance: continue
        joined, how = stock+piece, 'touches'
        if count(joined) != 1:
            found = link(stock, piece, legal, foot)
            if found is None: record.append(dict(piece_cm3=v*1e6, accepted=False, reason='no legal link joins it')); continue
            joined, how = found
        stock = joined; record.append(dict(piece_cm3=v*1e6, accepted=True, link=how))
    return stock, record


def wrap(c, group, foot=FOOT):
    """The support with its feet: (support, model, layout, record of every task's band)."""
    membrane = read_mesh(ROOT/group['file']); model = ExitModel(c, group['slots']); q = model.layout(membrane)
    rows = pressure_rows(model, q); pad = max(foot['outward'])+foot['padding']
    points = np.vstack([membrane.vertices, *[r['ground_material'] for r in rows]])
    legal = clip(G.box(points.min(0)-pad, points.max(0)+pad), planes(q.material))-solids(c.object_name, group)['F']
    stock, share = largest(G.solid(membrane))
    if share < 1-1e-9: raise ValueError('Step4 membrane is not one piece')
    record = []
    for r in rows:
        for d in foot['outward']:
            grown, pieces = grow(stock, rim(r, d, foot), legal, c.base.tolerance, foot)
            if d == foot['outward'][0]: first = grown, pieces
            if ground_enclosure(G.mesh(grown), [r])[0]['passed']: break
        else: (grown, pieces), d = first, None          # no band encloses: the hull-edge band; certify reports it
        stock = grown; record.append(dict(task=r['id'], outward_m=d, pieces=pieces))
    support = G.mesh(stock.simplify(0.))
    if G.volume(G.solid(membrane)-G.solid(support)) > c.base.tolerance: raise ValueError('the feet lost membrane')
    return support, model, model.layout(support), record


def certify(model, q, support):
    """Full audit of every task on the support, and its stance on the table in every installation."""
    candidate = dict(layout=q, support=support, tasks=model.prepare(q), construction={})
    proof = model.audit(candidate)
    ground = ground_enclosure(support, pressure_rows(model, q, candidate['tasks']))
    if not all(r['passed'] for r in ground): raise ValueError('A pressure hull leaves the ground faces: '+str(
        [(r['id'], r['pressure_hull_gap_m'], r['minimum_world_z_m']) for r in ground if not r['passed']]))
    return proof, ground


def inputs(object_name):
    data = ROOT/'whole_task_2/data'/object_name
    files = [data/COMBINE/'report.json', data/'step4_keepout/report.json', data/'step3_tool_membranes/report.json',
             *[ROOT/'whole_task_2'/p for p in ('step4/combine.py', 'step4/merge.py', 'step4/seats.py', 'step4/keepout.py',
                                             'step4/ground_oracle.py', 'step5/combined.py')]]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def run(object_name='B', output_root=DEFAULT_OUTPUT):
    began = time.monotonic(); c = context(object_name); folder = ROOT/'whole_task_2/data'/object_name/FOLDER
    folder.mkdir(parents=True, exist_ok=True); rows = []
    for n, group in enumerate(distinct(object_name)):
        name = f'r{n:02d}'; t0 = time.monotonic()
        row = dict(name=name, samples=group['samples'], step4_file=group['file'], step4_files=group['files'],
                   membrane_cm3=group['volume_cm3'], bbx_m3=group['bbx_m3'], seats=[len(ks) for _, ks in seats(group)],
                   joins=[dict(task=s['parents'][1], operation=s['operation'], seat=s['seat'], bbx_ratio=s['bbx_ratio'])
                          for s in group['chain']])
        try:
            support, model, q, pieces = wrap(c, group); row['pieces'] = pieces; proof, ground = certify(model, q, support)
        except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as ex:
            rows.append(dict(row, status='failed', reason=str(ex)[:300], seconds=time.monotonic()-t0))
            print('STEP5', name, 'FAILED', str(ex)[:160], flush=True); continue
        np.savez_compressed(folder/f'{name}.npz', vertices=support.vertices, faces=support.faces)
        (folder/f'{name}.obj').write_text(trimesh.exchange.obj.export_obj(support, digits=17, include_normals=False))
        rows.append(dict(row, status='passed', volume_cm3=float(support.volume*1e6),
                         feet_cm3=float(support.volume*1e6)-group['volume_cm3'], support_bbx_m3=obb(support.vertices),
                         bounds_m=support.bounds.tolist(), support_sha256=sha256(folder/f'{name}.npz'),
                         tasks=[dict(id=t['id'], samples=t['force']['passed_original_sample_count'],
                                     outer=t['force']['passed_continuous_outer_target_count'], arm_faces_hit=len(t['arm']['faces_hit']),
                                     exit_length_m=t['exit_length_m']) for t in proof['tasks']],
                         ground=[dict(id=r['id'], passed=r['passed'], pressure_hull_gap_m=r['pressure_hull_gap_m']) for r in ground],
                         seconds=time.monotonic()-t0))
        print('STEP5', name, 'PASS', round(support.volume*1e6, 1), 'cm3,', round(rows[-1]['feet_cm3'], 1), 'cm3 of feet,',
              round(time.monotonic()-t0), 's', flush=True)
        render(group, model, q, support, output_root/object_name/'step5'/f'{name}.png')
    report = dict(object=object_name, foot=FOOT, results=rows, input_sha256=inputs(object_name), seconds=time.monotonic()-began)
    write_json(folder/'report.json', plain(report))
    validate(object_name)
    return report


def validate(object_name='B'):
    """Disk replay: same inputs; every passed support rebuilt from Step4's membrane equals the saved one and is
    audited again."""
    c = context(object_name); folder = ROOT/'whole_task_2/data'/object_name/FOLDER
    report = json.loads((folder/'report.json').read_text())
    if report['input_sha256'] != inputs(object_name): raise ValueError('Step5 inputs changed since the build')
    groups = {g['file']: g for g in distinct(object_name)}; rows = []
    for row in report['results']:
        if row['status'] != 'passed': continue
        path = folder/f"{row['name']}.npz"
        if sha256(path) != row['support_sha256']: raise ValueError(row['name']+': changed support file')
        rebuilt, model, q, _ = wrap(c, groups[row['step4_file']]); support = read_mesh(path)
        d = G.volume(G.solid(support)-G.solid(rebuilt))+G.volume(G.solid(rebuilt)-G.solid(support))
        if d > c.base.tolerance: raise ValueError(f"{row['name']}: saved support differs from its rebuild: {d} m^3")
        proof, ground = certify(model, q, support)
        rows.append(dict(name=row['name'], rebuilt_symmetric_difference_m3=d,
                         samples=[t['force']['passed_original_sample_count'] for t in proof['tasks']]))
        print('STEP5', row['name'], 'DISK REPLAY PASS', flush=True)
    write_json(folder/'validation.json', plain(dict(status='passed', report_sha256=sha256(folder/'report.json'), results=rows)))
    return rows


# ---------------------------------------------------------------------------------------------
# Figures.

def overview(model, group, support, q, size, view=(.55, -1., .55)):
    """One image per seat (the support and the bunny in that seat, pink the areas of its tasks), then the
    support alone.  Drawn as the first task's installation stands it (that resting part down); one camera."""
    from whole_task_2.step1.reach_video import Scene
    from whole_task_2.step3.render_tool_membranes import image
    from whole_task_2.step3.render_independent import BLUE
    from whole_task_2.operations.geometry import moved_mesh
    W = np.linalg.inv(q.material[0]); support = moved_mesh(support, W)
    scene = Scene(support, model.setup, size, size)
    try:
        g = scene.r; blue = np.tile(BLUE, (len(support.faces), 1)); scene.view = np.asarray(view)/np.linalg.norm(view)
        rows = [(W@X, ks) for X, ks in seats(group)]
        scene.fit(np.vstack([support.vertices, *[moved_mesh(model.source, X).vertices for X, _ in rows]]), 1.06)
        images = []
        for X, ks in rows:
            bunny = g.Shape(scene.ctx, scene.shader, model.source.vertices, model.source.faces, g.GRAY, dynamic=True, metallic=.03, roughness=.63)
            colors = np.tile(g.GRAY, (len(model.source.faces), 1))
            for k in ks: colors[model.records[k]['working_face_ids']] = g.PINK
            bunny.paint(colors); images.append(image(scene, blue, [], None, [(bunny, X)]))
        return images+[image(scene, blue, [], None, [])]
    finally: scene.close()


def render(group, model, q, support, path, size=620, gap=40):
    """One figure: the seats and the support alone, above every task's use of it."""
    from .render_joined import use
    top = overview(model, group, support, q, size); bottom = [use(model, q, support, k, size) for k in range(len(model.records))]
    columns = max(len(top), len(bottom))
    sheet = Image.new('RGB', (columns*(size+gap)+gap, 2*(size+gap)+gap), 'white')
    for r, panels in enumerate((top, bottom)):
        for k, panel in enumerate(panels): sheet.paste(panel, (gap+k*(size+gap), gap+r*(size+gap)))
    path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)
    print('STEP5 FIGURE', path, flush=True)


def figures(object_name='B', output_root=DEFAULT_OUTPUT):
    """The figure of every saved support again, from disk."""
    c = context(object_name); folder = ROOT/'whole_task_2/data'/object_name/FOLDER
    report = json.loads((folder/'report.json').read_text()); groups = {g['file']: g for g in distinct(object_name)}
    for row in report['results']:
        if row['status'] != 'passed': continue
        g = groups[row['step4_file']]; support = read_mesh(folder/f"{row['name']}.npz")
        model = ExitModel(c, g['slots'])
        render(g, model, model.layout(support), support, output_root/object_name/'step5'/f"{row['name']}.png")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', default='B'); p.add_argument('--validate-only', action='store_true')
    p.add_argument('--figures-only', action='store_true')
    a = p.parse_args()
    validate(a.object) if a.validate_only else figures(a.object) if a.figures_only else run(a.object)
