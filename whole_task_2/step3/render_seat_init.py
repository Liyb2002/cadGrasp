"""Figures of a seat-first plan (step3/seat_init.py), no text.

seats.png: one row per seat (exit).  Every panel in a row is the same support -- the seat's, its exit sweep the
same -- with the object in it at one task's pose, as the robot sees it: the support blue, the task's work area pink
(with the faces it took over uncertified), the needle and head on its middle certified face.  Along a row only the pose changes, to bring another work area to the arm.

seats_parting.png: the big task cut along the parting line (two seats only): the object upright (mesh +y up), the
halves of exits d and -d in two colours, the line dashed, what no task works red; seen across d from two sides,
then the halves pulled apart along d.

seats_connected.png: every task's support on one base plate, as Step3's initialization.png
(render_tool_membranes.render_connected): the tasks of a seat are copies of one support at their own poses.  Top:
the objects in their seats with the needles and Step1's gold cones; bottom: the support alone.  For the picture the
fixtures stand in a row (tool_membranes.row) without widening for the exits or the arm and without the final recut:
it is not an audited support.
"""
import json
import shutil
import tempfile
from pathlib import Path

import manifold3d as md
import numpy as np
import trimesh
from PIL import Image

from robots_config.constraints import load_config
from whole_task_2.common.wrenches import resolve_policy
from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.operations.state import Layout
from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_render import contact, picture, tiles
from whole_task_2.step1.reach_video import Scene, tool_frame
from whole_task_2.step1.workspace import onto
from . import independent_geometry as G
from . import render_tool_membranes
from .render_independent import BLUE
from .render_tool_membranes import render_connected
from .tool_keepout import disc
from .tool_membranes import POLICIES, ToolMembraneModel, row_axis


def framed(mesh, z, name, V, s):
    """One task shifted by the work point (the scene's lights are about the origin): its pose, work faces, the
    needle's tip and axis on its middle face, and the points a picture of it shows."""
    point = np.asarray(s['work_point_m']); P = z[f'{name}_pose'].copy(); P[:3, 3] -= point
    faces = z[f'{name}_face_ids']; f = contact(mesh, faces, P); k = int(np.flatnonzero(faces == f)[0])
    tip = mesh.triangles_center[f]@P[:3, :3].T+P[:3, 3]; axis = P[:3, :3]@z[f'{name}_axes'][k]
    points = np.vstack([V@P[:3, :3].T+P[:3, 3], mesh.vertices@P[:3, :3].T+P[:3, 3], [tip, tip-axis*(s['needle_length_m']+.07)]])
    return P, np.r_[faces, z[f'{name}_absorbed_ids']].astype(int), tip, axis, points


def extent(scene, points):
    """Half the larger side of the points seen by the scene's camera."""
    right = np.cross([0, 0, 1], scene.view); right /= np.linalg.norm(right); up = np.cross(scene.view, right)
    return max(np.ptp(points@right), np.ptp(points@up))/2, right, up


def task_panel(scene, mesh, task, support, R):
    """One task with its seat's support, every panel at the same scale (half side R)."""
    g = scene.r; P, faces, tip, axis, points = task
    colors = np.tile(g.GRAY, (len(mesh.faces), 1)); colors[faces] = g.PINK
    shapes = [(support, P)]+[(part, tool_frame(tip, axis)) for part in scene.tools[False]]
    _, right, up = extent(scene, points); c = (points.min(0)+points.max(0))/2
    scene.fit(np.vstack([points, c+R*right, c-R*right, c+R*up, c-R*up]), 1.04)
    return picture(scene, P, colors, [], fit=False, shapes=shapes)


def rows(report, z, sup, mesh, out, size=400):
    s = report['setup']; a = np.asarray(s['working_direction_world'], float); h = -a.copy(); h[2] = 0; h /= np.linalg.norm(h)
    view = h+.6*np.cross([0., 0., 1.], h)+.55*np.array([0., 0., 1.])
    columns = max(len(seat['tasks']) for seat in report['seats']); images = []
    scene = Scene(mesh, s, size, size)
    try:
        scene.view = view/np.linalg.norm(view); g = scene.r
        tasks = {t['id']: framed(mesh, z, t['id'], sup[seat['id']+'_vertices'], s) for seat in report['seats'] for t in seat['tasks']}
        R = max(extent(scene, task[-1])[0] for task in tasks.values())
        for seat in report['seats']:
            V, F = sup[seat['id']+'_vertices'], sup[seat['id']+'_faces']
            support = g.Shape(scene.ctx, scene.shader, V, F, BLUE, metallic=.05, roughness=.55)
            row = [task_panel(scene, mesh, tasks[t['id']], support, R) for t in seat['tasks']]
            images += row+[Image.new('RGB', row[0].size, 'white')]*(columns-len(row))
        tiles(images, columns).save(out/'seats.png')
    finally:
        scene.close()
    print('FIGURE', out/'seats.png', flush=True)


HALVES, LEFT, INK = [(.95, .66, .22), (.40, .62, .85)], (.85, .12, .12), (.08, .08, .10)


def dashed(mesh, edges, dash):
    """The parting edges (vertex pairs) as a dashed line: chained, sampled along their length, every other dash kept;
    line pairs a little outside the surface so the depth test keeps them."""
    lift = mesh.vertices+mesh.vertex_normals*1e-3*float(np.linalg.norm(mesh.extents)); nxt = {}
    for u, v in edges: nxt.setdefault(u, []).append(v); nxt.setdefault(v, []).append(u)
    seen = set(); out = []
    for start in list(nxt):
        if start in seen: continue
        chain = [start]; seen.add(start)
        while True:
            step = [w for w in nxt[chain[-1]] if w not in seen]
            if not step: break
            chain.append(step[0]); seen.add(step[0])
        P = lift[chain]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        if s[-1] <= 0: continue
        t = np.arange(0, s[-1], dash/6); Q = np.c_[[np.interp(t, s, P[:, k]) for k in range(3)]].T
        on = (np.floor(t/dash) % 2 == 0)[:-1]
        out += [x for i in np.flatnonzero(on) for x in (Q[i], Q[i+1])]
    return np.asarray(out).reshape(-1, 3)


def parting_figure(report, z, mesh, out, size=560):
    d = np.asarray(report['parting']['direction_object']); A = mesh.face_normals@d > 0; left = z['left']
    edges = mesh.face_adjacency_edges[A[mesh.face_adjacency[:, 0]] != A[mesh.face_adjacency[:, 1]]]
    line = dashed(mesh, edges, .012*float(np.linalg.norm(mesh.extents)))
    R = np.eye(4); R[:3, :3] = onto([0., 1., 0.], [0., 0., 1.]); R[:3, 3] = -R[:3, :3]@mesh.bounds.mean(0)
    dw = R[:3, :3]@d; side = np.cross([0., 0., 1.], dw)
    side = side/np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([.4, -1., 0.])/np.linalg.norm([.4, -1.])
    gap = .35*float(np.ptp(mesh.vertices@d)); halves = []
    for k, part in enumerate((A, ~A)):
        ids = np.flatnonzero(part); sub = mesh.submesh([ids], append=True)
        c = np.tile(HALVES[k], (len(ids), 1)); c[left[ids]] = LEFT; halves.append((sub, c))
    scene = Scene(halves[0][0], report['setup'], size, size); g = scene.r
    try:
        other = g.Shape(scene.ctx, scene.shader, halves[1][0].vertices, halves[1][0].faces, HALVES[1], dynamic=True, metallic=.03, roughness=.63)
        other.paint(halves[1][1]); images = []
        def move(s): M = R.copy(); M[:3, 3] += s*dw; return M
        def ink(points, M): return [(points[i:i+4000]@M[:3, :3].T+M[:3, 3], INK) for i in range(0, len(points), 4000)]
        for view, apart in ((side+[0, 0, .35], 0.), (-side+[0, 0, .35], 0.), (side+[0, 0, .35], gap)):
            scene.view = np.asarray(view)/np.linalg.norm(view); top, low = move(apart/2), move(-apart/2)
            scene.fit(np.vstack([mesh.vertices@top[:3, :3].T+top[:3, 3], mesh.vertices@low[:3, :3].T+low[:3, 3]]), 1.06)
            raw = scene.draw(top, halves[0][1], [], segments=ink(line, top)+(ink(line, low) if apart else []), shapes=[(other, low)])
            images.append(Image.frombytes('RGB', scene.size, raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
        tiles(images, len(images)).save(out/'seats_parting.png')
    finally:
        scene.close()
    print('FIGURE', out/'seats_parting.png', flush=True)


def model_of(report, z, mesh, pairs):
    """Step3's tool membrane model over the (seat, task) pairs (what row() and render_connected read)."""
    obj = report['object']; physical = json.loads((ROOT/'objects'/obj/'meta.json').read_text()); policy = {}
    for p in POLICIES: policy.update(json.loads(p.read_text()))
    policy = resolve_policy(policy, load_config(ROOT/report['station']))
    with np.load(ROOT/'whole_task_2/data'/obj/'step3_mold_plate/fitted_outer.npz') as o: outer = trimesh.Trimesh(o['vertices'], o['faces'], process=False)
    records = [dict(id=t['id'], working_face_ids=np.r_[z[t['id']+'_face_ids'], z[t['id']+'_absorbed_ids']].astype(int).tolist(),
                    certified_face_ids=z[t['id']+'_face_ids'].tolist(),
                    axes=z[t['id']+'_axes'], joints=z[t['id']+'_joints'], pose=z[t['id']+'_pose'], angle_deg=0.)
               for _, t in pairs]
    policy['task_count'] = len(records)
    return ToolMembraneModel(mesh, outer, records, report['setup'], physical, policy, report['run'])


def joined_row(model, q, shells):
    """tool_membranes.connected for the picture: the fixtures in a row along row_axis, row_gap_m apart by their
    membranes and objects alone, each on a foot over its lowest part, on one plate."""
    p = model.policy; h = p['plate_thickness_m']; t = p['mold_outward_displacement_m']
    clear = 1.1*model.setup['needle_radius_m']; e = row_axis(model.setup); across = np.array([-e[1], e[0], 0.])
    left = 0.; pieces = []; prints = []; shifts = []
    for i, shell in enumerate(shells):
        body = moved_mesh(model.source, q.poses[i]).vertices; X = np.r_[shell.vertices@e, body@e]
        shift = (left-X.min())*e+[0., 0., h-shell.bounds[0, 2]]
        placed = shell.copy(); placed.apply_translation(shift); left = X.max()+shift@e+p['row_gap_m']
        top = min(h+t, body[:, 2].min()+shift[2]-clear); V = placed.vertices[placed.vertices[:, 2] <= top]
        if len(V) < 3: top = h+t; V = placed.vertices[placed.vertices[:, 2] <= top]
        dy = -((V@across).min()+(V@across).max())/2; shift += dy*across; placed.apply_translation(dy*across); V = V+dy*across
        pieces += [G.solid(placed), md.Manifold.hull_points(np.vstack([np.c_[V[:, :2], np.zeros(len(V))], np.c_[V[:, :2], np.full(len(V), top)]])/G.SCALE)]
        prints.append(V[:, :2]); shifts.append(shift)
    ring = disc(np.array([0., 0., 1.]), p['plate_edge_margin_m'], 16)[:, :2]
    F = (np.vstack(prints)[:, None]+ring[None]).reshape(-1, 2)
    pieces.append(md.Manifold.hull_points(np.vstack([np.c_[F, np.zeros(len(F))], np.c_[F, np.full(len(F), h)]])/G.SCALE))
    return G.mesh(G.union(pieces)), np.asarray(shifts)


class RoomyScene(Scene):
    """Step1's scene with room for every task's gold cone (its line buffer holds about 19)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lines = self.ctx.buffer(reserve=4096*12*8, dynamic=True)
        self.line_vao = self.ctx.vertex_array(self.line_shader, [(self.lines, '3f', 'position')])


def connected_figure(report, z, sup, mesh, out, height=900):
    # The camera looks across the row from the needle's side, so the row runs right to left: the last task first.
    pairs = [(seat, t) for seat in report['seats'] for t in seat['tasks']][::-1]
    model = model_of(report, z, mesh, pairs); poses = np.asarray([r['pose'] for r in model.records])
    exits = [z[t['id']+'_pose'][:3, :3]@seat['exit_object'] for seat, t in pairs]
    q = Layout(poses, np.tile(np.eye(4), (len(poses), 1, 1)), np.asarray(exits, float))
    shells = [moved_mesh(trimesh.Trimesh(sup[seat['id']+'_vertices'], sup[seat['id']+'_faces'], process=False), z[t['id']+'_pose'])
              for seat, t in pairs]
    joined, shifts = joined_row(model, q, shells)
    e = row_axis(model.setup); span = np.ptp(joined.vertices@e); tall = np.ptp(joined.vertices[:, 2])+model.setup['needle_length_m']+.12
    width = int(max(2000, height*span/tall*1.1))
    if width > 8000: height, width = int(height*8000/width), 8000           # the GL framebuffer's limit
    with tempfile.TemporaryDirectory() as tmp:
        render_tool_membranes.Scene = RoomyScene
        try: render_connected(model, q, joined, shifts, Path(tmp), width=width, height=height)
        finally: render_tool_membranes.Scene = Scene
        shutil.move(Path(tmp)/'initialization.png', out/'seats_connected.png')
    print('FIGURE', out/'seats_connected.png', flush=True)


def render(run, out):
    folder = ROOT/'whole_task_2/data'/run/'step3_seats'; report = json.loads((folder/'report.json').read_text())
    mesh = load_shape(ROOT/'objects'/report['object']/'mesh.stl'); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with np.load(folder/'plan.npz') as z: z = {k: z[k] for k in z.files}
    with np.load(folder/'supports.npz') as s: sup = {k: s[k] for k in s.files}
    rows(report, z, sup, mesh, out)
    if report.get('parting'): parting_figure(report, z, mesh, out)
    connected_figure(report, z, sup, mesh, out)
