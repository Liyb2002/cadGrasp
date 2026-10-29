"""How far the work arm can move the needle (its box, blue) decides how many working areas the bunny needs: a small box,
many; a large box, few.  In find_tasks.png's manner; no support.

A pose: the bunny in any orientation -- a needle direction v (object frame, one of DIRECTIONS on the sphere) turned
onto the station's and the bunny turned about it (one of TURNS, as Step2 leaves that turn free) -- then moved anywhere.
Its working area: the faces the needle reaches within the tilt cone about v (the screen of needle axes: needle and head
clear of the bunny; the same at every turn) whose centres lie in the box (edge s, the station's axes, so its
orientation on the bunny is the pose's).  The working areas: a greedy cover of every face some needle axis reaches -- each time the pose
and box (corners every s/CORNERS) taking the most area not yet taken.  With the larger boxes, pieces taking less than
IGNORE of the surface are not counted: their faces join the nearest working area (as Step2's assumed reachable faces).

The figure, a column per box (BOXES): on top the station, one pose for all boxes (common_pose: the bunny upright) --
the bunny on the table, the KUKA holding the needle with Step1's gold cone of tilts, the box about the needle's tip,
its working area pink with its border; below the bunny upright from two opposite sides, every working area in a shade of pink
(neighbours different), the borders dark.

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/area_subdivision/code/box_poses.py            the figure
  ... --sweep 20 40 77       working areas for these box edges (mm), no figure
The figure goes to idea/area_subdivision/box_poses.png.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_video import Scene, kuka, arm_links, tool_frame, GOLD
from whole_task_2.step1.find_tasks_figure import OUTLINE_COLOR, WORKSPACE_COLOR, tabletop, workspace_box, cone, dashed, tubes, picture, task_outline
from whole_task_2.step2 import split as S
from whole_task_2.step2.seats import fibonacci
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step2_cut import borders, CHUNK, unit

HERE = Path(__file__).resolve().parents[1]
RUN = 'B+big_01'        # the station (robots_config/random/07_kuka.json) and the object's screen of needle axes
BOXES = (('small', .036, False), ('large', .075, True), ('extra large', .12, True))   # m; pieces under IGNORE not counted
IGNORE = .005           # share of the surface (user, 10-29: for the 75 and 120 mm boxes)
DIRECTIONS = 240        # needle directions v tried (Fibonacci)
TURNS = 12              # turns about the needle tried per direction (every 30 deg)
CORNERS = 4             # box corners per box edge
PINKS = [(1., .52, .74), (1., .76, .87), (.86, .30, .55), (.98, .62, .80), (.74, .22, .45), (.93, .44, .62), (1., .86, .92)]
TOP, BOTTOM, LABEL, GAP = (1100, 760), 550, 70, 30   # px: a station panel, a bunny view, the labels' band, between columns
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
VIEW = (-.3, -1., .8)   # the station's camera, world (from the side towards the arm, above: the worked side shows)
RISE, TURN = .45, 35.   # the bunny views: their height, their turn off the area's side (deg)
AIM = .025              # m: the needle's line aimed this far below the bunny's centre (on its back, clear of the ears)


def inputs(run=RUN):
    """The station (Step2's setup), the bunny (object frame), the screen (axes, what each reaches), the faces to work."""
    work = ROOT/'whole_task_2/data'/run/'step1_reach'; s = json.loads((work/'plan.json').read_text())['setup']
    mesh = load_shape(ROOT/'objects/B/mesh.stl')
    with np.load(work/'screen.npz') as z: axes, reach = z['axes'], z['reach']
    return s, mesh, axes, reach, reach.any(0)


def working_areas(s, mesh, axes, reach, target, edge, ignore=0.):
    """Greedy cover: [dict(v, R (world <- object rotation), corner (world-oriented coordinates), area (the
    pose's whole working area), new (the faces it takes))]; each face's area index; the pieces not counted (taking
    less than `ignore` of the surface: their faces given to the working area of the nearest face counted)."""
    a = unit(s['working_direction_world']); cosb = np.cos(np.radians(s['max_tilt_deg'])); V = fibonacci(DIRECTIONS)
    A = mesh.area_faces; C = mesh.triangles_center; step = edge/CORNERS; k = CORNERS; poses = []
    for v in V:
        ok = reach[axes@v >= cosb-1e-12].any(0) & target
        if not ok.any(): continue
        for psi in 2*np.pi*np.arange(TURNS)/TURNS:
            R = Rotation.from_rotvec(a*psi).as_matrix()@S.toward(v, a); X = C@R.T; lo = X[ok].min(0)-edge
            poses.append(dict(v=v, R=R, ok=ok, lo=lo, cells=np.floor((X-lo)/step).astype(int)))
    left = target.copy(); out = []; owner = np.full(len(C), -1)
    while left.any():
        best = None; bound = np.asarray([A[p['ok'] & left].sum() for p in poses])
        for i in np.argsort(-bound):                    # the most a pose could still take, the largest first
            if best is not None and bound[i] <= best[0]: break
            p = poses[i]; m = p['ok'] & left
            if not m.any(): continue
            c = p['cells'][m]; H = np.zeros(c.max(0)+k+1); np.add.at(H, tuple(c.T), A[m])
            I = np.pad(H, ((1, 0),)*3).cumsum(0).cumsum(1).cumsum(2)
            W = (I[k:, k:, k:]-I[:-k, k:, k:]-I[k:, :-k, k:]-I[k:, k:, :-k]+I[:-k, :-k, k:]+I[:-k, k:, :-k]+I[k:, :-k, :-k]-I[:-k, :-k, :-k])
            j = np.unravel_index(np.argmax(W), W.shape)
            if best is None or W[j] > best[0]: best = (W[j], i, np.asarray(j))
        _, i, j = best; p = poses[i]; inside = p['ok'] & np.all((p['cells'] >= j) & (p['cells'] < j+k), axis=1)
        new = inside & left; owner[new] = len(out); left &= ~inside
        out.append(dict(v=p['v'], R=p['R'], corner=p['lo']+(edge/CORNERS)*j, area=inside, new=new))
    keep = [n for n, x in enumerate(out) if A[x['new']].sum() >= ignore*A[target].sum()]; dropped = len(out)-len(keep)
    if dropped:
        from scipy.spatial import cKDTree
        number = np.full(len(out), -1); number[keep] = np.arange(len(keep)); counted = np.flatnonzero(np.isin(owner, keep))
        owner = np.where(owner >= 0, number[np.maximum(owner, 0)], -1); stray = np.flatnonzero(target & (owner < 0))
        owner[stray] = owner[counted[cKDTree(C[counted]).query(C[stray])[1]]]; out = [out[n] for n in keep]
    return out, owner, dropped


def shades(mesh, owner, n):
    """A pink per working area, neighbours (sharing an edge) different where the shades allow, the least used first."""
    a, b = mesh.face_adjacency.T; near = {k: set() for k in range(n)}
    for x, y in zip(owner[a], owner[b]):
        if x != y and x >= 0 and y >= 0: near[int(x)].add(int(y)); near[int(y)].add(int(x))
    size = np.bincount(owner[owner >= 0], mesh.area_faces[owner >= 0], n); colour = {}
    for k in np.argsort(-size):
        used = [colour[j] for j in near[k] if j in colour]; count = np.bincount(list(colour.values()), minlength=len(PINKS))
        clash = np.bincount(used, minlength=len(PINKS))                       # all taken: the one fewest neighbours have
        colour[int(k)] = min(range(len(PINKS)), key=lambda c: (clash[c], count[c], c))
    return [PINKS[colour[k]] for k in range(n)]


def common_pose(s, mesh, axes, reach, target):
    """One pose for both station panels: the bunny upright (+Y up), turned about the vertical by the first of YAWS
    (every 30 deg) whose working area with the large box is largest; the box centred where the needle's line through
    AIM below the bunny's centre first meets it.  (world <- object rotation, that point, object frame; the faces reached.)"""
    a = unit(s['working_direction_world']); cosb = np.cos(np.radians(s['max_tilt_deg'])); U = Rotation.from_rotvec([np.pi/2, 0., 0.]).as_matrix()
    best = None
    for yaw in np.radians(np.arange(0., 360., 30.)):
        R = Rotation.from_euler('z', yaw).as_matrix()@U; v = R.T@a; ok = reach[axes@v >= cosb-1e-12].any(0) & target
        o = mesh.centroid-AIM*R[2]; loc = mesh.ray.intersects_location([o-v], [v], multiple_hits=False)[0]
        if not len(loc): continue
        n = float(mesh.area_faces[ok & inside(mesh, R, loc[0], BOXES[-1][1])].sum())
        if best is None or n > best[0]: best = (n, R, loc[0], ok)
    return best[1:]


def inside(mesh, R, centre, edge):
    """The faces whose centres lie in the box of edge `edge` about `centre` (object frame) along the world's axes (rows of R)."""
    return np.all(np.abs((mesh.triangles_center-centre)@R.T) <= edge/2, axis=1)


def station(s, mesh, axes, reach, R, p, ok, edge):
    """The station panel at the common pose: the bunny on the table, p over the work point; the KUKA holding the needle
    on p along the screened axis there nearest the nominal direction; the cone of tilts about it; the box of edge
    `edge` about p; its working area pink (the faces reached whose centres are in the box)."""
    a = unit(s['working_direction_world']); L, beta = s['needle_length_m'], s['max_tilt_deg']; v = R.T@a; cosb = np.cos(np.radians(beta))
    shift = np.asarray(s['work_point_m'], float); P = np.eye(4); P[:3, :3] = R; P[:3, 3] = shift-R@p
    P[2, 3] += s['table_height_m']-float((mesh.vertices@R.T+P[:3, 3])[:, 2].min()); area = ok & inside(mesh, R, p, edge)
    f = int(mesh.nearest.on_surface([p])[2][0]); k = np.flatnonzero((axes@v >= cosb-1e-12) & reach[:, f])
    x = axes[k[np.argmax(axes[k]@v)]] if len(k) else v; tip, d = R@p+P[:3, 3], R@x
    links, _ = arm_links(kuka(s), tip, d)
    links = links.copy(); links[:, :3, 3] -= shift; W = P.copy(); W[:3, 3] -= shift; t = tip-shift
    setup = dict(s, surface_workspace=dict(bounds_world_m=[(tip-edge/2).tolist(), (tip+edge/2).tolist()]))
    scene = Scene(mesh, setup, *TOP, arm=True); r = scene.r
    try:
        scene.view = unit(VIEW); shapes = tabletop(scene, setup, shift); _, corners = workspace_box(scene, setup, shift)
        tool = [(part, tool_frame(t, d)) for part in scene.tools[False]]; wire = cone(t-L*d, a, L, beta, 8)
        cones = [(r.Shape(scene.ctx, scene.shader, *tubes(dashed(wire), .00065), GOLD, metallic=.1, roughness=.5), np.eye(4))]
        colors = np.tile(r.GRAY, (len(mesh.faces), 1)); colors[area] = r.PINK
        fit = np.vstack([mesh.vertices@W[:3, :3].T+W[:3, 3], corners, t, t-(L+.07)*d, *[q for w in wire for q in w]])
        return picture(scene, W, colors, shapes+tool, fit, links, [], .3, cones, segments=task_outline(mesh, area, W))
    finally:
        scene.close()


def bunnies(s, mesh, owner, n):
    """The bunny upright (+Y up) from two opposite sides, each working area in a shade of pink, the borders dark."""
    U = Rotation.from_rotvec([np.pi/2, 0., 0.]).as_matrix(); up = trimesh.Trimesh(mesh.vertices@U.T, mesh.faces, process=False)
    colors = np.tile((.66, .69, .72), (len(mesh.faces), 1)); pal = shades(mesh, owner, n)
    for k in range(n): colors[owner == k] = pal[k]
    p = borders(up, owner); lines = [(p[i:i+CHUNK], OUTLINE_COLOR) for i in range(0, len(p), CHUNK)]; out = []
    scene = Scene(up, s, BOTTOM, BOTTOM)
    try:
        scene.ctx.line_width = 2.*BOTTOM/560
        for az in np.radians([-90.+TURN, 90.+TURN]):
            scene.view = unit(np.r_[np.cos(az), np.sin(az), RISE]); scene.fit(up.vertices, 1.06)
            out.append(Image.frombytes('RGB', scene.size, scene.draw(np.eye(4), colors, [], None, lines, [])).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    finally:
        scene.close()
    return out


def main(sweep=()):
    s, mesh, axes, reach, target = inputs(); A = mesh.area_faces
    if sweep:
        for mm in sweep:
            found, _, n = working_areas(s, mesh, axes, reach, target, mm/1e3, IGNORE)
            print(f'box {mm:g} mm: {len(found)+n} working areas, {len(found)} with pieces under {100*IGNORE:g}% not counted', flush=True)
        return
    columns = []; R, p, ok = common_pose(s, mesh, axes, reach, target)
    for name, edge, small in BOXES:
        found, owner, dropped = working_areas(s, mesh, axes, reach, target, edge, IGNORE if small else 0.)
        assert np.array_equal(owner >= 0, target)
        print(f'box {1e3*edge:g} mm: {len(found)} working areas ({dropped} pieces under {100*IGNORE:g}% not counted); at the common pose '
              f'it reaches {100*A[ok & inside(mesh, R, p, edge)].sum()/A[target].sum():.1f}% of the surface', flush=True)
        columns.append((name, edge, small, len(found), station(s, mesh, axes, reach, R, p, ok, edge), bunnies(s, mesh, owner, len(found))))
    w = TOP[0]; Hh = LABEL+TOP[1]+LABEL+BOTTOM
    sheet = Image.new('RGB', (len(columns)*w+(len(columns)-1)*GAP, Hh), 'white'); ink = ImageDraw.Draw(sheet)
    big = ImageFont.truetype(FONT, 38); blue = tuple(int(255*x) for x in WORKSPACE_COLOR)
    for c, (name, edge, small, n, top, views) in enumerate(columns):
        x = c*(w+GAP)
        ink.text((x+w//2, LABEL//2), f'{name} translation: box {1e3*edge:g} mm', fill=blue, font=big, anchor='mm')
        sheet.paste(top, (x, LABEL)); y = LABEL+TOP[1]
        ink.text((x+w//2, y+LABEL//2), f'{n} working areas', fill=(50, 50, 50), font=big, anchor='mm')
        for i, im in enumerate(views): sheet.paste(im, (x+i*BOTTOM, LABEL+TOP[1]+LABEL))
        if c: ink.line([(x-GAP//2, 20), (x-GAP//2, Hh-20)], fill=(215, 215, 215), width=2)
    path = HERE/'box_poses.png'; sheet.save(path); print('FIGURE', path, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--sweep', type=float, nargs='*', default=())
    main(p.parse_args().sweep)
