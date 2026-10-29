"""How Step2 (step2/directions.py) cuts a big working area, from a run's saved data: the screen of needle axes
(data/<run>/step1_reach/screen.npz), the plan and Step2's supports (data/<run>/step2_seats/supports.npz).  What a
direction reaches and the blocks are computed again with Step2's own functions, read only; nothing is written but the
figure.

One sheet: a row for the start and one per step, one column per chosen work direction (W01, W02, ...), every row of
a column seen from that direction's needle side, the bunny upright (object +Y up), one colour per direction throughout:

  Start    the big working area (pink);
  1 Cover  one picture, W01 as its column shows it: the faces its needle reaches (orange; pale: the rest of the area),
           Step1's needle and head at its seed with Step1's gold cone of tilts about it, and Step1's work cube (blue,
           its edge half the object's longest extent, along the station's axes at W01's pose, centred on the tip) --
           a sketch only: Step2 uses no work cube, the needle may work anywhere on the object; the next row: every
           chosen direction's faces (its colour), together covering the area;
  2 Split  the area cut into connected blocks of about equal size (the others' blocks pale), the borders dark, the
           block's seed a dark dot (the face that faces its needle most squarely, where it grows from);
  3 Check  the area as the plan keeps it (after Step1's exact needle test) in the support Step2 built for it alone (blue).

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/area_subdivision/code/step2_directions.py B+big_01
"""
import argparse
import json
from itertools import product
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, load_shape, object_of
from whole_task_2.step1.reach_video import Scene, tool_frame, GOLD
from whole_task_2.step1.find_tasks_figure import OUTLINE_COLOR, WORKSPACE_COLOR, cone, dashed, tubes
from whole_task_2.step2 import seats as SE
from whole_task_2.step2.directions import POLICY
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step5.video.raster import GRAY, PINK
from whole_task_2.idea.area_subdivision.code.step2_cut import HERE, unit, borders, CHUNK

AREA_COLORS = [(.96, .66, .22), (.42, .74, .42), (.64, .48, .84), (.90, .36, .28), (.88, .80, .28), (.28, .72, .66),
               (.62, .44, .30), (.55, .62, .25), (.45, .45, .70), (.86, .44, .66)]
PALE = (.97, .84, .89)  # the big area's faces outside a tile's own set
FADE = .62              # the other blocks' colours mixed this far toward white (step 2)
TILE = 300              # px
NEEDLE = (720, 390)     # px: row 1's picture
CONE_TUBE, DASH = .0012, (.008, .006)   # m: the gold cone's dashes, radius and on/off lengths
CUBE_TUBE = .0009       # m: the work cube's edges, radius
LABEL = 190             # px: the row labels' column
ROW_GAP = 16            # px
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
OFF_NEEDLE = 35.        # deg: a column's camera off the needle's side ...
SIDEWAYS = 50.          # deg: ... turned this far from the vertical about the needle
SEED = .025             # a seed dot's radius, a share of the object's largest extent
MARGIN = 1.08


def tile_view(v):
    """The camera of a column whose needle is v (upright frame): the needle's side, turned OFF_NEEDLE off it, SIDEWAYS
    from the vertical (a horizontal side when the needle is near vertical; Scene needs a view off the vertical)."""
    up = np.array([0., 0., 1.]); p = up-(up@v)*v
    if np.linalg.norm(p) < .2: p = np.cross(v, [1., 0., 0.])
    p = unit(p); b = np.radians(SIDEWAYS); side = np.cos(b)*p+np.sin(b)*unit(np.cross(v, p))
    a = np.radians(OFF_NEEDLE); view = unit(-np.cos(a)*v+np.sin(a)*side)
    if abs(view[2]) > .85: view = unit(np.r_[.53*unit(view[:2]) if np.linalg.norm(view[:2]) > 1e-9 else [.53, 0.], np.sign(view[2])*.85])
    return view


def cube_edges(centre, axes, edge):
    """The twelve edges of a cube of edge about centre along axes (rows)."""
    corners = {c: centre+.5*edge*np.asarray(c)@axes for c in product((-1, 1), repeat=3)}
    return [(corners[a], corners[b]) for a in corners for b in corners if a < b and sum(x != y for x, y in zip(a, b)) == 1]


def needle_picture(mesh, s, v, tip, colors, cube, view):
    """Row 1: the faces painted, Step1's needle and head along v with its tip at tip, Step1's gold cone of tilts about v
    (apex at the needle root, rim at the tip's distance) and the work cube's edges (cube), seen along view."""
    L, beta = s['needle_length_m'], s['max_tilt_deg']; scene = Scene(mesh, s, *NEEDLE)
    try:
        g = scene.r; wire = cone(tip-L*v, v, L, beta, 8)
        gold = g.Shape(scene.ctx, scene.shader, *tubes(dashed(wire, *DASH), CONE_TUBE), GOLD, metallic=.1, roughness=.5)
        box = g.Shape(scene.ctx, scene.shader, *tubes(cube, CUBE_TUBE), WORKSPACE_COLOR, metallic=.1, roughness=.5)
        tool = [(part, tool_frame(tip, v)) for part in scene.tools[False]]
        scene.view = view
        scene.fit(np.vstack([mesh.vertices, tip-(L+.07)*v, tip, *[q for w in wire+cube for q in w]]), MARGIN)
        data = scene.draw(np.eye(4), colors, [], None, [], tool+[(gold, np.eye(4)), (box, np.eye(4))])
    finally:
        scene.close()
    return Image.frombytes('RGB', NEEDLE, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def render(run, filename=None):
    work = ROOT/'whole_task_2/data'/run; plan = json.loads((work/'step1_reach/plan.json').read_text()); s = plan['setup']
    with np.load(work/'step1_reach/plan.npz') as z: target = z['target'].astype(bool)
    with np.load(work/'step1_reach/screen.npz') as z: axes, reach = z['axes'], z['reach']
    with np.load(work/'step2_seats/supports.npz') as z: saved = {k: z[k] for k in z.files}
    source = load_shape(ROOT/'objects'/object_of(run)/'mesh.stl')
    U = Rotation.from_rotvec([np.pi/2, 0., 0.]).as_matrix()                     # object +Y up
    mesh = trimesh.Trimesh(source.vertices@U.T, source.faces, process=False); size = float(mesh.extents.max())
    areas = plan['areas']; k = len(areas); colors = [np.asarray(AREA_COLORS[j % len(AREA_COLORS)]) for j in range(k)]

    # Step 1: what each chosen direction reaches (step2/directions.directions), object frame.
    V = np.asarray([unit(a['axis_object']) for a in areas]); cosb = np.cos(np.radians(s['max_tilt_deg']))
    sets = np.asarray([reach[axes@v >= cosb-1e-12].any(0) for v in V])
    ids = np.flatnonzero(target); root = ids[reach[:, ids].any(0)]
    # Step 2: the blocks (step2/seats.blocks, as seat_plan calls it), the seeds as it picks them.
    label = np.full(len(mesh.faces), -1); label[root] = SE.blocks(dict(mesh=source), root, sets, -V, POLICY)
    seeds = []
    for j in range(k):
        mine = root[sets[j][root]]; seeds.append(int(mine[np.argmax(source.face_normals[mine]@-V[j])]))
    final = np.full(len(mesh.faces), -1)
    for j, a in enumerate(areas): final[a['face_ids']] = j
    print('BLOCKS', run, f'{100*mesh.area_faces[target & (label == final)].sum()/mesh.area_faces[target].sum():.1f}% of the area',
          'in the same area as the plan', flush=True)

    Vu = V@U.T                                                                    # needles, upright frame
    big = np.tile(GRAY, (len(mesh.faces), 1)); start = big.copy(); start[target] = PINK
    pale = big.copy(); pale[target] = PALE
    faded = big.copy()
    for j in range(k): faded[label == j] = (1-FADE)*colors[j]+FADE
    flip = lambda data, scene: Image.frombytes('RGB', scene.size, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Row 1: W01 as its column shows it, with Step1's tool and work cube (the station's axes at W01's pose, upright frame).
    first = pale.copy(); first[sets[0] & target] = colors[0]; tip = mesh.triangles_center[seeds[0]]
    station = saved[areas[0]['id']+'_pose'][:3, :3]@U.T                          # rows: the world axes, upright frame
    cube = cube_edges(tip, station, .5*size)                                       # Step1: half the longest extent
    needle = needle_picture(mesh, s, Vu[0], tip, first, cube, tile_view(Vu[0]))
    grid = [[None]*k for _ in range(4)]
    scene = Scene(mesh, s, TILE, TILE)
    try:
        g = scene.r; scene.ctx.line_width = 2.*TILE/560
        p = borders(mesh, label); lines = [(p[i:i+CHUNK], OUTLINE_COLOR) for i in range(0, len(p), CHUNK)]
        for j in range(k):
            ball = trimesh.creation.icosphere(2, SEED*size)
            dot = g.Shape(scene.ctx, scene.shader, ball.vertices+mesh.triangles_center[seeds[j]], ball.faces, OUTLINE_COLOR,
                          metallic=.05, roughness=.5)
            P = saved[areas[j]['id']+'_pose']; R, t = P[:3, :3], P[:3, 3]
            support = (saved[areas[j]['id']+'_vertices']-t)@R@U.T                  # world -> object -> upright
            shell = g.Shape(scene.ctx, scene.shader, support, saved[areas[j]['id']+'_faces'], BLUE, smooth=False, metallic=.05, roughness=.55)
            scene.view = tile_view(Vu[j]); scene.fit(np.vstack([mesh.vertices, support]), MARGIN)   # one frame per column
            cover = pale.copy(); cover[sets[j] & target] = colors[j]
            split = faded.copy(); split[label == j] = colors[j]
            check = pale.copy(); check[final == j] = colors[j]
            for r, (painted, segments, shapes) in enumerate(((start, [], []), (cover, [], []), (split, lines, [dot]), (check, [], [shell]))):
                grid[r][j] = flip(scene.draw(np.eye(4), painted, [], None, segments, [(m, np.eye(4)) for m in shapes]), scene)
    finally:
        scene.close()

    # Rows: (label, pictures, height, a line above); the covers row goes on step 1, no line above it.
    rows = [('Start', grid[0], TILE, False), ('1  Cover', [needle], NEEDLE[1], True), ('', grid[1], TILE, False),
            ('2  Split', grid[2], TILE, True), ('3  Check', grid[3], TILE, True)]
    width = LABEL+k*TILE; height = sum(r[2] for r in rows)+(len(rows)-1)*ROW_GAP
    sheet = Image.new('RGB', (width, height), 'white'); ink = ImageDraw.Draw(sheet); font = ImageFont.truetype(FONT, 34); y = 0
    for name, pictures, tall, line in rows:
        for j, im in enumerate(pictures): sheet.paste(im, (LABEL+j*TILE, y))
        ink.text((24, y+tall//2), name, fill=(60, 60, 60), font=font, anchor='lm')
        if line: ink.line([(24, y-ROW_GAP//2), (width-24, y-ROW_GAP//2)], fill=(215, 215, 215), width=2)
        y += tall+ROW_GAP
    path = HERE/((filename or run.removeprefix(object_of(run)+'+')+'_directions')+'.png'); sheet.save(path)
    print('FIGURE', path, '|', k, 'areas', flush=True)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('run', nargs='+')
    p.add_argument('--name', help="the figure's file name (one run): a new variant, the earlier one kept")
    a = p.parse_args()
    for run in a.run: render(run, a.name)
