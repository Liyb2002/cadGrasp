"""Step2's figure (step2/directions.py): how it cuts a big working area, drawn from the run's saved data -- the screen
of needle axes (data/<run>/step1_reach/screen.npz), the plan, Step2's report and supports (data/<run>/step2_seats).
What a direction reaches is computed again with Step2's own function, read only, and the blocks are Step2's own (its
report; cut again for a run that did not keep them); nothing is written but the figure (output/<run>/step2/seats.png).  The same sheet as idea/area_subdivision/code/step2_directions.py.

One sheet: a row for the start and one per step, one column per chosen work direction (W01, W02, ...), one colour per
direction throughout.  The station stays and the bunny turns: every column shows the bunny in its area's pose (Step2's
placement, its direction along the station's needle), all from one camera that has the needle come in from the right.

  Start    the big working area (pink);
  1 Cover  one picture, W01 as its column shows it: the faces its needle reaches (orange; pale: the rest of the area),
           Step1's needle and head at its seed with Step1's gold cone of tilts about it, and Step1's work cube (blue,
           its edge half the object's longest extent, along the station's axes, centred on the tip) --
           a sketch only: Step2 uses no work cube, the needle may work anywhere on the object; the next row: every
           chosen direction's faces (its colour), together covering the area, each with the station's needle, head
           and gold cone of tilts at its seed;
  2 Split  the area cut into connected blocks of about equal size, every block in its direction's colour (as in step
           1), the borders dark, the column's seed a dark dot (the face that faces its needle most squarely, where its
           block grows from);
  3 Check  the area as the plan keeps it (after Step1's exact needle test) in the support Step2 built for it alone (blue).

  python -m whole_task_2.step2.directions B+big_01 --render-only
"""
import json
from itertools import product
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, load_shape, object_of
from whole_task_2.step1.reach_video import Scene, tool_frame, GOLD
from whole_task_2.step1.find_tasks_figure import OUTLINE_COLOR, WORKSPACE_COLOR, cone, dashed, tubes
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step5.video.raster import GRAY, PINK
from . import seats as SE

AREA_COLORS = [(.96, .66, .22), (.42, .74, .42), (.64, .48, .84), (.90, .36, .28), (.88, .80, .28), (.28, .72, .66),
               (.62, .44, .30), (.55, .62, .25), (.45, .45, .70), (.86, .44, .66), (.58, .20, .20), (.75, .25, .45)]
PALE = (.97, .84, .89)  # the big area's faces outside a tile's own set
TILE = 300              # px
NEEDLE = (720, 390)     # px: row 1's picture
CONE_TUBE, DASH = .0012, (.008, .006)   # m: the gold cone's dashes, radius and on/off lengths
CUBE_TUBE = .0009       # m: the work cube's edges, radius
LABEL = 190             # px: the row labels' column
ROW_GAP = 16            # px
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
RISE = .45              # the camera's rise over the horizontal (before normalising)
SEED = .025             # a seed dot's radius, a share of the object's largest extent
MARGIN = 1.08
OFFSET = .00012         # m: the borders lifted off the surface
CHUNK = 4000            # points per line draw (Scene's line buffer holds 4096)


def unit(v):
    v = np.asarray(v, float); return v/np.linalg.norm(v)


def borders(mesh, label):
    """Line pairs on the edges between faces of different labels (label < 0: not drawn on either side)."""
    a, b = mesh.face_adjacency.T; cut = (label[a] != label[b]) & ((label[a] >= 0) | (label[b] >= 0))
    lifted = mesh.vertices+OFFSET*mesh.vertex_normals
    return lifted[mesh.face_adjacency_edges[cut]].reshape(-1, 3)


def station_view(a):
    """The one camera (world): the needle a across the picture, its head on the right -- Scene's right (z x view) along
    the needle's horizontal back side -- a little from above."""
    h = np.r_[-np.asarray(a, float)[:2], 0.]; h = unit(h) if np.linalg.norm(h) > 1e-9 else np.array([1., 0., 0.])
    return unit(np.cross(h, [0., 0., 1.])+RISE*np.array([0., 0., 1.]))


def placed(points, P):
    """points (object frame) at the pose P (world from object)."""
    return np.asarray(points)@P[:3, :3].T+P[:3, 3]


def cube_edges(centre, axes, edge):
    """The twelve edges of a cube of edge about centre along axes (rows)."""
    corners = {c: centre+.5*edge*np.asarray(c)@axes for c in product((-1, 1), repeat=3)}
    return [(corners[a], corners[b]) for a in corners for b in corners if a < b and sum(x != y for x, y in zip(a, b)) == 1]


def needle_picture(mesh, s, P, v, tip, colors, cube, view):
    """Row 1: the object at the pose P, its faces painted, Step1's needle and head along v (world) with its tip at tip,
    Step1's gold cone of tilts about v (apex at the needle root, rim at the tip's distance) and the work cube's edges
    (cube), seen along view."""
    L, beta = s['needle_length_m'], s['max_tilt_deg']; scene = Scene(mesh, s, *NEEDLE)
    try:
        g = scene.r; wire = cone(tip-L*v, v, L, beta, 8)
        gold = g.Shape(scene.ctx, scene.shader, *tubes(dashed(wire, *DASH), CONE_TUBE), GOLD, metallic=.1, roughness=.5)
        box = g.Shape(scene.ctx, scene.shader, *tubes(cube, CUBE_TUBE), WORKSPACE_COLOR, metallic=.1, roughness=.5)
        tool = [(part, tool_frame(tip, v)) for part in scene.tools[False]]
        scene.view = view
        scene.fit(np.vstack([placed(mesh.vertices, P), tip-(L+.07)*v, tip, *[q for w in wire+cube for q in w]]), MARGIN)
        data = scene.draw(P, colors, [], None, [], tool+[(gold, np.eye(4)), (box, np.eye(4))])
    finally:
        scene.close()
    return Image.frombytes('RGB', NEEDLE, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def render(run, output_root=DEFAULT_OUTPUT, folder='step2_seats'):
    work = ROOT/'whole_task_2/data'/run; plan = json.loads((work/'step1_reach/plan.json').read_text()); s = plan['setup']
    report = json.loads((work/folder/'report.json').read_text()); policy = report['policy']
    with np.load(work/'step1_reach/plan.npz') as z: target = z['target'].astype(bool)
    with np.load(work/'step1_reach/screen.npz') as z: axes, reach = z['axes'], z['reach']
    with np.load(work/folder/'supports.npz') as z: saved = {k: z[k] for k in z.files}
    mesh = source = load_shape(ROOT/'objects'/object_of(run)/'mesh.stl'); size = float(mesh.extents.max())
    areas = plan['areas']; k = len(areas); colors = [np.asarray(AREA_COLORS[j % len(AREA_COLORS)]) for j in range(k)]
    poses = [saved[a['id']+'_pose'] for a in areas]; w = unit(s['working_direction_world']); view = station_view(w)

    # Step 1: what each chosen direction reaches (step2/directions.directions), object frame.
    V = np.asarray([unit(a['axis_object']) for a in areas]); cosb = np.cos(np.radians(s['max_tilt_deg']))
    sets = np.asarray([reach[axes@v >= cosb-1e-12].any(0) for v in V])
    ids = np.flatnonzero(target); root = ids[reach[:, ids].any(0)]
    # Step 2: the blocks as Step2 cut them (its report), each block in the column of the area with its direction (a
    # later cut over an earlier); a run that did not keep them: cut again (step2/seats.blocks), the seeds as it picks them.
    final = np.full(len(mesh.faces), -1)
    for j, a in enumerate(areas): final[a['face_ids']] = j
    label = np.full(len(mesh.faces), -1); seeds = [None]*k; cuts = report.get('blocks')
    if cuts:
        for cut in cuts:
            j = int(np.argmax(V@unit(cut['direction'])))
            if V[j]@unit(cut["direction"]) < 1-1e-6: continue              # a block whose area was cut again
            label[cut['faces']] = j; seeds[j] = cut['seed']
    else:
        label[root] = SE.blocks(dict(mesh=source), root, sets, -V, policy)
        print('STEP2 FIGURE BLOCKS cut again:', f'{100*mesh.area_faces[target & (label == final)].sum()/mesh.area_faces[target].sum():.1f}%',
              'of the area in the same area as the plan', flush=True)
    for j in range(k):
        if seeds[j] is None: mine = root[sets[j][root]]; seeds[j] = int(mine[np.argmax(source.face_normals[mine]@-V[j])])

    off = max(float(np.degrees(np.arccos(np.clip(P[:3, :3]@v@w, -1, 1)))) for P, v in zip(poses, V))
    if off > 1e-3: print('STEP2 FIGURE: an area pose turns its direction', round(off, 3), 'deg off the needle', flush=True)
    big = np.tile(GRAY, (len(mesh.faces), 1)); start = big.copy(); start[target] = PINK
    pale = big.copy(); pale[target] = PALE
    split = big.copy()
    for j in range(k): split[label == j] = colors[j]
    flip = lambda data, scene: Image.frombytes('RGB', scene.size, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Row 1: W01 as its column shows it, with Step1's tool and work cube (the station's axes, world).
    first = pale.copy(); first[sets[0] & target] = colors[0]; tip = placed(mesh.triangles_center[seeds[0]], poses[0])
    cube = cube_edges(tip, np.eye(3), .5*size)                                     # Step1: half the longest extent
    needle = needle_picture(mesh, s, poses[0], w, tip, first, cube, view)
    grid = [[None]*k for _ in range(4)]
    scene = Scene(mesh, s, TILE, TILE)
    try:
        g = scene.r; scene.ctx.line_width = 2.*TILE/560
        p = borders(mesh, label); scene.view = view
        for j in range(k):
            P = poses[j]; q = placed(p, P); lines = [(q[i:i+CHUNK], OUTLINE_COLOR) for i in range(0, len(q), CHUNK)]
            ball = trimesh.creation.icosphere(2, SEED*size)
            dot = g.Shape(scene.ctx, scene.shader, ball.vertices+mesh.triangles_center[seeds[j]], ball.faces, OUTLINE_COLOR,
                          metallic=.05, roughness=.5)
            support = saved[areas[j]['id']+'_vertices']                            # world, at the area's pose
            shell = g.Shape(scene.ctx, scene.shader, support, saved[areas[j]['id']+'_faces'], BLUE, smooth=False, metallic=.05, roughness=.55)
            body = np.vstack([placed(mesh.vertices, P), support])
            # The covers row with the station's needle at the seed (the frame takes the head in too).
            cover = pale.copy(); cover[sets[j] & target] = colors[j]; tool = [(placed(mesh.triangles_center[seeds[j]], P), w, False)]
            check = pale.copy(); check[final == j] = colors[j]
            for r, (painted, tools, segments, shapes) in enumerate(((start, [], [], []), (cover, tool, [], []), (split, [], lines, [(dot, P)]),
                                                                    (check, [], [], [(shell, np.eye(4))]))):
                scene.fit(np.vstack([body, scene.points(P, tools)]) if tools else body, MARGIN)   # one frame per column
                grid[r][j] = flip(scene.draw(P, painted, tools, None, segments, shapes), scene)
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
    path = Path(output_root)/run/'step2'/(folder.removeprefix('step2_')+'.png'); path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)
    print('STEP2 FIGURE', path, '|', k, 'areas', flush=True)
    return path
