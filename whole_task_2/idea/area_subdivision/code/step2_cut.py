"""How Step2 (step2/seats.py) cuts a big working area, drawn from a run's saved plan (data/<run>/step1_reach/plan.json);
nothing is computed again.  Three columns, the bunny upright (object +Y up), two opposite views (rows), no text:

  1 the big area (pink; grey: not to be worked);
  2 the big cut, seats: one colour per seat, its exit e an arrow of that colour in front of its faces (the object
    leaves the seat along e);
  3 the small cut, tasks: a seat's faces on one base -- one shade of its seat's colour per task, the tasks' borders
    drawn dark.

Run from the repository root:
  PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/area_subdivision/code/step2_cut.py B+whole_shape_free
"""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from whole_task_2.step1.common import ROOT, load_shape, object_of
from whole_task_2.step1.reach_video import Scene
from whole_task_2.step1.find_tasks_figure import OUTLINE_COLOR, tube, combine
from whole_task_2.step5.video.raster import GRAY, PINK

HERE = Path(__file__).resolve().parents[1]
SEAT_COLORS = [(.30, .60, .86), (.96, .66, .22), (.42, .74, .42), (.64, .48, .84), (.30, .78, .78), (.92, .46, .36),
               (.80, .78, .30), (.86, .44, .62)]
SIZE = 760              # px: a panel
GAP = 90                # px: between columns (the chevrons go here)
ROW_GAP = 10            # px
RISE = .45              # the views' height (before normalising)
AZIMUTH = 35.           # deg: the views turned off the area's side, so no exit arrow points at the camera
ARROW = .30             # an exit arrow's length, a share of the object's largest extent
SHAFT, HEAD, TIP = .009, .028, .075     # its shaft and head radii and head length, shares of the largest extent
CLEAR = .05             # its start this share of the largest extent in front of the object
MARGIN = 1.06
OFFSET = .00012         # m: the borders lifted off the surface
CHUNK = 4000            # points per line draw (Scene's line buffer holds 4096)


def unit(v):
    v = np.asarray(v, float); return v/np.linalg.norm(v)


def shades(color, n):
    """n shades of color, darker to lighter (just color when n == 1)."""
    color = np.asarray(color)
    if n == 1: return [color]
    return [(1-t)*.78*color+t*(.55*color+.45) for t in np.linspace(0, 1, n)]


def borders(mesh, label):
    """Line pairs on the edges between faces of different labels (label < 0: not drawn on either side)."""
    a, b = mesh.face_adjacency.T; cut = (label[a] != label[b]) & ((label[a] >= 0) | (label[b] >= 0))
    lifted = mesh.vertices+OFFSET*mesh.vertex_normals
    return lifted[mesh.face_adjacency_edges[cut]].reshape(-1, 3)


def arrow(start, e, size):
    """An arrow along e from start: (vertices, faces)."""
    end = start+ARROW*size*e; neck = end-TIP*size*e
    return combine([tube(start, neck, SHAFT*size), tube(neck, end, HEAD*size, 1e-6)])


def render(run, filename=None, width=SIZE):
    work = ROOT/'whole_task_2/data'/run/'step1_reach'; plan = json.loads((work/'plan.json').read_text())
    with np.load(work/'plan.npz') as z: target = z['target'].astype(bool)
    source = load_shape(ROOT/'objects'/object_of(run)/'mesh.stl')
    U = Rotation.from_rotvec([np.pi/2, 0., 0.]).as_matrix()                     # object +Y up
    mesh = trimesh.Trimesh(source.vertices@U.T, source.faces, process=False); size = float(mesh.extents.max())
    areas = plan['areas']; seats = list(dict.fromkeys(a['seat'] for a in areas))
    seat_of = np.full(len(mesh.faces), -1); task_of = np.full(len(mesh.faces), -1)
    for k, a in enumerate(areas): seat_of[a['face_ids']] = seats.index(a['seat']); task_of[a['face_ids']] = k

    # Colours: the big area, the seats, the tasks (a seat's tasks in shades of its colour, the largest darkest).
    big = np.tile(GRAY, (len(mesh.faces), 1)); big[target] = PINK
    by_seat = big.copy(); by_task = big.copy()
    for s, name in enumerate(seats):
        by_seat[seat_of == s] = SEAT_COLORS[s % len(SEAT_COLORS)]
        mine = sorted([k for k, a in enumerate(areas) if a['seat'] == name], key=lambda k: -areas[k]['area_m2'])
        for k, c in zip(mine, shades(SEAT_COLORS[s % len(SEAT_COLORS)], len(mine))): by_task[task_of == k] = c

    # Exit arrows, one per seat, in front of its faces along e (upright frame).
    arrows = []
    for s, name in enumerate(seats):
        a = next(a for a in areas if a['seat'] == name); e = unit(U@np.asarray(a['exit_object']))
        f = seat_of == s; w = mesh.area_faces[f]; c = (w[:, None]*mesh.triangles_center[f]).sum(0)/w.sum()
        start = c+e*(float(((mesh.vertices-c)@e).max())+CLEAR*size)
        arrows.append((arrow(start, e, size), .8*np.asarray(SEAT_COLORS[s % len(SEAT_COLORS)])))
    tips = np.vstack([v for (v, _), _ in arrows])

    # Views: from the side the big area faces (an area all round: the object's front, -Z) and from the other.
    n = mesh.area_faces[target]@mesh.face_normals[target]
    out = unit(n) if np.linalg.norm(n) > .05*mesh.area_faces[target].sum() else np.array([0., -1., 0.])
    turn = Rotation.from_rotvec([0., 0., np.radians(AZIMUTH)]).as_matrix()
    views = [unit(turn@np.array([v[0], v[1], 0.])+[0., 0., RISE]) for v in (out, -out)]

    scene = Scene(mesh, plan['setup'], width, width)
    try:
        g = scene.r; scene.ctx.line_width = 3*width/560
        shapes = [(g.Shape(scene.ctx, scene.shader, v, f, color, smooth=False, metallic=.05, roughness=.5), np.eye(4)) for (v, f), color in arrows]
        lines = {name: [(p[i:i+CHUNK], OUTLINE_COLOR) for i in range(0, len(p), CHUNK)] for name, p in
                 (('seat', borders(mesh, seat_of)), ('task', borders(mesh, task_of)))}
        panels = []
        for view in views:
            scene.view = view; scene.fit(np.vstack([mesh.vertices, tips]), MARGIN)
            row = []
            for colors, extra, segments in ((big, [], []), (by_seat, shapes, []), (by_task, [], lines['task'])):
                data = scene.draw(np.eye(4), colors, [], None, segments, extra)
                row.append(Image.frombytes('RGB', scene.size, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM))
            panels.append(row)
    finally:
        scene.close()

    sheet = Image.new('RGB', (3*width+2*GAP, 2*width+ROW_GAP), 'white'); ink = ImageDraw.Draw(sheet)
    for r, row in enumerate(panels):
        for c, panel in enumerate(row): sheet.paste(panel, (c*(width+GAP), r*(width+ROW_GAP)))
    for c in range(2):                                                             # chevrons between the columns
        x = (c+1)*width+c*GAP+GAP//2; y = width+ROW_GAP//2; h = GAP//3
        ink.polygon([(x-h//2, y-h), (x+h//2, y), (x-h//2, y+h)], fill=(150, 150, 150))
    path = HERE/((filename or run.removeprefix(object_of(run)+'+'))+'.png'); sheet.save(path)
    print('FIGURE', path, '|', len(seats), 'seats,', len(areas), 'tasks', flush=True)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('run', nargs='+')
    p.add_argument('--name', help="the figure's file name (one run): a new variant, the earlier one kept")
    a = p.parse_args()
    for run in a.run: render(run, a.name)
