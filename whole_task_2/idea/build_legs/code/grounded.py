"""How the legs are built (top row) and where they may go (bottom row), for contrast_vertical.py's big messy task:
its upright bunny, the big work area (not coloured) and pushes anywhere in the gold cone about straight down.  The bunny
sits in the cup (contrast_vertical.py's eggshell without its legs), its lowest point LIFT above the table.  Two rows of
three panels, one camera, no table, no robot, no text.  A schematic, not to scale.

Top row, the steps: 1 each load is the bunny's weight plus a push of PUSH weights at a point of the area along a
direction in the cone, and lands on the table at its centre of pressure (G.pressure_points), here as if the bunny's
lowest point were UNDER above the table, drawn DRAWN times as far from their mean: LOADS samples, DOTS of them drawn,
LINES lines of action (to the hull's corners) dashed; 2 the pressure hull (their convex hull); 3 the feet, as Step5
(combined.feet): candidates on the rings FOOT['outward'] outside the hull, a site every FOOT['spacing'], the cheapest
cycle of them with the hull inside (combined.enclosing; a foot costs its plate's area, the nearest rings first; dashed
blue), each joined (dashed) to the cup's point nearest it.  The marks on the table are drawn as
slides/Co-optimize/seperate_base_support_proof/videos/render03.py draws its cloud: flat camera-facing squares DOT_PX
across, the hull dotted every MARK_PX.

Bottom row, the same feet (Step5's flat cones, combined.plate, from the cup's nearest point straight down to the
site) in three places: the cheapest cycle (the top row's); three feet, one toward the camera; four feet, two toward
the camera, WIDE further out.  The last two are placed by hand about the hull's centre, on the smallest circle (in
5 mm steps) whose polygon holds the hull FOOT['outward'][0] inside.  Each support is the cup and its feet, minus the bunny swept straight up, one piece, with every
drawn centre of pressure inside its feet and the needle (not drawn) and its head clear of it.

Run from the repository root:
  PYTHONPATH=.:whole_task_2/idea/easy_task/code:whole_task_2/idea/build_legs/code .venv/bin/python -B whole_task_2/idea/build_legs/code/grounded.py
"""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
import manifold3d as md

from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_video import Scene, GOLD
from whole_task_2.step1.find_tasks_figure import OUTLINE_WIDTH_PX, CONE_OPACITY, Opacity, cone, tubes, combine
from whole_task_2.step1.find_tasks_figure import dashed as dashes
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step5.video.raster import GRAY
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step4.combine import exact_sweep
from whole_task_2.step5.combined import FOOT, enclosing, plate, widths, resample, joined
from vertical import LIFT, GAP, DOWN, FACE, TURN, first_hit, spin
from slanted import INFLATE, inside
from contrast_vertical import BIG, CONE, DEPTH, THICK, SMOOTH

HERE = Path(__file__).resolve().parents[1]
UNDER = THICK       # m: the centres of pressure as if the bunny's lowest point were this far above the table
PUSH = .5           # object weights: a push
DRAWN = 1.5         # the centres of pressure drawn (and the feet placed) this many times as far from their mean
LOADS = 4000        # sampled loads
DOTS = 500          # of them drawn (render03.py: 512)
LINES = 6           # lines of action drawn
WIDE = .03          # m: the last bottom-row feet this much further out
DOT_PX = OUTLINE_WIDTH_PX   # px: a dot, square (drawn as a line this long and this wide)
MARK_PX = 3.8       # px: a dotted hull, a dot every this far (render03.py)
DOT_COLOR = (.161, .584, .518)   # render03.py's shown pose, '#299584'
RISE = 1.           # camera height (the view's z before normalising)
MARGIN = 1.06
SIZE = (900, 820)   # px: one panel
GAP_PX = 24


def cup(m):
    """contrast_vertical.eggshell without its legs: the convex hull of the bunny below DEPTH above its lowest point,
    grown by THICK and rounded, up to that height (manifold, not yet cut by the bunny)."""
    top = m.bounds[0, 2]+DEPTH; part = m.vertices[m.vertices[:, 2] < top]
    ball = trimesh.creation.icosphere(2, THICK).vertices
    shell = md.Manifold.hull_points((part[:, None]+ball).reshape(-1, 3)/G.SCALE).smooth_out(90.).refine(SMOOTH)
    return shell.trim_by_plane([0, 0, -1.], -top/G.SCALE)


def loads(m, area, com, rng):
    """LOADS loads (the weight plus a push of PUSH at a point of area along a direction in the cone): (push points,
    push directions, the reactions' wrenches about com)."""
    face = rng.choice(np.flatnonzero(area), LOADS, p=m.area_faces[area]/m.area_faces[area].sum())
    p = np.einsum('ij,ijk->ik', rng.dirichlet(np.ones(3), LOADS), m.triangles[face])
    c = rng.uniform(np.cos(np.radians(CONE)), 1., LOADS); s = np.sqrt(1-c*c); phi = rng.uniform(0, 2*np.pi, LOADS)
    a = np.c_[s*np.cos(phi), s*np.sin(phi), -c]; F = DOWN+PUSH*a
    return p, a, np.c_[-F, -np.cross(p-com, PUSH*a)]


def discs(xy, radius, height=.001):
    v, f = combine([(c.vertices, c.faces) for c in (trimesh.creation.cylinder(radius, segment=[[*q, 0.], [*q, height]], sections=32) for q in xy)])
    return trimesh.Trimesh(v, f, process=False)


def marks(xy, ink, right, pixel, z=0.):
    """Dots DOT_PX across at xy (z on the table), as segments: lines DOT_PX long across the screen, drawn DOT_PX wide."""
    P = np.c_[xy, np.full(len(xy), z)]
    return np.stack([P-.5*DOT_PX*pixel*right, P+.5*DOT_PX*pixel*right], 1).reshape(-1, 3), ink


def lines(pairs, ink):
    return np.asarray(pairs, float).reshape(-1, 3), ink


def solid(stock, parts, sweep):
    body = G.union([stock, *parts]).trim_by_plane([0, 0, 1.], 0.)-sweep
    if not joined(body): raise ValueError('A support is not one piece')
    return G.mesh(body)


def ground(mesh):
    """The xy of a support's points on the table."""
    return mesh.vertices[np.abs(mesh.vertices[:, 2]) < 1e-7, :2]


def render(source, setup, view, fit, W, colors, outline, items, segments, wire=None):
    """One panel: the bunny, items (mesh, colour) and the segments (pairs, ink)."""
    scene = Scene(source, setup, *SIZE)
    try:
        r = scene.r
        scene.view = view; scene.fit(fit, MARGIN)
        shape = lambda x, c: (r.Shape(scene.ctx, scene.shader, x.vertices, x.faces, c, metallic=.05, roughness=.5), np.eye(4))
        cones = None if wire is None else [(r.Shape(scene.ctx, scene.shader, *tubes(dashes(wire), .0005), GOLD, metallic=.1, roughness=.5), np.eye(4))]
        compositor = Opacity(scene)
        try: data = compositor.draw(W, colors, [shape(*x) for x in items], None, [], cones=cones, cone_opacity=CONE_OPACITY,
                                    segments=outline+segments)
        finally: compositor.close()
    finally: scene.close()
    return Image.frombytes('RGB', SIZE, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def main():
    source = load_shape(ROOT/'objects/B/mesh.stl')
    setup = json.loads((ROOT/'whole_task_2/data/B+03_kuka/step1_reach/plan.json').read_text())['setup']
    setup['working_direction_world'] = DOWN.tolist()
    reach = np.asarray(setup['work_point_m'])[:2]-np.asarray(setup['base_position_world_m'])[:2]
    camera = np.degrees(np.arctan2(reach[1], reach[0]))+90+TURN
    view = np.r_[np.cos(np.radians(camera)), np.sin(np.radians(camera)), RISE]; view /= np.linalg.norm(view)

    W = np.eye(4); W[:3, :3] = spin(camera-FACE)@[[1, 0, 0], [0, 0, -1], [0, 1, 0]]   # original +Y up
    m = source.copy(); m.apply_transform(W)
    shift = -m.center_mass; shift[2] = LIFT-m.bounds[0, 2]; W[:3, 3] = shift; m.apply_translation(shift)
    com = m.center_mass
    top = np.r_[com[:2], first_hit(m, [[*com[:2], com[2]+1]], DOWN)[0]]; tip = top-GAP*DOWN
    area = (m.face_normals[:, 2] > 0) & (np.linalg.norm(m.triangles_center-top, axis=1) < BIG)
    grown = trimesh.Trimesh(m.vertices+INFLATE*m.vertex_normals, m.faces, process=False)
    sweep = exact_sweep(grown, -DOWN, DEPTH+.1); stock = cup(m)
    seat = solid(stock, [], sweep)

    # 1, 2: the centres of pressure, their hull.
    rng = np.random.default_rng(0)
    p, a, wrench = loads(m, area, com, rng); cop = G.pressure_points(wrench, com-[0, 0, LIFT-UNDER])
    cop = cop.mean(0)+DRAWN*(cop-cop.mean(0))                   # a schematic: drawn bigger
    hull = cop[ConvexHull(cop).vertices]; centre = hull.mean(0)
    angle = np.arctan2(*(hull-centre).T[::-1])
    picks = [int(np.argmin(np.abs(np.angle(np.exp(1j*(angle-t)))))) for t in np.linspace(-np.pi, np.pi, LINES, endpoint=False)]
    picks = [int(np.flatnonzero((cop == hull[k]).all(1))[0]) for k in picks]
    action = [(np.r_[cop[k], 0.], np.r_[cop[k], 0.]+(p[k, 2]/wrench[k, 2])*wrench[k, :3]) for k in picks]

    # 3: as combined.feet: candidates on rings outside the hull, moved out to legs FOOT['least'] long; the cheapest
    # cycle of them about the hull.
    sites, level = [], []
    for k, d in enumerate(FOOT['outward']):
        q = resample(np.asarray(orient(Polygon(hull).buffer(d, join_style=2), sign=1.).exterior.coords)[:-1], FOOT['spacing'])
        sites.append(q); level.append(np.full(len(q), k))
    sites, level = np.vstack(sites), np.concatenate(level)
    nearest = lambda xy: trimesh.proximity.closest_point(seat, np.c_[xy, np.full(len(xy), FOOT['thick']/2)])[:2]
    anchor, length = nearest(sites)
    for _ in range(3):
        short = length < FOOT['least']*(1-1e-3)
        if not short.any(): break
        out = sites[short]-anchor[short, :2]
        out = np.where(np.linalg.norm(out, axis=1)[:, None] > 1e-9, out, sites[short]-centre); out /= np.linalg.norm(out, axis=1)[:, None]
        rise = np.clip(anchor[short, 2]-FOOT['thick']/2, 0., FOOT['least'])
        sites[short] = anchor[short, :2]+out*np.sqrt(FOOT['least']**2-rise**2)[:, None]; anchor, length = nearest(sites)
    ok = length >= FOOT['least']*(1-1e-3); sites, level, anchor, length = sites[ok], level[ok], anchor[ok], length[ok]
    ground_r, upper = widths(anchor-np.c_[sites, np.zeros(len(sites))], FOOT)
    cost = np.maximum(length, FOOT['least'])*(ground_r+upper)+np.pi*ground_r**2
    for top in range(FOOT['first']-1, len(FOOT['outward'])):
        use = np.flatnonzero(level <= top); picked = enclosing(sites[use], cost[use], hull)
        if picked is not None: picked = use[picked]; break
    else: raise ValueError('No cycle of candidates holds the pressure hull')
    feet = sites[picked]

    # The bottom row: Step5's flat cones, at the cheapest cycle and at two placed by hand.
    def around(angles, out=0.):
        u = np.c_[np.cos(np.radians(angles)), np.sin(np.radians(angles))]; r = np.linalg.norm(hull-centre, axis=1).max()
        while not Polygon(centre+r*u).buffer(-FOOT['outward'][0]).contains(Polygon(hull)): r += .005
        xy = centre+(r+out)*u; return xy, nearest(xy)[0]
    layouts = [(feet, anchor[picked]), around(camera+np.r_[0, 120, 240]), around(camera+np.r_[45, 135, 225, 315], WIDE)]
    row = dict(world_from_support=np.eye(4), material=np.eye(4), hull=hull)
    supports = [solid(stock, [plate(row, s, q, FOOT)[0] for s, q in zip(*layout)], sweep) for layout in layouts]
    L, H, N = setup['needle_length_m'], setup['head_radius_m'], setup['needle_radius_m']
    axis = tip-np.linspace(0, L+.07, 171)[:, None]*DOWN
    for support in supports:
        if Delaunay(ground(support)[ConvexHull(ground(support)).vertices]).find_simplex(cop).min() < 0:
            raise ValueError('A centre of pressure leaves the feet')
        solids = np.vstack([trimesh.sample.sample_surface_even(x, n, seed=0)[0] for x, n in ((m, 60000), (support, 60000))])
        if (inside(solids, tip, tip-L*DOWN, N).any() or inside(solids, tip-L*DOWN, tip-(L+.07)*DOWN, H).any()
                or m.contains(axis).any() or support.contains(axis).any()):
            raise ValueError('The needle hits the bunny or a support')

    wire = cone(tip-L*DOWN, DOWN, L, CONE)
    fit = np.vstack([m.vertices, *(x.vertices for x in supports), *(q for s in wire for q in s)])
    right = np.cross([0, 0, 1.], view); right /= np.linalg.norm(right); up = np.cross(view, right)
    pixel = max(np.ptp(fit@right), np.ptp(fit@up)*SIZE[0]/SIZE[1])*MARGIN/SIZE[0]   # m a px, as Scene.fit
    cloud = marks(cop[rng.choice(len(cop), DOTS, replace=False)], DOT_COLOR, right, pixel)
    dotted = marks(resample(hull, MARK_PX*pixel), DOT_COLOR, right, pixel)
    cycle = [(np.r_[q, 0.], np.r_[w, 0.]) for q, w in zip(feet, np.roll(feet, -1, 0))]
    reach_lines = [(np.r_[s, FOOT['thick']/2], q) for s, q in zip(*layouts[0])]
    colors = np.tile(GRAY, (len(source.faces), 1))
    panels = [([(seat, BLUE)], [cloud, lines(dashes(action, .006, .004), GOLD)], wire),
              ([(seat, BLUE)], [cloud, dotted]),
              ([(seat, BLUE), (discs(feet, FOOT['tip']), BLUE)], [dotted, lines(dashes(cycle+reach_lines, .006, .004), BLUE)])]
    panels += [([(x, BLUE)], []) for x in supports]
    images = [render(source, setup, view, fit, W, colors, [], *p) for p in panels]
    out = Image.new('RGB', (3*(SIZE[0]+GAP_PX)+GAP_PX, 2*(SIZE[1]+GAP_PX)+GAP_PX), 'white')
    for k, image in enumerate(images): out.paste(image, (GAP_PX+k % 3*(SIZE[0]+GAP_PX), GAP_PX+k//3*(SIZE[1]+GAP_PX)))
    out.save(HERE/'grounded.png')
    print('RENDERED', HERE/'grounded.png', 'feet', [len(xy) for xy, _ in layouts])

if __name__ == '__main__': main()
