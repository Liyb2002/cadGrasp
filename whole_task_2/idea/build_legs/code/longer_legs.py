"""Longer legs, before and after, in grounded.py's scene: its bunny, cup, loads, feet and camera (a schematic, not to
scale).

The support rests on two bases: on table 1 as in grounded.py, and turned ALPHA about a level axis onto table 2
(rotational reuse onto another face).  In the support's frame a table is a plane n.x = d+L: d where it touches the
support (or the bunny) in that installation, L how much longer the legs are, the same for both tables.  A foot of
one table must stay above the other table, since the support turns onto it, so on each table only the side of
the crease (the line where the two planes meet) toward the support is legal.  The crease runs along the line of
sight (BETA), so the other table is seen nearly edge-on, as a line.  Each installation's loads are grounded.py's
in its own world (the weight plus a push of PUSH in the CONE about straight down, on the area about the top), its
centres of pressure drawn DRAWN times as far from their mean; its feet are grounded.py's (combined.feet): the
cheapest cycle of candidates about the pressure hull, among the legal ones.

Top row L = 0; bottom row the least L (combine.lowered: searched over the bunny's diagonal, then bisected) at which
every table's first ring of candidate feet (FOOT['outward'][0] about its pressure hull, combine.floor_gap) stays
above the other table.  The bunny stays where it is; the tables move.  Columns: on table 1 (TABLE_1), the cup,
both tables (see-through, their legal parts, the crease dark), the pressure hull dotted and its first ring of
candidate feet (the tips, blue; RED: below the other table); the support with its feet (Step5's flat cones); then
the same turned onto table 2 (TABLE_2).  With no legal cycle (top row) a table's cycle is chosen among all its
candidates, to show where its feet would have to stand: those below the other table are RED.

Run from the repository root:
  PYTHONPATH=.:whole_task_2/idea/easy_task/code:whole_task_2/idea/build_legs/code .venv/bin/python -B whole_task_2/idea/build_legs/code/longer_legs.py
"""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from whole_task_2.step1.common import ROOT, load_shape
from whole_task_2.step1.reach_video import Scene
from whole_task_2.step1.find_tasks_figure import Opacity
from whole_task_2.step3.render_independent import BLUE
from whole_task_2.step5.video.raster import GRAY
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step4.combine import exact_sweep
from whole_task_2.step4.skin import transform
from whole_task_2.step5.combined import FOOT, enclosing, plate, widths, resample, joined
from vertical import LIFT, DOWN, FACE, TURN, first_hit, spin
from slanted import INFLATE
from contrast_vertical import BIG, DEPTH
from grounded import cup, loads, discs, lines, ground, DRAWN, DOT_PX, MARK_PX, DOT_COLOR, RISE, MARGIN, SIZE, GAP_PX

HERE = Path(__file__).resolve().parents[1]
ALPHA = 80.                 # deg: table 2, the support turned this far onto it
BETA = 180.                 # deg: the way it turns, from the camera's right toward the back (its left: the
                            # crease runs along the line of sight, table 2 seen nearly edge-on)
TABLE_1 = (.66, .72, .80)   # see-through tables
TABLE_2 = (.86, .78, .62)
SEE = .35                   # their opacity
RED = (.80, .22, .18)       # a foot below the other table
CREASE = (.22, .22, .22)
SPAN = .025                 # m: a table drawn this far past what stands on it
INSIDE = .0004              # m: marks on a table lifted this far off it


def turn(a, b):
    """The least rotation taking unit a to unit b."""
    v = np.cross(a, b); c = float(a@b)
    if np.linalg.norm(v) < 1e-12: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3)+K+K@K/(1+c)


class Table:
    """A table in the support's frame: n.x = d+L, n out of the support.  T: its installation's world <- support (the
    table at z = 0)."""
    def __init__(self, n, d, L):
        self.n, self.e = n, d+L; self.T = np.eye(4); self.T[:3, :3] = turn(n, DOWN); self.T[2, 3] = self.e
        self.inv = np.linalg.inv(self.T)

    def below(self, x):
        """How far (m) support-frame points x are below this table."""
        return x@self.n-self.e


def pressure_hull(m, table):
    """grounded.py's pressure hull in table's installation (its world xy), drawn DRAWN times as far from the mean."""
    mk = m.copy(); mk.apply_transform(table.T); com = mk.center_mass
    top = np.r_[com[:2], first_hit(mk, [[*com[:2], com[2]+1]], DOWN)[0]]
    area = (mk.face_normals[:, 2] > 0) & (np.linalg.norm(mk.triangles_center-top, axis=1) < BIG)
    _, _, wrench = loads(mk, area, com, np.random.default_rng(0)); cop = G.pressure_points(wrench, com)
    cop = cop.mean(0)+DRAWN*(cop-cop.mean(0)); return cop[ConvexHull(cop).vertices]


def floor_gap(m, tables):
    """combine.floor_gap with the first ring of candidate feet (ring) for the grown hull: the lowest height (m) of a
    foot's tip on it above the other table; 0 when none is below."""
    return min(0., *(-float(depth(t, [o for o in tables if o is not t], ring(pressure_hull(m, t))).max()) for t in tables))


def lowered(m, normals, d):
    """combine.lowered: the least L, the same for every table, at which floor_gap holds (searched over the bunny's
    diagonal in 20 steps, then bisected 8 times)."""
    at = lambda L: [Table(n, e, L) for n, e in zip(normals, d)]
    if floor_gap(m, at(0.)) >= -1e-8: return 0.
    grid = np.linspace(0., float(np.linalg.norm(m.extents)), 21)
    for low, high in zip(grid[:-1], grid[1:]):
        if floor_gap(m, at(high)) < -1e-8: continue
        for _ in range(8):
            mid = (low+high)/2
            if floor_gap(m, at(mid)) >= -1e-8: high = mid
            else: low = mid
        return high
    raise ValueError('No L keeps the rings of feet above the other table')


def depth(table, others, sites):
    """How far (m) a foot at each site (table's world xy) goes below another table: its tip, a disc FOOT['tip'] in
    radius and FOOT['thick'] high; a foot is legal when it is not below (<= 0)."""
    phi = np.linspace(0, 2*np.pi, 16, endpoint=False); rim = FOOT['tip']*np.c_[np.cos(phi), np.sin(phi)]
    tip = transform(np.vstack([np.c_[(s+rim), np.full(len(rim), z)] for s in sites for z in (0., FOOT['thick'])]), table.inv)
    return np.max([o.below(tip).reshape(len(sites), -1).max(1) for o in others], axis=0)


def ring(hull):
    """The first ring of candidate feet about hull (combined.feet), table's world xy."""
    return resample(np.asarray(orient(Polygon(hull).buffer(FOOT['outward'][0], join_style=2), sign=1.).exterior.coords)[:-1], FOOT['spacing'])


def feet(seat, table, others, hull):
    """grounded.py's feet on table about hull (combined.feet): (sites world xy, anchors support frame, legal), the
    cheapest cycle of the legal candidates, else (none holding the hull) of all of them."""
    sites, level = [], []
    for k, d in enumerate(FOOT['outward']):
        q = resample(np.asarray(orient(Polygon(hull).buffer(d, join_style=2), sign=1.).exterior.coords)[:-1], FOOT['spacing'])
        sites.append(q); level.append(np.full(len(q), k))
    sites, level = np.vstack(sites), np.concatenate(level)
    nearest = lambda xy: trimesh.proximity.closest_point(seat, transform(np.c_[xy, np.full(len(xy), FOOT['thick']/2)], table.inv))[:2]
    anchor, length = nearest(sites)
    for _ in range(3):
        short = length < FOOT['least']*(1-1e-3)
        if not short.any(): break
        above = transform(anchor[short], table.T); out = sites[short]-above[:, :2]
        out = np.where(np.linalg.norm(out, axis=1)[:, None] > 1e-9, out, sites[short]-hull.mean(0)); out /= np.linalg.norm(out, axis=1)[:, None]
        rise = np.clip(above[:, 2]-FOOT['thick']/2, 0., FOOT['least'])
        sites[short] = above[:, :2]+out*np.sqrt(FOOT['least']**2-rise**2)[:, None]; anchor, length = nearest(sites)
    ok = length >= FOOT['least']*(1-1e-3); sites, level, anchor, length = sites[ok], level[ok], anchor[ok], length[ok]
    legal = depth(table, others, sites) <= 0; ground_r, upper = widths(transform(anchor, table.T)-np.c_[sites, np.zeros(len(sites))], FOOT)
    cost = np.maximum(length, FOOT['least'])*(ground_r+upper)+np.pi*ground_r**2
    for among in (legal, np.ones(len(sites), bool)):
        for top in range(FOOT['first']-1, len(FOOT['outward'])):
            use = np.flatnonzero(among & (level <= top))
            picked = enclosing(sites[use], cost[use], hull) if len(use) >= 3 else None
            if picked is not None: picked = use[picked]; return sites[picked], anchor[picked], legal[picked]
    raise ValueError('No cycle of candidates holds the pressure hull')


def build(stock, sweep, tables, plans):
    """The support (the cup and its legal feet, each cut by every table) and the feet below the other table (each
    cut by its own), all minus the bunny swept straight up, support frame."""
    good, bad = [stock], []
    for t, (sites, anchor, legal) in zip(tables, plans):
        row = dict(world_from_support=t.T, material=t.inv, hull=None)
        for s, a, ok in zip(sites, anchor, legal): (good if ok else bad).append(plate(row, s, a, FOOT)[0])
    support = G.union(good)
    for t in tables: support = support.trim_by_plane(tuple(-t.n), -t.e/G.SCALE)
    support = support-sweep
    if not joined(support): raise ValueError('A support is not one piece')
    return G.mesh(support), [G.mesh(f-sweep) for f in bad]


class Corner:
    """Where both tables are drawn: the crease (a point p, direction c) and, on each table, the direction u away
    from the crease toward the support."""
    def __init__(self, tables, centre):
        n1, n2 = tables[0].n, tables[1].n; self.c = np.cross(n1, n2); self.c /= np.linalg.norm(self.c)
        self.p = np.linalg.solve(np.array([n1, n2, self.c]), [tables[0].e, tables[1].e, self.c@centre])
        self.u, self.n = [], [t.n for t in tables]
        for t, o in zip(tables, tables[::-1]):
            u = -(o.n-(o.n@t.n)*t.n); self.u.append(u/np.linalg.norm(u))

    def quad(self, k, a, b):
        """Table k over a (along the crease) and b (away from it), two triangles facing the support, support frame."""
        P = np.array([self.p+x*self.c+y*self.u[k] for x, y in ((a[0], b[0]), (a[1], b[0]), (a[1], b[1]), (a[0], b[1]))])
        faces = np.array([[0, 1, 2], [0, 2, 3]]); n = np.cross(self.c, self.u[k])
        return trimesh.Trimesh(P, faces if n@self.n[k] < 0 else faces[:, ::-1], process=False)


def place(scene, target, width):
    """Scene.fit with a given target and screen width (m)."""
    right = np.cross([0, 0, 1.], scene.view); right /= np.linalg.norm(right); up = np.cross(scene.view, right)
    w = width/MARGIN/2; h = w*SIZE[1]/SIZE[0]
    scene.fit(np.array([target+x*right+y*up for x in (-w, w) for y in (-h, h)]), MARGIN, target)


def render(source, setup, view, target, width, W, items, ghosts, segments):
    """One panel: the bunny (placed by W), items and see-through ghosts (mesh, colour), the segments (pairs, ink)."""
    scene = Scene(source, setup, *SIZE)
    try:
        r = scene.r; scene.view = view; place(scene, target, width)
        shape = lambda x, c: (r.Shape(scene.ctx, scene.shader, x.vertices, x.faces, c, metallic=.05, roughness=.5), np.eye(4))
        compositor = Opacity(scene)
        try: data = compositor.draw(W, np.tile(GRAY, (len(source.faces), 1)), [shape(*x) for x in items], None,
                                    [shape(*x) for x in ghosts], opacity=SEE, segments=segments)
        finally: compositor.close()
    finally: scene.close()
    return Image.frombytes('RGB', SIZE, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def dots(P, right, pixel):
    """grounded.marks at 3D points P: lines DOT_PX long across the screen, drawn DOT_PX wide."""
    return np.stack([P-.5*DOT_PX*pixel*right, P+.5*DOT_PX*pixel*right], 1).reshape(-1, 3), DOT_COLOR


def moved(mesh, T):
    out = mesh.copy(); out.apply_transform(T); return out


def main():
    source = load_shape(ROOT/'objects/B/mesh.stl')
    setup = json.loads((ROOT/'whole_task_2/data/B+03_kuka/step1_reach/plan.json').read_text())['setup']
    setup['working_direction_world'] = DOWN.tolist()
    reach = np.asarray(setup['work_point_m'])[:2]-np.asarray(setup['base_position_world_m'])[:2]
    camera = np.degrees(np.arctan2(reach[1], reach[0]))+90+TURN
    view = np.r_[np.cos(np.radians(camera)), np.sin(np.radians(camera)), RISE]; view /= np.linalg.norm(view)
    right = np.cross([0, 0, 1.], view); right /= np.linalg.norm(right); up = np.cross(view, right)

    W = np.eye(4); W[:3, :3] = spin(camera-FACE)@[[1, 0, 0], [0, 0, -1], [0, 1, 0]]   # original +Y up
    m = source.copy(); m.apply_transform(W)
    shift = -m.center_mass; shift[2] = LIFT-m.bounds[0, 2]; W[:3, 3] = shift; m.apply_translation(shift)
    grown = trimesh.Trimesh(m.vertices+INFLATE*m.vertex_normals, m.faces, process=False)
    sweep = exact_sweep(grown, -DOWN, DEPTH+.1); stock = cup(m); seat = G.mesh(stock-sweep)

    # The two tables: out of the support straight down, and ALPHA from it toward h; each touching the support or
    # the bunny, whichever reaches further.
    back = -np.r_[view[:2], 0.]; back /= np.linalg.norm(back)
    h = np.cos(np.radians(BETA))*right+np.sin(np.radians(BETA))*back
    normals = [DOWN, np.cos(np.radians(ALPHA))*DOWN+np.sin(np.radians(ALPHA))*h]
    d = [max(float((seat.vertices@n).max()), float((m.vertices@n).max())) for n in normals]
    rows = []
    for L in (0., lowered(m, normals, d)):
        tables = [Table(n, e, L) for n, e in zip(normals, d)]
        hulls = [pressure_hull(m, t) for t in tables]
        plans = [feet(seat, t, [o for o in tables if o is not t], hull) for t, hull in zip(tables, hulls)]
        support, bad = build(stock, sweep, tables, plans)
        for t, hull in zip(tables, hulls):
            g = ground(moved(support, t.T)); g = np.vstack([g, *(ground(moved(f, t.T)) for f in bad)])
            if Delaunay(g[ConvexHull(g).vertices]).find_simplex(hull).min() < 0: raise ValueError('A centre of pressure leaves the feet')
        rows.append(dict(L=L, tables=tables, hulls=hulls, plans=plans, support=support, bad=bad, corner=Corner(tables, m.center_mass)))
        print(f'L {1000*L:.1f} mm, legal feet', [f'{int(p[2].sum())}/{len(p[2])}' for p in plans])

    # The tables' extent: what stands on them, both rows, SPAN further; drawn only on the support's side of the
    # crease.
    def extent(row, k):
        t, C = row['tables'][k], row['corner']
        on = [transform(np.c_[x, np.zeros(len(x))], t.inv) for x in (row['hulls'][k], row['plans'][k][0], ring(row['hulls'][k]))]
        shadow = np.vstack([m.vertices, row['support'].vertices]); shadow = shadow-np.outer(t.below(shadow), t.n)
        x = np.vstack([*on, shadow])-C.p; return x@C.c, x@C.u[k]
    for row in rows:
        ab = [extent(row, k) for k in (0, 1)]; a = np.r_[ab[0][0], ab[1][0]]; row['a'] = (a.min()-SPAN, a.max()+SPAN)
        row['quads'] = [row['corner'].quad(k, row['a'], (0., b.max()+SPAN)) for k, (_, b) in enumerate(ab)]

    panels = []
    for row in rows:
        tables, C, support, bad = row['tables'], row['corner'], row['support'], row['bad']
        crease = lines([(C.p+row['a'][0]*C.c, C.p+row['a'][1]*C.c)], CREASE)
        hull3 = [transform(np.c_[x, np.full(len(x), INSIDE)], t.inv) for x, t in zip(row['hulls'], tables)]
        ghosts = list(zip(row['quads'], (TABLE_1, TABLE_2)))
        # On table 1, then turned onto table 2: the cup, both tables, the table's pressure hull and its first ring of
        # candidate feet; then the support with its feet.
        for k, (t, hull) in enumerate(zip(tables, row['hulls'])):
            T = np.eye(4); T[:3, :3] = t.T[:3, :3]                         # turned, not moved: the bunny stays, the tables move
            q = ring(hull); legal = depth(t, [o for o in tables if o is not t], q) <= 0
            tips = [(moved(discs(q[ok], FOOT['tip']), t.inv), c) for ok, c in ((legal, BLUE), (~legal, RED)) if ok.any()]
            panels.append(dict(T=T, items=[(seat, BLUE), *tips], ghosts=ghosts, hulls=[hull3[k]], segments=[crease]))
            panels.append(dict(T=T, items=[(support, BLUE), *((f, RED) for f in bad)], ghosts=ghosts, hulls=[hull3[k]], segments=[crease]))
    for p in panels:
        T = p.pop('T'); p['W'] = T@W
        p['items'] = [(moved(x, T), c) for x, c in p['items']]; p['ghosts'] = [(moved(x, T), c) for x, c in p['ghosts']]
        p['hulls'] = [transform(x, T) for x in p['hulls']]; p['segments'] = [(transform(x, T), c) for x, c in p['segments']]

    # One scale for every panel; each column centred on both its rows.
    def screen(p):
        x = np.vstack([moved(m, p['W']@np.linalg.inv(W)).vertices, *(i.vertices for i, _ in p['items']), *(g.vertices for g, _ in p['ghosts'])])
        return x@right, x@up
    xy = [screen(p) for p in panels]
    n = len(panels)//2
    width = max(max(np.ptp(np.r_[xy[k][0], xy[k+n][0]]), np.ptp(np.r_[xy[k][1], xy[k+n][1]])*SIZE[0]/SIZE[1]) for k in range(n))
    pixel = width/SIZE[0]
    images = []
    for k, p in enumerate(panels):
        x = np.r_[xy[k % n][0], xy[k % n+n][0]]; y = np.r_[xy[k % n][1], xy[k % n+n][1]]
        target = right*(x.min()+x.max())/2+up*(y.min()+y.max())/2
        dotted = [dots(resample(hull, MARK_PX*pixel), right, pixel) for hull in p['hulls']]
        images.append(render(source, setup, view, target, width, p['W'], p['items'], p['ghosts'], dotted+p['segments']))
    out = Image.new('RGB', (n*(SIZE[0]+GAP_PX)+GAP_PX, 2*(SIZE[1]+GAP_PX)+GAP_PX), 'white')
    for k, image in enumerate(images): out.paste(image, (GAP_PX+k % n*(SIZE[0]+GAP_PX), GAP_PX+k//n*(SIZE[1]+GAP_PX)))
    out.save(HERE/'longer_legs.png')
    print('RENDERED', HERE/'longer_legs.png', 'L', [round(1000*row['L'], 1) for row in rows])

if __name__ == '__main__': main()
