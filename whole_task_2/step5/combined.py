"""Step5: every distinct support of step4/combine.py rebuilt as a frame, with its feet, certified for every task, and
its figure.

The placing arm installs the support for each task so that the task faces the work arm at its certified
pose, and the support rests on the table.

The frame (step5/frame.py): Step4's skin kept as big contact faces.  The peel deletes from the outside in (plane
cuts toward the bunnies, then boxes), each deletion kept when every task's paired forces and moments still hold and
it leaves no skin narrower than WIDTH where it was not; the narrow pieces left (the whiskers) go where the loads
let them; every seat's faces are then rebuilt from its layer with round edges (no cut steps, flaps or curtains)
and no holes, and faces left apart are joined by straight rods on cones.

Then feet (as Co-optimize's Step5, "wrap the pressure hull"): the ground faces enclose a task's pressure
hull when their convex hull does, so a few feet do, not a band around it.  The support turns and slides on
one table (every installation has the same table plane in the support frame), so it stands on the same ground
faces in every installation: the feet are planned once, about the convex hull of every task's pressure hull the
actual ground faces do not already enclose (all in one installation), not task by task (which left feet of two
tasks side by side, a fork):

  candidate feet stand on the table every FOOT['spacing'] around that hull, centred FOOT['outward'] outside it
  (a stance wider than the hull), where the tip of a foot (a disc FOOT['tip'] in radius, FOOT['thick'] high)
  fits in the legal space -- above every installation's floor, out of every task's body, keep-out and exit sweep
  (Step4's forbidden region), so a foot never blocks another task;
  a foot is one flat cone, straight from the support's point nearest its site down to the site (when the legal
  space cuts that leg off, from the next nearest point that keeps it: strut): the hull of two
  round-edged discs FOOT['thick'] thick in one plane (the plane through the leg, level across it), FOOT['tip'] in
  radius, the one on the table wider by FOOT['taper'] of the height it holds up, the one at the support by that of
  the reach (at most FOOT['base']: widest where it bears); a candidate the support comes nearer than FOOT['least']
  moves out until its leg is that long, else is none (no stubs: a low leg lies on the table); cut flat by the table,
  clipped to the legal space;
  the feet are the cheapest cycle of candidates (and of the actual ground faces' corners, free) about the
  hull with the hull left of every step, so their tips' convex hull holds it; a foot costs its material (its
  plate's area); the candidates nearest the hull first (the first FOOT['first'] rings together, then each further
  ring);
  a foot's leg may touch the object in another seat: the skin narrower than WIDTH it adds there is cut from it
  (as Step4 cuts thin skin);
  a foot the legal space or that cut cuts off the support or its tip is dropped and the cycle chosen again.

The whole frame and every foot stay.  As before every Step4 audit, the
original face cells at each task's final exit travel are recut (MembraneModel.refine_exits), removing the
Boolean slivers (~1e-10 m^3) the joins can leave on an exit.  Every task is then audited in full on
the support with its feet (more than 99% of the 32,768 paired samples, the continuous exit of
every source face, body and keep-out, the KUKA at every stored joint set), and in every installation the
actual ground faces must enclose the task's pressure hull.  Validation rebuilds every support from Step4's
membrane (frame and feet) and audits it again.

For the real station, next to every figure a folder output/<object>/step5/<name>/ (package): the support and
the object as STL (mm), info.json and tasks.npz.  support.stl is the certified support enlarged PRINT along x,
y and z, so the object has clearance in its seat; every installation is the certified one scaled about the
point of its ground under the object's centre (the support stands where it stood), the object centred in its seat.  The
certificates are the certified support's.  A run overwrites the last one: it first removes every earlier
result (clear()), so step5/ holds this run's results only, with no copies.

One figure per support, output/<object>/step5/<name>.png, no text: the top row, the support with the bunny
in each seat (pink: the areas of the seat's tasks), then the support alone; the bottom row, every task at
the station, the support as the placing arm installs it and the KUKA holding the needle on the task's
central certified face.  Beside it, Step4's support the result starts from, output/<object>/step4/<name>.png,
drawn the same way.
"""
import argparse
import json
import re
import shutil
import time
import numpy as np
import trimesh
import manifold3d as md
from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from scipy.spatial import ConvexHull

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.membranes import positive_components
from whole_task_2.step4.ground_oracle import pressure_rows, ground_enclosure
from whole_task_2.step4.seats import context, read_mesh
from whole_task_2.step4.merge import move
from whole_task_2.step4.combine import FOLDER as COMBINE, ExitModel, largest, plain, seats, solids, obb
from whole_task_2.step4.skin import WIDTH, SEAM, transform, piece, measure, legal, skin, regions, cut
from whole_task_2.step5 import peel as P
from whole_task_2.step5 import frame as FR

FOLDER = 'step5_combined'
FOOT = dict(thick=.006,                              # flat cones this thick, straight from the support's nearest point
            tip=.008,                                # to the table, this big in radius at either end, widened by
            taper=.35, base=.020,                    # this share of the height (ground end) or reach (support end), at most this
            least=WIDTH,                             # legs at least this long
            outward=(.012, .024, .036, .048, .064),  # candidate feet centred this far outside the pressure hull
            spacing=.012,                            # and this far apart along it
            first=2,                                 # the nearest rings tried together first
            attempts=24,                             # cycles chosen again after feet the legal space cut off
            padding=.03)                             # the legal space: a box this much around the farthest feet
PRINT = 1.01                                         # support.stl: the certified support enlarged this much


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


CRUMB = 1e-4                                         # pieces below this share of the material are export crumbs


def count(value):
    """The pieces of value, export crumbs (below CRUMB of the material) not counted."""
    parts = [float(p.volume()) for p in positive_components(value)]
    return sum(v >= CRUMB*sum(parts) for v in parts)


def joined(value):
    """Is value one piece, also as the mesh it is saved as (a piece touching it only along a seam parts there)?"""
    return count(value) == 1 and count(G.solid(G.mesh(value.simplify(0.)))) == 1


def resample(ring, spacing):
    """Points every spacing (at most) along a closed polyline, its corners among them."""
    out = []
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        n = max(int(np.ceil(np.linalg.norm(b-a)/spacing)), 1); out.append(a+(b-a)*np.arange(n)[:, None]/n)
    return np.vstack(out)


def enclosing(points, cost, hull, margin=1e-6):
    """The cheapest of points (each its cost) whose ground holds hull: a cycle about the hull with the hull
    left of every step (by margin).  Their ids, or None."""
    centre = hull.mean(0); order = np.argsort(np.arctan2(points[:, 1]-centre[1], points[:, 0]-centre[0]), kind='stable')
    P, w = points[order], cost[order]; n = len(P); d = P[None]-P[:, None]; h = hull[None, None]-P[:, None, None]
    length = np.linalg.norm(d, axis=-1)
    left = (d[..., None, 0]*h[..., 1]-d[..., None, 1]*h[..., 0] >= margin*length[..., None]).all(-1) & (length > 1e-12)
    # Every step of a cycle turns at most `span` about the hull's centre, so a cycle passes through every window
    # that wide: its points are the only starts needed, the sparsest window's.
    angle = np.arctan2(P[:, 1]-centre[1], P[:, 0]-centre[0]); i, j = np.nonzero(left)
    if not len(i): return None
    span = float(np.mod(angle[j]-angle[i], 2*np.pi).max())+1e-12
    within = (np.mod(angle[None]-angle[:, None], 2*np.pi) <= span) & ~np.eye(n, dtype=bool)
    starts = np.flatnonzero(within[within.sum(1).argmin()]); k = len(starts)
    if not k: return None
    dp = np.full((k, n+1), np.inf); back = np.full((k, n+1), -1); dp[:, 0] = w[starts]
    for j in range(1, n+1):
        uj = (starts+j) % n; ui = (starts[:, None]+np.arange(j)[None]) % n
        m = np.where(left[ui, uj[:, None]], dp[:, :j], np.inf); i = m.argmin(1); v = m[np.arange(k), i]
        dp[:, j] = v+(w[uj] if j < n else 0.); back[:, j] = np.where(np.isfinite(v), i, -1)
    b = int(dp[:, n].argmin())
    if not np.isfinite(dp[b, n]): return None
    ids, j = [], back[b, n]
    while j > 0: ids.append((starts[b]+j) % n); j = back[b, j]
    return order[[starts[b]]+ids[::-1]]


def section(row, legal, foot=FOOT):
    """The legal space through a tip's height on a task's table (its world xy, manifold units)."""
    world = move(row['world_from_support'], legal); h = foot['thick']/G.SCALE
    return world.slice(1e-3*h) ^ world.slice((1-1e-3)*h)


def own_thin(c, base, part):
    """part (a foot) without the skin narrower than WIDTH it adds over a seat (a leg touching the object in another
    seat; base: regions() of the support it joins), cut as Step4 cuts thin skin (leg_skin, for one foot)."""
    if part.is_empty(): return part
    S, _ = skin(c); low, high = G.mesh(part).bounds
    for X, R, own in base:
        mid = transform(S.mid, X); near = np.flatnonzero(~R & ((mid >= low) & (mid <= high)).all(1))
        hit = near[P.probed(part, mid[near])]
        if not len(hit): continue
        grown = R.copy(); grown[hit] = True; ids = np.flatnonzero(S.thin(grown, own=own) & ~R)
        if len(ids): part = cut(c, part, X, ids)
    return part


def widths(d, foot=FOOT):
    """The radii of a foot's ends, d (world) from its site up to its anchor: (on the table, at the support).  A foot
    is widest where it bears: its end on the table widens with the height it holds up (a tall foot stands like a
    cone on its base), its end at the support with the reach it spans (a low foot lies on the table like the
    reference's ground plates, tapering away from the support)."""
    grow = lambda x: np.minimum(foot['base'], foot['tip']+foot['taper']*np.maximum(x, 0.))
    return grow(d[..., 2]), grow(np.linalg.norm(d[..., :2], axis=-1))


def plate(row, site, anchor, foot=FOOT):
    """A flat cone from anchor (support frame, the support's point nearest the site) straight down to site (world xy
    of the task's table): the hull of two discs FOOT['thick'] thick with round edges, in the plane through the leg
    that is level across it (a plate lying on the table when the leg is low), their radii widths(), the one at the
    anchor sunk half the thickness into the support; cut flat by the table.  (the plate, the part of it that must
    stay: its disc FOOT['tip'] in radius at the site), support frame, manifold units."""
    a = transform(anchor[None], row['world_from_support'])[0]; s = np.r_[site, 0.]; d = a-s; length = np.linalg.norm(d)
    inward = np.r_[d[:2], 0.] if np.linalg.norm(d[:2]) > 1e-9 else np.r_[np.asarray(row['hull']).mean(0)-site, 0.]
    u = np.cross([0., 0., 1.], inward); u /= np.linalg.norm(u)
    n = np.cross(d, u) if length > foot['thick'] else np.array([0., 0., 1.]); n /= np.linalg.norm(n); v = np.cross(n, u)
    t = foot['thick']/2; phi = np.linspace(0., 2*np.pi, 64, endpoint=False)[:, None, None]
    psi = np.linspace(-np.pi/2, np.pi/2, 13)[None, :, None]
    table = row['world_from_support'][2]                           # world z of a support-frame point: table[:3]@p+table[3]

    def disc(centre, radius):
        p = (centre+(radius-t+t*np.cos(psi))*(np.cos(phi)*u+np.sin(phi)*v)+t*np.sin(psi)*n).reshape(-1, 3)
        return transform(p, row['material'])/G.SCALE

    def hull(points):
        return md.Manifold.hull_points(np.vstack(points)).trim_by_plane(tuple(table[:3]), -table[3]/G.SCALE)
    ground, top = widths(d, foot); keep = disc(s, foot['tip'])
    return hull([disc(s, ground), disc(a+t*d/max(length, 1e-9), top), keep]), hull([keep])


def foot_at(row, site, anchor, legal, stock, foot=FOOT, thin=lambda part: part):
    """A foot on a task's table at site (world xy): the flat cone (plate) from anchor, in the legal space, without
    the thin skin it adds (thin), support frame; None if the legal space or that cut cuts its tip or cuts it off
    the stock."""
    whole, tip = plate(row, site, anchor, foot)
    parts = positive_components(thin(whole ^ legal))
    if not parts: return None
    part = max(parts, key=lambda q: G.volume(q ^ tip))
    return part if G.volume(tip-part) <= 1e-3*G.volume(tip) and joined(stock+part) else None


def strut(row, site, anchor, mesh, legal, stock, foot=FOOT, thin=lambda part: part, tries=6):
    """foot_at from the support's point nearest the site, else from its next nearest vertices (each at least two
    tips from those tried, at most `tries`): still one straight leg, around another seat's keep-out instead of
    through it (B+big_01 B1's table: the leg from the nearest point crossed A1's tool space and was cut off)."""
    out = foot_at(row, site, anchor, legal, stock, foot, thin)
    if out is not None: return out
    p = transform(np.r_[site, foot['thick']/2][None], row['material'])[0]; tried = [np.asarray(anchor)]
    for v in np.argsort(np.linalg.norm(mesh.vertices-p, axis=1))[:512]:
        q = mesh.vertices[v]
        if min(np.linalg.norm(q-x) for x in tried) < 2*foot['tip']: continue
        tried.append(q); out = foot_at(row, site, q, legal, stock, foot, thin)
        if out is not None or len(tried) > tries: return out
    return None


def legal_space(c, group, q, points, pad):
    """Where material may go: in the box of points widened by pad, above every installation's floor, out of
    every task's body, keep-out and exit sweep (Step4's forbidden region)."""
    return legal(points, pad, q.material, solids(c.object_name, group)['F'])


def settled(c, group, support):
    """support with the seams the cell prisms leave collapsed (SEAM) and the forbidden region cut again, so
    that it settles under further Booleans (as step4/skin.py's wide)."""
    return piece(G.solid(support).simplify(SEAM/G.SCALE)-solids(c.object_name, group)['F'], 'settling splits the support')


def grounds(c, model, group, membrane, legs=False):
    """What the peel keeps for the tables, by Step4's own stance check (step4/combine.foot_ring, the rule combine
    accepted the support by): (points of membrane it keeps, a check of every deletion or None).  Where no ring of feet
    encloses a task's pressure hull, the corners of its ground faces stay.  Elsewhere a deletion is kept while every
    table keeps its ring (the check: the peel is otherwise free; points kept at the legs' starts, on the support's rim,
    stopped every cut from outside in); with legs, the points the rings' legs start from stay instead (wrap's second
    try, when the feet do not stand on the free frame)."""
    from whole_task_2.step4.combine import foot_ring

    def rings(mesh):
        q = model.layout(mesh); candidate = dict(layout=q, support=mesh, construction={}); candidate['tasks'] = model.prepare(q)
        return foot_ring(c, group['slots'], model, candidate)
    failed, starts = rings(membrane); q = model.layout(membrane); out = [starts if legs else np.zeros((0, 3))]
    for r in pressure_rows(model, q):
        if r['id'] not in failed: continue
        corners = np.asarray(ground_enclosure(membrane, [r])[0]['ground_contact_hull_xy_m'], float).reshape(-1, 2)
        out.append(transform(np.c_[corners, np.zeros(len(corners))], r['material']))
    return np.vstack(out), None if legs else (lambda value: set(rings(G.mesh(value))[0]) <= set(failed))


def feet(c, model, group, membrane, foot=FOOT):
    """membrane with its feet joined to it: (the solid, the record of every table's feet, whether every pressure
    hull is enclosed).  The installations with one table plane in the support frame (here all) stand on the same
    ground faces, so their feet are planned once, about the convex hull of the pressure hulls (in the first one's
    world) of all their tasks the ground faces do not yet enclose.  The candidates nearest that hull are tried
    first: the first FOOT['first'] rings of FOOT['outward'] together, then each further ring in turn."""
    q = model.layout(membrane)
    rows = pressure_rows(model, q)
    legal = legal_space(c, group, q, np.vstack([membrane.vertices, *[r['ground_material'] for r in rows]]), max(foot['outward'])+foot['padding'])
    stock, share = largest(G.solid(membrane))
    if share < 1-1e-9: raise ValueError('Step4 membrane is not one piece')
    base = regions(c, group['slots'], membrane); thin = lambda part: own_thin(c, base, part)
    tables = []
    for r in rows:
        same = next((t for t in tables if np.allclose(t[0]['world_from_support'][2], r['world_from_support'][2], atol=1e-9)), None)
        if same is None: tables.append([r])
        else: same.append(r)
    record = []; enclosed = True; disc = md.CrossSection.circle(foot['tip']/G.SCALE, 32)
    for same in tables:
        r = same[0]; ids = [s['id'] for s in same]
        need = [s for s, g in zip(same, ground_enclosure(G.mesh(stock), same)) if not g['passed']]
        if not need: record.append(dict(tasks=ids, feet=[])); continue
        hull = np.vstack([transform(s['ground_material'], r['world_from_support'])[:, :2] for s in need]); hull = hull[ConvexHull(hull).vertices]
        corners = np.asarray(ground_enclosure(G.mesh(stock), [r])[0]['ground_contact_hull_xy_m'], float).reshape(-1, 2)
        sites, level = [np.empty((0, 2))], [np.empty(0, int)]
        for k, d in enumerate(foot['outward']):
            ring = orient(Polygon(hull).buffer(d, join_style=2), sign=1.)
            for g in getattr(ring, 'geoms', [ring]):
                if g.is_empty: continue
                p = resample(np.asarray(g.exterior.coords)[:-1], foot['spacing']); sites.append(p); level.append(np.full(len(p), k))
        sites, level = np.vstack(sites), np.concatenate(level); free = section(r, legal, foot); mesh = G.mesh(stock)
        nearest = lambda xy: trimesh.proximity.closest_point(mesh, transform(np.c_[xy, np.full(len(xy), foot['thick']/2)], r['material']))[:2]
        # A leg is at least FOOT['least'] long (no stub): a site the support comes nearer than that moves out, along
        # the level line from its anchor (or from the hull), until it is that far -- again from the new nearest
        # point, at most thrice; a site still nearer is no candidate.
        anchor, length = nearest(sites) if len(sites) else (np.empty((0, 3)), np.empty(0))
        for _ in range(3):
            short = length < foot['least']*(1-1e-3)
            if not short.any(): break
            above = transform(anchor[short], r['world_from_support']); out = sites[short]-above[:, :2]
            out = np.where(np.linalg.norm(out, axis=1)[:, None] > 1e-9, out, sites[short]-hull.mean(0)); out /= np.linalg.norm(out, axis=1)[:, None]
            rise = np.clip(above[:, 2]-foot['thick']/2, 0., foot['least'])
            sites[short] = above[:, :2]+out*np.sqrt(foot['least']**2-rise**2)[:, None]; anchor, length = nearest(sites)
        ok = (length >= foot['least']*(1-1e-3)) & np.array([(disc.translate(tuple(s/G.SCALE))-free).area() <= 1e-6*disc.area() for s in sites], bool).reshape(-1)
        sites, level, anchor, length = sites[ok], level[ok], anchor[ok], length[ok]
        # A foot's material, as its plate's area.
        ground, top = widths(transform(anchor, r['world_from_support'])-np.c_[sites, np.zeros(len(sites))], foot)
        points = np.vstack([corners, sites]); cost = np.r_[np.zeros(len(corners)), np.maximum(length, foot['least'])*(ground+top)+np.pi*ground**2]
        level = np.r_[np.full(len(corners), -1), level]
        built = {}; found = None
        for top in range(min(foot['first'], len(foot['outward']))-1, len(foot['outward'])):
            for _ in range(foot['attempts']):
                use = np.flatnonzero((level <= top) & np.isfinite(cost))
                picked = enclosing(points[use], cost[use], hull) if len(use) >= 3 else None
                if picked is None: break
                picked = use[picked]; new = [int(k) for k in picked if k >= len(corners)]
                for k in new:
                    if k not in built: built[k] = strut(r, sites[k-len(corners)], anchor[k-len(corners)], mesh, legal, stock, foot, thin)
                    if built[k] is None: cost[k] = np.inf
                if any(built[k] is None for k in new): continue
                grown = G.union([stock]+[built[k] for k in new])
                if joined(grown) and all(g['passed'] for g in ground_enclosure(G.mesh(grown), need)): found = grown, new, len(picked)-len(new), top
                break
            if found is not None: break
        if found is None: enclosed = False; record.append(dict(tasks=ids, feet=None)); continue
        stock = found[0]
        used = [int(k) for k in picked if k < len(corners)]; on = [k-len(corners) for k in found[1]]
        record.append(dict(tasks=ids, need=[s['id'] for s in need], ground_corners=found[2], outward_m=foot['outward'][found[3]],
                           feet=[dict(site_support_m=transform(np.c_[sites[on], np.zeros(len(on))], r['material'])[k].tolist(),
                                      leg_m=float(length[j])) for k, j in enumerate(on)],
                           # The support's points the feet stand on: each leg's anchor, each ground corner used.
                           held_m=np.vstack([anchor[on].reshape(-1, 3),
                                             transform(np.c_[corners[used], np.zeros(len(used))], r['material']).reshape(-1, 3)]).tolist()))
    return stock, record, enclosed


def leg_skin(c, group, membrane, support):
    """support without the skin narrower than WIDTH its feet make that the membrane has not (peel.fresh, every such
    cell: a leg touching the object in another seat, or widening what a corner of the membrane touches): the
    feet's own layer there cut, until none is left; the membrane's material stays.  (support, the area cut, m^2)"""
    _, outer = skin(c); base = G.solid(membrane); old = P.thin_cells(c, group, membrane, WIDTH/2); value = G.solid(support); area = 0.
    while True:
        found, a = P.fresh(c, group, old, value, made=True, pieces=False)
        if not a: break
        layer = G.union([P.prisms(c, [(X, cells)]) ^ move(X, outer) for X, cells in found if len(cells)])
        if (layer ^ value).is_empty() or G.volume((layer ^ value)-base) < c.base.tolerance: break      # the feet have none there
        value = value-(layer-base); area += a
    return (piece(value.simplify(SEAM/G.SCALE), 'cutting the thin skin of the feet splits the support') if area else support), area


def wrap(c, group, foot=FOOT):
    """The support peeled, with its feet: (support, model, layout, record of every task's feet, record of the
    steps: the peel and the skin's least width)."""
    model = ExitModel(c, group['slots']); start = read_mesh(ROOT/group['file']); steps = {}
    for legs in (False, True):
        # The peel free of the tables but for every ring of feet staying; where the feet then do not stand on the frame,
        # once more keeping the points Step4's rings' legs start from (grounds).
        ground, stance = grounds(c, model, group, start, legs)
        membrane, steps['frame'] = FR.frame(c, model, group, start, ground, solids(c.object_name, group)['F'], stance=stance)
        steps.update(membrane_cm3=float(membrane.volume*1e6), peeled_bbx_m3=obb(membrane.vertices), legs_kept=legs)
        support, record = feet_on(c, model, group, membrane, foot, steps)
        if all(r['passed'] for r in ground_enclosure(support, pressure_rows(model, model.layout(support)))): break
    width = measure(c, group['slots'], support); steps['too_thin_m2'] = P.too_thin(c, group, support)
    steps['skin_width_m'] = width[0] if np.isfinite(width[0]) else None
    return support, model, model.layout(support), record, steps


def feet_on(c, model, group, membrane, foot, steps):
    """wrap's feet on the frame membrane: (support, record)."""
    # Feet, then the cuts after them (the thin skin the legs add, the settling, Step4's exit recut); a task those
    # leave off its ground gets feet again (at most twice more).
    began = time.monotonic(); support, record = membrane, []
    for rounds in range(3):
        stock, more, _ = feet(c, model, group, support, foot); record += more if not rounds else [f for f in more if f.get('feet')]
        stock = piece(stock, 'the feet split the support')
        # The feet only add.  (A Boolean difference cannot show it: cell prisms leave seams where one of the membrane
        # even with itself differs by mm^3 either way; the audit below judges the support as it is.)
        if G.volume(G.solid(stock)) < G.volume(G.solid(support))-c.base.tolerance: raise ValueError('the feet lost membrane')
        support, cut = leg_skin(c, group, membrane, stock); steps['feet_thin_cut_m2'] = steps.get('feet_thin_cut_m2', 0.)+cut
        # The forbidden region cut again: a union can leave a contact face triangulated unlike the object (the same
        # vertices, other diagonals), a thin lens between the two that the audit counts as overlap.
        support = settled(c, group, support)
        # Step4's own recut before an audit: every original face cell at the final exit travel.
        candidate = dict(layout=model.layout(support), support=support, tasks=None, construction={})
        candidate['tasks'] = model.prepare(candidate['layout']); model.refine_exits(candidate)
        record.append(dict(exit_refinement_rounds=candidate['construction']['exit_refinement_rounds'],
                           removed_cm3=float(support.volume-candidate['support'].volume)*1e6))
        # The audit judges the support the recut checked last.  Every re-mesh can leave a sliver where two coincident
        # surfaces meet (an inverted shell the audit counts: B+8pose, 1.7e-9 m^3 in a body after piece(), 1.7e-8 m^3 in
        # an exit prism after settling again), so a support the recut left as it was is not re-meshed.
        if candidate['construction']['exit_refinement_rounds']: support = piece(G.solid(candidate['support']), 'the exit recut splits the support')
        q = model.layout(support)
        if all(r['passed'] for r in ground_enclosure(support, pressure_rows(model, q))): break
    steps.update(feet_seconds=time.monotonic()-began, feet_rounds=rounds+1)
    return support, record


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
                                             'step4/ground_oracle.py', 'step4/skin.py', 'step5/peel.py', 'step5/frame.py', 'step5/combined.py')]]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def clear(object_name, output_root=DEFAULT_OUTPUT):
    """Every result of an earlier run, so that this one overwrites them: the supports, report and validation under
    data/<object>/step5_combined/, the figures rNN.png and packages rNN/ under output/<object>/step5/ and Step4's
    figures rNN.png under output/<object>/step4/ (nothing else there is touched).  No copies are kept."""
    for path in [ROOT/'whole_task_2/data'/object_name/FOLDER, output_root/object_name/'step5', output_root/object_name/'step4']:
        for p in (path.iterdir() if path.exists() else []):
            if re.fullmatch(r'r\d\d(\.png|\.npz|\.obj)?', p.name) or p.name in ('report.json', 'validation.json'):
                shutil.rmtree(p) if p.is_dir() else p.unlink()


def step4_figure(c, group, path):
    """Step4's support of a result, drawn as render() draws Step5's."""
    membrane = read_mesh(ROOT/group['file']); model = ExitModel(c, group['slots'])
    render(group, model, model.layout(membrane), membrane, path)


def run(object_name='B', output_root=DEFAULT_OUTPUT):
    began = time.monotonic(); c = context(object_name); folder = ROOT/'whole_task_2/data'/object_name/FOLDER
    clear(object_name, output_root); folder.mkdir(parents=True, exist_ok=True); rows = []
    for n, group in enumerate(distinct(object_name)):
        name = f'r{n:02d}'; t0 = time.monotonic()
        row = dict(name=name, samples=group['samples'], step4_file=group['file'], step4_files=group['files'],
                   membrane_cm3=group['volume_cm3'], bbx_m3=group['bbx_m3'], seats=[len(ks) for _, ks in seats(group)],
                   joins=[dict(task=s['parents'][1], operation=s['operation'], seat=s['seat'], bbx_ratio=s['bbx_ratio'])
                          for s in group['chain']])
        step4_figure(c, group, output_root/object_name/'step4'/f'{name}.png')
        try:
            support, model, q, pieces, steps = wrap(c, group); row.update(pieces=pieces, **steps)
            proof, ground = certify(model, q, support)
        except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as ex:
            rows.append(dict(row, status='failed', reason=str(ex)[:300], seconds=time.monotonic()-t0))
            print('STEP5', name, 'FAILED', str(ex)[:160], flush=True); continue
        np.savez_compressed(folder/f'{name}.npz', vertices=support.vertices, faces=support.faces)
        (folder/f'{name}.obj').write_text(trimesh.exchange.obj.export_obj(support, digits=17, include_normals=False))
        rows.append(dict(row, status='passed', volume_cm3=float(support.volume*1e6),
                         feet_cm3=float(support.volume*1e6)-steps['membrane_cm3'], support_bbx_m3=obb(support.vertices),
                         bounds_m=support.bounds.tolist(), support_sha256=sha256(folder/f'{name}.npz'),
                         tasks=[dict(id=t['id'], samples=t['force']['passed_original_sample_count'],
                                     outer=t['force']['passed_continuous_outer_target_count'], arm_faces_hit=len(t['arm']['faces_hit']),
                                     exit_length_m=t['exit_length_m']) for t in proof['tasks']],
                         ground=[dict(id=r['id'], passed=r['passed'], pressure_hull_gap_m=r['pressure_hull_gap_m']) for r in ground],
                         seconds=time.monotonic()-t0))
        print('STEP5', name, 'PASS', round(support.volume*1e6, 1), 'cm3,', round(steps['frame']['peel']['removed_cm3'], 1), 'cm3 peeled (bbx',
              round(group['bbx_m3']*1e6), '->', round(row['peeled_bbx_m3']*1e6), 'cm3),',
              round(rows[-1]['feet_cm3'], 1), 'cm3 of feet, skin', 'over 40' if steps['skin_width_m'] is None else round(steps['skin_width_m']*1000, 1), 'mm wide,', round(time.monotonic()-t0), 's', flush=True)
        render(group, model, q, support, output_root/object_name/'step5'/f'{name}.png', top=False)
        package(c, group, model, q, support, rows[-1], output_root/object_name/'step5'/name)
    report = dict(object=object_name, peel=dict(trim=P.TRIM, grids=P.GRIDS, budget=P.BUDGET), skin_width_m=WIDTH, least_width_m=P.THICK, foot=FOOT, results=rows, input_sha256=inputs(object_name), seconds=time.monotonic()-began)
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
        rebuilt, model, q, _, _ = wrap(c, groups[row['step4_file']]); support = read_mesh(path)
        # The same mesh, compared as a mesh: a Boolean difference of such a support with itself is ~1e-8 m^3 (the
        # cell prisms' seams), above the tolerance.
        if rebuilt.vertices.shape != support.vertices.shape or not np.array_equal(rebuilt.faces, support.faces):
            raise ValueError(f"{row['name']}: saved support differs from its rebuild (another mesh)")
        d = float(np.abs(rebuilt.vertices-support.vertices).max())
        if d > 1e-9: raise ValueError(f"{row['name']}: saved support differs from its rebuild: a vertex {d} m off")
        proof, ground = certify(model, q, support)
        rows.append(dict(name=row['name'], rebuilt_max_vertex_deviation_m=d,
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


def render(group, model, q, support, path, size=620, gap=40, top=True):
    """One figure: every task's use of the support, below the seats and the support alone when top (Step4's figure;
    Step5's keeps only the uses, user 2026-10-29: "只留下面这一行")."""
    from .render_joined import use
    rows = ([overview(model, group, support, q, size)] if top else [])+[[use(model, q, support, k, size) for k in range(len(model.records))]]
    columns = max(len(r) for r in rows)
    sheet = Image.new('RGB', (columns*(size+gap)+gap, len(rows)*(size+gap)+gap), 'white')
    for r, panels in enumerate(rows):
        for k, panel in enumerate(panels): sheet.paste(panel, (gap+k*(size+gap), gap+r*(size+gap)))
    path.parent.mkdir(parents=True, exist_ok=True); sheet.save(path)
    print('STEP5 FIGURE', path, flush=True)


def station(setup):
    """The station's robots_config file: the one Step1 recorded, else the one whose id is the setup's robot."""
    if setup.get('config_file'): path = ROOT/setup['config_file']
    else:
        path = next((p for p in sorted([*ROOT.glob('robots_config/*.json'), *ROOT.glob('robots_config/random/*.json')])
                     if json.loads(p.read_text()).get('id') == setup['robot']), None)
        if path is None: raise ValueError(f"no robots_config file has id {setup['robot']}")
    return path, json.loads(path.read_text())


def package(c, group, model, q, support, row, out):
    """What the real station needs of one support, in out/: support.stl (support frame; the certified support
    enlarged PRINT about its bounds' centre, for clearance) and object.stl (object frame); info.json: the support's facts, the station as robots_config gives it (the robot, its joint order and
    limits, its base, the tool point the joints were solved for), every seat (the object in the support) and every
    task (the object and the support at the station where Step1 certified the work arm, the exit, the feet, the
    checks, and the arm's joint sets at the central face of its area: one per test point, the face's three vertices
    and its centroid); tasks.npz: the same for every certified face.  Joints are the robot's own, in its joint
    order (an arm with fewer than seven joints is stored padded with zeros; the padding is dropped).  Seats and
    installations are support.stl's: the object centred in its seat, each installation the certified one scaled
    about the point of its ground under the object's centre.  Lengths mm (4x4 matrices too), angles deg."""
    from whole_task_2.step1.reach import test_points
    from whole_task_2.step1.reach_render import contact
    from whole_task_2.step3.tool_membranes import stem

    def mm(T):
        T = np.array(T, float); T[:3, 3] *= 1e3; return T
    setup = model.setup; rows = seats(group); arrays = {}; tasks = []
    O = support.bounds.mean(0); grow = lambda p: O+PRINT*(np.asarray(p)-O); centre = model.source.center_mass

    def seat(X):
        """The seat X in support.stl: the object centred in it."""
        Y = np.array(X, float); Y[:3, 3] += (PRINT-1)*(transform(centre[None], X)[0]-O); return Y

    def stand(P, X):
        """support.stl at the station and its ground: the certified installation P X^-1 scaled about the point of
        its ground (the support's lowest plane) under the object's centre, so the ground stays."""
        W = P@np.linalg.inv(X); point = transform(centre[None], P)[0]; point[2] = transform(support.vertices, W)[:, 2].min()
        T = np.eye(4); T[:3, 3] = -(PRINT-1)*(transform(point[None], np.linalg.inv(W))[0]-O); return W@T, point[2]
    out.mkdir(parents=True, exist_ok=True); printed = grow(support.vertices)
    trimesh.Trimesh(printed*1e3, support.faces, process=False).export(out/'support.stl')
    trimesh.Trimesh(model.source.vertices*1e3, model.source.faces, process=False).export(out/'object.stl')
    path, config = station(setup); spec = config['robot_specification']; dof = spec['dof']; limits = spec['joint_position_limits']
    bounds = np.asarray(limits['ranges'], float); bounds = np.degrees(bounds) if limits['unit'] == 'rad' else bounds
    tcp = config['feasibility'].get('tcp_transform'); tip = setup['needle_length_m']+.07    # step1.reach_video: flange 70 mm behind the needle root
    proofs = {t['id']: t for t in row['tasks']}; ground = {r['id']: r for r in row['ground']}
    feet = {}
    for f in row['pieces']:
        for t in f.get('tasks', []): feet.setdefault(t, []).extend(f['feet'] or [])
    for k, r in enumerate(model.records):
        X = model.X[k]; P = np.asarray(r['pose']); W, ground_z = stand(P, X); faces = np.asarray(r['certified_face_ids'])
        f = contact(model.source, faces, P); j = int(np.flatnonzero(faces == f)[0])
        axes = np.asarray(r['axes'])@P[:3, :3].T; joints = np.degrees(np.asarray(r['joints']))
        if np.abs(joints[..., dof:]).max(initial=0.) > 0: raise ValueError(r['id']+f': joints beyond the robot\'s {dof}')
        joints = joints[..., :dof]
        tasks.append(dict(
            id=r['id'], seat=next(n for n, (Y, _) in enumerate(rows) if np.allclose(X, Y, atol=1e-9)), turn_deg=r['angle_deg'],
            world_from_object=mm(P), world_from_support=mm(W),
            support_lowest_z_mm=float(transform(printed, W)[:, 2].min())*1e3,
            object_lift_mm=(PRINT-1)*(transform(centre[None], P)[0, 2]-ground_z)*1e3,
            exit_direction_world=q.exits[k], exit_length_mm=proofs[r['id']]['exit_length_m']*1e3,
            working_faces=len(r['working_face_ids']), working_area_cm2=float(model.source.area_faces[r['working_face_ids']].sum())*1e4,
            certified_faces=len(faces),
            central_face=dict(face=int(f), test_points_world_mm=(test_points(model.source)[f]@P[:3, :3].T+P[:3, 3])*1e3,
                              needle_direction_world=axes[j], arm_joints_deg=joints[j]),
            feet=[dict(tip_centre_support_mm=grow(np.asarray(s['site_support_m']))*1e3, leg_mm=s['leg_m']*1e3)
                  for s in feet.get(r['id'], [])],
            checks=dict(paired_samples_held=proofs[r['id']]['samples'], continuous_outer_targets_held=proofs[r['id']]['outer'],
                        arm_link_faces_hit=proofs[r['id']]['arm_faces_hit'], pressure_hull_gap_mm=ground[r['id']]['pressure_hull_gap_m']*1e3,
                        arm_joints_within_limits=bool(((joints >= bounds[:, 0]-1e-6) & (joints <= bounds[:, 1]+1e-6)).all()))))
        name = stem(r['id']); arrays.update({f'{name}_certified_faces': faces, f'{name}_needle_direction_world': axes,
                                             f'{name}_arm_joints_deg': joints, f'{name}_world_from_object_mm': mm(P),
                                             f'{name}_world_from_support_mm': mm(W)})
    np.savez_compressed(out/'tasks.npz', **arrays)
    info = dict(
        object=c.object_name, result=row['name'], units='lengths mm (4x4 matrices: translation mm), angles deg',
        frames=dict(object='object.stl: the object; tasks name its faces (face indices of object.stl)',
                    support=f'support.stl: the certified support enlarged {PRINT} along x, y and z about support.print_centre_mm, '
                            'for clearance; the object sits centred in a seat at seats[n].support_from_object',
                    world='the station (robots_config): support.stl stands at tasks[k].world_from_support (the certified '
                          'installation scaled about the point of its ground, z = support_lowest_z_mm, under the object\'s '
                          'centre); the object, centred in its seat, is tasks[k].object_lift_mm above tasks[k].world_from_object, '
                          'where Step1 certified the work arm\'s joint sets'),
        support=dict(print_scale=PRINT, print_centre_mm=O*1e3, print_volume_cm3=row['volume_cm3']*PRINT**3,
                     print_bounds_mm=np.array([printed.min(0), printed.max(0)])*1e3,
                     volume_cm3=row['volume_cm3'], membrane_cm3=row['membrane_cm3'], feet_cm3=row['feet_cm3'],
                     oriented_bbx_cm3=row['support_bbx_m3']*1e6, bounds_support_mm=np.asarray(row['bounds_m'])*1e3,
                     skin_least_width_mm=None if row['skin_width_m'] is None else row['skin_width_m']*1e3,
                     skin_narrow_and_thin_cm2=row['too_thin_m2']*1e4, peeled_cm3=row['frame']['peel']['removed_cm3'],
                     triangles=len(support.faces), watertight=bool(support.is_watertight),
                     step4=dict(file=row['step4_file'], samples=row['samples'])),
        station=dict(config=str(path.relative_to(ROOT)), id=config['id'], name=config['name'],
                     robot=dict(manufacturer=spec['manufacturer'], model=spec['model'], dof=dof, joint_order=spec['joint_order'],
                                joint_limits_deg=bounds, joint_limits_note=limits.get('applicability'),
                                arm_module=config['feasibility'].get('arm_module') or 'whole_task_2/step5/video/kuka.py'),
                     arm_base_world_mm=np.asarray(setup['base_position_world_m'])*1e3, arm_base_rotation_world=setup['base_rotation_world'],
                     tool=dict(needle_length_mm=setup['needle_length_m']*1e3, needle_radius_mm=setup['needle_radius_m']*1e3,
                               head_radius_mm=setup['head_radius_m']*1e3, source=setup.get('tool_source'),
                               solved_flange_to_needle_tip_mm=tip*1e3,
                               solved_rule='the joints place the needle tip on the face: flange (SO101: gripper frame) 70 mm '
                                           'behind the needle root, the needle along the tool axis',
                               config_tcp_transform=tcp,
                               config_tcp_matches_solved=None if tcp is None else bool(np.isclose(np.linalg.norm(tcp['translation_flange_m']), tip))),
                     working_direction_world=setup['working_direction_world'], tilt_cone_half_angle_deg=setup['max_tilt_deg'],
                     work_cube_bounds_world_mm=np.asarray(setup['surface_workspace']['bounds_world_m'])*1e3,
                     table_top_z_mm=setup['table_height_m']*1e3, table_bounds_world_mm=np.asarray(setup['table_bounds_world_m'])*1e3),
        seats=[dict(seat=n, support_from_object=mm(seat(X)), tasks=[model.records[k]['id'] for k in ks]) for n, (X, ks) in enumerate(rows)],
        tasks=tasks, files={p.name: sha256(p) for p in (out/'support.stl', out/'object.stl', out/'tasks.npz')})
    write_json(out/'info.json', plain(info))
    print('STEP5 PACKAGE', out, flush=True)


def figures(object_name='B', output_root=DEFAULT_OUTPUT):
    """The figure of every saved support again, from disk."""
    c = context(object_name); folder = ROOT/'whole_task_2/data'/object_name/FOLDER
    report = json.loads((folder/'report.json').read_text()); groups = {g['file']: g for g in distinct(object_name)}
    for row in report['results']:
        g = groups[row['step4_file']]; step4_figure(c, g, output_root/object_name/'step4'/f"{row['name']}.png")
        if row['status'] != 'passed': continue
        support = read_mesh(folder/f"{row['name']}.npz")
        model = ExitModel(c, g['slots'])
        render(g, model, model.layout(support), support, output_root/object_name/'step5'/f"{row['name']}.png", top=False)
        package(c, g, model, model.layout(support), support, row, output_root/object_name/'step5'/row['name'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', default='B'); p.add_argument('--validate-only', action='store_true')
    p.add_argument('--figures-only', action='store_true')
    a = p.parse_args()
    validate(a.object) if a.validate_only else figures(a.object) if a.figures_only else run(a.object)
