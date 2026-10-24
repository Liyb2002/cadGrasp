"""Step5: every distinct support of step4/combine.py peeled, with its feet, certified for every task, and its
figure.

The placing arm installs the support for each task so that the task faces the work arm at its certified
pose, and the support rests on the table.

The skin is at least skin.WIDTH (10 mm) wide along the object's surface everywhere (step4/skin.py; width is
across the skin, not its thickness): Step4 prunes every support to it before its oracle.

Then the peel (step5/peel.py), as Co-optimize's Step5.2: the material deleted a piece at a time -- plane cuts
of the outer material toward the bunnies, then boxes, the ones holding the most skin first -- each kept when
every task's paired forces and moments still hold (an LP certificate on the contact corners, solved again
only where a deletion takes a corner it uses), with the skin it leaves thinner than WIDTH and what falls off,
and no hole opens in the skin (from the periphery inward).

Then the support is made manufacturable: its least feature is 2r (ROUND['radius']).  Cutting it cell by cell
(Step4's thin skin, the peel's), by the keep-outs (hundreds of needle and head cylinders) and by planes leaves
thin teeth, spikes, knife edges and skin thin through the layer.  The rule, as the minimum length scale of
topology optimization (Sigmund 2007, open-close filters): every point of the support lies in a ball of radius r
inside it -- the support is its own morphological opening by that ball -- so no spike, fin or wall is thinner
than 2r and every convex edge is rounded to radius r at least.  On a grid ROUND['grid'] fine (a voxel is material
when the support holds its centre or one of its vertices, so a tooth thinner than a voxel counts), in the room:
the seats' layers (or the support, where the peel thickened it), above every floor, out of every body,
keep-out and exit sweep,
  - every contact is backed 2r thick (the material on the object thickened outward in the room: a thin wedge
    a plane cut leaves on the object holds loads, so it is thickened, not cut);
  - the support is closed by ROUND['close'] (the teeth join, small holes fill);
  - what sticks out from the objects thinner than 2R (ROUND['spike']: fins, hooks, fingers) goes: the support
    with the objects under it is opened by R, so the skin the objects back stays;
  - it is opened by r (ROUND['radius']): no spike, fin or wall thinner than 2r, every convex edge rounded;
  - about every contact corner a task's paired-load certificate uses (as the peel's, preferring corners deep in
    the skin) the backed skin within 2r stays;
  - a piece the openings part (one holding such skin however small) is joined to the rest by the shortest
    strut of radius r (its centre line where such a ball fits in the room, a geodesic); a piece of the support
    (at least 1e-3 of it) that no such strut reaches -- the room between it and the rest is narrower than 2r --
    by the shortest centre line through that room, its strut an exact tube of radius r (not smoothed below),
    cut to the legal space like the rest: as wide as the space lets it.
All of it in the room, so the result stays in it; one pass.
The surface is smoothed (a Gaussian of the signed distance, Taubin passes) and reaches 1.5 voxels into the
objects only, then, as a solid again, is cut to the exact legal space: its contact faces are the objects' own
(at an obtuse rim), nothing enters a body, keep-out or exit or goes below a floor.  What it adds in the prism
of a source face that then no longer exits goes again (the support's own material there stays).  The support is
audited like any other.

Then feet (as Co-optimize's Step5, "wrap the pressure hull"): the ground faces enclose a task's pressure
hull when their convex hull does, so a few feet do, not a band around it.  For every task whose hull the
actual ground faces do not already enclose, in its installation:

  candidate feet stand on the table every FOOT['spacing'] around the hull, centred FOOT['outward'] outside it
  (a little inside too: the pad still covers the hull's edge), where a pad FOOT['radius'] in radius and
  FOOT['height'] high fits in the legal space -- above every
  installation's floor, out of every task's body, keep-out and exit sweep (Step4's forbidden region), so a
  foot never blocks another task;
  a foot is its pad and a leg, the hull of the pad and a ball of the same radius at the support's nearest
  point (2 FOOT['radius'] = WIDTH wide everywhere), clipped to the legal space;
  the feet are the cheapest cycle of candidates (and of the actual ground faces' corners, free) about the
  hull with the hull left of every step but for a pad's radius, so their pads' convex hull holds it; a foot
  costs its material (its leg, pad and ball cap); the candidates nearest the hull first (the rings inside,
  on and just outside its edge together, then each further ring), so the feet hug it;
  a foot's leg may touch the object in another seat: the skin narrower than WIDTH it adds there is cut from it
  (as Step4 cuts thin skin);
  a foot the legal space or that cut cuts off the support or its pad is dropped and the cycle chosen again.

The whole peeled membrane and every foot stay.  As before every Step4 audit, the
original face cells at each task's final exit travel are recut (MembraneModel.refine_exits), removing the
Boolean slivers (~1e-10 m^3) the joins can leave on an exit.  Every task is then audited in full on
the support with its feet (32,768 paired samples and the continuous outer targets, the continuous exit of
every source face, body and keep-out, the KUKA at every stored joint set), and in every installation the
actual ground faces must enclose the task's pressure hull.  Validation rebuilds every support from Step4's
membrane (peel, rounding and feet) and audits it again.

For the real station, next to every figure a folder output/<object>/step5/<name>/ (package): the support and
the object as STL (mm), info.json and tasks.npz.  support.stl is the certified support enlarged PRINT along x,
y and z, so the object has clearance in its seat; every installation is the certified one scaled about the
point of its ground under the object's centre (the support stands where it stood), the object centred in its seat.  The
certificates are the certified support's.  A run overwrites the last one: it first removes every earlier
result (clear()), so step5/ holds this run's results only, with no copies.

One figure per support, output/<object>/step5/<name>.png, no text: the top row, the support with the bunny
in each seat (pink: the areas of the seat's tasks), then the support alone; the bottom row, every task at
the station, the support as the placing arm installs it and the KUKA holding the needle on the task's
central certified face.
"""
import argparse
import json
import re
import shutil
import time
import numpy as np
import trimesh
import manifold3d as md
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.measure import marching_cubes
from skimage.graph import MCP_Geometric
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.membranes import positive_components
from whole_task_2.operations.geometry import moved_mesh
from whole_task_2.step4.ground_oracle import pressure_rows, ground_enclosure
from whole_task_2.step4.seats import context, read_mesh
from whole_task_2.step4.merge import move
from whole_task_2.step4.combine import FOLDER as COMBINE, ExitModel, largest, plain, seats, solids, obb
from whole_task_2.step4.skin import WIDTH, SEAM, transform, piece, measure, legal, skin, regions, cut
from whole_task_2.step5 import peel as P

FOLDER = 'step5_combined'
FOOT = dict(radius=WIDTH/2, height=.005,             # pads and legs as wide as the skin's least width; pads 5 mm high
            outward=(-.004, 0., .004, .008, .016, .032, .064),   # candidate feet centred this far outside the
                                                     # pressure hull (inside: the pad still covers its edge)
            spacing=WIDTH/2,                         # and this far apart along it
            first=3,                                 # the nearest rings tried together first (inside, on, outside)
            attempts=24,                             # cycles chosen again after feet the legal space cut off
            padding=.03)                             # the legal space: a box this much around the farthest feet
ROUND = dict(grid=WIDTH/12, close=WIDTH/5, radius=.15*WIDTH,           # least feature radius r
             spike=.3*WIDTH,                                             # least protrusion radius R, the object backing it
             blur=1., taubin=10)                                         # (blur: voxels)
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


def enclosing(points, cost, hull, reach, margin=1e-6):
    """The cheapest of points (each its cost) whose ground holds hull: a cycle about the hull with the hull
    left of every step, at most the smaller reach (m: the radius of a foot's pad, 0 for a point) to its right
    (by margin).  Their ids, or None."""
    centre = hull.mean(0); order = np.argsort(np.arctan2(points[:, 1]-centre[1], points[:, 0]-centre[0]), kind='stable')
    P, w, r = points[order], cost[order], reach[order]; n = len(P); d = P[None]-P[:, None]; h = hull[None, None]-P[:, None, None]
    length = np.linalg.norm(d, axis=-1); slack = (margin-np.minimum(r[:, None], r[None]))[..., None]
    left = (d[..., None, 0]*h[..., 1]-d[..., None, 1]*h[..., 0] >= slack*length[..., None]).all(-1) & (length > 1e-12)
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
    """The legal space through a pad's height on a task's table (its world xy, manifold units)."""
    world = move(row['world_from_support'], legal); h = foot['height']/G.SCALE
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


def foot_at(row, site, anchor, legal, stock, foot=FOOT, thin=lambda part: part):
    """A foot on a task's table at site (world xy): its pad, joined to anchor (support frame) by the hull of
    the pad and a ball of its radius, in the legal space, without the thin skin it adds (thin), support frame;
    None if the legal space or that cut cuts the pad or cuts it off the stock."""
    pad = md.Manifold.cylinder(foot['height']/G.SCALE, foot['radius']/G.SCALE, -1., 32).translate((site[0]/G.SCALE, site[1]/G.SCALE, 0.))
    ball = md.Manifold.sphere(foot['radius']/G.SCALE, 32).translate(tuple(transform(anchor[None], row['world_from_support'])[0]/G.SCALE))
    pad = move(row['material'], pad); parts = positive_components(thin((pad+move(row['material'], ball)).hull() ^ legal))
    if not parts: return None
    part = max(parts, key=lambda p: G.volume(p ^ pad))
    return part if G.volume(pad-part) <= 1e-3*G.volume(pad) and joined(stock+part) else None


def legal_space(c, group, q, points, pad):
    """Where material may go: in the box of points widened by pad, above every installation's floor, out of
    every task's body, keep-out and exit sweep (Step4's forbidden region)."""
    return legal(points, pad, q.material, solids(c.object_name, group)['F'])


def settled(c, group, support):
    """support with the seams the cell prisms leave collapsed (SEAM) and the forbidden region cut again, so
    that it settles under further Booleans (as step4/skin.py's wide)."""
    return piece(G.solid(support).simplify(SEAM/G.SCALE)-solids(c.object_name, group)['F'], 'settling splits the support')


def grounds(c, model, group, membrane, foot=FOOT):
    """The points of membrane that must stay on the tables: in every installation where no ring of legal feet
    (step 5.3's candidates, legs aside) encloses the task's pressure hull, the corners of its ground faces."""
    q = model.layout(membrane); rows = pressure_rows(model, q); out = [np.zeros((0, 3))]
    legal = legal_space(c, group, q, np.vstack([membrane.vertices, *[r['ground_material'] for r in rows]]), max(foot['outward'])+foot['padding'])
    disc = md.CrossSection.circle(foot['radius']/G.SCALE, 32); pad = foot['radius']*np.cos(np.pi/32)-1e-4
    for r in rows:
        hull = np.asarray(r['hull']); free = section(r, legal, foot)
        sites = np.vstack([resample(np.asarray(g.exterior.coords)[:-1], foot['spacing']) for d in foot['outward']
                           for ring in [orient(Polygon(hull).buffer(d, join_style=2), sign=1.)] for g in getattr(ring, 'geoms', [ring]) if not g.is_empty])
        sites = sites[np.array([(disc.translate(tuple(s/G.SCALE))-free).area() <= 1e-6*disc.area() for s in sites], bool)]
        if len(sites) >= 3 and enclosing(sites, np.ones(len(sites)), hull, np.full(len(sites), pad)) is not None: continue
        corners = np.asarray(ground_enclosure(membrane, [r])[0]['ground_contact_hull_xy_m'], float).reshape(-1, 2)
        out.append(transform(np.c_[corners, np.zeros(len(corners))], r['material']))
    return np.vstack(out)


def pare(c, model, group, support):
    """The support after the peel (step5/peel.py), its skin's holes filled in the legal space, as the mesh it is
    saved as (its largest piece, should the export part a piece touching it at a seam; export crumbs dropped):
    (support, record)."""
    bunnies = np.vstack([transform(model.source.vertices, X) for X in model.X])
    pad = WIDTH+2*c.base.policy['mold_outward_displacement_m']; legal = legal_space(c, group, model.layout(support), support.vertices, pad)
    peeled, record = P.Peel(c, model, group, support, grounds(c, model, group, support)).run(bunnies, legal)
    peeled, record['filled_m2'] = P.fill(c, group, peeled, legal)
    value, record['saved_share'] = largest(G.solid(peeled))
    return piece(value, 'the peel splits the support'), record


def feet(c, model, group, membrane, foot=FOOT):
    """membrane with every task's feet joined to it: (the solid, the record of every task's feet, whether
    every pressure hull is enclosed).  The candidates nearest the hull are tried first: the first
    FOOT['first'] rings of FOOT['outward'] together, then each further ring in turn."""
    q = model.layout(membrane)
    rows = pressure_rows(model, q)
    legal = legal_space(c, group, q, np.vstack([membrane.vertices, *[r['ground_material'] for r in rows]]), max(foot['outward'])+foot['padding'])
    stock, share = largest(G.solid(membrane))
    if share < 1-1e-9: raise ValueError('Step4 membrane is not one piece')
    base = regions(c, group['slots'], membrane); thin = lambda part: own_thin(c, base, part)
    record = []; enclosed = True; disc = md.CrossSection.circle(foot['radius']/G.SCALE, 32)
    pad = foot['radius']*np.cos(np.pi/32)-1e-4                     # the pad polygon's inner radius, less a margin
    for r in rows:
        ground = ground_enclosure(G.mesh(stock), [r])[0]
        if ground['passed']:
            on = transform(np.c_[np.asarray(ground['ground_contact_hull_xy_m']).reshape(-1, 2), np.zeros(len(ground['ground_contact_hull_xy_m']))], r['material'])
            record.append(dict(task=r['id'], feet=[], ground_corners=len(on), held_m=on.tolist())); continue
        hull = np.asarray(r['hull']); corners = np.asarray(ground['ground_contact_hull_xy_m'], float).reshape(-1, 2)
        sites, level = [np.empty((0, 2))], [np.empty(0, int)]
        for k, d in enumerate(foot['outward']):
            ring = orient(Polygon(hull).buffer(d, join_style=2), sign=1.)
            for g in getattr(ring, 'geoms', [ring]):
                if g.is_empty: continue
                p = resample(np.asarray(g.exterior.coords)[:-1], foot['spacing']); sites.append(p); level.append(np.full(len(p), k))
        sites, level = np.vstack(sites), np.concatenate(level); free = section(r, legal, foot)
        ok = np.array([(disc.translate(tuple(s/G.SCALE))-free).area() <= 1e-6*disc.area() for s in sites], bool).reshape(-1)
        sites, level = sites[ok], level[ok]
        tops = transform(np.c_[sites, np.full(len(sites), foot['height'])], r['material'])
        anchor, length, _ = trimesh.proximity.closest_point(G.mesh(stock), tops) if len(sites) else (np.empty((0, 3)), np.empty(0), None)
        # A foot's material, as a length of leg: its leg, its pad and the cap of the ball.
        points = np.vstack([corners, sites]); cost = np.r_[np.zeros(len(corners)), length+foot['height']+2/3*foot['radius']]
        reach = np.r_[np.zeros(len(corners)), np.full(len(sites), pad)]; level = np.r_[np.full(len(corners), -1), level]
        built = {}; found = None
        for top in range(min(foot['first'], len(foot['outward']))-1, len(foot['outward'])):
            for _ in range(foot['attempts']):
                use = np.flatnonzero((level <= top) & np.isfinite(cost))
                ids = enclosing(points[use], cost[use], hull, reach[use]) if len(use) >= 3 else None
                if ids is None: break
                ids = use[ids]; new = [int(k) for k in ids if k >= len(corners)]
                for k in new:
                    if k not in built: built[k] = foot_at(r, sites[k-len(corners)], anchor[k-len(corners)], legal, stock, foot, thin)
                    if built[k] is None: cost[k] = np.inf
                if any(built[k] is None for k in new): continue
                grown = G.union([stock]+[built[k] for k in new])
                if joined(grown) and ground_enclosure(G.mesh(grown), [r])[0]['passed']: found = grown, new, len(ids)-len(new), top
                break
            if found is not None: break
        if found is None: enclosed = False; record.append(dict(task=r['id'], feet=None)); continue
        stock = found[0]
        used = [int(k) for k in ids if k < len(corners)]
        record.append(dict(task=r['id'], ground_corners=found[2], outward_m=foot['outward'][found[3]],
                           feet=[dict(site_m=sites[k-len(corners)].tolist(), leg_m=float(length[k-len(corners)])) for k in found[1]],
                           # The support's points the feet stand on: each leg's anchor, each ground corner used.
                           held_m=np.vstack([anchor[[k-len(corners) for k in found[1]]].reshape(-1, 3),
                                             transform(np.c_[corners[used], np.zeros(len(used))], r['material']).reshape(-1, 3)]).tolist()))
    return stock, record, enclosed


def leg_skin(c, group, membrane, support):
    """support without the skin its feet add narrower than WIDTH (a leg touching the object in another seat), cut
    as Step4 cuts thin skin: (support, the area cut, m^2)."""
    S, _ = skin(c); value = G.solid(support); area = 0.
    for (X, R, own), (_, old, _) in zip(regions(c, group['slots'], support), regions(c, group['slots'], membrane)):
        ids = np.flatnonzero(S.thin(R, own=own) & ~old)
        if len(ids): value = cut(c, value, X, ids); area += float(S.area[ids].sum())
    return (piece(value, 'cutting the thin skin of the feet splits the support') if area else support), area


def occupancy(solid, low, h, shape):
    """The voxel centres (grid low + (index + 1/2) h) inside solid: every z slice's cross-section filled, holes by
    even-odd."""
    nx, ny, nz = shape; out = np.zeros(shape, bool)
    for k in range(nz):
        polygons = solid.slice(float((low[2]+(k+.5)*h)/G.SCALE)).to_polygons(); inside = np.zeros((ny, nx), bool)
        for polygon in polygons:
            p = (np.asarray(polygon)*G.SCALE-low[:2])/h-.5
            if len(p) < 3: continue
            image = Image.new('1', (nx, ny), 0); ImageDraw.Draw(image).polygon([tuple(q) for q in p], fill=1); inside ^= np.asarray(image, bool)
        out[:, :, k] = inside.T
    return out


def touched(mesh, low, h, shape):
    """The voxels (grid low + index h) holding a vertex or a triangle's centre of mesh."""
    i = np.floor((np.vstack([mesh.vertices, mesh.triangles_center])-low)/h).astype(int)
    out = np.zeros(shape, bool); out[tuple(i[((i >= 0) & (i < shape)).all(1)].T)] = True
    return out


def struts(O, room, r, h, must, share=1e-3):
    """O (voxels) with every piece of at least `share` of it, or holding any of must, joined to the largest by the
    shortest strut of radius r whose centre line stays where a ball of radius r fits in room, the nearest piece
    first; then every piece of at least `share` no such strut reaches (the room between it and the rest narrower
    than 2r) by the shortest centre line through room, the strut as wide as room lets it: (O, struts, the centre
    lines of those narrow ones, voxel indices)."""
    edt = lambda mask: ndimage.distance_transform_edt(mask, sampling=h)
    label, n = ndimage.label(O)
    if n < 2: return O, 0, []
    size = np.bincount(label.ravel()); size[0] = 0; joined = label == size.argmax(); needed = set(np.unique(label[must]))
    todo = [k for k in range(1, n+1) if k != size.argmax() and (size[k] >= share*size.sum() or k in needed)]
    count, narrow = 0, []
    for space in (edt(room) > r, room):
        if space is room: todo = [k for k in todo if size[k] >= share*size.sum()]
        cost = np.where(space, 1., np.inf)
        while todo:
            paths = MCP_Geometric(cost, fully_connected=True); far, _ = paths.find_costs(np.argwhere(joined & space)); best = None
            for k in todo:
                ends = np.argwhere((label == k) & (space | must))
                if not len(ends): continue
                d = far[tuple(ends.T)]; j = int(d.argmin())
                if np.isfinite(d[j]) and (best is None or d[j] < best[0]): best = (d[j], k, tuple(ends[j]))
            if best is None: break
            path = np.asarray(paths.traceback(best[2])); line = np.zeros(O.shape, bool); line[tuple(path.T)] = True
            joined |= (label == best[1]) | ((edt(~line) <= r) & room); todo.remove(best[1]); count += 1
            if space is room: narrow.append(path)
    return joined, count, narrow


def tubes(paths, low, h, r):
    """Centre lines (voxel indices on the grid low + (index + 1/2) h) as exact tubes of radius r."""
    ball = trimesh.creation.icosphere(subdivisions=2, radius=r).vertices; parts = []
    for path in paths:
        p = low+(np.asarray(path, float)+.5)*h; p = np.vstack([p, p]) if len(p) == 1 else p
        parts += [md.Manifold.hull_points(np.vstack([ball+a, ball+b])/G.SCALE) for a, b in zip(p[:-1], p[1:])]
    return G.union(parts)


def holding(model, support, r, depth):
    """Every contact corner a task's paired-load certificate on support uses (the peel's), the certificate preferring
    corners deep in the skin (a corner costs exp(-depth/r)): (the corners, the object's outward normals there),
    support frame."""
    points, normals = [np.zeros((0, 3))], [np.zeros((0, 3))]
    for t in model.prepare(model.layout(support)):
        hold = P.Hold(t, support, lambda p: depth(p)*WIDTH/r)
        used = np.unique(np.concatenate([b for b in hold.bases if b is not None])); used = used[used > 0]-1
        _, _, faces = trimesh.proximity.closest_point(t['body'], hold.points[used])
        points.append(hold.points[used]); normals.append(t['body'].face_normals[faces])
    return np.vstack(points), np.vstack(normals)


def holds(model, support):
    """The first task whose contact corners on support no longer hold all its outer targets (the peel's
    certificate), or None."""
    for t in model.prepare(model.layout(support)):
        try: P.Hold(t, support, lambda p: np.zeros(len(p)))
        except ValueError: return t['id']
    return None


def rounded(c, model, group, support, r=ROUND):
    """support made manufacturable (above): (support, record)."""
    began = time.monotonic(); h = r['grid']; slots = group['slots']; F = solids(c.object_name, group)['F']; delta = 1.5*h
    pad = r['close']+2*r['radius']+2*r['spike']+delta+6*h; low = support.bounds[0]-pad; shape = np.ceil((support.bounds[1]+pad-low)/h).astype(int)
    box = G.box(low, low+shape*h); edt = lambda mask: ndimage.distance_transform_edt(mask, sampling=h)
    close = lambda mask, d: edt(edt(~mask) <= d) > d
    opened = lambda mask, d: edt(~(edt(mask) > d)) <= d
    voxel = lambda p: tuple(np.clip(np.floor((np.asarray(p)-low)/h).astype(int), 0, shape-1).T)
    def balls(centres, d):
        seed = np.zeros(shape, bool); seed[voxel(centres)] = True; return edt(~seed) <= d
    bodies = G.union([move(np.asarray(s['X'], float), G.solid(c.base.source)) for s in slots]) ^ box
    _, outer = skin(c); value = G.solid(support); material = model.layout(support).material
    space = legal(support.vertices, WIDTH, material, F)                                   # exact: floors, out of F
    A = occupancy(value, low, h, shape) | touched(support, low, h, shape); B = occupancy(bodies, low, h, shape)
    hard = ~occupancy(legal(support.vertices, WIDTH, material, md.Manifold()) ^ box, low, h, shape) | (occupancy(F ^ box, low, h, shape) & ~B)
    room = occupancy((G.union([move(np.asarray(s['X'], float), outer) for s in slots])+value) ^ box, low, h, shape) & ~B & ~hard
    near = edt(~B); reach = np.clip(1-(near-delta)/(2*h), 0, 1); air = edt(A | B)

    def solid(O):
        """O smoothed, as a solid in the legal space, the exit kept: (support, share kept, exit rounds)."""
        field = ndimage.gaussian_filter(edt(O)-edt(~O), r['blur'])+delta*reach          # into the objects only
        v, f, _, _ = marching_cubes(np.pad(field, 1, constant_values=-1.), 0., spacing=(h, h, h))
        mesh = trimesh.Trimesh(v+low-.5*h, f, process=True)
        if mesh.volume < 0: mesh.invert()
        trimesh.smoothing.filter_taubin(mesh, iterations=r['taubin'])
        body = G.solid(mesh) if not narrow else G.solid(mesh)+tubes(narrow, low, h, radius)
        out, share = largest(body ^ space)
        for rounds in range(4):                 # a source face that no longer exits: what was added in its prism goes
            m = G.mesh(out); candidate = dict(layout=model.layout(m), support=m, tasks=None, construction={})
            candidate['tasks'] = model.prepare(candidate['layout']); rows, bad = model.path_checks(out, model.exit_paths(candidate))
            if all(row['passed'] for row in rows) or not bad: break
            for _, _, prism in bad: out = out-(prism-value)
            out = largest(out)[0]
        return piece(out, 'rounding splits the support'), share, rounds
    radius, spike = r['radius'], r['spike']
    contact = A & (near <= 1.5*h)
    backed = A | ((edt(~contact) <= 2*radius) & (near <= 2*radius+h) & room)        # every contact backed 2r thick
    points, normals = holding(model, support, radius, lambda p: air[voxel(p)])
    own = balls(points+.5*h*normals, 2*radius) & backed               # the backed skin about every corner the loads use
    closed = close(backed, r['close']) & room                                        # the teeth joined
    closed = opened(closed | (B & (near <= 2*spike+h)), spike) & room                 # what sticks out thinner than 2R goes
    O = opened(closed, radius) | own                                                 # no feature thinner than 2r
    O, joins, narrow = struts(O, room | own, radius, h, own)
    out, share, rounds = solid(O); failed = holds(model, out)
    if failed is not None: raise ValueError(f'{failed}: the manufacturable support (features {radius*1000:g} mm, protrusions {spike*1000:g} mm) no longer holds its loads')
    return out, dict(radius_m=radius, spike_m=spike, grid=[int(n) for n in shape], kept_share=float(share), struts=joins,
                     narrow_struts=len(narrow), corners=len(points), exit_rounds=rounds,
                     volume_cm3=[float(support.volume*1e6), float(out.volume*1e6)], seconds=time.monotonic()-began)


def wrap(c, group, foot=FOOT):
    """The support peeled, with its feet: (support, model, layout, record of every task's feet, record of the
    steps: the peel and the skin's least width)."""
    model = ExitModel(c, group['slots']); membrane = read_mesh(ROOT/group['file']); steps = {}
    began = time.monotonic(); before = membrane.volume; membrane, steps['peeled'] = pare(c, model, group, membrane)
    steps['peeled'].update(removed_cm3=float(before-membrane.volume)*1e6, seconds=time.monotonic()-began)
    membrane, steps['rounded'] = rounded(c, model, group, membrane)
    steps.update(membrane_cm3=float(membrane.volume*1e6), peeled_bbx_m3=obb(membrane.vertices))
    began = time.monotonic(); stock, record, _ = feet(c, model, group, membrane, foot); steps['feet_seconds'] = time.monotonic()-began
    support = piece(stock, 'the feet split the support')
    # The feet only add.  (A Boolean difference cannot show it: cell prisms leave seams where one of the membrane
    # even with itself differs by mm^3 either way; the audit below judges the support as it is.)
    if G.volume(G.solid(support)) < G.volume(G.solid(membrane))-c.base.tolerance: raise ValueError('the feet lost membrane')
    support, steps['feet_thin_cut_m2'] = leg_skin(c, group, membrane, support)
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
    # The skin as it is: its least width (None: nothing narrower than 4 WIDTH) and the area narrower than WIDTH and
    # thinner than THICK (load-bearing skin the legal space lets be neither wider nor thicker).  Recorded.
    width = measure(c, group['slots'], support); steps['too_thin_m2'] = P.too_thin(c, group, support)
    steps['skin_width_m'] = width[0] if np.isfinite(width[0]) else None
    return support, model, model.layout(support), record, steps


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
                                             'step4/ground_oracle.py', 'step4/skin.py', 'step5/peel.py', 'step5/combined.py')]]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def clear(object_name, output_root=DEFAULT_OUTPUT):
    """Every result of an earlier run, so that this one overwrites them: the supports, report and validation under
    data/<object>/step5_combined/, the figures rNN.png and packages rNN/ under output/<object>/step5/ (nothing else
    there is touched).  No copies are kept."""
    for path in [ROOT/'whole_task_2/data'/object_name/FOLDER, output_root/object_name/'step5']:
        for p in (path.iterdir() if path.exists() else []):
            if re.fullmatch(r'r\d\d(\.png|\.npz|\.obj)?', p.name) or p.name in ('report.json', 'validation.json'):
                shutil.rmtree(p) if p.is_dir() else p.unlink()


def run(object_name='B', output_root=DEFAULT_OUTPUT):
    began = time.monotonic(); c = context(object_name); folder = ROOT/'whole_task_2/data'/object_name/FOLDER
    clear(object_name, output_root); folder.mkdir(parents=True, exist_ok=True); rows = []
    for n, group in enumerate(distinct(object_name)):
        name = f'r{n:02d}'; t0 = time.monotonic()
        row = dict(name=name, samples=group['samples'], step4_file=group['file'], step4_files=group['files'],
                   membrane_cm3=group['volume_cm3'], bbx_m3=group['bbx_m3'], seats=[len(ks) for _, ks in seats(group)],
                   joins=[dict(task=s['parents'][1], operation=s['operation'], seat=s['seat'], bbx_ratio=s['bbx_ratio'])
                          for s in group['chain']])
        try:
            support, model, q, pieces, steps = wrap(c, group); row.update(pieces=pieces, **steps)
            row.update(peeled_cm3=steps['peeled']['removed_cm3'])
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
        print('STEP5', name, 'PASS', round(support.volume*1e6, 1), 'cm3,', round(row['peeled_cm3'], 1), 'cm3 peeled (bbx',
              round(group['bbx_m3']*1e6), '->', round(row['peeled_bbx_m3']*1e6), 'cm3),',
              round(rows[-1]['feet_cm3'], 1), 'cm3 of feet, skin', 'over 40' if steps['skin_width_m'] is None else round(steps['skin_width_m']*1000, 1), 'mm wide,', round(time.monotonic()-t0), 's', flush=True)
        render(group, model, q, support, output_root/object_name/'step5'/f'{name}.png')
        package(c, group, model, q, support, rows[-1], output_root/object_name/'step5'/name)
    report = dict(object=object_name, peel=dict(trim=P.TRIM, grids=P.GRIDS, budget=P.BUDGET), skin_width_m=WIDTH, foot=FOOT, results=rows, input_sha256=inputs(object_name), seconds=time.monotonic()-began)
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
    feet = {f['task']: f for f in row['pieces'] if 'task' in f}
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
            feet=[dict(pad_centre_support_mm=grow(transform(np.r_[s['site_m'], 0.][None], q.material[k])[0])*1e3, leg_mm=s['leg_m']*1e3)
                  for s in feet.get(r['id'], {}).get('feet') or []],
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
                     skin_narrow_and_thin_cm2=row['too_thin_m2']*1e4, skin_thickened_cm2=row['peeled']['thickened_m2']*1e4,
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
        if row['status'] != 'passed': continue
        g = groups[row['step4_file']]; support = read_mesh(folder/f"{row['name']}.npz")
        model = ExitModel(c, g['slots'])
        render(g, model, model.layout(support), support, output_root/object_name/'step5'/f"{row['name']}.png")
        package(c, g, model, model.layout(support), support, row, output_root/object_name/'step5'/row['name'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', default='B'); p.add_argument('--validate-only', action='store_true')
    p.add_argument('--figures-only', action='store_true')
    a = p.parse_args()
    validate(a.object) if a.validate_only else figures(a.object) if a.figures_only else run(a.object)
