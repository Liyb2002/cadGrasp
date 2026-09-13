"""The same contacts, joined into TWO printed pieces -- and whether two is honest.

`pipeline_a1f` leaves one wall standing under each contact: four loose objects,
each of which has to be placed and held. A fixture is not four loose objects, so
the question this figure asks is whether the four can be carried by two solids,
and it is a geometry question with a real answer rather than a drawing exercise.

--- what a piece is ----------------------------------------------------------

A piece is the group's walls plus a BASE PLATE: a slab of thickness `PLATE_H`
lying on the floor that runs from one wall's foot to the next. Nothing about the
contacts moves -- the bearing faces are the same tangent planes at the same
points, so the force answer `capped_a1f` proved is still the force answer. The
only new geometry is the plate, and the only new question is where it may run.

--- where a plate may run ----------------------------------------------------

A plate occupies the height band `z in [0, PLATE_H]`. It therefore has to avoid
exactly one thing: the part's own footprint WITHIN THAT BAND, which is a very
different and much smaller set than the part's whole silhouette. A1-f in this
pose rests on a ridge, so below six millimetres it occupies a strip a few
centimetres wide while the rest of it leans out over open floor. That strip is
the obstacle; everything else on the ground plane is free, including ground the
part overhangs.

So the route is a shortest path on the ground, around that strip, and it is
computed rather than drawn: a 1.5 mm grid over the free space, Dijkstra between
wall feet, a minimum spanning tree when a group has more than two of them. An MST
over shortest paths is not a Steiner tree, so the plate length reported is an
upper bound on the best plate for that grouping -- which is the safe direction,
since the claim being made is that a grouping WORKS.

--- how the grouping is chosen -----------------------------------------------

Not by eye. Four contacts split into two non-empty groups seven ways and all
seven are routed. Least plate cannot be the whole rule, because a group of ONE
needs no plate at all and would win every time -- and a piece carrying one pad is
not a merge, it is figure 1's loose wall with a foot bolted on. So splits where
both pieces carry at least two contacts are ranked first, and within them the one
that needs the least plate wins.

On A1-f pose 6 that comes out 12 / 34, and the reason is a fact about the pose
rather than about the objective: the workpiece stands on a ridge running the
length of it, the ridge cuts the floor in two, and contacts 1 and 2 are on one
side of it while 3 and 4 are on the other. The winning split is the one that
never asks a plate to cross the ridge, and the three splits that do cost 60 to
76 per cent more plate.

--- and then it is checked ---------------------------------------------------

`manifold3d` is installed, so the intersection tests here are exact boolean
volumes and not samples: piece against workpiece, piece against piece. Each pad
is checked to still touch its contact (distance to the piece's surface) and to
still stop there (a point half a millimetre past the contact, along the push, must
be OUTSIDE the piece -- otherwise the "pad" has eaten into the part). Each piece
is checked to be one connected body, because a plate routed through a gap that
turns out to be too narrow comes back in two halves and would otherwise be
reported as a success. The clearance figure quoted is measured more than ten
millimetres from any pad: the pads TOUCH, by construction and on purpose, so a
minimum taken over the whole piece is zero whatever happens and says nothing.

    python slides/tools/merge_a1f.py A1-f
    python slides/tools/merge_a1f.py A1-f --pose 2
"""
from __future__ import annotations
import coordinates as COORD

import argparse
import io
import itertools
import shutil

import matplotlib
import numpy as np
import shapely
import trimesh
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from common import obj_path                                        # noqa: E402
from cover import PAPER                                            # noqa: E402
from disturbances import _font                                     # noqa: E402
from pipeline_a1f import (FLOOR, camera, fit_wall, number, project, replicate,
                          solid_shot, steps_of, visible)            # noqa: E402
from shrink_support import paint                                   # noqa: E402

PLATE_H = 0.006        # 6 mm of plate: thin enough to slide under a leaning part
RIB = 0.016            # 16 mm wide, so a rib is stiff about the axis that matters
CLEAR = 0.002          # 2 mm of air between fixture and workpiece, everywhere
GRID = 0.0015          # the routing grid; a plate rib is ten cells across
MARGIN = 0.05          # how far outside the part the router may wander

PIECE_COL = ["#d2450f", "#1f5f9c"]        # the two pieces, and their mujoco materials
PIECE_MAT = ["propA", "propB"]
RIDGE = "#8d8a80"      # what the part occupies inside the plate's own 6 mm
GHOST = "#d8d5cc"      # its whole silhouette, for context


# ------------------------------------------------------------ the free floor ---

def low_shadow(world: trimesh.Trimesh, h: float) -> Polygon:
    """The ground the part actually occupies below height `h`.

    Not the silhouette. The silhouette is what the part covers seen from above,
    and a plate six millimetres thick does not care what is a hand's width over
    its head -- it cares what is IN its own slab. Slicing first and projecting
    the slice is the difference between a fixture that may cross under a leaning
    part and one that may not.
    """
    low = world.slice_plane([0, h, 0], [0, -1, 0], cap=True)
    tri = COORD.floor(low.triangles)
    return unary_union([Polygon(t) for t in tri if
                        abs(np.cross(t[1] - t[0], t[2] - t[0])) > 1e-12]).buffer(0)


class Floor:
    """A grid of the free ground, and shortest paths across it."""

    def __init__(self, blocked, lo, hi, step=GRID):
        self.step, self.lo = step, np.asarray(lo, float)
        self.nx = int(np.ceil((hi[0] - lo[0]) / step)) + 1
        self.ny = int(np.ceil((hi[1] - lo[1]) / step)) + 1
        gx, gy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        self.xy = np.stack([self.lo[0] + gx * step, self.lo[1] + gy * step], -1)
        pts = self.xy.reshape(-1, 2)
        self.free = shapely.contains_xy(blocked, pts[:, 0], pts[:, 1]) \
            .reshape(self.nx, self.ny) == False       # noqa: E712 -- ndarray, not bool
        # 8-connected, so a diagonal costs what a diagonal costs
        rows, cols, w = [], [], []
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            a = np.s_[max(0, -dx):self.nx - max(0, dx), max(0, -dy):self.ny - max(0, dy)]
            b = np.s_[max(0, dx):self.nx - max(0, -dx), max(0, dy):self.ny - max(0, -dy)]
            ok = self.free[a] & self.free[b]
            ia = (np.arange(self.nx * self.ny).reshape(self.nx, self.ny)[a])[ok]
            ib = (np.arange(self.nx * self.ny).reshape(self.nx, self.ny)[b])[ok]
            d = step * np.hypot(dx, dy)
            rows += [ia, ib]
            cols += [ib, ia]
            w += [np.full(len(ia), d)] * 2
        n = self.nx * self.ny
        self.graph = coo_matrix((np.concatenate(w),
                                 (np.concatenate(rows), np.concatenate(cols))),
                                shape=(n, n)).tocsr()

    def node(self, p) -> int:
        """The nearest FREE cell to a point -- a wall foot may sit just off one."""
        d = np.linalg.norm(self.xy - np.asarray(p)[:2], axis=-1)
        d[~self.free] = np.inf
        return int(np.argmin(d))

    def path(self, a: int, b: int):
        """The shortest free route between two cells, or None if there is none."""
        dist, pred = dijkstra(self.graph, indices=a, return_predecessors=True)
        if not np.isfinite(dist[b]):
            return None, np.inf
        route, cur = [b], b
        while cur != a:
            cur = pred[cur]
            route.append(cur)
        idx = np.array(route[::-1])
        return self.xy.reshape(-1, 2)[idx], float(dist[b])


def spanning(floor: Floor, nodes):
    """The paths that join a group's feet: an MST over grid shortest paths.

    A true Steiner tree would be shorter, so what this returns is an upper bound
    on the plate a grouping needs. That is the direction that keeps the argument
    honest -- the claim is that a grouping IS possible, and a longer plate that
    works still proves it.
    """
    if len(nodes) < 2:
        return [], 0.0
    m = len(nodes)
    paths, cost = {}, np.full((m, m), np.inf)
    for i, j in itertools.combinations(range(m), 2):
        pth, c = floor.path(nodes[i], nodes[j])
        paths[(i, j)] = pth
        cost[i, j] = cost[j, i] = c
    if not np.isfinite(cost[np.triu_indices(m, 1)]).all():
        return None, np.inf
    seen, total, out = {0}, 0.0, []
    while len(seen) < m:
        best = min(((i, j) for i in seen for j in range(m) if j not in seen),
                   key=lambda e: cost[e])
        i, j = best
        out.append(paths[(min(i, j), max(i, j))])
        total += cost[i, j]
        seen.add(j)
    return out, total


# ---------------------------------------------------------------- the pieces ---

def plate_polygon(feet, anchors, paths, keepout):
    """The plate's ground plan: ribs along the routes, a gusset out to each wall.

    Routing a centre-LINE and thickening it afterwards is the mistake this exists
    to avoid: a path may pass two millimetres from the workpiece and be perfectly
    legal, and the eight-millimetre-wide rib grown around it then sits four
    millimetres inside the part. So the thickened plate is cut against the keepout
    at the end, unconditionally. Whatever survives cannot touch the workpiece --
    and if too little survives, the piece falls apart, which the connected-body
    count reports rather than hides.
    """
    polys = []
    for pth in paths:
        # the grid path is a staircase; simplifying it first is what turns the
        # plate into something a drawing can show and a mill could cut
        polys.append(LineString(np.asarray(LineString(pth).simplify(0.0025).coords))
                     .buffer(RIB / 2, cap_style=2, join_style=2))
    for f, q in zip(feet, anchors):
        # the wall's own foot out to where the plate can safely start: a gusset,
        # which is what a fixture would have there anyway
        polys.append(unary_union([f, Point(q).buffer(RIB / 2, cap_style=3)]).convex_hull)
    return unary_union(polys).difference(keepout)


def build_piece(walls, plate2d, h=PLATE_H) -> trimesh.Trimesh:
    """One printed body: the group's walls and the plate that joins their feet."""
    solids = list(walls)
    if plate2d is not None and not plate2d.is_empty:
        for g in (plate2d.geoms if plate2d.geom_type == "MultiPolygon" else [plate2d]):
            if g.area > 1e-8:
                solids.append(COORD.mesh(trimesh.creation.extrude_polygon(g, h)))
    out = trimesh.boolean.union(solids, engine="manifold")
    out.process(validate=True)
    return out


def under_part(poly, world: trimesh.Trimesh, h=PLATE_H, step=0.002):
    """How much of a plate lies UNDER the workpiece, and by how little.

    The whole reason a two-piece merge is possible on a tipped part is that a
    plate may go under the overhang instead of the long way round, so it is worth
    measuring rather than asserting: sample the plate on a 2 mm grid, keep the
    samples the part is over, and shoot a ray up from the plate's top face.
    """
    if poly.is_empty:
        return 0.0, float("inf")
    lo, hi = np.array(poly.bounds[:2]), np.array(poly.bounds[2:])
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + step, step),
                         np.arange(lo[1], hi[1] + step, step), indexing="ij")
    q = np.c_[gx.ravel(), gy.ravel()]
    q = q[shapely.contains_xy(poly, q[:, 0], q[:, 1])]
    if not len(q):
        return 0.0, float("inf")
    org = COORD.lift_floor(q,h+1e-5)
    loc, ray, _ = world.ray.intersects_location(
        org, np.tile([0, 1.0, 0], (len(org), 1)), multiple_hits=False)
    if not len(loc):
        return 0.0, float("inf")
    return float(len(np.unique(ray)) * step ** 2) * 1e4, float(loc[:, 1].min() - h) * 1e3


def foot_polygon(w: trimesh.Trimesh, h=PLATE_H) -> Polygon:
    """What a wall covers inside the plate's own height band.

    The wall is convex, so its cross-section is the hull of the vertices in the
    band. Taking the hull of the WHOLE wall instead would make a leaning wall's
    footprint the shadow of its top, which is where it is not.
    """
    v = np.asarray(w.vertices)
    low = COORD.floor(v[v[:, 1] <= h + 1e-9])
    if len(low) < 3:
        low = COORD.floor(v)
    return MultiPoint(low).convex_hull


# ----------------------------------------------------------------- the check ---

def audit(pieces, world, pts, us, groups) -> dict:
    """Exact booleans where they exist, and say so where they do not.

    `trimesh` hands boolean work to an external engine and there is no guarantee
    one is installed; `manifold3d` is, so these are true intersection volumes.
    A volume, not a flag, because "they touch" and "one is buried 4 mm inside the
    other" are different findings and a boolean would report them the same.
    """
    out = {"engine": "manifold3d exact boolean", "pieces": []}
    for i, m in enumerate(pieces):
        inter = trimesh.boolean.intersection([m, world], engine="manifold")
        hits = []
        for j in groups[i]:
            d = float(trimesh.proximity.closest_point(m, [pts[j]])[1][0])
            past = bool(m.contains([pts[j] + 5e-4 * us[j]])[0])
            hits.append({"contact": int(j) + 1, "gap_to_pad_m": d, "eats_into_part": past})
        out["pieces"].append({
            "group": [int(j) + 1 for j in groups[i]],
            "volume_cm3": float(m.volume) * 1e6,
            "watertight": bool(m.is_watertight),
            "bodies": int(len(m.split(only_watertight=False))),
            "overlap_with_part_mm3": float(inter.volume) * 1e9 if len(inter.faces) else 0.0,
            # the pads touch by definition, so the minimum over ALL of the piece is
            # zero whatever happens and says nothing. What can be read is the
            # clearance AWAY from the pads, which is the number a fixture is judged on
            "min_gap_to_part_mm": float(np.abs(trimesh.proximity.signed_distance(
                world, m.vertices[np.linalg.norm(
                    m.vertices[:, None, :] - np.asarray(pts)[None, groups[i], :],
                    axis=2).min(axis=1) > 0.010])).min()) * 1e3,
            "contacts": hits})
    if len(pieces) == 2:
        inter = trimesh.boolean.intersection(list(pieces), engine="manifold")
        out["overlap_between_pieces_mm3"] = (float(inter.volume) * 1e9
                                             if len(inter.faces) else 0.0)
    return out


# ---------------------------------------------------------------- the drawing ---

def plan(world, shadow, pieces_2d, pts, us, groups, px) -> Image.Image:
    """The floor seen from above: the ridge the part stands on, and the two plates.

    This is the panel that has to carry the grouping argument, so it draws the one
    fact the argument rests on -- the strip of ground the part occupies inside the
    plate's own six millimetres -- and lets the reader see that one plate is east
    of it and the other west, and that neither has to cross.
    """
    fig = plt.figure(figsize=(5.0, 5.4), dpi=px / 5.4, facecolor=PAPER)
    ax = fig.add_subplot(111, facecolor=PAPER)
    sil = unary_union([Polygon(t) for t in COORD.floor(world.triangles)
                       if abs(np.cross(t[1] - t[0], t[2] - t[0])) > 1e-12]).buffer(0)
    for poly, col, a, lab in [(sil, GHOST, 1.0, "the part, seen from above"),
                              (shadow, RIDGE, 1.0,
                               f"what it occupies below {PLATE_H * 1000:.0f} mm")]:
        for g in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly]):
            ax.fill(*np.asarray(g.exterior.coords).T, color=col, alpha=a, zorder=1,
                    label=lab, lw=0)
            lab = None
    for i, poly in enumerate(pieces_2d):
        for g in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly]):
            ax.fill(*np.asarray(g.exterior.coords).T, color=PIECE_COL[i], alpha=.85,
                    zorder=3, lw=0,
                    label=f"piece {'AB'[i]}  ·  contacts "
                          f"{' '.join(str(j + 1) for j in groups[i])}")
    for i, (p, u) in enumerate(zip(pts, us)):
        col = PIECE_COL[[k for k, g in enumerate(groups) if i in g][0]]
        h = COORD.floor(u) / max(np.linalg.norm(COORD.floor(u)), 1e-9)
        # the arrow starts where the push comes from and runs into the contact,
        # the same convention the spheres use
        ax.annotate("", xy=COORD.floor(p), xytext=COORD.floor(p) - 0.020 * h, zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                    shrinkA=0, shrinkB=7))
        ax.scatter(*COORD.floor(p), s=170, c="white", edgecolors=col, linewidths=2.2, zorder=6)
        ax.text(*COORD.floor(p), str(i + 1), color=col, fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=7)
    ax.set_aspect("equal")
    ax.set_axis_off()
    # under the axes, not in a corner of them: which corner is empty depends on the
    # pose, and a legend that lands on a plate hides the thing being argued about
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=9,
              frameon=False, handlelength=1.3, columnspacing=1.4)
    ax.set_title("the ground plan  ·  a plate may pass under an overhang,\n"
                 "it may not cross the strip the part stands on",
                 fontsize=12, color="#1b1b1a", pad=8, linespacing=1.35)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PAPER)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def block(dr, x, y, lines, font, lead=1.42, ink=(40, 40, 38)):
    for ln, col in lines:
        dr.text((x, y), ln, fill=col or ink, font=font)
        y += int(font.size * lead)
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    # pose 6 by default: see the note in pipeline_a1f on why this pose
    ap.add_argument("--pose", type=int, default=6)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    px = args.size
    d = obj_path(args.object)
    out_dir = d / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = d / "_merge_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    r = replicate(args.object, args.pose, args.k, args.seed)
    T, pts, us = r["T"], r["pts"], r["us"]
    n = len(pts)
    world = r["mesh"].copy()
    world.apply_transform(T)
    fitted = [fit_wall(pts[i], us[i], world) for i in range(n)]
    walls, drafts = [f[0] for f in fitted], [f[1] for f in fitted]
    for i, (w, dr_, vol) in enumerate(fitted):
        print(f"  wall {i + 1}: {w.volume * 1e6:5.2f} cm3, draft {dr_:.1f} deg, "
              f"inside the part {vol * 1e9:.4f} mm3")
    feet = [foot_polygon(w) for w in walls]
    shadow = low_shadow(world, PLATE_H)
    lo = np.minimum(COORD.floor(world.bounds[0]), np.array([f.bounds[0] for f in feet]).min(0)) - MARGIN
    hi = np.maximum(COORD.floor(world.bounds[1]), np.array([f.bounds[2] for f in feet]).max(0)) + MARGIN

    # ---- every way of splitting the contacts into two non-empty groups, routed
    print(f"pose {args.pose}: routing all {2 ** (n - 1) - 1} two-group splits "
          f"({n} contacts)")
    trials = []
    for bits in range(1, 2 ** (n - 1)):
        A = [i for i in range(n) if (bits >> i) & 1]
        B = [i for i in range(n) if i not in A]
        if not A or not B:
            continue
        groups = [A, B]
        total, routes, anchors, ok = 0.0, [], [], True
        for g, other in ((A, B), (B, A)):
            # the other piece's walls are obstacles: two fixture pieces may no more
            # pass through each other than through the workpiece. The obstacle is
            # inflated by half a rib as well as the clearance, so that the rib
            # BODY is legal and not merely its centre line
            blocked = unary_union([shadow] + [feet[j] for j in other]) \
                .buffer(CLEAR + RIB / 2)
            floor = Floor(blocked, lo, hi)
            node = [floor.node(np.asarray(feet[j].centroid.coords[0])) for j in g]
            paths, cost = spanning(floor, node)
            if paths is None:
                ok = False
                break
            routes.append(paths)
            anchors.append([floor.xy.reshape(-1, 2)[k] for k in node])
            total += cost
        trials.append(dict(groups=groups, ok=ok, plate=total if ok else np.inf,
                           real=min(len(A), len(B)) >= 2,
                           routes=routes if ok else None,
                           anchors=anchors if ok else None))
        print(f"   {'/'.join(''.join(str(j + 1) for j in g) for g in groups):>12}  "
              + (f"plate {1000 * total:6.1f} mm" if ok else "no route")
              + ("" if trials[-1]["real"] else "   (a group of one is not a merge)"))
    # least plate ALONE is a degenerate objective: a group of one needs no plate at
    # all, so it would always win and the winner would be the un-merged answer with
    # a foot bolted on. A merge has to actually merge, so a split where some piece
    # carries a single contact is ranked behind every split where both carry two.
    best = min(trials, key=lambda t: (not t["real"], t["plate"]))
    groups, routes, anchors = best["groups"], best["routes"], best["anchors"]
    print(f"  -> grouping {'/'.join(''.join(str(j + 1) for j in g) for g in groups)}, "
          f"plate {1000 * best['plate']:.1f} mm")

    # the plates are cut against the workpiece, and the second is cut against the
    # first as well: two pieces that overlap are one piece with extra steps
    keepout = shadow.buffer(CLEAR)
    pieces_2d, pieces = [], []
    for i, g in enumerate(groups):
        poly = plate_polygon([feet[j] for j in g], anchors[i], routes[i],
                             unary_union([keepout] + [p.buffer(CLEAR)
                                                      for p in pieces_2d]))
        pieces_2d.append(poly)
        pieces.append(build_piece([walls[j] for j in g], poly))
    rep = audit(pieces, world, pts, us, groups)
    for i, p in enumerate(rep["pieces"]):
        p["under_cm2"], p["headroom_mm"] = under_part(pieces_2d[i], world)
    for p in rep["pieces"]:
        print(f"   piece {p['group']}: {p['volume_cm3']:.1f} cm3, {p['bodies']} body, "
              f"overlap with part {p['overlap_with_part_mm3']:.3f} mm3, "
              f"min gap {p['min_gap_to_part_mm']:.2f} mm, "
              f"{p['under_cm2']:.1f} cm2 of plate under the part "
              f"(headroom {p['headroom_mm']:.1f} mm)")
        for h in p["contacts"]:
            print(f"      contact {h['contact']}: pad gap {1e6 * h['gap_to_pad_m']:.1f} um, "
                  f"eats into part {h['eats_into_part']}")
    print(f"   piece against piece: {rep['overlap_between_pieces_mm3']:.3f} mm3")

    # ---- the drawing
    frac, gain = steps_of(r["targets"], r["avail"], r["chosen"])
    parts = paint(r["part"], T, r["inside"], set(), tmp, f"mg_p{args.pose}",
                  rel="_merge_tmp")
    # the plan draws what the piece covers on the ground: plate AND wall feet
    ground = [unary_union([pieces_2d[i]] + [feet[j] for j in g])
              for i, g in enumerate(groups)]
    plan_im = plan(world, shadow, ground, pts, us, groups, int(px * 1.55))

    for i, m in enumerate(pieces):
        m.export(tmp / f"piece{i}.stl")
    for i, w in enumerate(walls):
        w.export(tmp / f"wall{i}.stl")
    loose = [(f"_merge_tmp/wall{i}.stl", "propA") for i in range(n)]
    merged = [(f"_merge_tmp/piece{i}.stl", PIECE_MAT[i]) for i in range(len(pieces))]
    V = np.vstack([world.vertices] + [m.vertices for m in pieces])
    wlo, whi = V.min(axis=0), V.max(axis=0)

    sw, sh_ = int(px * 1.00), int(px * 0.78)
    labh = int(px * 0.062)
    shots = []
    for solids, ms, az, show, lab in [
            (loose, walls, 118.0, True, f"before  ·  {n} loose walls"),
            (merged, pieces, 118.0, True, "after  ·  two pieces, from the front"),
            (merged, pieces, 298.0, True, "after  ·  the same two, from behind"),
            # the same camera as `after, from the front`, with the workpiece taken
            # away: two objects, and nothing about the picture that is not printed
            (merged, pieces, 118.0, False, "what you would print")]:
        cam = camera(wlo, whi, az, -18.0, zoom=1.22, lift=0.40)
        im = solid_shot(args.object, T, parts if show else [], solids, cam, sw, sh_)
        # with the workpiece gone nothing can be hidden by it, so every number is
        # solid there; otherwise a contact is pale when its BODY is out of sight
        seen = visible(cam, world if show else trimesh.Trimesh(), ms)
        per = np.zeros(n, bool)
        if len(ms) == n:
            per = seen
        else:
            for gi, g in enumerate(groups):
                per[g] = seen[gi]
        number(im, project(cam, sw, sh_, pts), range(1, n + 1), sw, size=0.052,
               seen=per)
        shots.append((im, lab))

    wide = len(shots) * sw + (len(shots) - 1) * 10
    body = _font(int(px * 0.030))
    head = _font(int(px * 0.040))
    small = _font(int(px * 0.026))

    strip = Image.new("RGB", (wide, sh_ + labh), "white")
    dsr = ImageDraw.Draw(strip)
    for i, (im, lab) in enumerate(shots):
        strip.paste(im, (i * (sw + 10), labh))
        dsr.text((i * (sw + 10) + sw // 2, int(labh * 0.5)), lab, fill=(40, 40, 38),
                 font=head, anchor="mm")

    # ---- the verdict, in words, next to the plan
    pw_ = wide - plan_im.size[0] - int(px * 0.04)
    text = Image.new("RGB", (pw_, plan_im.size[1]), "white")
    dt = ImageDraw.Draw(text)
    y = int(px * 0.03)
    y = block(dt, 0, y, [(f"{args.object}  pose {args.pose}  ·  {n} contacts, "
                          f"two pieces", None)], head, 1.6)
    rows = [("how the grouping was chosen", (110, 110, 106))]
    rows.append(("   all seven splits routed; a group of one is not a merge, so those",
                 (110, 110, 106)))
    rows.append(("   are ranked last, and the rest by how much plate they need",
                 (110, 110, 106)))
    for t in sorted(trials, key=lambda t: (not t["real"], t["plate"])):
        tag = "/".join("".join(str(j + 1) for j in g) for g in t["groups"])
        mark = "   <-- taken" if t is best else ""
        rows.append((f"   {tag:<10}"
                     + (f"plate {1000 * t['plate']:6.1f} mm" if t["ok"]
                        else "no route: a group cannot be joined")
                     + ("" if t["real"] else "   one piece carries one contact")
                     + mark,
                     (30, 30, 28) if t is best else (110, 110, 106)))
    rows.append(("", None))
    rows.append(("what the merge is checked against  ·  "
                 + rep["engine"], (110, 110, 106)))
    for p in rep["pieces"]:
        rows.append((f"   piece {'AB'[rep['pieces'].index(p)]}  contacts "
                     f"{' '.join(str(c) for c in p['group'])}  ·  "
                     f"{p['volume_cm3']:.0f} cm3  ·  {p['bodies']} connected "
                     f"{'body' if p['bodies'] == 1 else 'BODIES'}", None))
        rows.append((f"      inside the workpiece: {p['overlap_with_part_mm3']:.3f} mm3"
                     f"   ·   nearest approach away from its pads: "
                     f"{p['min_gap_to_part_mm']:.2f} mm",
                     (150, 40, 30) if p["overlap_with_part_mm3"] > 1e-3 else None))
        if p["under_cm2"] > 0.5:
            rows.append((f"      {p['under_cm2']:.0f} cm2 of its plate runs UNDER the "
                         f"workpiece, headroom {p['headroom_mm']:.1f} mm",
                         (150, 40, 30) if p["headroom_mm"] < 1.0 else None))
        for h in p["contacts"]:
            rows.append((f"      contact {h['contact']}: pad sits "
                         f"{1e6 * h['gap_to_pad_m']:.0f} um off the surface, "
                         f"past the contact: {'YES' if h['eats_into_part'] else 'no'}",
                         (150, 40, 30) if h["eats_into_part"] or
                         h["gap_to_pad_m"] > 1e-5 else None))
    rows.append((f"   piece against piece: "
                 f"{rep['overlap_between_pieces_mm3']:.3f} mm3",
                 (150, 40, 30) if rep["overlap_between_pieces_mm3"] > 1e-3 else None))
    rows.append(("", None))
    # the one sentence a reader should be able to take away, and it has to follow
    # from the numbers above it rather than from the fact that the picture is tidy
    won = (all(p["overlap_with_part_mm3"] <= 1e-3 and p["bodies"] == 1
               and not any(h["eats_into_part"] for h in p["contacts"])
               for p in rep["pieces"])
           and rep["overlap_between_pieces_mm3"] <= 1e-3)
    tight = min(p["headroom_mm"] for p in rep["pieces"])
    rows.append((f"two pieces: {'YES' if won else 'NO'}   ·   both bodies connected, "
                 f"neither inside the workpiece, nothing between them",
                 (20, 90, 60) if won else (150, 40, 30)))
    if np.isfinite(tight):
        rows.append((f"the margin that decides it is vertical: {tight:.1f} mm of "
                     f"headroom over the thinner plate", (110, 110, 106)))
    block(dt, 0, y, rows, body, 1.55)

    top = Image.new("RGB", (wide, plan_im.size[1]), "white")
    top.paste(plan_im, (0, 0))
    top.paste(text, (plan_im.size[0] + int(px * 0.04), 0))

    caption = (
        f"{args.object}  pose {args.pose}   ·   the same {n} contacts "
        f"{pipeline_note(frac)}   ·   nothing about the force answer moves: the "
        f"bearing faces are the same tangent planes at the same {n} points\n"
        f"the plate is {PLATE_H * 1000:.0f} mm thick and {RIB * 1000:.0f} mm wide, "
        f"kept {CLEAR * 1000:.0f} mm clear of the workpiece   ·   routes are shortest "
        f"paths on a {GRID * 1000:.1f} mm grid over the free floor, joined by a "
        f"minimum spanning tree")
    font = _font(int(px * 0.032))
    headh = int(px * 0.13)
    page = Image.new("RGB", (wide, headh + top.size[1] + strip.size[1]), "white")
    page.paste(top, (0, headh))
    page.paste(strip, (0, headh + top.size[1]))
    ImageDraw.Draw(page).text((16, int(headh * .18)), caption, fill=(90, 90, 90),
                              font=font)
    out = out_dir / f"merged_pose{args.pose}.png"
    page.save(out)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {out}  {page.size[0]}x{page.size[1]}")


def pipeline_note(frac):
    return f"that answered {100 * frac[-1]:.0f}% of the requirement in figure 1"


if __name__ == "__main__":
    main()
