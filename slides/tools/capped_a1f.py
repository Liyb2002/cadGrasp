"""The same support-design question as `reach.py`, asked of contacts that can only push so hard.

`reach.py` asks whether the requirement

    T = up - K*d          one weight up, minus the disturbance

lies in the CONE the chosen pushes span. A cone is scale-free, so that question
throws magnitude away: a direction counts as answered however much force along it
the answer would take. Here the contacts are given a strength, and the test
becomes a linear program instead,

    lam0*up + sum_i lam_i * u_i = T,     lam0 >= 0,     0 <= lam_i <= 1

so a direction can sit comfortably inside the cone and still be out of reach
because the force it needs is more than the supports can deliver. That is the
whole of the change, and it is why the answer gets bigger rather than smaller.

The three modelling decisions, taken as given:

  * the process push is at most one body weight (K = 1, as before);
  * each support delivers at most one body weight -- that is the new bound;
  * the floor stays unbounded. It is the ground; it carries whatever it is given,
    so `lam0` has no upper bound and that is what keeps the south of the sphere
    free.

And one that is a modelling choice rather than physics, so it is stated as one:
EACH DISTINCT PUSH DIRECTION MAY BE USED AT MOST ONCE. Two supports bearing on
the same face are, in force space, one support of twice the strength, so without
this rule the cap could always be bought off by repetition and would mean
nothing. The rule is what makes "how many supports" and "how strong is a support"
two separate questions.

Nothing else about the physics moves. Translation only, a contact contributes
only the direction it pushes, and a force is still drawn WHERE IT COMES FROM.

--- the search ---------------------------------------------------------------

A1-f offers 850-1950 candidate directions per pose, so subsets cannot be
enumerated and an LP per candidate subset per requirement is far too slow. Two
facts make it tractable.

The first is that feasibility has a closed form. The achievable set

    S = ray(up)  +  sum_i [0, u_i]

is a Minkowski sum of a ray and segments, so it is a polyhedron whose every facet
is spanned by two of its generators; its facet normals are therefore the pairwise
cross products, and `T` is achievable exactly when `y.T <= sum_i max(0, y.u_i)`
for each of them with `y.up <= 0`. That is a couple of dozen dot products instead
of a linear program, it is exact rather than tolerance-bound, and it batches over
every requirement at once. It agrees with `scipy.optimize.linprog` on every
requirement of every pose, which is checked and recorded, and the LP still gives
the final verdict.

The second is that the same support function gives a LOWER bound that can be
computed without searching at all. For ANY direction `y` with `y.up <= 0`, a set
of supports that answers everything must satisfy

    max_j (y . T_j)  <=  sum_i x_i * max(0, y.u_i),      x the 0/1 indicator

which is one linear covering constraint over the candidates. Sampling `y` over
the lower hemisphere and minimising `sum x` subject to those constraints is a
covering MILP, and because every constraint is necessary its optimum is a valid
lower bound on the number of supports, whatever they are. `reach.py`'s compass
bound is still valid too -- a capped answer is an uncapped answer, so anything the
cone test forbids the cap forbids as well -- but it is weaker here and the covering
bound subsumes it on every pose. The search is then a
cutting-plane loop: solve the MILP, test its answer exactly, and if it fails, add
the facet normals it violated as new rows -- they are necessary conditions too,
so the bound stays valid and gets tighter. When the MILP's own answer passes the
exact test, the bound is attained and the count is PROVEN MINIMAL, not merely
found. On A1-f the loop closes in a handful of rounds on every pose.

This replaces, and does not reuse, `reach.py`'s beam over pairs: a beam reports an
upper bound, and there is no need to settle for one here. It also cannot use
greedy growth, for the reason `reach.py` gives -- `up` with a single support spans
a plane, a plane has no volume, and every candidate scores zero on step one. The
greedy fallback in `grow` is only reached if the MILP runs out of budget, and it
starts from the MILP's incumbent rather than from nothing.

`cover.py`'s `covered()` is not used anywhere here. Its residual tolerance is
loose enough that a 0.003-degree wobble in the generators has been measured
turning a true 58% into a false 100%, and the whole point of this tool is a
quantity that tolerance would swamp.

--- the extra panel ----------------------------------------------------------

`reach.py`'s reach panel drew the cone, which under a cap is no longer the answer.
What replaces it is the RADIAL EXTENT of the achievable set: for every direction
`v` on the sphere, the largest force

    rho(v) = max { r >= 0 : r*v in S }

the chosen supports plus the floor can put along it, painted as a graded field.
`S` is convex and contains the origin, so a requirement is answered exactly when
`|T| <= rho(T_hat)` -- the field is not an illustration of the test, it IS the
test, drawn. Bare grey means `rho = 0`, no mixture points that way at all; pale
through deep teal is 0.25 through 2+ body weights, which is the range the
requirements occupy (`|T| = |up - d|` runs to 2). So the reader can tell a
direction that is unreachable from one that is reachable but not strongly enough,
which a single flat teal could not say.

    python slides/tools/capped_a1f.py A1-f
    python slides/tools/capped_a1f.py A1-f --poses 0 --reuse
"""
from __future__ import annotations

import argparse
import itertools
import shutil
import time

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.spatial import cKDTree

from common import obj_path, read_json, write_json         # noqa: E402
from cover import (AZIM, DONE, ELEV, HAVE, INK, NEED, PUSH_BLUE, UP, globe_png,
                   neighbours, paint, sheet, shot, tiling)  # noqa: E402
from disturbances import _font                             # noqa: E402
from reach import compass                                  # noqa: E402
from shrink_support import angled_pushes                   # noqa: E402
from supports import CONTACT_EPS, region_mask              # noqa: E402
from work_regions import BOUNDARY_EDGE, gun_directions, spot_region  # noqa: E402

FLOOR_MARK = "#3d3d3a"
# one hue, six lightnesses: the quantity is an amount of force, and an amount reads
# as light-to-dark. A rainbow would invent categories that are not there. The palest
# band still has to be plainly TEAL, because the thing it must not be confused with
# is BARE -- a direction no mixture points in at all -- and that is a different fact,
# not a smaller number, so it gets its own dead grey.
SHADE = ["#c9e7e1", "#a3d8cf", "#7ec8bd", "#55ab9f", "#33897d", "#186158"]
BREAK = [0.25, 0.5, 1.0, 1.5, 2.0, np.inf]      # upper edge of each shade, body weights
BARE = "#cbc6ba"
SHORT = "#8A3FA8"        # owed, the direction IS available, it is simply not strong enough.
                         # violet rather than a second red, matching capped_cuboid: two reds a
                         # shade apart read as one, and this is the distinction the cap exists
                         # to show
MARK = "#7a2a06"         # the ring around a contact number on the part
FOVY = 45.0              # SCENE names no fovy, so this is mujoco's default
CAP = 1.0                                        # what one support can deliver


# ------------------------------------------------------------------ the model ---

def normals(G: np.ndarray) -> np.ndarray:
    """Unit `y` with `y.up <= 0` that between them cut out the achievable set.

    Every facet of a Minkowski sum of segments and a ray is spanned by the
    generators that lie in it, and a two-dimensional face needs two independent
    ones, so the facet normals are the pairwise cross products. Any `y` in the
    half space is a VALID inequality whether or not it is a facet, so a longer
    list only ever costs time -- but a short one is wrong, and a flat generator
    set is exactly where it comes up short: if the generators are rank two the
    cross products all collapse onto the plane normal and the set looks unbounded
    inside its own plane. That is the `+1` panel, so it is not a corner case, and
    the in-plane edge normals are added back by crossing again with each
    generator.
    """
    m = G.shape[1]
    if m < 2:
        return np.zeros((0, 3))
    idx = np.array(list(itertools.combinations(range(m), 2)))
    c = np.cross(G[:, idx[:, 0]].T, G[:, idx[:, 1]].T)
    n = np.linalg.norm(c, axis=1)
    c = c[n > 1e-10] / n[n > 1e-10, None]
    if len(c) and np.linalg.matrix_rank(G, tol=1e-9) < 3:
        extra = np.cross(np.repeat(c, m, axis=0), np.tile(G.T, (len(c), 1)))
        n = np.linalg.norm(extra, axis=1)
        c = np.vstack([c, extra[n > 1e-10] / n[n > 1e-10, None]])
    Y = np.vstack([c, -c])
    return Y[Y @ UP <= 1e-12]


def screen(mesh, T, px, pts):
    """Where the contacts land on `shot`'s picture, so they can be numbered.

    Five supports is two more than `reach.py` ever placed, and with one camera
    for the whole figure some of them end up behind the part or under the floor
    -- on this pose two consecutive renders came out pixel for pixel identical,
    so a reader could not count them and could not tell the steps apart. `shot`
    does not hand back its camera, so the camera it builds is rebuilt here:
    same azimuth and elevation, same 1.25x framing of the mesh bounding box.
    Checked by drawing the numbers over a render whose arrows ARE visible; they
    sit on the arrowheads.
    """
    V = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
    lo, hi = V.min(axis=0), V.max(axis=0)
    fov = np.tan(np.deg2rad(FOVY / 2))
    dist = 1.25 * float(np.linalg.norm(hi - lo)) / 2 / fov
    a, e = np.deg2rad(AZIM + 180.0), np.deg2rad(-ELEV)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([np.sin(a), -np.cos(a), 0.0])
    v = np.asarray(pts) - ((lo + hi) / 2 - dist * fwd)
    s = np.stack([v @ right, v @ np.cross(right, fwd)], 1) / (v @ fwd)[:, None] / fov
    return np.stack([(0.5 + 0.5 * s[:, 0]) * px, (0.5 - 0.5 * s[:, 1]) * px], 1)


def radial(G: np.ndarray, V: np.ndarray) -> np.ndarray:
    """The largest force the generators can put along each direction of `V`.

    `S` is convex and holds the origin, so it meets the ray through `v` in a
    segment and this one number settles every question about `v`: zero means the
    direction cannot be produced at all, and a requirement `T` is answered exactly
    when `|T| <= rho(T_hat)`.
    """
    Y = normals(G)
    if not len(Y):                       # the floor on its own: straight up, forever
        return np.where(V @ UP > 1 - 1e-9, np.inf, 0.0)
    h = np.maximum(0.0, Y @ G[:, 1:]).sum(axis=1) if G.shape[1] > 1 else np.zeros(len(Y))
    A = Y @ V.T
    return np.where(A > 1e-12, h[:, None] / np.maximum(A, 1e-300), np.inf).min(axis=0)


def answered(G: np.ndarray, T: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(T, axis=1)
    return mag <= radial(G, T / mag[:, None]) + 1e-9


def lp_answered(G: np.ndarray, T: np.ndarray) -> np.ndarray:
    """The same question put to `linprog`, which is what the model literally says."""
    n = G.shape[1] - 1
    c, bnd = np.zeros(1 + n), [(0, None)] + [(0, CAP)] * n
    return np.array([linprog(c, A_eq=G, b_eq=t, bounds=bnd, method="highs").status == 0
                     for t in T])


def least_effort(G: np.ndarray, T: np.ndarray) -> np.ndarray:
    """The smallest total SUPPORT force that answers each requirement.

    `lam` is not unique, so reporting "the force a contact carries" needs a stated
    choice of solution; this is the cheapest one the supports can get away with.
    The floor is left out of the objective because it is free -- charging for it
    would push the answer onto the supports, which is the opposite of the truth.
    """
    n = G.shape[1] - 1
    c = np.r_[0.0, np.ones(n)]
    bnd = [(0, None)] + [(0, CAP)] * n
    out = np.zeros((len(T), n))
    for j, t in enumerate(T):
        r = linprog(c, A_eq=G, b_eq=t, bounds=bnd, method="highs")
        out[j] = r.x[1:] if r.status == 0 else np.nan
    return out


# ----------------------------------------------------------------- the search ---

def rows(Y: np.ndarray, targets: np.ndarray, avail: np.ndarray):
    """One covering constraint per sampled direction: capacity along `y` >= demand."""
    return np.maximum(0.0, Y @ avail.T), (Y @ targets.T).max(axis=1)


def grow(targets, avail, start):
    """Fallback only: add whichever candidate answers the most that is still owed.

    Never used from an empty start -- `up` plus one support spans a plane and every
    candidate would score zero -- so it takes the MILP's incumbent as its seed and
    is reached only when the MILP runs out of budget. It reports an upper bound.
    """
    chosen = list(start)
    mag = np.linalg.norm(targets, axis=1)
    left = np.flatnonzero(~answered(np.column_stack([UP] + [avail[c] for c in chosen]),
                                    targets))
    G = np.column_stack([UP] + [avail[c] for c in chosen])
    was = float(np.maximum(0.0, mag[left] - radial(G, targets[left] / mag[left, None])).sum())
    while len(left) and len(chosen) < 14:
        m, dirs = mag[left], targets[left] / mag[left, None]
        best, score, mask = None, None, None
        for k in range(len(avail)):
            if k in chosen:
                continue
            r = radial(np.column_stack([UP] + [avail[c] for c in chosen] + [avail[k]]), dirs)
            ok = m <= r + 1e-9
            s = (int(ok.sum()), -float(np.maximum(0.0, m - r).sum()))
            if score is None or s > score:
                best, score, mask = k, s, ok
        if score[0] == 0 and -score[1] >= was - 1e-12:      # nothing left to add
            break
        chosen.append(best)
        left, was = left[~mask], -score[1]
    return chosen


def search(targets, avail, tiles, cuts=14, milp_seconds=420.0):
    """The fewest supports, by a covering MILP tightened until its answer is true.

    Every row is a necessary condition, so the MILP optimum is a lower bound on
    any answering set at every iteration; when its own answer passes the exact
    test the bound is attained and the count is minimal rather than merely found.
    """
    note, mag = [], np.linalg.norm(targets, axis=1)
    Y = tiles[tiles[:, 2] <= 1e-12]                 # y.up <= 0 is the whole admissible set
    Y = Y[(Y @ targets.T).max(axis=1) > 1e-9]       # a row demanding nothing constrains nothing
    n, t0, bound = len(avail), time.time(), 0
    incumbent = None
    for it in range(cuts):
        C, R = rows(Y, targets, avail)
        res = milp(np.ones(n), constraints=[LinearConstraint(C, lb=R, ub=np.inf)],
                   integrality=np.ones(n), bounds=Bounds(0, 1),
                   options=dict(time_limit=milp_seconds, mip_rel_gap=0.0))
        if res.status != 0 or res.x is None:
            note.append(f"the covering MILP stopped early ({res.message.strip()}); "
                        f"the count below is an upper bound only")
            break
        bound = int(round(res.fun))
        S = list(np.flatnonzero(res.x > 0.5))
        incumbent = S
        G = np.column_stack([UP] + [avail[i] for i in S])
        ok = answered(G, targets)
        if ok.all():
            note.append(f"{bound} is optimal: the covering bound forbids fewer and this "
                        f"set attains it ({it + 1} MILP solves, {len(Y)} rows, "
                        f"{time.time() - t0:.0f}s)")
            return S, bound, True, note
        # the facet normals this set violates are necessary conditions the sample
        # missed, so feeding them back cuts off the whole family it belongs to
        # rather than just this one set, and keeps the bound valid
        fn = normals(G)
        hv = np.maximum(0.0, fn @ avail[S].T).sum(axis=1)
        v = (fn @ targets.T).max(axis=1) - hv
        add = np.argsort(-v)[:24]
        add = add[v[add] > 1e-9]
        if not len(add):
            note.append("no violated facet normal to add; the sampled bound has stalled")
            break
        Y = np.vstack([Y, fn[add]])
    chosen = grow(targets, avail, incumbent if incumbent is not None else [])
    note.append(f"greedy growth from the MILP incumbent: {len(chosen)} supports, "
                f"an upper bound; the proven lower bound is {bound}")
    return chosen, bound, False, note


def order_by_gain(targets, avail, chosen):
    """Put the chosen supports in the order that makes each panel add the most.

    A set has no order, and the +1 +2 +3 panels need one; taking it greedily is
    the only choice that makes the numbers under them climb for a reason.
    """
    rest, out = list(chosen), []
    while rest:
        best = max(rest, key=lambda k: answered(
            np.column_stack([UP] + [avail[c] for c in out + [k]]), targets).mean())
        out.append(best)
        rest.remove(best)
    return out


# ---------------------------------------------------------------- the drawing ---

def shades(rho):
    """The graded field, one fill per band, with rho = 0 painted rather than left bare.

    Left to the default the zero band comes out a washed-out near-white, which is
    the one reading it must not have: `no force at all this way` is not `a little`.
    """
    out, lo = [(np.flatnonzero(rho <= 0), BARE, .95)], 0.0
    for hi, col in zip(BREAK, SHADE):
        out.append((np.flatnonzero((rho > lo) & (rho <= hi)), col, .95))
        lo = hi
    return out


def legend(draw, x, y, w, font):
    """The ramp spelled out, because a shade means a number here."""
    draw.text((x, y - int(1.5 * w)), "force available, body weights",
              fill=(90, 90, 90), font=font)
    draw.rectangle([x, y, x + w, y + w], fill=BARE, outline=(200, 200, 200))
    draw.text((x + w // 2, y + w + int(.3 * w)), "none", fill=(120, 120, 120),
              font=font, anchor="ma")
    x += 2 * w
    labels = ["0", ".25", ".5", "1", "1.5", "2", "2+"]
    draw.text((x, y + w + int(.3 * w)), labels[0], fill=(120, 120, 120), font=font,
              anchor="ma")
    for i, col in enumerate(SHADE):
        draw.rectangle([x + i * w, y, x + (i + 1) * w, y + w],
                       fill=col, outline=(200, 200, 200))
        draw.text((x + (i + 1) * w, y + w + int(.3 * w)), labels[i + 1],
                  fill=(120, 120, 120), font=font, anchor="ma")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--poses", type=int, nargs="+", default=None)
    ap.add_argument("--region", choices=("spray", "spot"), default="spray")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=840)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push-points", type=int, default=90)
    ap.add_argument("--push-dirs", type=int, default=24)
    ap.add_argument("--cuts", type=int, default=14)
    ap.add_argument("--milp-seconds", type=float, default=420.0)
    # the search is minutes and the figure is seconds, so redrawing has to be able
    # to skip it
    ap.add_argument("--reuse", action="store_true",
                    help="take the supports from the existing capped_k*.json")
    args = ap.parse_args()

    d = obj_path(args.object)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"]
    poses = args.poses if args.poses is not None else range(len(examples))
    ico, tiles = tiling(4)
    tree = cKDTree(tiles)
    adjacency = neighbours(ico)
    px = args.size
    out_dir = d / "capped"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = d / "_capped_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    old = {}
    if args.reuse and (out_dir / f"capped_k{args.k:g}.json").exists():
        old = {r["pose"]: r for r in
               read_json(out_dir / f"capped_k{args.k:g}.json")["poses"]}
    # what the uncapped tool found, for the one comparison worth making
    unc = {}
    if (d / "reach" / f"reach_k{args.k:g}.json").exists():
        unc = {r["pose"]: r for r in
               read_json(d / "reach" / f"reach_k{args.k:g}.json")["poses"]}

    records = []
    for pose in poses:
        T = np.asarray(examples[pose]["T_world_mesh"])
        R, t = T[:3, :3], T[:3, 3]
        # ---- exactly reach.py's pipeline, seed and all; n_targets and the region
        # area are checked against reach_k1.json and match on all ten poses
        rng = np.random.default_rng(args.seed + 1000 * pose)
        if args.region == "spot":
            reg = spot_region(mesh, T, rng, float(rng.uniform(0.17, 0.33)),
                              BOUNDARY_EDGE * float(np.linalg.norm(mesh.extents)))
            part, inside = reg.pop("mesh"), reg.pop("mask")
            n_passes = reg["n_passes"]
        else:
            n_passes = int(rng.integers(1, 4))
            part = mesh
            inside = region_mask(mesh, T, gun_directions(rng, n_passes),
                                 mesh.triangles_center, mesh.face_normals)
        area = float(part.area_faces[inside].sum() / part.area)

        pw, pu = angled_pushes(part, T, inside, args.push_points, args.push_dirs,
                               args.seed + pose)
        targets = UP - args.k * pu
        keep = np.linalg.norm(targets, axis=1) > 1e-9
        targets, pw, pu = targets[keep], pw[keep], pu[keep]
        mag = np.linalg.norm(targets, axis=1)
        unit = targets / mag[:, None]

        on_floor = (part.triangles_center @ R.T + t)[:, 2] <= CONTACT_EPS
        off = np.flatnonzero(~inside & ~on_floor)
        push = -(part.face_normals[off] @ R.T)
        push /= np.linalg.norm(push, axis=1, keepdims=True)
        hit = tree.query(push)[1]
        first = {}
        for k, h in enumerate(hit):
            first.setdefault(h, k)
        rep = np.array([first[h] for h in sorted(first)])
        avail, avail_face = push[rep], off[rep]

        if pose in old and args.reuse:
            # matched by direction, not by index: the candidate list is rebuilt here
            # and only the directions themselves are stable across runs
            chosen = [int(np.argmax(avail @ np.asarray(s["push"])))
                      for s in old[pose]["supports"]]
            bound, proven, note = (old[pose]["lower_bound"],
                                   old[pose]["proven_minimal"], old[pose]["notes"])
        else:
            chosen, bound, proven, note = search(targets, avail, tiles, args.cuts,
                                                 args.milp_seconds)
            chosen = order_by_gain(targets, avail, chosen)

        G = np.column_stack([UP] + [avail[c] for c in chosen])
        # the final verdict goes to the linear program, and the fast test is held
        # against it rather than trusted; a rank check as well, because a flat
        # generator set is the one shape both could agree about and be wrong
        st = answered(G, targets)
        lp = lp_answered(G, targets)
        disagree = int((st != lp).sum())
        rank = int(np.linalg.matrix_rank(G, tol=1e-9))
        lam = least_effort(G, targets[lp])
        peak = np.nanmax(lam, axis=0) if len(lam) else np.zeros(len(chosen))
        at_cap = int((lam >= CAP - 1e-6).any(axis=1).sum()) if len(lam) else 0
        head = float(np.min(radial(G, unit) / mag)) if len(mag) else float("inf")
        frac = float(lp.mean())
        span = compass(targets)
        pts = part.triangles_center[avail_face[chosen]] @ R.T + t
        us = avail[chosen]

        # ---- a force is drawn WHERE IT COMES FROM, at -F, arrow running into the
        # centre, so the floor's G sits at the south pole where the floor is.
        h_job, h_back, h_want = tree.query(-pu)[1], tree.query(pu)[1], tree.query(-unit)[1]
        job = np.flatnonzero(sheet(adjacency, h_job))
        back = np.flatnonzero(sheet(adjacency, h_back))
        want = sheet(adjacency, h_want)
        near = cKDTree(-unit).query(tiles)[1]
        band = np.flatnonzero(sheet(adjacency, tree.query(-avail)[1]))
        few = np.random.default_rng(11).choice(len(pu), min(26, len(pu)), replace=False)
        fewt = np.random.default_rng(14).choice(len(unit), min(26, len(unit)), replace=False)

        panels = [([(job, PUSH_BLUE, .95)], [], [(pu[few], PUSH_BLUE, False)], [],
                   "the job"),
                  ([(back, NEED, .95)], [], [(-pu[few], NEED, False)], [],
                   "reverse the pushes\nthe far side, sideways in full"),
                  ([(np.flatnonzero(want), NEED, .95)], [], [(unit[fewt], NEED, False)],
                   [(-UP, FLOOR_MARK, "G")],
                   f"+ its own weight\n{span:.0f}° wide, |T| up to {mag.max():.2f}w"),
                  ([(band, HAVE, .95)], [(-avail, HAVE, 26, True)], [(avail, HAVE, False)],
                   [(-UP, FLOOR_MARK, "G")],
                   f"what we have\n{len(avail)} pushes, 1w each")]
        for i in range(len(chosen) + 1):
            Gi = np.column_stack([UP] + [avail[c] for c in chosen[:i]])
            rho = radial(Gi, -tiles)          # the field, drawn where it acts from
            sti = answered(Gi, targets)
            marks = [(-UP, FLOOR_MARK, "G")]
            marks += [(-avail[chosen[j]], "#d2450f", str(j + 1)) for j in range(i)]
            # what is still owed is owed for one of two different reasons, and under
            # a cap they are the whole distinction: the deep red is a direction no
            # mixture points in, the pale red is one that IS available and is simply
            # not strong enough. A single red would hide exactly what changed.
            owed = want & ~sti[near]
            fills = shades(rho)
            fills += [(np.flatnonzero(want & sti[near]), DONE, .95),
                      (np.flatnonzero(owed & (rho > 0)), SHORT, .95),
                      (np.flatnonzero(owed & (rho <= 0)), NEED, .95)]
            arr = [(unit[fewt][~sti[fewt]], NEED, False)] if not sti.all() else []
            # the title reports what capped_cuboid reports: how much is done and
            # how hard the worst contact is working at THAT step. Under a cap the
            # reach percentage is no longer the interesting number -- it hits 100%
            # while most of the requirement is still unmet.
            step_lam = least_effort(Gi, targets[sti]) if i and sti.any() else None
            title = (f"{'floor only' if i == 0 else f'+{i}'}\n"
                     f"{100 * sti.mean():.0f}% done")
            if step_lam is not None and len(step_lam) and np.isfinite(step_lam).any():
                title += f"  ·  peak {np.nanmax(step_lam):.2f} W"
            panels.append((fills, [], arr, marks, title))
        # the field on its own, with the requirement only outlined, so the strength
        # UNDER the requirement can be read instead of being painted over. Two tiles
        # of ring, because one is thinner than the jaggedness of the boundary itself
        # and reads as speckle.
        rho = radial(G, -tiles)
        inner = want.copy()
        for _ in range(2):
            inner &= ~((adjacency @ ~inner) > 0)
        tight = int(np.argmin(radial(G, unit) / mag))
        panels.append((shades(rho) + [(np.flatnonzero(want & ~inner), INK, .95),
                                      (np.flatnonzero(want & ~st[near]), NEED, .95)],
                       [], [], [(-UP, FLOOR_MARK, "G"),
                                (-unit[tight], "#B02A26", "tight")],
                       f"the strength field\nasks {mag.max():.2f}w · margin {head:.2f}"))

        parts = paint(part, T, inside, set(), tmp, f"cap_p{pose}", rel="_capped_tmp")
        blue = [(pw[j], pu[j]) for j in few]
        bare = shot(args.object, T, parts, px, [], [], triad=False)
        forces = shot(args.object, T, parts, px, blue, [], triad=False)
        shots = [bare, forces, forces, forces]
        # number the contacts on the part, as capped_cuboid does. With four and
        # five supports the one shared camera buries some of them -- on that
        # object two consecutive renders came out pixel for pixel identical --
        # so the count has to be written on rather than left to be seen.
        xy = screen(mesh, T, px, pts) if len(chosen) else np.zeros((0, 2))
        numf = _font(int(px * 0.055))

        def numbered(im, n):
            dr = ImageDraw.Draw(im)
            for j in range(n):
                dr.text(tuple(xy[j]), str(j + 1), fill="white", font=numf, anchor="mm",
                        stroke_width=4, stroke_fill=MARK)
            return im

        shots += [numbered(shot(args.object, T, parts, px, blue,
                                [(pts[j], us[j]) for j in range(i)], triad=False), i)
                  for i in range(len(chosen) + 1)]
        shots.append(shots[-1])               # the field panel sits over the finished part
        assert len(shots) == len(panels), (len(shots), len(panels))

        per = (len(panels) + 1) // 2
        blocks = []
        for ps, sh_ in [(panels[:per], shots[:per]), (panels[per:], shots[per:])]:
            g = globe_png(ico, ps, px)
            gw = g.size[0] // max(len(ps), 1)
            gh = int(g.size[1] * 0.99)
            blk = Image.new("RGB", (gw * len(ps), gh + gw), "white")
            blk.paste(g.crop((0, 0, g.size[0], gh)), (0, 0))
            for i, im in enumerate(sh_):
                blk.paste(im.resize((gw, gw)), (i * gw, gh))
            blocks.append(blk)
        caption = (
            f"{args.object}  pose {pose}   ·   BLUE = the disturbances   ·   GREY = "
            f"answered   ·   PALE RED = that way is available, just not hard enough   "
            f"·   DEEP RED = that way is not available at all\n"
            f"TEAL = the largest force the chosen pushes can put along that direction, "
            f"STONE = none   ·   ORANGE = a push this workpiece can supply   ·   "
            f"G = the floor, 1 2 3 = the supports")
        font = _font(int(px * 0.042))
        small = _font(int(px * 0.026))
        headh = int(px * 0.16)
        capw = ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textbbox(
            (16, 0), caption, font=font)[2] + 16
        wide = max(max(b.size[0] for b in blocks), capw + int(px * 0.45))
        page = Image.new("RGB", (wide, headh + sum(b.size[1] for b in blocks)), "white")
        y = headh
        for b in blocks:
            page.paste(b, (0, y))
            y += b.size[1]
        dr = ImageDraw.Draw(page)
        dr.text((16, int(headh * .18)), caption, fill=(90, 90, 90), font=font)
        legend(dr, capw + int(px * 0.05), int(headh * .34), int(px * 0.038), small)
        out = out_dir / f"pose{pose}.png"
        page.save(out)

        records.append({"pose": pose, "n_passes": int(n_passes),
                        "region_area_fraction": area,
                        "n_targets": int(len(targets)),
                        "compass_span_deg": span,
                        "max_target_magnitude": float(mag.max()),
                        "n_available": int(len(avail)),
                        "n_supports": len(chosen),
                        "lower_bound": int(bound),
                        "proven_minimal": bool(proven),
                        "answered_fraction": frac,
                        "uncapped_n_supports": unc.get(pose, {}).get("n_supports"),
                        "min_margin_rho_over_T": head,
                        "requirements_driving_a_contact_to_its_cap": at_cap,
                        "fast_test_vs_linprog_disagreements": disagree,
                        "generator_rank": rank,
                        "notes": note,
                        "supports": [{"p": [float(x) for x in pts[i]],
                                      "push": [float(x) for x in us[i]],
                                      "peak_force": float(peak[i])}
                                     for i in range(len(chosen))]})
        print(f"pose {pose:2d}: {len(avail):4d} pushes, {span:5.1f}° wide "
              f"-> {len(chosen)} supports (lower bound {bound}"
              f"{', PROVEN' if proven else ''}), {100 * frac:5.1f}% answered, "
              f"peak {np.array2string(peak, precision=2)}, margin {head:.2f}   {out}",
              flush=True)
        for n in note:
            print(f"          {n}", flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    path = out_dir / f"capped_k{args.k:g}.json"
    if args.poses is not None and path.exists():
        keep_old = [r for r in read_json(path)["poses"]
                    if r["pose"] not in {r2["pose"] for r2 in records}]
        records = sorted(records + keep_old, key=lambda r: r["pose"])
    write_json(path, {"object": args.object, "k": args.k, "cap": CAP,
                      "poses": records})


if __name__ == "__main__":
    main()
