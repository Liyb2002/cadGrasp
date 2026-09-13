"""`capped_a1f.py`'s support design, re-asked with a 90-degree rule on the chosen set.

`capped_a1f.py` answers a SAMPLE of disturbances and says nothing about the
directions between them. Measured on the designs it produced for A1-f, the
horizontal safety factor -- how much sideways force the supports can raise along
the worst compass bearing, over how much the worst requirement asks for there --
runs 1.000 to 1.084. The designs pass because the weak bearings happen to fall
where nothing was sampled, not because they are strong there.

The geometric fix. Supports capped at F span the zonotope
Z = sum_i [0, F] u_i, and the force available along `v` is the radial extent
rho(v) = F / max_i lam_i for v = sum_i lam_i u_i. Asking for rho >= F everywhere
in the patch the supports span -- cone(u_1..u_m) on the sphere -- forces a
condition on the BOUNDARY of that patch:

    every pair of ADJACENT chosen directions is at most 90 degrees apart.

Adjacent means the pair spans a face of the cone: every other chosen direction
lies on one side of the plane the pair spans with the origin, which is the sign of
(u_i x u_j) . u_k over all other k. That pair is the right one because on the arc
between two rays bounding the patch every other ray sits on one side, so their
out-of-plane parts share a sign, and non-negative coefficients force them all to
zero -- nobody else can push there, the arc is a two-ray problem, and two rays
hold F between them exactly to 90 degrees.

Note that it is a NECESSARY condition and not a sufficient one, which is why this
tool measures min rho over the patch afterwards rather than assuming it.

The floor is exempt. Its `up` is unbounded, so it is not one of the capped
generators and it is not part of the patch; the rule binds the supports only.

--- the two encodings -------------------------------------------------------

STRICT, an upper bound on the cost. Require every chosen pair to be at most 90
degrees, u_i . u_j >= 0, whether adjacent or not. In the covering MILP that is one
conflict row x_i + x_j <= 1 per obtuse pair, so it composes with what
`capped_a1f.search` already builds and the answer stays provably minimal. It is
stricter than the geometry needs, so what it costs bounds what the real condition
costs.

A1-f has 850-1950 candidates a pose, hence up to two million conflict rows, so
they are never written down. Instead the candidate list is PROPAGATED to a
fixpoint: a feasible strict set containing `i` lies inside N[i], the candidates
within 90 degrees of `i`, so if N[i] intersected with the survivors cannot even
meet the one-sided covering bound `max_j y.T_j <= sum_k max(0, y.u_k)` for some
`y` in the lower hemisphere, then `i` belongs to no feasible set and drops out.
Dropping shrinks everyone else's neighbourhood, so the rule is re-applied until
nothing moves. Every removal is sound, so an empty fixpoint is a PROOF of
infeasibility, not a failure to find something.

EXACT, the condition as stated. Adjacency depends on the chosen set, so it is not
a fixed pairwise constraint and it cannot be enumerated. It can still be
SEPARATED exactly. For an obtuse pair (i, j) with plane normal w = u_i x u_j, the
pair is non-adjacent exactly when some chosen k has w.u_k > 0 and some other has
w.u_k < 0, which is two linear rows

    sum_{k: w.u_k > 0} x_k >= x_i + x_j - 1        and the same for w.u_k < 0

that are valid for EVERY feasible set, not just for the one that produced them.
So they can be added lazily to the same covering MILP without ever weakening the
bound: solve, look at the answer, add the rows for whichever obtuse pairs came
back adjacent, add `capped_a1f`'s violated facet normals if the answer also fails
the exact force test, and re-solve. When an answer comes back with no adjacent
obtuse pair and every requirement met, the MILP optimum is attained by a feasible
set and the count is PROVEN MINIMAL. If the budget runs out first the count is an
upper bound and the last MILP value is the lower bound, and both are reported as
such. The one convention baked in is the tolerance: `w.u_k` inside 1e-6 of zero
counts as ON the plane and does not break adjacency, which makes the rows very
slightly stronger than the bare geometry -- a support a millionth of a radian off
the plane is not something this model can tell from a support in it.

AND THE THING ITSELF, because the 90-degree rule turned out to buy almost nothing
and it is worth knowing what would. `rho >= F` over the whole sphere is not a
harder KIND of constraint than the ones already here -- it is the same one:

    rho >= F everywhere  <=>  B(0,F) inside Z  <=>  h_Z(y) >= F for every unit y
                         <=>  sum_i x_i * max(0, y.u_i) >= 1 for every unit y

covering rows on the full sphere with demand 1, alongside the requirement's rows
on the lower half. It is stronger than the patch-only question the 90-degree rule
comes from, so it upper-bounds that cost, and the two coincide when the chosen
cone is all of R3 -- which is what these designs come out as.

--- what is measured afterwards ---------------------------------------------

The constraint is a means, so the tool reports the end as well.

  * min rho over the patch, EXACTLY. rho(v) = min over facets y of Z of
    h(y)/(y.v), so its minimum over the cone on the sphere is
    min_y h(y) / max_{v in K, |v|=1} (y.v) = min_y h(y) / |P_K(y)|, with P_K the
    projection onto the cone, which is one non-negative least squares per facet.
    Checked against a 40962-point sphere sweep.
  * how much of the sphere the patch is, and how much of the REQUIREMENT lies
    inside it -- because a guarantee over the patch says nothing about a
    direction outside it.
  * the horizontal safety factor, the number the constraint is meant to buy,
    before and after.

--- reimplemented rather than imported --------------------------------------

`capped_a1f`'s per-pose pipeline -- work region, angled pushes, requirement,
candidate pushes -- lives inside its `main()` and is not importable, so `build()`
below is a line-for-line copy of it. It is checked rather than trusted: `--check`
compares n_targets, region_area_fraction and n_available against the recorded
`capped/capped_k1.json` on every pose and refuses to go on unless they match, and
the ten baselines are re-searched from scratch rather than read back.

    python slides/tools/adj_a1f.py --check
    python slides/tools/adj_a1f.py
    python slides/tools/adj_a1f.py --poses 1 8 9 --reuse-baseline
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time

import numpy as np
import trimesh
from scipy.optimize import Bounds, LinearConstraint, milp, nnls
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from capped_a1f import (CAP, answered, least_effort, lp_answered, normals, radial,
                        rows, search)
from common import obj_path, read_json, write_json
from cover import UP, tiling
from reach import compass
from shrink_support import angled_pushes
from supports import CONTACT_EPS, region_mask
from work_regions import gun_directions

OBJECT = "A1-f"
REGION = "spray"          # A1-f's region, as capped_a1f is run for it -- not `spot`
ACUTE = -1e-9             # u_i . u_j at or above this counts as within 90 degrees
SIGN = 1e-6               # |w.u_k| below this counts as ON the plane, so it cannot
                          # break adjacency; see the docstring
SWEEP = 6                 # icosphere subdivisions for the patch sweep, 40962 points


# ------------------------------------------------------------------ the model ---

def build(pose_T, mesh, tree, pose, seed=0, k=1.0, n_points=90, n_dirs=24):
    """capped_a1f.main's per-pose pipeline, copied because it is not importable.

    Same seeds, same order of draws. `--check` is what makes the copy worth
    anything; an earlier tool in this project replicated a pipeline with the wrong
    seed and nothing said so.
    """
    R, t = pose_T[:3, :3], pose_T[:3, 3]
    rng = np.random.default_rng(seed + 1000 * pose)
    n_passes = int(rng.integers(1, 4))
    inside = region_mask(mesh, pose_T, gun_directions(rng, n_passes),
                         mesh.triangles_center, mesh.face_normals)
    area = float(mesh.area_faces[inside].sum() / mesh.area)

    pw, pu = angled_pushes(mesh, pose_T, inside, n_points, n_dirs, seed + pose)
    targets = UP - k * pu
    keep = np.linalg.norm(targets, axis=1) > 1e-9
    targets, pu = targets[keep], pu[keep]

    on_floor = (mesh.triangles_center @ R.T + t)[:, 2] <= CONTACT_EPS
    off = np.flatnonzero(~inside & ~on_floor)
    push = -(mesh.face_normals[off] @ R.T)
    push /= np.linalg.norm(push, axis=1, keepdims=True)
    hit = tree.query(push)[1]
    first = {}
    for j, h in enumerate(hit):
        first.setdefault(h, j)
    rep = np.array([first[h] for h in sorted(first)])
    return dict(n_passes=n_passes, area=area, targets=targets, pu=pu,
                avail=push[rep], avail_face=off[rep], inside=inside)


# ----------------------------------------------------------- the 90-degree rule ---

def adjacent_obtuse(U: np.ndarray):
    """Chosen pairs that are both past 90 degrees AND span a face of the cone.

    The face test is the sign of (u_i x u_j).u_k over the others: all on one side
    means the pair bounds the patch, and then nothing else can push on the arc
    between them. An exactly antipodal pair has no plane -- `w` vanishes, both
    sides come out empty -- and falls out of this test as a violation, which is
    the right answer: two opposed rays bound nothing that holds F.
    """
    out = []
    for a, b in itertools.combinations(range(len(U)), 2):
        if U[a] @ U[b] >= ACUTE:
            continue
        w = np.cross(U[a], U[b])
        nw = float(np.linalg.norm(w))
        s = U @ (w / nw) if nw > 1e-9 else np.zeros(len(U))
        s[[a, b]] = 0.0
        if (s > SIGN).sum() == 0 or (s < -SIGN).sum() == 0:
            out.append((a, b))
    return out


def face_rows(avail, S, U):
    """For each adjacent obtuse chosen pair, the two rows that make it non-adjacent.

    Valid for every feasible set and not only for `S`, which is the whole point:
    they tighten the MILP permanently instead of merely cutting off one answer,
    so the objective stays a lower bound on the constrained optimum throughout.
    """
    out, lo = [], []
    for a, b in adjacent_obtuse(U):
        i, j = S[a], S[b]
        w = np.cross(U[a], U[b])
        nw = float(np.linalg.norm(w))
        s = avail @ (w / nw) if nw > 1e-9 else np.zeros(len(avail))
        for side in (1.0, -1.0):
            r = np.zeros(len(avail))
            r[side * s > SIGN] = 1.0
            r[i], r[j] = r[i] - 1.0, r[j] - 1.0     # k == i or k == j has s == 0 anyway
            out.append(r)
            lo.append(-1.0)
    return out, lo


def strict_survivors(Y, R, avail, verbose=True):
    """Candidates that could belong to SOME pairwise-acute answering set.

    A feasible strict set containing `i` lies inside N[i], so its capacity along
    any `y` is at most N[i]'s; if that already falls short of the demand, `i` is
    in nothing. Killing `i` shrinks every other neighbourhood, so the rule is
    re-applied to a fixpoint. Sound at every step, therefore an empty fixpoint is
    a proof that no pairwise-acute set answers the pose at all.
    """
    M = np.maximum(0.0, Y @ avail.T)                # capacity of each candidate along each y
    D = avail @ avail.T
    alive = np.ones(len(avail), bool)
    trail = [int(alive.sum())]
    for _ in range(40):
        N = (D >= ACUTE) & alive[None, :]
        cap = M @ N.T.astype(float)                 # cap[y, i] = what N[i] can raise along y
        keep = alive & (cap >= R[:, None] - 1e-9).all(axis=0)
        if keep.sum() == alive.sum():
            break
        alive = keep
        trail.append(int(alive.sum()))
        if not alive.any():
            break
    if verbose:
        print(f"          strict propagation: {' -> '.join(map(str, trail))}", flush=True)
    return alive, trail


def search_strict(targets, avail, tiles, cuts=14, secs=420.0):
    """Fewest supports with EVERY chosen pair inside 90 degrees, or a proof of none.

    Propagate first, because it is a fixpoint over necessary conditions and it is
    cheap; only what survives ever reaches the MILP, and only then are the
    conflict rows written down -- over the survivors there are few enough of them
    to state explicitly, which keeps the search exact.
    """
    note = []
    Y = tiles[tiles[:, 1] <= 1e-12]
    R = (Y @ targets.T).max(axis=1)
    Y, R = Y[R > 1e-9], R[R > 1e-9]
    t0 = time.time()
    for it in range(cuts):
        alive, trail = strict_survivors(Y, R, avail)
        if not alive.any():
            note.append(f"INFEASIBLE, proven: the pairwise-acute candidate set "
                        f"propagates to empty in {len(trail) - 1} rounds "
                        f"({' -> '.join(map(str, trail))}), so no set of supports "
                        f"with every pair inside 90 deg answers this pose, at any count")
            return None, None, False, note
        idx = np.flatnonzero(alive)
        sub = avail[idx]
        C, Rr = rows(Y, targets, sub)
        pair = [(a, b) for a, b in itertools.combinations(range(len(idx)), 2)
                if sub[a] @ sub[b] < ACUTE]
        cons = [LinearConstraint(C, lb=Rr, ub=np.inf)]
        if pair:
            data = np.ones(2 * len(pair))
            ij = np.array(pair)
            conf = csr_matrix((data, (np.repeat(np.arange(len(pair)), 2), ij.ravel())),
                              shape=(len(pair), len(idx)))
            cons.append(LinearConstraint(conf, lb=-np.inf, ub=1.0))
        res = milp(np.ones(len(idx)), constraints=cons, integrality=np.ones(len(idx)),
                   bounds=Bounds(0, 1), options=dict(time_limit=secs, mip_rel_gap=0.0))
        if res.status == 2:
            note.append(f"INFEASIBLE, proven: {len(idx)} candidates survive propagation "
                        f"but the covering MILP with their conflict rows has no solution")
            return None, None, False, note
        if res.status != 0 or res.x is None:
            note.append(f"the strict MILP stopped early ({res.message.strip()})")
            return None, None, False, note
        S = [int(idx[i]) for i in np.flatnonzero(res.x > 0.5)]
        bound = int(round(res.fun))
        G = np.column_stack([UP] + [avail[i] for i in S])
        if answered(G, targets).all():
            note.append(f"{bound} is optimal under the all-pairs rule: the covering "
                        f"bound forbids fewer and this set attains it "
                        f"({it + 1} MILP solves, {time.time() - t0:.0f}s)")
            return S, bound, True, note
        fn = normals(G)
        v = (fn @ targets.T).max(axis=1) - np.maximum(0.0, fn @ avail[S].T).sum(axis=1)
        add = np.argsort(-v)[:24]
        add = add[v[add] > 1e-9]
        if not len(add):
            note.append("no violated facet normal to add; the sampled bound has stalled")
            return S, bound, False, note
        Y = np.vstack([Y, fn[add]])
        R = np.r_[R, (fn[add] @ targets.T).max(axis=1)]
    note.append(f"cut budget exhausted; {len(S)} supports is an upper bound, {bound} a "
                f"lower bound")
    return S, bound, False, note


def search_adjacent(targets, avail, tiles, cuts=60, secs=300.0, budget=1800.0):
    """Fewest supports with no ADJACENT chosen pair past 90 degrees.

    Lazy separation rather than enumeration, on rows that are globally valid, so
    the MILP value is a lower bound on the constrained optimum at every iteration
    and equality with a clean answer is a proof of minimality.
    """
    note = []
    Y = tiles[tiles[:, 1] <= 1e-12]
    Y = Y[(Y @ targets.T).max(axis=1) > 1e-9]
    n, t0 = len(avail), time.time()
    extra, elo, S, bound = [], [], None, 0
    for it in range(cuts):
        C, R = rows(Y, targets, avail)
        cons = [LinearConstraint(C, lb=R, ub=np.inf)]
        if extra:
            cons.append(LinearConstraint(csr_matrix(np.array(extra)),
                                         lb=np.array(elo), ub=np.inf))
        res = milp(np.ones(n), constraints=cons, integrality=np.ones(n),
                   bounds=Bounds(0, 1), options=dict(time_limit=secs, mip_rel_gap=0.0))
        if res.status == 2:
            note.append("INFEASIBLE, proven: the covering MILP with the accumulated "
                        "face rows has no solution")
            return None, None, False, note
        if res.status != 0 or res.x is None:
            note.append(f"the MILP stopped early ({res.message.strip()}); the count "
                        f"below is an upper bound only")
            break
        bound = int(round(res.fun))
        S = [int(i) for i in np.flatnonzero(res.x > 0.5)]
        U = avail[S]
        bad = adjacent_obtuse(U)
        G = np.column_stack([UP] + [avail[i] for i in S])
        ok = answered(G, targets)
        if not bad and ok.all():
            note.append(f"{bound} is optimal under the adjacency rule: every row is a "
                        f"necessary condition, so the MILP forbids fewer, and this set "
                        f"attains it with no adjacent pair past 90 deg "
                        f"({it + 1} MILP solves, {len(Y)} covering rows, {len(extra)} "
                        f"face rows, {time.time() - t0:.0f}s)")
            return S, bound, True, note
        if bad:
            r, lo = face_rows(avail, S, U)
            extra += r
            elo += lo
        if not ok.all():
            fn = normals(G)
            v = (fn @ targets.T).max(axis=1) - np.maximum(0.0, fn @ U.T).sum(axis=1)
            add = np.argsort(-v)[:24]
            add = add[v[add] > 1e-9]
            if len(add):
                Y = np.vstack([Y, fn[add]])
        if time.time() - t0 > budget:
            note.append(f"time budget spent after {it + 1} solves")
            break
    note.append(f"NOT proven: {len(S) if S else 0} supports is only an upper bound and "
                f"it may not even satisfy the rule; {bound} is a valid lower bound")
    return S, bound, False, note


def search_guarantee(targets, avail, tiles, cuts=30, secs=900.0, budget=1200.0):
    """Fewest supports that actually deliver rho >= F, not just the boundary condition.

    Worth asking because the 90-degree rule is only necessary. The condition it
    stands in for -- rho >= F over the whole sphere -- turns out to be LINEAR in the
    same variables and in the same form as the covering rows already here:

        rho >= F everywhere  <=>  F*S2 inside Z  <=>  B(0,F) inside Z
                             <=>  h_Z(y) >= F for every unit y
                             <=>  sum_i x_i * max(0, y.u_i) >= 1 for every unit y

    the last step because the support function of a zonotope is the one-sided sum
    of dot products, and a convex body contains a ball exactly when its support
    function dominates the ball's. So the ball rows are covering rows over the FULL
    sphere with demand 1, next to the requirement's covering rows over the lower
    half with demand max_j y.T_j, and one MILP takes both.

    Separation is exact rather than sampled: min over unit y of h_Z(y) is the
    inradius of Z about the origin, and a polytope's inradius is the least distance
    from the origin to a FACET plane, so checking the zonotope's own facet normals
    settles it and any that fall short go back in as rows.

    This is stronger than the patch-only question the 90-degree rule comes from --
    it asks for F in directions the supports may not even span -- so its count is
    an upper bound on what the patch-only guarantee costs. The two coincide exactly
    when the chosen cone is all of R3, which is what these designs come out as.
    """
    note = []
    Y = tiles[tiles[:, 1] <= 1e-12]
    Y = Y[(Y @ targets.T).max(axis=1) > 1e-9]
    # the ball rows live on both halves of the sphere, and they start DENSE even
    # though the separation below is exact and a coarse start would still be
    # correct. Measured on pose 1: the whole 2562-point sample proves the optimum
    # in two solves and 558s, a 162-point start leaves the relaxation so loose that
    # a single solve runs past 300s without closing the gap. The rows are cheap;
    # the branching they save is not
    B = tiles.copy()
    n, t0, S, bound = len(avail), time.time(), None, 0
    for it in range(cuts):
        C, R = rows(Y, targets, avail)
        Call = np.vstack([C, np.maximum(0.0, B @ avail.T)])
        Rall = np.r_[R, np.full(len(B), CAP)]
        res = milp(np.ones(n), constraints=[LinearConstraint(Call, lb=Rall, ub=np.inf)],
                   integrality=np.ones(n), bounds=Bounds(0, 1),
                   options=dict(time_limit=secs, mip_rel_gap=0.0))
        if res.status == 2:
            note.append("INFEASIBLE, proven: no subset of the candidates can hold F in "
                        "every direction at all")
            return None, None, False, note
        if res.status != 0 or res.x is None:
            note.append(f"the MILP stopped early ({res.message.strip()})")
            break
        bound = int(round(res.fun))
        S = [int(i) for i in np.flatnonzero(res.x > 0.5)]
        U = avail[S]
        G = np.column_stack([UP] + list(U))
        ok = answered(G, targets)
        fz = zono_normals(U)
        h = CAP * np.maximum(0.0, fz @ U.T).sum(axis=1)
        thin = fz[h < CAP - 1e-9]
        if ok.all() and not len(thin):
            note.append(f"{bound} is optimal for rho >= F everywhere: every row is a "
                        f"necessary condition and this set attains the bound "
                        f"({it + 1} MILP solves, {time.time() - t0:.0f}s)")
            return S, bound, True, note
        if len(thin):
            B = np.vstack([B, thin[np.argsort(h[h < CAP - 1e-9])[:24]]])
        if not ok.all():
            fn = normals(G)
            v = (fn @ targets.T).max(axis=1) - np.maximum(0.0, fn @ U.T).sum(axis=1)
            add = np.argsort(-v)[:24]
            add = add[v[add] > 1e-9]
            if len(add):
                Y = np.vstack([Y, fn[add]])
        if time.time() - t0 > budget:
            note.append(f"time budget spent after {it + 1} solves")
            break
    note.append(f"NOT proven: {len(S) if S else 0} supports is an upper bound, {bound} "
                f"a lower bound")
    return S, bound, False, note


# ------------------------------------------------- what the design now guarantees ---

def zono_normals(U: np.ndarray) -> np.ndarray:
    """Facet normals of the supports' zonotope, both signs, floor excluded.

    `capped_a1f.normals` cannot be reused: it keeps only the half `y.up <= 0`,
    which is right when the unbounded floor is one of the generators and wrong
    here, where the question is about the capped supports on their own.
    """
    m = len(U)
    if m < 2:
        return np.zeros((0, 3))
    idx = np.array(list(itertools.combinations(range(m), 2)))
    c = np.cross(U[idx[:, 0]], U[idx[:, 1]])
    nc = np.linalg.norm(c, axis=1)
    c = c[nc > 1e-10] / nc[nc > 1e-10, None]
    # a flat set of generators collapses every cross product onto one plane normal
    # and the zonotope looks unbounded inside its own plane; the in-plane edge
    # normals put the sides back
    if len(c) and np.linalg.matrix_rank(U.T, tol=1e-9) < 3:
        extra = np.cross(np.repeat(c, m, axis=0), np.tile(U, (len(c), 1)))
        ne = np.linalg.norm(extra, axis=1)
        c = np.vstack([c, extra[ne > 1e-10] / ne[ne > 1e-10, None]])
    return np.vstack([c, -c])


def zono_radial(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """rho(v): the largest force the capped supports alone can put along each `v`."""
    Y = zono_normals(U)
    if not len(Y):
        return np.zeros(len(V))
    h = CAP * np.maximum(0.0, Y @ U.T).sum(axis=1)
    A = Y @ V.T
    return np.where(A > 1e-12, h[:, None] / np.maximum(A, 1e-300), np.inf).min(axis=0)


def patch_min_rho(U: np.ndarray):
    """min of rho over the patch, exactly, and the direction that attains it.

    rho is a min of h(y)/(y.v) over facets, so its minimum over the cone on the
    sphere swaps the two: min_y h(y) / max_{v in K, |v| = 1} (y.v), and that inner
    maximum is |P_K(y)|, the norm of the projection of `y` onto the cone -- one
    non-negative least squares in three rows. Facets whose projection vanishes
    point away from the whole cone and constrain nothing there.
    """
    Y = zono_normals(U)
    if not len(Y):
        return float("inf"), None
    h = CAP * np.maximum(0.0, Y @ U.T).sum(axis=1)
    best, arg = np.inf, None
    for y, hy in zip(Y, h):
        lam = nnls(U.T, y)[0]
        p = U.T @ lam
        nrm = float(np.linalg.norm(p))
        if nrm <= 1e-9:
            continue
        if hy / nrm < best:
            best, arg = hy / nrm, p / nrm
    return float(best), arg


def horizontal_factor(U: np.ndarray, targets: np.ndarray, n_compass=7200):
    """Sideways capacity over sideways demand, at the worst compass bearing.

    `up` carries no compass bearing and the floor's projection is zero, so this is
    the one part of the requirement only the supports can meet -- which is why it,
    and not the reach percentage, is where these designs are actually thin.
    """
    th = np.linspace(0.0, 2 * np.pi, n_compass, endpoint=False)
    N = np.stack([np.cos(th), np.sin(th)], 1)
    dem = (targets[:, :2] @ N.T).max(axis=0)
    cap = np.maximum(0.0, U[:, :2] @ N.T).sum(axis=0)
    m = dem > 1e-9
    return float((cap[m] / dem[m]).min())


def pairs_str(m):
    """the pair-angle spread, or a dash when there is only one support to pair."""
    if m["pair_angle_min_deg"] is None:
        return "-"
    return f"{m['pair_angle_min_deg']:.0f}-{m['pair_angle_max_deg']:.0f} deg"


def measure(U, targets, grid):
    """Everything the constraint was supposed to buy, on one design."""
    mag = np.linalg.norm(targets, axis=1)
    unit = targets / mag[:, None]
    G = np.column_stack([UP] + list(U))
    rho_grid = zono_radial(U, grid)
    inside = rho_grid > 1e-9
    mn, arg = patch_min_rho(U)
    D = np.clip(U @ U.T, -1.0, 1.0)
    iu = np.triu_indices(len(U), 1)
    ang = np.degrees(np.arccos(D[iu])) if len(U) > 1 else np.zeros(0)
    return dict(
        n_supports=len(U),
        answered_fraction=float(lp_answered(G, targets).mean()),
        min_margin_rho_over_T=float(np.min(np.where(
            mag > 0, radial(G, unit) / mag, np.inf))),
        horizontal_safety_factor=horizontal_factor(U, targets),
        pair_angle_min_deg=float(ang.min()) if len(ang) else None,
        pair_angle_max_deg=float(ang.max()) if len(ang) else None,
        n_obtuse_pairs=int((D[iu] < ACUTE).sum()) if len(ang) else 0,
        n_adjacent_obtuse_pairs=len(adjacent_obtuse(U)),
        patch_fraction_of_sphere=float(inside.mean()),
        patch_min_rho=mn,
        patch_min_rho_grid=float(rho_grid[inside].min()) if inside.any() else None,
        patch_min_rho_at=[float(x) for x in arg] if arg is not None else None,
        guarantee_holds=bool(mn >= CAP - 1e-9),
        requirement_inside_patch_fraction=float((zono_radial(U, unit) > 1e-9).mean()),
    )


# ------------------------------------------------------------------------ main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuts", type=int, default=60)
    ap.add_argument("--milp-seconds", type=float, default=300.0)
    ap.add_argument("--adj-budget", type=float, default=1800.0)
    # the rho >= F MILP carries 2562 more dense rows than the others and needs the
    # room: pose 1 closes in two solves of roughly 280s each
    ap.add_argument("--guarantee-seconds", type=float, default=900.0)
    ap.add_argument("--guarantee-budget", type=float, default=1200.0)
    ap.add_argument("--check", action="store_true",
                    help="only replicate the pipeline and compare with capped_k1.json")
    ap.add_argument("--reuse-baseline", action="store_true",
                    help="take the unconstrained design from capped_k1.json instead of "
                         "re-searching it")
    args = ap.parse_args()

    d = obj_path(OBJECT)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"]
    poses = args.poses if args.poses is not None else list(range(len(examples)))
    _, tiles = tiling(4)
    tree = cKDTree(tiles)
    rec = {r["pose"]: r for r in
           read_json(d / "capped" / f"capped_k{args.k:g}.json")["poses"]}
    sphere = trimesh.creation.icosphere(subdivisions=SWEEP)
    grid = np.asarray(sphere.vertices)
    grid = grid / np.linalg.norm(grid, axis=1, keepdims=True)

    # ---- the replication gate. an earlier tool in this project silently rebuilt a
    # pipeline with the wrong seed, so nothing downstream runs until the rebuilt
    # requirement matches the recorded one on every pose asked for
    bad = []
    built = {}
    for pose in poses:
        b = build(np.asarray(examples[pose]["T_world_mesh"]), mesh, tree, pose,
                  args.seed, args.k)
        built[pose] = b
        r = rec[pose]
        same = (len(b["targets"]) == r["n_targets"]
                and abs(b["area"] - r["region_area_fraction"]) < 1e-12
                and len(b["avail"]) == r["n_available"])
        print(f"pose {pose}: n_targets {len(b['targets'])} vs {r['n_targets']}, "
              f"region_area_fraction {b['area']:.9f} vs {r['region_area_fraction']:.9f}, "
              f"n_available {len(b['avail'])} vs {r['n_available']}   "
              f"{'MATCH' if same else 'MISMATCH'}", flush=True)
        if not same:
            bad.append(pose)
    if bad:
        sys.exit(f"pipeline replication failed on poses {bad}; nothing else is trustworthy")
    print("replication: all poses match the recorded capped_k1.json\n", flush=True)
    if args.check:
        return

    out_dir = d / "adjacent"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pose in poses:
        b = built[pose]
        targets, avail = b["targets"], b["avail"]
        print(f"pose {pose}: {len(avail)} candidate pushes, {len(targets)} requirements, "
              f"{compass(targets):.1f} deg of compass", flush=True)

        if args.reuse_baseline:
            base = [int(np.argmax(avail @ np.asarray(s["push"])))
                    for s in rec[pose]["supports"]]
            base_bound, base_proven = rec[pose]["lower_bound"], rec[pose]["proven_minimal"]
            base_note = ["taken from capped_k1.json"]
        else:
            t0 = time.time()
            base, base_bound, base_proven, base_note = search(
                targets, avail, tiles, cuts=14, milp_seconds=args.milp_seconds)
            base_note = list(base_note) + [f"{time.time() - t0:.0f}s"]
        m_base = measure(avail[base], targets, grid)
        agree = (m_base["n_supports"] == rec[pose]["n_supports"]
                 and base_bound == rec[pose]["lower_bound"]
                 and base_proven == rec[pose]["proven_minimal"])
        print(f"   baseline  : {len(base)} supports (bound {base_bound}"
              f"{', PROVEN' if base_proven else ''}), "
              f"{100 * m_base['answered_fraction']:.1f}% answered, "
              f"pairs {pairs_str(m_base)}, "
              f"adjacent-obtuse {m_base['n_adjacent_obtuse_pairs']}, "
              f"HSF {m_base['horizontal_safety_factor']:.4f}, "
              f"patch {100 * m_base['patch_fraction_of_sphere']:.1f}% of sphere, "
              f"min rho {m_base['patch_min_rho']:.4f}   "
              f"[{'matches' if agree else 'DIFFERS FROM'} capped_k1.json]", flush=True)

        strict, s_bound, s_proven, s_note = search_strict(
            targets, avail, tiles, cuts=args.cuts, secs=args.milp_seconds)
        m_strict = measure(avail[strict], targets, grid) if strict else None
        if strict is None:
            print(f"   strict    : {s_note[-1]}", flush=True)
        else:
            print(f"   strict    : {len(strict)} supports (bound {s_bound}"
                  f"{', PROVEN' if s_proven else ''}), "
                  f"{100 * m_strict['answered_fraction']:.1f}% answered, "
                  f"HSF {m_strict['horizontal_safety_factor']:.4f}, "
                  f"min rho {m_strict['patch_min_rho']:.4f}", flush=True)

        adj, a_bound, a_proven, a_note = search_adjacent(
            targets, avail, tiles, cuts=args.cuts, secs=args.milp_seconds,
            budget=args.adj_budget)
        m_adj = measure(avail[adj], targets, grid) if adj else None
        if adj is None:
            print(f"   adjacency : {a_note[-1]}", flush=True)
        else:
            print(f"   adjacency : {len(adj)} supports (bound {a_bound}"
                  f"{', PROVEN' if a_proven else ''}), "
                  f"{100 * m_adj['answered_fraction']:.1f}% answered, "
                  f"pairs {pairs_str(m_adj)}, "
                  f"adjacent-obtuse {m_adj['n_adjacent_obtuse_pairs']}, "
                  f"HSF {m_adj['horizontal_safety_factor']:.4f}, "
                  f"patch {100 * m_adj['patch_fraction_of_sphere']:.1f}% of sphere, "
                  f"min rho {m_adj['patch_min_rho']:.4f}, "
                  f"requirement inside patch "
                  f"{100 * m_adj['requirement_inside_patch_fraction']:.1f}%", flush=True)
        for n in a_note:
            print(f"               {n}", flush=True)

        gua, g_bound, g_proven, g_note = search_guarantee(
            targets, avail, tiles, cuts=args.cuts, secs=args.guarantee_seconds,
            budget=args.guarantee_budget)
        m_gua = measure(avail[gua], targets, grid) if gua else None
        if gua is None:
            print(f"   rho>=F    : {g_note[-1]}", flush=True)
        else:
            print(f"   rho>=F    : {len(gua)} supports (bound {g_bound}"
                  f"{', PROVEN' if g_proven else ''}), "
                  f"{100 * m_gua['answered_fraction']:.1f}% answered, "
                  f"pairs {pairs_str(m_gua)}, "
                  f"HSF {m_gua['horizontal_safety_factor']:.4f}, "
                  f"patch {100 * m_gua['patch_fraction_of_sphere']:.1f}% of sphere, "
                  f"min rho {m_gua['patch_min_rho']:.4f}, "
                  f"guarantee holds {m_gua['guarantee_holds']}", flush=True)
        for n in g_note:
            print(f"               {n}", flush=True)

        # the peak force is quoted over the requirements the design actually answers;
        # an unanswered one has no solution and would come back nan
        lam = np.zeros((0, 0))
        if adj:
            Ga = np.column_stack([UP] + list(avail[adj]))
            lam = least_effort(Ga, targets[lp_answered(Ga, targets)])
        records.append({
            "pose": pose,
            "n_passes": int(b["n_passes"]),
            "region_area_fraction": b["area"],
            "n_targets": int(len(targets)),
            "n_available": int(len(avail)),
            "compass_span_deg": compass(targets),
            "recorded": {kk: rec[pose][kk] for kk in
                         ("n_supports", "lower_bound", "proven_minimal",
                          "answered_fraction", "min_margin_rho_over_T")},
            "baseline": dict(m_base, lower_bound=int(base_bound),
                             proven_minimal=bool(base_proven), notes=list(base_note),
                             pushes=[[float(x) for x in avail[i]] for i in base]),
            "strict": (None if strict is None else
                       dict(m_strict, lower_bound=int(s_bound),
                            proven_minimal=bool(s_proven), notes=list(s_note),
                            pushes=[[float(x) for x in avail[i]] for i in strict])),
            "strict_infeasible": strict is None,
            "strict_notes": list(s_note),
            "adjacency": (None if adj is None else
                          dict(m_adj, lower_bound=int(a_bound),
                               proven_minimal=bool(a_proven), notes=list(a_note),
                               peak_force=([float(v) for v in np.nanmax(lam, axis=0)]
                                           if len(lam) else None),
                               pushes=[[float(x) for x in avail[i]] for i in adj])),
            "guarantee": (None if gua is None else
                          dict(m_gua, lower_bound=int(g_bound),
                               proven_minimal=bool(g_proven), notes=list(g_note),
                               pushes=[[float(x) for x in avail[i]] for i in gua])),
        })

    path = out_dir / f"adjacent_k{args.k:g}.json"
    if args.poses is not None and path.exists():
        old = [r for r in read_json(path)["poses"]
               if r["pose"] not in {r2["pose"] for r2 in records}]
        records = sorted(records + old, key=lambda r: r["pose"])
    write_json(path, {"object": OBJECT, "k": args.k, "cap": CAP, "region": REGION,
                      "poses": records})
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
