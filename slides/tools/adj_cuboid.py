"""`capped_cuboid.py`'s search again, with a guarantee attached to the answer.

`capped_cuboid.py` asks only whether the sampled requirement can be produced.
That is feasibility and nothing more: it says nothing about the directions
BETWEEN the samples, and the measured margins on its designs sit at 1.00-1.08,
so the designs pass because the weak directions happen to fall where nothing was
sampled. A jig that is one badly-aimed disturbance from failing is not a design.

What a capped set of supports is actually worth along a direction is the radial
extent of the zonotope Z = sum_i [0, F] u_i,

    rho(v) = max{ r >= 0 : r v in Z }   and, writing v = sum_i lam_i u_i,
    rho(v) = F / max_i lam_i

for the representation that minimises that largest coefficient. Asking for
rho >= F everywhere on the patch cone(u_i) gives a boundary condition: on an arc
of the patch bounded by two rays with nothing between them, every other support
sits on one side of the plane those two span, so its coefficient would have to go
negative to help and is therefore zero. The arc is a two-support problem, and a
two-support problem holds F only out to 90 degrees (`min_force.py`). So

    every pair of ADJACENT chosen directions must be at most 90 degrees apart.

Adjacency is a property of the CHOSEN SET, not of the pair, which is what makes
this awkward: it is not a fixed pairwise constraint and cannot simply be handed
to a solver. Two encodings are run here and both are reported.

  * STRICT -- require every chosen pair inside 90 degrees, u_i . u_j >= 0. One
    conflict per negative-dot pair; a MILP would take it directly. It constrains
    non-adjacent pairs too, which the real rule does not, so what it returns is
    an UPPER BOUND on the true cost.
  * EXACT -- enumerate. Five or six candidates per pose is at most 64 subsets, so
    every subset is built, tested for adjacency, and the cheapest admissible one
    kept. No bound, no relaxation: the number is the true minimum.

The floor is not in any of this. It is unbounded, it is not a capped generator,
rho is not the right question for it, and the constraint is applied among the
chosen supports alone.

Two corrections to the rule as stated were forced by this object, and both are
recorded in `arcs()`. The candidate pushes on a box are the six face inward
normals -- plus and minus ONE orthonormal triad -- so the cone is never in
general position and its facets carry three rays, not two. The pair spanning
such a facet is antiparallel: it spans no plane at all, the cross-product sign
test degenerates to all-zeros, and reading that as "adjacent" would reject a set
whose rho never drops below F. The honest generalisation is that the constraint
belongs to CONSECUTIVE rays within a facet, and that an angular gap of 180
degrees or more is not part of the cone and constrains nothing. `guarantee()`
then checks the thing the rule is a proxy for -- min rho over the patch -- by
linear program, so the combinatorial answer is never trusted on its own.

On this object the answer is not sampled, it is provable, and the proof is worth
stating because it is the whole result. Every candidate here is one of the six
inward face normals of a box, so any chosen set S is a subset of plus and minus
ONE orthonormal triad. A unit direction v in cone(S) needs coefficient |v_i| on
the axis it leans along and nothing on that axis's opposite, so the smallest
largest coefficient is max_i |v_i| <= |v| = 1 and

    rho(v) = F / max_i |v_i| >= F        for every v in cone(S), every S,

with equality exactly at the rays. So EVERY subset of the candidates already
carries the guarantee, the constraint can cost nothing, and the 180 degree pairs
in the unconstrained designs were never a weakness -- they are opposite faces,
they are never adjacent, and each has a third face at 90 degrees sitting between
them. `guarantee()` measures this rather than assuming it, and agrees.

Nothing else moves. Same region, same seeds, same candidate directions, same
feasibility LP imported from `capped_cuboid`, and the unconstrained column is
recomputed here from the same subset enumeration rather than copied, so a
mismatch against `capped/capped_k1.json` would show up as a mismatch and not be
absorbed.

    python slides/tools/adj_cuboid.py
    python slides/tools/adj_cuboid.py --poses 2
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import trimesh
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from capped_cuboid import CAP, capacity, feasible          # noqa: E402
from common import obj_path, read_json, write_json         # noqa: E402
from cover import UP, covered, tiling                      # noqa: E402
from reach import compass                                  # noqa: E402
from shrink_support import angled_pushes                   # noqa: E402
from supports import CONTACT_EPS                           # noqa: E402
from work_regions import BOUNDARY_EDGE, spot_region        # noqa: E402

TOL = 1e-9
RIGHT = 1e-9               # slack on cos(90 deg) = 0; the dots here are 0 or +-1


# ------------------------------------------------------- the pipeline, reused ---

def stage(mesh, examples, pose, args):
    """reach.py's setup, seed for seed, so the only new thing is the constraint.

    Copied rather than imported because `capped_cuboid.main` interleaves it with
    rendering and does not hand the intermediate arrays back. Every seed here is
    the same expression it is there -- `seed + 1000*pose` for the region, `seed +
    pose` for the pushes -- and `n_targets` and `region_area_fraction` are checked
    against the recorded run before anything is built on top.
    """
    T = np.asarray(examples[pose]["T_world_mesh"])
    R, t = T[:3, :3], T[:3, 3]
    rng = np.random.default_rng(args.seed + 1000 * pose)
    reg = spot_region(mesh, T, rng, float(rng.uniform(0.17, 0.33)),
                      BOUNDARY_EDGE * float(np.linalg.norm(mesh.extents)))
    part, inside = reg.pop("mesh"), reg.pop("mask")
    area = float(part.area_faces[inside].sum() / part.area)

    pw, pu = angled_pushes(part, T, inside, args.push_points, args.push_dirs,
                           args.seed + pose)
    targets = UP - args.k * pu
    keep = np.linalg.norm(targets, axis=1) > 1e-9
    targets = targets[keep]

    on_floor = (part.triangles_center @ R.T + t)[:, 1] <= CONTACT_EPS
    off = np.flatnonzero(~inside & ~on_floor)
    push = -(part.face_normals[off] @ R.T)
    push /= np.linalg.norm(push, axis=1, keepdims=True)
    tree = cKDTree(tiling(4)[1])
    hit = tree.query(push)[1]
    first = {}
    for k, h in enumerate(hit):
        first.setdefault(h, k)
    rep = np.array([first[h] for h in sorted(first)])
    return dict(targets=targets, avail=push[rep], area=area,
                n_passes=int(reg["n_passes"]))


def all_masks(targets, avail):
    """Which targets EVERY subset answers, not just the ones a search walks into.

    `capped_cuboid.exhaustive` stops at the first size that reaches the ceiling,
    which is all it needs; here three different admissibility rules have to be
    scored against the same table, and the constrained rules cannot use the
    all-candidates set as their ceiling because that set is usually not
    admissible. So the whole lattice is built once. The monotone trick is the
    same one and does the same work: a target answered by S minus an element is
    answered by S, so it never gets a second solve.
    """
    n = len(avail)
    masks = {frozenset(): feasible([], targets)}
    for size in range(1, n + 1):
        for pick in itertools.combinations(range(n), size):
            known = np.zeros(len(targets), bool)
            for i in pick:
                known |= masks[frozenset(pick) - {i}]
            todo = np.flatnonzero(~known)
            got = known.copy()
            if len(todo):
                got[todo] = feasible([avail[i] for i in pick], targets[todo])
            masks[frozenset(pick)] = got
    return masks


def cheapest(masks, n, admissible):
    """Smallest admissible subset that answers as much as any admissible subset.

    Same contract and the same tie-break as `capped_cuboid.exhaustive` -- ties go
    to whichever subset sorts last, which is arbitrary and harmless because at
    the minimum size every subset reaching the ceiling is equally optimal. The
    ceiling is taken over the admissible subsets only, which is the whole reason
    this is not just a filter bolted onto the old search: under a constraint the
    best possible answer is itself a smaller number.
    """
    ok = [p for p in masks if admissible(sorted(p))]
    ceiling = max(float(masks[p].mean()) for p in ok)
    for size in range(n + 1):
        rows = [(float(masks[p].mean()), tuple(sorted(p))) for p in ok if len(p) == size]
        if not rows:
            continue
        best, pick = max(rows)
        if best >= ceiling - 1e-12:
            return list(pick), best, ceiling
    return [], 0.0, ceiling


# ------------------------------------------------------------ the constraint ---

def strict_ok(us):
    """Every chosen pair inside 90 degrees. Stricter than the rule, and easy."""
    return all(float(us[i] @ us[j]) >= -RIGHT
               for i, j in itertools.combinations(range(len(us)), 2))


def arcs(us):
    """The boundary arcs of the patch: pairs of rays with nothing between them.

    A pair bounds an arc of cone(us) when the plane it spans SUPPORTS the cone --
    every other ray on one side -- which is the sign of (u_i x u_j) . u_k over
    all other k. Two things the plain reading of that test gets wrong on this
    object, both because the candidates here are plus and minus one orthonormal
    triad and so the cone is never simplicial:

    A facet can hold more than two rays. The facet of cone(+x,-x,+y,+z) at y = 0
    holds +x, -x and +z, and the pair that spans its plane is +x,-x at 180
    degrees -- but +z sits between them, the arc is really two arcs of 90
    degrees, and rho never drops below F on it. So the rays IN the plane are
    collected and the constraint is put on CONSECUTIVE ones, not on the pair that
    happened to find the plane.

    An antiparallel pair spans no plane, so its cross product vanishes and the
    sign test returns all-zeros -- vacuously "adjacent", and vacuously 180
    degrees apart. Those pairs are skipped: cone(+x,-x) is a LINE, not the arc
    between the two rays, and rho on a line through a ray is F.

    Consecutive gaps of 180 degrees or more are dropped for the same reason: the
    wedge between two rays lies in the cone only while it is less than a half
    turn, and outside the cone there is nothing to guarantee.
    """
    n = len(us)
    out, seen = [], set()
    for i, j in itertools.combinations(range(n), 2):
        c = np.cross(us[i], us[j])
        m = float(np.linalg.norm(c))
        if m <= 1e-9:
            continue
        c = c / m
        s = us @ c
        if not ((s >= -TOL).all() or (s <= TOL).all()):
            continue
        face = tuple(np.flatnonzero(np.abs(s) <= TOL))
        if face in seen or len(face) < 2:
            continue
        seen.add(face)
        e1 = us[face[0]]
        e2 = np.cross(c, e1)
        v = us[list(face)]
        ang = np.arctan2(v @ e2, v @ e1)
        order = np.argsort(ang)
        a, idx = ang[order], np.asarray(face)[order]
        for k in range(len(a)):
            gap = (a[(k + 1) % len(a)] - a[k]) % (2 * np.pi)
            if gap < np.pi - 1e-9:
                out.append((int(idx[k]), int(idx[(k + 1) % len(a)]), float(gap)))
    return out


def adjacent_ok(us):
    """The rule itself: no arc of the patch is bounded by rays over 90 apart."""
    return all(g <= np.pi / 2 + 1e-9 for _, _, g in arcs(us))


# ------------------------------------------------------------- the guarantee ---

def rho(gens, dirs):
    """How hard the capped supports alone can push along each direction.

    `capped_cuboid.capacity` answers the same question with the floor bolted in
    as an unbounded generator, which is right for reading a figure against the
    requirement and wrong here: the patch and the 90 degree rule are about the
    CAPPED generators, and an unbounded one would swamp anything with lift in it.
    So it is reimplemented without the floor. Zero comes back for a direction
    outside the cone, which is also how membership is tested.
    """
    m = len(gens)
    if not m:
        return np.zeros(len(dirs))
    A = np.column_stack([np.zeros(3)] + [-g for g in gens])
    bounds = [(0, m * CAP + 1.0)] + [(0, CAP)] * m
    c = np.r_[-1.0, np.zeros(m)]
    out = np.zeros(len(dirs))
    for j, d in enumerate(dirs):
        A[:, 0] = d
        r = linprog(c, A_eq=A, b_eq=np.zeros(3), bounds=bounds, method="highs")
        if r.status == 0:
            out[j] = r.x[0]
    return out


def patch_dirs(gens, rng, n):
    """Directions inside the patch, by construction rather than by rejection.

    Sampling the sphere and keeping what lands in the cone dies on exactly the
    cases that matter: a two-ray patch is an arc and a coplanar set is a great
    circle, both of measure zero, and rejection would return nothing. Every point
    of the cone is a non-negative mix of the rays, so mixing the rays IS the
    sampler, and a small Dirichlet concentration pushes mass onto the edges and
    the rays where the minimum is expected to sit. The rays themselves and the
    directions perpendicular to each pair are added outright: those are where the
    argument says the weak spots are, and a sampler should not be asked to find a
    measure-zero point.
    """
    G = np.asarray(gens)
    v = [G]
    for i, j in itertools.combinations(range(len(G)), 2):
        c = np.cross(G[i], G[j])
        if np.linalg.norm(c) > 1e-9:
            v.append(np.stack([c, -c]) / np.linalg.norm(c))
    for a in (0.15, 0.5, 1.0, 4.0):
        w = rng.dirichlet(np.full(len(G), a), size=n)
        v.append(w @ G)
    d = np.concatenate(v)
    nrm = np.linalg.norm(d, axis=1)
    d = d[nrm > 1e-9]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def guarantee(gens, rng, n=500, rounds=3):
    """min rho over the patch: what the constrained design is actually worth.

    The combinatorial rule is only a NECESSARY condition, so it is not evidence.
    This is the quantity it is a proxy for, measured. Coarse mixes first, then
    the weight vector behind the worst one is jittered and resampled, because the
    minimum of rho sits at a ray or on an edge and a coarse mix lands near those
    without landing on them.
    """
    if not len(gens):
        return 0.0, np.zeros(3)
    G = np.asarray(gens)
    d = patch_dirs(G, rng, n)
    val = rho(G, d)
    # the perpendiculars patch_dirs throws in are candidates, not members: on an
    # orthogonal triad u_i x u_j is another axis, which may be outside the chosen
    # cone entirely. rho is strictly positive on every ray OF the cone, so zero
    # is exactly the "not in the patch" flag and those are dropped rather than
    # reported as a guarantee of nothing
    keep = val > 1e-7
    d, val = d[keep], val[keep]
    for _ in range(rounds):
        k = int(np.argmin(val))
        # jitter in MIXTURE space, not on the sphere: a perturbed mix is still
        # inside the cone, a perturbed direction need not be
        w0 = np.maximum(np.linalg.lstsq(G.T, d[k], rcond=None)[0], 0)
        if w0.sum() <= 1e-12:
            break
        w = np.maximum(w0 + rng.normal(0, 0.15 * (w0.max() + 1e-3), (n, len(G))), 0)
        nd = w @ G
        nrm = np.linalg.norm(nd, axis=1)
        nd = nd[nrm > 1e-9] / nrm[nrm > 1e-9, None]
        nv = rho(G, nd)
        d = np.concatenate([d, nd[nv > 1e-7]])
        val = np.concatenate([val, nv[nv > 1e-7]])
    k = int(np.argmin(val))
    return float(val[k]), d[k]


def horizontal(us, targets, n=1440):
    """The compass safety factor: how much sideways push there is per sideways ask.

    `up` is projected out of both sides because the floor supplies it without
    limit, so only the horizontal part of a requirement is ever a demand on the
    supports. The numerator is the most the supports can put along a compass
    bearing -- each capped at one weight, and one that would have to pull
    contributes nothing -- and the denominator is the most any requirement asks
    along it. Below 1 somewhere means some direction is genuinely short, whatever
    the sampled feasibility says.

    The bearing count belongs with the number, because the minimum sits in a
    narrow notch and a coarse compass steps over it: pose 6 reads 1.2993 at 72
    bearings, 1.2668 at 360, 1.2640 at 1440 and 1.2632 at 5760.
    """
    if not len(us):
        return 0.0
    p = np.asarray(us) - np.outer(np.asarray(us) @ UP, UP)
    Th = targets - np.outer(targets @ UP, UP)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    N = np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1)
    num = CAP * np.maximum(0.0, p @ N.T).sum(0)
    den = (Th @ N.T).max(0)
    live = den > 1e-9
    return float((num[live] / den[live]).min()) if live.any() else float("inf")


def margins(gens, targets):
    """How much harder than asked the design can push, in the asked direction.

    The number the researcher measured at 1.00-1.08 on the unconstrained designs.
    The floor IS included here -- this is about the requirement, which every
    target carries a lift component of -- so it is `capped_cuboid.capacity` on
    the target directions, read against the magnitude each target wants.
    """
    mag = np.linalg.norm(targets, axis=1)
    unit = targets / mag[:, None]
    return capacity(list(gens), unit) / mag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="cuboid_baseline")
    ap.add_argument("--poses", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push-points", type=int, default=90)
    ap.add_argument("--push-dirs", type=int, default=24)
    args = ap.parse_args()

    d = obj_path(args.object)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"]
    poses = args.poses if args.poses is not None else range(len(examples))
    # the recorded run is the control. an earlier error in this project came from
    # replicating a pipeline with the wrong seed and not noticing, so the two
    # numbers that would move if a seed moved are compared before anything else
    ref = {r["pose"]: r for r in
           read_json(d / "capped" / f"capped_k{args.k:g}.json")["poses"]}

    rules = [("none", lambda us: True), ("strict", strict_ok), ("exact", adjacent_ok)]
    records = []
    for pose in poses:
        t0 = time.time()
        st = stage(mesh, examples, pose, args)
        targets, avail = st["targets"], st["avail"]
        r = ref.get(pose)
        check = (r is not None and r["n_targets"] == len(targets)
                 and abs(r["region_area_fraction"] - st["area"]) < 1e-9
                 and r["n_available"] == len(avail))
        print(f"pose {pose:2d}: n_targets {len(targets)} (recorded {r['n_targets']}), "
              f"area {st['area']:.6f} (recorded {r['region_area_fraction']:.6f}), "
              f"{len(avail)} candidates (recorded {r['n_available']}) -> replication "
              f"{'MATCHES' if check else 'DIFFERS'}", flush=True)
        if not check:
            raise SystemExit("pipeline replication does not match the recorded run")

        masks = all_masks(targets, avail)
        rng = np.random.default_rng(101 + pose)
        rec = {"pose": pose, "n_targets": int(len(targets)),
               "region_area_fraction": st["area"], "n_passes": st["n_passes"],
               "n_available": int(len(avail)),
               "compass_span_deg": compass(targets),
               "max_required_force": float(np.linalg.norm(targets, axis=1).max()),
               "recorded_n_supports": r["n_supports"],
               "recorded_answered_fraction": r["answered_fraction"], "designs": {}}
        for name, rule in rules:
            chosen, frac, ceiling = cheapest(masks, len(avail),
                                             lambda p, rule=rule: rule(avail[list(p)]))
            us = avail[chosen]
            g, weak = guarantee(us, rng)
            mg = margins(us, targets)
            # the patch is the SUPPORTS' cone. whether the requirement even lands
            # in it is a separate question from whether the design answers the
            # requirement, and the guarantee only covers what lands in it.
            # membership is `covered`, not `rho`, because it is vectorised and a
            # per-direction LP over two thousand targets is not worth its cost
            inpatch = (covered(us.T, targets) if len(us)
                       else np.zeros(len(targets), bool))
            # the horizontal version is the one with a chance: the floor supplies
            # `up` without limit, so what the supports are really asked for is the
            # part of the requirement left after `up` is projected out
            ph = np.asarray(us) - np.outer(us @ UP, UP) if len(us) else np.zeros((0, 3))
            ph = ph[np.linalg.norm(ph, axis=1) > 1e-9]
            Th = targets - np.outer(targets @ UP, UP)
            live = np.linalg.norm(Th, axis=1) > 1e-9
            hpatch = (covered(ph.T, Th[live]) if len(ph)
                      else np.zeros(int(live.sum()), bool))
            rec["designs"][name] = {
                "n_supports": len(chosen),
                "answered_fraction": frac, "ceiling_fraction": ceiling,
                "min_rho_over_patch": g,
                "rho_at_least_cap": bool(g >= CAP - 1e-6),
                "weakest_direction": [float(x) for x in weak],
                "max_pair_angle_deg": float(np.degrees(np.arccos(np.clip(
                    min((us[i] @ us[j] for i, j in
                         itertools.combinations(range(len(us)), 2)), default=1.0),
                    -1, 1)))),
                "arc_angles_deg": sorted(round(float(np.degrees(a)), 3)
                                         for _, _, a in arcs(us)),
                "requirement_in_patch_fraction": float(inpatch.mean()),
                "horizontal_requirement_in_patch_fraction": float(hpatch.mean()),
                "margin_min": float(np.nanmin(mg)), "margin_max": float(np.nanmax(mg)),
                "margin_min_over_answered": (float(np.nanmin(mg[masks[frozenset(chosen)]]))
                                             if masks[frozenset(chosen)].any() else 0.0),
                "horizontal_safety_factor": horizontal(us, targets),
                "supports": [[float(x) for x in avail[i]] for i in chosen]}
            e = rec["designs"][name]
            print(f"   {name:6s}: {len(chosen)} supports, {100 * frac:5.1f}% answered "
                  f"(ceiling {100 * ceiling:5.1f}%)  min rho {g:.3f} W  "
                  f"widest arc {max([a for _, _, a in arcs(us)], default=0) * 57.2958:5.1f} deg  "
                  f"widest pair {e['max_pair_angle_deg']:5.1f} deg  "
                  f"HSF {e['horizontal_safety_factor']:.3f}  "
                  f"margin {e['margin_min']:.3f}  "
                  f"requirement in patch {100 * e['requirement_in_patch_fraction']:.1f}%",
                  flush=True)
        # ---- the rule is NECESSARY, not sufficient, so passing it is not the
        # same as carrying the guarantee. Every subset is priced against the
        # thing the rule stands in for -- min rho over its own patch -- and the
        # two verdicts are compared. A subset the rule rejects while rho holds is
        # a subset the search threw away for nothing.
        audit = {"n_subsets": 0, "strict_ok": 0, "adjacent_ok": 0, "rho_ok": 0,
                 "rule_rejects_but_rho_holds": [], "rule_accepts_but_rho_fails": []}
        rng2 = np.random.default_rng(9001 + pose)
        for p in sorted(masks, key=lambda s: (len(s), sorted(s))):
            us = avail[sorted(p)]
            a, s = adjacent_ok(us), strict_ok(us)
            g = guarantee(us, rng2, n=80, rounds=1)[0] if len(us) else 0.0
            r = bool(len(us)) and g >= CAP - 1e-6
            audit["n_subsets"] += 1
            audit["strict_ok"] += int(s and len(us) > 0)
            audit["adjacent_ok"] += int(a and len(us) > 0)
            audit["rho_ok"] += int(r)
            if len(us) and not a and r:
                audit["rule_rejects_but_rho_holds"].append([sorted(p), round(g, 4)])
            if len(us) and a and not r:
                audit["rule_accepts_but_rho_fails"].append([sorted(p), round(g, 4)])
        rec["audit"] = audit
        print(f"   audit : {audit['n_subsets']} subsets, strict admits "
              f"{audit['strict_ok']}, adjacency admits {audit['adjacent_ok']}, "
              f"rho>=F holds for {audit['rho_ok']}; "
              f"{len(audit['rule_rejects_but_rho_holds'])} rejected for nothing, "
              f"{len(audit['rule_accepts_but_rho_fails'])} admitted without the "
              f"guarantee", flush=True)
        records.append(rec)
        print(f"          {time.time() - t0:.0f}s", flush=True)

    out_dir = d / "adjacent"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"adj_k{args.k:g}.json"
    keep = {r["pose"]: r for r in (read_json(out_json)["poses"] if out_json.exists() else [])}
    keep.update({r["pose"]: r for r in records})
    write_json(out_json, {"object": args.object, "k": args.k, "cap_per_support": CAP,
                          "search": "full subset lattice; strict and exact adjacency",
                          "poses": [keep[p] for p in sorted(keep)]})


if __name__ == "__main__":
    main()
