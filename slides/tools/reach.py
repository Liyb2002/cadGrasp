"""What the supports can synthesise, drawn against what they are asked for.

Everything here lives on one sphere, in force space. A contact contributes only
the direction it pushes; the floor contributes straight up for free; and for a
disturbance `d` of one body weight the contacts have to produce

    T = up - K*d

so the question is never "how many targets did we hit" but the plainer one:

    does the REACH -- every direction the chosen pushes can be mixed into --
    contain the whole REQUIREMENT?

Two things this fixes about the older `cover.py` figures.

The second step was described wrongly, not drawn wrongly. `T = (-d) + up`
decomposes exactly and both halves are real: the reversal is exact and
full-strength in the horizontal, so a push slanting downward still has to be met
sideways. What was false was calling the second step "gravity removes the
southern hemisphere". Gravity removes nothing. It adds one weight upward to every
vector, and since `T.up = 1 - d_z` is never negative for a unit `d`, the south
was never occupied in the first place -- the shift is what carries the set north.

And the reach is not the generators. A handful of isolated push directions look
hopeless until you notice that any non-negative mix of them is available too --
two directions buy the arc between them, three buy a patch. Drawing the
generators and leaving the reader to imagine their mixtures is what made the
orange ball unreadable.

The reach needs no hull algorithm and no case analysis. `covered` already
answers "can this direction be mixed out of these generators"; running it over
the sphere's own tiles IS the region. A pairwise cross-product construction was
tried first and gets degenerate cones wrong -- for {north pole, south pole, a
point on the equator} it returns a whole great circle where the truth is half of
one.

    python slides/tools/reach.py A1-f
    python slides/tools/reach.py cuboid_baseline --region spot
"""
from __future__ import annotations

import argparse
import itertools
import shutil

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from common import obj_path, read_json, write_json         # noqa: E402
from cover import (DONE, HAVE, NEED, PUSH_BLUE, UP, covered, globe_png,
                   neighbours, paint, sheet, shot, tiling)  # noqa: E402
from disturbances import _font                             # noqa: E402
from shrink_support import angled_pushes                   # noqa: E402
from supports import CONTACT_EPS, region_mask              # noqa: E402
from work_regions import BOUNDARY_EDGE, gun_directions, spot_region  # noqa: E402

REACH = "#7ec8bd"          # what the chosen pushes can be mixed into
FLOOR_MARK = "#3d3d3a"


def compass(targets):
    """How much of the compass the requirement occupies once `up` is projected out.

    A lower bound on the number of supports, exact and free. Writing a target as
    a non-negative mix sends `up` to zero under this projection, so the projected
    target must lie in the planar cone of the projected supports -- and two
    directions span at most a half plane. A requirement wider than 180 degrees
    therefore cannot be met by two supports, whatever they are, and no search is
    needed to know it.
    """
    p = targets - np.outer(targets @ UP, UP)
    p = p[np.linalg.norm(p, axis=1) > 1e-9]
    if len(p) < 2:
        return 0.0
    a = np.sort(np.arctan2(p[:, 1], p[:, 0]))
    return float(np.degrees(2 * np.pi - np.diff(np.r_[a, a[0] + 2 * np.pi]).max()))


def _batch(targets, gens, chunk=600):
    """Fraction of targets inside each cone, for many cones at once.

    With `up` plus two supports the cone is simplicial, so membership is one
    3x3 solve per cone rather than a walk over Caratheodory subsets, and the
    solves batch. That is what makes half a million pairs tractable.
    """
    out = np.zeros(len(gens))
    Tt = targets.T
    for s in range(0, len(gens), chunk):
        A = gens[s:s + chunk]
        ok = np.abs(np.linalg.det(A)) > 1e-9
        f = np.zeros(len(A))
        if ok.any():
            lam = np.linalg.solve(A[ok], np.broadcast_to(Tt, (int(ok.sum()), 3, len(targets))))
            f[ok] = (lam >= -1e-7).all(axis=1).mean(axis=1)
        out[s:s + chunk] = f
    return out


def search(targets, avail, tree, beam=240):
    """The fewest pushes whose reach contains the whole requirement.

    The compass bound settles the lower end without searching, and on every pose
    measured so far it says three. That makes the search one-sided: any triple
    that works is provably minimal, and only failing to FIND one is a risk. So
    pairs are enumerated exactly, and triples are grown from the best pairs --
    a beam, and it is reported as one.

    Greedy one-at-a-time, which `cover.py` uses, cannot start at all here: `up`
    with a single support spans a plane, a plane has no volume, and the gain is
    exactly zero for every candidate. It stalls on step one and reports nothing.
    """
    note = []
    if covered(UP.reshape(3, 1), targets).all():
        return [], 0.0, ["the floor alone answers everything"]
    low = 2 if compass(targets) <= 180.0 else 3
    if low == 3:
        note.append("the compass bound rules out 1 and 2 supports without searching")

    pair_idx = np.array(list(itertools.combinations(range(len(avail)), 2)))
    G = np.stack([np.broadcast_to(UP, (len(pair_idx), 3)),
                  avail[pair_idx[:, 0]], avail[pair_idx[:, 1]]], axis=2)
    got = _batch(targets, G)
    if low == 2 and got.max() >= 1.0:
        return list(pair_idx[int(np.argmax(got))]), 1.0, note

    seed = pair_idx[np.argsort(-got)[:beam]]
    note.append(f"triples grown from the best {len(seed)} of {len(pair_idx)} pairs (a beam)")
    best, best_f = None, -1.0
    for p in seed:
        rest = np.setdiff1d(np.arange(len(avail)), p)
        f = np.array([covered(np.column_stack([UP, avail[p[0]], avail[p[1]], avail[k]]),
                              targets).mean() for k in rest])
        j = int(np.argmax(f))
        if f[j] > best_f:
            best, best_f = [p[0], p[1], rest[j]], float(f[j])
        if best_f >= 1.0:
            break
    if best_f >= 1.0:
        note.append("three is optimal: the compass bound forbids fewer")
        return best, best_f, note

    # nothing at three -- keep adding, still never one at a time from scratch
    chosen = list(best)
    while best_f < 1.0 and len(chosen) < 8:
        rest = np.setdiff1d(np.arange(len(avail)), chosen)
        f = np.array([covered(np.column_stack([UP] + [avail[c] for c in chosen] + [avail[k]]),
                              targets).mean() for k in rest])
        j = int(np.argmax(f))
        if f[j] <= best_f + 1e-12:
            note.append(f"no further push adds anything; the ceiling is {100 * best_f:.1f}%")
            break
        chosen.append(int(rest[j]))
        best_f = float(f[j])
    return chosen, best_f, note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--poses", type=int, nargs="+", default=None)
    ap.add_argument("--region", choices=("spray", "spot"), default="spray")
    ap.add_argument("--k", type=float, default=1.0)
    # the render size, not the page size: the globe strip is drawn at 3.5 inches
    # and px/3.5 dots to the inch, so the finished page comes out 3.4x wider.
    # mujoco's offscreen buffer is 1400 wide, so this must stay under it.
    ap.add_argument("--size", type=int, default=840)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push-points", type=int, default=90)
    ap.add_argument("--push-dirs", type=int, default=24)
    args = ap.parse_args()

    d = obj_path(args.object)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"]
    poses = args.poses if args.poses is not None else range(len(examples))
    ico, tiles = tiling(4)
    tree = cKDTree(tiles)
    adjacency = neighbours(ico)
    px = args.size
    out_dir = d / "reach"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = d / "_reach_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    records = []
    for pose in poses:
        T = np.asarray(examples[pose]["T_world_mesh"])
        R, t = T[:3, :3], T[:3, 3]
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
        unit = targets / np.linalg.norm(targets, axis=1, keepdims=True)

        on_floor = (part.triangles_center @ R.T + t)[:, 1] <= CONTACT_EPS
        off = np.flatnonzero(~inside & ~on_floor)
        push = -(part.face_normals[off] @ R.T)
        push /= np.linalg.norm(push, axis=1, keepdims=True)
        hit = tree.query(push)[1]
        first = {}
        for k, h in enumerate(hit):
            first.setdefault(h, k)
        rep = np.array([first[h] for h in sorted(first)])
        avail, avail_face = push[rep], off[rep]

        chosen, frac, note = search(targets, avail, tree)
        span = compass(targets)
        pts = part.triangles_center[avail_face[chosen]] @ R.T + t
        us = avail[chosen]

        # ---- one rule for every ball: a force is drawn WHERE IT COMES FROM, at
        # -F, and its arrow runs from there into the centre. So a push from above
        # puts its patch on top with the arrow driving down, which is what the
        # blue ball already did; red and orange were mirrored against it and are
        # brought into line here. Nothing about the test changes, because
        # T in cone{u} exactly when -T in cone{-u}.
        h_job, h_back, h_want = tree.query(-pu)[1], tree.query(pu)[1], tree.query(-unit)[1]
        job = np.flatnonzero(sheet(adjacency, h_job))
        back = np.flatnonzero(sheet(adjacency, h_back))   # the reversal, now visible
        want = sheet(adjacency, h_want)
        near = cKDTree(-unit).query(tiles)[1]
        band = np.flatnonzero(sheet(adjacency, tree.query(-avail)[1]))
        few = np.random.default_rng(11).choice(len(pu), min(26, len(pu)), replace=False)
        fewt = np.random.default_rng(14).choice(len(unit), min(26, len(unit)), replace=False)

        # titles stay two short lines: a panel is three inches wide and a long one
        # runs into its neighbour's
        panels = [([(job, PUSH_BLUE, .95)], [], [(pu[few], PUSH_BLUE, False)], [],
                   "the job"),
                  # T = (-d) + up decomposes exactly, and both halves are real. The
                  # reversal is not cosmetic: T's horizontal part is MINUS d's, at
                  # full strength, so a push that slants downward still has to be
                  # met sideways -- the floor takes only the vertical share. What
                  # the older figure got wrong was calling the second step "gravity
                  # removes the southern hemisphere". Gravity removes nothing; it
                  # adds one weight upward to every vector, and that shift is what
                  # carries the whole set north.
                  ([(back, NEED, .95)], [], [(-pu[few], NEED, False)], [],
                   "reverse the pushes\nthe far side, sideways in full"),
                  ([(np.flatnonzero(want), NEED, .95)], [], [(unit[fewt], NEED, False)],
                   [(-UP, FLOOR_MARK, "G")],
                   f"+ its own weight\n{span:.0f}° wide, so 3 at least"),
                  ([(band, HAVE, .95)], [(-avail, HAVE, 26, True)], [(avail, HAVE, False)],
                   [(-UP, FLOOR_MARK, "G")],
                   f"what we have\n{len(avail)} pushes + the floor")]
        for i in range(len(chosen) + 1):
            G = np.column_stack([UP] + [avail[c] for c in chosen[:i]])
            can = covered(G, -tiles)                      # the reach, drawn where it acts from
            st = covered(G, targets)
            marks = [(-UP, FLOOR_MARK, "G")]
            marks += [(-avail[chosen[j]], "#d2450f", str(j + 1)) for j in range(i)]
            fills = [(np.flatnonzero(can), REACH, .95),
                     (np.flatnonzero(want & st[near]), DONE, .95),
                     (np.flatnonzero(want & ~st[near]), NEED, .95)]
            arr = [(unit[fewt][~st[fewt]], NEED, False)] if not st.all() else []
            panels.append((fills, [], arr, marks,
                           f"{'floor only' if i == 0 else f'+{i}'}\n"
                           f"reach {100 * can.mean():.0f}%  ·  {100 * st.mean():.0f}% done"))

        # ---- the page, laid out as cover.py's: the globe strip is rendered at
        # the full figure width, and the caption is sized against THAT, not
        # against the render size. Sizing it against `px` made it four times too
        # big for the page and it ran over the panel titles.
        parts = paint(part, T, inside, set(), tmp, f"rc_p{pose}", rel="_reach_tmp")
        blue = [(pw[j], pu[j]) for j in few]
        bare = shot(args.object, T, parts, px, [], [], triad=False)
        forces = shot(args.object, T, parts, px, blue, [], triad=False)
        shots = [bare, forces, forces, forces]
        shots += [shot(args.object, T, parts, px, blue,
                       [(pts[j], us[j]) for j in range(i)], triad=False)
                  for i in range(len(chosen) + 1)]

        per = (len(panels) + 1) // 2
        blocks = []
        for k, (ps, sh_) in enumerate([(panels[:per], shots[:per]),
                                       (panels[per:], shots[per:])]):
            g = globe_png(ico, ps, px)
            gw = g.size[0] // max(len(ps), 1)
            gh = int(g.size[1] * 0.99)
            blk = Image.new("RGB", (gw * len(ps), gh + gw), "white")
            blk.paste(g.crop((0, 0, g.size[0], gh)), (0, 0))
            for i, im in enumerate(sh_):
                blk.paste(im.resize((gw, gw)), (i * gw, gh))
            blocks.append(blk)
        caption = (
            f"{args.object}  pose {pose}   ·   BLUE = the disturbances   ·   "
            f"RED = still owed, GREY = answered\n"
            f"TEAL = the REACH, every direction the chosen pushes mix into   ·   "
            f"ORANGE = a push this workpiece can supply   ·   G = the floor")
        font = _font(int(px * 0.042))
        head = int(px * 0.16)
        wide = max(max(b.size[0] for b in blocks),
                   ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textbbox(
                       (16, 0), caption, font=font)[2] + 16)
        page = Image.new("RGB", (wide, head + sum(b.size[1] for b in blocks)), "white")
        y = head
        for b in blocks:
            page.paste(b, (0, y))
            y += b.size[1]
        ImageDraw.Draw(page).text((16, int(head * .22)), caption,
                                  fill=(90, 90, 90), font=font)
        out = out_dir / f"pose{pose}.png"
        page.save(out)

        records.append({"pose": pose, "n_passes": int(n_passes),
                        "region_area_fraction": area,
                        "n_targets": int(len(targets)),
                        "compass_span_deg": span,
                        "n_available": int(len(avail)),
                        "n_supports": len(chosen),
                        "answered_fraction": float(frac),
                        "notes": note,
                        "supports": [{"p": [float(x) for x in pts[i]],
                                      "push": [float(x) for x in us[i]]}
                                     for i in range(len(chosen))]})
        print(f"pose {pose:2d}: {len(avail):4d} pushes, requirement spans {span:5.1f}° "
              f"-> {len(chosen)} supports, {100 * frac:5.1f}% answered   {out}", flush=True)
        for n in note:
            print(f"          {n}", flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    write_json(out_dir / f"reach_k{args.k:g}.json",
               {"object": args.object, "k": args.k, "poses": records})


if __name__ == "__main__":
    main()
