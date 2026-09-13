"""The same support-design question as `reach.py`, but with contacts of finite strength.

`reach.py` asks whether the requirement

    T = up - K*d          (one body weight of disturbance, K = 1)

lies in the CONE the chosen pushes span. A cone is scale-free: a direction is
either available at every magnitude or at none, because the contact magnitudes
are unbounded. That throws away half of what a support has to be. A jig can be
pointed the right way and still be nowhere near strong enough.

So the magnitudes are bounded here, and the researcher's model is taken as given:

  * the process push is at most one body weight, K = 1, unchanged;
  * each support delivers at most ONE body weight;
  * the floor stays unbounded -- it is the ground, it carries whatever it is given;
  * each distinct push direction may be used AT MOST ONCE.

That last one is a modelling choice, not a physical law, and it is what makes
the cap mean anything. Two supports on the same face push the same way, so they
are one support of twice the strength; allow that and the cap can always be
bought off by doubling up, and the whole question collapses back to the cone.
One support per direction is the honest reading of "each support can deliver at
most 1 W", and the candidate list is already one direction per tile, so a subset
of it never repeats a direction.

The test therefore stops being cone membership and becomes linear feasibility:

    does there exist  lam0 >= 0  and  0 <= lam_i <= 1  with
    lam0*up + sum_i lam_i*u_i = T ?

which is `scipy.optimize.linprog` with a zero objective; `status == 0` is
feasible. Nothing else about the pipeline moves -- same region, same pushes,
same candidate directions -- so every difference from `reach.py` is the cap.

Only five or six candidate directions exist per pose on this part, so the search
is EXHAUSTIVE over all 2^n subsets, smallest first. There is no heuristic here
and no beam: the number reported is the true minimum, not an upper bound on it.

The drawing has to change with the model. `reach.py`'s teal panel is the REACH,
the set of directions the pushes mix into, and a set is all a cone can be. Under
a cap the same panel would be a lie by omission, so what is painted here is the
radial support function: for each direction on the ball, the LARGEST force the
chosen supports and the floor can actually deliver along it, banded at 0.5, 1,
1.5 and 2 body weights. Bare background still means "not available at all"; a
teal band means "available, and this hard".

That field alone would not settle the question a reader has to ask, because the
requirement carries a magnitude too and the eye cannot compare a shade with a
sheet. So the requirement is painted over it in three states, not two: GREY
answered, RED where the direction is not in the reach at all -- the only failure
`reach.py` could ever show -- and VIOLET where the direction IS in the reach and
the supports simply cannot push that hard. Violet is the whole of what this tool
adds; under a cone test it cannot exist. On pose 9 with four supports the reach
is 100% of the requirement and the answer is 43%, and the panel is mostly violet.

    python slides/tools/capped_cuboid.py
    python slides/tools/capped_cuboid.py --poses 2
"""
from __future__ import annotations

import argparse
import itertools
import shutil

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from common import obj_path, read_json, write_json         # noqa: E402
from cover import (AZIM, DONE, ELEV, HAVE, NEED, PUSH_BLUE, UP, covered,
                   globe_png, neighbours, paint, sheet, shot, tiling)  # noqa: E402
from disturbances import _font                             # noqa: E402
from reach import FLOOR_MARK, compass                      # noqa: E402
from shrink_support import angled_pushes                   # noqa: E402
from supports import CONTACT_EPS                           # noqa: E402
from work_regions import BOUNDARY_EDGE, spot_region        # noqa: E402

CAP = 1.0                  # what one support can deliver, in body weights
WEAK = "#8A3FA8"           # reachable, but the supports cannot push that hard
# how hard the chosen supports can push, banded. The bands are read against the
# requirement, whose magnitudes run from 0 to 2 W, so 1 W and 2 W are the two
# edges a reader actually needs.
LEVELS = (0.5, 1.0, 1.5, 2.0)
RAMP = ("#a9dcd1", "#79c7b8", "#4aae9c", "#22907d", "#0f6b5e")
# the requirement sits near the SOUTH of the ball, because it is drawn at -T and
# T points up; from this one camera that is the far side, seen through the
# sphere. An opaque field in front of it therefore erases it -- which is what
# happens in reach.py's cone panel, where the answered requirement is simply
# invisible. Half-transparent keeps both: the field reads as a haze whose bands
# are still ordered, and the requirement shows through it.
FIELD_ALPHA = 0.5
RMAX = 3.0                 # the ray up the floor gives is unbounded; clip it to draw it
MARK = "#7a2a06"
FOVY = 45.0                # SCENE names no fovy, so this is mujoco's default


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


def feasible(gens, targets):
    """Which targets the floor plus these capped supports can actually produce.

    The model test, and the only one that decides anything below. One LP per
    target because the constraint set changes with the target's magnitude --
    which is exactly the information a cone test does not have.
    """
    m = len(gens)
    A = np.column_stack([UP] + list(gens)) if m else UP.reshape(3, 1)
    bounds = [(0, None)] + [(0, CAP)] * m
    zero = np.zeros(m + 1)
    return np.array([linprog(zero, A_eq=A, b_eq=t, bounds=bounds,
                             method="highs").status == 0 for t in targets], bool)


def effort(gens, targets):
    """The least-effort decomposition, so a peak force can be quoted at all.

    The feasibility LP has a zero objective, so the multipliers it happens to
    return are whatever the solver's vertex was, and reporting those as "the
    force this contact carries" would be reporting an artefact. Minimising the
    total support force is a policy -- one of many -- but it is a stated one,
    and it is the one a designer would want: spend as little as the geometry
    allows. The floor is free and is left out of the objective.
    """
    m = len(gens)
    A = np.column_stack([UP] + list(gens))
    bounds = [(0, None)] + [(0, CAP)] * m
    c = np.r_[0.0, np.ones(m)]
    out = np.full((len(targets), m), np.nan)
    for j, t in enumerate(targets):
        r = linprog(c, A_eq=A, b_eq=t, bounds=bounds, method="highs")
        if r.status == 0:
            out[j] = r.x[1:]
    return out


def capacity(gens, dirs):
    """The largest force the floor plus these supports can put along each direction.

    This is the radial support function of the reachable set, and it is the
    honest replacement for `reach.py`'s cone panel. The cone said "yes" or "no";
    this says HOW HARD, which is the whole point of a cap. Directions the cone
    already rules out are priced at zero without an LP -- `covered` is vectorised
    and settles them for free -- so only the reachable ones cost a solve.
    """
    m = len(gens)
    G = np.column_stack([UP] + list(gens))
    live = covered(G, dirs)
    A = np.column_stack([np.zeros(3), -UP] + [-g for g in gens])
    bounds = [(0, RMAX), (0, None)] + [(0, CAP)] * m
    c = np.r_[-1.0, np.zeros(m + 1)]
    out = np.zeros(len(dirs))
    for j in np.flatnonzero(live):
        A[:, 0] = dirs[j]
        r = linprog(c, A_eq=A, b_eq=np.zeros(3), bounds=bounds, method="highs")
        if r.status == 0:
            out[j] = r.x[0]
    return out


def cone(gens, targets):
    """reach.py's scale-free test, kept so the two models can be quoted side by side."""
    return covered(np.column_stack([UP] + list(gens)), targets)


def exhaustive(targets, avail, test=feasible):
    """Every subset, smallest first, and therefore the true minimum.

    With five or six candidates there are at most 64 subsets, so the beam search
    `reach.py` needs is pure overhead here and, worse, would leave the answer as
    an upper bound. Two things keep it cheap. Both tests are monotone in the
    subset, so a target already answered by S minus one element is answered by S
    and never gets a second solve. And the ceiling -- what all the candidates
    together manage -- is computed first, so the loop stops at the smallest
    subset that reaches it, whether that ceiling is 100% or not.
    """
    n = len(avail)
    memo = {frozenset(): test([], targets)}
    ceiling = float(test([avail[i] for i in range(n)], targets).mean())
    for size in range(n + 1):
        rows = []
        for pick in itertools.combinations(range(n), size):
            known = np.zeros(len(targets), bool)
            for i in pick:
                known |= memo[frozenset(pick) - {i}]
            todo = np.flatnonzero(~known)
            got = known.copy()
            if len(todo):
                got[todo] = test([avail[i] for i in pick], targets[todo])
            memo[frozenset(pick)] = got
            rows.append((got.mean(), pick))
        # ties go to whichever subset sorts last, and that is fine: at the
        # minimum size every subset reaching the ceiling is equally optimal and
        # the figure only draws one of them
        best, pick = max(rows)
        if best >= ceiling - 1e-12:
            return list(pick), float(best), ceiling
    return [], 0.0, ceiling


def bands(cap_field, skip):
    """Quantise the capacity field into the fills `globe` can take.

    Nothing is painted where the requirement is: those tiles carry a state --
    answered, too weak, out of reach -- and a tile can only hold one colour, so
    the state wins there and the field speaks for the rest of the ball.
    """
    edge = np.digitize(cap_field, LEVELS)
    return [(np.flatnonzero((cap_field > 1e-9) & (edge == k) & ~skip), RAMP[k], FIELD_ALPHA)
            for k in range(len(RAMP))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="cuboid_baseline")
    ap.add_argument("--poses", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=840)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push-points", type=int, default=90)
    ap.add_argument("--push-dirs", type=int, default=24)
    # the search is exact on the full requirement and costs about forty seconds
    # a pose, so it is left exact; the flag is here only to reproduce the
    # subsampled numbers the model was first measured on
    ap.add_argument("--subsample", type=int, default=0)
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

    records = []
    for pose in poses:
        # ---- everything down to `avail` is reach.py's pipeline, seed for seed,
        # so that the only thing separating the two figures is the cap
        T = np.asarray(examples[pose]["T_world_mesh"])
        R, t = T[:3, :3], T[:3, 3]
        rng = np.random.default_rng(args.seed + 1000 * pose)
        reg = spot_region(mesh, T, rng, float(rng.uniform(0.17, 0.33)),
                          BOUNDARY_EDGE * float(np.linalg.norm(mesh.extents)))
        part, inside = reg.pop("mesh"), reg.pop("mask")
        n_passes = reg["n_passes"]
        area = float(part.area_faces[inside].sum() / part.area)

        pw, pu = angled_pushes(part, T, inside, args.push_points, args.push_dirs,
                               args.seed + pose)
        targets = UP - args.k * pu
        keep = np.linalg.norm(targets, axis=1) > 1e-9
        targets, pw, pu = targets[keep], pw[keep], pu[keep]
        unit = targets / np.linalg.norm(targets, axis=1, keepdims=True)

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

        sub = (np.random.default_rng(7 + pose).choice(len(targets), args.subsample,
                                                      replace=False)
               if args.subsample and args.subsample < len(targets)
               else np.arange(len(targets)))
        chosen, frac, ceiling = exhaustive(targets[sub], avail)
        # the same exhaustive search under reach.py's scale-free test, so the
        # price of the cap can be quoted rather than asserted: how many supports
        # the cone model would have asked for, and what its answer is actually
        # worth once the supports are only a body weight each
        loose, loose_frac, _ = exhaustive(targets[sub], avail, cone)
        loose_capped = float(feasible([avail[i] for i in loose], targets).mean())
        span = compass(targets)
        pts = part.triangles_center[avail_face[chosen]] @ R.T + t
        us = avail[chosen]

        lam = np.zeros((len(targets), 0))

        # ---- one rule for every ball, inherited unchanged: a force is drawn
        # WHERE IT COMES FROM, at -F, with its arrow running from there into the
        # centre. A push from above puts its patch on top and drives down, and
        # the floor's G therefore sits at the south pole, which is where the
        # floor is.
        h_job, h_back, h_want = tree.query(-pu)[1], tree.query(pu)[1], tree.query(-unit)[1]
        job = np.flatnonzero(sheet(adjacency, h_job))
        back = np.flatnonzero(sheet(adjacency, h_back))
        want = sheet(adjacency, h_want)
        near = cKDTree(-unit).query(tiles)[1]
        band = np.flatnonzero(sheet(adjacency, tree.query(-avail)[1]))
        few = np.random.default_rng(11).choice(len(pu), min(26, len(pu)), replace=False)
        fewt = np.random.default_rng(14).choice(len(unit), min(26, len(unit)), replace=False)
        need_mag = float(np.linalg.norm(targets, axis=1).max())

        # titles stay two short lines: a panel is three inches wide and a long
        # one runs into its neighbour's
        panels = [([(job, PUSH_BLUE, .95)], [], [(pu[few], PUSH_BLUE, False)], [],
                   "the job"),
                  ([(back, NEED, .95)], [], [(-pu[few], NEED, False)], [],
                   "reverse the pushes\nthe far side, sideways in full"),
                  # the compass bound is about directions only, so the cap cannot
                  # weaken it; what is new on this panel is the second number,
                  # the biggest force the requirement ever asks for
                  ([(np.flatnonzero(want), NEED, .95)], [], [(unit[fewt], NEED, False)],
                   [(-UP, FLOOR_MARK, "G")],
                   f"+ its own weight\n{span:.0f}° wide, up to {need_mag:.2f} W"),
                  ([(band, HAVE, .95)], [(-avail, HAVE, 26, True)], [(avail, HAVE, False)],
                   [(-UP, FLOOR_MARK, "G")],
                   f"what we have\n{len(avail)} pushes, {CAP:g} W each")]
        for i in range(len(chosen) + 1):
            gens = [avail[c] for c in chosen[:i]]
            # the field replaces reach.py's cone. Same place on the ball, but it
            # answers "how hard" instead of "at all", and a reader can only see
            # the cap bite if the drawing carries a magnitude.
            field = capacity(gens, -tiles)
            ok = feasible(gens, targets)
            inreach = cone(gens, targets)               # the old, scale-free answer
            # priced for the design AS IT STANDS, not for the finished one: a
            # quoted peak has to belong to the jig in the picture, and the last
            # support added can change what the first one is asked to carry
            lam = effort(gens, targets) if i else lam
            marks = [(-UP, FLOOR_MARK, "G")]
            marks += [(-avail[chosen[j]], "#d2450f", str(j + 1)) for j in range(i)]
            # three states, not two: grey answered, VIOLET where the direction is
            # available but not at the strength wanted, RED where the direction is
            # not available at all. Violet is the entire reason for this tool --
            # under the cone test it does not exist.
            fills = bands(field, want)
            fills += [(np.flatnonzero(want & ok[near]), DONE, .95),
                      (np.flatnonzero(want & inreach[near] & ~ok[near]), WEAK, .95),
                      (np.flatnonzero(want & ~inreach[near]), NEED, .95)]
            arr = [(unit[fewt][inreach[fewt] & ~ok[fewt]], WEAK, False),
                   (unit[fewt][~inreach[fewt]], NEED, False)]
            title = f"{'floor only' if i == 0 else f'+{i}'}\n{100 * ok.mean():.0f}% done"
            if i and ok.any():
                title += f"  ·  peak {np.nanmax(lam):.2f} W"
            panels.append((fills, [], arr, marks, title))
        peak = (np.nanmax(lam, axis=0) if len(chosen) and np.isfinite(lam).any()
                else np.zeros(len(chosen)))

        # ---- the page, laid out as reach.py's: the globe strip is rendered at
        # the full figure width and the caption is sized against THAT
        parts = paint(part, T, inside, set(), tmp, f"cap_p{pose}", rel="_capped_tmp")
        blue = [(pw[j], pu[j]) for j in few]
        bare = shot(args.object, T, parts, px, [], [], triad=False)
        forces = shot(args.object, T, parts, px, blue, [], triad=False)
        shots = [bare, forces, forces, forces]
        xy = screen(mesh, T, px, pts) if len(chosen) else np.zeros((0, 2))
        numf = _font(int(px * 0.055))
        for i in range(len(chosen) + 1):
            im = shot(args.object, T, parts, px, blue,
                      [(pts[j], us[j]) for j in range(i)], triad=False)
            dr = ImageDraw.Draw(im)
            for j in range(i):
                dr.text(tuple(xy[j]), str(j + 1), fill="white", font=numf, anchor="mm",
                        stroke_width=4, stroke_fill=MARK)
            shots.append(im)

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
        # four short lines rather than three long ones: the page is only as wide
        # as its panels, and a caption line wider than that pads the sheet with
        # a blank gutter down the right
        caption = (
            f"{args.object}  pose {pose}   ·   every support pushes at most {CAP:g} body "
            f"weight, each direction is used once, the floor is unbounded\n"
            f"BLUE = the disturbances   ·   GREY = answered   ·   "
            f"RED = the direction is not there at all\n"
            f"VIOLET = the direction is there but the supports are not strong enough   ·   "
            f"ORANGE = a push this workpiece can supply   ·   G = the floor\n"
            f"TEAL = the hardest the chosen supports can push that way "
            f"(pale ≤{LEVELS[0]:g} W … dark >{LEVELS[-1]:g} W)   ·   "
            f"numbered contacts are the supports so far, some behind the part")
        font = _font(int(px * 0.042))
        head = int(px * 0.26)
        wide = max(max(b.size[0] for b in blocks),
                   ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textbbox(
                       (16, 0), caption, font=font)[2] + 16)
        page = Image.new("RGB", (wide, head + sum(b.size[1] for b in blocks)), "white")
        y = head
        for b in blocks:
            page.paste(b, (0, y))
            y += b.size[1]
        ImageDraw.Draw(page).text((16, int(head * .12)), caption,
                                  fill=(90, 90, 90), font=font)
        out = out_dir / f"pose{pose}.png"
        page.save(out)

        records.append({"pose": pose, "n_passes": int(n_passes),
                        "region_area_fraction": area,
                        "n_targets": int(len(targets)),
                        "n_searched": int(len(sub)),
                        "compass_span_deg": span,
                        "max_required_force": need_mag,
                        "n_available": int(len(avail)),
                        "n_supports": len(chosen),
                        "answered_fraction": float(frac),
                        "ceiling_fraction": ceiling,
                        "cap_per_support": CAP,
                        "peak_force": float(peak.max()) if len(chosen) else 0.0,
                        "uncapped_n_supports": len(loose),
                        "uncapped_answered_fraction": float(loose_frac),
                        "uncapped_choice_under_cap": loose_capped,
                        "supports": [{"p": [float(x) for x in pts[i]],
                                      "push": [float(x) for x in us[i]],
                                      "peak_force": float(peak[i])}
                                     for i in range(len(chosen))]})
        print(f"pose {pose:2d}: {len(avail)} candidates, requirement spans {span:5.1f}° "
              f"and reaches {need_mag:.2f} W -> {len(chosen)} supports, "
              f"{100 * frac:5.1f}% answered (ceiling {100 * ceiling:5.1f}%)   {out}",
              flush=True)
        print("          peak force per support: "
              + ", ".join(f"{p:.2f}" for p in peak), flush=True)
        print(f"          uncapped would take {len(loose)} at {100 * loose_frac:.1f}%; "
              f"those {len(loose)} answer {100 * loose_capped:.1f}% once capped", flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    # a --poses run must not throw away the poses it did not run. reach.py's
    # record writes whatever this invocation happened to cover, so
    # objects/cuboid_baseline/reach/reach_k1.json on disk holds pose 0 alone
    # while ten reach/pose*.png sit beside it -- the figures and their record
    # disagree, and nothing says so. Merging by pose number keeps them together.
    out_json = out_dir / f"capped_k{args.k:g}.json"
    keep = {r["pose"]: r for r in (read_json(out_json)["poses"] if out_json.exists() else [])}
    keep.update({r["pose"]: r for r in records})
    write_json(out_json,
               {"object": args.object, "k": args.k, "cap_per_support": CAP,
                "one_support_per_direction": True, "search": "exhaustive over all subsets",
                "poses": [keep[p] for p in sorted(keep)]})


if __name__ == "__main__":
    main()
