"""A fillet is a SHARP CORNER AT ITS CENTRE OF CURVATURE. Companion to wrap_vs_pads.

The same bracket, wrapping two edges of the same cube: (a) a sharp right angle,
(b) a rounded one. The question is whether rounding the edge -- which every real
workpiece has, A1-f's is about 3 mm on a 190 mm face -- changes what the support can
supply. The answer is sharper than "no".

FORCE -- EXACTLY IDENTICAL, AT EVERY RADIUS. A fillet's outward normals sweep
continuously from n_A to n_B, and that is precisely the fan the sharp edge's normal
cone already is. Nothing is added and nothing is missing, and the radius never enters.

MOMENT -- THE FILLET IS A SHARP CORNER AT q. A point on the fillet is p = q + r n with
q the centre of curvature, and a frictionless contact there pushes along u = -n, so

    m = (p - c) x u = (q - r u - c) x u = (q - c) x u - r (u x u) = (q - c) x u

The r (u x u) term VANISHES IDENTICALLY. The radius does not enter the moment either;
only q does. So the rounded edge's moment cone is the sharp-corner cone taken from a
DIFFERENT POINT -- and for a 90 deg edge that point sits r*sqrt(2) inside the vertex.
The two cones share the image of the fan's bisector exactly -- the vertex minus q is
parallel to it, so the difference (vertex - q) x u is zero there -- and part company
towards the ends, by an amount LINEAR IN r. That is the whole discrepancy, and the run
measures it at six radii.

THE DRAWING TRAP, and it is dimension.py's normals(): at true scale the fillet would be
a single pixel and the fan a smudge -- and at this camera the near edge is seen so
obliquely that even a tenth of a side barely bends it. So the fillet HERE IS DRAWN FAT
-- r = 0.12 L, eight times A1-f's -- and every number is reported TWICE, at the drawn
radius and at A1-f's real one, so nobody reads the drawn discrepancy as the real one.
The drawn row is the picture's; the real row is the one to believe.

Why not fatter still, since fatter is easier to see: q climbs the page as r grows, and
past about 0.13 L it reaches the centre of mass and the two land on the same page
abscissa. 0.12 L is the widest fillet that leaves them legible apart. (The com's
roundel is no longer drawn -- see `scene` -- but the point is still there and `q`
running into it still crowds the page.)

Reused wholesale from wrap_vs_pads.py, imported rather than copied so the two figures
cannot drift: the palette, the camera and light, the cube, the contact point P, the
bracket section, the page size, the fan-panel geometry, and unit/between/frame/fan/
moment/outside/agree/prism/faces/shaded/outline/cast/veil/Arrow3D/disc/at/trim. Forked
because they hard-code one body and one cone: scene() and fanel().

Every number on the page is computed at run time and printed. No randomness, so a
clean re-run reproduces the PNG byte for byte.

    python slides/tools/wrap_sharp_vs_round.py    ->  slides/tools/figures/wrap_sharp_vs_round.png
"""
from __future__ import annotations

from common import figure_path

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.patches import FancyArrowPatch, Wedge      # noqa: E402
from mpl_toolkits.mplot3d.art3d import (Line3DCollection,  # noqa: E402
                                        Poly3DCollection)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrap_vs_pads import (                                 # noqa: E402
    INK, MUTED, PAPER, ORANGE, FAN, BODY3, SUPPORT3, GROUND3, SUPPORT_INK,
    GROUND_INK, VIEW, SUN, NEAR, SOFT, CORE, DEEP, ELEV, AZIM,
    L, H, UP3, CENTRE, N_A, N_B, P, ZT, LN, NFAN3, NFAN2, BRACKET,
    R_FILL, R_ARC, R_LAB, XLIM, YLIM, W, HT, TOP, BOT,
    Arrow3D, agree, at, between, cast, disc, fan, faces, frame, moment, outline,
    prism, shaded, trim, unit, veil)

OUT = figure_path('wrap_sharp_vs_round.png')

# the fillet, twice over. R_DRAW is a LIE ABOUT SCALE and the only one on the page:
# it is what the fillet has to be to be seen at all. R_REAL is A1-f's, 3 mm of radius
# on a 190 mm face, and it is the one to believe
R_DRAW, R_REAL = 0.12 * L, 3.0 / 190.0
LADDER = (0.0, R_REAL, 0.02, 0.05, R_DRAW, 0.20)           # the radii tabulated
NARC = 25                                                  # facets across the fillet


# ------------------------------------------------------------------- the numbers


def curvature(r):
    """The centre of curvature of the fillet on the (+x, -y) edge, at P's height.

    For a right angle it sits r*sqrt(2) in along the bisector, which is dimension.py's
    `rounded` specialised to 90 deg -- the only case this figure has.
    """
    return np.array([H - r, -H + r, ZT])


def angle_to_cone(v, g1, g2):
    """The smallest angle, in degrees, from `v` to any ray of cone{g1, g2}.

    outside() gives a residual, which is the right thing when the answer should be
    zero; here it is not zero and a LENGTH would depend on how long m happens to be.
    Resolve v in the cone's own plane: if the projection lands inside, the nearest ray
    is that projection and the answer is how far v leans off the plane; if not, the
    nearest ray is whichever generator is nearer.
    """
    e1, e2 = frame(g1, g2)
    c = np.array([v @ e1, v @ e2])
    a = np.arctan2(c[1], c[0])
    if 0.0 <= a <= np.radians(between(g1, g2)):
        off = np.linalg.norm(v - c[0] * e1 - c[1] * e2)
        return float(np.degrees(np.arcsin(off / np.linalg.norm(v))))
    return min(between(v, g1), between(v, g2))


def measure(r, u_fan, sharp):
    """One row of the ladder: what a fillet of radius `r` does to the moment cone."""
    q = curvature(r)
    m_r = np.array([moment(u, q) for u in u_fan])
    return dict(
        r=r, back=r * 2 ** 0.5, arm=float(np.linalg.norm(q - CENTRE)),
        wide=between(m_r[0], m_r[-1]),
        off=max(angle_to_cone(v, sharp[0], sharp[-1]) for v in m_r),
        drop=100 * max(1 - np.linalg.norm(a) / np.linalg.norm(b)
                       for a, b in zip(m_r, sharp)))


# -------------------------------------------------------------------- the solids


def rounded(section, k, r, n=NARC):
    """`section` with vertex k -- a convex right angle -- replaced by a fillet arc.

    Returns the new section, the indices that are arc, and the centre of curvature.
    The arc runs from the tangent point on the incoming edge to the one on the
    outgoing edge, so the winding, and with it prism()'s faces, is unchanged.
    """
    s = [np.asarray(v, float) for v in section]
    hub = curvature(r)[:2]
    t0 = s[k] - unit(s[k] - s[(k - 1) % len(s)]) * r        # tangent point, incoming
    a0 = np.arctan2(*(t0 - hub)[::-1])
    t = np.linspace(a0, a0 + np.pi / 2, n)                  # 90 deg, always CCW here
    arc = hub + r * np.stack([np.cos(t), np.sin(t)], axis=1)
    return ([v for v in s[:k]] + list(arc) + [v for v in s[k + 1:]],
            set(range(k, k + n)), hub)


def solid(section, arc_idx, z0, z1):
    """prism(), plus a flag per face: True where the face is one facet of the arc.

    A 25-facet fillet drawn with an ink edge on every facet reads as a fluted column,
    not a rounded corner. The facets are sealed with their own colour instead, and the
    two TANGENT edges survive because their quads are not arc quads.
    """
    polys = prism(section, z0, z1)
    n = len(section)
    return polys, [False, False] + [i in arc_idx and j in arc_idx
                                    for i, j in zip(range(n),
                                                    list(range(1, n)) + [0])]


CUBE = [(-H, -H), (H, -H), (H, H), (-H, H)]
SHARP_BODY = solid(CUBE, set(), 0.0, L)
SHARP_WRAP = solid(BRACKET, set(), 0.0, ZT)
ROUND_BODY = solid(*rounded(CUBE, 1, R_DRAW)[:2], 0.0, L)
ROUND_WRAP = solid(*rounded(BRACKET, 5, R_DRAW)[:2], 0.0, ZT)


# ------------------------------------------------------------------- the drawing


def scene(ax, body, wrap, contacts, pushes, tip, floor, lo, hi, hub=None):
    """One panel. wrap_vs_pads.scene(), with the body and the contact set free.

    Two things are new and both are the claim: the arm is drawn to whatever point the
    moments are actually taken from -- p on the sharp edge, q inside the round one --
    and on the round panel every push is continued as a dotted line to q, so the page
    shows the fan converging there rather than only asserting it.
    """
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_axis_off()
    ax.patch.set_visible(False)

    ax.add_collection3d(Poly3DCollection([floor], facecolor=GROUND3,
                                         edgecolor=GROUND_INK, lw=1.4, zorder=0))
    thrown = [cast(s) for s, _ in (body, wrap)]
    for i, t in enumerate((np.arange(SOFT) + 1) / SOFT):
        ax.add_collection3d(Poly3DCollection(
            [np.hstack([q.mean(0) + (1 - CORE * t) * (q - q.mean(0)),
                        np.zeros((len(q), 1))]) for q in thrown],
            facecolor=veil(DEEP * t ** 2), edgecolor="none", zorder=1 + 0.01 * i))

    for (polys, smooth), base, tone, ink, z in ((body, BODY3, SUN, INK, 4),
                                                (wrap, SUPPORT3, NEAR, SUPPORT_INK, 6)):
        seen = [(Q, n, sm) for (Q, n), sm in zip(faces(polys), smooth) if n @ VIEW > 0]
        for want in (False, True):                         # flat faces, then the arc
            got = [(Q, n) for Q, n, sm in seen if sm == want]
            if not got:
                continue
            col = shaded([n for _, n in got], base, *tone)
            ax.add_collection3d(Poly3DCollection(
                [Q for Q, _ in got], facecolor=col,
                edgecolor=col if want else ink, lw=0.6 if want else
                (1.1 if z == 4 else 1.0), zorder=z))
        ax.add_collection3d(Line3DCollection(outline(polys), colors=ink,
                                             lw=2.4 if z == 4 else 1.8, zorder=z + .4))

    # NO centre-of-mass roundel, and that is the one mark this figure gives up.
    # `c` is where the moments are taken about and it does not move; but the
    # roundel is 0.075 across and sits within its own radius of `q`, and `q` --
    # with the fillet arc that meets there -- is the entire subject of panel (b).
    # A mark that hides the thing the page is about is not worth the point it
    # locates, so the dashed arm is left to locate `c` by its own endpoint.
    ax.plot(*np.array([CENTRE, tip]).T, color=MUTED, lw=1.5, ls=(0, (5.0, 3.2)),
            zorder=8)
    for p, u in zip(contacts, pushes):
        ax.add_artist(Arrow3D(p - LN * u, p, zorder=12, color=ORANGE, lw=3.0,
                              arrowstyle="-|>,head_width=.22,head_length=.42"))
    for p in contacts:
        disc(ax, p, 0.010 if hub is not None else 0.032, INK, 11)
    if hub is not None:                                    # the fan, continued inward
        for p in contacts:                                 # OVER the arrows: every
            ax.plot(*np.array([p, hub]).T, color=MUTED, lw=1.5,   # push line meeting
                    ls=(0, (2.2, 2.4)), zorder=12.6)       # at q is the panel's point
        disc(ax, hub, 0.032, "white", 12.8)
    off = 0.20 * unit(np.cross(UP3, VIEW))
    marks = [(np.array([H, 0.24, 0.42]), "A", MUTED, 20),
             (np.array([-0.24, -H, 0.42]), "B", MUTED, 20)]
    # no mark for the sharp vertex: q, the middle contact and the vertex are COLLINEAR
    # along the fan's bisector, so anything drawn at the vertex lands on the middle
    # arrow. The shift shows instead by comparing the two panels' arms, which is where
    # it belongs -- (a)'s runs out to the corner, (b)'s stops short at q
    marks += ([(hub - 1.00 * off + 0.02 * UP3, "q", INK, 20)] if hub is not None
              else [(P - off + 0.10 * UP3, "p", INK, 20)])
    for q, s, col, fs in marks:
        ax.text(*q, s, color=col, fontsize=fs, zorder=13, ha="center", va="center")


# --------------------------------------------------------- the two abstract panels


def fanel(ax, wide_a, wide_b, rays, labels, title, note):
    """Two cones about a SHARED BISECTOR, at the same radius: orange sharp, ink round.

    wrap_vs_pads' fanel drew one cone twice because its two designs coincided. Here
    the force panel still does -- the dashed arc lands exactly on the orange one and
    that IS the force claim -- while the moment panel does not, and the dashed arc
    stops short at both ends by half the difference in opening.

    Sharing the bisector is not a drawing convenience: (V - q) is parallel to the
    fan's bisector, so the two moment cones agree there exactly and nowhere else. What
    the panel CANNOT show is that the round cone leaves the sharp one's plane, so it
    is not a sub-cone however much the picture nests them -- `note` carries that.
    """
    a0, a1 = 90.0 - wide_a / 2, 90.0 + wide_a / 2
    b0, b1 = 90.0 - wide_b / 2, 90.0 + wide_b / 2
    ax.add_patch(Wedge((0, 0), R_FILL, a0, a1, facecolor=FAN, alpha=0.55,
                       edgecolor="none", zorder=1))
    for a in rays:
        ax.plot(*np.array([(0, 0), at(a, R_FILL)]).T, color=ORANGE, lw=0.9, zorder=3)
    ax.plot(*np.array([(0, 0), at(90.0, R_ARC)]).T, color=MUTED, lw=1.3,
            ls=(0, (6.0, 2.5, 1.0, 2.5)), zorder=4)        # the shared ray
    for a, lab, ha in zip((a0, a1), labels, ("left", "right")):
        ax.add_patch(FancyArrowPatch((0, 0), at(a, R_FILL), zorder=6,
                                     arrowstyle="-|>", mutation_scale=26, shrinkA=0,
                                     shrinkB=0, joinstyle="miter", lw=4.2,
                                     color=ORANGE))
        ax.text(*at(a, R_LAB), lab, color=INK, fontsize=19, zorder=7, ha=ha,
                va="center")
        ax.plot(*np.array([at(a, R_ARC - .10), at(a, R_ARC + .10)]).T, color=MUTED,
                lw=1.2, zorder=5)
    for a in (b0, b1):
        ax.plot(*np.array([at(a, R_ARC - .07), at(a, R_ARC + .07)]).T, color=INK,
                lw=1.2, zorder=5.5)
    for lo_, hi_, colour, lw, ls, z in ((a0, a1, ORANGE, 4.2, "-", 5),
                                        (b0, b1, INK, 1.8, (0, (4.4, 4.0)), 6)):
        t = np.radians(np.linspace(lo_, hi_, 361))
        ax.plot(R_ARC * np.cos(t), R_ARC * np.sin(t), color=colour, lw=lw, ls=ls,
                zorder=z, solid_capstyle="butt")
    ax.text(0, R_ARC + 0.05, f"{wide_a:.4f}°", color=ORANGE, fontsize=19, zorder=7,
            ha="center", va="bottom")
    ax.text(0, R_ARC - 0.065, f"{wide_b:.4f}°", color=INK, fontsize=19, zorder=7,
            ha="center", va="top")
    ax.plot([0], [0], marker="o", ms=6.0, color=INK, zorder=7)
    for i, (mark, s) in enumerate(((dict(color=ORANGE, lw=4.2), "(a) sharp edge"),
                                   (dict(color=INK, lw=1.8, ls=(0, (4.4, 4.0))),
                                    "(b) rounded edge"))):
        y = -0.14 - 0.18 * i
        ax.plot([XLIM[0] + 0.06, XLIM[0] + 0.42], [y, y], zorder=7, **mark)
        ax.text(XLIM[0] + 0.52, y, s, color=INK, fontsize=15, ha="left", va="center")
    ax.text(XLIM[1] - 0.04, -0.14, note, color=MUTED, fontsize=13.5, ha="right",
            va="center", zorder=7)
    ax.set_title(title, color=INK, fontsize=20, pad=12)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()


# ----------------------------------------------------------------------- the page


def main():
    hub = curvature(R_DRAW)
    u_fan = fan(N_A, N_B, NFAN2)                           # the force fan, both edges
    sharp = np.array([moment(u) for u in u_fan])           # (a): all from p
    rnd = np.array([moment(u, hub) for u in u_fan])        # (b): all from q

    # (1) the fillet ACTUALLY DRAWN is the radius the page quotes -- measured off the
    # polygon that goes into the scene, so the numbers cannot describe another fillet
    arc = np.array([v for i, v in enumerate(rounded(CUBE, 1, R_DRAW)[0])
                    if i in rounded(CUBE, 1, R_DRAW)[1]])
    drawn_r = float(np.abs(np.linalg.norm(arc - hub[:2], axis=1) - R_DRAW).max())
    assert drawn_r < 1e-12, drawn_r

    # (2) the FORCE claim: the fillet's own normals, read off the drawn arc, are the
    # sharp edge's normal cone -- both ways round, as sets
    n_geo = np.array([np.hstack([unit(v - hub[:2]), 0.0]) for v in arc])
    res_f = agree(-n_geo, (N_A, N_B))
    assert res_f < 1e-15, res_f

    # (3) the MECHANISM: r (u x u) is identically zero, so the long way round the
    # fillet surface and the short way from q give the same moment
    p_geo = np.array([hub + R_DRAW * n for n in n_geo])
    res_r = float(max(np.linalg.norm(moment(-n, p) - moment(-n, hub))
                      for n, p in zip(n_geo, p_geo)))
    assert res_r < 1e-15, res_r

    # (4) and so the rounded edge's moment set IS a sharp corner's, taken at q
    res_m = agree(np.array([moment(-n, p) for n, p in zip(n_geo, p_geo)]),
                  (moment(N_A, hub), moment(N_B, hub)))
    assert res_m < 1e-15, res_m

    rows = [measure(r, u_fan, sharp) for r in LADDER]
    d, real = rows[LADDER.index(R_DRAW)], rows[LADDER.index(R_REAL)]
    wide_f, wide_s = between(N_A, N_B), rows[0]["wide"]

    # the framing: wrap_vs_pads' own, over all four solids and both fans of tails
    r_ax = unit(np.cross(UP3, VIEW))
    page = np.array([r_ax, np.cross(VIEW, r_ax)])
    cube = np.abs(page).sum(axis=1)
    solids = [s for s, _ in (SHARP_BODY, SHARP_WRAP, ROUND_BODY, ROUND_WRAP)]
    floor = np.vstack([cast(s) for s in solids] + [np.vstack(s)[:, :2] for s in solids])
    lo2, hi2 = floor.min(axis=0) - 0.14, floor.max(axis=0) + 0.14
    floor = np.array([[hi2[0], hi2[1], 0.], [hi2[0], lo2[1], 0.],
                      [lo2[0], lo2[1], 0.], [lo2[0], hi2[1], 0.]])
    tails = [P - LN * u for u in fan(N_A, N_B, 41)] + \
            [hub - (R_DRAW + LN) * u for u in fan(N_A, N_B, 41)]
    flat = np.vstack([np.vstack(s) for s in solids] + [floor] + tails) @ page.T
    span, mid = flat.max(axis=0) - flat.min(axis=0), (flat.max(0) + flat.min(0)) / 2
    side = 1.04 * max(span / cube)
    lo, hi = mid @ page - side / 2, mid @ page + side / 2
    tall = (TOP - BOT) / (span[1] / (side * cube[1]))
    wide = tall * HT / W / (cube[1] / cube[0])

    fig = plt.figure(figsize=(W, HT), dpi=210, facecolor=PAPER)
    u3 = fan(N_A, N_B, NFAN3)
    for cx, body, wrapb, pts, tip, hb in (
            (0.25, SHARP_BODY, SHARP_WRAP, [P] * NFAN3, P, None),
            (0.75, ROUND_BODY, ROUND_WRAP, [hub - R_DRAW * u for u in u3], hub, hub)):
        scene(fig.add_axes([cx - wide / 2, (TOP + BOT) / 2 - tall / 2, wide, tall],
                           projection="3d", computed_zorder=False),
              body, wrapb, pts, u3, tip, floor, lo, hi, hb)

    box = 0.480 * W / HT / ((XLIM[1] - XLIM[0]) / (YLIM[1] - YLIM[0]))
    fanel(fig.add_axes([0.010, 0.215, 0.480, box]), wide_f, wide_f,
          [90 - wide_f / 2 + between(N_A, u) for u in u_fan], ("$n_A$", "$n_B$"),
          "FORCE — what each design can push along", "the arcs coincide")
    fanel(fig.add_axes([0.510, 0.215, 0.480, box]), wide_s, d["wide"],
          [90 - wide_s / 2 + between(sharp[0], m) for m in sharp],
          ("$m_A$", "$m_B$"), "MOMENT — what each design can turn the cube about",
          f"and {d['off']:.2f}° out of this plane")

    def v(q):
        q = np.where(np.abs(q) < 5e-5, 0.0, np.asarray(q, float))
        return f"({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f})"
    for cx, rowset in (
            (0.25, [f"$n_A$ = {v(N_A)}", f"$n_B$ = {v(N_B)}",
                    "a fillet's normals sweep $n_A → n_B$ — the sharp edge's cone, "
                    "exactly",
                    f"equal as SETS to {res_f:.1e}, AT EVERY RADIUS"]),
            (0.75, [f"sharp   $m = (p−c)×u$,   $|p−c|$ = "
                    f"{np.linalg.norm(P - CENTRE):.4f} L",
                    f"round   $m = (q−c)×u$,   $|q−c|$ = {d['arm']:.4f} L   —   "
                    f"the $r(u×u)$ term is {res_r:.1e}",
                    f"so a fillet is a SHARP CORNER at q — r√2 = {d['back']:.4f} L "
                    "inside the vertex",
                    f"and its cone is that corner's, to {res_m:.1e}"])):
        for i, s in enumerate(rowset):
            fig.text(cx, 0.196 - 0.0225 * i, s, color=INK, fontsize=15.5,
                     ha="center", va="center")

    for cx, head in ((0.25, ("(a) a SHARP right-angle edge",
                             "one contact point, and its normal cone is the whole "
                             "fan —",
                             f"the bracket pushes anywhere in it, {NFAN3} of a "
                             "continuum")),
                     (0.75, ("(b) a ROUNDED edge — EXAGGERATED",
                             f"drawn at r = {R_DRAW:.2f} L, {R_DRAW / R_REAL:.0f}× "
                             "A1-f's, or it could not be seen —",
                             "a contact per direction, and every push runs "
                             "through q"))):
        for i, (s, size, col) in enumerate(zip(head, (23, 16, 16),
                                               (INK, MUTED, MUTED))):
            fig.text(cx, 0.985 - 0.0225 * i, s, color=col, fontsize=size,
                     ha="center", va="center")

    for i, s in enumerate((
            "ROUNDING THE EDGE CHANGES THE FORCE SET NOT AT ALL, and moves the "
            "moment set only by moving the POINT.",
            "A fillet is a SHARP CORNER at q: $p = q + r n$ and $u = −n$, so "
            "$m = (q − c) × u$ — the radius cancels.",
            f"Drawn {R_DRAW / R_REAL:.0f}× life size (r = {R_DRAW:.2f} L): cone "
            f"{wide_s - d['wide']:.2f}° narrower, worst axis {d['off']:.2f}° off, "
            f"|m| down {d['drop']:.1f} %.",
            f"A1-f's REAL fillet, 3 mm on a 190 mm face (r = {R_REAL:.4f} L), gives "
            f"{real['off']:.2f}° and {real['drop']:.1f} %. Frictionless — METHOD §1.")):
        fig.text(0.5, 0.080 - 0.0225 * i, s, color=INK, fontsize=17, ha="center",
                 va="center")

    fig.savefig(OUT, facecolor=PAPER)
    plt.close(fig)
    trim(OUT)

    print(f"cube side {L:.4f}   c = {v(CENTRE)}   p = vertex = {v(P)}   "
          f"|p - c| = {np.linalg.norm(P - CENTRE):.4f} L")
    print(f"drawn fillet r = {R_DRAW:.4f} L   q = {v(hub)}   "
          f"r*sqrt(2) = {R_DRAW * 2 ** .5:.4f} L inside the vertex")
    print(f"real fillet   r = {R_REAL:.6f} L  (A1-f, 3 mm on a 190 mm face) -- the "
          f"drawn one is {R_DRAW / R_REAL:.1f}x life size")
    print()
    print(f"FORCE   n_A = {v(N_A)}   n_B = {v(N_B)}   {wide_f:.4f} deg apart")
    print("        a fillet's normals sweep n_A -> n_B and stop there: same cone, "
          "any radius")
    print(f"MOMENT  sharp m_A = {v(sharp[0])}   m_B = {v(sharp[-1])}   "
          f"{wide_s:.4f} deg apart")
    print(f"        round m_A = {v(rnd[0])}   m_B = {v(rnd[-1])}   "
          f"{d['wide']:.4f} deg apart  (drawn r)")
    print("        the two share the bisector exactly: (vertex - q) is parallel to "
          f"it, residual {np.linalg.norm(np.cross(P - hub, unit(N_A + N_B))):.1e}")
    print()
    print(f"ASSERT  drawn arc is a circle of radius {R_DRAW} about q   "
          f"worst deviation {drawn_r:.3e}")
    print(f"        force sets equal, sharp vs round        worst residual {res_f:.3e}")
    print(f"        r (u x u) vanishes, so (p-c)xu = (q-c)xu       max gap "
          f"{res_r:.3e}")
    print(f"        round moment set IS a sharp corner's at q      residual "
          f"{res_m:.3e}")
    print()
    print("MEASURED  the fillet's moment cone against the sharp edge's, "
          f"over {NFAN2} rays of the fan")
    print(f"  {'radius':>14}  {'':10}  {'q back by':>10}  {'|q-c|':>7}  "
          f"{'opening':>9}  {'worst axis off':>14}  {'|m| falls':>9}")
    for row in rows:
        tag = ("sharp" if row["r"] == 0 else
               "A1-f, REAL" if row["r"] == R_REAL else
               "DRAWN" if row["r"] == R_DRAW else "")
        print(f"  r = {row['r']:.6f}  {tag:<10}  {row['back']:10.4f}  "
              f"{row['arm']:7.4f}  {row['wide']:8.4f}°  {row['off']:13.4f}°  "
              f"{row['drop']:8.3f} %")
    print("  the discrepancy is linear in r and goes to zero with it; the DRAWN row "
          "is the picture's, the REAL row is A1-f's")
    print()
    a = plt.imread(OUT)
    print(f"{OUT}   {a.shape[1]} x {a.shape[0]}")


if __name__ == "__main__":
    main()
