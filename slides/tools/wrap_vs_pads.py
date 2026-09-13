"""Wrapping a convex edge buys nothing: the bracket and the two pads span one cone.

The intuition to kill is that a support which WRAPS an edge -- one bracket, touching
both faces and reaching round the corner -- holds a workpiece in more ways than two
separate pads, one on each face. It does not.

FORCE. A frictionless contact pushes only along the inward normal. At a CONVEX EDGE
that normal is not unique: the normal cone there is the whole fan between the two
faces' inward normals, and a body wrapping the edge really can push along any of it.
That is the whole of its apparent advantage, and it is worth nothing, because the fan
between n_A and n_B IS the set of non-negative combinations of n_A and n_B -- which is
what the two pads already give.

MOMENT. Not automatic, and the reason is worth having. At a FIXED point p the map
u |-> (p - c) x u is LINEAR, so it carries a cone to a cone and an interpolation to an
interpolation: the wrap's fan goes onto the fan between m_A = (p - c) x n_A and
m_B = (p - c) x n_B, which is exactly the cone the pads' two moments span. Same
conclusion, same reason one level up. What the map does not preserve is ANGLE -- 90
deg of force fan comes out 143.13 deg of moment fan, and the evenly spaced rays of the
left panel come out visibly bunched in the right one. Linear, not conformal.

THE CAVEAT. The force equality needs only the same DIRECTIONS. The moment equality
needs the same POINTS as well, because m depends on p -- METHOD s11.1's "a contact is
a point AND a direction". Both designs are drawn hugging the edge and both are
declared to contact the same p, and that is what makes the moment claim exact: slide
the pads out onto the faces and the force cone does not move at all while the moment
set leaves the plane m_A and m_B span. The run measures it; the caption carries it.

THE OTHER CAVEAT. This is the frictionless model, which is what METHOD s1 locks. The
mu = 0.5 that turns up in the pipeline's inputs is Passive Grippers' number (s8), and
friction "quantitatively" is on s5's list of what is NOT modelled. Nothing here is
claimed under friction.

Every number on the page is computed at run time and printed. No randomness, so a
clean re-run reproduces the PNG byte for byte.

    python slides/tools/wrap_vs_pads.py    ->  slides/tools/figures/wrap_vs_pads.png
"""
from __future__ import annotations

from common import figure_path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.patches import FancyArrowPatch, Wedge      # noqa: E402
from mpl_toolkits.mplot3d import proj3d                    # noqa: E402
from mpl_toolkits.mplot3d.art3d import (Line3DCollection,  # noqa: E402
                                        Poly3DCollection)
from scipy.spatial import ConvexHull                       # noqa: E402

# the house palette, unchanged from dimension.py. FAN is the pale orange that file
# fills a cone with on the ball, and a cone is what the lower panels are filled with
INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
ORANGE, FAN = "#E08A24", "#F2C88E"
BODY3, SUPPORT3, GROUND3 = "#b0b5ae", "#bfc5c8", "#f1f2ed"
SUPPORT_INK, GROUND_INK, DUSK = "#79838a", "#a3a9a4", "#49525c"

OUT = figure_path('wrap_vs_pads.png')

# dimension.py's camera and light, unchanged. The azimuth is NOT free here: an edge is
# visible only when both faces meeting along it are, and at -45 those are +x and -y,
# so the edge between them is the near vertical one, standing on the silhouette
ELEV, AZIM, LELEV, LAZIM = 36.0, -45.0, 64.0, -95.0
VIEW, LIGHT = (np.array([np.cos(np.radians(e)) * np.cos(np.radians(a)),
                         np.cos(np.radians(e)) * np.sin(np.radians(a)),
                         np.sin(np.radians(e))])
               for e, a in ((ELEV, AZIM), (LELEV, LAZIM)))
SUN = 0.80, 0.40, 0.25              # ambient, key, bounce -- the body
# forked off dimension.py's HAZE and it has to be: there the chocks stand BEHIND the
# plate and are flattened for aerial perspective, here they stand in FRONT of it, and
# a flattened solid in the foreground reads as a smudge. Same colour, wider key
NEAR = 0.84, 0.32, 0.18
SOFT, CORE, DEEP = 14, 0.45, 0.19   # the shadow's steps, how far in, how dark

L, H = 1.0, 0.5                     # the cube's side, and its half extent
UP3 = np.array([0.0, 0.0, 1.0])
CENTRE = np.array([0.0, 0.0, H])    # uniform, so the centre of mass is the centre
N_A = np.array([-1.0, 0.0, 0.0])    # the +x face's INWARD normal -- the way a support
N_B = np.array([0.0, 1.0, 0.0])     # pressing it can push. Likewise the -y face

# where on the edge the contact sits, and it is not free. The moment generators
# subtend arccos(-H^2 / (H^2 + z^2)) with z the height of p above the centre, so at
# the edge's MID height that is 180 deg: the fan's bisector runs straight through the
# centre of mass, its moment is zero, and the moment cone collapses from a wedge to a
# line. 3H/2 stands a quarter of the cube clear of that, gives |p - c| = 3L/4 and
# arm = sqrt(5)L/4 exactly, and puts the contact at the TOP of a support that stands
# on the floor -- so p is not buried behind the design's own material
P = np.array([H, -H, 3 * H / 2])
ARM = P - CENTRE

T, A_ARM, ZT = 0.20 * L, 0.40 * L, P[2]        # support thickness, reach, height
LN, NFAN3, NFAN2 = 0.38, 7, 25                 # arrow length; rays drawn, and drawn
GSLIDE = 0.30 * L                              # how far the caveat slides the pads

# (a) ONE body, going round the outside of the corner. (b) TWO, meeting only along the
# edge line. Their union is (a) minus the block at the outer corner, and that missing
# block is the whole visible difference -- the only material the wrap has that the
# pads do not, and it touches nothing
BRACKET = [(H, -H + A_ARM), (H + T, -H + A_ARM), (H + T, -H - T),
           (H - A_ARM, -H - T), (H - A_ARM, -H), (H, -H)]
PAD_A = [(H, -H), (H + T, -H), (H + T, -H + A_ARM), (H, -H + A_ARM)]
PAD_B = [(H - A_ARM, -H - T), (H, -H - T), (H, -H), (H - A_ARM, -H)]


# ------------------------------------------------------------------- the numbers


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def between(a, b):
    """The angle between two vectors, in degrees."""
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def frame(g1, g2):
    """An orthonormal basis of the plane two generators span, first axis on g1."""
    e1 = unit(g1)
    return e1, unit(g2 - (g2 @ e1) * e1)


def fan(a, b, n):
    """`n` unit directions evenly spaced IN ANGLE along the arc from a to b.

    Even spacing is for the drawing: it is what makes the bunching in the moment
    panel visible, the linear map being unable to keep them even.
    """
    e1, e2 = frame(a, b)
    t = np.linspace(0.0, np.radians(between(a, b)), n)
    return np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2


def moment(u, at=P):
    """(p - c) x u -- the moment a push at `at` makes about the centre of mass."""
    return np.cross(np.asarray(at, float) - CENTRE, u)


def outside(v, g1, g2):
    """The distance from `v` to cone{g1, g2}. Zero exactly when v lies in the cone.

    A residual and not a verdict, so the two asserts are reported without a
    tolerance anywhere. The cone is two-dimensional: resolve v in the plane's own
    basis, and what is left over is how far off that plane it lies and how far
    round the nearer bounding ray it has slipped.
    """
    e1, e2 = frame(g1, g2)
    c = np.array([v @ e1, v @ e2])
    top = np.radians(between(g1, g2))
    off = np.linalg.norm(v - c[0] * e1 - c[1] * e2)
    a = np.arctan2(c[1], c[0])
    gap = 0.0 if 0.0 <= a <= top else min(
        np.linalg.norm(c - max(c @ d, 0.0) * d)
        for d in (np.array([1.0, 0.0]), np.array([np.cos(top), np.sin(top)])))
    return float(np.hypot(off, gap))


def agree(wrap, pads):
    """The two designs' cones as SETS: the worst residual, taken both ways."""
    return max([outside(v, *pads) for v in wrap]
               + [outside(v, wrap[0], wrap[-1]) for v in pads])


# -------------------------------------------------------------------- the solids


def prism(section, z0, z1):
    """A cross-section in the xy plane, extruded between two heights."""
    s = np.asarray(section, float)
    lo, hi = (np.hstack([s, np.full((len(s), 1), z)]) for z in (z0, z1))
    return [lo[::-1], hi] + [np.array([lo[i], lo[j], hi[j], hi[i]])
                             for i, j in zip(range(len(s)),
                                             list(range(1, len(s))) + [0])]


CUBE = prism([(-H, -H), (H, -H), (H, H), (-H, H)], 0.0, L)
WRAP = [prism(BRACKET, 0.0, ZT)]
PADS = [prism(PAD_A, 0.0, ZT), prism(PAD_B, 0.0, ZT)]


# ------------------------------------------------------------------- the drawing


def faces(polys):
    """Each face with its outward normal, taken from the winding and the centre."""
    hub = np.vstack(polys).mean(axis=0)
    out = []
    for Q in polys:
        n = np.cross(Q[1] - Q[0], Q[2] - Q[1])
        out.append((Q, n / np.linalg.norm(n) * np.sign(n @ (Q.mean(axis=0) - hub))))
    return out


def shaded(ns, base, amb, key, bounce):
    """A colour per face: Lambert, plus what the near-white floor throws back up."""
    n = np.atleast_2d(np.asarray(ns, float))
    f = amb + key * np.clip(n @ LIGHT, 0, None) + bounce * np.clip(-n[:, 2], 0, None)
    return np.clip(f[:, None] * np.array(matplotlib.colors.to_rgb(base)), 0, 1)


def outline(polys):
    """The silhouette: edges with a visible face on one side only, drawn heavier.

    Generic rather than dimension.py's axis-aligned version, because the bracket is
    an L and has edges no box has.
    """
    seen = {}
    for Q, n in faces(polys):
        for i in range(len(Q)):
            k = tuple(sorted([tuple(np.round(Q[i], 9)),
                              tuple(np.round(Q[(i + 1) % len(Q)], 9))]))
            seen.setdefault(k, []).append(n @ VIEW > 0)
    return [np.array(k) for k, v in seen.items() if len(v) == 2 and v[0] != v[1]]


def cast(polys):
    """A solid's vertices dropped along the light onto the floor: its shadow."""
    Q = np.vstack(polys)
    Q = Q[:, :2] - np.outer(Q[:, 2] / LIGHT[2], LIGHT[:2])
    return Q[ConvexHull(Q).vertices]


def veil(t):
    """The floor colour with `t` of the light taken out of it. Opaque, not
    transparent: two crossing shadows would otherwise seam along an edge that is
    not there."""
    g, d = (np.array(matplotlib.colors.to_rgb(c)) for c in (GROUND3, DUSK))
    return (1 - t) * g + t * d


class Arrow3D(FancyArrowPatch):
    """A flat arrow with both ends pinned to points in space: mplot3d has none and
    its quiver draws the head as bare lines. dimension.py's, unchanged."""

    def __init__(self, tail, tip, **kw):
        super().__init__((0, 0), (0, 0), mutation_scale=10, **kw)
        self.ends = np.array([tail, tip], float)

    def do_3d_projection(self, renderer=None):
        x, y, _ = proj3d.proj_transform(*self.ends.T, self.axes.M)
        self.set_positions((x[0], y[0]), (x[1], y[1]))
        return 0.0                                         # do not depth sort


def disc(ax, c, r, colour, z, n=48):
    """A filled circle built in the VIEW plane, so it stays a circle on the page."""
    e1 = unit(np.cross(UP3, VIEW))
    e2 = np.cross(VIEW, e1)
    t = np.radians(np.arange(0, 360, 360 / n))[:, None]
    ax.add_collection3d(Poly3DCollection(
        [c + r * (np.cos(t) * e1 + np.sin(t) * e2)], facecolor=colour, edgecolor=INK,
        lw=1.3, zorder=z))
    return e1, e2


def scene(ax, design, pushes, floor, lo, hi):
    """One panel: the cube, one design's supports, and what that design can push."""
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)          # a CUBE, or the elevation drawn is not ELEV
    ax.set_axis_off()
    ax.patch.set_visible(False)         # the rects overlap their neighbours

    ax.add_collection3d(Poly3DCollection([floor], facecolor=GROUND3,
                                         edgecolor=GROUND_INK, lw=1.4, zorder=0))
    thrown = [cast(s) for s in [CUBE] + design]
    for i, t in enumerate((np.arange(SOFT) + 1) / SOFT):
        ax.add_collection3d(Poly3DCollection(
            [np.hstack([q.mean(0) + (1 - CORE * t) * (q - q.mean(0)),
                        np.zeros((len(q), 1))]) for q in thrown],
            facecolor=veil(DEEP * t ** 2), edgecolor="none", zorder=1 + 0.01 * i))

    for solid, base, tone, ink, z in ([(CUBE, BODY3, SUN, INK, 4)]
                                      + [(s, SUPPORT3, NEAR, SUPPORT_INK, 6)
                                         for s in design]):
        seen = [(Q, n) for Q, n in faces(solid) if n @ VIEW > 0]
        ax.add_collection3d(Poly3DCollection(
            [Q for Q, _ in seen], facecolor=shaded([n for _, n in seen], base, *tone),
            edgecolor=ink, lw=1.1 if z == 4 else 1.0, zorder=z))
        ax.add_collection3d(Line3DCollection(outline(solid), colors=ink,
                                             lw=2.4 if z == 4 else 1.8, zorder=z + .4))

    e1, e2 = disc(ax, CENTRE, 0.075, "white", 9)           # the centre-of-mass roundel
    for a in (90, 270):
        t = np.radians(np.arange(a, a + 91, 5))[:, None]
        ax.add_collection3d(Poly3DCollection(
            [np.vstack([CENTRE, CENTRE + 0.075 * (np.cos(t) * e1 + np.sin(t) * e2)])],
            facecolor=INK, edgecolor="none", zorder=9.1))
    # the arm, drawn over the body because that is where most of it runs. Without it
    # on the page there is nothing to say what the moments are measured from
    ax.plot(*np.array([CENTRE, P]).T, color=MUTED, lw=1.5, ls=(0, (5.0, 3.2)), zorder=8)
    for u in pushes:                    # head AT the contact, tail out along -u
        ax.add_artist(Arrow3D(P - LN * u, P, zorder=12, color=ORANGE, lw=3.0,
                              arrowstyle="-|>,head_width=.22,head_length=.42"))
    disc(ax, P, 0.032, INK, 11)
    off = 0.20 * unit(np.cross(UP3, VIEW))                 # one page-width, leftwards
    for q, s, col in ((CENTRE + 0.15 * UP3, "c", INK), (P - off + 0.10 * UP3, "p", INK),
                      (np.array([H, 0.24, 0.42]), "A", MUTED),
                      (np.array([-0.24, -H, 0.42]), "B", MUTED)):
        ax.text(*q, s, color=col, fontsize=20, zorder=13, ha="center", va="center")


# --------------------------------------------------------- the two abstract panels

R_FILL, R_ARC, R_LAB = 1.00, 1.17, 1.36
XLIM, YLIM = (-1.62, 1.62), (-0.42, 1.44)


def at(ang, r):
    return r * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])


def fanel(ax, opening, rays, labels, title):
    """One cone, drawn twice: as (a)'s continuum and as (b)'s two generators.

    The panel's whole job is that the two land on top of each other, so the designs
    get DIFFERENT MARKS AT THE SAME PLACE rather than different places: (a)'s cone
    is a solid orange arc and (b)'s a dashed ink one laid over it at the same
    radius. That is the ordinary way of saying two curves coincide and, unlike
    nesting them, it implies no containment either way. The muted ticks make the
    ENDS checkable by eye: both arcs have to stop on them.
    """
    a0, a1 = 90.0 - opening / 2, 90.0 + opening / 2        # bisector up the page
    ax.add_patch(Wedge((0, 0), R_FILL, a0, a1, facecolor=FAN, alpha=0.55,
                       edgecolor="none", zorder=1))
    for a in rays:                                         # (a): a sampled continuum
        ax.plot(*np.array([(0, 0), at(a, R_FILL)]).T, color=ORANGE, lw=0.9, zorder=3)
    for a, lab, ha in zip((a0, a1), labels, ("left", "right")):
        ax.add_patch(FancyArrowPatch((0, 0), at(a, R_FILL), zorder=6,   # (b): two
                                     arrowstyle="-|>", mutation_scale=26, shrinkA=0,
                                     shrinkB=0, joinstyle="miter", lw=4.2,
                                     color=ORANGE))
        ax.text(*at(a, R_LAB), lab, color=INK, fontsize=19, zorder=7, ha=ha,
                va="center")
        ax.plot(*np.array([at(a, R_ARC - .10), at(a, R_ARC + .10)]).T, color=MUTED,
                lw=1.2, zorder=5)
    t = np.radians(np.linspace(a0, a1, 361))
    for colour, lw, ls, z in ((ORANGE, 4.2, "-", 5), (INK, 1.8, (0, (4.4, 4.0)), 6)):
        ax.plot(R_ARC * np.cos(t), R_ARC * np.sin(t), color=colour, lw=lw, ls=ls,
                zorder=z, solid_capstyle="butt")
    ax.text(0, R_ARC + 0.05, f"{opening:.4f}°", color=INK, fontsize=19, zorder=7,
            ha="center", va="bottom")
    ax.plot([0], [0], marker="o", ms=6.0, color=INK, zorder=7)
    for i, (mark, s) in enumerate(((dict(color=ORANGE, lw=4.2), "(a) wrap"),
                                   (dict(color=INK, lw=1.8, ls=(0, (4.4, 4.0))),
                                    "(b) pads"))):
        y = -0.14 - 0.18 * i
        ax.plot([XLIM[0] + 0.06, XLIM[0] + 0.42], [y, y], zorder=7, **mark)
        ax.text(XLIM[0] + 0.52, y, s, color=INK, fontsize=15, ha="left", va="center")
    ax.set_title(title, color=INK, fontsize=20, pad=12)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()


def trim(out, pad=20):
    """Cut the saved image down to what was drawn: the framing cube leaves slack
    above and below a scene that is wide and low, and cropping distorts nothing."""
    a = plt.imread(out)
    ink = (a[:, :, :3] < 0.96).any(axis=2)
    r, c = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    matplotlib.image.imsave(out, a[max(r.min() - pad, 0):r.max() + 1 + pad,
                                   max(c.min() - pad, 0):c.max() + 1 + pad])


# ----------------------------------------------------------------------- the page

W, HT = 16.0, 16.0                  # the figure, in inches
TOP, BOT = 0.915, 0.545             # the band the SCENES' ink is asked to fill


def main():
    n_f = fan(N_A, N_B, NFAN2)                             # (a)'s force generators
    m_a, m_b = moment(N_A), moment(N_B)                    # (b)'s moment generators
    n_m = np.array([moment(u) for u in n_f])               # (a)'s, by the linear map
    open_f, open_m = between(N_A, N_B), between(m_a, m_b)
    arm = float(np.linalg.norm(m_a))

    # the two asserts, and they are the claim: the cones are equal as SETS
    res_f, res_m = agree(n_f, (N_A, N_B)), agree(n_m, (m_a, m_b))

    # the caveat, measured: slide each pad along its own face, away from the edge
    s_a, s_b = moment(N_A, P + GSLIDE * N_B), moment(N_B, P + GSLIDE * N_A)
    e1, e2 = frame(m_a, m_b)
    tilt = max(float(np.degrees(np.arcsin(
        np.linalg.norm(v - (v @ e1) * e1 - (v @ e2) * e2) / np.linalg.norm(v))))
        for v in (s_a, s_b))
    rank = int(np.linalg.matrix_rank(np.column_stack([m_a, m_b, s_a, s_b]), tol=1e-9))
    slid_f = max(outside(v, N_A, N_B) for v in (N_A, N_B))

    # the camera's page axes, the floor both panels share, and the cube that frames
    # them. The framing box must be a CUBE, so the axes rectangle has to carry the
    # aspect a cube projects to; that rectangle comes out taller than the scene
    # inside it, and `TOP`/`BOT` are where the INK is asked to land rather than the
    # rectangle. The rectangles overlap their neighbours and their patches are off
    r = unit(np.cross(UP3, VIEW))
    page = np.array([r, np.cross(VIEW, r)])
    cube = np.abs(page).sum(axis=1)                        # a unit cube's page size
    solids = [CUBE] + WRAP + PADS
    floor = np.vstack([cast(s) for s in solids] + [np.vstack(s)[:, :2] for s in solids])
    lo2, hi2 = floor.min(axis=0) - 0.14, floor.max(axis=0) + 0.14
    floor = np.array([[hi2[0], hi2[1], 0.], [hi2[0], lo2[1], 0.],
                      [lo2[0], lo2[1], 0.], [lo2[0], hi2[1], 0.]])
    flat = np.vstack([np.vstack(s) for s in solids] + [floor]
                     + [P - LN * u for u in fan(N_A, N_B, 41)]) @ page.T
    span, mid = flat.max(axis=0) - flat.min(axis=0), (flat.max(0) + flat.min(0)) / 2
    side = 1.04 * max(span / cube)
    lo, hi = mid @ page - side / 2, mid @ page + side / 2
    tall = (TOP - BOT) / (span[1] / (side * cube[1]))      # the rect, from the ink
    wide = tall * HT / W / (cube[1] / cube[0])

    fig = plt.figure(figsize=(W, HT), dpi=210, facecolor=PAPER)
    # BOTH panels get the SAME fan, and that is the whole point of the page.
    # An earlier cut drew (a) as a fan and (b) as its two generators, which is
    # true of the GENERATORS -- a pad on one face has exactly one normal -- and
    # badly false as a first impression: a figure whose claim is "these two are
    # the same" must not open with them looking different. What each design can
    # PUSH ALONG is the non-negative combinations of what it touches, and for
    # both of them that is `cone{n_A, n_B}`, the fan drawn here.
    for cx, design, ps in ((0.25, WRAP, fan(N_A, N_B, NFAN3)),
                           (0.75, PADS, fan(N_A, N_B, NFAN3))):
        scene(fig.add_axes([cx - wide / 2, (TOP + BOT) / 2 - tall / 2, wide, tall],
                           projection="3d", computed_zorder=False),
              design, ps, floor, lo, hi)

    box = 0.480 * W / HT / ((XLIM[1] - XLIM[0]) / (YLIM[1] - YLIM[0]))
    fanel(fig.add_axes([0.010, 0.215, 0.480, box]), open_f,
          [90 - open_f / 2 + between(N_A, u) for u in n_f], ("$n_A$", "$n_B$"),
          "FORCE — what each design can push along")
    fanel(fig.add_axes([0.510, 0.215, 0.480, box]), open_m,
          [90 - open_m / 2 + between(m_a, w) for w in n_m], ("$m_A$", "$m_B$"),
          "MOMENT — what each design can turn the cube about")

    def v(q):                           # a vector, with no negative zeros in it
        q = np.where(np.abs(q) < 5e-5, 0.0, np.asarray(q, float))
        return f"({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f})"
    for cx, rows in (
            (0.25, [f"$n_A$ = {v(N_A)}", f"$n_B$ = {v(N_B)}",
                    "the fan IS the non-negative combinations of the two",
                    f"(a) and (b) agree, as SETS, to {res_f:.1e}"]),
            (0.75, [f"$m_A$ = {v(m_a)}", f"$m_B$ = {v(m_b)}",
                    f"arm {arm:.4f} L,   |p − c| = {np.linalg.norm(ARM):.4f} L,   "
                    f"$u ↦ (p−c)×u$ is LINEAR",
                    f"so it carries the fan onto the fan — agreeing to {res_m:.1e}"])):
        for i, s in enumerate(rows):
            fig.text(cx, 0.196 - 0.0225 * i, s, color=INK, fontsize=15.5,
                     ha="center", va="center")

    for cx, rows in ((0.25, ("(a) ONE bracket, WRAPPING the edge",
                             "it touches face A, face B and the edge between them",
                             f"and pushes anywhere in this fan — {NFAN3} of a continuum")),
                     (0.75, ("(b) TWO pads, one on each face",
                             "two bodies, the same point, the same two faces —",
                             "and the SAME fan: the corner block adds no direction")))  :
        for i, (s, size, col) in enumerate(zip(rows, (23, 16, 16), (INK, MUTED, MUTED))):
            fig.text(cx, 0.985 - 0.0225 * i, s, color=col, fontsize=size,
                     ha="center", va="center")

    for i, s in enumerate((
            "Both push into the same normal cone at the same p: WRAPPING THE EDGE "
            "BUYS NOTHING, in force and in moment alike.",
            "Force needs only the same DIRECTIONS; moment needs the same POINTS too. "
            f"Slide the pads {GSLIDE:.2f} L onto the faces:",
            f"the force cone does not move ({slid_f:.1e}), the moment set tilts "
            f"{tilt:.2f}° out of this plane, rank 2 → rank {rank}.",
            "Frictionless contact throughout — the model METHOD §1 locks. Nothing "
            "here is claimed under friction.")):
        fig.text(0.5, 0.080 - 0.0225 * i, s, color=INK, fontsize=17, ha="center",
                 va="center")

    fig.savefig(OUT, facecolor=PAPER)
    plt.close(fig)
    trim(OUT)

    print(f"cube side {L:.4f}   c = {v(CENTRE)}   p = {v(P)}   p - c = {v(ARM)}   "
          f"|p - c| = {np.linalg.norm(ARM):.4f} L  (3L/4)")
    print()
    print(f"FORCE   n_A = {v(N_A)}   n_B = {v(N_B)}   {open_f:.4f} deg apart")
    print(f"MOMENT  m_A = {v(m_a)}   m_B = {v(m_b)}   {open_m:.4f} deg apart")
    print(f"        arm |m_A| = |m_B| = {arm:.6f} L = sqrt(5) L/4 = {5 ** .5 / 4:.6f}")
    print(f"        linear, not conformal: {open_f:.2f} deg of force fan comes out "
          f"{open_m:.2f} deg of moment fan")
    for t, u in zip(np.linspace(0, open_f, 5), fan(N_A, N_B, 5)):
        print(f"          force ray {t:6.2f} deg off n_A  ->  moment ray "
              f"{between(m_a, moment(u)):6.2f} deg off m_A")
    print()
    print(f"ASSERT  each of (a)'s {NFAN2} generators in (b)'s cone, and (b)'s 2 in (a)'s:")
    print(f"          force cones equal as sets    worst residual {res_f:.3e}")
    print(f"          moment cones equal as sets   worst residual {res_m:.3e}")
    print()
    print(f"CAVEAT  slide each pad {GSLIDE:.4f} L along its own face, off the edge:")
    print(f"          force cone unchanged, residual {slid_f:.3e}")
    print(f"          m_A' = {v(s_a)}   m_B' = {v(s_b)}")
    print(f"          {tilt:.4f} deg out of the m_A, m_B plane; the four together "
          f"have rank {rank}, not 2")
    print("        frictionless (METHOD s1). mu = 0.5 is Passive Grippers' (s8), and "
          "s5 lists friction as not modelled")
    print()
    a = plt.imread(OUT)
    print(f"{OUT}   {a.shape[1]} x {a.shape[0]}")


if __name__ == "__main__":
    main()
