"""The weakest direction two capped supports can push in, and the 90 degree wall.

`dimension.py` ends on the observation that the angular SIZE of a cone of contact
forces says nothing about what can actually be pushed along it: two supports 179
deg apart reach 179 deg of directions and can put 1.7 % of one support's strength
into them. This figure is that observation on its own, in the plane, with nothing
else in the picture.

The set-up is two contacts pushing along unit directions `t` apart, each able to
supply at most the same force `F`. In the plane those two pushes are a basis for
every direction between them, so a unit direction `phi` off the first push is
reached one way and one way only:

    d = c1 u1 + c2 u2      c1 = sin(t - phi) / sin t     c2 = sin(phi) / sin t

Both coefficients are non-negative on the arc, so neither contact is asked to
pull, and scaling the whole thing by `m` scales both. The cap binds on whichever
is larger, so the pair can deliver

    m(phi) = F / max(c1, c2)

and the guarantee -- what the pair is worth in the WORST direction the arc
contains -- is the minimum of that over phi. It comes out

    min m  =  F              t <= 90 deg
           =  sin(t) * F     t >  90 deg

There is a second way to see the same thing, and it is the one the top row of the
figure draws. Each contact contributes a segment of forces, `[0, F] u`, so what
the PAIR can supply is the Minkowski sum of two segments: the RHOMBUS with corners
0, F u1, F u2 and F(u1 + u2). "How hard can the pair push along phi" is then just
"how far is it from the corner at 0 to the far boundary of that rhombus, along
phi", and the guarantee is the nearest point of that far boundary. The far
boundary is two straight edges, the one through F u2 parallel to u1 and the one
through F u1 parallel to u2, so the guarantee is a point-to-SEGMENT distance and
the whole answer is elementary:

    the foot of the perpendicular from 0 onto the edge through u2 sits at
    parameter -cos t along it, which is OUTSIDE the segment while t <= 90 and
    INSIDE it once t > 90

Outside, the nearest point is the segment's own end -- a corner of the rhombus, a
push direction, at distance F. Inside, the nearest point is the foot, at distance
|u2 + (-cos t) u1| = sin(t) F. The 90 deg wall is exactly the moment the foot of a
perpendicular slides onto the edge, and nothing else. `checks()` computes the
guarantee this second way too, from point-to-segment distances alone, and gets the
same numbers as the sweep.

Two things about that are worth a figure, because both are easy to get wrong.

FIRST, THE WEAKEST DIRECTION IS NOT THE BISECTOR. The bisector is the strongest
part of the arc, not the weakest: it splits the load evenly, c1 = c2, and reaches
`2 cos(t/2) F`. The floor sits at the ENDS while the pair is inside a right angle
-- a push direction is weakest, because only one contact contributes to it at all
-- and past 90 deg it sits at `phi = 90` and `phi = t - 90`, square to one push or
the other, where the OTHER contact is asked for `1/sin t` of the total.

SECOND, THE THRESHOLD IS 90 DEG AND NOT 120. 120 deg is where the bisector value
`2 cos(t/2)` falls to `F`, which is a real number about the middle of the arc and
the wrong one for a guarantee. At 120 deg the bisector still reads 1.000 F and the
true weakest direction is already down to 0.866 F.

Every arrow in the figure is drawn on ONE encoding -- length and line width and
head all proportional to the force it stands for -- so a weakest-direction arrow
at `sin(t) F` can be compared by eye against a push arrow at full `F` and against
the same arrow in the next panel. That is the same encoding the other figures in
this series use. `arrow()` records the one liberty taken with it, which is that the
two colours have different base weights so that a blue arrow landing exactly on an
orange one can still be seen.

    python slides/tools/min_force.py     ->  slides/tools/figures/min_force.png

Deterministic: no randomness anywhere, so a clean re-run reproduces the file byte
for byte. Every number printed is computed here, and the closed form above is
checked against dense sampling before anything is drawn.
"""
from __future__ import annotations

from common import figure_path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.patches import Polygon                   # noqa: E402

INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
BLUE, ORANGE = "#2563EB", "#E08A24"

OUT = figure_path('min_force.png')

# the five pairs drawn across the top. 60 and 90 are both inside the wall, so
# they must look alike -- that is the point of showing two of them; 120 is the
# angle the bisector argument gets wrong; 150 is half strength; 175 is the
# collapse. 180 itself is NOT drawn as a pair, because there is no wedge left to
# draw; it is handled on the curve instead, and the reason is in `curve()`
SHOWN = [60.0, 90.0, 120.0, 150.0, 175.0]

# the angles quoted in the printed table. They are the ones dimension.md's table
# already carries, so the two documents can be checked against each other, plus
# 170 and 175 to fill in the tail
TABLE = [30.0, 60.0, 89.0, 90.0, 91.0, 100.0, 120.0, 150.0, 170.0, 175.0, 179.0]

F = 1.0                                                  # the cap, and the unit


def coeffs(t, phi):
    """How much of each contact a unit direction `phi` off the first push needs.

    Solving `d = c1 u1 + c2 u2` with u1 at 0 and u2 at t, for d at phi: the
    cross product with u2 kills c2 and leaves c1 sin t = sin(t - phi), and the
    cross product with u1 leaves c2 sin t = sin phi. Both are >= 0 for phi in
    [0, t], which is what makes this a contact problem at all -- outside that arc
    one of them turns negative and the pair would have to PULL.
    """
    s = np.sin(t)
    return np.sin(t - phi) / s, np.sin(phi) / s


def reach(t, phi):
    """The largest force the pair can put along `phi`, with each capped at F.

    Scaling the direction scales both coefficients together, so the cap binds on
    the larger one and on nothing else. Note the two ends: at phi = 0 the answer
    is exactly F for every t whatsoever, because c1 = 1 and c2 = 0 -- one contact
    doing all of it. The arc always leaves its own endpoints alone; what collapses
    is everything in between.
    """
    c1, c2 = coeffs(t, phi)
    return F / np.maximum(c1, c2)


def weakest(t_deg):
    """The closed form: the guarantee, and where on the arc it is attained.

    max(c1, c2) is what has to be MAXIMISED, and c1 = sin(t - phi)/sin t is just a
    sine sweeping the interval [0, t] backwards. If t <= 90 that sweep never gets
    past the top of the sine, so c1 <= 1 with equality at phi = 0, and c2 <= 1 with
    equality at phi = t: the cap is reached first at the two ENDS and the whole arc
    holds F. If t > 90 the sweep passes the top: c1 hits 1/sin t at phi = t - 90 and
    c2 hits it at phi = 90, and 1/sin t > 1, so those two interior directions are
    strictly worse than the ends.

    Both branches are checked against dense sampling in `checks()`; nothing here is
    taken on faith.
    """
    t = np.deg2rad(t_deg)
    if t_deg <= 90.0:
        return F, [0.0, t_deg]                           # the push directions themselves
    return np.sin(t) * F, sorted({t_deg - 90.0, 90.0})


def bisector(t_deg):
    """What the MIDDLE of the arc reaches -- the seductive wrong answer.

    c1 = c2 = sin(t/2)/sin t = 1/(2 cos(t/2)), so the middle reaches 2 cos(t/2) F.
    It falls to F at t = 120, which is where the "120 deg" folklore comes from: a
    rhombus whose diagonal equals its side. It is the arc's BEST direction, so it
    can only ever overstate the guarantee, and the overstatement is exactly
    1/sin(t/2) -- 15.5 % at 120 deg, 41.4 % at 90.
    """
    return 2 * np.cos(np.deg2rad(t_deg) / 2) * F


def by_geometry(t_deg):
    """The guarantee again, from point-to-segment distance and nothing else.

    A wholly independent route to the same number, so that a slip in the
    trigonometry above cannot pass unnoticed. What the pair can supply is the
    rhombus 0, u1, u2, u1+u2 (the Minkowski sum of the two contacts' segments),
    and the guarantee is the distance from the corner at 0 to the NEAREST point of
    the far boundary -- the two edges [u1, u1+u2] and [u2, u1+u2].

    Written out for the edge that starts at u2 and runs along u1: the foot of the
    perpendicular sits at s = -u2.u1 = -cos t, and clamping s to [0, 1] is the
    entire content of the two branches. It returns the clamped s as well, because
    s crossing 0 IS the wall: s <= 0 while t <= 90, s in (0, 1) after.
    """
    t = np.deg2rad(t_deg)
    u1, u2 = np.array([1.0, 0.0]), np.array([np.cos(t), np.sin(t)])
    s = float(np.clip(-u2 @ u1, 0.0, 1.0))               # foot, clamped to the edge
    return F * np.linalg.norm(u2 + s * u1), s


def sampled(t_deg, n=200001):
    """The same minimum found by brute force, with the minimisers it lands on.

    Endpoints included, because on the flat branch that is exactly where the
    answer lives and a sampler that opened the interval would miss it entirely.
    Near-ties are clustered so that both minimisers come back rather than whichever
    one argmin happened to see first.
    """
    t = np.deg2rad(t_deg)
    phi = np.linspace(0.0, t, n)
    m = reach(t, phi)
    lo = m <= m.min() + 1e-9
    idx = np.where(lo)[0]
    cuts = np.where(np.diff(idx) > 1)[0]
    return m.min(), [np.degrees(phi[g].mean()) for g in np.split(idx, cuts + 1)]


# --------------------------------------------------------------- the checks


def checks():
    """Everything the figure claims, re-derived numerically and printed.

    Run before a single artist is created, so a broken claim never reaches paper.
    """
    print("the guarantee: closed form, a 200001-point sweep of the arc, and the "
          "rhombus\n")
    print("    t     min m over the arc   = ?       where it sits, phi        "
          "the bisector    overstates")
    worst = wg = 0.0
    for deg in TABLE:
        m_c, where_c = weakest(deg)
        m_s, where_s = sampled(deg)
        m_g, _ = by_geometry(deg)
        worst, wg = max(worst, abs(m_c - m_s)), max(wg, abs(m_c - m_g))
        # the two lists must agree as sets, not just the values
        assert len(where_c) == len(where_s), (deg, where_c, where_s)
        for a, b in zip(where_c, where_s):
            assert abs(a - b) < 0.01, (deg, where_c, where_s)
        tag = "F     " if deg <= 90 else "sin t "
        seat = "a push itself" if deg <= 90 else "square to a push"
        at = " and ".join(f"{w:.0f}" for w in where_s) + " deg"
        print(f"{deg:6.0f}         {m_s:.4f} F      {tag}   {at:<14} {seat:<17} "
              f"{bisector(deg):.4f} F     x {bisector(deg) / m_s:.3f}")
    print(f"\n   closed form vs sweep, worst disagreement over those angles: "
          f"{worst:.2e}")
    print(f"   closed form vs point-to-segment distance on the rhombus:     "
          f"{wg:.2e}")

    # a fine sweep of t as well, so the claim is about the whole range and not
    # about eleven chosen angles
    bad, at = 0.0, None
    for t in np.deg2rad(np.linspace(0.05, 179.95, 3600)):
        deg = np.degrees(t)
        m = reach(t, np.linspace(0.0, t, 20001)).min()
        d = abs(m - weakest(deg)[0])
        if d > bad:
            bad, at = d, deg
    print(f"   and over 3600 angles from 0.05 to 179.95 deg: {bad:.2e} at t = "
          f"{at:.2f} deg")

    # WHERE the wall is, found rather than asserted. Two things blunt a bisection
    # here and both are known in advance, which is what makes the residual
    # evidence rather than noise. The minimum leaves F QUADRATICALLY --
    # sin(90 + e) = 1 - e^2/2 -- so accepting "still F" within `tol` cannot place
    # the wall inside sqrt(2 tol) radians of it. And a sweep of n points can miss
    # the minimiser by up to half a step d, which lifts the sampled minimum by
    # about d^2/2 and blunts it the same way. So the answer must land in
    # [90 + sqrt(2 tol), 90 + sqrt(2 tol + d^2)] -- and it does, at every
    # tolerance, from one that is all tolerance to one that is all grid
    print("\n   the wall, by bisection on the SAMPLED minimum (not on the formula)")
    n = 40001
    for tol in (1e-4, 1e-7, 1e-10, 1e-13):
        lo, hi = 45.0, 135.0
        for _ in range(64):
            mid = (lo + hi) / 2
            if sampled(mid, n)[0] >= F - tol:
                lo = mid
            else:
                hi = mid
        d = np.deg2rad(lo) / (n - 1) / 2                  # half a sample step, radians
        band = (np.degrees(np.sqrt(2 * tol)),
                np.degrees(np.sqrt(2 * tol + d * d)))
        ok = "yes" if band[0] <= lo - 90 <= band[1] * 1.0001 else "NO"
        print(f"      accept within {tol:.0e} of F  ->  {lo:11.7f} deg  =  90 + "
              f"{lo - 90:.7f}   predicted band {band[0]:.7f} .. {band[1]:.7f}"
              f"   {ok}")
    print("      the residual is entirely tolerance and grid, so the wall is at 90.")

    # the same wall with no bisection and no sampling at all: the foot of the
    # perpendicular from 0 onto the rhombus edge sits at s = -cos t, so it is off
    # the end of the segment (clamped to 0) for every t up to 90 and on it after.
    # There is nothing to converge to here -- s crosses at exactly 90 by the sign
    # of a cosine
    print("\n   and with no search at all: s = clamp(-cos t) is where the nearest "
          "point of\n   the rhombus edge sits. s = 0 means a CORNER (a push); s > 0 "
          "means the edge itself")
    for deg in (60.0, 89.0, 90.0, 91.0, 120.0, 150.0):
        m_g, s = by_geometry(deg)
        print(f"      t = {deg:5.0f}   s = {s:.4f}   "
              + ("nearest point is the corner u2 -- min = F"
                 if s == 0 else f"nearest point is inside the edge -- min = sin t "
                                f"= {m_g:.4f} F"))

    # and the wrong threshold, for contrast
    lo, hi = 90.0, 179.0
    for _ in range(64):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if bisector(mid) >= F else (lo, mid)
    print(f"   watching the BISECTOR instead puts the wall at {lo:.7f} deg -- 120, "
          f"and wrong by 30")

    # the minimiser is never the bisector except in the limit
    print("\n   is the minimiser ever the bisector? m(bisector)/min m = 1/sin(t/2)")
    for deg in (30.0, 90.0, 120.0, 150.0, 175.0, 179.0):
        m_s, where = sampled(deg)
        mid = deg / 2
        off = min(abs(w - mid) for w in where)
        print(f"      t = {deg:5.0f}   bisector at phi = {mid:5.1f}, nearest "
              f"minimiser at {min(where, key=lambda w: abs(w - mid)):5.1f}  "
              f"-- {off:4.1f} deg away;  ratio {bisector(deg) / m_s:.4f}")
    print("      the gap closes only as t -> 180, where the whole arc is collapsing"
          " together.")
    print()


# --------------------------------------------------------------- the drawing


def unit(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a)])


PUSH_LW, PUSH_MS = 4.6, 20.0                             # an arrow standing for F
WEAK_LW, WEAK_MS = 2.7, 12.5                             # ditto, drawn narrower


def arrow(ax, tip, colour, mag, lw, ms, z=10):
    """One force, drawn on the one encoding this series uses.

    Length AND line width AND head all scale with the magnitude, so an arrow at
    half the force is half as long and half as heavy and reads as half whichever
    way the eye takes it. Nothing is floored: at t = 175 the weakest direction is
    0.0872 F and it comes out a whisker, because that is what it is. Scaling only
    the length would leave a full-weight stroke standing for a force of nothing.

    The two base weights differ on purpose and it is the one liberty taken here.
    Below the wall the weakest direction IS a push, so the blue arrow lands
    exactly on the orange one, and drawn at equal weight the pair would be a
    single arrow of ambiguous colour. Narrower, the blue sits inside the orange as
    a core with the orange showing round it, which is the finding rather than a
    collision. Length is the quantity to read across the two colours; width
    compares blue against blue.
    """
    ax.annotate("", tip, (0.0, 0.0), zorder=z,
                arrowprops=dict(arrowstyle="-|>,head_width=.30,head_length=.55",
                                mutation_scale=ms * mag, color=colour,
                                lw=lw * mag, shrinkA=0, shrinkB=0))


def panel(ax, t_deg, first):
    """One pair, everything it can deliver, and the direction it delivers least in.

    The pair is drawn symmetric about straight up so the five panels can be laid
    in a row and read as one thing opening. That is free: only the angle between
    the pushes appears in the mathematics.

    The blue region is what the pair can supply -- every direction of the arc out
    to `m(phi)`. It is traced from `reach` point by point rather than asserted,
    and it comes back a RHOMBUS every time, which is a check in itself: the sum of
    two segments of forces can be nothing else. Its two lower corners are pinned
    to the arrow TIPS at every t, because a push direction always reaches exactly
    F, so the reader watches the middle fall away from the ends with the ends
    held still.
    """
    m_min, phis = weakest(t_deg)
    half = t_deg / 2
    # phi is measured from the first push; put that push on the left
    glob = lambda phi: 90.0 + half - phi

    # the dashed arc is the F level AND the span of directions at once: it runs
    # from one push to the other, so it is the arc being minimised over, drawn at
    # the height a single support reaches. Anything inside it is a direction the
    # pair cannot put a whole support into
    a = np.deg2rad(np.linspace(90 - half, 90 + half, 361))
    ax.plot(np.cos(a), np.sin(a), color=MUTED, lw=1.15, ls=(0, (4, 3)), zorder=3)

    # hairlines out to that arc along the weakest directions, so a whisker of an
    # arrow still says WHERE it points and the shortfall from F reads as the gap
    # between its tip and the arc. Below the wall they hide under the pushes,
    # which is correct -- there the weakest directions are the pushes
    for p in phis:
        ax.plot([0, unit(glob(p))[0]], [0, unit(glob(p))[1]], color=MUTED,
                lw=0.85, ls=(0, (1.3, 2.0)), zorder=2)

    phi = np.linspace(0.0, t_deg, 721)
    m = reach(np.deg2rad(t_deg), np.deg2rad(phi))
    lobe = (m[:, None] * np.stack([unit(g) for g in glob(phi)]))
    ax.add_patch(Polygon(np.vstack([[0.0, 0.0], lobe]), closed=True,
                         facecolor=BLUE, alpha=.15, edgecolor="none", zorder=4))
    ax.plot(lobe[:, 0], lobe[:, 1], color=BLUE, lw=1.5, alpha=.85, zorder=5)

    # the far corner of the rhombus: both contacts at full F at once, which is the
    # bisector and the arc's BEST direction. Drawn so it can be seen that the
    # bisector is the TOP of the region and never the bottom
    b = bisector(t_deg)
    ax.plot([0, 0], [0, b], color=MUTED, lw=1.1, ls=(0, (1.6, 2.2)), zorder=6)
    ax.plot([0], [b], marker="o", ms=4.6, mfc=PAPER, mec=MUTED, mew=1.3, zorder=7)

    # the two pushes, at full F -- the same encoding with mag = 1
    for g in (90 - half, 90 + half):
        arrow(ax, unit(g), ORANGE, 1.0, PUSH_LW, PUSH_MS, z=10)

    # and the weakest directions, at mag = the guarantee
    for p in phis:
        ax.plot(*(m_min * unit(glob(p))), marker="o", ms=4.0, mfc=BLUE,
                mec="none", zorder=13)
        arrow(ax, m_min * unit(glob(p)), BLUE, m_min, WEAK_LW, WEAK_MS, z=12)

    ax.set_title(f"t = {t_deg:.0f}°", color=INK, fontsize=16, pad=8)
    seat = ("the pushes themselves" if t_deg < 90 else
            "the pushes — and square to them" if t_deg == 90 else
            "square to a push")
    ax.text(0, -0.33, f"weakest  {m_min:.3f} F", ha="center", va="center",
            color=BLUE, fontsize=14)
    ax.text(0, -0.55, "φ = " + " and ".join(f"{p:.0f}°" for p in phis),
            ha="center", va="center", color=MUTED, fontsize=11)
    ax.text(0, -0.72, seat, ha="center", va="center", color=MUTED, fontsize=11)
    ax.text(0, -0.93, f"bisector reaches {b:.3f} F", ha="center", va="center",
            color=MUTED, fontsize=11)
    if first:
        # labelled once. Both get a leader: a floating "reach = F" beside a panel
        # whose arc only spans 60 deg is an unattached caption, and the reader has
        # to guess which of three dashed things it names
        ax.annotate("reach = F", xy=(-0.46, 0.898), xytext=(-1.06, 1.30),
                    color=MUTED, fontsize=10.5, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                    shrinkA=3, shrinkB=2))
        ax.annotate("what the\npair reaches", xy=(0.20, 1.06), xytext=(0.56, 0.62),
                    color=BLUE, fontsize=10.5, ha="left", va="center",
                    linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9, alpha=.8,
                                    shrinkA=4, shrinkB=2))
    ax.set_xlim(-1.10, 1.10)
    ax.set_ylim(-1.03, 1.90)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return m_min, phis


def curve(ax):
    """The guarantee against t, with the bisector beside it for contrast.

    Both curves are evaluated, not drawn from a sketch of the shape: `weakest` is
    the same function the panels use and the same one `checks()` puts against a
    sweep.
    """
    t = np.linspace(0.001, 179.999, 4000)
    m = np.array([weakest(x)[0] for x in t])
    b = bisector(t)

    ax.fill_between(t, m, b, color=MUTED, alpha=.09, lw=0, zorder=1)
    ax.axhline(F, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(1.5, 1.035, "F", color=MUTED, fontsize=12)

    ax.plot(t, b, color=MUTED, lw=1.9, ls=(0, (5, 3)), zorder=4,
            label="the MIDDLE of the arc,  2 cos(t/2) F  —  its best direction")
    ax.plot(t, m, color=BLUE, lw=3.2, zorder=6,
            label="the WEAKEST direction on the arc,  min m")

    # the five panels above, tied to the curve they are samples of
    for x in SHOWN:
        y = weakest(x)[0]
        ax.plot([x, x], [0, y], color=MUTED, lw=0.9, ls=(0, (1.4, 2.4)), zorder=3)
        ax.plot([x], [y], marker="o", ms=6.5, mfc=BLUE, mec=PAPER, mew=1.2, zorder=8)

    # the corner. It is a genuine corner -- flat, then sin t, the two branches
    # meeting with different slopes -- and an eye following the bisector curve
    # smoothly down through 120 never sees it
    ax.annotate("the corner is at 90°\nflat before it,  sin t after",
                xy=(90, 1.0), xytext=(46, 0.62), color=BLUE, fontsize=13,
                ha="center", va="center", linespacing=1.5, zorder=9,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.1,
                                shrinkA=8, shrinkB=3))
    ax.text(45, 1.08, "every direction of the arc takes a full support",
            color=BLUE, fontsize=12, ha="center", va="bottom")

    # 120: the seductive wrong answer, drawn as the gap it actually is
    ax.plot([120, 120], [weakest(120)[0], bisector(120)], color=ORANGE, lw=2.4,
            zorder=7, solid_capstyle="butt")
    for y in (bisector(120), weakest(120)[0]):
        ax.plot([120], [y], marker="o", ms=6, mfc=ORANGE, mec=PAPER, mew=1.1,
                zorder=8)
    ax.annotate("at 120° the middle still reads 1.000 F\n"
                "while the weakest direction is already 0.866 F.\n"
                "120° is where the MIDDLE drops to F — not the arc",
                xy=(120, 0.93), xytext=(148, 1.92), color=ORANGE, fontsize=12.5,
                ha="center", va="center", linespacing=1.5, zorder=9,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.1,
                                shrinkA=10, shrinkB=4))

    # the tail is labelled just PAST the corner and not out at 160, because the
    # two curves converge as t -> 180 -- the ratio between them is 1/sin(t/2) --
    # and a label out there would sit on both of them at once
    ax.annotate("sin t · F", xy=(128, np.sin(np.deg2rad(128))), xytext=(134, 0.58),
                color=BLUE, fontsize=14, ha="center", va="center", zorder=9,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.1,
                                shrinkA=8, shrinkB=3))

    # THE TWO DEGENERATE ENDS, marked as what they are rather than left to run off
    # the plot. At exactly 180 the pushes span a LINE and not a wedge: there are no
    # directions strictly between them, so the minimum is over an EMPTY arc and not
    # a small one. What the pair can still do is push along the line itself, either
    # way, at exactly F. So the curve's limit of 0 is a statement about directions
    # just OFF the line, and the honest pair of marks is an OPEN circle at 0 -- a
    # limit, not a value -- with a FILLED one at F for what actually happens there.
    # t = 0 is degenerate in the opposite direction and gets the same treatment:
    # the pushes coincide, the arc is one direction, and both contacts drive it, so
    # the pair puts 2F along it. There the BISECTOR curve is the one that is right
    ax.plot([180], [0.0], marker="o", ms=9, mfc=PAPER, mec=BLUE, mew=2.0, zorder=10,
            clip_on=False)
    ax.plot([180], [1.0], marker="o", ms=9, mfc=BLUE, mec=PAPER, mew=1.3, zorder=10,
            clip_on=False)
    ax.text(93, 0.205, "at exactly 180° the pushes span a LINE, not a wedge:\n"
                       "no directions in between, so nothing to minimise over.\n"
                       "Along that line the pair still delivers a full F  (●).\n"
                       "The curve's 0  (○)  is the limit for directions just off it.",
            color=INK, fontsize=10.5, ha="left", va="center", linespacing=1.6,
            zorder=9)
    ax.plot([0], [2.0], marker="o", ms=8, mfc=MUTED, mec=PAPER, mew=1.2, zorder=10,
            clip_on=False)
    ax.text(4, 2.18, "at 0° the two pushes coincide: one direction, both contacts "
                     "driving it, 2F along it",
            color=MUTED, fontsize=11, ha="left", va="center")

    ax.set_xlim(-2, 186)
    ax.set_ylim(0, 2.32)
    ax.set_xticks(range(0, 181, 30))
    ax.set_xticklabels([f"{d}°" for d in range(0, 181, 30)])
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("t   —   the angle between the two pushes", color=INK,
                  fontsize=13.5, labelpad=8)
    ax.set_ylabel("force the pair can deliver,\nin units of one support's cap F",
                  color=INK, fontsize=12.5, linespacing=1.6, labelpad=8)
    ax.tick_params(colors=MUTED, labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.0)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.004, 0.010), frameon=True,
                    fontsize=12, handlelength=2.8, borderpad=0.8, labelspacing=0.7)
    leg.get_frame().set_facecolor(PAPER)
    leg.get_frame().set_edgecolor(MUTED)
    leg.get_frame().set_linewidth(0.8)
    for txt, c in zip(leg.get_texts(), (MUTED, BLUE)):
        txt.set_color(c)


def main():
    checks()

    fig = plt.figure(figsize=(15.0, 10.6), dpi=210, facecolor=PAPER)
    # the row of pairs is drawn with `aspect="equal"`, so its axes boxes shrink to
    # the data they hold and the height ratio is what stops the row from sitting
    # in a band of its own whitespace. 1 : 1.42 is where the panels are limited by
    # the column width rather than by the row height, which is as large as five
    # across a 15 inch page can be
    gs = fig.add_gridspec(2, len(SHOWN), height_ratios=[1.0, 1.42],
                          left=0.078, right=0.988, top=0.895, bottom=0.062,
                          wspace=0.02, hspace=0.10)
    fig.text(0.5, 0.960, "Two supports t apart, each capped at F:  which direction "
             "between them is the weakest, and how bad it gets",
             ha="center", va="center", color=INK, fontsize=18)

    for i, t_deg in enumerate(SHOWN):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(PAPER)
        panel(ax, t_deg, i == 0)

    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor(PAPER)
    curve(ax)

    fig.savefig(OUT, facecolor=PAPER)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
