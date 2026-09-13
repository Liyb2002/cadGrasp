"""Two supports both pushing at F: how much of it actually arrives along a direction.

Pure plane statics, flat on the page, and one operation throughout: RESOLVE each
force along a chosen direction and ADD the two components. Contacts push along
unit directions `u1` and `u2`, `t` apart, each with magnitude `F`. Pick any
direction `w` between them, drop a perpendicular from each force's tip onto the
`w` axis, and the foot of that perpendicular is that force's component along `w`.
The pair delivers the sum of the two:

    F(u1.w) + F(u2.w) = F(u1 + u2).w

Writing `w` at angle `psi` from the bisector, so that the two forces stand at
`t/2 - psi` and `t/2 + psi` from it, the components are `F cos(t/2 - psi)` and
`F cos(t/2 + psi)` and they add to

    F (u1 + u2).w = 2F cos(t/2) cos(psi)

Nothing here is quoted from anywhere. `audit()` recomputes every number at run
time -- from the dot products, against the closed forms, over a sweep -- and
asserts it before a single line is drawn.

**On the bisector each force gives F cos(t/2) and the pair delivers 2F cos(t/2):
1.7321 F at 60 deg, 1.4142 F at 90 deg, 0.0872 F at 175 deg, and 0.0000 F at
180 deg.** The far factor `cos(psi)` never exceeds one, so the bisector is the
best direction there is and `2F cos(t/2)` is the whole of what a pair can do.

**The 180 deg panel is the point of the figure, not a degenerate case.**
`u1 + u2 = 0` exactly, so the sum of the two projections is zero **in every
direction w**, not merely on the bisector -- two opposed supports both pushing at
full F deliver nothing, anywhere, and `audit` checks that over a full circle. The
nuance that keeps the panel honest: such a pair CAN drive a net force along its
own line, but only by pushing UNEQUALLY -- `c1 u1 + c2 u2 = (c1 - c2) u1` -- and
pushing both at F is exactly the case that gives zero. At 175 deg the same
cancellation is all but complete: resolved onto the other push's direction, a
force contributes `F cos 175 deg = -0.9962 F`, so one force very nearly undoes
the other, and past 90 deg apart one of the two components is NEGATIVE over most
of the wedge -- drawn pointing backwards, because that is what it does.

One thing this figure is careful NOT to be. Resolving and adding is not the same
operation as DECOMPOSING -- solving `v = c1 u1 + c2 u2` for what each support
must supply to deliver a wanted `v`. That asks for `1/(2 cos(t/2))` from each,
which is 0.5774 where resolving gives 0.8660, and the two agree only where
`cos(t/2) = 1/(2 cos(t/2))`, i.e. only at t = 90 deg. Each panel says so in one
line and spends no arrows on it.

    python slides/tools/projection_angles.py   ->  slides/tools/figures/projection_angles.png

Deterministic: no sampling, no randomness, no time stamps, and every panel's size
on the page is computed from its own contents rather than set by hand. A clean
re-run reproduces the PNG byte for byte, and the PNGs written by the other
scripts in `slides/tools/` must not move at all.
"""
from __future__ import annotations

from common import figure_path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch  # noqa: E402

INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
BLUE, ORANGE = "#2563EB", "#E08A24"

OUT = figure_path('projection_angles.png')

# the four spreads, in the order they are drawn: two that behave, dimension.py's
# near-opposed pair, and the opposed one the figure is really about
APART = (60.0, 90.0, 175.0, 180.0)

# every force in the figure is F, and F is 1. The figure is in units of F
F = 1.0

# a total below this is smaller than its own arrowhead and gets a ring round it
# rather than a bigger arrow than it deserves
TINY, RING = 0.25, 0.42

PAGE, MARGIN = 12.6, 0.30                                # inches: drawable, and margin
COLGAP, ROWGAP = 0.34, 0.62                              # inches between panels
SMALL, MID = 8.0, 9.2                                    # label point sizes
LINES = 1.55                                             # label line spacing


def at(deg):
    """The unit vector at `deg` degrees, measured the usual way from +x."""
    a = np.deg2rad(deg)
    return np.array([np.cos(a), np.sin(a)])


def pushes(t):
    """The two push directions, `t` apart and straddling straight up.

    Putting the BISECTOR up rather than putting u1 along +x earns its keep twice:
    every panel is then symmetric about the vertical and reads the same way, and
    the near-opposed pairs at 175 and 180 deg come out lying almost along the
    horizontal, which is the shape of a panel.
    """
    return at(90 + t / 2), at(90 - t / 2)


def resolve(t, psi):
    """Each force resolved along w, straight off the dot product.

    `psi` is w's angle from the bisector, positive towards u1, so w lies between
    the pushes exactly while |psi| <= t/2. Nothing is precomputed here: these are
    the projections themselves, and `audit` is what checks them against the
    closed forms cos(t/2 -+ psi).
    """
    w = at(90 + psi)
    u1, u2 = pushes(t)
    return w, F * float(u1 @ w), F * float(u2 @ w)


def delivered(t, psi):
    """The closed form for what the pair delivers along w: 2F cos(t/2) cos(psi).

    The `cos(psi)` factor is at most one, so the bisector is always the best
    direction and 2F cos(t/2) is the most the pair can do in any direction at
    all. At t = 180 the leading factor is zero and so is everything else.
    """
    return 2 * F * np.cos(np.deg2rad(t / 2)) * np.cos(np.deg2rad(psi))


def backwards(t):
    """The share of the wedge on which one component comes out NEGATIVE.

    u1.w = cos(t/2 - psi) turns negative once psi < t/2 - 90, and u2.w once
    psi > 90 - t/2, so the components are both forwards exactly while
    |psi| < 90 - t/2. Below 90 deg apart that covers the whole wedge and nothing
    ever points backwards; above it the bad share is 2 max(0, t - 90) / t and
    runs to the whole wedge as t runs to 180. `audit` checks this against a sweep
    rather than trusting it.
    """
    return 100 * 2 * max(0.0, t - 90) / t


def rebuild(t):
    """What DECOMPOSING would ask for, kept only for the one line of contrast.

    Solving v = c1 u1 + c2 u2 for a unit v on the bisector gives c1 = c2 =
    1/(2 cos(t/2)) -- what each support must SUPPLY to deliver one unit -- where
    resolving asks what arrives from a supply of F. The two are reciprocal in
    2 cos(t/2) and agree only where cos(t/2) = 1/(2 cos(t/2)), i.e. only at
    90 deg. None if the pair is singular, where no decomposition exists at all.
    """
    c = np.cos(np.deg2rad(t / 2))
    return None if abs(c) < 1e-9 else 1 / (2 * c)


# ------------------------------------------------------------------ the numbers

def audit():
    """Re-derive every claim the figure makes, and assert it.

    Runs before anything is drawn, so a broken identity stops the script instead
    of producing a confident-looking picture.
    """
    print("resolve each force along w, then add: F(u1.w) + F(u2.w) = F(u1+u2).w")
    for t in APART:
        u1, u2 = pushes(t)
        both = u1 + u2
        print(f"\n  {t:5.1f} deg apart     u1 + u2 = ({both[0]:+.6f}, {both[1]:+.6f})"
              f"     |u1 + u2| = {np.linalg.norm(both):.6f}")

        # the whole wedge, not only the bisector: 721 directions between the two
        # pushes, each resolved from the dot products and checked against the
        # closed forms
        psis = np.linspace(-t / 2, t / 2, 721)
        worst_pair, worst_sum, least = 0.0, 0.0, []
        for psi in psis:
            w, p1, p2 = resolve(t, psi)
            worst_pair = max(worst_pair,
                             abs(p1 - F * np.cos(np.deg2rad(t / 2 - psi))),
                             abs(p2 - F * np.cos(np.deg2rad(t / 2 + psi))))
            worst_sum = max(worst_sum, abs(p1 + p2 - delivered(t, psi)))
            least.append(min(p1, p2))
        assert worst_pair < 1e-12, "the components must be F cos(t/2 -+ psi)"
        assert worst_sum < 1e-12, "the sum must be 2F cos(t/2) cos(psi)"
        # -1e-12 rather than 0: at t = 90 the wedge edge resolves to 6.1e-17,
        # positive by luck of rounding, and a bare < 0 would report a backwards
        # component that is not there
        share = 100 * float((np.array(least) < -1e-12).mean())
        assert abs(share - backwards(t)) < 100 / (len(psis) - 1), \
            "the closed form for the backwards share must agree with the sweep"

        _, p1, p2 = resolve(t, 0.0)
        print(f"        on the bisector   each force gives {p1:.4f} F = F cos(t/2), "
              f"and the pair delivers {p1 + p2:.4f} F = 2F cos(t/2)")
        _, e1, e2 = resolve(t, -t / 2)
        # `signed` and not a bare format: at t = 90 the edge component is
        # u1.u2 = cos 90 deg, which lands on -1.1e-16, and "-0.0000 F" would
        # advertise a sign that is rounding and nothing else
        print(f"        at the wedge edge, w along u2   {signed(e1)} F and "
              f"{signed(e2)} F, delivering {abs(round(e1 + e2, 4)):.4f} F "
              f"= F(1 + cos t)")
        print(f"        one component points BACKWARDS over {share:.1f} % of the wedge, "
              f"fighting the other force rather than helping it")

        # and now the claim the 180 deg panel rests on, made for every t: sweep w
        # right round the circle, not merely across the wedge
        ring = np.linspace(0, 360, 2881)
        tot = np.stack([at(a) for a in ring]) @ both
        assert abs(tot.max() - np.linalg.norm(both)) < 1e-12
        print(f"        over a FULL circle of w the most the pair delivers is "
              f"{tot.max():.6f} F, at {ring[tot.argmax()]:.1f} deg -- the bisector "
              f"is 90.0")
        if np.linalg.norm(both) < 1e-9:
            assert np.abs(tot).max() < 1e-15
            print(f"        u1 + u2 is EXACTLY zero, so the sum is "
                  f"{np.abs(tot).max():.2e} F in every one of the {len(ring)} "
                  f"directions swept.")
            print(f"        Two opposed supports pushing EQUALLY deliver nothing, "
                  f"anywhere. Pushing unequally does work,")
            print(f"        but only along their own line:")
            for c1, c2 in ((1.0, 1.0), (1.4, 0.6), (2.0, 0.5)):
                net = c1 * u1 + c2 * u2
                print(f"          c = ({c1:.1f}, {c2:.1f})   net "
                      f"({net[0]:+.3f}, {net[1]:+.3f}) = {abs(c1 - c2):.3f} F along "
                      f"{'u1' if c1 > c2 else 'u2'}")
        c = rebuild(t)
        print(f"        (decomposing is the other question: to BUILD one unit along "
              f"the bisector each push must supply "
              + ("nothing that exists -- the pair is rank one)"
                 if c is None else f"{c:.4f} F, not {p1:.4f} F)"))

    print("\non the bisector, which is the best direction any pair has:")
    for t in APART:
        _, p1, p2 = resolve(t, 0.0)
        print(f"  {t:5.1f} deg   each force gives {p1:.4f} F   the pair delivers "
              f"{p1 + p2:.4f} F")


# --------------------------------------------------------------- one construction

def gauge(c):
    """Line width for a component of size `c` -- the SECOND encoding of magnitude.

    Length already carries it, the arrow being c long, and this carries it again.
    Straight proportion rather than a square root: everything in this figure lies
    between 0 and 2F, so a linear map spans the range without flattening it, and
    a component of 0.04 F is meant to come out as a hairline, because it is one.
    """
    return 0.85 + 2.55 * abs(c)


def shaft(ax, tail, tip, colour, lw, z=4, ls="-"):
    """One arrow. FancyArrowPatch rather than annotate, so the head scales with lw."""
    ax.add_patch(FancyArrowPatch(tail, tip, arrowstyle="-|>",
                                 mutation_scale=6.5 + 2.0 * lw, color=colour, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, joinstyle="miter",
                                 capstyle="butt", zorder=z))


def unitarc(ax, o, t):
    """The unit circle between the two pushes: the wedge, and the scale.

    Every force in the figure is F long, so every force's arrowhead sits on this
    arc, and it is what the length of a component is read against.
    """
    ax.add_patch(Arc(o, 2 * F, 2 * F, theta1=90 - t / 2, theta2=90 + t / 2, color=MUTED,
                     lw=0.9, alpha=.6, zorder=1))


def square(ax, foot, a, b, s=0.085):
    """The right-angle mark at the foot of a perpendicular.

    The one mark in the figure that is not a quantity. It is here because the
    whole construction turns on the drop being PERPENDICULAR to the axis, and a
    dropped line without it is only a line to somewhere.
    """
    ax.plot(*np.stack([foot + s * a, foot + s * (a + b), foot + s * b]).T, color=MUTED,
            lw=1.0, zorder=3)


def signed(x):
    """Four places, with a sign -- except on an exact zero, which has no sign.

    At 180 deg the components are cos(90 deg) = 6.1e-17, and "+0.0000 F" would be
    a sign on a quantity that has none.
    """
    return "0.0000" if abs(x) < 5e-5 else f"{x:+.4f}"


def bones(t, psi):
    """Every point one resolution construction is built from.

    Shared by the drawing and by the layout, so that a panel is sized from the
    same numbers it is drawn from and the two cannot drift apart.
    """
    w, p1, p2 = resolve(t, psi)
    u1, u2 = pushes(t)
    n = np.array([-w[1], w[0]])                          # the left-hand normal of w
    tot = p1 + p2
    # the sum is set out BESIDE the axis rather than on it: on the axis it would
    # be a third arrow lying along the two components and invisible under them.
    # The offset is whatever clears both force arrows, because any smaller one
    # crosses them -- every force leaves the same origin the axis does, so a
    # parallel line nearer than the forces' own reach must cut across one. It
    # gets its own dotted baseline and two dotted ties, which is what says the
    # black arrow is the orange chain measured over again.
    #
    # The offset is F + 0.3 in EVERY cell, which is the least that clears a force
    # arrow in the worst case -- a force square to the axis, which is what a
    # near-opposed pair gives. Making it uniform rather than tight costs a third
    # of an F in width and buys the thing the figure most needs: the black answer
    # arrow stands in the same place in all twelve cells, so the four panels can
    # be read across
    off = F + 0.30
    # when the two components have OPPOSITE signs they lie on the same stretch of
    # the axis pointing opposite ways, and head to tail draws the second exactly
    # over the first. Past 90 deg apart that is most of the wedge, and it is the
    # case the figure most needs legible -- one force fighting the other -- so
    # the second component steps aside by a tenth of an F. A step is a pure
    # translation: every length in it is still true, and a dotted connector and a
    # dotted tie put its two ends back where they belong
    step = (-0.11 * n) if p1 * p2 < -1e-18 else np.zeros(2)
    span = (min(0.0, p1, tot) - 0.30, max(F * 1.16, p1, tot) + 0.26)
    return dict(w=w, n=n, p1=p1, p2=p2, tot=tot, off=off, span=span, step=step,
                u1=u1, u2=u2)


def resolved(ax, o, t, psi):
    """One direction w, with both forces resolved onto it and the two added.

    The construction, in the order it is read: the two forces at F; the w axis;
    the perpendicular dropped from each force's tip onto that axis, with a right
    angle marked at the foot; the two components head to tail ALONG the axis, the
    first of them ending exactly on the first foot; and the total set out beside
    them, tied to the chain at both ends so the two read as one length.

    A negative component is drawn pointing backwards along w, at its true length,
    because that is what a force fighting the axis does. It is never drawn as its
    absolute value.
    """
    b = bones(t, psi)
    w, n, p1, p2, tot = b["w"], b["n"], b["p1"], b["p2"], b["tot"]
    ax.plot(*np.stack([o + b["span"][0] * w, o + b["span"][1] * w]).T, color=MUTED,
            lw=0.9, ls=(0, (1, 2.6)), zorder=1)
    ax.text(*(o + (b["span"][1] + 0.13) * w), "$w$", ha="center", va="center",
            color=INK, fontsize=10.5, zorder=8)
    unitarc(ax, o, t)
    for u, p, name in ((b["u1"], p1, "$u_1$"), (b["u2"], p2, "$u_2$")):
        shaft(ax, o, o + F * u, ORANGE, gauge(F), z=4)
        # centred above the tip rather than off to the side: a near-opposed pair
        # lays its forces along the horizontal, and a label set out sideways
        # there runs straight into the sum bar's baseline
        ax.text(*(o + 1.13 * F * u), name, ha="center", va="bottom", color=ORANGE,
                fontsize=10.5, zorder=8)
        foot = o + p * w
        drop = o + F * u - foot
        if np.linalg.norm(drop) > 1e-9:                  # w along a force: no drop
            ax.plot(*np.stack([o + F * u, foot]).T, color=MUTED, lw=0.9,
                    ls=(0, (2.2, 2.2)), zorder=2)
            towards = -np.sign(p) * w if abs(p) > 1e-9 else w
            square(ax, foot, towards, drop / np.linalg.norm(drop))
        ax.scatter(*foot, s=17, color=MUTED, zorder=3)
    # the two components, head to tail on the axis. The first ends exactly on the
    # first foot, which is the whole reason for having dropped the perpendicular
    if abs(p1) > 1e-9:
        shaft(ax, o, o + p1 * w, ORANGE, gauge(p1), z=5)
    if abs(p2) > 1e-9:
        start = o + p1 * w + b["step"]
        shaft(ax, start, start + p2 * w, ORANGE, gauge(p2), z=5)
        if b["step"].any():
            ax.plot(*np.stack([o + p1 * w, start]).T, color=MUTED, lw=0.8,
                    ls=(0, (1, 2)), zorder=2)
            ax.plot(*np.stack([start + p2 * w, o + tot * w]).T, color=MUTED, lw=0.8,
                    ls=(0, (1, 2)), zorder=2)
    base = o - b["off"] * n
    ax.plot(*np.stack([base + b["span"][0] * w, base + b["span"][1] * w]).T,
            color=MUTED, lw=0.7, ls=(0, (1, 2.6)), zorder=1)
    for tie in (0.0, tot):
        ax.plot(*np.stack([o + tie * w, base + tie * w]).T, color=MUTED, lw=0.9,
                ls=(0, (1, 2.6)), zorder=1)
    if abs(tot) > 1e-9:
        shaft(ax, base, base + tot * w, INK, gauge(tot), z=6)
    else:
        # 180 deg: the two components cancel exactly and there is no arrow left
        # to draw. A ring on the spot says so, where a zero-length arrow would
        # simply be missing
        ax.scatter(*base, s=115, facecolor=PAPER, edgecolor=INK, lw=1.6, zorder=6)
    return (f"$w$ {'on the bisector' if abs(psi) < 1e-9 else f'{abs(psi):.4g}° off it'}",
            f"$F(u_1\\!\\cdot\\!w)$ = {signed(p1)} F",
            f"$F(u_2\\!\\cdot\\!w)$ = {signed(p2)} F",
            f"adding to {abs(round(tot, 4)):.4f} F")


def fanned(ax, o, t, n=7):
    """The same sum as w sweeps the whole wedge, and the curve of it.

    2F cos(t/2) cos(psi) is a cosine in psi, so the tips of the totals trace a
    CIRCLE of diameter 2F cos(t/2) sitting on the bisector -- the polar curve
    r = A cos(psi). Drawing the curve as well as the arrows means the panel
    answers for every direction between the pushes and not only for the nine that
    are drawn. At 180 deg that circle has diameter zero: it is a point, and there
    are no arrows to draw at all.
    """
    psis = np.linspace(-t / 2, t / 2, 361)
    curve = o + delivered(t, psis)[:, None] * np.stack([at(90 + p) for p in psis])
    unitarc(ax, o, t)
    for u, name in ((pushes(t)[0], "$u_1$"), (pushes(t)[1], "$u_2$")):
        shaft(ax, o, o + F * u, ORANGE, gauge(F), z=7)
        ax.text(*(o + 1.13 * F * u), name, ha="center", va="bottom", color=ORANGE,
                fontsize=10.5, zorder=8)
    for psi in np.linspace(-t / 2, t / 2, n + 2)[1:-1]:
        w = at(90 + psi)
        ax.plot(*np.stack([o, o + 1.12 * F * w]).T, color=MUTED, lw=0.6,
                ls=(0, (1, 3.4)), zorder=1)
        s = delivered(t, psi)
        if abs(s) > 1e-9:
            # a much lighter weight than `gauge`: seven totals radiating from one
            # origin at full weight close up into a solid black wedge, and this
            # cell is here to show the SHAPE of the sweep, not to be measured
            shaft(ax, o, o + s * w, INK, 0.65 + 0.85 * abs(s), z=6)
    top, edge = delivered(t, 0.0), delivered(t, t / 2)
    if top > 1e-9:
        ax.plot(*curve.T, color=INK, lw=1.1, zorder=5)
    else:
        ax.scatter(*o, s=46, color=INK, zorder=6)
    if top < TINY:
        # at 175 and 180 deg the whole locus is smaller than an arrowhead. The
        # ring says where to look, which is more honest than drawing it any
        # bigger than it is
        ax.add_patch(Circle(o, RING, facecolor="none", edgecolor=INK, lw=1.0,
                            ls=(0, (2, 2)), zorder=5))
    return ("$w$ swept across the wedge",
            "each total $2F\\cos(t/2)\\cos\\psi$",
            f"mid {top:.4f} F, edge {edge:.4f} F",
            "the tips trace a circle" if top > 1e-9 else "the circle is a POINT")


# --------------------------------------------------------------------- the layout

def R(t, psi):
    return {"kind": "resolve", "t": t, "psi": psi, "lines": 4}


def Fan(t):
    return {"kind": "fan", "t": t, "lines": 4}


def reach(c):
    """Everything one cell covers, as a box relative to its own origin.

    The panels are laid out and then SIZED from these, so changing an angle or a
    sample re-flows the figure instead of breaking it.
    """
    t = c["t"]
    arc = np.stack([at(a) for a in np.linspace(90 - t / 2, 90 + t / 2, 181)])
    if c["kind"] == "fan":
        psis = np.linspace(-t / 2, t / 2, 361)
        pts = np.vstack([[[0, 0]], 1.22 * F * arc,
                         delivered(t, psis)[:, None] * np.stack([at(90 + p)
                                                                 for p in psis])]
                        + ([[[-RING, -RING], [RING, RING]]]      # the callout ring
                           if delivered(t, 0.0) < TINY else []))
    else:
        b = bones(t, c["psi"])
        w, n = b["w"], b["n"]
        pts = np.vstack([[[0, 0]], 1.22 * F * arc,
                         [b["span"][0] * w, (b["span"][1] + 0.24) * w,
                          b["p1"] * w, b["tot"] * w, -b["off"] * n,
                          -b["off"] * n + b["tot"] * w,
                          b["p1"] * w + b["step"], b["tot"] * w + b["step"]]])
    return pts.min(axis=0), pts.max(axis=0)


def place(bands, gx, gy, scale):
    """Origins for every cell, and the box the panel needs, in data units.

    Bands are laid out one above another and the cells inside a band left to
    right, each standing on its band's baseline. Labels hang below their cell.

    `scale` is data units per inch, so a label's allowance is a fixed PHYSICAL
    size: the text comes out the same size on the page whatever a panel's scale.
    """
    lab_h, nudge = 0.235 * scale, 0.12 * scale
    laid, y = [], 0.0
    for band in bands:
        boxes = [reach(c) for c in band]
        deep = max(hi[1] - lo[1] for lo, hi in boxes)
        base, x = y - deep, 0.0
        for c, (lo, hi) in zip(band, boxes):
            laid.append([c, np.array([x - lo[0], base - lo[1]]),
                         np.array([x - lo[0] + (lo[0] + hi[0]) / 2, base - nudge])])
            x += (hi[0] - lo[0]) + gx
        y = base - lab_h * max(c["lines"] for c in band) - nudge - gy
    lo = np.min([o + reach(c)[0] for c, o, _ in laid], axis=0)
    hi = np.max([o + reach(c)[1] for c, o, _ in laid], axis=0)
    lo[1] = min(p[1] - lab_h * c["lines"] for c, _, p in laid)
    return laid, lo, hi


def draw(ax, laid):
    """Draw every cell where `place` put it, and hang its label under it."""
    for c, o, p in laid:
        if c["kind"] == "fan":
            lines, colour = fanned(ax, o, c["t"]), INK
        else:
            lines, colour = resolved(ax, o, c["t"], c["psi"]), MUTED
        ax.text(*p, "\n".join(lines), color=colour, fontsize=SMALL, linespacing=LINES,
                ha="center", va="top", zorder=8)


# ---------------------------------------------------------------------- the page

def told(t):
    """A panel's title, and the one line it spends on the other question.

    Both are built from numbers computed here and now, so a panel cannot end up
    captioned with a figure it is not drawing.
    """
    _, p1, p2 = resolve(t, 0.0)
    said = (f"{t:.4g}° apart — {p1:.4f} F from each force, "
            f"{p1 + p2:.4f} F delivered along the bisector")
    c = rebuild(t)
    if c is None:
        note = ("The pair is not useless: pushing UNEQUALLY, $c$ = (1.4, 0.6), still "
                f"drives {abs(1.4 - 0.6):.3f} F\nalong the line. It is pushing them "
                "EQUALLY that delivers nothing, in every direction.\n"
                "And resolving is not decomposing: no coefficients build a force OFF "
                "the line at all.")
    else:
        note = ("Resolving is not decomposing. To BUILD one unit along the bisector "
                "each push must\n"
                f"SUPPLY $1/(2\\cos(t/2))$ = {c:.4f} F, not the {p1:.4f} F it "
                "contributes here.\nThe two agree only where "
                "$\\cos(t/2) = 1/(2\\cos(t/2))$ — which is to say only at 90°.")
    return said, note


def figure():
    """Four panels, two by two, all four at one scale.

    Each panel holds the same three cells -- the bisector, a direction off it,
    and the whole sweep -- so the four read across as one argument. Their
    contents come out very nearly the same size, so rather than let each take its
    own scale they are all given the largest box any of them needs, and an arrow
    in one panel can be compared with an arrow in another directly. Nothing is
    set by hand: the box is measured and the page follows it.
    """
    box = (PAGE - COLGAP) / 2
    plans = {}
    for t in APART:
        # a label's allowance is physical, so it depends on the panel's scale,
        # which depends on the width the layout comes out at, which depends on
        # the allowance. Three passes settle it to well under a pixel
        bands = [[R(t, 0.0), R(t, -t / 4), Fan(t)]]
        scale = 1.0
        for _ in range(3):
            laid, lo, hi = place(bands, 0.80, 0.0, scale)
            scale = (hi[0] - lo[0]) / box
        plans[t] = [laid, lo, hi]

    wide = max(hi[0] - lo[0] for _, lo, hi in plans.values())
    deep = max(hi[1] - lo[1] for _, lo, hi in plans.values())
    foot_h = 3.5 * 0.235 * (wide / box)                  # two lines of panel note
    for t in APART:                                      # centre each in the common box
        laid, lo, hi = plans[t]
        lo = lo - [(wide - (hi[0] - lo[0])) / 2, deep - (hi[1] - lo[1])]
        plans[t] = [laid, np.array([lo[0], lo[1] - foot_h]),
                    np.array([lo[0] + wide, lo[1] + deep])]

    hgt = box * (deep + foot_h) / wide
    head, foot = 1.34, 0.12                              # inches of furniture
    tall = head + foot + 2 * hgt + ROWGAP
    fig = plt.figure(figsize=(PAGE + 2 * MARGIN, tall), dpi=210, facecolor=PAPER)
    fig.text(0.5, 1 - 0.26 / tall,
             "Two supports, both pushing at $F$: how much of it arrives along a "
             "direction between them",
             ha="center", va="top", color=INK, fontsize=15)
    fig.text(0.5, 1 - 0.52 / tall,
             "orange: the two forces, and each one's component along $w$ — the foot of "
             "the perpendicular dropped from its tip onto the $w$ axis.\n"
             "Long AND thick when large, and pointing backwards when negative.      "
             "black: the two components added, which is all the pair delivers along "
             "$w$.\nEvery force is $F$ long, so the arc through the two arrowheads is "
             "one $F$ from the origin.",
             ha="center", va="top", color=MUTED, fontsize=9.6, linespacing=1.7)

    for k, t in enumerate(APART):
        laid, lo, hi = plans[t]
        col, row = k % 2, k // 2
        left = MARGIN + col * (box + COLGAP)
        bottom = tall - head - (row + 1) * hgt - row * ROWGAP
        ax = fig.add_axes([left / (PAGE + 2 * MARGIN), bottom / tall,
                           box / (PAGE + 2 * MARGIN), hgt / tall])
        ax.set_facecolor(PAPER)
        draw(ax, laid)
        said, note = told(t)
        ax.text((lo[0] + hi[0]) / 2, lo[1] + foot_h * 0.88, note, ha="center", va="top",
                color=MUTED, fontsize=SMALL, linespacing=1.7, zorder=8)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(said, color=INK, fontsize=11.2, pad=5)
    return fig, wide


def main():
    audit()
    fig, wide = figure()
    fig.savefig(OUT, facecolor=PAPER)
    plt.close(fig)
    print(f"\nall four panels are drawn at one scale, {wide:.2f} F across half a page")
    print(f"\n{OUT}")


if __name__ == "__main__":
    main()
