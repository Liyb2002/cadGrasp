"""The right-angle case, drawn large: what two perpendicular pushes cost, and what they can do.

This is the reference panel the other projection figures lean on, so it is
deliberately unhurried -- two panels, one scale, and nothing on either that is
not load-bearing.

Two contacts push along u1 and u2, exactly 90 deg apart. Neither can pull. A
force asked for along their 45 deg bisector is built from them by the ordinary
parallelogram, which at this angle is a SQUARE, and the two facts the figure
exists to hold side by side are:

    a unit along the bisector costs 0.7071 from each support -- 1.4142 in TOTAL,
    more than the one unit it delivers (the triangle inequality); and yet

    with each support capped at F, the bisector is the direction the pair can
    push HARDEST in, sqrt(2) F = 1.4142 F, while the push directions themselves
    are the WEAKEST at exactly F.

Those two read like a contradiction and are not, and the reconciliation is one
line: spreading a demand over two supports raises the SUM the supports pay
(1.0000 -> 1.4142) and lowers the LARGEST single share (1.0000 -> 0.7071). The
bill is the sum; a per-support cap binds on the max. Panel 1 draws the sum,
panel 2 draws the max, and they share a scale so the 1.0000 of one and the
1.4142 of the other can be compared by eye.

The second thing this panel is for: at 90 deg, and at NO other angle, the
orthogonal projection -- the plain dot product u_i . f -- IS the decomposition.
That is why everyone's intuition about resolving forces is built on this case,
and it is exactly the step that breaks when the pushes are not square. The
figure draws it as the dropped perpendicular whose foot lands precisely on the
component arrow; `main` verifies it numerically over a sweep of targets and
prints how badly the same move fails at 60 and 120 deg.

Magnitude is encoded TWICE, in arrow LENGTH and in arrow WIDTH, so that a big
force is a long thick arrow and a small one a short thin arrow, and the reader
never has to read a number to see which of two arrows is larger.

Nothing here is quoted from memory: every number in the figure and in the
printout is computed at run time by the functions below. There is no randomness
anywhere, so a clean re-run reproduces the PNG byte for byte.

    python slides/tools/projection_right_angle.py   ->  slides/tools/figures/projection_right_angle.png
"""
from __future__ import annotations

from common import figure_path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch, Polygon  # noqa: E402

# the house palette, unchanged: orange is what a contact SUPPLIES, blue is what
# is DEMANDED of it, muted is construction, ink is text and hard boundaries
INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
BLUE, ORANGE = "#2563EB", "#E08A24"

OUT = figure_path('projection_right_angle.png')

# half the angle between the pushes. 45 puts them 90 deg apart and stands the
# bisector straight up the page, which buys the figure its mirror symmetry: the
# two components are equal, so the reader sees the equality instead of reading it
HALF = 45.0
CAP = 1.0                                                  # the per-support cap, F


def unit(deg):
    """A unit vector at `deg` measured from the +x axis, counter-clockwise."""
    t = np.deg2rad(deg)
    return np.array([np.cos(t), np.sin(t)])


U1, U2 = unit(90 - HALF), unit(90 + HALF)                  # the two push directions
BIS = unit(90.0)                                           # their bisector, straight up
BASIS = np.column_stack([U1, U2])                          # columns, so BASIS @ [a,b] = force


# ------------------------------------------------------------------- the numbers


def decompose(f):
    """The two coefficients that BUILD f out of the pushes: solve BASIS @ c = f.

    This is the physical question -- how much does each contact have to supply --
    and it is a linear solve, not a projection. The two coincide only when the
    pushes are orthonormal, which is the whole point of this figure.
    """
    return np.linalg.solve(BASIS, f)


def project(f):
    """The two orthogonal projections, u_i . f. The intuitive answer.

    Correct at 90 deg and wrong everywhere else, because it asks each push
    separately how much of f lies along it and never checks that the answers add
    back up to f.
    """
    return BASIS.T @ f


def reach(deg_from_u1, cap=CAP):
    """The largest force obtainable along a direction when each push is capped.

    The demand scales the coefficients linearly, so the cap binds on the LARGER
    of the two: multiply the unit decomposition by cap / max(a, b) and stop.
    Coefficients are non-negative throughout the octant between the pushes, so
    no contact is ever asked to pull.
    """
    c = decompose(unit(90 - HALF + deg_from_u1))
    return cap / np.max(c)


def cost(deg_from_u1):
    """Total support force spent per unit delivered along a direction: a + b."""
    return np.sum(decompose(unit(90 - HALF + deg_from_u1)))


def apart(deg, target_from_u1=None):
    """The same two questions for a pair `deg` apart -- used only for contrast.

    Returns (decomposition, projection) of the unit vector at `target_from_u1`
    degrees off the first push, defaulting to the bisector. Nothing here is
    drawn; it exists so the claim "90 deg and only 90 deg" is a measurement in
    the printout rather than an assertion in a comment.
    """
    if target_from_u1 is None:
        target_from_u1 = deg / 2.0
    v1, v2 = unit(0.0), unit(deg)
    f = unit(target_from_u1)
    B = np.column_stack([v1, v2])
    return np.linalg.solve(B, f), B.T @ f


# ------------------------------------------------------------------- the drawing

# magnitude -> line width, in points. Linear, so that reading widths off the page
# is reading magnitudes, and pinned at the two magnitudes the figure actually
# draws: 0.7071 comes out at 3.33 pt and 1.4142 at 5.67 pt, a factor of 1.7 on a
# factor of 2 in magnitude -- enough to be unmistakable without the thin arrows
# turning into hairlines
def width(m):
    return 1.00 + 3.30 * m


# and magnitude -> arrowhead size. The head has to grow with the shaft or a thick
# arrow ends in a point narrower than itself, which reads as a stub
def head(m):
    return 14.0 + 8.5 * m


# how far to the right of an arrow tip its label sits. Enough to clear the
# construction line that leaves the same tip: at the tip of either resultant a
# 45 deg edge runs away downwards, and a label hung directly off the point has
# its second line struck through by it
LAB = 0.20


def force(ax, at, vec, color, z=10):
    """One force, drawn as an arrow whose LENGTH and WIDTH both carry its size.

    FancyArrowPatch rather than `annotate`, because annotate ties the head size
    to the text size and this figure needs the head tied to the magnitude.
    """
    m = float(np.linalg.norm(vec))
    ax.add_patch(FancyArrowPatch(at, np.asarray(at) + vec, zorder=z,
                                 arrowstyle="-|>", mutation_scale=head(m),
                                 shrinkA=0, shrinkB=0, joinstyle="miter",
                                 lw=width(m), color=color))


def right_angle(ax, at, e1, e2, s=0.085, z=6):
    """The small square that says two lines meet at 90 deg.

    Drawn at the feet of the dropped perpendiculars, which is the one place in
    the figure where the right angle is a CLAIM and not just the arrangement of
    the pushes: it is what makes the foot land on the component.
    """
    p = np.array([np.asarray(at) + s * e1,
                  np.asarray(at) + s * (e1 + e2),
                  np.asarray(at) + s * e2])
    ax.plot(p[:, 0], p[:, 1], color=MUTED, lw=1.3, zorder=z,
            solid_joinstyle="miter", solid_capstyle="butt")


def wedge(ax, a0, a1, r, label, at=(0.0, 0.0), lift=0.075):
    """An angle mark between two directions, with its size written in it."""
    ax.add_patch(Arc(at, 2 * r, 2 * r, angle=0, theta1=a0, theta2=a1,
                     color=MUTED, lw=1.3, zorder=6))
    p = np.asarray(at) + (r + lift) * unit((a0 + a1) / 2)
    ax.text(p[0], p[1], label, color=MUTED, fontsize=12, zorder=7,
            ha="center", va="center")


def tag(ax, at, text, colour, ha, gloss=None, size=15):
    """A number set beside the arrow it measures, with an optional gloss under it.

    The gloss is muted and smaller so that the numbers stay the thing the eye
    lands on: this figure is read by comparing arrow sizes, and the words are
    there to name what has been compared, not to be read first.
    """
    at = np.asarray(at, float)
    ax.text(at[0], at[1], text, color=colour, fontsize=size, zorder=7,
            ha=ha, va="center")
    if gloss is not None:
        ax.text(at[0], at[1] - 0.115, gloss, color=MUTED, fontsize=12, zorder=7,
                ha=ha, va="center")


def rays(ax, out=1.24):
    """The two lines of action, extended past the arrows.

    Without them a dropped perpendicular has nothing to be perpendicular TO, and
    the foot cannot be seen to be a foot.
    """
    for u, lab, ha in ((U1, "u₁", "left"), (U2, "u₂", "right")):
        ax.plot([0, out * u[0]], [0, out * u[1]], color=MUTED, lw=1.1, zorder=2)
        p = (out + 0.055) * u
        ax.text(p[0], p[1], lab, color=MUTED, fontsize=15, zorder=7,
                ha=ha, va="bottom")


def frame(ax, title, lines):
    """One panel's furniture: identical limits on both, so the two share a scale.

    Sharing the scale is the argument, not the styling. Panel 1's resultant is
    1.0000 long and panel 2's is 1.4142 long, and drawn at one scale that
    difference is a thing the reader sees rather than a pair of numbers they
    have to compare.
    """
    ax.set_title(title, color=INK, fontsize=15, pad=10)
    for i, s in enumerate(lines):
        ax.text(0.0, -0.30 - 0.20 * i, s, color=INK, fontsize=12.5,
                ha="center", va="center", zorder=7)
    ax.set_xlim(-1.34, 1.34)
    ax.set_ylim(-0.66, 1.56)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor(PAPER)


def built(ax, a, b):
    """Panel 1: the force asked for, and the two pushes that build it.

    The square is drawn as two orange arrows (what the contacts supply) and two
    dashed edges (construction). The dashed edges are simultaneously the
    parallelogram's far sides and the perpendiculars dropped from the tip onto
    each push line -- at 90 deg those are the same two segments, which is the
    identity the panel is here to show.
    """
    rays(ax)
    f = BASIS @ np.array([a, b])                           # the resultant they make
    p1, p2 = a * U1, b * U2                                # the two component tips

    for p in (p1, p2):                                     # the far sides of the square
        ax.plot([p[0], f[0]], [p[1], f[1]], color=MUTED, lw=1.4, zorder=3,
                ls=(0, (5.0, 3.2)))
    # the right angle at each foot. Its two arms run back down the push line and
    # along the dashed edge, so the mark sits inside the square
    right_angle(ax, p1, -U1, U2)
    right_angle(ax, p2, -U2, U1)

    force(ax, (0, 0), p1, ORANGE, z=10)
    force(ax, (0, 0), p2, ORANGE, z=10)
    force(ax, (0, 0), f, BLUE, z=11)

    wedge(ax, 90 - HALF, 90, 0.40, f"{HALF:.0f}°")
    wedge(ax, 90, 90 + HALF, 0.40, f"{HALF:.0f}°")

    # the component labels sit outside the push lines, level with the arrow tips
    for p, u, ha, c in ((p1, U1, "left", a), (p2, U2, "right", b)):
        n = np.array([u[1], -u[0]]) * (1 if ha == "left" else -1)
        # the gloss ties the number to the angle marks: the components are equal
        # and are cos of the half-angle, which is the whole of the arithmetic
        tag(ax, p + 0.135 * n, f"{c:.4f}", ORANGE, ha, f"= cos {HALF:.0f}°")
    tag(ax, f + (LAB, 0.0), f"|f| = {np.linalg.norm(f):.4f}", BLUE, "left",
        "the force wanted")
    # the drawing convention, stated once, in the dead space this panel has
    # above it. It is dead precisely because the resultant here is 1.0000 long
    # where the next panel's is 1.4142, which is the comparison the shared scale
    # is for
    ax.text(0.0, 1.30, "arrow length and arrow width both carry the magnitude",
            color=MUTED, fontsize=12.5, zorder=7, ha="center", va="center")


def capped(ax, cap=CAP):
    """Panel 2: the same pair with a cap on each, and how far that reaches.

    Every force the pair can supply is `a u1 + b u2` with 0 <= a, b <= cap, and
    that set is exactly a SQUARE standing on the origin. So the reachable
    boundary is the square's two far edges, and the reader can read the answer
    for every direction at once off the picture: the far corner, on the
    bisector, sticks out to sqrt(2) cap, and the only two directions where the
    square touches the circle of radius cap are the push directions themselves.
    """
    rays(ax)
    p1, p2 = cap * U1, cap * U2
    corner = p1 + p2

    ax.add_patch(Polygon([(0, 0), p1, corner, p2], closed=True, facecolor=ORANGE,
                         alpha=0.13, edgecolor="none", zorder=1))
    # the circle of radius cap, for contrast: what the pair would reach if a cap
    # of F meant F in every direction. It is inside the square everywhere except
    # at the two push directions, where it touches
    ax.add_patch(Arc((0, 0), 2 * cap, 2 * cap, angle=0, theta1=90 - HALF,
                     theta2=90 + HALF, color=MUTED, lw=1.5, zorder=4,
                     ls=(0, (5.0, 3.2))))
    for p in (p1, p2):                                     # the reachable frontier
        ax.plot([p[0], corner[0]], [p[1], corner[1]], color=INK, lw=2.0, zorder=5)
    right_angle(ax, p1, -U1, U2)
    right_angle(ax, p2, -U2, U1)

    force(ax, (0, 0), p1, ORANGE, z=10)
    force(ax, (0, 0), p2, ORANGE, z=10)
    force(ax, (0, 0), corner, BLUE, z=11)

    for p, u, ha in ((p1, U1, "left"), (p2, U2, "right")):
        n = np.array([u[1], -u[0]]) * (1 if ha == "left" else -1)
        tag(ax, p + 0.135 * n, f"F = {cap:.4f}", ORANGE, ha, "at its cap")
    # the sqrt(2) in this label is the 45 deg case written out longhand. HALF is
    # a free parameter for the GEOMETRY and not for the TEXT: move it and this
    # string, the panel captions and the title all have to be rewritten
    tag(ax, corner + (LAB, 0.0), f"√2 F = {np.linalg.norm(corner):.4f}", BLUE,
        "left", "the pair's strongest")
    # name the dashed arc under it and to one side: the bisector is taken by the
    # blue arrow and the frontier crowds the arc everywhere above it
    p = 0.85 * cap * unit(90 + 0.55 * HALF)
    ax.text(p[0], p[1], "radius F", color=MUTED, fontsize=12.5, zorder=7,
            ha="center", va="top")


def main():
    a, b = decompose(BIS)                                  # the unit demand, split
    pr = project(BIS)
    total = float(abs(a) + abs(b))                         # what the two supports pay

    # --- check 1: over the whole octant, is the dot product the decomposition?
    phis = np.linspace(0.0, 90.0, 901)
    err = max(float(np.max(np.abs(decompose(unit(90 - HALF + p))
                                  - project(unit(90 - HALF + p))))) for p in phis)

    # --- check 2: the capped reach across the octant
    rr = np.array([reach(p) for p in phis])
    cc = np.array([cost(p) for p in phis])

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.5), dpi=210, facecolor=PAPER)
    built(axes[0], a, b)
    frame(axes[0], f"Two pushes {2 * HALF:.0f}° apart, one unit along the bisector",
          [f"{a:.4f} + {b:.4f} = {total:.4f} for {1.0:.4f} delivered — "
           f"the triangle inequality",
           f"the perpendicular foot lands ON the component — only at {2 * HALF:.0f}°"])
    capped(axes[1])
    frame(axes[1], "The same pair, each capped at F",
          ["shaded — every force the pair can supply under the cap",
           "√2 F at the corner, on the bisector — only F along either push"])

    # the reconciliation, spelled out once, under both panels because it is the
    # one claim that needs both of them
    fig.text(0.5, 0.070,
             f"Spreading one unit over both supports raises the TOTAL paid, "
             f"{1.0:.4f} → {total:.4f}, and halves the LARGEST single share, "
             f"{1.0:.4f} → {max(a, b):.4f}.",
             color=INK, fontsize=13.5, ha="center", va="center")
    fig.text(0.5, 0.026,
             "The bill is the sum; the cap binds on the max — so the bisector is "
             "at once the most expensive direction and the strongest.",
             color=INK, fontsize=13.5, ha="center", va="center")
    fig.subplots_adjust(0.01, 0.115, 0.99, 0.935, 0.02)
    fig.savefig(OUT, facecolor=PAPER)
    plt.close(fig)

    print(f"pushes {2 * HALF:.0f} deg apart      "
          f"u1 ({U1[0]:+.4f},{U1[1]:+.4f})   u2 ({U2[0]:+.4f},{U2[1]:+.4f})   "
          f"u1.u2 = {U1 @ U2:+.3e}")
    print()
    print("A UNIT ALONG THE BISECTOR, and what it costs")
    print(f"  components                 a = {a:.4f}   b = {b:.4f}   "
          f"(cos 45 deg = {np.cos(np.deg2rad(45)):.4f})")
    print(f"  they sum in magnitude to   {total:.4f}   against the "
          f"{np.linalg.norm(BASIS @ np.array([a, b])):.4f} they deliver"
          f"   -- the triangle inequality")
    print(f"  the largest single share   {max(a, b):.4f}, DOWN from the 1.0000 "
          f"one support would carry alone")
    print()
    print("THE DOT PRODUCT IS THE DECOMPOSITION -- at 90 deg and nowhere else")
    print(f"  bisector:  solve  a = {a:.16f}   b = {b:.16f}")
    print(f"             dot    a = {pr[0]:.16f}   b = {pr[1]:.16f}")
    print(f"  over {len(phis)} targets across the octant, the largest "
          f"disagreement is {err:.3e}")
    for deg in (60.0, 90.0, 120.0):
        c, p = apart(deg)
        made = np.linalg.norm(np.column_stack([unit(0.0), unit(deg)]) @ p)
        print(f"  {deg:5.1f} deg apart, bisector:  solve {c[0]:.4f} {c[1]:.4f}   "
              f"dot {p[0]:.4f} {p[1]:.4f}   gap {np.max(np.abs(c - p)):.4f}   "
              f"-- the dot answer builds {made:.4f} of what was asked"
              + ("   (they agree)" if np.max(np.abs(c - p)) < 1e-12 else ""))
    print()
    print(f"CAPPED AT F = {CAP:.4f} EACH, what the pair can reach")
    print(f"  most            {rr.max():.4f} F at {phis[int(np.argmax(rr))]:.1f} deg "
          f"off u1 -- the bisector, and sqrt(2) = {np.sqrt(2):.4f}")
    print(f"  least           {rr.min():.4f} F at "
          f"{phis[int(np.argmin(rr))]:.1f} and "
          f"{phis[len(phis) - 1 - int(np.argmin(rr[::-1]))]:.1f} deg off u1 "
          f"-- the push directions themselves")
    print("  support spent per unit delivered (a + b):  "
          f"{cc[0]:.4f} at the pushes,  {cc[len(cc) // 2]:.4f} at the bisector")
    print(f"  the cap binds at 1 / {max(a, b):.4f} = {1 / max(a, b):.4f} F, "
          f"which is the same {np.sqrt(2):.4f}")
    print()
    print(f"widths drawn: {width(a):.2f} pt at {a:.4f}, {width(1.0):.2f} pt at "
          f"1.0000, {width(np.sqrt(2)):.2f} pt at {np.sqrt(2):.4f}")
    print(OUT)


if __name__ == "__main__":
    main()
