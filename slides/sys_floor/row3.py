r"""Row (3) in three steps, in `slides/setup/equations/three_equations.png`'s own symbols.

    python slides/sys_floor/row3.py  ->  slides/sys_floor/row3.png

`three_equations.png` writes row (3) as an integral, which is right and is not
readable out loud.  This page is the same row said in three steps, and it takes
the other page's names unchanged -- `F_push`, `q`, `mg`, `z`, `c`, `sys_floor`
-- so a reader carries one vocabulary across both. `F_push` is the complete
force vector, with fixed magnitude |F_push| = K mg = 0.5 mg. This page explains
the coplanar construction in the drawing and checks the workpiece and supports
as one assembly.

STEP 2 IS WRITTEN IN TWO PIECES, `p = X + tW` and then `t`, and it is not
collapsed into one.  Collapsed, it reads as a formula to be trusted; split, it
reads as what it is -- a point on a line, and the one number that says how far
along.  The minus sign is then `t`'s own and needs no accounting for.

IT IS AN INTERSECTION, AND NOTHING MORE.  Two loads, two lines of action, one
crossing point; the resultant acts along the line through that crossing; extend
it to `z = 0`.  That is the whole of step 2, and step 2 is the same operation as
the one that finds any line's floor crossing -- it is used once, not twice.

WHAT THAT REPLACED, and why the replacement is the honest one.  An earlier cut
gave step 2 as a lever rule -- slide each load down its own line to the floor at
`a` and `b`, drop the horizontal parts, see-saw the two vertical ones:

    p = (w_a a + w_b b) / (w_a + w_b),   w_a = mg,   w_b = -F_push . z

That is the same point -- `scene` asserts it to 1e-12 -- and it is three ideas
where one will do.  It also needed two new symbols, and `a` was the first thing
a reader asked about: it is a POINT, and the formula opened `mg . a`, a weight
times a position, which reads as a MOMENT.  The intersection needs no `a`, no
`b`, and no weights.

The lever rule is still worth having somewhere, because it reads the BEHAVIOUR
off at a glance -- no push and `p` sits under the weight; push down and it
slides toward the push; push up and the second weight goes NEGATIVE, so `p`
leaves the segment entirely -- and `slides/setup/equations/equations.md` keeps it for that.
(`coverage.md` used to, and is deleted with its page.)

ONE CAVEAT, and it is not on the page.  In THREE dimensions two lines of action
generally do not meet; the pair then reduces to a wrench and "the resultant's
line" does not exist.  `p` is still exactly the formula above and it is still
the centre of pressure -- it is the CONSTRUCTION that needs the two lines
coplanar.  This picture is a plane figure and the construction is exact in it.
So: "the load's line of action" beside a plane figure, "the centre of pressure"
in general.

ALSO NOT ON THE PAGE: the floor has to absorb the load's HORIZONTAL part, which
it can only do by friction.  Step 2 does not care -- friction acts in `z = 0`,
so its moment about a horizontal axis is zero and it cannot move `p` -- but step
3 is not the whole of the assembly's equilibrium.

NO PROSE ON THE SHEET.  Three numbered equations, the quantifier they hold
under, and a picture of four things: the push, the weight, the force they add up
to, and that force's line taken down to the floor.  Everything the earlier cuts
explained in words -- what `X` is, what `W` is, what happens as `W . z` goes to
zero, what the two lines of action are -- is either read off the picture or is
in this docstring.  A slide is read in ten seconds and a paragraph on it is not.

WHAT CAME OFF THE PICTURE, and why none of it is missed: the two dashed lines of
action and the marked crossing `X` (the two arrows already meet there, and the
resultant is drawn FROM that meeting), the letters `c` and `q` (the arrows start
at them), and the orange footprint segment with its label (step 3 is a sentence
about `p`, and the picture's job is to produce `p`).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import FancyArrow, Polygon           # noqa: E402
from support_polygon import K                                # noqa: E402

OUT = Path(__file__).resolve().parent / "row3.png"
INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
LOAD, PUSH, FOOT = "#b02a26", "#1b1b1a", "#db8a24"
EQ, LAB, SUB = 27, 19, 18

MG = 1.0                       # weight; the process has magnitude K * MG
TILT = np.radians(20.0)         # the block, tipped onto one corner
LEAN = np.radians(12.0)         # the push, off the face normal and inside
                                # METHOD s1's 15 deg cone: straight down the
                                # normal the crossing sat almost on the plumb
                                # line and the construction was invisible


def scene():
    """The picture's numbers, computed and not drawn by hand."""
    R = np.array([[np.cos(TILT), -np.sin(TILT)],
                  [np.sin(TILT), np.cos(TILT)]])
    part = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) @ R.T
    c = part.mean(axis=0)
    e0, e1 = part[3], part[2]                    # the face turned up and left
    q = e0 + 0.75 * (e1 - e0)
    d = -np.array([-(e1 - e0)[1], (e1 - e0)[0]])
    d /= np.linalg.norm(d)                       # into the surface
    d = np.array([[np.cos(LEAN), -np.sin(LEAN)],
                  [np.sin(LEAN), np.cos(LEAN)]]) @ d

    F_push = K * MG * d                         # the complete force vector
    W = np.array([0.0, -MG]) + F_push            # the resultant
    X = q + ((c[0] - q[0]) / d[0]) * d           # gravity's line is x = c_x
    p = X - (X[1] / W[1]) * W                    # and down to z = 0
    assert abs(p[1]) < 1e-12

    # THE LEVER RULE AGREES, and this assert is now the ONLY place that is
    # recorded: the page no longer says it, so nothing else would catch a drift
    a, b = np.array([c[0], 0.0]), q - (q[1] / d[1]) * d
    wa, wb = MG, -F_push[1]
    assert abs(p[0] - (wa * a[0] + wb * b[0]) / (wa + wb)) < 1e-12
    return part, c, q, d, X, W, p


def picture(ax, part, c, q, d, X, W, p):
    r"""Four things: the push, the weight, what they add up to, and where that
    goes.

    THE TWO LOADS ARE DRAWN WHERE THEY ACT -- the push landing ON THE SURFACE at
    `q`, the weight starting at `c` -- and both in ink, because they are the
    given.  Only the resultant is red, and only its continuation to the floor is
    dashed: solid is a force, dashed is the same force's line carried on to
    `z = 0`, which is the one construction on the sheet.

    Two earlier cuts got this wrong in ways worth naming.  One drew both loads
    as arrows POINTING AT `X`, which put the push in mid-air and drew gravity
    UPWARD -- directions are not a drafting choice.  The other slid both loads
    to `X`: correct, since a force is a sliding vector, and still wrong for this
    page, because a reader has to see the process touching the part.

    `X` IS DEFINED BY THE PICTURE AND NOT BY A SENTENCE.  Step 2's formula
    names it, and for one cut nothing on the sheet said what it was -- the
    legend line went out with the rest of the prose and left an undefined symbol
    in a formula, which is a broken figure.  The fix is not a sentence: the two
    loads' lines of action are drawn as thin dashes, and where they cross IS
    `X`.  Two lines cost less than a clause and they cannot be misread.

    `conv` IS SPELLED OUT, and the picture is left alone.  `X` needed a picture
    because it is a CONSTRUCTION -- a crossing, which two lines show and no word
    does.  `convex hull` is not a construction, it is a NAME, and an abbreviated
    name is the only thing that was wrong with it: a reader took `conv` for
    `contained`, which the line already says with its own symbol.  Drawing the
    footprint to explain a word would have been the wrong repair, and it was
    tried first.

    Why a HULL, for the record and not for the sheet: the floor can only push,
    so its reaction is a non-negative combination of the contacts, and a
    non-negative combination that sums to one is exactly a convex one.

    AND IT IS SET UPRIGHT, `\mathrm{X}`.  In mathtext an italic capital X is
    read as a lowercase x beside an italic lowercase `p`, and the first reader
    asked which was which.
    """
    ax.axhline(0, color=MUTED, lw=1.4, zorder=1)
    ax.add_patch(Polygon(part, closed=True, facecolor="#eceae6",
                         edgecolor="#b9b6ae", lw=1.4, zorder=2))


    def arrow(tail, vec, col, lw=0.012):
        ax.add_patch(FancyArrow(*tail, *vec, width=lw, head_width=0.066,
                                head_length=0.080, length_includes_head=True,
                                color=col, zorder=6))

    # the two lines of action, thin: where they cross IS `X`, and that is the
    # only definition of `X` the sheet carries
    ax.plot([c[0], c[0]], [0.02, X[1] + 0.30], color="#9a9a94", lw=1.0,
            ls=(0, (4, 4)), zorder=3)
    ax.plot(*np.stack([q - 0.86 * d, q + 0.30 * d]).T, color="#9a9a94", lw=1.0,
            ls=(0, (4, 4)), zorder=3)

    G = np.array([0.0, -0.46])                     # the weight, at c, DOWNWARD
    arrow(c, G, PUSH)
    arrow(q - 0.50 * d, 0.50 * d, PUSH)            # the push, landing ON q
    n = W / np.linalg.norm(W)
    arrow(X, 0.58 * n, LOAD, lw=0.017)             # the resultant, from X
    ax.plot(*np.stack([X + 0.58 * n, p]).T, color=LOAD, lw=2.0,
            ls=(0, (6, 5)), zorder=5)              # its line, on to the floor
    ax.plot([p[0]], [0], "o", ms=10, color=LOAD, zorder=7)

    ax.text(*(c + 0.55 * G + [-0.06, 0]), r"$-mg\,\hat{z}$", color=PUSH,
            fontsize=SUB, ha="right", va="center")
    ax.text(*(q - 0.56 * d + [-0.25, 0.04]), r"$F_{\rm push}$",
            color=PUSH, fontsize=SUB, ha="center", va="bottom")
    ax.text(*(X + 0.30 * n + [0.08, 0]), r"$W$", color=LOAD, fontsize=EQ,
            ha="left", va="center")
    # `X` sits a hair up-left of `q`, so its label goes further out still: on
    # the short offset it landed on the push's own arrowhead
    ax.plot([X[0]], [X[1]], "o", ms=8, color=LOAD, zorder=7)
    ax.text(X[0] - 0.16, X[1] + 0.09, r"$\mathrm{X}$", color=LOAD, fontsize=SUB,
            ha="right", va="bottom")
    ax.plot([X[0] - 0.14, X[0] - 0.03], [X[1] + 0.08, X[1] + 0.015],
            color=LOAD, lw=1.0, zorder=6)
    ax.text(p[0] + 0.09, 0.05, r"$p$", color=LOAD, fontsize=EQ, ha="left",
            va="bottom")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.52, 1.34); ax.set_ylim(-0.26, 1.98)


def main():
    part, c, q, d, X, W, p = scene()

    fig = plt.figure(figsize=(14.0, 6.6), dpi=200, facecolor=PAPER)
    t = lambda y, s, **k: fig.text(k.pop("x", 0.03), y, s, ha="left",
                                   va="center", color=k.pop("c", INK),
                                   fontsize=k.pop("fs", EQ), **k)
    NUM_X, EQ_X = 0.030, 0.085

    t(0.930, "The workpiece and its supports do not tip over", fs=32)
    for y, num, eq in (
            (0.720, "1", (r"$W \;=\; -mg\,\hat{z} \;+\; F_{\rm push}$",)),
            (0.500, "2", (r"$p \;=\; \mathrm{X} \;+\; t\,W$",
                          r"$t \;=\; -\,\frac{\mathrm{X} \cdot \hat{z}}"
                          r"{W \cdot \hat{z}}$")),
            (0.245, "3", (r"$p \;\in\; \mathrm{convex\ hull}\;"
                          r"(\mathrm{sys\_floor})$",))):
        t(y, num, x=NUM_X, fs=EQ, c=MUTED)
        for i, line in enumerate(eq):
            t(y - 0.108 * i, line, x=EQ_X, fs=EQ)
    t(0.095, r"for every $q$ and admissible $F_{\rm push}$, "
             rf"$|F_{{\rm push}}| = {K:g}\,mg$", x=EQ_X, c=MUTED, fs=SUB)

    picture(fig.add_axes([0.475, 0.075, 0.50, 0.80]), part, c, q, d, X, W, p)

    fig.savefig(OUT, facecolor=PAPER)
    plt.close(fig)
    im = plt.imread(str(OUT))
    ink = (im[:, :, :3] < 0.96).any(axis=2)
    r_, c_ = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    matplotlib.image.imsave(str(OUT), im[max(r_.min() - 30, 0):r_.max() + 31,
                                         max(c_.min() - 30, 0):c_.max() + 31])
    print(f"drawn at K = {K}: the lines cross at X = "
          f"({X[0]:.4f}, {X[1]:.4f}), W = ({W[0]:.4f}, {W[1]:.4f}), and the "
          f"line meets the floor at p = {p[0]:.4f}")
    print("   the lever rule gives the same p to 1e-12 (asserted in `scene`)")
    print(f"{OUT}  md5 {hashlib.md5(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    # The current presentation uses a fixed B/pose_2 snapshot. Preserve the
    # historical experiment and its library functions behind an explicit flag.
    import runpy
    import sys
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy')
        main()
    elif len(sys.argv) > 1:
        raise SystemExit('Use slides/render.py for current figures; --legacy enables historical options.')
    else:
        runpy.run_path(str(Path(__file__).with_name('presentation.py')), run_name='__main__')
