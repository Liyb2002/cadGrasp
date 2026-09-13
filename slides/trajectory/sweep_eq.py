r"""slides/trajectory -- the support slides in: one definition, one condition.

    python slides/trajectory/sweep_eq.py  ->  slides/trajectory/sweep_eq.png

    Sweep(supp, a)  =  { x - t a : x in supp, t >= 0 }
    Sweep(supp, a)  and  interior(obj)  are disjoint

That is the whole page.  A support `supp` is a solid standing on the floor; it
is slid into place along the floor in one horizontal direction `a`.  The first
line is the space it sweeps on the way -- the support and everything behind it
along `a` -- and the second says that space never enters the workpiece `obj`,
held at its target pose.  Touching the surface at the end is allowed: that is
the pad.

WHY `interior(obj)` AND NOT `obj` (asked, 2026-08-30, and now answered ON the
page).  `obj` is closed, so it contains its own surface, and `Sweep cap obj =
empty` would rule out every support that actually touches the workpiece --
which is every support.  THIS PAGE'S OWN PICTURE would be the counterexample:
`supp`'s left face lies on the line of the block's right face and the two share
a segment, printed on every run.  Taking the interior hands the boundary back:
touch, do not cross.  It does not go the other way round either -- `int(Sweep)
cap int(obj) = empty` is too weak, since it lets the sweep's own BOUNDARY pass
through the workpiece, a zero-thickness slice straight through the part.  The
closed sweep against the open workpiece is the one that means no penetration,
and it is what `sweep_demo.py` measures: the exact boolean VOLUME, which is 0
for two solids meeting on a face.

THE NAMES ARE `three_equations.png`'S OWN (owner, 2026-08-30; this page said
`S` and `W` before).  `supp` and `obj` are the words that page's domains are
built out of -- `supp_obj`, `sys_floor` -- so the two pages now name the same
two bodies the same way.  `W` in particular had to go: in METHOD s2 `W` is the
WORK REGION, the patch the process reaches and the supports must keep off, so
a `W` meaning the whole workpiece was a collision with the one symbol this
project most needs to stay sharp.

BUT THE DIRECTION STAYS `a`, AND NOT `t`.  `t` is already in the line, as the
scalar that runs from 0 to infinity, and it is the letter every reader takes
for a scalar parameter besides.  `x - t a` with the two swapped reads `x - s t`
-- a product of two scalars, on the page whose whole point is that one of them
is a direction.  Renaming both is possible and buys nothing; `a` for the
approach it is.

TWO LINES AND NOT ONE.  Folding the definition into the condition gives
`{ x - t a : ... } cap int(obj) = empty`, one line with no name in it -- and
then the picture's strip would carry a label the page never defines.  The
definition is the first line so that `Sweep(supp, a)` on the strip means
something.

THE PICTURE.  `order_book.py`'s tipped block for `obj`, on its corner on the
floor; a trapezoid `supp` on the floor whose left face lies on the line of the
block's right face, so it touches along a segment and cannot cross; the arrow
`a` along the floor toward the block; and the swept region as a pale strip
from `supp` back to the panel edge along `-a`.  Its geometry is computed from
the block's face line, not drawn by hand.  Same page style as
`order_book.py` (deleted 2026-09-01); same names as `slides/setup/equations/three_equations.png`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import FancyArrow, Polygon           # noqa: E402

OUT = Path(__file__).resolve().parent / "sweep_eq.png"
INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
EQ, LAB, SUB = 27, 19, 18

TILT = np.radians(20.0)         # the block, tipped onto one corner (row3's)
H_SUPP = 0.70                   # the support's height
X_SUPP = 1.60                   # the support's back face
X_EDGE = 2.80                   # the panel edge the strip runs to


def scene():
    """The picture's numbers, computed and not drawn by hand."""
    R = np.array([[np.cos(TILT), -np.sin(TILT)],
                  [np.sin(TILT), np.cos(TILT)]])
    part = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) @ R.T
    c = part.mean(axis=0)
    e0, e1 = part[2], part[1]                    # the right face, top to bottom
    u = (e0 - e1) / np.linalg.norm(e0 - e1)      # up the face
    foot = e1 - (e1[1] / u[1]) * u               # the face line, at the floor
    top = foot + (H_SUPP / u[1]) * u             # the face line, at H_SUPP
    supp = np.array([foot, [X_SUPP, 0.0], [X_SUPP, H_SUPP], top])
    sweep = np.array([foot, [X_EDGE, 0.0], [X_EDGE, H_SUPP], top])
    a = np.array([-1.0, 0.0])                    # into place: toward the block
    assert e1[1] < H_SUPP, "the support is below the block's corner: no pad"
    return part, c, supp, sweep, a, (e1, top)


def picture(ax, part, c, supp, sweep, a, pad):
    """The floor, the block `obj`, the support `supp` at its final place, the
    arrow `a`, and what `supp` swept on the way."""
    ax.add_patch(Polygon(sweep, closed=True, facecolor="#cfcbc3", alpha=0.5,
                         edgecolor="none", zorder=0.5))
    ax.axhline(0, color=MUTED, lw=1.4, zorder=1)
    ax.add_patch(Polygon(part, closed=True, facecolor="#eceae6",
                         edgecolor="#b9b6ae", lw=1.4, zorder=2))
    ax.add_patch(Polygon(supp, closed=True, facecolor="#d9d6cf",
                         edgecolor="#8f8c84", lw=1.4, zorder=3))

    tail = np.array([2.35, -0.12])
    ax.add_patch(FancyArrow(*tail, *(0.60 * a), width=0.012, head_width=0.066,
                            head_length=0.080, length_includes_head=True,
                            color=INK, zorder=6))
    ax.text(tail[0] + 0.06, tail[1], r"$a$", color=INK, fontsize=SUB,
            ha="left", va="center")

    ax.text(c[0], c[1], r"$\mathrm{obj}$", color=INK, fontsize=SUB,
            ha="center", va="center")
    s = supp.mean(axis=0)
    ax.text(s[0] + 0.04, s[1], r"$\mathrm{supp}$", color=INK, fontsize=SUB,
            ha="center", va="center")
    ax.text(0.5 * (X_SUPP + X_EDGE), 0.5 * H_SUPP,
            r"$\mathrm{Sweep}(\mathrm{supp}, a)$",
            color=MUTED, fontsize=SUB, ha="center", va="center", zorder=7)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.55, X_EDGE); ax.set_ylim(-0.32, 1.42)


def main():
    part, c, supp, sweep, a, pad = scene()

    fig = plt.figure(figsize=(14.0, 6.6), dpi=200, facecolor=PAPER)
    t = lambda y, s, **k: fig.text(k.pop("x", 0.03), y, s, ha="left",
                                   va="center", color=k.pop("c", INK),
                                   fontsize=k.pop("fs", EQ), **k)
    t(0.930, "The support slides in", fs=32)
    t(0.865, "along the floor, in one straight line", c=MUTED, fs=SUB)
    t(0.760, r"$\mathrm{Sweep}(\mathrm{supp}, a) \;=\; \{\, x - t\,a \;:\;"
             r" x \in \mathrm{supp},\; t \geq 0 \,\}$", c=INK, fs=EQ)
    t(0.655, r"$\mathrm{Sweep}(\mathrm{supp}, a) \;\cap\;"
             r" \mathrm{interior}(\mathrm{obj}) \;=\; \emptyset$", c=INK, fs=EQ)
    t(0.560, "what it sweeps on the way never enters the workpiece", c=MUTED,
      fs=SUB)
    t(0.495, r"$\mathrm{interior}$, not $\mathrm{obj}$: the pad ends up touching",
      c=MUTED, fs=SUB)
    t(0.040, r"for every support  ·  $a$ horizontal,  $\mathrm{obj}$ at the "
             r"target pose", c=MUTED, fs=SUB)

    picture(fig.add_axes([0.44, 0.02, 0.55, 0.50]), part, c, supp, sweep, a,
            pad)

    fig.savefig(OUT, facecolor=PAPER)
    plt.close(fig)
    im = plt.imread(str(OUT))
    ink = (im[:, :, :3] < 0.96).any(axis=2)
    r_, c_ = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    matplotlib.image.imsave(str(OUT), im[max(r_.min() - 30, 0):r_.max() + 31,
                                         max(c_.min() - 30, 0):c_.max() + 31])
    lo, hi = pad
    print(f"supp touches obj along its right face from ({lo[0]:.4f}, {lo[1]:.4f}) "
          f"to ({hi[0]:.4f}, {hi[1]:.4f}); slid in along a = ({a[0]:+.0f}, "
          f"{a[1]:+.0f})")
    print(f"{OUT}  md5 {hashlib.md5(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
