r"""The problem, as three lines of one shape.

    python slides/setup/equations/three_equations.py   ->  slides/setup/equations/three_equations.png

There was a long form, `four_equations.png`, which split the left-hand side
into skin and floor and carried the assembly's force row as a fourth line. It
was wrong and it is deleted; this page is the only statement of the problem, and
it makes four deliberate trades against what that page tried to say. Naming them,
because the page they are traded against is no longer there to check against: the
two left-hand integrals become ONE; the fourth row becomes the third row read
again; the assembly's force row is BOUGHT with friction rather than dropped; and
the pressure cap is DEDUCTED to the assumptions slide. The other headings below
are drafting decisions, not trades.

VECTORS SINCE 2026-09-01: `d_supp` AND `d_push` ARE DELETED. `F_supp` and
`F_push` are now VECTORS -- the press and the push, magnitude and direction in
one symbol -- where before each was a magnitude times its own unit direction.
What it buys: two symbols a side instead of three, and both torque rows shorten
to `r x F`. The field/parameter distinction the old `d` pair carried now rides
where it belongs anyway: `F_supp` lives under the integral, one vector per
patch of skin, and `F_push` is one given vector per push. What it costs, named:
one-sidedness can no longer be written `F_supp >= 0` -- the headline says it in
words (it presses, it cannot pull) and the legend carries the frictionless
reading, `F_supp` along the skin's own inward normal; and row 3 no longer shows
`u = z` inside the equation (the old `F_supp r_supp x z` carried it), so rows 2
and 3 are now the SAME LINE with only the free body changed, and the note under
row 3 alone says every contact there is vertical. The down-push check survives
the rewrite: `F_push = -F z` gives `(mg + F) z` -- the supports carry MORE.

THE FLOOR IS JUST ANOTHER CONTACT. Its direction is `up` and its pressure
cannot be negative, which is what every other contact already is. So the two
left-hand integrals of the long form -- skin and floor -- become one, and
nothing is lost but the ability to point at the floor's share.

WHICH IS WHY THE DOMAINS CAN BE NAMED FOR THEIR INTERFACES. An earlier cut wrote
`part` and `ground`, meaning every contact the object has and every contact the
ground has, and that hid the PIVOT: the workpiece is tipped about an edge of its
own footprint and never leaves the floor (METHOD s0), so it presses there still.
Naming the first domain `supp_obj` is exact only once the ground under the
workpiece is counted as one of the supports -- which is the paragraph above, and
is the line the page now carries. The pivot is then in it by right: it is a
contact on the object.

AND THE THIRD DOMAIN IS `sys_floor`, NOT `supp_floor`. That is a free body and
not a name. Rows (1) and (2) cut around the WORKPIECE, so their domain is
everything pressing on the object. Row (3) cuts around the workpiece AND its
supports, so its domain is everything pressing on THAT -- every support's foot,
and the workpiece's own pivot, which is on the floor and is not on any support.
`supp_floor` drops the pivot unless the floor-is-a-support convention is dragged
in to save it, and then it has to mean "the floor's contact with the floor",
which is nothing. `sys` is the assembly, the free body row (3) is actually
written about, and under that name the pivot needs no rescuing.

AN UNDERSCORE AND NOT A HYPHEN, AND THAT IS NOT TYPOGRAPHY. Inside mathtext a
hyphen IS a math minus, so `\int_{\rm supp-obj}` set the domain as `supp - obj`
with the minus's own spacing around it -- the one name this cut exists to
introduce, rendered as a subtraction. `\_` is the literal, and the `\;` after
the subscript is there for the same reason: at `\,` the domain name still butted
into the `F` it is a subscript of, which is where the eye loses which letters
belong to the integral sign and which to the integrand.

THE RIGHT-HAND SIDE IS THE LOAD TURNED ROUND, AND THE PAGE HAS TO SAY SO.
Three readings in a row stopped at the force row's minus and asked why it was not
a plus -- "the supports carry the weight AND the push, so surely they add". They
do add: the right-hand side is `-(load)` with the load expanded into its two
terms, and the terms started with opposite signs. Gravity is `-mg z`, so reversed
it is `+mg z`; the process is `+F_push`, so reversed it is `-F_push`. The plus
between the two reversals is there; what is visible is the second one's own
sign. The decisive check, and the one worth saying out loud: push straight DOWN,
`F_push = -F z`, and the right-hand side becomes `(mg + F) z` -- the supports
carry MORE. Write it `+` instead and a downward push would make them carry LESS.

So the right-hand side is now written as ONE BRACKET with ONE minus in front of
it: `-( -mg z + F_push )`. Inside the bracket every term is a load acting
on the workpiece, with its own real sign -- gravity points DOWN and says so --
and the minus outside is the single act of turning the whole thing round. The
expanded form `mg z - F_push` is the same vector and was read four times
as "gravity MINUS the push", because the gravity term's reversal is invisible
there (the notation writes its direction as `z`, already flipped) while the
push's is not. The bracket makes both reversals one operation. Rows 2 and 3 take
the bracket too, so the minus means the same thing in all three, and the legend
now says `gravity is -mg z` where it defines `mg`.

WHY GRAVITY IS IN THE FORCE ROW AND NOT THE TORQUE ROWS. It acts at `c`, and
`(c - c) x F = 0`. That is exact for (1) and (2), whose free body is the
workpiece. For (3), whose free body is the workpiece AND its supports, it holds
only because the supports are treated as MASSLESS: give them weight and it acts
at the assembly's centre of mass, not at `c`, and leaves a torque. Nothing in
METHOD.md declares this -- searched, and support mass appears nowhere -- so it
is an assumption the model has been making silently. It belongs on the
assumptions slide beside the pressure cap, not in the statement.

THE THIRD LINE IS THE SAME EQUATION. "Centre of pressure inside the footprint"
is not a different kind of statement: it is the torque row again, over the
GROUND footprint, where every contact points `up`. Under the vector notation
that identity is typographic -- rows 2 and 3 are the same line with only the
domain changed -- and the fact that every `F_supp` there is vertical moves to
the note under the row, where the old form wrote it into the equation as
`u = z`. What makes it the support polygon is not the equation but
one-sidedness underneath it -- a floor that cannot pull.

(3) OF THE LONG FORM IS BOUGHT, NOT DROPPED. Declaring the floor's friction
sufficient turns the assembly's FORCE row from an equation with an empty left
side into an inequality to check, `K|d_h| <= mu (1 - K d_z)`, needing a friction
coefficient the model does not declare. What it does NOT do is delete the
horizontal load: friction acts at the floor while the centre of mass is at
height `h`, so not sliding puts `h K mg |d_h|` straight into the third line.

NO COLOUR. An earlier cut drew the torque rows in violet. METHOD s3.0 already
spends violet on "owed, direction available, not strong enough", so a violet
equation is a clash, not an emphasis. Three lines of one shape do not need one
of them shouting.

WHAT IS ON THE RIGHT. `F(d)` and `tau(d)` are the two demand balls the previous
slide draws, so the page before this one IS the right-hand side. Expanded, they
separate into gravity and the process, and that separation carries a fact worth
the two lines it costs: gravity is in the force row and NOT in the torque row,
acting at `c` and so turning nothing about it.

NAMED IN PAIRS, WHICH IS WHY THERE IS NO `F(d)` LEFT. An earlier cut called the
support side `lambda, u, r` and the process side `f, d, r_q`, described BOTH
directions as "which way it pushes", and put them a page apart -- leaving no way
to tell a FIELD (one vector per patch, inside the integral) from a PARAMETER
(one vector per push, given). Named `_supp` and `_push` the pairing is in the
symbols, so the two sides can be written on ONE LINE and read against each
other:

    int_supp_obj  r_supp x F_supp  dA  =  -( r_push x F_push )

That killed the second block. `F(d)` and `tau(d)` existed only to keep the lines
short enough to leave the right-hand sides for an expansion below; with the
names doing the work the expansion IS the line, and `F(d)` would now collide
with `F_supp` besides.

ONE ABUSE, DELIBERATE. `F_supp` is a PRESSURE -- force per unit area, which is
what `dA` is there to integrate -- while `F_push` is a force. So the pair is
parallel in role and not in units. The alternative, `P_supp`, keeps the units
honest and breaks the pairing that makes the page readable in ten seconds; this
page is for saying out loud. Nothing carries the units-honest form any more --
the long form that did is deleted -- so the abuse is now undeclared anywhere but
here, and the assumptions slide is where it should be said.

WHAT IS DEDUCTED. The pressure cap. The long form carries `P lambda` with
`lambda` in [0,1]; here `lambda >= 0` alone, because one-sidedness is what makes
the problem's shape and the cap is a number -- it belongs on the assumptions
slide with `A0 = mg/P = 5.4 cm^2`, not in the statement of what is being solved.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

from pathlib import Path                                     # noqa: E402

OUT = str(Path(__file__).with_name("three_equations.png"))
INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"

EQ, LAB, SUB = 30, 21, 20


def main() -> None:
    # every line LEFT-aligned on one column, and wide: at 13 in the symbol line
    # and the third row ran off both edges and lost the symbol each was there to
    # define -- the trap the deleted long form hit
    fig = plt.figure(figsize=(16.0, 8.6), dpi=200, facecolor=PAPER)
    t = lambda y, s, **k: fig.text(k.pop("x", 0.5), y, s, ha=k.pop("ha", "center"),
                                   va="center", color=k.pop("c", INK),
                                   fontsize=k.pop("fs", EQ), **k)

    LAB_X, EQ_X = 0.030, 0.150     # the name of the row, then the row
    COL = (0.030, 0.165, 0.520, 0.720)

    t(0.958, r"Find a pressure field  $F_{\rm supp}$  — it presses, it cannot pull",
      fs=36)
    for y, who, row in (
            (0.885, "the supports", (r"$F_{\rm supp}$  what it presses with",
                                     r"$r_{\rm supp}$  its arm",
                                     r"$\hat{y}$  up")),
            (0.835, "the process", (r"$F_{\rm push}$  the push, one vector",
                                    r"$r_{\rm push}$  its arm",
                                    r"$q$  where it lands"))):
        t(y, who, x=COL[0], ha="left", fs=SUB, c=INK)
        for x, cell in zip(COL[1:], row):
            t(y, cell, x=x, ha="left", fs=SUB, c=MUTED)
    t(0.790, r"$mg$  the workpiece's weight, so gravity is $-mg\,\hat{y}$   ·   "
             r"$c$  the centre of mass, which both arms are measured from",
      x=COL[0], ha="left", fs=SUB, c=MUTED)
    t(0.742, r"$F_{\rm supp}$ points along the skin's own inward normal   ·   "
             r"the floor under the workpiece counts as a support",
      x=COL[0], ha="left", fs=SUB, c=MUTED)
    t(0.694, "every row: the left is what the supports SUPPLY, the bracket is "
             "every load, the minus turns it round", x=COL[0], ha="left",
      fs=SUB, c=MUTED)

    RHS = r"r_{\rm push} \times F_{\rm push}"
    for y, lab, eq in (
            (0.600, "force",
             r"$\int_{\rm supp\_obj}\; F_{\rm supp} \; dA \;=\;"
             r" -\left( -mg\,\hat{y} \;+\; F_{\rm push} \right)$"),
            (0.450, "torque",
             r"$\int_{\rm supp\_obj}\; r_{\rm supp} \times F_{\rm supp}"
             r" \; dA \;=\; -\left( " + RHS + r" \right)$"),
            (0.300, "no tipping",
             r"$\int_{\rm sys\_floor}\; r_{\rm supp} \times F_{\rm supp}"
             r" \; dA \;=\; -\left( " + RHS + r" \right)$")):
        t(y, lab, x=LAB_X, ha="left", fs=LAB, c=MUTED)
        t(y, eq, x=EQ_X, ha="left", fs=EQ)
    t(0.212, r"$\rm sys$ is the workpiece and its supports as one body, and its "
             r"contacts with the floor all point up", x=EQ_X, ha="left",
      fs=SUB, c=MUTED)
    t(0.158, r"the same row, on that footprint   ·   $F_{\rm supp}\parallel\hat{y}$, "
             r"pressing up, is what makes it the support polygon", x=EQ_X,
      ha="left", fs=SUB, c=MUTED)
    t(0.088, r"gravity is in the force row and not the turning rows: acting at "
             r"$c$, it turns nothing about $c$", x=COL[0], ha="left", fs=SUB,
      c=MUTED)
    t(0.026, r"for every push", fs=26)

    fig.savefig(OUT, facecolor=PAPER)
    plt.close(fig)
    a = plt.imread(OUT)
    ink = (a[:, :, :3] < 0.96).any(axis=2)
    r, c = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    matplotlib.image.imsave(OUT, a[max(r.min() - 30, 0):r.max() + 31,
                                   max(c.min() - 30, 0):c.max() + 31])
    print("F_supp and F_push are VECTORS since 2026-09-01: d_supp and d_push are")
    print("gone, the torque rows read r x F, and one-sidedness is said in words")
    print("line 3 IS line 2, over the ground; every contact there is vertical,")
    print("and a floor that cannot pull is what makes it the support polygon")
    print("gravity is in the force row and NOT the torque rows: at c it turns")
    print("nothing.  the cap stays out: `A0 = mg/P` on the assumptions page")
    print(OUT)


if __name__ == "__main__":
    main()
