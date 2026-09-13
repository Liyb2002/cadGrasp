# projection_angles.png — what it argues, and what not to break

Handoff for `slides/tools/projection_angles.py`, which writes exactly one file,
`slides/tools/figures/projection_angles.png`. Everything below was re-derived on 2026-08-14 by
running the script; every number is from that run, not from memory.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp/tools
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u projection_angles.py      # ~2 s, one PNG, prints every number it draws
```

Deterministic — no sampling, no randomness, no time stamps — and two clean runs
give the same bytes. `slides/tools/figures/projection_angles.png` is
`aa2326489ebc537d94894175c5b36e47`, 2772 × 1868. It writes nothing else, and the
two frozen plane figures next to it must stay where they are:
`dimension_flat.png` `42b00980c594ab1c83ed3ead6f6a9491` and `normals.png`
`9a82bdb6f7d5d66197e14492a6ab2248`.


## The argument

**One operation, four angles.** Two supports push along unit directions `u1` and
`u2`, `t` apart, **each with magnitude F**. Pick a direction `w` between them,
drop a perpendicular from each force's tip onto the `w` axis, and the foot is
that force's component along `w`. The pair delivers the sum:

    F(u1.w) + F(u2.w) = F(u1 + u2).w

Write `w` at `psi` from the bisector, so the forces stand `t/2 - psi` and
`t/2 + psi` from it:

    each component  F cos(t/2 -+ psi)          the sum  2F cos(t/2) cos(psi)

`cos(psi)` never exceeds one, so **the bisector is the best direction a pair
has** and `2F cos(t/2)` is the whole of what it can do — in any direction on the
page, not just between the pushes. The script checks that by sweeping `w` right
round the circle, not merely across the wedge.

**This is not decomposition, and the figure says so once per panel and then drops
it.** Decomposing solves `v = c1 u1 + c2 u2` for what each support must SUPPLY to
deliver a wanted `v`; resolving asks what ARRIVES from a supply of `F`. On the
bisector the two answers are `1/(2cos(t/2))` and `cos(t/2)` — reciprocal in
`2cos(t/2)`, equal only where `cos(t/2) = 1/(2cos(t/2))`, which is to say **only
at 90°**. At 60° that is 0.5774 F against 0.8660 F; at 175° it is 11.4628 F
against 0.0436 F. Whoever revisits this: the panel note is one sentence and
should stay one sentence. An earlier draft of this figure spent half its arrows
on the decomposition and buried what the panels are actually about.

Printed by the run:

```
   t        each force        the pair delivers      at the wedge edge (w along u2)
  60 deg     0.8660 F            1.7321 F            +0.5000 and +1.0000 -> 1.5000 F
  90 deg     0.7071 F            1.4142 F             0.0000 and +1.0000 -> 1.0000 F
 175 deg     0.0436 F            0.0872 F            -0.9962 and +1.0000 -> 0.0038 F
 180 deg     0.0000 F            0.0000 F            -1.0000 and +1.0000 -> 0.0000 F
```

The edge column is `F(1 + cos t)`, and it is where the cancellation is easiest to
see: at 175° a force resolved onto the OTHER push's direction contributes
**−0.9962 F**, so one support all but exactly undoes the other and 2 F of push
arrives as 0.0038 F.


## 180° is the punchline, not the degenerate case

`u1 + u2` is **exactly zero**, so `F(u1+u2).w` is zero for **every** `w`. Swept
over 2881 directions round the full circle the largest magnitude found is
**1.22e-16 F**, which is the floating point zero. Two opposed supports both
pushing at full `F` deliver **nothing, anywhere** — not "nothing between them",
not "nothing on the bisector".

**The nuance that keeps the panel honest, and which must not be dropped:** such a
pair is not useless. `c1 u1 + c2 u2 = (c1 - c2) u1`, so pushing UNEQUALLY drives
a net force along their own line — the run prints `c = (1.4, 0.6)` giving
**0.800 F** and `c = (2.0, 0.5)` giving **1.500 F**. It is pushing them EQUALLY
that gives zero. This is the same pair `dimension.md` calls "form closure along
one axis, the strongest thing two supports can do to one degree of freedom", and
the figure would be a lie about it without that line.

175° is the same cancellation two degrees short of complete, and the figure puts
them side by side for exactly that reason.


## Reading a panel

Four panels, one per angle, **all four at one scale** (9.00 F across half a
page), so an arrow in one can be compared with an arrow in another directly.
Each panel holds the same three cells, left to right:

1. **`w` on the bisector** — the full construction, largest numbers.
2. **`w` a quarter of the wedge off the bisector** (`psi = -t/4`) — because the
   claim is about ANY direction between the pushes, not only the middle one, and
   because past 90° this is where a component goes negative.
3. **`w` swept right across the wedge** — seven totals and the curve of them.

- **Orange** is force: the two supports at `F`, and each one's component along
  `w`. Length and line width both carry magnitude (`gauge` is linear, 0.85 +
  2.55|c|, because everything here lies between 0 and 2 F).
- **Black** is the total the pair delivers — the two components added.
- **The dashed line from a force's tip to the axis** is the perpendicular, and it
  carries a **right-angle mark at the foot**. That mark is the one thing in the
  figure that is not a quantity; it is there because a dropped line without it is
  just a line to somewhere.
- **The grey arc** through the two arrowheads is one `F` from the origin: the
  wedge, and the scale everything else is read against.
- **The two components are drawn head to tail ALONG the axis**, and the first one
  ends exactly on the first foot, which is what having dropped the perpendicular
  bought.
- **A negative component points backwards along `w`, at its true length.** It is
  never drawn as its absolute value.

**The sweep cell draws a circle, and that is not decoration.** `2F cos(t/2)
cos(psi)` is a cosine in `psi`, so the polar curve `r = A cos(psi)` is a CIRCLE
of diameter `A = 2F cos(t/2)` sitting on the bisector. The tips of the totals lie
on it, so the cell answers for every direction in the wedge and not only for the
seven drawn. At 180° that circle has diameter zero: it is a point, and there are
no arrows to draw at all.


## Decisions that look free and are not

**The bisector is drawn straight up in every panel.** It costs nothing and buys
two things: the panels are symmetric about the vertical and read the same way,
and the near-opposed pairs at 175° and 180° come out lying along the horizontal,
which is the shape of a panel.

**The sum bar is set out beside the axis, at `F + 0.30` in every one of the
twelve cells.** On the axis it would be a third arrow lying along the two
components and invisible under them. Any offset smaller than `F` crosses a force
arrow — every force leaves the same origin the axis does, so a parallel line
nearer than the forces' own reach must cut one. `F + 0.30` is the least that
clears the worst case (a force square to the axis, which is what a near-opposed
pair gives), and making it uniform rather than tight costs a third of an `F` in
width and puts the black answer in the same place in all twelve cells. It gets
its own dotted baseline and two dotted ties, which is what says the black arrow
is the orange chain measured over again.

**When the two components have opposite signs the second one steps aside by
0.11 F.** Head to tail on the axis would draw it exactly over the first, and past
90° apart that is 97.1 % of the wedge at 175° and effectively all of it at 180° —
the case the figure most needs legible. The step is a pure translation, every
length in it is still true, and a dotted connector and a dotted tie put its two
ends back where they belong.

**The fan is drawn at a much lighter weight than the constructions** (0.65 +
0.85|s| against `gauge`) and samples the INTERIOR of the wedge. At full weight
seven totals radiating from one origin close up into a solid black wedge; sampled
at the edges, the outermost arrow lands exactly on a force and draws over it.

**A total below 0.25 F gets a dashed ring round it instead of a bigger arrow.**
At 175° and 180° the whole locus is smaller than an arrowhead. The ring says
where to look, which is more honest than drawing anything larger than it is.

**Panel heights are computed, not set.** Each panel takes half the page width, so
what is free is its height, and that is whatever keeps its own scale square; the
figure's height is the sum. A label's allowance is a fixed PHYSICAL size, which
depends on the panel's scale, which depends on the width the layout comes out at,
which depends on the allowance — `figure()` runs the layout three times to settle
it. Anything added to a cell must go into `reach()` too, or it will hang out of
the box and land on the labels. That is exactly how the sweep cell's ring first
went wrong.


## The checks, all of which run before anything is drawn

`audit()` asserts, and stops the script rather than let a confident-looking
picture out:

- the components come from the **dot products**, and match `F cos(t/2 -+ psi)` to
  better than 1e-12 at 721 directions across each wedge;
- their sum matches `2F cos(t/2) cos(psi)` to the same tolerance;
- the largest total over a full circle of `w` equals `|u1 + u2|` — 1.732051,
  1.414214, 0.087239 and 0.000000 F, all found at 90.0°, which is the bisector;
- at 180° the sum is below 1e-15 in every one of 2881 directions;
- the share of the wedge on which a component is negative matches the closed form
  `2 max(0, t - 90)/t` to within one sample: **0.0 %, 0.0 %, 97.1 %, 99.9 %**.

The last two need a tolerance of `-1e-12` rather than `< 0`, because at t = 90
the edge component is `cos 90°` and lands on −1.1e-16; a bare `< 0` reports a
backwards component that is not there, and a bare format prints "−0.0000 F".


## Known open issues

- **The 175° and 180° bisector cells are nearly empty.** The components are
  0.0436 F and 0.0000 F, so the chain, the feet and the sum bar are all specks.
  That IS the message, and the labels carry the numbers, but a reader skimming
  the picture alone sees two horizontal arrows and very little else. The sweep
  cell's ring is the only place those panels point at their own smallness.
- **`psi = -t/4` for the second cell is a choice**, not a derived quantity. It is
  generic (no degenerate drop, no coincidence with a force) and it is negative so
  that the backwards component appears at 175° and 180°. Any `psi` in
  `(t/2 - 90, 0)` would do the same job.
- **Output path is absolute and hard-coded** in `OUT`. The script cannot be run
  from a clone elsewhere without editing it.
- **`MID` and `BLUE` are defined and unused.** `BLUE` went with the draft that
  drew decomposition beside resolution; both are left in place because the house
  palette is shared with `slides/tools/dimension.py` and should stay whole.
- **The figure is one page wide and does not shrink gracefully.** Label sizes are
  physical, so halving `PAGE` would not halve the text; the label lines are cut
  to about thirty characters to fit a cell at the current width, and anything
  longer runs into the next cell.
