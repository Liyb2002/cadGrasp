# min_force.png — what it argues, and what not to break

Handoff for `slides/tools/min_force.py` and the one PNG it writes into `slides/tools/figures/`.
Everything below was derived on 2026-08-14 by running the script; every number is
from that run, none from memory.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp/tools
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u min_force.py          # ~1.3 s, writes one PNG, prints every number first
```

| file | md5 | note |
|---|---|---|
| `min_force.png` | `3cd278bb177194d7b554326ca795330d` | 3150 × 2226 |

The script is deterministic — no randomness anywhere — and two consecutive runs
were checked byte for byte. It touches nothing else: `dimension_flat.png` was
`42b00980c594ab1c83ed3ead6f6a9491` and `normals.png` `9a82bdb6f7d5d66197e14492a6ab2248`
before and after.


## The argument

This is the last paragraph of `dimension.md` on its own. There the lesson was that
the angular SIZE of a cone of contact forces says nothing about what can be pushed
along it — 179° of directions with 1.7 % of a support's strength in them. Here that
is the whole figure, in the plane, with no body, no floor and no third dimension.

Two contacts push along unit directions `t` apart, each able to supply at most the
same `F`. In the plane the two pushes are a basis for everything between them, so a
unit direction `phi` off the first is reached one way and one way only:

    d = c1 u1 + c2 u2      c1 = sin(t - phi)/sin t      c2 = sin(phi)/sin t

Both coefficients are non-negative on the arc — outside it one turns negative and a
contact would have to PULL — and scaling `d` scales both, so the cap binds on the
larger and on nothing else:

    m(phi) = F / max(c1, c2)

The **guarantee** is the worst of that over the arc, and it comes out

    min m  =  F              t <= 90 deg
           =  sin(t) * F     t >  90 deg

**The second derivation is the one the top row draws, and it is the one to keep.**
Each contact contributes a segment of forces `[0, F] u`, so what the PAIR can supply
is the Minkowski sum of two segments — the **rhombus** `0, F u1, F u2, F(u1 + u2)`.
"How hard can the pair push along `phi`" is then "how far is it from the corner at 0
to the far boundary of that rhombus", and the guarantee is the nearest point of that
boundary. The far boundary is two straight edges, so this is a point-to-segment
distance and nothing more:

> the foot of the perpendicular from 0 onto the edge through `u2` sits at parameter
> `s = -cos t` along it. That is OUTSIDE the segment while `t <= 90` and INSIDE it
> once `t > 90`.

Outside, the nearest point is the segment's own end — a rhombus corner, which is a
push direction, at distance `F`. Inside, it is the foot, at distance
`|u2 - cos(t) u1| = sin(t)`. **The 90° wall is the torque the foot of a perpendicular
slides onto an edge, and it is nothing else.** The script computes the guarantee this
way as well as by sweeping, and the two agree to `0.00e+00`.

The lobe in each panel is traced from `reach()` point by point rather than drawn as a
rhombus, and it comes back a rhombus every time. That is a standing check: if it ever
comes back curved, `reach` and the geometry have parted company.


## The numbers

Printed by the run. `phi` is measured from the first push, and the two minimisers are
symmetric — `t - 90` and `90`, which coincide with the two ends `0` and `t` below the
wall.

```
    t     min m over the arc   = ?       where it sits, phi        the bisector   overstates
    30         1.0000 F      F        0 and 30 deg   a push itself     1.9319 F     x 1.932
    60         1.0000 F      F        0 and 60 deg   a push itself     1.7321 F     x 1.732
    89         1.0000 F      F        0 and 89 deg   a push itself     1.4265 F     x 1.427
    90         1.0000 F      F        0 and 90 deg   a push itself     1.4142 F     x 1.414
    91         0.9998 F      sin t    1 and 90 deg   square to a push  1.4018 F     x 1.402
   100         0.9848 F      sin t    10 and 90 deg  square to a push  1.2856 F     x 1.305
   120         0.8660 F      sin t    30 and 90 deg  square to a push  1.0000 F     x 1.155
   150         0.5000 F      sin t    60 and 90 deg  square to a push  0.5176 F     x 1.035
   170         0.1736 F      sin t    80 and 90 deg  square to a push  0.1743 F     x 1.004
   175         0.0872 F      sin t    85 and 90 deg  square to a push  0.0872 F     x 1.001
   179         0.0175 F      sin t    89 and 90 deg  square to a push  0.0175 F     x 1.000
```

**These agree with `dimension.md`'s table**, which quotes 1.0000 at 60 and 90, 0.9848 at
100, 0.8660 at 120, 0.5000 at 150 and 0.0175 at 179, and puts the minimiser at
`phi = t - 90`. That is the same set of angles read out at the lower of the two
symmetric minimisers; this figure names both. **If the two tables ever disagree, one
of the two scripts has been broken.**

Three independent routes to those numbers, and the disagreements between them:

```
closed form vs a 200001-point sweep of the arc          2.38e-12
closed form vs point-to-segment distance on the rhombus 0.00e+00
closed form vs sweep, over 3600 angles 0.05 .. 179.95   1.22e-09   worst, at t = 127.56
```


## The two things the figure exists to correct

**1. The weakest direction is not the bisector — the bisector is the arc's BEST
direction.** It splits the load evenly (`c1 = c2`) and reaches `2 cos(t/2) F`, which is
the far corner of the rhombus, both contacts at full `F` at once. It can therefore only
ever *overstate* the guarantee, by exactly `1/sin(t/2)`:

```
   t =    30   bisector at phi =  15.0, nearest minimiser at  30.0  -- 15.0 deg away;  ratio 1.9319
   t =    90   bisector at phi =  45.0, nearest minimiser at   0.0  -- 45.0 deg away;  ratio 1.4142
   t =   120   bisector at phi =  60.0, nearest minimiser at  90.0  -- 30.0 deg away;  ratio 1.1547
   t =   150   bisector at phi =  75.0, nearest minimiser at  60.0  -- 15.0 deg away;  ratio 1.0353
   t =   175   bisector at phi =  87.5, nearest minimiser at  85.0  --  2.5 deg away;  ratio 1.0010
   t =   179   bisector at phi =  89.5, nearest minimiser at  89.0  --  0.5 deg away;  ratio 1.0000
```

The minimiser never lands on the bisector at any `t`, and the gap closes only as
`t -> 180`, where the whole arc is collapsing onto itself and every direction in it is
weak together. **Below the wall the minimiser is a push direction itself** — obvious
once said, since only one contact contributes to it at all — and above the wall it is
square to a push, where the *other* contact is asked for `1/sin t` of the total.

**2. The wall is at 90°, not 120°.** 120° is where the *bisector* falls to `F`, which
the script also finds, to make the contrast concrete:

```
   watching the BISECTOR instead puts the wall at 120.0000000 deg -- 120, and wrong by 30
```

**At 120° the bisector still reads 1.000 F while the true weakest direction is already
down to 0.866 F.** That is the whole reason both curves are on the lower plot.

The 90° is found rather than asserted, by bisecting on the SAMPLED minimum with no
formula in the loop. Two known effects blunt that bisection and both are predicted
before it runs — the minimum leaves `F` *quadratically* (`sin(90 + e) = 1 - e²/2`), so a
tolerance `tol` cannot place the wall inside `sqrt(2 tol)` radians; and a sweep of `n`
points can miss the minimiser by half a step `d`, which lifts the sampled minimum by
about `d²/2`. So the answer has to land in `[90 + sqrt(2 tol), 90 + sqrt(2 tol + d²)]`,
and it does at every tolerance, from one that is all tolerance to one that is all grid:

```
   accept within 1e-04 of F  ->   90.8102915 deg   predicted band 0.8102847 .. 0.8102855   yes
   accept within 1e-07 of F  ->   90.0256386 deg   predicted band 0.0256235 .. 0.0256482   yes
   accept within 1e-10 of F  ->   90.0012709 deg   predicted band 0.0008103 .. 0.0013864   yes
   accept within 1e-13 of F  ->   90.0011252 deg   predicted band 0.0000256 .. 0.0011253   yes
```

The residual is entirely tolerance and grid; there is no third thing in it, and the
wall is at 90. **And with no search at all**, `s = clamp(-cos t, 0, 1)` crosses zero at
exactly 90 by the sign of a cosine — 0.0000 at 60, 89 and 90, then 0.0175 at 91, 0.5000
at 120, 0.8660 at 150. There is nothing here to converge to.


## 180°, and 0°, are degenerate and are marked as such

**At exactly 180° the two pushes span a LINE, not a wedge.** There are no directions
strictly between them, so "the weakest direction on the arc" has nothing to range over:
the minimum is over an EMPTY set, not a small one. What the pair can still do is push
along the line itself, either way, at exactly `F` — that is `dimension.md`'s opposed
pair, form closure along one axis, the strongest thing two contacts can do to one degree
of freedom.

So the curve's limit of 0 is a statement about directions just OFF the line and not
about the case itself, and it is marked that way: an **open circle at (180, 0)** for a
limit that is not attained, a **filled dot at (180, F)** for what actually happens there,
and four lines of text saying so. **Do not let the curve run into the corner unremarked**
— read carelessly it says two opposed supports are worth nothing, which is the opposite
of the truth.

`t = 0` is degenerate in the other direction and gets one line and one grey dot: the
pushes coincide, the arc is a single direction, both contacts drive it, and the pair puts
**2F** along it. There the *bisector* curve is the one that is right and the limit is the
one that is wrong. It is on the plot because a reader who has just been told the bisector
is the wrong thing to watch should see the one place it is the whole answer.

**180° is not drawn as a panel in the top row**, and that is deliberate: there is no
wedge left to draw, and a panel of it would be two opposed arrows with an empty region
between them, which looks like a mistake rather than a degeneracy. 175° carries the
collapse instead.


## Reading the picture

**Top row — five pairs, 60°, 90°, 120°, 150°, 175°.** Two of them are inside the wall on
purpose: 60 and 90 have to look alike, because they *are* alike. 120 is the angle the
bisector argument gets wrong, 150 is half strength, 175 is the collapse.

- **Orange arrows are the two pushes**, drawn at length `F`.
- **The blue region is what the pair can supply** — every direction of the arc out to
  `m(phi)`, which is the rhombus. Its two lower corners are pinned to the orange arrow
  TIPS at every `t`, exactly, because a push direction always reaches `F`. The reader
  watches the middle fall away with the ends held still.
- **The dashed arc does two jobs at once**: it is the span of directions being minimised
  over, drawn at the height one support reaches. Anything inside it is a direction the
  pair cannot put a whole support into.
- **Blue arrows are the weakest directions**, at the guarantee. Below the wall they lie
  ON the pushes — that overlap is the finding, not a collision.
- **The grey open circle** at the top of each region is the rhombus's far corner, both
  contacts at full `F`: the bisector, and the region's high point, never its low one.
- **Grey hairlines** run from the origin out to the dashed arc along each weakest
  direction, so that a whisker of an arrow still says WHERE it points, and the shortfall
  reads as the gap between its tip and the arc.

**Every arrow uses one encoding: length AND line width AND head all scale with the force.**
Nothing is floored — at 175° the weakest direction is 0.0872 F and it is drawn as a
whisker, because that is what it is. This matches the other new figures in the series.

**The one liberty**: the two colours have different base weights, orange 4.6 pt at `F`
against blue 2.7 pt. Below the wall the weakest direction IS a push, so the arrows
coincide exactly, and at equal weight the pair would be a single arrow of ambiguous
colour; narrower, the blue sits inside the orange as a core with the orange showing round
it. **Length is the quantity to read across the two colours; width compares blue against
blue.** Anyone equalising the two weights has to solve the overlap some other way first.

**Bottom — the guarantee against `t`**, with the bisector dashed beside it and the band
between them shaded. The five panels are tied to it by dotted verticals and dots. The
corner at 90° is annotated, the `sin t` tail is labelled just PAST the corner rather than
out at 160 — the two curves converge as `t -> 180`, their ratio being `1/sin(t/2)`, and a
label out there would sit on both at once.


## Decisions that look free and are not

**The pairs are drawn symmetric about straight up.** Only the angle between the pushes
appears in the mathematics, so the orientation is free, and it is spent on making five
panels comparable at a glance.

**All five panels share one scale**, `xlim = ±1.10`, `ylim = -1.03 .. 1.90`. The arrows
are only comparable across panels because of that. **Do not tighten the limits of one
panel** — the 60° region reaching 1.732 is what sets the top, and the captions inside the
axes are what set the bottom.

**The gridspec height ratio is 1 : 1.42 and it is not decorative.** The top row uses
`aspect="equal"`, so its axes boxes shrink to the data they hold; at that ratio the
panels are limited by the column width rather than the row height, which is the largest
five across a 15 inch page can be. A larger top share just adds whitespace above and
below the row.

**Both plotted curves are evaluated, not sketched.** `weakest()` draws the lower curve
and it is the same function `checks()` puts against the sweep and against the rhombus. A
hard-coded `np.where(t < 90, 1, np.sin(t))` in the drawing would let the picture and the
check drift apart.

**The printed table's angles are `dimension.md`'s angles**, plus 170 and 175 for the
tail. That is what makes the two documents checkable against each other, so **do not
prune `TABLE` down to the five drawn angles.**


## Known open issues

- **The output path is absolute and hard-coded**, like the rest of `slides/tools/`. The script
  cannot be run from a clone elsewhere without editing `OUT`.
- **The 175° panel's blue whisker is 0.26 pt wide and about 12 px long**, which is at the
  edge of what a raster shows. That is the honest consequence of not flooring the
  encoding; the tip dot and the hairline guide are what make it findable. If this figure
  is ever printed small, the dot is the thing to enlarge — not the arrow.
- **The three near-vertical dotted lines in the 150° and 175° panels** (two weakest-
  direction guides and the bisector) very nearly coincide as `t -> 180`. Correct, and
  slightly crowded.
- **`m(phi)` is a plane result and the script says so, but nothing enforces it.** In
  space, two capped contacts span a 2-D subspace and the same rhombus argument applies
  *within that plane*; the guarantee over a solid cone of three or more is a different
  and harder question and is not what this figure answers.
- **`checks()` is 0.62 s of the 1.3 s run**, nearly all of it in the 3600-angle sweep
  and the four bisections. It runs before anything is drawn on purpose, so a broken claim
  never reaches paper. Anyone tempted to trim it should trim the sweep, not the three
  independent routes to the same number.


## Open questions for the human

1. **Whether 60° and 90° both earn a panel.** They are drawn to make the point that
   nothing happens between them, which costs a fifth of the top row. If space is ever
   needed, dropping 60 and adding 100 would show the wall being crossed instead — a
   different point, equally true.
2. **Whether the `t = 0` mark belongs on the plot.** It is a second degeneracy in a
   figure whose honest-handling budget is mostly spent on 180°, and it could go in this
   note alone.
