# three_pushes.png — what it argues, and what not to break

Handoff for `slides/tools/three_pushes.py` and the one PNG it writes into `slides/tools/figures/`.
Everything below was re-derived on 2026-08-14 by running the script; the numbers are
from that run or from the closed forms it checks itself against, not from memory.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u slides/tools/three_pushes.py     # ~1.3 s, rewrites one PNG, prints every number
```

There is no randomness anywhere in it and a clean re-run reproduces the file byte for
byte — checked twice here, `85f3b87ef02a9a87fb9ee54ae8e61d6c` both times, with stdout
identical apart from the elapsed-time line. It writes nothing else: `dimension_flat.png`
is still
`42b00980c594ab1c83ed3ead6f6a9491` and `normals.png` still
`9a82bdb6f7d5d66197e14492a6ab2248` across every run of it.


## The argument

Three contacts, each able to push one way only and each capped at the same `F`, one
support's whole strength. Together they can supply the cone of non-negative
combinations `v = Σ cᵢuᵢ`, and — since scaling the whole combination scales every
coefficient together — the cap binds on the largest of them:

    m(v) = F / maxᵢ cᵢ

`dimension_ball.png` paints which directions the cone reaches and says nothing about
how hard, and on its own that half is not merely incomplete, it is the misleading
half. This figure draws both at once: the cone is the painted patch, and **the paint
IS the magnitude**.

Three balls, one camera, one colour scale, and **one arrangement posed the same way in
all three**: `u₁` straight up, because in the scene these figures are about one
generator is not a chock but the **floor**, and the floor's push does not move when the
chocks are re-cut. What changes across the row is the other two. The panels are in the
order the argument wants rather than in order of angle.

- **(a) the three pushes mutually square** — one up and two horizontal, the coordinate
  axes. The cone is the corner of a cube: an octant, `12.500 %` of the ball, **exactly
  one eighth**, and every direction in it takes `F` or more — from exactly `F` at the
  three pushes up to `√3 F = 1.73 F` down the middle.
- **(b) the three pushes close together, 50°.** The two swing UP towards the vertical
  one and cluster round it — shims driven under an edge rather than blocks stood
  beside it. A much smaller cone, `2.902 %`, under a quarter of the octant, and again
  everything in it takes `F` or more, now up to `2.62 F`. Narrowing buys strength in
  the middle and costs reach, and it does **not** raise the floor.
- **(c) the three pushes spread wide, 110°.** The two swing DOWN to 20° below
  horizontal and splay apart — clamps hooked over an edge. Twice the octant's reach,
  `25.550 %` — and the field has fallen **through** `F`. The weakest direction of the
  patch is worth `0.80 F`, and only `4.9 %` of the patch still takes a full support's
  worth.

90° is the hub and the two panels beside it are the two ways off it, which is why it
is drawn first rather than in the middle. Reach went up in (c) and capability went
down, which is the reason the angular size of a cone is the wrong thing to quote
about a support arrangement.


## The pose, and why it is the whole of this revision

**2026-08-14.** The figure's numbers were already right and were not touched. What was
wrong was the pose. `tripod` used to return three vectors **symmetric about +z** with
the camera looking down their common axis, so **none of the three was vertical at any
angle** and panel (a) read as two rays going up-left and up-right with one straight
down. That is the same cone, and it is disconnected from the scene it belongs to,
where one generator IS straight up: it is the floor.

`tripod` now returns `u₁ = +z` at every angle, and the other two at polar angle `t`
and azimuths `±A/2`. **`A` is not a new number.** With `u₁` at the pole the arcs
`u₁→u₂` and `u₁→u₃` are meridians, so the angle between them is just the difference in
azimuth — which is the spherical triangle's own corner angle at `u₁`, the same

    cos A = cos t / (1 + cos t)

that `covers()` integrates. Every pairwise angle is then exactly `t`: `u₁·u₂ = cos t`
by the polar angle, and `u₂·u₃ = sin²t cos A + cos²t = cos t` by that identity. The
run asserts both, at every angle drawn.

At 90° that gives `A = 90°` and the three are exactly the coordinate axes — one up,
two horizontal, and the patch is the axis-aligned octant. **The family is the physical
one:** the floor's push never moves, and the two chock pushes tilt up as the angle
narrows (a shim under an edge) or down below the horizontal as it opens (a clamp over
an edge). At 110° they stand 20° below horizontal.

The degenerate end still behaves. `cos(120°)` comes back as `-0.4999999999999998`, and
carried through it makes `cos A = -1.0000000000000004`, a NaN out of `arccos` rather
than the flat tripod 120° really is; the cosine is snapped to 12 places first and the
ratio clipped — a bug found in an earlier pass, kept fixed in its new form.


## The camera, and the one place the brief could not be followed

`ELEV, AZIM = 20.0, -12.0`. Read them in the page basis, where roll is 0 and so world
`+z` always projects straight up the page:

    +z                            ->  (0, cos elev)
    horizontal at page-angle th   ->  (cos th, -sin th sin elev)

with `th` from page-right and **negative page-y meaning coming out of the page at the
reader**. `AZIM` is the offset between the eye and the arrangement's own mirror plane,
so at 90° the two horizontals land at `th = 45 + AZIM` and `th = 135 + AZIM`.

Measured on the run — the script prints all nine every time:

```
      t              push    polar    page x    page y   page len   off-axis
    90°    u_1  the floor     0.0°    +0.000    +0.940      0.940     70.00°
    90°               u_2    90.0°    +0.839    -0.186      0.859     59.22°
    90°               u_3    90.0°    -0.545    -0.287      0.616     37.99°
    50°    u_1  the floor     0.0°    +0.000    +0.940      0.940     70.00°
    50°               u_2    50.0°    +0.546    +0.420      0.689     43.57°
    50°               u_3    50.0°    -0.281    +0.360      0.457     27.17°
   110°    u_1  the floor     0.0°    +0.000    +0.940      0.940     70.00°
   110°               u_2   110.0°    +0.897    -0.417      0.989     81.59°
   110°               u_3   110.0°    -0.706    -0.534      0.885     62.21°
```

So panel (a) draws as a coordinate triad: one straight up at `0.940`, one across the
page at `0.859`, one foreshortened towards the reader at `0.616`. `u₁` is at
`(0, +0.940)` in all three panels, which is the check the revision exists for.

**Two rules, both asserted, both printed.** No push within **20°** of the view axis —
a ray nearer than that keeps under a third of its length on the page and reads as a
dot rather than an arrow. And no push past **84°** from it — 90° is the limb, and the
patch is the cone ON the pushes, so a push round the back puts paint round the back
too. The second is tight, not decorative: at 110° the receding push stands `81.59°`.

**The brief for this revision asked for `elev 30` with the horizontals at `th = 20`
and `110`, and that camera does not exist.** It is a better triad — page lengths
`0.955` and `0.581` against `0.940` up — and it is exactly reproducible from the
formulae above, so it was tried first. It stands 25° off the mirror plane, and
measured over the other two panels it puts

- the 50° panel's near push **12.16°** off the view axis, with `0.211` of its length
  left on the page, and
- the 110° panel's far push **96.28°** from the eye, six degrees **round the back of
  the ball**.

No elevation rescues the second. At 25° of offset the eye cannot come within `23.55°`
of the 110° tripod's 3-fold axis, and its pushes stand `71.06°` off that axis, so one
of them is over the horizon at every elevation. The feasible band, scanned: the limb
is reached at 15° of offset from elev 15 and at only 7° from elev 30, and lowering the
elevation to buy offset flattens the ball towards a disc, since the horizontal plane
of directions projects to an ellipse of semi-minor axis `sin(elev)`. `20, -12` is the
corner of that trade.

**One camera and one pose for all three panels, and both have to be single.** The
reader compares the SIZE of the three patches. Turning the arrangement about the
vertical between panels would be geometrically free — `u₁` is the axis of that turn,
so it would move nothing and change no angle — and it would buy back the brief's
camera for panel (a) alone. It is not done, because then the row would no longer be
one arrangement opening and closing: "the two swing up" and "the two swing down" would
be false, and that sentence is the figure.


## The numbers, and the checks they are made to pass

Printed by the run. The `exact` columns are closed forms; the `sweep` columns are a
uniform Fibonacci sweep of `N = 1 000 000` directions.

```
      t           cone covers             weakest           strongest   of the patch,
             sweep      exact     sweep     exact     sweep     exact   at or above F
    50°     2.903%     2.902%     1.003     1.000     2.607     2.619        100.00%
    60°     4.387%     4.387%     1.001     1.000     2.444     2.449        100.00%
    90°    12.500%    12.500%     1.000     1.000     1.729     1.732        100.00%
   110°    25.549%    25.550%     0.803     0.803     1.145     1.147          4.95%
```

and, per panel,

```
        t      cone covers    weakest    strongest    of the patch at or above F
  (a)  90 deg   12.500 %      1.0000 F    1.7321 F         100.00 %
  (b)  50 deg    2.902 %      1.0000 F    2.6185 F         100.00 %
  (c) 110 deg   25.550 %      0.8028 F    1.1472 F           4.95 %
```

60° is not a panel; it is in the table because it is a row the human's own run
reports, so a regression shows up here as a number moving rather than as a figure
merely looking different. The three panels take 90, 50 and 110.

**The `sweep` column moved in the third decimal when the pose changed, and that is
correct rather than a regression.** The Fibonacci set is fixed in space; turning the
cone through it changes which samples land inside. Before the revision the same
columns read `2.902 / 1.001 / 2.608` at 50°, `1.002 / 2.442` at 60° and `1.147` at
110°. Every `exact` column is unchanged to every digit printed, which is the point of
having both.

**Areas are measured on a UNIFORM sweep and never by counting a mesh**, and the run
now measures the trap instead of asserting it. The same three cones, counted three
ways:

```
      t     exact     sweep   lat-lon cells  geodesic cells
    90°    12.500    12.500          12.500          12.480
    50°     2.902     2.903           4.818           2.773
   110°    25.550    25.549          22.526          25.503
```

Read the lat-lon column across. A lat-lon mesh has cells whose areas go as `sin θ`, so
a cell count over-weights the pole. It is **exact** at 90° — that octant is cut by the
equator and by two meridians, so the over-weighting cancels across its own boundaries,
which is the same reason `dimension_ball.png` gets away with counting quads. One panel
over, at 50°, the same mesh reports **two thirds more patch than there is**, because
that patch hugs the pole where the cells are slivers; at 110° it reports a tenth too
little. A method that is exact on the case you check it on and 66 % out on the case
beside it is worse than one that is uniformly rough. The geodesic mesh the figure is
drawn on is within half a per cent everywhere and is still not used for measuring: a
drawing mesh answers to the drawing. `sweep()` does the measuring; `shell()` only
draws.

**`covers()` is the closed form for a regular spherical triangle of side `t`.** The
spherical law of cosines gives `cos A = cos t / (1 + cos t)` and the area is the excess
`3A − π`. At 90° that is `π/2` exactly, an eighth of the sphere — and the same `A` is
what `tripod` uses to place `u₂` and `u₃`, so the shape and its area are built from one
number.

**The swept floor sits a little ABOVE the exact one below 90°, and that is the
argument rather than an error.** The floor there is AT the push directions and no
Fibonacci sample lands on one, so the sweep can only come back from just inside:
`1.003` at 50°, `1.001` at 60°, `1.000` at 90°. The swept ceiling sits a little below
the exact one for the same reason at the centroid. Neither gap shrinks by taking a
finer grid in any way worth doing; the closed forms are what the captions quote.

**Both closed forms have one-line proofs and both are in the docstrings.**

- `weakest()` — the floor is at the direction that makes ONE coefficient as large as
  possible: maximise `c₁` over `{c ≥ 0, c'Gc = 1}`. The unconstrained maximum is
  `√((G⁻¹)₁₁)`, at `c ∝ G⁻¹e₁` — but those weights are only non-negative, and so only
  name a direction the three can actually reach, when the pairwise cosine is negative.
  For a non-negative cosine the constrained maximum falls back on `c = e₁`, the push
  direction itself, and the answer is exactly `F`. **That switch is the threshold, and
  it happens at `cos t = 0`.**
- `strongest()` — minimising `maxᵢ cᵢ` over `{c ≥ 0, |Σcᵢuᵢ| = 1}` is the same as
  MAXIMISING `|Σcᵢuᵢ|` over the box `0 ≤ cᵢ ≤ 1`, and a convex function over a box
  takes its maximum at a vertex. Three distinct vertices by symmetry: `(1,1,1)` gives
  `√(3+6cos t)`, `(1,1,0)` gives `√(2+2cos t)`, `(1,0,0)` gives `1`. The first wins
  below **104.48°** (`cos t = −1/4`) and the second past it — spread wide enough, a
  PAIR pulls harder across its own bisector than all three do down the middle, because
  the third is then more than 90° from that middle and its neighbours have to cancel
  each other. At 110° the strongest directions are the three edge midpoints, not the
  centre, and that is why (c)'s three pale lenses hug the rim instead of sitting in
  the middle of the patch. The `1` keeps the ceiling from ever falling below `F`.

**The `4.9 %` in (c) is the share of the PATCH, not of the sphere.** Of the sphere it
is `1.26 %`. Quote the right one: the patch share is what says the arrangement has
stopped delivering over the very set it reaches.


## The mechanism, and why the number is 90 and not something else

The threshold is a statement about ONE PAIR. The weakest direction of a pair
`{uⱼ, uₖ}` is the one perpendicular to both, `dᵢ = uⱼ × uₖ`: neither of them can put
anything along it, so only the third support is left. As the pairwise angle opens,
`dᵢ` walks INTO the patch. Printed by the run:

```
      t       c_i   c_j = c_k   in the cone   angle to u_i    m(d_i)     cos t
    50°    1.4185     -0.5550         False         45.17°    0.0000   +0.6428
    60°    1.2247     -0.4082         False         35.26°    0.0000   +0.5000
    89°    1.0003     -0.0172         False          1.40°    0.0000   +0.0175
    90°    1.0000      0.0000          True          0.00°    1.0000   +0.0000
    91°    1.0003      0.0178          True          1.43°    0.9997   -0.0175
   110°    1.2457      0.6475          True         36.60°    0.8028   -0.3420
```

Read the third and fourth columns together. Below 90° the two other coefficients are
**negative** — `dᵢ` is outside the cone, 45.17° away from its own push at 50°, and the
constraint it carries cannot bite. At exactly 90° they are **zero** and `dᵢ` **is**
`uᵢ`, sitting on top of it to the last decimal. Past 90° they are **positive** and
`dᵢ` is strictly inside, 36.60° in from the corner at 110°, and `m` there is `0.8028`,
which is the whole of (c)'s weakness. (The `+ 0.0` in that print line is there only to
stop the 90° row showing a negative zero; the value is exactly `-0.0` and which side of
zero the last bit falls on is the arithmetic's business.)

The crossing is algebraic, not numerical. Decomposing `dᵢ` gives

    c_j = c_k = −cos t / [ (1 − cos t)(1 + 2 cos t) |u^i| ]

and every factor but the leading `−cos t` is positive over `0 < t < 120°`, the whole
range in which three symmetric directions are independent at all. **The sign therefore
turns at `cos t = 0` and nowhere else.**

**2026-08-14 — the sentence that used to follow that one was wrong and is now gone.**
It said the threshold "is 90° for three of them and 90° for any number: a fourth
support does not mend a direction that two of its neighbours are square to." It does.
What bounds the blind direction `dᵢ = unit(uⱼ × u_k)` is what the OTHER supports can
put along it,

    h(dᵢ) = F Σ_{l ∉ {j,k}} max(0, dᵢ·u_l)

which with THREE supports has exactly ONE term, `F cos β`, and so is at most `F` with
equality only when the third support IS `dᵢ` — the mutually square arrangement, and
the whole of the 90° rule. With four or more the sum acquires further terms and can
reach `F`. The **square pyramid** `{+z, ±x, ±y}` contains two exactly OPPOSED pairs,
180° apart, and its weakest direction is still exactly `F`, because the direction its
pair `{+z, +x}` is blind to is `±y` and `+y` is itself a support; meanwhile the regular
**tetrahedron**, every pair 109.47°, falls to `0.8165 F`. So past three supports the
pairwise angle is not the criterion at all. **`slides/tools/figures/more_pushes.png` is that
counterexample drawn**, and `slides/tools/more_pushes.md` carries the replacement rule and
the linear program the magnitude becomes once the decomposition stops being unique.

Nothing in this figure or in any of its numbers changed when that was corrected —
`three_pushes.png` is byte-identical, `85f3b87ef02a9a87fb9ee54ae8e61d6c`, before and
after — because the false claim lived in the prose and never in the arithmetic. Every
statement above it, about three supports, still holds exactly.

`slides/tools/more_pushes.md` carries what happens past three supports, where that
threshold stops being the criterion at all.

**Where `m = F` exactly is a curve and not a fitted contour.** `cᵢ = 1` is the plane
`v · u^i = 1`, which meets the ball in the circle centred on `unit(u^i)` — the
mechanism direction — at angular radius `arccos(1/|u^i|)`. All three circles pass
through their own `uᵢ`, since `uᵢ` decomposes as `eᵢ`; what changes with the angle is
whether anything ELSE of the circle lies inside the cone. Below 90° nothing does, so
`contour()` hands back nothing and the below-`F` region is that single touching point.
At 110° the circles cut three real lenses out of the patch and their boundary is drawn
from the algebra, not traced off a raster.

`contour()` needs **two** conditions and not one, and the second is easy to miss:
`cᵢ = 1` holds all the way round the circle, so a stretch of it can run through the
INSIDE of the region where another support is over ITS cap, where the field is below
`F` on both sides and the curve bounds nothing. Dropped, the three circles come back
as one long arc straight across the 110° patch instead of as the three symmetric
lenses that are really there. That was a real bug here, alongside a `dual()` written
as `inv(Uᵀ).ᵀ` instead of `inv(Uᵀ)` — which returns three vectors of about the right
size and so shows up as a plausible curve in the wrong place rather than as an
exception.


## Reading the picture

- **The orange arrows are the three pushes**, each capped at `F`. Same colour, same
  weight, same PAGE length everywhere, which is what says they are three of the same
  thing; what changes across the row is only where they stand. **The topmost one is
  the same arrow in all three panels** — it is the floor.
- **The ellipse across the ball is the HORIZON.** `+z` is the floor's push, so `z = 0`
  is the set of horizontal directions, seen from 20° above. In (a) the two other
  pushes sit exactly ON it; in (b) they are above it; in (c) they are under it. It is
  also the only far-side mark and so the only thing that says the shell is
  translucent.
- **The paint is `m(v)` and is read as three states with a hard break, not as one
  scale.** Red is a direction the three can push but not with a full support behind
  it; blue is one support's strength or more; **bare grey shell is a direction they
  cannot push at all**, which is a third state and not a small magnitude, and is why
  it is left unpainted rather than given the bottom of the ramp.
- **The ink curve inside (c)** is `m = F` exactly. It exists only past 90°.
- **The ink rings are `uⱼ × uₖ`.** In (a) each one lands exactly on a push dot and
  encircles it; in (b) they are outside the patch, out on the bare shell — two of them
  down on the horizon and the third round the back beside the vertical push, drawn
  faint; in (c) they are well inside the red. That walk is the threshold, drawn.
  Measured on the page in ball radii, ring to its own push's dot: `0.694` and `0.763`
  at 50°, `0.012`–`0.019` at 90° — which is only the `1.04` against `1.02` stand-off,
  i.e. exactly on top — and `0.278`–`0.458` at 110°.
- The pale corners of (a) and (b) are the fact the captions state: **the weakest
  directions of those two patches are the push directions themselves**, at exactly
  `F`, and the ramp is built to leave white fast so that they show up as bright spots
  in an otherwise deep patch.


## Decisions that look free and are not

**The colour key is shared verbatim with `more_pushes.py`.** Same two ramps, same
break at `F`, so a colour means the same thing in both figures that answer this
question. They differ in exactly one number: the blue ramp's far end. `more_pushes`'s is
`3.54 F`, the middle of its five-ray cluster; this one's is `2.62 F`, the middle of a 50°
patch, read off `strongest()` rather than typed. Neither figure may clip its own
strongest point, so that number has to follow the panels.

**The ball is tessellated GEODESICALLY, not in longitude and latitude.**
`dimension.py`'s ball uses lat-lon and is right to: its patch is nowhere near a pole.
Here the pole is where `u₁` stands — **a corner of all three patches**, and the one
mark the whole figure is posed around — so a lat-lon mesh would leave a bullseye of
converging seams exactly on it, in all three panels at once. An icosahedron subdivided
five times gives 20480 cells about 2° a side, every one within a few per cent of every
other, so the field's staircase is the same width all over the patch and the corner is
no worse than the middle — and half a cell is 1°, which is what the exact curves drawn
over it are wide enough to hide.

**The shell and the paint are TWO collections, and the split is forced.** Agg
rasterises every polygon on its own, so two that share an edge leave a hairline of
whatever is behind them between the two — 20480 cells' worth of hairline, which prints
the tessellation right across the ball. Giving each cell an outline in its own colour
closes it, but only where the fill is OPAQUE: over the translucent shell the outline
blends a second time and draws the same mesh a shade darker instead. So the painted
cells want a line width and the bare ones want none.

A per-face `linewidths` array cannot do that, and this is worth knowing before anyone
tries it again: **`Poly3DCollection` re-orders its faces by depth on every draw and
permutes the face and edge COLOURS to match, but the line widths are the base
`Collection`'s and are left where they were.** The widths then land on whichever cells
happen to be at those depths, and the ball comes out sprinkled with a few hundred
outlined triangles in both the bare and the painted parts. Turning antialiasing off is
not the answer either — with it off Agg hands the boundary pixels to BOTH polygons and
the translucent shell comes out with a dark lattice all over it. Both were tried and
both are in the crop history.

Splitting costs the one thing a single collection buys — mplot3d gives a whole
collection one averaged depth, so the two cannot interleave — and here that costs
nothing, because **the painted patch is entirely on the near side in all three
panels**. That is not luck: the patch is the cone ON the three pushes, so for a unit
`v` in it, `v·EYE = Σcᵢ(uᵢ·EYE)` with `Σcᵢ ≥ 1`, which puts the whole patch no further
from the eye than its furthest PUSH. The 84° rule on the pushes therefore *implies*
this one, and there is an `assert` on it anyway: at 110° the receding push stands
`81.59°` from the eye and the furthest painted cell centre is `80.36°`, nine and a half
degrees clear of the limb.

**Only the arrows' LENGTH is corrected for foreshortening, and their tails are not.**
A radial arrow on a ball has no foreshortening cue on it to contradict — it is one
straight mark pointing away from the middle of a disc — so giving it the 3-D length
whose PAGE length is the one wanted costs nothing. Uncorrected, the nine pushes keep
between `0.457` and `0.989` of their length and the reader is handed nine arrows of
nine different lengths all standing for the same `F` — which would be worse here than
anywhere else, because the three pushes of one panel are drawn at three angles to the
eye ON PURPOSE, and a reader who took length for magnitude would read the pose as the
physics. The tail still stands at `1.03` on the shell, because that is a POSITION, and
positions are the one thing this figure projects honestly. `SEEN_FLOOR` no longer binds
in this pose — the closest push to the axis keeps `0.457` — and is a guard only, kept
so that moving the camera degrades the figure instead of exploding it.

**The arrows are `Arrow3D`, not `quiver`, for `dimension.py`'s reason.** mplot3d builds
a quiver head out of two barbs in a plane it picks from the shaft alone, so the more
nearly an arrow points at the reader the more nearly its head is edge on. The nine
pushes stand between `27.17°` and `81.59°` off the camera's axis on purpose, so nine
quiver heads would come out at nine different weights.

**Orthographic.** mplot3d's default is a perspective projection at focal length 1,
strong enough that the near side of a ball of radius 1 draws visibly larger than the
far side — and this figure asks the reader to compare the AREA of three patches which,
in this pose, sit at three different distances from the view axis: their centroids
stand `29.2°`, `54.7°` and `71.1°` off the vertical push, so they cannot all face the
camera. Under perspective the reader would be comparing three areas scaled by three
different factors he was given no way to know about.

**`ZOOM` is capped by the CAPTION, and `LOWER` is what made that possible.** The
framing box has to stay a CUBE — mplot3d rescales whatever box it is given to a fixed
diagonal and then stretches the result to fill the axes rectangle — which leaves two
levers, how big the cube is drawn and where its centre is put. Measured over all three
panels the drawing runs from `−1.000` to `+1.388` ball radii vertically and from
`−1.062` to `+1.305` sideways, and it is **not** centred on the ball, because the
vertical push now stands out of the top of every panel and nothing balances it
underneath. So the cube is lowered by `LOWER = 0.28` of a radius along world `+z`,
which is page-up. Without it the drawing runs to `0.906` of the page height and the
arrowhead is drawn straight through the second line of the caption — measured, and it
is what this revision's first render did. With it the drawing occupies `0.286`–`0.844`
of the page, clearing the caption line above by `0.018` and the number line below by
`0.019`; `ZOOM = 1.80` is then capped by the two of them together.

**The captions are formatted from the same values the paint is**, so the words under
each picture cannot drift from the colours in it, and the swatches in the key are put
through the same Lambert term and the same transparency the ball is (`onpaper()`)
rather than being eyeballed to match. Panel (a)'s "= exactly one eighth" is written
only where it is exactly true — the tail is guarded on `covers(90)` landing on `12.5`
to machine precision, and no other angle in the figure lands on a round fraction at
all.


## Known open issues

- **Two of (c)'s three pale lenses are cut by the patch rim and read as thin claws.**
  That is geometry and not drawing: the strongest directions at 110° are the three
  EDGE midpoints, so each lens hugs a boundary arc and is half a lens by right. Any
  camera puts at least one of them there.
- **(c)'s receding push is close to the limb, `81.59°`, and its corner of the patch is
  squashed against the silhouette.** That is the price of `u₁ = +z`: the 110° tripod's
  3-fold axis stands `71.06°` off the vertical, and the eye is at `70°`, so the patch
  is seen from near its own edge. It is `8.4°` from the wall, and the 84° rule in
  `pose()` is what stops anyone walking into it.
- **The painted cells still show a faint 2° texture** where the field is steepest, the
  colour being taken at each cell's centre. It is most visible in (b), whose patch is
  small and whose field is steep. It is under a per cent of the fill and does not
  survive being looked at from a normal distance; the fix would be a finer mesh at a
  rendering cost, or a per-pixel raster, which cannot be translucent.
- **The row is not centred on the page before `trim()`.** Every panel's arrows reach
  further right (`+1.305` radii) than left (`−1.062`), so the balls sit left of centre
  in their own rectangles; `trim()` crops to the ink and the printed figure is fine,
  but panel (c)'s rightmost arrowhead clears its axes rectangle by only `0.007` of the
  figure width, and anything that lengthens the arrows will clip it rather than
  overflow it — everything in a 3-D axes is clipped to that rectangle.
- **The output path is absolute and hard-coded** to `/Users/yuanboli/.../figures/` in
  `main()`, as everywhere else in `slides/tools/`.
- **`VMAX` is a module global set in `main()`.** `ramp()`, `key()` and `onpaper()` all
  read it, so importing the module and calling `ball()` without going through `main()`
  raises. It is a global because the ramp's top has to be derived from `CASES` rather
  than typed, and threading it through five call sites bought nothing.


## Where it sits beside the other figures

| file | what it answers |
|---|---|
| `dimension_ball.png` | WHICH directions a contact set reaches — point, arc, patch |
| `dimension_pairs.png` | how hard, for TWO pushes, as a fan along the arc |
| **`three_pushes.png`** | **how hard, for THREE, as a field over the patch** |
| `more_pushes.png` | how hard, for FOUR and FIVE — clustered, and spread right out |

`dimension_pairs.png` puts the magnitude in the LENGTH of an arrow because a
two-generator cone is a curve and a curve has room for arrows along it. A
three-generator cone is a region, arrows over a region collide, and the magnitude goes
into the paint instead. That is the only reason the two figures encode the same
quantity differently, and neither encoding transfers.

**`ninety.py` and `ninety.png` were deleted on 2026-08-14.** They were an unasked-for
figure from an earlier pass — the 90° threshold as a continuum, plus a tetrahedron panel
— and `more_pushes.png` supersedes both halves, with the mechanism behind the threshold
recorded in `more_pushes.md`. Their numbers had agreed with this file's where the two
overlapped (`4.4 %` at 60°, `12.5 %` at 90°), which is how the agreement was checked
before they went. Nothing references them any more.


## The file

| file | md5 | note |
|---|---|---|
| `three_pushes.png` | `85f3b87ef02a9a87fb9ee54ae8e61d6c` | three balls, 2650 × 1292 |


## 2026-08-14 — the shell went clear, and the inside got drawn

On the human's instruction: "我要透明的球，那就是内部是白色透明的，然后可以看到顶点与圆心的
连线和角度". The ball is now WHITE at **13 % alpha** with its Lambert term nearly all
ambient (`SHELL_LIGHT = 0.94, 0.06`), because shading is what made the old grey shell read
as a solid, and a solid is what this is not.

Two things had to be given back once the fill stopped carrying the form:

- **The limb** — the great circle square to the eye, which is the ball's outline. A grey
  fill ended somewhere and that was the edge; a white fill at 13 % on near-white paper has
  no edge at all, and without the limb the ball reads as a patch with some arrows near it.
- **`interior()`** — the three radii out of the centre, the three arcs of radius
  `MARK = 0.34` standing between them, the centre itself, and the angle written once.

**The radii are DASHED and drawn over everything, and that is a deliberate lie about
depth.** Honest depth hides them: at any point of the disc the sphere's near surface is at
`sqrt(1 − r²)` and every interior line is nearer the centre than that, so the opaque patch
occludes all three radii wherever it covers them — which in (a) and (c) is the whole
middle of the ball. Drawn properly buried they showed *nothing*; the first render of this
change had visible radii only in (b), where the patch is small and high. A dash is the
ordinary way to say "behind what it crosses", it costs no accuracy, and it is the only
reading that survives an opaque patch. The patch stays opaque for the reasons already
recorded: a colour read through a translucent shell is not the colour on the key, and a
translucent mesh draws its own tessellation in hairlines.

**Only one angle is labelled per panel.** All three pairs are equal by construction and
`tripod()` asserts it to nine decimals, so three copies of the same number is noise. The
one that gets the label is the arc whose midpoint projects furthest from the other two, so
the text never lands on a radius; it carries a paper-coloured box because in (a) and (c)
it falls on the patch.

The three arcs are a scaled copy of the patch's own rim, which is not a coincidence and is
worth seeing: **the solid angle at the centre and the patch on the surface are the same
object read at two radii.**
