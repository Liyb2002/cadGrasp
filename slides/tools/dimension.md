# dimension.png — what it argues, and what not to break

Handoff for `slides/tools/dimension.py` and the five PNGs it writes into `slides/tools/figures/`.
Everything below was re-derived on 2026-08-13 by reading the script and running it, the
pairs figure again on 2026-08-14 when the fan went in, the scene figure the same day when
the row of producible forces went onto the body, and the pairs figure's TOP row a second
time on 2026-08-14 when the blue fan came off it again and the capped/uncapped pair of
percentages went into its captions; numbers are from those runs or from checks against the
code, not from memory.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp/tools
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u dimension.py          # ~3 s, rewrites all five PNGs, prints every number
```

The script is deterministic — there is no randomness anywhere in it — and a clean re-run
reproduces every file byte for byte. If a run changes a file you did not intend to
change, that is a real regression, not sampling noise.

**2026-08-13 — the body was laid down.** Every earlier version balanced it on a single
corner. That premise is gone, on the human's instruction, and the whole camera half of
the script went with it. What replaced it, and why the replacement was forced rather than
chosen, is the first section below. The two plane figures are untouched and still
byte-identical.


## The argument

The statics are translation only. A contact contributes exactly one thing, the
direction it pushes; where it touches never enters the equations. The floor is a free
generator pointing straight up. A disturbance `d` of one body weight is survived when
the contacts can produce

    T = up - d      as a non-negative combination of the generators

and not otherwise. `gap()` measures the distance from `T` to that cone; zero means
survivable. With at most three generators the non-negative least-squares projection is
the best of eight ordinary solves (the empty subset — supply nothing — plus the seven
non-empty ones), which is why there is no LP anywhere.

Every support adds one generator, and **every generator adds a dimension** to the set
of survivable disturbances. The floor alone gives a ray, so the set is a single
direction: a point. One support opens the ray into a wedge, so the set is an arc. Two
supports open the wedge into a solid cone, so the set is a patch. That progression —
point, arc, patch; 0, 1, 2 dimensions — is the whole figure. A table of percentages
destroys it, because an arc has no area and would be entered as 0 %, which is exactly
the reading the figure exists to refute.

**But "every generator adds a dimension" is only true while each new push points out of
the span of the ones before it, and that is a real condition, not a technicality.** A
push already inside that span adds reach without adding dimension, and the worst case is
an opposed pair, which adds *nothing at all* to the directions available: non-negative
combinations of `u` and `-u` are the multiples of `u`, so the cone is a **line** and it
meets the ball in **two points**, not in an arc. Swept at n = 200 000 and clustered at 6°:

```
generators                    hits   pieces   across
{u}                              4     1        0.7 deg     a point
{u, 179 deg away}              791     1      179.4 deg     an arc, nearly a half circle
{u, -u}                          7     2        0.6 deg     TWO POINTS -- the arc collapses
{up, u}                        399     1       90.5 deg     an arc
{up, u, -u}                    793     1      179.9 deg     three pushes, STILL an arc
{u, v, -u, -v} coplanar       1586     1      360   deg     four pushes, still no area
```

The conic hull is not continuous in its generators: 179° apart gives an arc 179° long and
180° apart gives two isolated points. So three supports do not have to give a patch, and
four do not either — `{up, u, -u}` is three generators and still zero area, and four
coplanar ones close a whole great circle and still have zero area. The figure's
configuration is safe because its three pushes stand **mutually square** and so are in
general position; anyone changing `PROPPED` has to check that they still are.

**Nothing physical jumps there, though, and it is worth knowing why.** The arc's LENGTH
runs smoothly up to the wall — 90.4°, 120.6°, 150.6°, 170.7°, 179.4°, 179.9°, then two
points — but what can actually be pushed along it collapses smoothly to nothing well
before. Two pushes `t` apart, a cap of `F` on each: a unit direction `φ` along the arc
decomposes as `a = sin(t−φ)/sin t`, `b = sin(φ)/sin t`, and the cap binds on the larger of
them, so it reaches `F / max(a,b)`.

```
  t       arc      at the middle    the WEAKEST on the arc    where that sits
  60 deg   60 deg    1.7321 F           1.0000 F              phi =  0 deg, a push itself
  90 deg   90 deg    1.4142 F           1.0000 F              phi =  0 deg
 100 deg  100 deg    1.2856 F           0.9848 F  = sin t     phi = 10 deg
 120 deg  120 deg    1.0000 F           0.8660 F  = sin t     phi = 30 deg
 150 deg  150 deg    0.5176 F           0.5000 F  = sin t     phi = 60 deg
 179 deg  179 deg    0.0175 F           0.0175 F  = sin t     phi = 89 deg
```

**The whole arc takes at least `F` only while the two pushes are at most 90° apart.** Past
90° the floor is `sin(t)·F`, and it does NOT sit at the bisector — it sits at `t − 90°`,
square to the other push. (An earlier pass here quoted 120° for this. 120° is where the
*bisector* drops to `F`, equivalently where a rhombus's diagonal equals its side; it is a
real number about the middle of the arc and the wrong one for the guarantee.)

The figure's own pushes stand exactly 90° apart, which is the last arrangement where the
guarantee holds: over the whole octant the reachable magnitude runs from `F` to `√3 F`,
and the minimum sits precisely at the three push directions. Anyone spreading `PROPPED`
wider loses it immediately.

**Narrowing below 90° does not buy a better floor, and it is worth knowing why before
anyone tries.** Pushing exactly along one support's own direction decomposes as
`c = (1,0,0)` — uniquely, since three independent generators leave no other way — so that
direction reaches exactly `F` whatever the angles are. Checked at 50°, 60° and 90°: all
three return `F/1.000000`. The floor is pinned at `F` for every arrangement up to 90° and
cannot be raised. What narrowing moves is the *interior* and the *coverage*, in opposite
directions:

```
 pairwise t   the cone covers   weakest in it   strongest in it
    50 deg        2.902 %            F              2.61 F
    60 deg        4.387 %            F              2.44 F
    75 deg        7.556 %            F              2.13 F
    90 deg       12.500 %            F              1.73 F   <- the figure
   100 deg       17.554 %         0.963 F           1.40 F
   110 deg       25.550 %         0.803 F           1.15 F
```

The coverage column is checked against the closed form for a regular spherical triangle
of side `t` — `cos A = cos t/(1+cos t)`, area `3A − π` — which reproduces the swept figures
to three decimals, and gives exactly `12.500 %` at 90°, the octant. **Measure it on a
uniform sweep (`sweep()`'s Fibonacci spiral), never by counting `shell()`'s quads**: those
are equal in longitude and latitude, not in area, so a quad count weights a cell by
`sin θ` and the answer depends on where the patch sits. An earlier pass here reported
10.6 / 13.2 / 22.9 / 33.7 % from a quad count; every one of those was wrong.

The trap has a sharp edge worth recording, because it decides whether an existing number
in this file is safe. Counted on `shell()`'s quads, the octant comes back as **12.500 %
when it is axis-aligned** — the `sin θ` weighting cancels across cuts that follow the
mesh's own lines — and as **22.786 % when the same octant is centred on the pole**. So
`balls()`'s printed 12.500 %, which measures the figure's own axis-aligned octant on that
grid, is exact and stays; it was the pole-centred trios of the later check that the quad
count ruined. On a uniform sweep both read 12.500 %, which is the reason to use one:
it does not care where the patch is.

So it is a straight trade — narrower is stronger in the middle and reaches fewer
directions; wider reaches more and past 90° some of them can no longer take a full
support — and 90° is the corner of it. For this figure the angle is not free anyway: a
box lying on its broad face has vertical sides, so every chock's push is horizontal and
therefore square to the floor's, and adjacent sides are square to each other. All three
pairs are 90° by the geometry rather than by choice, and getting below it would mean
chocks that push up-and-inward — shims under an edge rather than blocks beside it.

At 179° the pair reaches 179° of directions and can put **1.7 % of one support's strength**
into them; only 1.1 % of that arc is good for half a support. So the discontinuity is in
the SET and not in the capability, and any statement about what a support arrangement is
worth should be made about the capped reach, not about the angular size of the cone —
which is the same lesson as the cube against the simplex.

**And the collapsed case is not a failure of the supports, only of this way of counting
them.** `u` with `-u` can supply *any* force along that line, both signs, any magnitude —
opposed contacts, form closure along one axis, the strongest thing two supports can do to
one degree of freedom. They buy completeness along one axis where a square pair buys
extent across two, and a dimension count sees only the second. `slides/tools/opposing_supports.py`
is where the first belongs.

**Any change that flattens the point → arc → patch progression is wrong, however good
it looks.** That includes anything that makes panel 1 look like it has a range of
capability, anything that makes panels 2 and 3 look like the same kind of thing, and
anything that turns the counting of arrows into an estimate of area.


## The pose, and why it is not a free choice

The plate **lies on its broad face**. `HALF3 = [0.76, 0.70, 0.14]`, ratios
5.43 : 5.00 : 1, and `LEVEL = argmin(HALF3) = 2` is the axis it lies on. Everything else
about the pose is derived from that one index.

It got there the long way, and the reasoning is worth keeping because it closes the
question rather than settling it by taste.

**Balancing a flat body on itself stands it on end.** Balance means the centre of mass
sits over the contact, so the tilt works out to `arccos(h_2/|h|)` — and that runs to 90°
as the plate is thinned. Measured for this plate: **82.28° on a corner**, and resting on
an edge does not escape it (**68.5°** and **66.8°** for the two bottom edges). Thinning
made it worse, not better, and did not even shrink it: taking `h_2` from 0.30 to 0.14
moved the body's page height from 1.99 to 1.97 while standing it 6.1° straighter, because
the page size follows the diagonal `2|h|`, which the two broad extents own.

**So there is exactly one pose that shows a flat plate whole**, and the figure now takes
it. Lying down, the body covers **2.07 wide × 1.44 tall** on the page where standing it
covered 1.64 × 1.97.

**The statics do not notice.** A contact enters the model as a push direction and nothing
else, so a broad face resting on the floor is the same single upward generator that a
single corner was. Panel 1 still shows one surviving direction. What is lost is the
rhetorical hook — *a legitimate equilibrium, and still only one disturbance survives* —
and what replaces it is at least as good and considerably more intuitive: a plate on a
frictionless floor slides under any sideways push at all.


## The ball — `dimension_ball.png`

**The scene figure cannot say which way an arrow points, and this one can.** An
orthographic view collapses a direction's inclination and its azimuth onto the single
angle the arrow is drawn at, and the collapse is not even order-preserving. Measured on
the panel-3 set at the camera the figure uses:

```
true inclination      drawn on the page
    -36.9 deg              -15.1 deg
    -16.6 deg              +45.0 deg     <- leaning DOWN, drawn above the horizontal one
      0.0 deg              +30.4 deg
    +19.5 deg              +90.0 deg     <- drawn straight up the page
```

Three remedies were tried on the scene and none of them works. Dropping each arrow's plan
projection onto the floor: the projections are hidden under the body and the arrows
themselves, and at 21 arrows they read as clutter (`scratchpad/plan_6_tie.png`). Lowering
the camera: **worse**, not better — scoring by the largest inversion, a lower-inclined
arrow drawn further up the page than a higher one, elev 36° gives 104° and elev 18° gives
161°, because a low camera flattens "away" onto "along". Fewer arrows: relieves the
crowding, does nothing to the inversion. The collapse is what an axonometric projection
of a two-parameter set of directions *is*; no amount of care with the arrows removes it.

**So the ball is drawn from the other end: it shows the cone the contacts span.** A
contact can pull nothing, so what a set of them can supply is the non-negative
combinations of their push directions — a cone, whose trace on the ball is exactly the
region *between the arrows*. `spanned()` reads it off `gap`, the same solver the counts
and the scene use, so the painted region and the percentage are one claim.

An earlier pass painted the *survivable disturbances* here instead, in blue. It was
dropped as unreadable: the reader has three orange arrows in front of them and a blue
region that is not between them, on the far side of the ball, related to them by a
reflection nobody can perform by eye. **The cone is the same counting and it is the one
the arrows explain themselves.**

**All three answers are exact.** The floor pushes straight up and each chock pushes along
a horizontal it shares with no one, so the three generators stand mutually square:

| | the set | what it is | area |
|---|---|---|---|
| one force — the floor | `{up}` | **one direction**, a point | 0 |
| two forces | the arc from `up` to `u₁` | an **arc**, 90° | 0 |
| three forces | `{a·up + b·u₁ + c·u₂, a,b,c ≥ 0}` | the **spherical triangle** they close on | an octant, **12.500 %** |

Measured off the very tessellation that is painted: **0.000 %, 0.000 %, 12.500 %**. The
middle panel's arc is one of the three edges of the third panel's patch, and every panel
draws its edges, so the reader sees the arc opening rather than being told about it.

**Two figures, two percentages, and that is correct.** The ball's 12.5 % is the cone of
contact forces; the scene's 25 % is the set of disturbances that cone survives. They are
in bijection — `d = up - T` renormalised — so they always have the same *dimension*,
which is the entire argument, but that map does not preserve area and there is no reason
the two numbers should agree. Do not quote one for the other.

**The camera looks straight down the cone's own axis**, `axis()`, which for these three
generators is elevation 35.26° and azimuth −45° — the isometric direction, because they
stand mutually square. That is what puts **all three arrows on the near half of the ball**
with none foreshortened more than another, and the patch facing the reader square instead
of hiding round the back. It is not the scene's camera (36°, −45°) and it should not be;
it is derived from the generators, so it follows them if they ever change.

**`ball()` is shared with `dimension_pairs.png`, and everything the pairs figure needed
extra went in behind one opt-in flag, `fan=False`.** That is not tidiness. This panel's
whole subject is the point → arc → patch progression, and a fan of capped magnitudes
drawn across it would argue a second thing in the middle of the first. With the flag off
the function is what it was, and `dimension_ball.png` is byte-identical across the
change. What the flag turns on — the fan itself, an orthographic projection, and a
generator arrow measured on the page rather than in the world — is in the pairs section
below.

Two drawing points that are not free. **The shell is shaded**, by the same Lambert term
the solids take: filled with one flat colour a ball projects to a disc and the reader has
to take the third dimension on trust from the equator alone. **The arcs are cut at the
silhouette and drawn twice**, near half over the shell and far half under it at 45 %
alpha, because mplot3d gives a whole line one depth and would put all of it in front —
the far half then reads as a loop floating over the ball instead of a curve lying on it.
The paint itself needs no such care: it is one collection with a colour per quad, which
is the only way a translucent ball sorts against itself. The region takes the arrows' own
hue washed out, `FAN` against `ORANGE`, because it is what they span and not a second
thing.


## The three pairs — `dimension_pairs.png`

Six panels, three columns: a scene on top, its ball underneath, for two pushes **180°,
175° and 90°** apart. It exists to make one thing impossible to miss — **the first two
columns are five degrees apart in setup and nothing alike in result.** The two scenes are
all but indistinguishable; the two balls are two isolated points against an arc that runs
right round the front of the ball.

- **180°** — the cone is a LINE, so it meets the ball in **two points**. Not a short arc,
  not a thin one: no arc at all.
- **175°** — an arc **175°** long, five degrees short of a half circle.
- **90°** — an arc **90°** long.

All three have zero area, measured 0.000 % on the same tessellation the ball figure paints.
That is the point of putting them in a row: three quite different sets, one number.

**2026-08-14 — the balls gained a fan, and it is the half of the answer the arc was
leaving out.** The arc says WHICH directions the pair reaches and nothing about how hard,
and on its own that is not merely incomplete, it is the misleading half: it scores 175°
as nearly a half circle of capability when the pair can barely push across the middle of
it at all. Every direction in the region now carries an arrow along itself, standing off
the shell, whose **length and line width both run with the largest force that can be
produced along it**.

**The quantity.** Every support is the same physical thing and takes the same cap `F`. A
direction inside the cone is `v = Σ cᵢ uᵢ` with every `cᵢ ≥ 0`, and scaling the whole
combination scales every coefficient together, so the cap binds on the largest and

    m(v) = F / maxᵢ cᵢ

For two pushes `t` apart, a unit direction `φ` along from the first has
`c = (sin(t−φ), sin φ) / sin t`. That was checked three independent ways before it was
drawn — the closed form, `strongest()`'s subset walk, and a brute-force walk of the
boundary of the capped set `{c₁u₁ + c₂u₂ : 0 ≤ cᵢ ≤ F}`, which knows no formula at all.
They agree to 1e-9, 1e-4 for the brute force, which is grid-limited. Every sampled
direction also comes back with `gap = 0`, so the fan hangs on exactly the set `spanned()`
paints and not on a second one. Printed by the run:

```
  t        at each end     across the middle    2 cos(t/2)
 180 deg    1.0000 F        no middle at all     --      the cone is a LINE; the two
                                                         ends ARE the whole set
 175 deg    1.0000 F        0.0872 F             0.0872
  90 deg    1.0000 F        1.4142 F             1.4142
```

**A push direction is the WEAKEST place on its own arc, not the strongest**, and that is
the fact the fan is drawn to make unmissable. Both ends of all three arcs are worth
exactly one support, `1.0000 F`, because a push can be supplied by its own support working
alone. What the second support adds is everything in between — `√2 F` at 90°, and
`0.0872 F` at 175°, an eleventh of what a single chock gives. So the 90° column swells in
the middle and the 175° column is full-size at the two ends and gone to hairline stubs
between them, which is the picture of why an arc of 175° is worth almost nothing while an
arc of 90° is worth more than the two pushes that make it. (`0.0872` is `2 cos(t/2)`; the
arc's true floor is `sin t = 0.0872` at `φ = t − 90° = 85°`, the same number to three
places at this `t`. They part company well below 175 — see the table further up.)

**The fan is sampled by ANGLE, every 9°**, so the pitch is the same in every column and a
longer arc holds more arrows: 18 arrows at 175° (pitch 9.21°), 9 at 90° (pitch 9.00°), and
**none at 180°**, where the answer is not an omission — the cone is the line, it meets the
ball in the two generators themselves, and those are already drawn. **The two ends of
every fan are the generator arrows**, which is what anchors the scale: a generator arrow
is drawn at the length `F` earns, so the reader has the unit in front of them and an arrow
longer than one is a direction the pair pushes harder than either push alone.

```
             m        line width       page length     (what it is)
 90 deg   1.0000 F     2.20 pt           113 px        both ends, the generators
          1.4142 F     3.11              160           the bisector
175 deg   1.0000 F     2.20              113           both ends
          0.3550 F     0.78               40           one 9.21 deg step in
          0.0872 F     0.75  (floored)     10          the middle
```

Those page lengths are what the arrows are *given*; each draws about 4 % shorter than
that, and the stubs about 19 % shorter, for the reason in the known issues below.

**Both cues, because neither survives the whole range alone.** Below `0.341 F` the width
floor `FAN_THIN = 0.75 pt` binds — 16 of the 18 arrows in the 175° column, none in the
90° one — because 0.087 F asks for 0.19 pt and a 0.19 pt line does not print. Across that
weak middle the width has stopped counting and only says "small", and the length, which
is left un-floored on purpose, is what still carries the number. At the other end the
width is what makes the swell unmistakable at a glance. A hairline that is also a stub is
unmistakably a stub; a hairline of full length reads as a drawing error.

**The length is a PAGE length, and that is `onpage()`.** An arrow's 3-D length cannot be
read off the page — the projection scales it by `sqrt(1 − (d·v)²)`, which here runs from
1.000 for a push lying across the view down to **0.669 at the bisector of the 90° pair**,
which is exactly the direction that most needs drawing long. Left uncorrected the 90° fan
swells 41 % in truth and **11 % on the page**, and the two ends of the 90° column come out
15 % shorter than the two ends of the 175° column although both stand for the same `F`. So
each arrow is given the 3-D length whose *page* length is the magnitude. Nothing is lost
by it: a radial arrow on a ball has no foreshortening cue on it to contradict. Only the
length is treated this way — the tail still stands at 1.03 on the shell, because that is a
POSITION, and positions are the one thing this figure projects honestly. The clamp
`FAN_SEEN = 0.55` is a guard against a camera change and is never reached; the worst case
measured is the 0.669 above.

**The fan panels are drawn orthographic**, `set_proj_type("ortho")` under the same flag.
mplot3d's default is a *perspective* projection at focal length 1, strong enough that the
near side of a ball of radius 1 is visibly larger than the far side. That cost nothing
while every arrow was the same length and only its direction was read; it is fatal the
torque a length carries a number, and the worst case is again the bisector of the 90°
pair, aimed nearly down the camera's own axis. `dimension_ball.png` keeps the default,
which is one of the reasons it is byte-identical across this change.

**The fan arrows are `Arrow3D`, not `quiver`, and that is not a preference either.**
mplot3d builds a quiver head out of two barbs in a plane it picks from the shaft alone, so
an arrow aimed at the reader has its head projected edge on and simply loses it — the
first render came out with the strongest arrow in the figure drawn as a bar with a square
end. A `FancyArrowPatch` re-projects both ends and draws its head on the page. `shrinkA`
and `shrinkB` have to be zeroed with it: the default pulls 2 pt off each end, and the
weakest arrows here are 3.4 pt long in total.

**The box is still ±1.5 and that looks like luck, so it is worth writing down why it is
not.** A fan arrow reaches `1.03·seen + 0.44·m` from the middle of the page, and the two
terms trade against each other — the arrows drawn longest are the ones pointing most
nearly at the reader, whose tails the same foreshortening has pulled in. Measured across
all three columns the largest is **1.4700**, which is the generator arrows lying across
the view, exactly what the figure already framed before the fan existed. The fan changed
no ball's size, and nothing since has either except the caption: five lines of it needed
`subplots_adjust`'s `top` at **0.81** instead of 0.83, which took the balls from **475 to
464 px** across and the saved image from 1561 to **1547 px** tall. (Measured as the shell's
vertical extent on the trimmed raster, column 1. An earlier pass wrote 514 px here; that is
not what this measurement gives at either `top`, and whatever it counted, the point it was
making — that the fan did not resize the balls — is still right.)

**The captions carry the measured numbers**, the magnitude formatted from `strongest()` at
the bisector and the two percentages from `carried()` over a sweep, rather than typed, so
the words under the picture cannot drift from the arrows in it.

**The contacts are on the body's vertical edges in the first two columns and on its faces
in the third**, and that is forced rather than styled. `touching()` takes the support
point in the direction the push comes from, which for a face normal averages the four
vertices that tie and gives the face centre, and otherwise gives an edge. A plain box has
parallel sides, so two FACE contacts on opposite sides are exactly 180° apart and nothing
between 180° and the corner cone is reachable that way — **175° has to be taken on an
edge**, or on a tapered part, or through friction. The 90° pair, whose pushes are square
to two faces, gets face contacts.

**Each pair's spin about the vertical is derived, not chosen.** It changes no angle
between the pushes, so it is free, and `apart()` spends it on legibility: it maximises the
shorter of the two arrows' page lengths subject to the bisector still facing the reader.
The reason is the one this figure keeps running into — an arrow near the view axis has no
length on the page, and a *horizontal* push drawn near it comes out pointing **down** the
page, which is the one thing it is not. What the rule picks, at elev 42° and azim −45°:

```
180 deg   pushes at azimuth -135.0 and  +45.0   both drawn at 100 % of length
175 deg   pushes at azimuth  +42.5 and -132.5   both at 100 %, bisector faces us by 0.74
 90 deg   pushes at azimuth    0.0 and  -90.0   both at  85 %, bisector faces us by 0.74
```

A 180° pair has no bisector to constrain, so the rule puts both ends on the silhouette
where an arrow keeps all of its length. The elevation is 42° rather than the scene's 36°
for the same reason: a horizontal push is at least `elev` off the view axis whatever its
azimuth, so it keeps at least `sin(elev)` of its length — 50 % at 30°, 67 % at 42°.

Those percentages are what `apart()` optimises and they still describe the **scenes**. On
the **balls** they no longer describe what is drawn: with the fan on, a generator arrow is
`UNIT_LN / onpage()` long, so all six of them are drawn at the same 113 px whatever their
azimuth, and the 90° column's two arrows are 17 % longer on the page than they were before
this change. They have to be — they are the two ends of a fan that means the same thing in
every column, and an anchor that changed length from column to column would be a lie about
`F`. It does not let `apart()` off: a badly turned pair still throws the *arc* round the
back of the ball, which is the constraint that rule is really for.


## The blue family on the pairs scenes — two shoves, and a question mark

**2026-08-14, second pass on this row. The fan is gone.** It went in on the human's
instruction ("加上试试") as the whole survivable set, swept off the sphere and thinned to
34° by `spread_out()` — **10, 21 and 13** arrows across the three panels, re-derived here by
re-running that sampler before it was deleted — and the verdict on it was
**有点儿丑，算了吧**: it is ugly, drop it. What is drawn instead is narrower and worth more:
**here are two shoves, and 180° and 175° cannot hold them.**

**The floor is still in the generator set, and still only for this row.** Two horizontal
pushes on their own would have to produce `up − d` out of a purely horizontal wedge, which
forces `d` straight up — the one direction `measure` throws away. Swept, the pair alone
leaves 20, 157 and 87 of 40000 directions, all of it tolerance residue on an empty set.
So the SCENE row is floor + two chocks, while the BALL row stays the two chocks alone,
because that pairwise geometry is what the figure is for. **The two rows answer different
questions and the captions say so on every column.**

### the model, and why the chocks are capped and the floor is not

A disturbance `d` is held when

    up − d = c₀·up + c₁u₁ + c₂u₂,   every c ≥ 0,   c₁ and c₂ ≤ F

**The chocks take a cap of one body weight each; the floor takes none.** That asymmetry is
not an oversight and it is the whole of what this row now argues. The floor is already
carrying the weight, it is the one generator no arrangement has to earn, and every other
section of this file rests on it staying a free generator. A chock is a block like any
other and can be leaned on only so hard, and `F` is the same unit the fan on the balls
below is drawn in, so the two rows are priced alike.

**`carried()` measures the rank before it solves, and that is not fastidiousness.** At 180°
the two pushes are exactly opposite, the three generators are **rank 2**, and
`np.linalg.solve` on that matrix returns coefficients of order **1e15** with no warning —
it reports directions as held that the contacts cannot touch. So the deficient case is
answered properly: `lstsq` gives one solution, the null space gives the line of all the
others, and each coefficient's box cuts an interval out of that line; the contacts can do
it exactly when the intervals overlap. Checked against a linear program —
`scipy.optimize.linprog`, `A_eq = G.T`, `bounds = [(0,None),(0,F),(0,F)]` — over all 20000
directions of all three columns, capped and free: **120000 programs, not one
disagreement.** The closed form is what ships because it vectorises and this script sweeps
the sphere several times a run.

### the table that is the figure

```
              generator rank    force free     chocks capped at F
  180 deg           2              0.00 %           0.00 %
  175 deg           3             48.61 %           1.58 %
   90 deg           3             25.00 %          25.00 %
```

**175° holds nearly half of everything while the force is free and 1.58 % of it once each
chock is limited to one body weight — a collapse by a factor of thirty (30.7, printed by
the run). 90° loses NOTHING to the cap. 180° holds nothing at all, ever.** Those six
numbers are the three captions, formatted from the measurement rather than typed.

**The free column has a closed form and the sweep reproduces it.** The cone two pushes `t`
apart span with the floor is a spherical triangle of sides 90°, 90° and `t`, so it covers
`(t + 90 + 90 − 180)/720` of the ball; the survivable set is **exactly twice** that —
checked at 40, 60, 90, 120, 150 and 175°, coming back at 2.0000 every time — which is
`t/3.6 %`. So 175° is **48.611 %** and 90° is **25.000 %** exactly. Measure the cone by
counting `shell()`'s quads instead and 175° comes back as 24.479 against its true 24.306:
the same `sin θ` weighting trap this file warns about further up, in a fresh place.

**At 180° that closed form runs smoothly on to 50 % and the true answer is 0.** It is the
arc-versus-two-points discontinuity again, in a second quantity: the generators are rank
2, the survivable set is a CURVE on the sphere, and a curve has no area.

**`PAIR_N = 20000`, not the `N = 40000` the rest of the script uses**, and the reason is
that it is the density which reproduces every closed form to the two places the captions
print: 48.610 and 25.005, printing as 48.61 and 25.00. At 40000 the sweep gives 48.615 and
rounds **away** from its own closed form, to 48.62.

**The 1.7 % the 180° caption used to carry was a tolerance band, not a result**, and it had
to go. `measure()` counts a direction as surviving when the true set passes within one grid
step of it, which is what makes a point, an arc and a patch comparable at all — but quoting
that count as a percentage of the sphere, for a set of measure zero, reports the width of
the ring rather than the size of the set. Nothing in this row goes through that band any
more. **Do not put a number from `measure()` into a caption on this figure.**

### the two shoves themselves

**Exactly two arrows, the same two in all three panels, and the only thing that changes is
whether they carry a question mark.** Both are HORIZONTAL — a shove against a plate on a
frictionless floor is what a chock is for — and both stand 25° either side of straight away
from the reader, `SHOVE`, derived from the camera and from nothing else.

*Away* is where they have to be. `apart()` turns the 90° pair so its bisector faces the
reader, so the quarter sphere that pair survives, `{d·u₁ ≤ 0, d·u₂ ≤ 0}`, is centred on the
direction pointing away — the `pressed()` cost this figure has always paid. They keep
**0.739** of their length on the page because of it. 25° is the trade either side: wider
and one chock goes idle as the shove lines up square to the other, narrower and the two
arrows stop being told apart. At 25° they are **50° apart in the world and 70° apart as
drawn**, and they land on the two *different* faces the reader can see — so `anchors()`
finds one arrow per face, gives each the centre of its own face, and does it identically in
all three panels, which is what lets the reader see they are the same two.

Printed and **asserted** by every run:

```
                          shove A  (-0.342,+0.940, 0)   shove B  (-0.940,+0.342, 0)
  180 deg   rank 2      not in the cone at all         not in the cone at all
  175 deg   rank 3      c = (1.00, 10.18, 10.60)       c = (1.00, 10.60, 10.18)
   90 deg   rank 3      c = (1.00,  0.34,  0.94)       c = (1.00,  0.94,  0.34)
```

`c` is `(floor, chock 1, chock 2)`. **175° reaches both shoves and would need ten body
weights on each chock to answer them; 180° cannot reach them at all; 90° holds both, with
its chocks at a third and at 94 % of one body weight.** The run asserts exactly this list
of six answers, so if the camera, the angles or `SHOVE` ever move the arrows off the claim
the script stops instead of drawing something else.

**24.88 % of the sphere is held by 90° and failed by both the others**, so the choice was
wide and it was made for the page: longest on the page, most separated, landing on
different faces, and pointing like a shove rather than a lift.

**The question mark is `query()`** — blue, because it belongs to the blue family and not to
the contacts, and standing on the arrow's own line past its tail at `QUERY_LN = 1.00`
against `arrow3`'s 0.70, so it reads as a query about that arrow and not as a mark on the
floor. The floor is grown to sit under the marks for the same reason it is grown under the
tails: they stand at the plate's own mid height and would otherwise be the one thing in the
picture floating over bare paper.

### why counting arrows was the wrong instrument here

**`solid()` can count arrows and this figure cannot.** There the three panels have
different generator COUNTS and `reachable`'s simplex sampling gives 1, 9, 21 — a count that
tracks the DIMENSION, which is that figure's entire claim. Here all three panels have three
generators. The old fan dodged that by sweeping the set and thinning it to a fixed angular
spacing, so the count tracked extent instead — but extent is not area either (a long thin
set collects arrows along its length), and a reader counting 9 against 16 was being invited
to compare two things that no number under the picture supported. **The percentage is the
number, and now there are two of them.**

**And the free/capped pair is the cube against the simplex, one more time.** With the force
free, two nearly-opposed pushes and the floor span very nearly a half space and answer
nearly half of every disturbance; capped, the middle of that span costs more than any chock
has, and 48.61 % becomes 1.58 %. The 90° pair spans half as much and loses none of it. **A
support arrangement that covers a great many directions and can barely push in any of them
is exactly what a dimension count cannot see** — which is why the top row of this figure is
now two arrows and a question mark rather than a set drawn out in full.


## Reading the picture

Three panels, left to right: no supports, one chock, two chocks. One scene, one camera,
one floor, one body in all three — nothing changes across the panels except the number
of chocks, so the panels can be read by counting.

- **Orange arrows are generators** — contact push directions. Same colour, same weight,
  same length everywhere, so counting them is the point. One in panel 1 (the floor,
  always free), two in panel 2, three in panel 3.
- **Blue arrows are survivable disturbances**, drawn arriving on the face they meet most
  squarely. **1, 9 and 21**. That 1 → line → sheet is the dimension count made visible.
- **The row of orange arrows on top of the plate, each starting from a dot, is what the
  contacts can PRODUCE** — length *and* line width in proportion to how hard that
  direction can be pushed. **1, 3 and 6**. Its own section is below.
- **The roundel** is the usual centre-of-mass mark. In `normals.png` it is there
  specifically so the reader can see that nothing points at it.
- **The ground** is a bounded quad in z = 0, drawn with its own edges visible.
- **The chocks** are blocks standing on the floor, inner wall flush against a vertical
  side of the plate, rising to twice the plate's thickness and sloping away behind.

Panel 3's arrows land on faces `(0,+1)`, `(1,−1)` and `(2,+1)` — the two visible sides
and the top.


## The row on the body — built, then switched OFF

**`ROW = False`. The row is not drawn, on the human's instruction: "dimension.png在物体上的
橙色箭头是啥意思？？把他们去掉".** That is the whole verdict on it — a third family of
arrows on a panel already carrying 21 blue ones and up to 3 orange generators does not
read, and a reader who cannot tell what an arrow means is not helped by how carefully it
was placed. Turning `ROW` back on reproduces the figure exactly as it was; turning it on
*without* also thinning the blue fan will reproduce the complaint. The same quantity is
drawn legibly on the ball in `dimension_pairs.png`, which is where it belongs.

Switching it off restored `dimension.png` and the four `_iso_` variants to the md5s they
had before it went in — a clean revert, checked rather than assumed. Everything below is
kept because the placement work is sound and the numbers are right.

### what it was

**2026-08-14, on the human's instruction: 回到dimension png那个图，你需要在物体上也画出可能
的力，一排，和他们的大小.** A row of arrows across the plate, each pointing along its own
direction, sized by how hard the contacts can push that way.

**It is the other half of the picture, and the blue arrows cannot supply it.** Blue says
what the supports SURVIVE — a set of disturbances. This says what they can PUSH WITH — a
set of forces. Same contacts, two different sets, and until now the figure only drew one
of them. The orange generator arrows name the directions the supports push in but say
nothing about what any combination of them is worth.

**The quantity is the pairs figure's, exactly.** Each support is capped at `F`; a
direction in the cone is `v = Σ cᵢuᵢ` with every `cᵢ ≥ 0`; scaling the combination scales
every coefficient together, so the cap binds on the largest and

    m(v) = F / maxᵢ cᵢ

which is `strongest()`, the same function `dimension_pairs.png`'s fan is drawn from. The
encoding is that figure's too — **length and line width both in proportion to `m`**, no
curve on either — so the two pictures speak one language.

**Here the three generators stand mutually square** (printed each run: pairwise dots
`+0.000, +0.000, +0.000`), so `cᵢ = v·uᵢ` and every answer is exact. Printed by the run,
and asserted on every drawn arrow against `spanned()` (each is in the cone) and against
`F / max(v·uᵢ)` (each magnitude is what the closed form says):

```
one force      1 arrow    1.0000 F
two forces     3 arrows   1.0000  1.4142  1.0000 F
three forces   6 arrows   1.4142  1.0000  1.4142  1.0000  1.4142  1.0000 F
```

**The shape of that is the point, and it is the opposite of where a reader looks for a
weak spot: the PUSH DIRECTIONS ARE THE WEAKEST places in the cone.** A push can be
supplied by its own support working alone and reaches exactly `F` however the supports are
arranged; everything between two of them has both to draw on. `√2 F = 1.4142` between any
two, `√3 F = 1.7321` down the middle of the octant. The row alternates long, short, long,
short round the rim, and every short one is drawn **parallel to one of the generator
arrows already in the panel**, which is what lets the reader match them up.

**What is drawn is the cone's RIM — its generators and the arcs joining them — and not
its inside. That is forced by the camera and it is the one real compromise here.** The
scene looks down elev 36°, azim −45°; the cone's axis is elev 35.26°, azim −45°. They are
**0.7° apart**, so the `√3 F` direction points essentially at the reader and keeps
**0.013** of its length on the page. It cannot be drawn at this camera in any style. Worse
than that, an honest projection *inverts the whole message*: measured over the octant,

```
                       m        keeps       drawn on the page
  a push itself     1.0000 F    0.809            0.809
  an edge middle    1.4142 F    0.572            0.809   <- dead level with F
  the axis          1.7321 F    0.013            0.022   <- a dot
```

Every direction in this cone is within 54.7° of the camera and the ones that can be pushed
hardest are the ones nearest it, so foreshortening runs exactly against magnitude. The rim
is the part that stands clear: **the worst foreshortening along it is 0.572**, above the
`0.55` floor `onpage()` clamps at, and each arrow is given the 3-D length whose *page*
length is the magnitude — the same correction, and the same reasoning, as the ball's fan.
Drawn page lengths are **0.280 at `F` and 0.396 at `√2 F`**. The `√3 F` peak is printed by
the run and painted whole on `dimension_ball.png`, which is where interior directions
belong.

**The pitch is 45°**, so a 90° edge gives its two ends and its middle and nothing else:
`F, √2 F, F`. Coarser than that loses the middle and the swell with it; finer is
unaffordable — 22.5° would put **12** arrows on panel 3 rather than 6, on top of 21 blue
ones.

**The arrangement is a row, and three things about it are not free.** The tails run along
the **widest chord of the top face measured on the page**, so the arrows can be compared
side by side; that chord is a world direction that turns with the camera, so it is taken
from `VIEW` and the four view variants each get their own. The face is the **horizontal
plane itself**, which is what rescues the horizontal forces: an arrow lying in the top
face reads as horizontal, where the same arrow drawn anywhere else reads as pointing down
the page. And the **spacing is fixed, not the span** — the same rule `anchors()` follows
for the blue arrows — so the row GROWS, 1 → 3 → 6, and what the reader sees change is the
set and not the layout.

Within the row the arrows are sorted by **the angle they are DRAWN at**, descending, which
makes the row a sweep: each arrow leans a little further round than the one to its left,
like a clock hand strobed across the plate. Panel 3's rim is a closed loop and covers a
whole turn of the page, and no order can lay a whole turn along a line without one end
leaning back over its neighbours; this sort puts the leaning one at the right-hand end and
makes it a push direction, which is the shortest arrow there is. Sorted this way panel 2's
three arrows are the second, third and fourth of panel 3's six, in that order.

**A dot under every tail**, and it earns its place: six arrows pointing six ways on the
page do not read as a row until the reader knows where each one starts. Nothing else in
the picture stands in an even line.

### Keeping the two families apart

The warning was real — panel 3 already carried 21 blue arrows, two chocks and a roundel.
Four things separate the row from them, and the first is the important one.

- **Colour.** Orange is what the contacts supply, blue what they survive. The row is made
  of the orange arrows' own combinations, so it takes their hue; a third colour would say
  it was a third kind of thing, which it is not.
- **Sense.** A blue arrow ARRIVES, tip on the face. A row arrow LEAVES, tail on the face,
  which is where a force's arrow starts.
- **Place.** The blues spread over three faces and come in from outside the body; the row
  is one line across the top of it, with dots.
- **Weight.** `ROW_LW = 1.9` at `F` against the generators' `3.0`, so the two pushes stay
  the heaviest orange in the panel.

**Nothing was taken away from the blue arrows to make room. `1, 9, 21` is unchanged**, and
the m = 6 sampling that produces the 21 is unchanged with it.

**The layout was derived rather than eyeballed, and `crowding()` is the measure.** It
reports two numbers, because two things can go wrong and they are not equally bad. NEAR is
the least the row comes to anything that would confuse if it touched — another row arrow,
a generator, an arrowhead against a blue shaft either way round, or the roundel; a blue
shaft *crossed* at a steep angle is not counted, being two colours crossing. FAMILY is the
least it comes to another ORANGE arrow, and it is the one that decides how wide the row
may be. `ROW_STEP`, `ROW_LN`, `ROW_AT` and `ROW_OFF` were picked by sweeping the whole
face and maximising NEAR subject to FAMILY ≥ 0.20. What that costs and buys, in world
units on the page (the plate is 2.06 wide):

```
                                   NEAR                            FAMILY
row centred on the face           0.028   panel 1 on the ROUNDEL    0.210
nudged along the chord only       0.021   panel 2 head to head       0.220
                                          with a blue arrow
as shipped, 0.02 and 0.08         0.058   panel 3 near the roundel  0.252
wider: step 0.38, span past
  the plate's corners             0.060                             0.130   <- rejected
```

The wide layouts score *better* on NEAR and are still wrong, and the reason is worth
keeping. **At this camera the three pushes project 120° apart and therefore sum to nothing
on the page, so every `√2 F` direction is drawn exactly ANTIPARALLEL to the one push it
does not contain.** Spread the row past the plate's corners and its outermost arrow comes
down along a chock's own generator — same hue, same line, 0.130 apart. Kept inside them it
is 0.252 away. Both numbers are printed by every run, for every one of the four azimuths
(variants **b** and **d** are the tightest, at 0.055 and 0.226).


## The numbers, and the two checks

Lying down, all three answers are **exact**, which is new. Every push is horizontal, so
`d` survives exactly when it leans away from each push: the floor alone leaves one
direction, one chock leaves a **half great circle, 180.0°**, and two leave the **quarter
sphere `{d·u₁ ≤ 0, d·u₂ ≤ 0}`, 25.000 %**. The sweep has to reproduce those, less the
band `measure` cuts out around straight up.

Printed by the run (all reproduced):

```
floor only          1 direction survives, (-0.001,+0.002,-1.000) -- straight down; 3 of 40000
floor + 1 support   an arc 176.1 deg long, and no area at all (344 of 40000);
                    twice the push's 90.0 deg lean predicts 180.0,
                    less the 4.06 deg cap = 175.9
floor + 2 supports  a patch, 25.8 % of the sphere (10336 of 40000)

nine times the samples          n = 40000            n = 360000
  point                       3 hits   0.007 %        3 hits   0.001 %
  arc                       344 hits   0.860 %     1057 hits   0.294 %
  patch                   10336 hits  25.840 %    91043 hits  25.290 %
```

**Check 1 — grid refinement (`refine`).** `measure()` counts a direction as surviving
when the true set passes within one grid step `h = sqrt(4π/n)` of it. That tolerance is
what makes the three sets comparable at all: no finite grid lands on a set of measure
zero. It also forces the counts to behave: a 0-dimensional set keeps a fixed handful of
samples however fine the grid, a 1-dimensional set collects hits like `sqrt(density)`,
and only a 2-dimensional set holds a fixed share. Nine times the samples must therefore
give **1×, 3× and 9×** the hits. Measured: 3 → 3 (1.00×), 344 → 1057 (3.07×),
10336 → 91043 (8.81×). Those are the three dimensions, read off directly.

A real regression looks like a ratio moving to a neighbouring integer — an arc scaling
like 9, or a patch like 3. Noise it cannot be. The patch's 8.81 rather than 9.00 is
expected and not a defect: the tolerance band fattens the boundary, so the percentage
converges **from above**, 25.840 % → 25.290 %, **towards the exact 25.000 %**. Being able
to name the limit is the gain from laying the body down; the old pose could only say
"at or just below 18.0 %". Quote 25.840 % as the count at n = 40000, never as the area.

**Check 2 — the arc's length, geometrically and independently.** The wedge's two edges
are the floor's own ray, which gives straight down, and the chock's ray, which gives
straight down *reflected in the push direction*. A reflection turns a vector through
twice the angle to the mirror, so the arc is exactly **twice the push's lean from
vertical** — and the push is horizontal, so that is 180.0° exactly.

**The sign of the correction flipped when the body was laid down, and this is the thing
to know.** The far end of the arc is now *straight up*, which is precisely the direction
`measure` throws away, so the sweep must come back **short by the 4h = 4.06° cap** rather
than long by a step at each end. Predicted 175.9, measured **176.1** — 0.2° apart. If it
ever comes back long, or short by much more than the cap, the generator set or the
tolerance has changed. The same reflection identity still holds unchanged in the plane
figure: the support there pushes at `(-0.882,+0.471)`, leaning 61.93° from vertical, and
the measured run is exactly `(-90.00°, +33.75°)` = 123.75°, against 2 × 61.93 = 123.86°,
with a circle step of 0.25°.


## The files

| file | md5 | note |
|---|---|---|
| `dimension.png` | `3fed2dbdd2b4c1e86e3509d072cb8590` | the scene, 3289 × 596 |
| `dimension_ball.png` | `6d24a472d66f2991b0c09cfde161b5fd` | the cone the arrows span, 2377 × 823 |
| `dimension_pairs.png` | `e00662c8754a8705cf33d8a3a88475a0` | 180, 175 and 90 deg apart, 2612 × 1547 |
| `dimension_flat.png` | `42b00980c594ab1c83ed3ead6f6a9491` | **frozen — must not change** |
| `normals.png` | `9a82bdb6f7d5d66197e14492a6ab2248` | **frozen — must not change** |

**The main figure is variant `a`.** `BEST = -45.0`, which is `AZIMS[0]`, so
**The four `dimension_iso_*` views are no longer written.** `VARIANTS = False`. They were
scaffolding for choosing `BEST` by looking, that choice is settled and argued below, and
nothing read them afterwards; set `VARIANTS = True` to get them back, at which point
`dimension.png` and `dimension_iso_a.png` are the same bytes. That was the check that
tells you which view is in play: if the two md5s ever diverge, `BEST` was changed.

**The two frozen files are the two plane figures**, and they belong to different
arguments. `dimension_flat.png` is the same statics read at a glance in 2-D and lives
beside the solid figure; `normals.png` answers a separate question entirely (whether
frictionless contact forces on a tilted face are parallel or aimed at the centre of
mass — neither: they follow the *local* normal). They have been held byte-identical
across every edit to the 3-D half of the script, including this one, which is why
`BODY3`, `SUPPORT3`, `GROUND3`, `SUPPORT_INK` and `GROUND_INK` are forked constants
rather than edits to `BODY`, `SUPPORT` and `GROUND`. **Do not touch `BODY`, `SUPPORT`,
`GROUND`, `HALF`, `block`, `draw`, `along`, `spans`, `com`, `generator`, `arrow`,
`rounded` or `normals` while working on the solid figure.** Run the script and diff the
md5s of those two files before you call any change done.


## Decisions that look free and are not

**The chocks press the faces turned AWAY from the reader, and that is `pressed()`.** Both
faces of an axis do identical work — same plate, mirrored — so the choice is the
picture's. It settles two things at once: a chock on a face the reader can see must come
between reader and body, and on the far face it never can, which is why `panel()` can now
hand every chock the same zorder. While the body balanced on a corner this had to be
decided per support and per azimuth, and from two of the four a support came forward and
covered the resting contact.

It has one cost and it is unavoidable. The survivable set is `{d : d·u ≤ 0 for every push
u}`; pressing the far faces puts both pushes on the camera's side, so the direction
pointing straight *away* from the reader belongs to the set, and an arrow along it has no
length on the page. Pressing the near faces would put the direction straight *towards*
the reader in the set at exactly the same cost, with occlusion on top, so there is nothing
to be had by swapping. What is left is to keep the drawn samples off that axis — see the
sampling density below.

**Panel 3 is sampled at m = 6, giving 21 arrows, and it is not a taste.** Of the
densities that give a workable count, the shortest arrow on the page keeps

```
m = 3    6 arrows   0.410 of 0.70   59 %
m = 4   10 arrows   0.199           28 %
m = 5   15 arrows   0.043            6 %   <- one arrow becomes an arrowhead
m = 6   21 arrows   0.182           26 %
m = 7   28 arrows   0.192           27 %
```

`m = 5` puts a sample almost exactly on the view axis. 6 is the first density above it
that does not. Panel 2 is at m = 9, where the shortest arrow keeps 0.410, and the count
comes out at 9 as it always has.

**`reachable` drops `up` itself, and now it has to.** It is the same artefact `measure`
removes: a disturbance of exactly one body weight pulling straight up leaves `T = 0`, so
the contacts do nothing, it survives on an equality, and it says nothing about what a
support bought. It never arose while the plate stood on a corner, because no push there
was horizontal. Lying down every push is, so `g` horizontal gives `g·up = 0` and hands
back `up` exactly — a whole edge of the weight simplex collapsing onto one useless
direction, five of the twenty-one in panel 3 among them.

**The elevation is 36° and answers to nothing but a trade.** Both of the old ceilings —
the resting corner must stay visible, and it must stay the lowest thing on the page —
existed to protect a corner that no longer exists. The contact is a whole face now, and
every part of it that matters is on the silhouette at any elevation. What is left: too
low and the broad face closes up, and it is the face the argument is read off; too high
and the two thin faces close up instead, until the plate reads as a rectangle painted on
the ground and the chocks lose the height that tells them from their own shadows. 36° is
very nearly the balance point — the broad face keeps `sin(elev)` = **58.8 %** of its true
area on the page and each thin face `cos(elev)·cos45` = **57.2 %** of its own, so no face
is favoured.

**The azimuth is −45° of four candidates, and the reason changed.** It used to be
occlusion; with every chock behind the plate at every azimuth, occlusion no longer
separates them. What separates them is the light, which does not turn with the camera:

```
azim  -45   faces 0.800  0.975  1.160   narrowest gap 0.175
azim  +45         0.800  0.800  1.160                 0.000   <- an edge disappears
azim +135         0.800  0.815  1.160                 0.015
azim +225         0.815  0.975  1.160                 0.159
```

−45 wins outright and +45 is unusable. All four are still written out so the choice can
be made by looking; set `VARIANTS = True` and open the four together before changing
`BEST`.

**The spin is zero, and it is spent on the 3-D read rather than on the light.** The
plate's four sides stand vertical, so a spin that aims one of them at the camera hands
the reader a rectangle head on: the body flattens into a shape on the page and the floor
under it stops receding. This was tried — optimising the spin for shading alone drove it
to 45.5°, and the result reads as a flat rectangle with a grey band under it. Square to
the world axes, with the camera at an isometric corner, both visible sides come in at 45°
and the body projects to a hexagon that can only be a solid. The light is what pays for
the shading instead; see `LAZIM`.

**The light no longer has to be steep, and that is what freed it.** The old rule was that
the floor is sized to hold whatever is thrown onto it and every unit it grows comes off
the size of the body on the page, so the light had to stand high to keep the shadows
short. A plate lying down throws a shadow barely longer than itself, and the floor is now
sized by the **arrow tails** instead. Folding the shadow in costs **1.01 %** of the widest
side and 2.07 % of the area, and that number does not move at all as the light is lowered
from 64° to 45°. So `LELEV` could be spent on the shading: setting the two visible
verticals one step apart and the broad face the same step above them solves out at
`tan(LELEV) = 2`, or 63.4°, with the light square to one of them. 64° and −95° sit a
whisker off that, and the 5° of slack past square is deliberate — the far face is then
clipped to the ambient by a margin instead of sitting exactly on the knife edge.

**`SUN = 0.80, 0.40, 0.25` — but the bounce term has gone inert.** It was put in because
the only faces a key above a floor cannot reach are the ones looking down, and blue
arrows used to land on a downward face of a body balanced on its corner. Lying down, the
body turns no face downwards that the reader can see, so `shaded`'s bounce multiplies zero
on everything drawn. It is kept because it costs nothing and the term is right; it is no
longer load-bearing, and nobody should re-derive the figure's lighting from it. The three
visible faces come back at **0.800 / 0.975 / 1.160**, gaps of **0.175 and 0.185** — the
widest and the most even this figure has had (0.157 and 0.177 balanced on a corner, 0.095
and 0.195 thinned and still standing), with the darkest sitting *at* the ambient because
it stands edge on to the light and takes no key at all.

**The floor's generator is drawn at `foot()`, not at the origin.** Balanced on a corner
the body touched the floor at the origin and the arrow was drawn there. Lying down it
touches over its whole underside and the origin is the middle of that, under the plate,
where nothing can be seen. Position is no part of the model, so this is legibility only:
the corner of the footprint nearest the reader, on the silhouette, clear of the body in
every panel.

**`trim()` crops the saved raster, and the framing box has to stay a cube.** This was
tried the other way round first and it does not work. `set_box_aspect` rescales whatever
box it is given to a fixed diagonal and then stretches the result to fill the axes
rectangle, so a box that is not a cube comes out both shrunk *and* squashed — the plate
was drawn at 59 % of the panel and distorted with it, and the floor was clipped at the
panel edges. A cube framed to the width leaves two fifths of the page height empty above
and below every panel alike, and cropping the raster afterwards is the one fix that
cannot distort anything, because it moves no pixel that it keeps. All three panels share
a vertical extent, so one cut serves for all of them. `bbox_inches="tight"` is not a
substitute: it works from artist extents, and for a 3-D axes that is the whole axes.

**The ground is a bounded quad, not an infinite sheet.** `tile()`'s docstring records
what the sheet did: run it off all four sides of the picture and the only boundary left
in view is the cut across it, and an orthographic cut across a ground plane is dead
straight and, at any of these four azimuths, dead horizontal. That is a band under a
line — the plane figure's floor — and no elevation makes it recede, because a plane
recedes on the page through the convergence of its own edges and there are none. Bounded
and squared to the *world* axes rather than to the page, all four edges are in view and
none is horizontal: the quad projects to a rhombus that opens with `sin(elev)`.

**Cast shadows are opaque, not alpha-blended.** `veil(t)` mixes `GROUND3` towards `DUSK`
and returns a flat opaque colour. With alpha, two shadows crossing would darken each
other and draw a seam along an edge that is not there. The floor is one flat colour, so
the blend is computed once and an overlap simply looks like more of the same shadow —
which is what two shadows crossing actually do. The softness is 14 nested copies per
shadow shrinking **inwards** (`SOFT, CORE, DEEP = 14, 0.45, 0.19`) — a penumbra with no
blur. A second, tighter ramp (`BED = (0.18, 0.035)`, `BEDDED = 0.38`) darkens the floor
immediately around each contact, because the one thing a cast shadow cannot show is
contact — its own edge runs through it. `bed()` survived the change of pose unaltered: it
sweeps a disc over whatever points lie on the floor, which is now a rounded rectangle
around the plate's footprint and another around each chock's, where it used to be a disc
under a single corner.

**Matplotlib's automatic depth sorting is off.** `computed_zorder=False` on the 3-D
axes, and `Arrow3D.do_3d_projection` returns `0.0` so the axes is told not to sort the
arrows either. Left to itself mplot3d sorts every collection and patch by one averaged
depth, so an arrow landing on the body counts as nearer or further than the whole of it,
and the answer flips as the arrow moves. Turned off, the explicit zorders decide, exactly
as in the plane figure. **Everything drawn in `panel()` depends on this**, in this order:
ground 0, shadow ramps 1.00–1.27, chock 2 (its generator 3), body faces 4, silhouette
4.4, roundel 8 and 9, blue arrows 11, floor generator 12, the producible-force row's dots
12.9 and its arrows 13. An artist added without an explicit zorder lands wherever it
happens to be in the list, so if you add one, give it a number. **The row is last on
purpose**: it is the only thing in the panel whose LENGTH carries a number, and a number
read off a part-hidden arrow is read wrong.

**Panel 1 has exactly one blue arrow, and that is load-bearing.** In the plane, `spans()`
throws away runs thinner than `floor_deg = 2.0°`, so the floor-only panel gets no run at
all and `draw()` falls into the `not runs` branch; in space, `measure()` drops the
directions that pass only because the force they ask for is next to nothing. What is left
is one direction — exactly `(0, −1)` in the plane, `(−0.001, +0.002, −1.000)` on the
sphere. **One direction is one arrow.** Copies of it along the face would inflate a
0-dimensional set into a row of capabilities and destroy the argument.

**Panel 1's producible row is one arrow for the same reason, and it has to stay one.**
The floor alone spans a single ray, so there is exactly one force the contacts can supply,
straight up, at exactly `F` — no range, nothing to sample, nothing to sweep. `producible()`
returns that one direction for a single generator and takes no pitch to get there. An
arrangement that gave panel 1 a fan of any kind would say the first contact bought a range
of capability, which is precisely the reading this figure exists to refute.

**Blue arrows spread along a face; they never converge on a vertex.** Position is no part
of the model — a force enters as a direction and nothing else — so where an arrow lands
is legibility only. `landing()` sends each push to the visible face it meets most
squarely, which is also the only way its shaft stays outside a convex body, and ranks it
within that face by the direction it arrived from. `anchors()` then stretches each face's
range across the face, with the range taken over **all three panels at once**, so a push
surviving in more than one panel keeps its place and the growth from arrow to line to
sheet is the set growing rather than the spacing changing. Nothing here is placed by hand.

**`casters()` takes `props=None`, not `props=PROPPED`.** A default argument binds once at
import. Written the obvious way the function goes on reading whatever `PROPPED` was when
the module loaded, and hands `chock` an axis it cannot build against — which is exactly
how it failed while this pose was being prototyped.


## Known open issues

Recorded by earlier passes, or found here, and deliberately not fixed.

- **Half the arrow tails are below the floor plane.** The plate lies a tenth of its own
  width off the ground, so every survivable direction with any lift in it puts its tail
  under z = 0: the lowest reaches **z = −0.6253** and **15 of 31** tails are below the
  plane (17 below `tile`'s `touching = 0.20`), against 3 of 25 when the body stood on a
  corner. They are drawn over the ground rather than through it — zorder 11 against the
  floor's 0 — so nothing looks wrong, but `tile()` is carrying the floor out to sit under
  all of them, and that floor is 3.010 × 2.890 where the plate is 1.52 × 1.40. Shortening
  `arrow3`'s `ln = 0.70` is the lever if the floor is ever the thing to shrink.
- **One arrow in panel 3 points nearly straight into the page.** It is the direction
  running away from the reader, it genuinely belongs to the set, and it cannot be removed
  by any choice of azimuth (see `pressed`). At m = 6 it keeps 26 % of its length and reads
  as a short arrow rather than as an arrowhead.
- **The two-chock panel is crowded, and the row made it more so.** Twenty-one blue
  arrows, six orange ones, two chocks and a roundel now share one silhouette. Nothing is
  mis-drawn and the row was placed by maximising the measured clearance rather than by
  eye, but 0.058 on a plate 2.06 wide is not much, and it is what caps the row's arrows
  at `ROW_LN = 0.28` and its pitch at 45°. **The next thing added to this panel will not
  fit.** If one has to be, the lever is the blue count — but see the note above about
  what 1 → 9 → 21 is carrying.
- **A row arrow at `F` is drawn shorter than a generator arrow, and both stand for `F`.**
  0.280 against 0.372 on the page. They are different kinds of arrow — a generator is a
  direction mark of one fixed length, drawn so the reader can count them; a row arrow is a
  magnitude — but panel 1 puts the two of them in one picture pointing the same way, and
  nothing in the drawing says the difference. Matching them would need `ROW_LN = 0.372`,
  which does not fit between the two chock generators (see `crowding`), and shortening
  `generator3` would spend a well-settled part of the figure on it. What anchors the scale
  instead is that panel 1's single arrow **is** `F`, and the same arrow stands in all three
  panels because `up` belongs to every cone.
- **The row draws the rim of the cone, never its inside, so `√3 F` is never on the page.**
  It is printed and it is on `dimension_ball.png`; here it is 0.7° off the camera. A reader
  who counts the row's arrows is counting a sample of a rim and not a set — six is
  `ROW_PITCH`, not a property of the octant. The dimension count is the blue arrows' job
  and the row must never be read for it.
- **Every horizontal producible force is drawn pointing DOWN the page**, because a
  horizontal direction at elev 36° projects below the horizon. Standing the row on the
  broad face is what makes that legible — the face *is* the horizontal plane, so those
  arrows lie in it — but it is the same collapse the ball figure exists to answer, and at
  this camera it has a sharp form worth knowing: the three pushes project 120° apart and
  sum to nothing, so each `√2 F` arrow is drawn exactly opposite the push it does not
  contain. Two arrows drawn back to back on the page are **not** opposed forces.
- **A drawn fan arrow is a few per cent shorter than the length it was given.**
  Matplotlib pulls an arrowhead back from the endpoint by roughly 1.4 × the line width, so
  a nominal 113.1 px arrow at `F` draws 109 and a nominal 159.9 at `√2 F` draws 153 —
  the **same 96 %** in both, because the width scales with the magnitude too, so the fan
  is uniformly scaled rather than distorted. Where `FAN_THIN` pins the width the bite is
  fixed instead: the 0.087 F stub is nominally 9.9 px and draws 8. That understates the
  weakest arrows and never the strongest, which is the safe direction for this figure, and
  8 px is still a visible mark. Measured with a standalone `FancyArrowPatch` at this dpi,
  not read off the picture.
- **The bottom row of `dimension_pairs.png` is heavier on the right than on the left.**
  The 90° fan hangs **1.255 ball radii** below the middle, where the 175° fan does not
  reach past the ball's own silhouette at all and the 180° column has nothing below its
  equator, so the row does not balance. It is the arrows saying what they are there to
  say. Nothing is clipped, and that was checked rather than assumed: in the untrimmed
  2772 × 1638 raster the bottom row's axes floor is **1.599 radii** below the ball centre
  against the 1.255 the lowest arrow uses, **80 px of margin** — measured off that raster
  with the ball at radius 231.5 px. (An earlier pass wrote 1.265 / 0.712 / 1.607 and 87 px,
  at the marginally larger ball this row had before the caption grew to five lines.)
- **The top row is now the LIGHTER of the two, and that is new.** With the blue fan gone
  it carries two arrows where the row under it carries eleven, and the scene sits in the
  middle of a square axes with clear paper above and below it. Nothing is wrong with it and
  the alternative — putting a set back on it — is exactly what was removed. The lever, if
  it ever has to be pulled, is `subplots_adjust`'s `hspace`, not the arrows.
- **The patch percentage is grid-dependent.** 25.840 % at n = 40000, 25.290 % at 360000,
  converging from above to the exact 25 % because the `h`-wide tolerance fattens the
  boundary. Any comparison must be at matched `n`.
- **The fillet in `normals.png` is drawn far fatter than the real thing.** Deliberate — at
  true scale the fan would be one pixel. The comment cites "about 3 mm on a 190 mm face"
  for A1-f; `objects/A1-f/meta.json` gives extents 0.10 × **0.18** × 0.10 m, so the face
  is 180 mm, not 190. The 3 mm radius itself was not verified here.
- **Output paths are absolute and hard-coded** to `/Users/yuanboli/.../figures/` in
  `main()`. The script cannot be run from a clone elsewhere without editing.
- **`solid()` binds a local `flat` that shadows the module-level `flat()` function.** No
  bug today — `shadow()` and `ground()` resolve `flat` in module globals — but it is a
  trap for anyone moving code between the two.
- **One comment/code mismatch left**, not affecting the drawing: `main()`'s note that the
  admissible arc is "symmetric about" the support's outward normal `push` is symmetric
  about **−push**, the direction contacts there actually push — the arc's midpoint is
  exactly `(+0.882, −0.472)`.


## Open questions for the human

1. **Which of the four view variants to adopt.** `dimension.png` comes from `BEST = -45.0`,
   variant `a`, and the shading table above says it is the only azimuth where all three
   visible faces separate. That is a strong argument but it is an argument, and the four
   are written out side by side precisely so the choice can be made by looking.
2. **~~Whether "balanced on a single corner" stays the premise.~~ Settled 2026-08-13: it
   does not.** The human asked to see the body lying down; the geometry above says that is
   the only pose in which a flat plate can be seen whole. Everything in the camera half of
   this script is now downstream of *lying down* instead, so if the corner ever comes
   back, the elevation, the azimuth, the spin and the light all re-open with it.
3. **Whether the chocks are the right size.** `SEAT = (0.46, 0.46, 0.56)` — half the side
   covered, 0.46 out, a crest at twice the plate's thickness. The crest is the number
   doing the work: below about 1.5 thicknesses a chock standing behind the plate is hidden
   by it. Nothing sets the other two but the look of the thing.
