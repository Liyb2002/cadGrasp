# more_pushes.png — what it argues, and what not to break

Handoff for `slides/tools/more_pushes.py` and the one PNG it writes into `slides/tools/figures/`.
Everything below was re-derived on 2026-08-14 by running the script; the numbers are
from that run or from the closed forms it checks itself against, not from memory.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u slides/tools/more_pushes.py     # ~6 s, rewrites one PNG, prints every number
```

There is no randomness anywhere in it and a clean re-run reproduces the file byte for
byte — checked twice, `b9bf3efd7a5fb11e227581be52c3b56e` both times, with stdout
identical apart from the elapsed-time line. It writes nothing else: after every run
`three_pushes.png` is still `85f3b87ef02a9a87fb9ee54ae8e61d6c`, `dimension_flat.png`
`42b00980c594ab1c83ed3ead6f6a9491` and `normals.png`
`9a82bdb6f7d5d66197e14492a6ab2248`.


## Why this is a new file and not a fourth panel of `three_pushes.py`

The brief allowed either. It is separate because the two figures do not share their
spine, only their palette.

`three_pushes.py` is built on the fact that **three independent generators decompose a
direction uniquely**. `decompose` is one `np.linalg.solve`, `capped` is a formula, and
`covers`, `weakest` and `strongest` are three closed forms in one variable — the
pairwise angle `t` — because the whole family it draws is one regular tripod opening
and closing. None of that survives a fourth ray. The decomposition becomes a linear
program, the family becomes four unrelated polyhedra, `t` stops being a parameter, and
every closed form has to be rebuilt out of the zonotope instead. Folding that into
`three_pushes.py` would have doubled it and left its own note arguing two things at
once, and the risk it carries — see the next section — is exactly the kind that gets
lost in a file that already works.

What IS shared is copied verbatim so a reader can put the two figures side by side:
the palette, both ramps and the break at `F`, the geodesic mesh, `Arrow3D`, `split`,
`shaded`, `onpage`, `onpaper`, `trim`, and the camera `ELEV, AZIM = 20, -12`. Panel
(a) of this figure IS panel (a) of `three_pushes.png`, same arrangement, same pose,
same camera, deliberately, as the baseline the rest of the row departs from.


## The one thing that breaks with more than three rays, and it breaks silently

With three independent generators `v = Σcᵢuᵢ` has one solution and `m(v) = F/maxᵢcᵢ`
is a formula. With four or more it has a whole affine family of solutions and the
contacts get to choose, so

    m(v)  =  F / min { maxᵢ cᵢ  :  c ≥ 0,  Σ cᵢuᵢ = v }

which is a **linear program**, not a solve. `reach_lp` states it the way round that
makes infeasibility impossible: maximise `r` subject to `r·v = Σcᵢuᵢ` and `0 ≤ cᵢ ≤ F`,
with `r ≥ 0`. `r = 0, c = 0` is always feasible, so an unreachable direction comes back
as `r = 0` rather than as an exception — which is the third state the figure needs.

`wrong_ways` **runs** the three obvious substitutions rather than warning about them.
All three are one line, all three are already used elsewhere in this repo, and all
three are wrong without leaving an exception behind.

**1. `np.linalg.solve`, which is `three_pushes.py`'s own `decompose`, on a set that is
flat and does not say so.** Its own tripod family reaches that at 120°:

```
       t      cond(U^T)   solve max|c|   raised?   the LP says
   110.0000°  2.061e+00     1.1768e+00      no      0.8498 F
   119.0000°  6.991e+00     1.9223e+00      no      0.5202 F
   119.9900°  7.044e+01     1.0912e+01      no      0.0916 F
   119.9999°  7.044e+02     1.0057e+02      no      0.0099 F
   120.0000°  1.911e+16     2.8285e+15      no      0.0000 F
```

At 120° the three rays are coplanar, the matrix is singular to `1.9e16`, and **solve
returns** — coefficients of order `1e15` for a direction the three cannot reach at
all. Nothing downstream can tell. `facets` refuses rank-deficient input by raising
instead, and the LP answers 0.

**2 and 3, the two decompositions that exist once there are more than three rays:**

```
                         the cone is  lstsq calls it  nnls calls it  nnls m too small on  by up to
             (a) 3 rays      12.50 %         12.50 %        87.48 %               0.00 %      0.0 %
             (b) 4 rays     100.00 %          0.00 %       100.00 %               0.00 %      0.0 %
             (c) 5 rays      50.00 %          0.00 %       100.00 %               0.00 %      0.0 %
             (d) 6 rays     100.00 %          0.00 %       100.00 %               0.00 %      0.0 %
  four, unevenly spread      48.90 %          2.17 %        98.78 %              18.81 %     33.4 %
```

`lstsq`/`pinv` give the minimum-**norm** decomposition, which has a negative entry in
almost every direction once there are four rays, so a `capped` written on it paints
**nothing** — 0 % where the truth is 100 %. `nnls` is the dangerous one: it is
feasible, it is smooth, and if its residual is discarded it calls the octant **87 %**
of the ball. And note the last two columns on the four symmetric sets: **`nnls`'s
magnitudes happen to match the LP exactly there.** That is the worst possible luck,
because a check run only on the sets this figure draws would pass; on an unevenly
spread four it is short over a fifth of the cone by up to a third.

**The LP is the definition and it is not what paints the figure.** 20480 mesh cells
times four panels is 81920 solves. What paints is `reach`, a closed form, and the
licence for substituting it is re-earned on every run — `main` runs both on 1200
directions of each of the six sets and asserts they agree, which they do to `8e-10`.

The closed form is worth stating because it is what makes every other number here
exact as well. The set the contacts can supply within their caps is the **zonotope**

    Z = { Σ cᵢuᵢ : 0 ≤ cᵢ ≤ F }

and `m(v)` is exactly `Z`'s **radial function** — how far one may go along `v` and
still be inside. Every facet of a zonotope in three dimensions is spanned by
generators lying in its plane, so every facet normal is `±unit(uᵢ × uⱼ)` for some
pair; and its support function is `h(a) = F Σᵢ max(0, a·uᵢ)` by inspection, since to
go furthest along `a` you turn on exactly the generators with a positive component
along it. Hence

    m(v) = min over facet normals a with a·v > 0 of  h(a) / (a·v)

vectorised over a million directions at once. A facet with `h(a) = 0` passes through
the origin: that is a **wall of the cone**, and it is how "cannot push that way at
all" arrives as a value rather than as a special case.


## The finding, and it is new

`dimension.md` records that for THREE pushes the guarantee *every direction the cone
reaches takes at least `F`* holds exactly while the pairwise angles are ≤ 90°.
`three_pushes.md` and `three_pushes.py` both then generalise that sentence to any
number of supports. **That generalisation is false, and this figure is the
counterexample.**

```
                                    rays   widest pair   weakest   guarantee
       (a) 3 rays, mutually square     3        90.00°    1.0000       holds
       (b) 4 rays, the tetrahedron     4       109.47°    0.8165        LOST
    (c) 5 rays, the square pyramid     5       180.00°    1.0000       holds
 (d) 6 rays, the three axes both ways  6       180.00°    1.0000       holds
           5, triangular bipyramid     5       180.00°    0.8660        LOST
     4, the tetrahedron, cube corners  4       109.47°    0.8165        LOST
```

Read the middle two columns against each other. **109.47° loses the guarantee and
180° keeps it.** And the triangular bipyramid has the SAME widest pair as the square
pyramid — 180°, and the same narrowest, 90° — and loses it anyway. Whatever the
criterion is, it is not an angle between two rays, and no ordering of the pairwise
angles recovers the verdicts.

**The criterion that does work** is the old one with the term that was invisible at
n = 3 written out. For a pair `{uᵢ, uⱼ}` let `a = unit(uᵢ × uⱼ)` — the direction
neither can put anything along. The most the arrangement can put there is **what is
left over**:

    h(a) = F Σ_{k ∉ {i,j}} max(0, a·u_k)

and the guarantee survives that pair when `h(a) ≥ F`, or when the cap the pair's plane
cuts off misses the cone altogether — which is `three_pushes.py`'s "`dᵢ` walks into the
patch", unchanged. Printed by the run, for the binding pair of each set:

```
                                     blind pair  rays left    h(a)/F   h(a)/|P(a)|
       (a) 3 rays, mutually square     u_1, u_2          1    1.0000        1.0000
       (b) 4 rays, the tetrahedron     u_1, u_4          2    0.8165        0.8165
    (c) 5 rays, the square pyramid     u_1, u_2          3    1.0000        1.0000
 (d) 6 rays, the three axes both ways  u_1, u_3          4    1.0000        1.0000
           5, triangular bipyramid     u_1, u_3          3    0.8660        0.8660
     4, the tetrahedron, cube corners  u_1, u_2          2    0.8165        0.8165
```

**With three rays that sum has exactly ONE term**, `F cos β` with `β` the angle from
the third ray to `a`, so it is at most `F` and equals `F` only when the third ray IS
`a` — which is precisely the mutually-square arrangement. *"Every pair at most 90°"
was never a statement about pairs.* It was the n = 3 shadow of "whatever is left over
must be able to cover the pair's blind direction", and the shadow detaches the torque
the sum acquires a second term. The square pyramid's pair `{+z, +x}` is blind along
`±y` — **and `+y` and `−y` are themselves rays**, so `h = F` exactly. The
tetrahedron's pairs leave two rays whose positive parts along `a` sum to `0.8165 F`,
and that number IS its weakest direction, to every digit.

`h(a)/|P(a)|` in the last column is the quantity `weakest` actually minimises: `P(a)`
is `a`'s projection onto the cone, so `|P(a)| = max{a·v : v in the cone, |v| = 1}` and
the ratio is the bound the facet imposes on the cone. It equals `h(a)` in all six rows
here only because every binding facet's normal happens to lie inside its own cone; in
`three_pushes.py`'s 50° panel it does not, which is the whole reason that panel keeps
the guarantee with `h(a) = 0.705 F`.

**And the other half is the trade the extra rays buy.** Three at 90° reach an eighth
of the ball with everything at `F` or more. The square pyramid's five reach **half**,
still with everything at `F` or more — *four times the coverage for nothing*, and its
fifth ray is not exotic, it is the fourth chock. Six reach the whole ball and keep it.
The tetrahedron buys the whole ball too and pays the guarantee for it, and it is the
arrangement a reader would reach for first, because "spread them as evenly as possible"
is the obvious instinct and it is the wrong one.


## The four panels

**Rebuilt on 2026-08-14, and the first version's mistake is worth keeping.** It drew four
arrangements SPREAD as far apart as they go — the tetrahedron, the square pyramid, the
three axes both ways — and every one of them reached most or all of the ball. The reader's
response was the right one: *"4 pushes 怎么占了一个球？？不应该都是一块一块的吗？？"* Four
pushes covering everything is true, and it is what force closure means, but a figure whose
every panel is a full ball teaches nothing about four pushes. Whether a set closes on a
small patch turns on how CLUSTERED it is, not on how many rays it has.

So three of the four panels now cluster, and the fourth is kept as the contrast:

- **(a) three, leaning 45° round one axis.** The three rays are the corners of a spherical
  TRIANGLE and the cone is the piece they close on: `7.693 %` of the ball, everything in it
  from `F` at the corners to `2.1213 F` down the middle.
- **(b) four, leaning 45°.** One more ray, same lean. A QUADRILATERAL, `10.817 %` —
  barely wider than the triangle — but the middle has gone from `2.1213 F` to
  `2.8284 F`.
- **(c) five, leaning 45°.** A PENTAGON, `12.234 %`, middle `3.5355 F`. Adding rays to a
  cluster buys almost no reach and a great deal of strength.
- **(d) the same four as (b), spread right out** — the regular tetrahedron, pairwise
  109.47°. They now positively span space, so the cone is EVERYTHING, and `94.0 %` of it
  is under `F`: weakest `0.8165 F = √(2/3)`, strongest only `1.1547 F = 2/√3`, `5.988 %`
  at or above `F`. The ball is almost entirely red, which is the panel.

**Two closed forms make the row worth putting side by side**, and `assert` checks both
against the LP on every run:

- along any CORNER ray the reach is exactly `F`, whatever `k` and `theta` are. Only that
  ray can push there without something else needing to be cancelled — the same fact the
  three-ray figure shows, and it does not care how many rays there are;
- along the AXIS it is exactly `k cos(theta) F`, because every ray contributes its full
  `cos(theta)` and none of them fight. At 45°: `2.12 F`, `2.83 F`, `3.54 F` for three,
  four and five.

The (b) → (d) step is the one the reader is meant to make: **the same four rays, moved.**

## The numbers, and the checks they are made to pass

Every column but one is a closed form, with the sweep printed beside it.

```
                                     widest            reaches   weakest  strongest      at or above F
                                       pair     exact    sweep                        of the ball  of the cone
       (a) 3 rays, mutually square   90.00°   12.500%  12.500%    1.0000     1.7321      12.500%     100.000%
       (b) 4 rays, the tetrahedron  109.47°  100.000% 100.000%    0.8165     1.1547       5.988%       5.988%
    (c) 5 rays, the square pyramid  180.00°   50.000%  50.000%    1.0000     1.7321      50.000%     100.000%
 (d) 6 rays, the three axes both ways 180.00° 100.000% 100.000%    1.0000     1.7321     100.000%     100.000%
           5, triangular bipyramid  180.00°  100.000% 100.000%    0.8660     1.4142      59.808%      59.808%
     4, the tetrahedron, cube corners 109.47° 100.000% 100.000%    0.8165     1.1547       5.988%       5.988%
```

The bipyramid is not a panel; it is in the table because it is the sharpest evidence
the figure has and there was no room for a fifth ball. The cube-corner tetrahedron is
there to pin the handoff's orientation — see below.

**`covers` is Girard's excess.** The cone meets the sphere in a convex spherical
polygon whose sides lie on the cone's walls, so its area is the sum of the interior
angles minus `(k−2)π` for `k` walls; the interior angle where two walls meet is `π`
minus the angle between their inward normals. Three degenerate counts sit under that:
no wall is the whole sphere, one wall is a half-sphere, two walls are a lune of twice
their dihedral. The octant comes out `π/2` exactly and the square pyramid — whose only
wall is `z = 0` — exactly a half.

**`weakest` is a minimum over facets, not a search.** `m = min_a h(a)/(a·v)`, so the
floor over the cone is `min_a h(a) / max{a·v : v in the cone, |v| = 1}`, and that inner
maximum is `|P(a)|`, the length of `a`'s projection onto the cone — a non-negative
least squares, `dimension.py`'s `gap` seen from the other side.

**`strongest` is the longest subset sum.** Minimising `maxᵢcᵢ` over `{c ≥ 0, |Σcᵢuᵢ| = 1}`
is the same as maximising `|Σcᵢuᵢ|` over the box `0 ≤ cᵢ ≤ F`, and a convex function
over a box takes its maximum at a vertex. `2ⁿ` of them with `n ≤ 6`, and no cleverness
is worth the risk of missing one. It gives `√3` for the three box-like sets, `2/√3` for
the tetrahedron and `√2` for the bipyramid.

**The share at or above `F` is exact too, and it is the one that could have been
fudged.** `m(v) ≥ F` means `v` is inside `Z`, so the set is the sphere with one cap cut
off by each facet standing closer to the origin than the sphere does. Every arrangement
here has all such facets at the same depth, so the caps are equal and their pairwise
overlaps are two copies of one integral (`half_lens`). **Triple overlaps are checked
rather than assumed**: each overlapping pair's boundary circles are intersected and the
two crossing points tested against every other cap. For the tetrahedron three adjacent
caps meet in a single point — the caps are `35.264°` across and their normals stand
`60°` apart, so they overlap in pairs and the triple meets exactly at
`(1,1,1)/√3`-type directions — and inclusion–exclusion stops at pairs. It gives
`5.988 %`, and the million-direction sweep returns `5.992 %`. The bipyramid's caps are
`30°` across with normals `60°` apart, so they only touch, and the answer collapses to
`3√3/2 − 2 = 59.808 %`; the sweep returns `59.815 %`. **That agreement is what says the
combinatorics were right**, and it is the reason both are printed.

**Areas are measured on a UNIFORM sweep and never by counting a mesh.** The run
measures the trap rather than asserting it:

```
           exact     sweep   lat-lon cells  geodesic cells
   (a)    12.500    12.500          12.500          12.480
   (b)   100.000   100.000         100.000         100.000
   (c)    50.000    50.000          50.000          50.313
   (d)   100.000   100.000         100.000         100.000
```

**and the lat-lon column looks innocent here, which is exactly why the check belongs in
`three_pushes.py`'s file and not only in this one.** All four of these cones are cut by
the equator, by meridians, or by nothing at all, so the `sin θ` over-weighting cancels
across their own boundaries every time. One figure over, at 50°, the same mesh reports
**two thirds more patch than there is**. A method that is exact on all four cases you
happen to check and 66 % out on the fifth is worse than one that is uniformly rough.


## The rows this figure was handed, reproduced

The brief carried four measured rows. `anchors()` reproduces them digit for digit on a
4000-direction sweep, printed beside the closed forms:

```
                                 reaches       weakest            strongest        at or above F
                                          sweep     exact     sweep     exact       of the ball
   4, tetrahedron 109.47 deg    100.00 %  0.8165 F  0.8165 F  1.1433 F  1.1547 F        6.10 %
           5, square pyramid     50.00 %  1.0002 F  1.0000 F  1.7157 F  1.7321 F       50.00 %
     5, triangular bipyramid    100.00 %  0.8661 F  0.8660 F  1.4038 F  1.4142 F       59.72 %
 6, the three axes both ways    100.00 %  1.0001 F  1.0000 F  1.7157 F  1.7321 F      100.00 %
```

Two things fall out of that, and both are worth having.

**The orientation matters to the swept columns and to nothing else, and it is what
identifies the handoff's tetrahedron.** The Fibonacci set is fixed in space, so turning
an arrangement through it changes which samples land where the field is extreme. At the
**cube's corners** the sweep returns `1.1433 F` and `6.10 %`, which is the handoff row;
with a **vertex up** it returns `1.1504 F` and `6.05 %`. Every exact column is identical
to every digit either way. The figure draws the vertex-up one, because in the scene
these figures are about, one ray is the floor.

**Two of the handoff's entries are closed forms rather than sweeps.** The square
pyramid's `1.0000 F` and `1.7321 F` are exactly `F` and exactly `√3 F`; the same sweep
returns `1.0002` and `1.7157`, because the strongest direction is the single corner
`(1,1,1)/√3` of a box and no 4000-point sample lands within a degree of it. The
bipyramid's row is swept throughout and reproduces to all four decimals, which is what
identifies the sweep as the same construction.

**The last column of the handoff is a share OF THE BALL.** `three_pushes.md`'s
corresponding number is a share of the PATCH — it says so explicitly, and it is right
to, because "the arrangement has stopped delivering over the very set it reaches" is
the sentence that matters. Both are printed here, in two labelled columns, because for
the square pyramid they differ by a factor of two: `50.000 %` of the ball and
`100.000 %` of the cone are the same fact.


## The pose, and the one camera rule that had to be loosened

```
  panel    ray    polar    page x    page y   page len   off-axis    margin
      a    u_1     0.0°    +0.000    +0.940      0.940     70.00°    70.00°
      a    u_2    90.0°    +0.839    -0.186      0.859     59.22°    59.22°
      a    u_3    90.0°    -0.545    -0.287      0.616     37.99°    37.99°
      b    u_1     0.0°    +0.000    +0.940      0.940     70.00°    70.00°
      b    u_2   109.5°    +0.500    -0.587      0.771     50.41°    50.41°
      b    u_3   109.5°    +0.443    -0.029      0.444    153.67°    26.33°
      b    u_4   109.5°    -0.942    -0.324      0.997     94.77°    85.23°
      c    u_1     0.0°    +0.000    +0.940      0.940     70.00°    70.00°
      c    u_2    90.0°    +0.602    -0.273      0.661     41.37°    41.37°
      c    u_3    90.0°    +0.799    +0.206      0.825    124.44°    55.56°
      c    u_4    90.0°    -0.602    +0.273      0.661    138.63°    41.37°
      c    u_5    90.0°    -0.799    -0.206      0.825     55.56°    55.56°
      d    u_1     0.0°    +0.000    +0.940      0.940     70.00°    70.00°
      d    u_2   180.0°    +0.000    -0.940      0.940    110.00°    70.00°
      d    u_3–u_6                              same as (c)
```

`u₁ = +z` exactly in every panel, so it projects to `(0, +0.940)` — straight up the
page, the same arrow, four times. Panel (a)'s three rays land exactly where
`three_pushes.png` puts them.

**`three_pushes.py`'s 84° rule cannot be kept and its 20° rule can.** That file keeps
every push between 20° and 84° of the eye: 20° because a ray nearer the view axis than
that keeps under a third of its length on the page and reads as a dot, 84° because 90°
is the limb and a push round the back puts the paint it bounds round the back with it.
**Four, five and six rays cannot all be on the near side.** Five rays whose azimuths run
all the way round — which is what a square pyramid is — put at least two of themselves
more than 90° from any eye whatever. So the rule becomes: no ray within `SHY = 20°` of
the view axis **or of its opposite**, which is the condition an arrow needs in order to
read. Measured over all eighteen rays the worst margin is `26.33°`, panel (b)'s `u₃`.
The paint's own condition is separate and is asserted where it belongs — see below.

**The azimuth of each arrangement about the vertical is FREE and is spent.** Turning a
set about `+z` carries it onto a congruent set, so coverage, weakest, strongest and
share are all invariant and only the pose moves. `three_pushes.py` deliberately refuses
that freedom, because its three panels are ONE arrangement opening and closing and
turning it between panels would make "the two swing up" false. Here the four panels are
four different objects and there is no such sentence to protect. The phases were picked
off the same table the run prints — 20° for the tetrahedron, which keeps the worst
margin at `26.33°` while putting three of its six pale lenses on the near side, and 25°
for the four horizontals, shared by (c) and (d).

**Far rays are drawn OVER the ball, dashed, with a hollow dot at the foot.** That is the
same deliberate lie about depth the interior radii tell, and here it is not a nicety.
Honest depth deletes them: an arrow standing on the far side projects INSIDE the disc —
its tail lands at `1.03·√(1−(u·EYE)²)` of a radius, under 1 whenever the ray is more
than a few degrees over the limb — so an opaque painted ball hides it completely, and in
(d) that would silently turn six pushes into three. A dash is the ordinary way to say
"behind what it crosses".


## Decisions that look free and are not

**The paint-in-front-of-shell split needed a weaker condition than `three_pushes.py`'s,
and the weaker one is measured rather than argued.** The ball is drawn as two
`Poly3DCollection`s — bare cells with no edge line, painted cells with an edge line in
their own fill colour — because Agg leaves a hairline between adjacent polygons and
closing it with an edge only works where the fill is opaque, and because a per-face
`linewidths` array does not survive `Poly3DCollection`'s depth re-ordering (it permutes
the face and edge COLOURS and leaves the widths where they were). Splitting costs depth
sorting between the two collections. `three_pushes.py` can assert its whole patch is on
the near side; **that is false here — the square pyramid paints the entire upper half of
the ball.** What "paint in front" actually requires is weaker: wherever a painted
direction is on the far side, the point of the ball at the same place on the PAGE — its
mirror in the plane square to the eye — must be painted too, so that nothing bare is
being drawn over. For the square pyramid the mirror of a far point with `z ≥ 0` has a
larger `z`, so it holds; for (b) and (d) there is no bare cell at all; for (a) no
painted cell is on the far side. The assert checks that on 40000 directions in every
panel, and it is the thing that will trip first if anyone adds an arrangement.

**`VMAX` is `√3 F`, read off `strongest` rather than typed, and it is not
`three_pushes.png`'s `2.62 F`.** Neither figure may clip its own strongest point, so
the blue ramp's far end has to follow the panels. The cost is real and worth stating:
panel (a) is the same arrangement as `three_pushes.png`'s panel (a) and is painted with
a different stretch of blue. The break at `F` is in the same place in both, which is
the reading that matters, and each figure carries its own key. The alternative — pinning
this figure to `2.62 F` — would leave a third of its ramp unused and squash the whole
figure into the pale end, when three of its four panels reach exactly `√3`.

**`LOWER` had to go to 0 and the PAGE had to grow to pay for it.**
`three_pushes.py` lowers its framing cube by 0.28 of a radius, because in all three of
its panels the vertical push stands out of the top and nothing balances it underneath.
Panel (d)'s sixth ray is the lid: it points straight DOWN and reaches
`1.03 + 0.42/0.940 = 1.477` radii below the middle, outside a cube lifted by 0.28, and a
3-D axes CLIPS to its rectangle rather than overflowing it — the first render simply cut
the arrowhead off. **And the zoom is not the lever it looks like.** Measured: the drawn
size of the ball depends on the axes rectangle's WIDTH alone and not at all on its
height — a figure 0.9 in taller at the same zoom draws a ball of exactly the same 280
pixels. So a shorter page cannot be paid for by zooming out without shrinking every
ball, and at 6.9 in the zoom that fitted (d)'s downward arrow was 1.62, drawing balls
2.80 in across against `three_pushes.png`'s 3.20 in — 12 % smaller than the figure whose
panel (a) this one repeats. At **7.8 in** the fit is **1.82** and the balls are 3.15 in.
Every text row is placed at the same INCH offset from the page edge as it had at 6.9 in,
so nothing else moved.

**The angle label is placed by a search and not by a fraction, and the reason is
specific.** For every regular arrangement here the midpoint of the arc between two rays
is the direction that PAIR pulls hardest in — the tetrahedron's six strongest directions
ARE its six edge midpoints, and (b)'s pale lenses stand exactly there — so a
paper-backed label at the arc's midpoint sits on top of the one feature the panel exists
to show. It did, on the first render, hiding the largest of (b)'s three visible lenses
almost entirely. `label_spot` instead walks the arc and takes the point whose PAGE
position stands furthest from everything else drawn: the rays' dots, the dashed radii
as SEGMENTS (a label clear of both ends of a line can still sit across its middle), the
centre, and the strongest directions. A lens is a region about 30° across and a radius
is a hairline, so a ball radius of clearance next to the first is worth less than next
to the second; `LENS_W = 0.45` buys that, and without it the search settles for clipping
a lens's corner in order to stand further off a dash.

**An opposed pair has no angle arc, so one is chosen.** Every great semicircle from `u`
to `−u` is 180° and picking one is picking a plane. The one picked is the semicircle
through the direction square to the pair that stands furthest from every other ray,
with the tie — the choice always comes in mirror pairs — going to the one facing the
camera. Without the second half the label lands behind the ball as often as in front of
it; without the first, (d)'s 180° mark runs straight down two of the other four pushes.

**Only ONE angle is marked per panel, and it is the WIDEST pair.** There are up to
fifteen of them. What the figure argues is that the widest pairwise angle predicts
nothing, so the widest is the one worth drawing, and the reader is meant to read it
straight off against the line underneath saying whether the guarantee held.

**`three_pushes.py`'s mechanism rings are gone, on purpose.** They mark `uⱼ × u_k` for
each pair, and their walk into the patch is that figure's whole account of the
threshold. With five rays there are ten of them and with six, fifteen; and worse, the
thing they stand for has changed — the bound `m(dᵢ) ≤ F(uᵢ·dᵢ)` they carry is the ONE
term the sum `h(a)` has when there are three rays, and it stops being the whole story
at four. Drawing ten rings that each state a bound which no longer binds would be
drawing the superseded argument. What replaces them is the `h(a)` table above, and the
red region in (b) with the exact `m = F` curve round it.

**The `m = F` curve is drawn from the algebra, and only panel (b) has one.** `m(v) ≥ F`
is `v ∈ Z`, so the curve is the sphere met with each facet plane `a·v = h(a)` — a circle
at angular radius `arccos h(a)` — kept only where no OTHER facet is already violated.
That second condition is the same one `three_pushes.py` needed and is just as easy to
miss: `a·v = h(a)` holds all the way round its circle, and a stretch of it can run
through the inside of the region where another facet is over ITS bound, where the field
is below `F` on both sides and the curve bounds nothing. In (a), (c) and (d) every facet
stands at distance `F` or more and merely touches the ball, so the run comes back empty
and the panels have no ink curve at all — which is itself the statement that nothing in
them is below `F`.


## Reading the picture

- **The orange arrows are the rays**, each capped at `F`. Same colour, same weight, same
  PAGE length everywhere — length carries no number, the paint carries the magnitude —
  so what changes across the row is only how many there are and where they stand. Solid
  with a filled dot is the near side; **dashed with a hollow dot is round the back.**
- **The topmost arrow is the same arrow in all four panels.** It is the floor.
- **The paint is `m(v)` and is read as three states with a hard break.** Red is a
  direction they can push but not with a full support behind it; blue is one support's
  strength or more; **bare white shell is a direction they cannot push at all**, which
  is a third state and not a small magnitude, and is why it is left unpainted rather
  than given the bottom of the ramp. (a) and (c) have bare shell; (b) and (d) have none.
- **(b)'s pale lenses are the only directions the tetrahedron can still put a full `F`
  into**, and they sit BETWEEN the pushes, at the six edge midpoints, not on them. The
  ink curve round them is `m = F` exactly.
- **The ellipse across the ball is the HORIZON**, `z = 0`, seen from 20° above. In (c)
  and (d) the four chock pushes stand exactly on it; in (c) it is also the rim of the
  cone, which is why the orange runs along it.
- **The dashed radii and the arc between two of them** are inside the ball, where they
  are. The angle written is the arrangement's widest pair.


## Four bugs the rebuild exposed, all silent

Clustering the rays broke things that had worked only because every earlier arrangement
was axis-aligned, with coordinates of 0 and ±1 whose cancellations were exact. None of
them raised.

**`facets()` left rounding residue in the wall support values.** A generator lying IN a
facet plane must contribute exactly nothing to `h(a) = cap Σ max(0, a·uᵢ)`, and `a` comes
out of a cross product, so `a·u` for such a generator is `~5e-10` rather than 0.
`max(0, ·)` keeps it. A WALL of the cone — whose `h` is zero precisely because every
generator lies in it or behind it — then came back at `h ≈ 5e-10`, and `reach = h/(a·v)`
read a small positive value on the whole outside of that wall. **6416 of 20480 mesh cells
were painted for a cone covering 1574 of them**, all at the ramp's black end, so the patch
grew a torn black fringe. Fixed by clipping `|a·u| < FLAT = 1e-7` to zero: a real
contribution is `O(0.1)` or larger, so the threshold is not delicate. After: 1562 cells,
and `reach` straight down is exactly 0.

**`share()` counted the sphere minus a few caps, ignoring the cone's own walls.** It
subtracts a cap for every facet with `0 < h < 1` and skips those with `h ≤ 0` — but those
are the walls, and while every cone was the whole ball there were none. For a cluster it
returned `92 %` of the ball for a cone covering `7.7 %`, printing "1199 % of the cone at
or above F". Now the easy case is taken first and exactly: **if the weakest direction is
already at or above the cap, the share at or above `F` IS the cover** — no caps, no
inclusion–exclusion. The cap formula is left for the case it was written for and asserts
that the cone is the whole ball before using it.

**The paint could not be one collection in front of the shell.** The earlier version
asserted no bare near-side cell ever covered a painted far-side one, which held for its
arrangements and stopped holding at once for a cluster: a cone leaning 45° round the
vertical wraps past the limb, so part of its patch really is round the back, and a single
collection in front drew that part THROUGH the ball. The paint is now split at the limb
and drawn twice, near half over the shell and far half under it — which costs nothing,
because the shell between them is 13 % opaque and is what makes the far half read as far.

**The LP and the closed form disagreed by 1.8e-6 on grazing directions**, from the same
residue: at the wall both `h` and `a·v` go to zero together and their rounding no longer
cancels. The check now compares only STRICTLY INSIDE the cone and compares relative
error — on 400 random interior directions of the clustered triple the two agree to
`2.2e-16`. A `WALL_EPS = 1e-7` deadband keeps `reach` itself off the same `0/0`.


## Known open issues

- **Panels (a), (b) and (c) carry visible empty space under the ball**, because the
  framing cube is shared and panel (d) is the only one with a ray pointing down. Giving
  each panel its own cube would recover it and would make the four balls incomparable in
  size, which is the one thing the row cannot afford.
- **(b)'s six lenses are unevenly placed on the page** — three on the near side at
  `32.0°`, `61.3°` and `77.0°` off the eye, the last of those crowding the limb, and
  three round the back at `103.0°`, `118.7°` and `148.0°`. That is the tetrahedron's
  3-fold symmetry against a camera that cannot sit on its axis, and no azimuth fixes it;
  20° is the phase that gets three clear while keeping the ray margin at 26°.
- **The angle label in (b) now sits across the vertical dashed radius** — the search
  traded a lens for a dash, deliberately, and a dash crossing behind a paper-backed box
  is the ordinary reading.
- **The painted cells still show a faint 2° texture** where the field is steepest, the
  colour being taken at each cell's centre. Most visible in (d), whose field runs from
  `F` to `√3 F` twelve times over.
- **`share()` only handles equal cap depths** and asserts it. Every arrangement here is
  regular enough for that; an uneven one would need the unequal-radius lens formula.
- **The output path is absolute and hard-coded**, as everywhere else in `slides/tools/`.
- **`VMAX` is a module global set in `main()`.** `ramp`, `key` and `onpaper` all read
  it, so importing the module and calling `ball()` without going through `main()` raises.


## Where it sits beside the other figures

| file | what it answers |
|---|---|
| `dimension_ball.png` | WHICH directions a contact set reaches — point, arc, patch |
| `dimension_pairs.png` | how hard, for TWO pushes, as a fan along the arc |
| `three_pushes.png` | how hard, for THREE, as a field over the patch, against the angle |
| **`more_pushes.png`** | **how hard, for FOUR and FIVE — clustered, and spread; where the angle rule dies** |
| `min_force.png` | the weakest direction of a PAIR against the angle, in the plane |

**`three_pushes.md` and `three_pushes.py` both had to be corrected**, and only in the
one place where they generalise the 90° rule beyond three supports. Both said, in the
same words, that "the threshold is 90° for three of them and 90° for any number: a
fourth support does not mend a direction that two of its neighbours are square to."
The square pyramid mends exactly that. Neither the figure nor any number in it changed
— `three_pushes.png` is byte-identical, `85f3b87ef02a9a87fb9ee54ae8e61d6c`, before and
after — because the claim was in prose and never in the arithmetic.

`ninety.py` and `ninety.png` were **deleted** on 2026-08-14. They were an unasked-for
fifth figure from an earlier pass, carrying the 90° threshold as a continuum plus a
tetrahedron panel; this file supersedes both halves, and the `uⱼ × uₖ` mechanism it
derived is recorded above. Nothing references them any more.


## The file

| file | md5 | note |
|---|---|---|
| `more_pushes.png` | `b9bf3efd7a5fb11e227581be52c3b56e` | four balls, 3607 × 1608 |
