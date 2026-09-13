# Current setup slides: five object/pose examples

![Five target poses](target_poses.png)

`presentation.py` reads the saved B/pose_2, B/pose_3, A1-f/pose_2,
A1-f/pose_3 and C5/pose_2 snapshots. All five use the shared white canvas,
grey workpiece, green working surface, camera and compact floor.
The separate [working-area diagram](../working_area.png) shows multiple possible
process forces acting at different positions and directions on the green patch.
These are separate possible loads, with illustrative arrow lengths.

Run `python slides/render.py --only setup` in the cadgrasp environment.
Each selected case writes its own `pose.png`; `target_poses.png` collects all five.
B/pose_2 also writes `target_pose.png` and `tip_B.png` for existing slide links.
The figures depict held target poses; no new placement trajectory is asserted.

Old figures were deleted. The full pose catalogue and setup inputs are preserved.
`target_poses.py` remains a separate dataset builder; rendering never invokes it.
The numerical experiments below are historical and do not describe the new images.

## Historical tipping search

# `big_tip` — flat on the floor, then over onto ONE point

The setup slide, in three pictures a row: **a workpiece nobody designed for this was
standing on the floor; somebody chose one point of its footprint and pushed it over that
point through 60 to 70 degrees; it now hangs in a pose it has no equilibrium in, past the
angle where gravity stops resisting the tip and starts driving it, with the patch a
process has to reach painted on the surface it can no longer be laid down to reach.**

**One page an object, up to `ROWS` = 5 rows of it, written to `tip_<name>.png`** — the filenames
`tip_sequence.py` also writes, and this is what they now hold: the same three columns
through the same page fitter, the tips five times bigger, one mark instead of four. Run one
script or the other, not both; `slides/setup/poses/README.md` carries the same warning beside that
script's own md5 table.

**As many rows as the workpiece has DIFFERENT tips, up to five, and no page is padded.**
A1-f and C5 have five; B has two and gets a two-row page. What widened on 2026-08-28 is
where the placements come from — `poses.json` is what a physics sampler happened to find,
and when `spread` comes up short the convex hull is enumerated for the rest — and that is
stated in full two paragraphs down, beside the belief it replaced.

The argument, the constants and every trap paid for are in the script's own module
docstring, beside the code they describe. This page keeps the pictures, the numbers and
the evidence for each.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u slides/setup/poses/big_tip.py
```

19 s on this machine, all three pages, and it prints every number they rest on.

| file | px | md5 | rows |
|---|---|---|---|
| `tip_A1-f.png` | 2384 × 3744 | `3a7c20cb1ec45a2b9da2b1ce6eb506fb` | 5 |
| `tip_B.png` | 2384 × 1584 | `5b3b5fd46687a60cc8afe27b60327316` | **2** |
| `tip_C5.png` | 2384 × 3744 | `939a8f266e94a087b3de7cb33f5ef9e8` | 5 |

The *of them different tips* column this table used to carry is gone, because it can no
longer be anything but the row count. **A1-f's and C5's md5s are the ones they had before B
was fixed**: the widening below is gated on `spread` coming up short, and a page that fills
itself never reaches it.

## B has TWO rows, and why it is not five

Until 2026-08-28 B's page was five rows of one tip, and this section argued that this was
the workpiece and not the code. **The finding was right and the reason given for it was
wrong**, so both are kept.

**What was believed.** Of B's 3600 (placement, bearing) pairs — five placements, 720
bearings each, none of them sampled — 31 clear 60°, 13 clear 60° + `MIN_WEDGE` = 68°, and 8
survive the rest of `candidates`; all eight are the same corner (vertex 64) of placement 4
over a 3.5° fan of bearing, 278.0° to 281.5°, all of them at *exactly* 60.0° because
`wedge = min(WEDGE, limit − 60°)` absorbs the whole surplus whenever the limit is under 80°.
`spread` calls those one tip and is right, so a `top_up` function drew the most *unlike*
four of them a second time and labelled each with the separation it achieved. The sweep
behind that was exhaustive over what it was handed — and **what it was handed was
incomplete**.

**What was wrong with it.** `poses.json` is a SAMPLE of the stable placements and not the
set of them: `slides/tools/drop_sample.py` dropped B forty times, found **nine** distinct
placements and kept **five** (`n_kept` = 5, coverage 0.85). A body resting on a plane rests
on a facet of its own convex hull, and B's hull is 262 facets over 231 distinct normals, of
which **fifteen are statically stable** — the centre of mass projects inside the footprint
of the vertices left within `TOUCH` of the floor. Five of the fifteen are `poses.json`'s.
So the old paragraph's "five placements, 720 bearings each, 3600 pairs" was one third of the
workpiece.

**What is actually true.** *The placement count was never the limit.* Enumerating all
fifteen and sweeping all 720 bearings on each:

Ordered by the widest window `limit − balance` any of its 720 bearings offers; *limit* and
*balance* are read at that bearing, and *best limit* is the largest landing limit anywhere
on the placement, which is usually a different bearing.

| | best limit | widest `limit − balance` | limit there | balance there | candidates |
|---|---|---|---|---|---|
| hull facet 233 | 67.89° | **39.73°** | 67.89° | 28.16° | 0 — misses `WANT[0] + MIN_WEDGE` = 68° by **0.11°** |
| placement 4 = **hull facet 110** | **72.71°** | **38.29°** | 72.71° | 34.43° | **8** |
| **hull facet 113** (new) | **71.48°** | **32.72°** | 71.41° | 38.69° | **5** |
| hull facet 247 | 45.54° | 20.25° | 34.59° | 14.34° | 0 |
| hull facet 80 | 52.43° | 18.35° | 22.08° | 3.72° | 0 |
| hull facet 50 | 51.67° | 18.20° | 49.74° | 31.55° | 0 |
| hull facet 70 | 46.79° | 13.53° | 27.66° | 14.12° | 0 |
| hull facet 145 | 28.09° | 13.37° | 24.21° | 10.85° | 0 |
| placement 3 = hull facet 144 | 29.25° | 12.01° | 15.72° | 3.71° | 0 |
| placement 0 = hull facet 24 | 38.78° | 8.55° | 25.72° | 17.17° | 0 |
| hull facet 25 | 35.86° | 6.47° | 22.76° | 16.29° | 0 |
| hull facet 128 | 31.22° | 6.15° | 13.83° | 7.68° | 0 |
| hull facet 154 | 29.61° | 3.88° | 29.61° | 25.73° | 0 |
| placement 1 = hull facet 5 | 24.16° | 3.65° | 15.71° | 12.06° | 0 |
| placement 2 = hull facet 117 | 10.96° | −2.94° | 1.99° | 4.92° | 0 |

**A round workpiece rolls onto another part of itself before the tip clears its own balance
point.** That is the whole of B's difficulty, and it is a fact about curvature, not about
how many placements were on the list: **thirteen of the fifteen never reach a landing limit
of 68° at all**, and the window a tip has to live in, `limit − balance`, exceeds 30° on only
three of them. (The old paragraph's "outside placement 4 the best limit anywhere on the part
is 38.8°" was placement 0's, and true of the five placements it knew about; over all fifteen
it is hull facet 233's 67.89°, which still buys nothing — 0.11° short of the 68° a 60° tip
with `MIN_WEDGE` of daylight needs.) A rotation-only tip of 60°+ needs a sharp-cornered
footprint and the bunny is round.

**So B gets two rows**, one from each of the two placements that can carry a big tip, and
the second of them (hull facet 113) is a placement `poses.json` never recorded. `top_up` is
gone; the row count is variable and nothing is padded. Two was the ceiling under the old
reading too — what changed is that the second tip is now *drawn* instead of a first tip
drawn four more times.

> **Facet 113 is a STATICALLY stable placement and is labelled as one.** Its weight comes
> down **1.72 mm** inside its own footprint, against 9.55 mm for placement 4. `rest_pose`'s
> static test passes it — the same test, run as a filter rather than an assert, is what
> `hull_places` screens with — but `slides/tools/drop_sample.py`'s forty MuJoCo drops never settled
> the bunny there, so **nothing has re-verified it dynamically**. A page that wanted a
> dynamically confirmed placement would have to widen `drop_sample`, not this script.

A combined `big_tip.png` and a set of per-panel PNGs were both written at earlier points and
both were dropped on request; `sweep` has every panel in memory if either is wanted back.

**Deterministic, and MEASURED to be**: run twice from scratch and the PNG comes out
byte-identical, with an identical log line for line. The only random draws are the work
region's seeds, from a generator seeded by the object's NAME (METHOD §7: never by list
index). A file that moves when nobody meant it to is a regression.

## The twelve rows

Every number below is off the run that wrote the three md5s above. **Every row is a
different tip by `spread`'s rule**, so the old *different tip?* column is gone; *seed* is the
face the work region's disc was grown from, which is what says two rows are not sharing a
patch.

| | placement | bearing | contact | tip | wedge | limit | balance | past by | `w(0)` | seed | region | of the drawn part |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A1-f** 1 | 4 | 272.5° | v95 | **70.0°** | 20.0° | 90.0° | 24.6° | **45.4°** | 98.5 mm | 148 | 13.84 % | 13.7 % |
| **A1-f** 2 | 4 | 87.5° | v0 | **70.0°** | 20.0° | 90.0° | 25.2° | **44.8°** | 98.0 mm | 19974 | 13.99 % | 13.5 % |
| **A1-f** 3 | 4 | 45.5° | v28 | **70.0°** | 20.0° | 90.0° | 32.0° | **38.0°** | 81.2 mm | 225 | 13.74 % | 38.3 % |
| **A1-f** 4 | 4 | 316.0° | v15 | **70.0°** | 20.0° | 90.0° | 31.6° | **38.4°** | 81.6 mm | 2473 | 14.00 % | 42.0 % |
| **A1-f** 5 | 4 | 224.0° | v81 | **70.0°** | 20.0° | 90.0° | 31.6° | **38.4°** | 81.6 mm | 148 | 14.63 % | 32.5 % |
| **B** 1 | 4 | 281.5° | v64 | **60.0°** | 8.8° | 68.8° | 34.7° | **25.3°** | 29.6 mm | 1038 | 10.18 % | 19.2 % |
| **B** 2 | **hull 113** | 270.5° | v50 | **60.0°** | 9.6° | 69.6° | 38.8° | **21.2°** | 26.2 mm | 1072 | 10.20 % | 9.9 % |
| **C5** 1 | 0 | 272.5° | v4755 | **65.5°** | 20.0° | 85.5° | 50.1° | **15.4°** | 22.4 mm | 10082 | 9.25 % | 3.0 % |
| **C5** 2 | 2 | 11.0° | v5597 | **63.9°** | 20.0° | 83.9° | 57.3° | **6.6°** | 26.9 mm | 10555 | 9.12 % | 9.3 % |
| **C5** 3 | 3 | 6.5° | v3 | **60.0°** | 8.0° | 68.0° | 42.9° | **17.1°** | 12.2 mm | 14068 | 13.30 % | 6.0 % |
| **C5** 4 | 0 | 305.5° | v4755 | **60.0°** | 13.5° | 73.5° | 55.0° | **5.0°** | 3.8 mm | 15586 | 12.93 % | 0.0 % |
| **C5** 5 | 3 | 303.5° | v267 | **60.0°** | 13.3° | 73.3° | 47.0° | **13.0°** | 14.0 mm | 10301 | 13.76 % | 28.9 % |

**B's two rows are two different poses of the bunny, not two pushes on one.** They start
from different placements — one physics-sampled, one enumerated, **39.3° apart in resting
normal** — so the part lies on a different part of itself in column 1 and arrives somewhere
else in column 3. They also turn on different vertices, v64 and v50, though those are only
12.26 mm apart on the mesh: both are on the same low-curvature foot pad, which is the only
thing on this workpiece that can carry a big tip at all. Within one placement `unlike` would
call two contacts that close the same tip; across placements it calls them infinitely
different, and the rendered page agrees. The region seeds differ too, which is a symptom
rather than a setting: `regions` is seeded `SEED ^ crc32(name)` per OBJECT — METHOD §7's rule, never by
list index — and draws its seeds from the faces admissible *at that row's own `T_star`*, so
when B's five rows were five copies of one pose, three of them (1, 4 and 5) drew the same
face 1038 and grew nearly the same patch. Two genuinely different poses give 1038 and 1072
with no change to the seeding at all. A1-f's rows 1 and 5 also both draw face 148, from
different admissible sets, and grow 13.84 % and 14.63 % of the surface from it.

**tip + wedge = limit, and the limit is the workpiece's own geometry.** That identity is
the whole of what this page can and cannot do, and it is not a convention — requiring
daylight under the workpiece (every vertex higher than `tan α` times its own horizontal
distance from the contact, a cone of air opening from the point it stands on) reduces
*exactly* to `φ − t ≥ α` for a vertex in the plane of the tip. So the degrees held back
from the limit ARE the angle the underside makes with the floor.

At 5° of it — which this page spent for a while, drawing A1-f at 85° of its 90° limit —
the underside is 5° off the ground, its own shadow fills the gap, and the panel reads as
LYING DOWN however big the number in the caption. The rule now is **spend everything above
60° on daylight, up to `WEDGE` = 20°**, and refuse a bearing that cannot afford
`MIN_WEDGE` = 8°. A1-f's 90° limit buys 70° + 20°; B's 68.8° buys 60° + 8.8°.

Every region is ONE patch inside the [8 %, 15 %] band. **Which rows** is `spread`'s: the
largest tips first, skipping any that is the same picture as one already taken — the same
placement, a contact within `APART` = 15 mm and a bearing within `SEP` = 25°. A1-f offers
1008 qualifying bearings and C5 250, so their five are a choice; B has 13 bearings over two
placements and they come to exactly two rows.

## How a pose is found

Two poses a row, and **neither is searched for numerically: both are closed form, off the
mesh and `poses.json`, with no simulator, no marching and no sampling over angles.** The
mesh throughout is the one `refine` returns — the raw mesh if it is finer than `COARSE` =
2000 faces, uniformly 4-split until no face holds more than `QUANTUM` = 0.5 % of the surface
if it is not — so the vertices the contact, the limit and the patch are read off are the
vertices that get drawn.

### The initial pose — the stable placement it starts from

1. **Read, not computed.** `objects/<name>/poses.json` is `slides/tools/drop_sample.py`'s: the
   workpiece dropped from uniformly random orientations and settled in MuJoCo, the settled
   trials clustered on the direction of gravity in the mesh frame (which is the whole of
   what decides a resting pose, yaw being free), the most frequent clusters kept with the
   cluster frequency as that placement's probability. Five an object, and **all five are
   tried** — which of them a row comes from is an output, not a setting. Only
   `T_world_mesh` is read; the recorded probability is not used to rank anything here.
2. **Re-seated** (`rest_pose`), and this is not cosmetic. A settle stops when the motion
   stops, not when the contact is exact, so the recorded pose carries a residual tilt —
   0.001° to 0.12° over these fifteen placements, a tenth of a millimetre across a 60 mm
   part, and enough to ruin what comes next, because the tip turns about a VERTEX and picks
   it as the one reaching furthest along the push *among those on the floor*. Under a
   residual tilt the patch's true outer vertex sits a hair ABOVE the cut, the vertex behind
   it is picked instead, and the tip then ends a few degrees later when the outer one comes
   down — which threw out every bearing of every C-series part before this function existed.

   A body resting on a plane rests on a facet of its own **convex hull**, so the exact seat
   needs no simulator. Take the hull face whose world normal has the most negative `z`, the
   one pointing most nearly straight down; turn the whole pose by the smallest rotation
   carrying that normal exactly onto `−ẑ` — about their common perpendicular, by the angle
   between them, so nothing is spun about the vertical that does not have to be; then drop
   until the lowest vertex of the mesh is at `z = 0`. The tilt taken out and the drop are
   returned and print on every placement.
3. **Two asserts, and between them they are what "a stable placement" means.**
   - The facet found was already within `TILT` = 3° of level (`arccos(−n_z)` in degrees), or
     the recorded pose is not resting on it and the fault is upstream, not here.
   - The centre of mass projects **inside the footprint** — the 2-D convex hull of the hull
     vertices standing at `z ≤ TOUCH` = 0.1 mm in the seated pose — tested by evaluating
     that polygon's own half-plane equations at the plumb point and requiring the worst of
     them to be ≤ 0. The margin comes back with the pose and prints: 2.4 to 49.2 mm over the
     fifteen recorded placements the three objects offer, and 0.5 to 5.8 mm over the ten
     `hull_places` adds for B.

   Neither test *rejects* a placement. Both are asserts, so a placement that fails one stops
   the run rather than being skipped — which is exactly why `hull_places` runs the second of
   them itself, as a filter, before handing anything to `rest_pose`.
4. **All the placements go into one pool.** Every candidate is tagged with its placement —
   an index for one of `poses.json`'s, `hull <facet>` for one `hull_places` enumerated — and
   that placement's own `T_rest`, and `spread` then chooses across the whole pool, so a
   page's rows may come from several placements or from one. If nothing anywhere on an
   object offers a single candidate the run stops (`assert picks`).

### The target pose — the tip

`candidates(mesh, T_rest)`, once per placement: **every one of `NB` = 720 horizontal
bearings, each a complete (direction, contact, limit) in closed form.**

1. **The push direction `e1`** = `(cos θ, sin θ, 0)` for `θ = 2πk/720` — the horizontal
   direction the workpiece is pushed over in. **This is the only free choice in the whole
   construction**; everything after it is forced.
2. **The contact is then FORCED**: among the vertices on the floor (`z ≤ TOUCH` = 0.1 mm in
   the seated pose) the one reaching furthest along `e1` — a supporting line of the
   footprint, METHOD §0's condition. Not a preference: any floor vertex further out along
   `e1` sits at `φ = atan2(0, +) = 0` and goes under the floor on the first degree of the
   turn. `TOUCH` is deliberately **not** `CONTACT_EPS` here — at a 1.5 mm tolerance the
   vertex reaching furthest along `e1` may be one floating 1.5 mm up, and the workpiece then
   stands on a pedestal in every panel of the row.
3. **The axis** is horizontal through that point, `ẑ × e1`, which carries the part up and
   over it. (METHOD §0 allows any line through a corner pivot; what the tilted ones are
   worth is measured under *why not just choose a better point?* — +1° to +5° of lean, for
   most of the rotation spent on yaw.)
4. **The limit, in closed form.** Write every vertex in the plane of the tip relative to the
   contact: `x` along `e1`, `z` vertical, `r = hypot(x, z)`, `φ = atan2(z, x)`. The rotation
   takes it to `(x cos t + z sin t, −x sin t + z cos t)`, so its height after a turn of `t`
   is `r sin(φ − t)` and it reaches the floor at exactly `t = φ`.

   > `limit = min φ` over the vertices with `φ > 0` that are further than `NEAR` = 2 mm from
   > the contact, capped at 180° — and 180° if no vertex qualifies at all

   Three exclusions, and every one of them was paid for:

   - **`r ≤ 1e-9` — the vertices ON the axis.** They do not move at all, and `atan2(0, 0)`
     is 0, which would read as a landing at zero degrees and kill the bearing. They come
     back for the patch test in step 6, which is where an edge pivot is actually caught.
   - **`φ ≤ 0` — the vertices behind the contact.** After the drop these can only be other
     vertices of the resting patch: `dz ≤ 0` forces `dx ≤ 0`, since anything at floor level
     *ahead* of the contact would have been the contact. The tip LIFTS them — their
     admissible interval is `[φ + π, φ + 2π]` — so they bind at `φ + 2π ≥ 180°` and not at
     `φ`. Reading `min φ` straight off instead makes a vertex one micrometre lower than the
     contact report a limit of −180° and throws the whole bearing away; A1-f's resting face
     carries 1199 vertices within 0.1 mm of the floor and every one of its bearings died
     this way.
   - **`r ≤ NEAR` = 2 mm — the contact's own neighbourhood**, which is a filleted corner's
     tessellation, rolling rather than landing. See *the contact rolls*; the dip this allows
     is priced by `DEPTH` in step 6 rather than forgiven.
5. **The split.** `wedge = min(WEDGE, limit − 60°)` and then `tip = min(limit − wedge,
   120°)`, where `WANT` = (60°, 120°) is the band of tips this page is for. The bearing is
   dropped if the wedge is under `MIN_WEDGE` = 8° or the tip under 60° — and a limit below
   60° gives a negative wedge, so it is dropped by the same test. Tip and wedge are one
   budget and the budget is the limit; see *the twelve rows*.
6. **Three filters, all evaluated at the drawn angle `t = tip` and all closed form.** Every
   height is `h = r sin(φ − t)`, over EVERY vertex this time, the on-axis ones included:
   they have `r = 0`, so `h = 0`, so they count as touching — **and that is the whole of
   what makes an edge pivot an edge, that its far end never lifts.** Leaving them out of the
   patch test, as the limit has to, let every edge bearing through as "one contact".
   - `patch` — what still touches, `h ≤ TOUCH`, must lie within `PATCH` = 4 mm of the
     contact. **This is the ONE POINT test.** An edge pivot's far end is 80 mm away and
     never lifts at all, so it fails by 80 mm; a filleted corner's own neighbours are inside
     it. The distance is measured in the pose on disk, because a rotation about a line
     through the contact does not change any vertex's distance from it. It replaced a "the
     second nearest vertex must clear 4 mm" rule that no refined mesh could survive: on a
     fine tessellation there are vertices 2 to 4 mm out, they rise to 2 to 4 mm at 85°, and
     every bearing of A4 was thrown out for it.
   - `wide` — the same width read at `CONTACT_EPS`'s 1.5 mm must be within `LIE` = 30 mm.
     Not the one-point test but a picture criterion: a workpiece with 50 mm of itself within
     a millimetre and a half of the floor READS as lying on that feature whatever the
     tolerance says. C5 has such a bearing — 0.52 mm wide at `TOUCH` and 49.99 mm at 1.5 mm,
     its leg pad almost flat on the ground — and it is not drawn.
   - `depth` — the deepest vertex of the whole mesh, `max(0, −min h)`, must stay inside
     `DEPTH` = 0.2 mm. This is what the `NEAR` exclusion costs, charged rather than hidden.

   `clear`, the height of the lowest thing NOT touching, is measured and printed beside them
   and is not a filter.
7. **What survives is recorded whole**: the bearing, the contact and its vertex index, the
   limit, the tip, the wedge, `clear`, `depth`, `patch`, `wide`, and where the weight ends
   up. That last is plane-of-the-tip bookkeeping, with the centre of mass at `(xc, zc)`
   relative to the contact and `s` its component along the axis:
   **`ahead = xc cos t + zc sin t`** is how far past the contact the plumb line lands in the
   plane the camera sees square-on; **`aside = s`** is the part of the same offset the
   camera looks down and foreshortens to nothing, and the turn does not change it;
   **`arm = hypot(ahead, s)`** is METHOD §11.4's `w(0)`; and **`balance = atan2(−xc, zc)` is
   the tip angle at which the weight passes over the contact**, so `tip − balance` is the
   *past by* column and `ahead > 0` says the same thing.
8. **Which rows get drawn** (`spread`). Sorted by the largest tip first, ties
   broken on the most legible overhang `ahead − |aside|` — which puts the weight across the
   page rather than into it — then taken greedily, skipping any candidate that is the same
   picture as one already taken: **the same placement AND a contact within `APART` = 15 mm
   AND a bearing within `SEP` = 25°**, all three at once. So a different corner of the same
   placement is a row, and so is the same corner pushed a different way. A1-f offers 1008
   qualifying bearings and C5 250, so their five are a choice. **`ROWS` = 5 is a ceiling and
   not a quota**: a workpiece with fewer different tips gets fewer rows, and nothing pads
   what this function returns. Every candidate `spread` looks at is asserted to have
   `ahead > 0` — **past the balance point is asserted, not preferred**: a candidate whose
   weight came down short of the contact would fall back to where it started, and stops the
   run rather than being quietly skipped. (`hull_places` therefore screens `ahead > 0` on
   what *it* adds: a recorded placement offering a candidate that would fall back is a fault
   upstream, but on an enumerated one it is a fact about a round part, so it is counted,
   printed, and never put in front of the assert.)

   > **The test and the distance are one function**, `unlike(c, p)`, added 2026-08-28:
   > `inf` for two candidates from different placements, otherwise
   > `max(‖Δcontact‖ / APART, ∠(e1, e1′) / SEP)`. Under 1 is exactly `spread`'s old
   > three-way `and`, so nothing about which five A1-f and C5 draw moved — **both pages came
   > out byte-identical to the run before it.** What the ratio adds is a reading of *how*
   > different, which a threshold does not have. `top_up`, the function that needed it, is
   > gone (step 9); `unlike` stays, because the `inf` it returns for two candidates from
   > different placements is what makes B's second row a row.

9. **And what happens when there are not five** (`hull_places`). `spread`'s rule is a
   threshold and a threshold does not degrade: a workpiece with one different tip gets one
   row, which is the truth and is also a page with four empty rows on it. B was that
   workpiece. **The answer is not to pad the page — it is to stop handing `spread` a third
   of the workpiece.**

   `poses.json` is a SAMPLE of the stable placements: `slides/tools/drop_sample.py` drops the part
   in MuJoCo a fixed number of times, clusters where it settles and keeps the commonest few
   (B: forty drops, nine distinct placements, five kept at 0.85 coverage). A body resting on
   a plane rests on a facet of its own CONVEX HULL, so the rest are enumerable in closed
   form. When `spread` comes up short — **and only then**, which is the whole of why A1-f's
   and C5's pages are byte-identical — `sweep` calls `hull_places`, which walks every hull
   facet in turn:

   - seat it with `down` (the smallest rotation carrying its normal onto −ẑ) and drop the
     mesh onto the floor;
   - take the footprint, the hull vertices left within `TOUCH` of the floor, and keep the
     facet when the centre of mass projects **strictly inside** that polygon. This is
     `rest_pose`'s own stability test, run here as a FILTER — `rest_pose`'s copy is an
     assert, and handing it a facet the part cannot balance on would stop the run instead of
     skipping the facet. Strictly inside, so `rest_pose`'s `deep ≤ 0` can never fire on what
     comes back. `sweep` asserts the two agree to a micrometre on every facet it takes.
   - drop it if its normal is within `DEDUPE` = 1° of one already taken, or of one
     `poses.json` has already given. A tessellated pad is several facets of one plane and
     standing on any of them is the same placement — and one placement entered twice under
     two names would be *infinitely unlike itself* by `unlike`, which is the bug this whole
     change removes.

   Then every surviving placement is swept whole, all 720 bearings, nothing sampled, and its
   candidates go into the same pool `spread` was already choosing from. **On B: 262 hull
   facets, 15 statically stable, 5 of them `poses.json`'s, 10 offered, and exactly one of
   the ten — facet 113 — yields a candidate.** `spread` is asked again and returns 2, and 2
   is the row count.

   > **Facet 113's weight comes down 1.72 mm inside its footprint**, against 9.55 mm for the
   > placement `poses.json` did record. `rest_pose`'s static test passes it, and MuJoCo has
   > never been asked: `drop_sample`'s forty drops never settled the bunny there. It is a
   > **statically** stable placement and this page says so rather than implying a physics
   > result it does not have.

   **What this replaced, and why that was wrong.** Until 2026-08-28 this step was `top_up`,
   which filled a short page from the same pool by repeatedly taking whatever was least like
   the rows already drawn, and labelled every row it added with the separation it actually
   achieved (B's four came out at 0.14, 0.06, 0.04 and 0.02 of what `spread` asks for). It
   was honest and it was not a picture: four of B's five panels were the same push from 0.5°
   to 3.5° away, all five at exactly 60.0° because the wedge absorbs the surplus below an 80°
   limit. Its docstring asserted "**B is that workpiece and it is not a bug**" and argued it
   from "five placements, 720 bearings each, 3600 pairs" — an enumeration that was exhaustive
   over `poses.json` and `poses.json` was five placements of fifteen. See *B has TWO rows*
   for what the corrected sweep says. `top_up` is deleted rather than narrowed: there is no
   longer any caller for "fill a page that cannot fill itself", because a page that cannot
   fill itself is now simply short.

   **What was measured and NOT taken**, because none of it buys B a *different* tip either —
   all of it swept over the full placement set × 720 bearings, with `spread`'s `ahead > 0`
   assert read as a skip so a low floor could be scored at all rather than stopping the
   sweep:

   | relaxation | B's different tips, over `poses.json`'s five placements | what it costs |
   |---|---|---|
   | band floor 60° → 55° / 50° / 45° / 40° / 30° (`MIN_WEDGE` = 8°) | 1 / 1 / 2 / 2 / 2 | rows captioned 60–120° that are 30–55° |
   | `MIN_WEDGE` 8° → 5° / 3° / 0° (floor 60°) | 1 / 1 / 1 | daylight the page exists to show |
   | `LIE` 30 mm → 40 mm with `MIN_WEDGE` 7° | 2 | `wide` 8.7 mm → 37.0 and 38.5 mm on the two rows it adds, **and row 1's own daylight, 8.8° → 7.1°** |

   Widening the *placement source* costs none of that and is what was taken instead. It buys
   B a second row and not a third: with all fifteen placements swept at the shipped
   constants, thirteen never reach a 68° landing limit at all. The band, the wedge, `LIE`,
   `PATCH` and `DEPTH` are untouched, and the same refusal stands — the closest call remains
   the `LIE` = 40 mm rung, which reaches B's second corner at the price of quadrupling the
   1.5 mm patch and taking daylight off row 1, and B's one-point claim is already the
   weakest of the three (*Known issues*).
10. **The rotation, and the re-seat.** `T* = rot_about_line(ẑ × e1, q, tip) · T_rest` about
   the footprint vertex `q` step 2 chose, then dropped so its own lowest vertex is exactly
   at `z = 0`; column 2 is the same thing at `MIDWAY` = 0.5 of the tip, re-seated the same
   way. The re-seat moves the pose by a
   fraction of a millimetre in EITHER direction and both directions are the same fact — a
   real contact is a rolling one and a mesh's is a chord of it. DOWN by up to `TOUCH`,
   because the vertex the tip turns on is only required to be on the floor to that tolerance
   and the pose would otherwise stand on a 0.07 mm pedestal; UP by up to `DEPTH`, because
   the fillet's next facet dips under as the contact rolls onto it. The signed amount is
   `lift`, and it prints.
11. **Then the contact is read BACK off the drawn pose** — the lowest vertex of `T*`, with
    its `z` zeroed — and the marker, the couple's arm and the plumb offset all use that,
    rather than the vertex the rotation was built about. `roll` is the horizontal distance
    between the two, and it is the contact migration this page allows and prints.
12. **Checked on the finished transform, not on the formula that chose it.** Nothing is
    under the floor by as much as `1e-12` m and the lowest vertex is within `1e-9` m of it,
    so the pose is neither through the ground nor on a pedestal; the touching set,
    re-measured in the drawn pose as a horizontal distance from the re-read contact, is at
    most `PATCH` wide; `outward` — the side of the pivot line the resting centre of mass is
    NOT on, which it computes and asserts is unambiguous — agrees with the push bearing to
    `1e-9`, so the tip really does turn the workpiece toward the outside of its own
    footprint; and `margin` asserts METHOD §11.4's couple arm against the flat
    plumb-to-contact distance this page would draw, which agree to `0.0e+00` m here.

## The three columns

| | shows | how it is built |
|---|---|---|
| **at rest** | the stable placement, with the chosen ground point already marked | `poses.json`, re-seated on the hull facet it rests on (`rest_pose`) |
| **over the point** | the same part halfway through the turn, with the push that makes it | `rot_about_line(axis, point, tip/2)`, arrow from `tip_sequence.push_site` |
| **the target pose** | `T*` — held nowhere, standing on the marked point, work region green | the full rotation, then `regions()` |

## What is checked, on every run

- **It is one rotation about one point of the ground.** METHOD §0's condition: the contact
  is never broken and never relocated. The one exception the tessellation forces is
  measured and printed rather than hidden — see *the contact rolls* below.
- **The workpiece clears the floor the whole way, exactly.** A vertex at `(x, z)` in the
  plane of the tip stands at `r sin(φ − t)` after a turn of `t`, so it reaches the floor
  at exactly `t = φ` and the limit is `min φ` — closed form, no marching, no sampling.
  The finished transform is then re-checked: the lowest vertex of the drawn pose is on the
  floor to `1e-9` m and none is under it.
- **It stands on ONE point.** At the drawn angle the set of vertices still ON the floor is
  under `PATCH` = 4 mm wide (0.00–0.73 mm over the twelve rows). An edge pivot's far end
  never lifts at all
  and fails this by 80 mm; it is what `PATCH` is for. **The generous reading is printed
  beside it and they differ**: at CONTACT_EPS's 1.5 mm tolerance the same patch runs to
  2.18–2.72 mm on A1-f, 2.49–3.74 mm on C5 and **12.23–12.42 mm on B**, whose feet are
  low-curvature pads that lie
  nearly flat — METHOD §11.10's strip, met again. `LIE` = 30 mm refuses a bearing whose
  1.5 mm patch is wider than that, because at 50 mm the part READS as lying on the feature
  whatever the tolerance says; C5 had such a bearing and it is not drawn.
- **The placement it starts from is stable.** The centre of mass comes down 2.4 to 49.2 mm
  inside the polygon the part stands on, asserted on each of the fifteen recorded placements
  the three objects offer, drawn or not — and 0.5 to 5.8 mm on the ten `hull_places`
  enumerates for B, where the same test is applied as a filter first and then re-asserted.
  The one enumerated placement that gets drawn, B's row 2, has 1.72 mm of it and has never
  been confirmed in MuJoCo; see *B has TWO rows*.

## Only the pivot is marked, and where the eye stands

**One mark, the point the workpiece turns on** — a filled orange disc inside a ring it
does not touch, drawn flat over the render so it is never behind the part. Everything
`tip_sequence`'s column 3 also draws — the centre-of-mass symbol, the dashed plumb line,
the red tipping-margin bar and its millimetres — is **computed, asserted and printed here
and drawn nowhere**. The push arrow stays: it is not a mark on the workpiece, it is the
agent that puts it there.

**And no words on the panels either.** The angles `tip_sequence` writes in each panel's
corner are gone; they are in the log, in this page's own table and in every row's record.
What is left on the sheet is the object's name, the three column heads and a row label.

**A grey floor, and no shadow.** The floor is `work_regions.SCENE`'s own infinite plane —
no thickness, no edge, not a table — darkened from the `rgba 0.97` it ships with to
`FLOOR_RGBA` = **0.80**, which is where a warm-white workpiece stands against it without
five rows of it becoming the subject. 0.88, 0.72 and 0.62 were rendered beside it. At 12°
of elevation the horizon is in frame, so each panel is a white strip above a grey ground;
`tip_sequence.render`'s flattening of the sky to white is what keeps that a clean edge
rather than the band it calls a wall.

**Every panel keeps its own floor** (`apart`, `GUTTER` = 10 px). That module's `page`
pastes its tiles flush, which is right when a panel ends in white sky and wrong the moment
the ground is drawn: the grey runs to the panel's edge, the three panels of a row join into
one continuous floor, and the eye reads a single scene cut by two vertical seams. Ten
pixels of page colour on each side is twenty between neighbours, and `PAPER` is the sheet's
own `#fcfcfb`, so the margin shows only where it has something to separate.

The **cast shadow is off**, and it is the one thing here that was decided by the geometry
rather than by taste: the scene lights from nearly overhead, so a part perched with its far
end 68 mm up threw its shadow 34 mm from under itself — the wedge `candidates` spends up to
20° of tip to buy came back filled with the part's own shadow, which is exactly the picture
of something lying down. A side light shifts the shadow clear and takes the floor's
brightness with it (`sin 63°` to `sin 41°` of the lamp, most of the contrast between lit
floor and shadow), so those panels came back with a flat grey floor and no readable shadow
anyway. A ground plate with a rim, and six ground colours, were also tried and dropped.

## Why not just choose a better point?

Asked directly, and the answer is that **the point is not the free variable**. Given a
horizontal push direction, the contact is forced: it must be the footprint vertex reaching
furthest along that direction, because any floor vertex further out sits at `φ = atan2(0,
+) = 0` and goes under the floor on the first degree of the turn. So "choose a better
point" is "choose a better direction", and all 720 of them are swept on all five
placements of each object.

What *is* free and was not used is the **axis direction** — METHOD §0 allows any line
through a corner pivot, and this page always took the horizontal one perpendicular to the
push. A tilted axis can steer the colliding feature sideways instead of into the floor,
and it has the same closed form: `z'(t) = C + A cos t + B sin t` with `C = (u·a)a_z`,
`A = u_z − C`, `B = (a×u)·ẑ`, so the landing angle is `atan2(B, A) + arccos(−C/M)`. It
was measured over 1410 sampled axes per contact, against the exact 720-bearing horizontal
sweep, scoring the LEAN (`arccos(a_z² + (1−a_z²) cos t)`) rather than the turn, because a
tilted axis spends part of its rotation on yaw and yaw does not read as a tip at all:

| | best lean, horizontal axis | best lean, any axis | what the tilted one costs |
|---|---|---|---|
| A1-f | 85.0° | 89.7° | a 45°-tilted axis and a **194°** turn |
| B | 67.7° | 69.0° | a 35°-tilted axis, 88° turn |
| C5 | 80.5° | 74.6° | — (the sampling never beat the horizontal sweep) |
| A2 | 119.2° | 128.5° | a 26°-tilted axis and a **212°** turn |

**+1° to +5° of lean, for most of the rotation spent on yaw** — column 2 stops reading as
*pushed over* and starts reading as *twisted*, and the camera can no longer look along the
axis at a 12° elevation. Not taken.

**So the ceiling is the workpiece.** `limit` is 90° on A1-f, 85.5° on C5 and 68.8° on B,
and tip + wedge cannot exceed it. Both at once — a big tip AND obvious daylight — needs a
part with a bigger limit: `A2` reaches 129.2° and `A5` 130.2° on essentially every bearing
they have, which would buy a 100° tip with a 29° wedge. One line in `OBJECTS`.

## The contact rolls, and it is priced rather than hidden

A convex corner in this library is FILLETED, and a fillet does not tip about a point — it
**rolls**, the contact migrating as the part turns. Its tessellation says so: C5's tip is
stopped at 76.9° by a vertex **0.27 mm** away, which is not a second contact but the same
one, one facet along. So vertices inside `NEAR` = 2 mm of the contact are left out of the
limit and their dip below the floor is priced instead (`DEPTH` = 0.2 mm, never more than a
few micrometres in practice), and the drawn pose is **re-seated** — dropped or lifted so
its lowest vertex is exactly on the floor. Measured over the twelve rows: A1-f and B
0 µm on every row, C5 −69 to +109 µm; contact migration 0.00 mm everywhere except C5's
rows 2, 3 and 5, at 0.60, 0.17 and 0.97 mm. The marker, the couple's arm and
the red bar are then all read back off the *drawn* pose, so they stand on the point the
part actually touches.

## The work region is regenerated here, and that is forced

`objects/<name>/region/region.json` stores one patch per pose of `tips.json`, and these
are not those poses. The rules are that file's own `seed_rule` and `growth_rule`, quoted:
an area-uniform seed among faces that turn up far enough for a gun above the part
(`n_z > 0.35`), stand clear of the floor band and are visible from outside along their own
normal; then a geodesic disc grown by Dijkstra on the face-adjacency graph weighted by the
distance between face centres, never carried past 15 % of the surface, stopping at a target
drawn from [9 %, 14 %].

> **The disc grows only through faces that pass the SEED'S OWN TEST, changed 2026-08-27.**
> `region.json`'s rule as written filters the seed and then lets the disc run with nothing
> in its way but the floor band, so a patch starts on a face a gun above the workpiece can
> see and crawls over the edge onto the underside — which that gun cannot reach at all.
> **0.1 % to 42.9 % of the patches drawn before the fix were surface no declared process
> touches**, and those faces are what make `slides/sys_floor`'s row (3) blow up: their
> inward normal points UP, so the 15° push cone on them reaches within a hair of straight
> up, and one body weight pushed there leaves the assembly's total normal force at 0.027 —
> on the point of floating, with the load landing 3 to 8 part widths away.
>
> The fix is not a new assumption — *the gun is above the workpiece* was already declared —
> and it is feasible on every pose: the reachable set is **41–88 %** of the surface with a
> largest connected component of **40–88 %**, against a band asking for 8–15 %. What it
> costs is legibility: the patch now sits on the upward-facing side, which a 12° camera
> sees nearly edge-on, so it draws over 0–42 % of the part where it used to draw over
> 10–40 %. Two of C5's rows show almost none of it.

Twelve are grown and **all twelve print**; the one drawn is the one that covers most of
the drawn part. A work region is a declared INPUT to the problem (PIPELINE §1), so
choosing which patch a slide declares is legitimate — choosing one and not saying how is
not. The spread between candidates is large and is the reason the choice is made at all:
on C5 the twelve run from **3.4 % to 26.0 %** of the drawn part, all of them legal patches
inside the band.

> **Scored on the DRAWN area, not on the facing area.** `tip_sequence.facing` measures the
> share of a patch's own surface that turns toward the eye, which is right when the camera
> is being walked round a fixed patch and wrong when the patch is what is being chosen: a
> flat face seen nearly edge-on scores 100 % there and draws as a green thread. An earlier
> version of this page went out that way.

## What it borrows, and the three things it does not

The framing, the flat overlay, the push arrow, the tipping margin and the page fitter are
all `tip_sequence.py`'s, imported rather than copied — this file is that fitter's fourth
caller and not its fifth copy (`torque/demo_fig/README.md`'s standing complaint). What it
does **not** take is that module's `choose`: the camera is `cameras`' here — pinned on
the axis and scored on whether the mark lands where the contact is — for the reasons under
*only the pivot is marked*. Three of its
constants are set from here rather than edited there, each for a stated reason:

- `TITLES` — the three column heads.
- `NOTE` — `place` keeps a label off the whole bottom strip rather than its bottom-left
  corner. It is inert now that the millimetres are not drawn, and is left set so that
  restoring the margin bar does not silently reintroduce the collision.
- `MUTED` — the note is written darker, because at 12° of elevation and 85° of tip the
  part throws a long shadow across the bottom of the panel and `tip_sequence`'s grey is
  within a few levels of it.

## Which workpieces can be drawn this way

`A1-f`, `B` and `C5` — the working set the rest of `slides/poses` and PIPELINE §3 draw, so
the same three parts carry every page of the deck. They are not the easy three. Swept over
the whole library with the 60° floor lifted (720 bearings a placement, five placements an
object), **15 of the 21 objects clear 60° somewhere and 6 cannot**:

| | best tip | |
|---|---|---|
| roomiest | `A2` 119.2°, `A5` 120.0° | every bearing, or nearly |
| comfortable | `A4`, `cuboid_baseline`, `A1-f`, `A1-s`, `C7`, `A3` 80° | hundreds to thousands of bearings |
| tight | `C2` 79.5°, `D8` 79.2°, `C3` 76.3°, `C5` 76.0°, `C6` 72.7°, `C4` 69.5° | 19 to 251 bearings |
| tightest | **`B` 62.7°** | **9 of its 1784 bearings** — its own body overhangs its feet |
| cannot | `C1` 54.8°, `D1` 54.2°, `D3` 45.7°, `C8` 23.2°, `D4` 20.5°, `D2` 11.1° | a second point lands first |

Any of them can be drawn by editing `OBJECTS`; nothing else in the file knows which parts
these are.

> **This table is over `poses.json`'s five placements an object and has not been re-swept
> against `hull_places`.** It is therefore a LOWER bound on every row of it, and B is the
> proof: every number in its row was measured over five of the bunny's fifteen statically
> stable placements, and the placement that gives B its second row is not among the five.
> The six objects listed as *cannot* may or may not stay there. Re-sweeping the library through
> `hull_places` is unwritten; the three drawn objects are measured exhaustively either way,
> because `sweep` enumerates whenever `spread` comes up short.

## Known issues

- **One of the twelve rows still pays the `HIDDEN` charge** (C5's second): no rung of `OUT`, at either
  end of its axis, leaves the contact unoccluded at the target pose, so its mark is drawn
  over a face that is in front of it. The marker being flat over the render is what makes
  that survivable; an elevation chosen per row is the unwritten fix.
- **The floor says where the ground is; nothing says the part is off it.** With no cast
  shadow the wedge — 8–20°, real, and drawn in the silhouette — is the only cue that the
  workpiece is not resting along an edge. That is the standing cost of the look, and it is
  why `cameras` scores `drop` at all.
- **A4 and `cuboid_baseline` need their meshes split before a region can be grown on them**
  and `A2` does not get it: the split is uniform and gated at `COARSE` = 2000 faces, so
  A2's 14.37 %-of-the-surface face stays whole and is a legal one-triangle work region.
  Splitting only the faces over the quantum leaves T-junctions, `face_adjacency` loses the
  seam, and the Dijkstra disc stalls under the band — A1-f's patches came out at 4–7 % that
  way. A red-green refinement would fix both and is unwritten.
- **B's contact is one point only at the tight reading.** 0.00 mm wide at `TOUCH`, 12.23 and
  12.42 mm wide at CONTACT_EPS's 1.5 mm on its two rows: it turns on a low-curvature foot
  pad, so the surface takes twelve millimetres to climb one and a millimetre is what the
  tolerance allows. Both numbers print. Nothing here decides which reading a paper should
  quote — and it is why the `LIE` = 40 mm rung under *what happens when there are not five*
  is still refused now that it would not even buy a row: the one-point claim on B is the
  page's weakest and is not the place to buy a picture.
- **B's page is two rows, and two is the workpiece.** This bullet used to read *"B's page
  is five rows of one tip and there is no fix inside this script"*, and named the fix as
  upstream: `poses.json` keeps 5 of B's 9 recorded distinct placements, so stable placements
  of the bunny were never offered to the sweep at all. **That half was right and the
  conclusion drawn from it was wrong** — the fix was not upstream, because the hull carries
  every stable placement whether a sampler found it or not, and `hull_places` now enumerates
  them here. What survives of the old bullet is the count: B has 15 statically stable
  placements, 13 of them cannot reach a 68° landing limit, and no amount of further
  enumeration will make a two-row page a five-row one. The remaining degree of freedom is
  still METHOD §0's free **axis direction**, measured under *why not just choose a better
  point?* and not taken — worth +1.3° of lean on B for an 88° turn about a 35°-tilted axis,
  which is a different construction and not this page's.
- **The one enumerated placement drawn is statically stable and nothing more.** B's row 2
  stands on hull facet 113 with its weight 1.72 mm inside the footprint. `rest_pose`'s test
  passes it; `drop_sample`'s forty MuJoCo drops never landed there, so no dynamic
  verification exists for it. A slide that needs one would have to widen `drop_sample`, not
  this file.
