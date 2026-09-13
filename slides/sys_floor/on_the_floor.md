# Floor loads: five object/pose examples

The current [five landing clouds](on_the_floor.png) and B/pose_2 [resultant diagram](row3.png)
share the workpiece, camera, colours and compact floor of `head_total_force.png`.
Run `python slides/render.py --only floor` in the cadgrasp environment.

`presentation.py` reads saved paired loads for B/pose_2, B/pose_3, A1-f/pose_2,
A1-f/pose_3 and C5/pose_2 without changing baseline outputs. Each case writes
`on_the_floor_{object}_{pose}.png` and matching provenance JSON. It independently
recomputes all 32,768 landings per case and checks them against
Step4's corresponding rows (the saved array also begins with a gravity-only row).
Hidden floor points are occluded by the workpiece. The cloud is sampled, not a
continuous boundary certificate.

World coordinates are Y-up. About the floor origin, let
`W = -mg e_y + F_push`, `M = c x (-mg e_y) + q x F_push`, and `N = -W_y`.
For positive N the required floor point is `p = (-M_z/N, 0, M_x/N)`.
The illustration's downward process force and gravity are parallel; their
weighted application point defines the resultant line. No intersection of skew
3-D force lines is assumed. Convex-hull containment is necessary for tipping
resistance; friction and yaw still require a joint bearing check.

`on_the_floor.py` and `row3.py` now render the current diagrams by default.
Old figures were deleted. The numbers and coordinate conventions below describe
earlier experiments only; they do not describe the current images.

## Historical experiments

# `on_the_floor` — the target pose, with the landings on the ground

**2026-09-11 接入 baseline：** `support_polygon.cop` 的整体压力中心公式被新 Step4 的等价六维需求映射采用；当前 baseline 使用 Step1 的当前姿态及 0–0.5mg 全范围，含零加工力，并另存连续外包。下面的旧图、固定力度和旧姿态样本不作为当前认证。见 [Step4](../baseline_algo/step4_floor_contact/README.md)。

**Current scope (2026-09-06): [problem_statement.md](../problem_statement.md#当前决定与讨论记录).** Only the full-load set
`L_K={p(q,d,K)}`, `K=0.5`, is required. References below to varying `[0,K]` describe an
optional extension and historical area measurements; they add no zero-load requirement.
The floor hull check treats the workpiece and its supports as one assembly and uses the
convex hull of the assembly's floor contacts. Separate support-by-support floor checks
are outside the current scope; see [equations.md](../setup/equations/equations.md).

`slides/setup/poses/big_tip.py`'s target pose, **unchanged** — same pose, same camera angles,
same grey floor, same green work region, **and no mark on the contact** — with the
places the load can land scattered on the ground. Nothing else on the panel.

`row3.py` / `row3.png` explain the construction in the drawing's plane: intersect the
two coplanar load lines at `X`, then extend their resultant to the floor at `p`.
The construction remains planar. `F_push` is the complete force vector, and both the
resultant equation and arrow label use that notation. The quantified load is fixed at
`|F_push| = 0.5 mg`; no range of force magnitudes is required.

**Two declarations changed on 2026-08-27, and between them they are the difference between
a fixture and a building.** The work region now grows only through faces the process can
reach (`big_tip.md`), and **`K = 0.5`**: the process pushes with half a body weight, not a
whole one. The load's reach from the plumb point went from **0.48–7.78 part widths to
0.18–0.32**, and **3 of the 11 poses that had no bounded region at all now have one** —
every pose does.

```
cd /Users/yuanboli/Documents/GitHub/cadGrasp
source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
python -u slides/sys_floor/on_the_floor.py
```

| file | md5 | rows |
|---|---|---|
| `on_the_floor_A1-f.png` | `6b86ed85138910d02035afacf21834bf` | 5 |
| `on_the_floor_B.png` | `9236ec1a137156051aa73ec2c4381379` | **2** |
| `on_the_floor_C5.png` | `00e5565465092e4b0440af0203e97826` | 5 |

**Regenerated and checked on 2026-09-06 after the visibility correction.** Across all
12 poses, the visibility mask, workpiece rendering and dot projection use the same
final camera. All 196 394 drawn landing centres pass the workpiece ray test, and every
retained landing fits inside the panel with room for the dot radius. The five floor-model
regressions pass. Setup generation and rendering scratch stayed in temporary copies of
the object directories; the three generated setup PNGs matched the originals byte for
byte, and the original setup PNGs were unchanged.

`row3.png` was regenerated with fixed vector magnitude `|F_push| = 0.5 mg` (md5
`047f295fc4bca5a387224f84f78ab3b1`). Its planar line-intersection result agrees with both
the lever rule and `support_polygon.cop` to `1e-12`; the planar construction is retained.

**B went to two rows on 2026-08-28**, and A1-f's and C5's md5s did not move with it. Twice:
it went from one row to five that morning, when `big_tip` padded a short page with repeats
of the tip it had (`top_up`), and to two that evening, when the padding was replaced by a
wider PLACEMENT SOURCE (`big_tip.md`'s `hull_places`) and every row on every page became a
genuinely different tip. The five were the same corner 0.5–3.5° apart and their landing sets
were near-copies, 41.2–44.1 mm, a spread of 7 %. **The two are two different poses of the
bunny** and their landing sets are not copies: 43.8 mm and 68.2 mm, 0.18 and 0.27 part
widths.

**The previous render was checked for determinism:** two runs gave three identical
md5s before the 2026-09-06 visibility correction. The poses, camera angles and work
regions are `big_tip.sweep`'s own — this page calls it and reads the records it hands
back. The floor page then fits its frame to the workpiece and landings.

## The dots sample the cone rim; the boundary also depends on folds and visibility

Row (3) cuts the free body around the **workpiece and its supports**, so the supports'
forces on the workpiece are internal and cancel, and where the load lands on the floor
depends only on gravity and the process push. The formulas below describe that map at
**`K = 0.5`**. This page evaluates 220 contact points and 96 rim directions per point;
it does not compute the complete continuous boundary.

**1 · A lever rule.** With `g = (c_x, c_y, 0)` the plumb point of the centre of mass, and
`h = q − (q_z/d_z) d` the point where the push's *own* line of action crosses the floor,

> **`p = ( 1·g + (−t d_z)·h ) / ( 1 − t d_z )`**

— the centre of pressure is the average of those two points, weighted by their vertical
force components (1 down for gravity, `−t d_z` for the push). `t = 0` gives `g`; a downward
push puts `p` between the two; an upward one puts it on the far side of `g`, and
`t d_z → 1` sends it to infinity, which is the assembly on the point of floating.

**2 · Optional magnitude extension: the region is a fan.** With `r = q − g`,

> `p − g = t (r_z d_h − d_z r_h) / (1 − t d_z)`

and `t ↦ t/(1 − t d_z)` is monotone on `[0, K]`, so a fixed `(q, d)` sweeps the straight
segment from `g` to its full-magnitude end. The union over those magnitudes is
**star-shaped about `g`**; each ray reaches furthest at **`t = K = 0.5`**, which is the
magnitude drawn on this page. Only that endpoint set is required by the current task;
its interior is not filled by an assumed `[0,K]` load range.

**3 · Stereographic projection gives a rational map.** Substitute
`u = d_h/(1 − d_z)`, the projection of the push direction from the north pole, for which
`d_h = 2u/(1+|u|²)` and `d_z = (|u|²−1)/(1+|u|²)`. At full magnitude:

> `p(u) = g + K[(1−|u|²)r_h + 2r_z u] / [(1+K) + (1−K)|u|²]`
>
> **`p(u) = g + [(1−|u|²)r_h + 2r_z u] / [3+|u|²]` at `K = 0.5`.**

The former quadratic expression is the special case **`K = 1`**. A full, unoccluded
cone that excludes the north pole maps to a disc; one containing the pole requires the
exterior chart or a second projection chart. The original formula in `d` is well defined
even at the pole because `1 − K d_z ≥ 0.5`.

**4 · The boundary can come from the rim, a fold, or an occlusion edge.** For a work-region
point above the floor (`r_z > 0`), the determinant at `K = 0.5` is

> `det DΦ(u) = 4r_z [r_z(3−|u|²) − 4r_h·u] / (3+|u|²)³`.

Thus the critical directions satisfy **`d·r = K r_z = 0.5 r_z`**. For the full cone,

> **`∂Φ(C_full) ⊆ Φ(∂C_full) ∪ Φ(C_full ∩ {d : d·r = 0.5 r_z})`.**

The script counts how many sampled contact points have a full cone intersecting that
critical circle. This is a diagnostic before visibility filtering: it does not establish
which parts of the fold remain reachable. The line-of-sight filter may also create
boundaries inside the original cone. Even a zero fold count would therefore not certify
the rim as the entire visible boundary. The `N_PHI = 96` rim angles themselves are
finite samples.

**Rechecked on 2026-09-05 at `K = 0.5`: 0 of 2 640 sampled full cones meet the critical
circle**, over the current 12 poses (5 A1-f, 2 B, 5 C5). Each pose has 0 of 220. The sampled
landings reach **0.18–0.32 part widths** from `g`; no point is dropped by the frame clip.
At that check, all three floor PNGs reproduced the pre-correction hashes. The setup PNGs
were regenerated in a temporary directory, matched their originals, and the originals
were left untouched.

**5 · The region is bounded at half a body weight.** A singular denominator would require
`1 − K d_z = 0`. At `K = 1` that can happen for a push straight up. At **`K < 1` it cannot
happen at all**, since
`d_z ≤ 1` gives `R ≥ 1 − K`: at `K = 0.5` the assembly's total normal force never falls
below half a body weight and the denominator's amplification is at most 2. The current
`support_polygon.exact_R_min` includes the interior maximum `d_z = 1` when a full cone
contains straight up. Its minimum over full cones is a lower bound after visibility
filtering; all current poses have positive total normal reaction.

**6 · Historical area of the magnitude-extended set.** That set is star-shaped about `g`, so **`A = ½∫₀^{2π} ρ(θ)² dθ`** with `ρ` the radial
support of the union — a shoelace sum once the outline is a polygon. **This page does not
compute it**: the two pages that did, the plan view and the region page, are deleted, and
what survives of them is `support_polygon.py`'s `cop` and `exact_R_min` alone.

> **The assembly normal-reaction equations admit some force distribution when every
> required landing is inside the convex hull of its floor contact.** The floor can only
> push, which permits eliminating its forces into a hull test. This is the current
> assembly-level floor check, alongside the workpiece contact equations (1)(2).
> The picture is the argument: the workpiece touches the ground at ONE point — every pose
> here is a `pivot == "point"` pose — a point's convex hull is itself, and the landings are
> all over the floor round it. **The contact is not marked**, so that one point is where the
> part meets its shadow and the sentence has to carry it; the setup page is where it is
> drawn.

### `K`, and what it is

`K` is METHOD §1's declared disturbance magnitude, **in body weights of the workpiece**:
against a push of `K` body weights the contacts must produce `T = up − K·d`.
**`support_polygon.K = 0.5`** is shared by this page. Both `cop(..., t=K)` and
`exact_R_min(..., t=K)` take `t` as the actual magnitude in body weights, with that
default. The removed METHOD and PIPELINE documents used the historical `K = 1` model.

In newtons, `K = 0.5` is **2.7 N** on A1-f (555 g), **3.6 N** on B (735 g) and **0.09 N**
on C5 (17 g) — still several times a real spray gun's ~1 N on the two heavy parts. Which
is the honest caveat about `K` itself: it is *dimensionless*, so one `K` for every object
declares a different physical force for each, and a gun does not know what the part weighs.
At a fixed 1 N a real gun would be `K` = 0.18 on A1-f, 0.14 on B and **5.9 on C5** — a
17 g part fails this project's own premise of *too heavy to lift* long before its supports
do.

### What sampling the cone's interior was costing

The version before this drew 48 directions taken at random from *inside* each cone.
For these historical rows, sampling the rim found more distant landings. These are two
finite-sample measurements, not a comparison against an exact continuous boundary:

| | A1-f 1 | A1-f 2 | A1-f 3 | A1-f 4 | A1-f 5 | B 1 | C5 1 | C5 2 | C5 3 | C5 4 | C5 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| interior sample | 0.79 | 1.39 | 2.47 | 3.07 | 3.10 | 0.48 | 5.41 | 5.52 | 5.60 | 2.20 | 1.97 |
| **the rim** | **0.85** | **1.52** | **3.21** | **3.39** | **3.34** | **0.48** | **5.78** | **7.78** | **6.99** | **2.36** | **2.10** |

(Both rows are at `K = 1` and the old work region.) These historical reaches do not
validate the boundary at `K = 0.5`; the current point cloud samples both contact points
and rim angles, and does not draw the fold or occlusion-edge images.

## Drawing decisions

- **The frame is pulled back until the landings are in it.** The setup page frames on the
  workpiece and the load lands further out than that on every row, so at its own framing the
  dots run off the panel. The camera keeps the row's elevation and azimuth; its look-at
  point and distance are fitted to the workpiece and all retained landings, including
  hidden ones. Visibility is then evaluated using this final camera, also used for
  rendering and projection. This avoids a framing/visibility loop and keeps newly
  revealed dots inside the panel. Landings beyond `CLIP` = 3 box diagonals
  are dropped instead and the count prints: zero on every one of the twelve rows now (it
  was zero on eight of fifteen, at most 49, before the poses last changed).
- **The dots are drawn flat on the finished render, not put in the scene.** 21 120 of them
  is 21 120 geoms and `tip_sequence.render` is handed 64. What that costs is the depth test,
  so it is done by hand and exactly: a dot is drawn only if the ray from its landing point
  to the eye misses the workpiece (`tip_sequence.buried`'s test, vectorised). The part
  hides the ground behind it, as it should. Visibility always uses the final camera;
  the original implementation calculated it before fitting the frame and reused a
  stale mask after the eye moved.
- **The colour is the deleted plan view's own orange**, which is what
  `torque/code/demo_scene.py` already spends on this quantity. It is not the setup page's
  green work region, and it no longer has to be told apart from that page's orange-red
  contact target: `render`'s `draw` is left empty here, so the target is not painted at all.
  Two warm marks on one panel read as one set, and the landings are what this page is for.

## Known issues

- **Nothing on the panel says where the hull would have to reach.** An earlier version drew
  the smallest three feet that contain the landings, and a plan view beside them; both were
  cut as clutter. Three unrestricted vertices can enclose any bounded planar landing
  set. This page draws the assembly's required landings; it does not choose an actual
  footprint or construct support geometry.
- **`big_tip.sweep` is re-run to get the rows**, so this page rewrites `slides/poses`'s own
  three PNGs as a side effect. They come out byte-identical whichever script writes them,
  and that is now checked — it was NOT true for a while: this page rebound `tip_sequence`'s
  shared `TITLES` at import, so the setup pages went out with this page's single column
  head. The rebinding now happens only around this page's own `page` call.
- **Nothing draws the set as a REGION any more.** `coverage.py` did — its outline, its
  convex hull, and the ring of feet that answers it — and it is deleted with the plan view.
  This page is the point cloud and there is no longer a page for the shape or for the
  requirement; `slides/setup/equations/equations.md` carries both in algebra.
- **Contact points and rim directions are sampled:** 220 area-uniform points over the
  stored work region and 96 azimuths at each. The drawn magnitude is `t = K = 0.5`.
- **The fold and visibility boundaries are not drawn.** The diagnostic tests the full
  cone against `d·r = 0.5 r_z`; it does not trace that curve or the boundaries created by
  occlusion. The point cloud is therefore not a complete boundary certificate.
