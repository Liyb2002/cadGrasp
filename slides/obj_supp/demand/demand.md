# Demand and Step 3 contact constraints

[Joint mechanics and insertion](demand_equation.png) ·
[Head total force: B / pose 2](head_total_force.png) ·
[Common head sweep: B / pose 2](head_sweep.png) ·
[Force and moment demand fields](demand_pairs.png)

## Paired demand, no uplift, and common insertion

The English figure groups **force demand**, **moment demand**, and **no uplift**
into one mechanics block (01–03). Common insertion remains a separate geometric
block (04). The force and moment form one paired demand in R6, retaining the
notation and unbroken equations from [the supplied reference](pasted-movie.png):

\[
\operatorname{demand}(F_{\rm push},\mathrm{pt})=(F_D,\tau_D)\in\mathbb R^6,
\qquad
(F_D,\tau_D)=\left(mg\hat z-F_{\rm push},\;-(\mathrm{pt}-c)\times F_{\rm push}\right).
\]

`F_push` is the applied process force, `pt` its location in the work region,
`c` the workpiece center of mass, and `z` points upward. The figure shows
`0 <= |F_push| <= 0.5 mg`, as in the reference.

For each covered load, find **one** passive contact reaction field that
simultaneously supplies this R6 demand and satisfies no uplift. The figure
encloses the following conditions in one box to show their shared unknowns:

\[
\left(\sum_{\rm contacts}F_{\rm supp},\;
\sum_{\rm contacts}r_{\rm supp}\times F_{\rm supp}\right)=(F_D,\tau_D),
\]

\[
\sum_{\rm heads}F_{\rm supp}\cdot\hat z\geq0.
\]

The external demand remains six-dimensional. No uplift restricts the feasible
reaction set; it does not add a seventh independent external demand. A lifted
solver representation is separate from this physical demand definition.

`F_supp` is the passive contact force on the workpiece at each contact, not
one fixed active force applied at every point. The reactions can depend on
the applied load. The complete workpiece balance includes both head contacts
and the workpiece's original contact with the ground. The no-uplift condition sums
**head forces only**, excluding that original workpiece–floor reaction,
because it concerns uplift of the connected support itself.

For one connected, massless, unanchored support, this total vertical force
on the workpiece must be nonnegative. Individual heads may push downward.
The redundant floor-normal equation is omitted from the figure. No uplift
is necessary, not a certificate of full support equilibrium, friction or
resistance to tipping. See [the force illustration](head_total_force.md).
That illustration draws the opposite forces: the workpiece acting on the
support. Their total vertical component must therefore be nonpositive;
the definition of `F_supp` in the equations above remains unchanged.

The fourth condition is independent of the applied load and holds once for
the selected design. With `a` denoting insertion and `-a` withdrawal,

\[
\exists\text{ an allowed unit insertion direction }a:\quad
\mathrm{Sweep}(\mathrm{supp},a)\cap\operatorname{int}(\mathrm{obj})=\varnothing,\qquad
\mathrm{Sweep}(\mathrm{supp},a)\cap\{z<0\}=\varnothing,
\]
\[
\mathrm{Sweep}(\mathrm{supp},a)=\{x-ta:x\in\mathrm{supp},\ t\geq0\}.
\]

`supp` is the union of selected head solids and `obj` is the workpiece.
`Sweep(supp,a)` is the space occupied as the heads withdraw along `-a`.
The notation matches [the original sweep figure](../../trajectory/sweep_eq.png);
the additional abbreviations `C(a)` and `H` are omitted.
The direction is chosen from the finite catalogue after the work-side and floor
filters. Step 3 certifies a common head sweep; exhaustion of this catalogue
alone does not prove all directions impossible. Surface contact is allowed,
interior penetration is not. The opposed-head illustration includes its own
stronger local-direction check in [head_sweep.md](head_sweep.md).

Every greedy round requires gravity-only equilibrium with no uplift and a
common head direction. Working-load coverage may be partial and grows as
heads are selected; the schedule uses at most three heads. Step 5 constructs
and checks the frame, base, connectors and complete assembly trajectory.

Regenerate the mathematics:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/obj_supp/demand/demand_equation.py
```

The demand fields below retain the same paired external force and moment
data. They are not a plot of the new support constraints or their coverage.

## The referenced drawing, and what is reused

The visual reference was the former pipeline step 2 (removed in the root cleanup).
Its **COVER force** is binary paint
on a sphere. Its **COVER moment** is a stepped radial relief: each triangular
face rises to `1+h`, adjacent heights are joined by side walls, and dashed rings
provide a ruler. Both are rendered by
[`cover.globe` / `cover.relief_shell`](../../tools/cover.py), which this page
calls directly.

Only that visual mechanism is reused. Step 2's fields describe capped *supply*
mixtures and must not be substituted for demand. Here red means **sampled demand**.
The former line-frame spheres and internal endpoint clouds have been replaced.
The former step-1 demand picture used
the same red language but a nonlinear `L/2` height map; this page uses step 2's
linear height mechanism instead.

## Sampling, signs and scales

Both fields come from **547,488 paired process samples** on current setup **B,
target pose 1**. The shared sampler requests 1,440 area-weighted work-region
points × 384 directions (`seed=1000`), within the 15-degree local inward-normal
cone; obstructed incoming rays are rejected. This is the existing dense drawing
rung, applied to the current setup pose at `K=0.5`. The 2,141-row load table
[`area/demand_B_tip1.npz`](../area/demand_B_tip1.npz) is read only to check that
the pose and centre of mass agree. Its load rows were retained from the former
`invoices/` page when the area demos became self-contained.

- **Force coverage:** each `F_D/|F_D|` is displayed at its antipode, **`-F_D/|F_D|`**,
  following the pipeline's “where the force comes from” convention. This display
  sign does not change the equation. Red is the occupied demand-direction domain,
  not the directions a support can provide or pass. On the 5,120-tile icosphere,
  196 bins are occupied. The existing three-round sheet closing is used; on this
  dense sample it changes no bins. All force magnitudes remain in the paired
  data; highlighted magnitudes are printed. The full range is
  `1.165401–1.500000 mg`.
- **Moment relief:** bins are indexed by **`+tau_D/|tau_D|`**, with no sign reversal.
  Each bin stores the largest **actual sampled** `|tau_D|` assigned to it, in
  `mg mm`. There are 3,694 occupied bins. Unsampled bins remain bare at radius 1;
  bare is not a proof that the continuous domain contains no such direction.
  The measured largest bin value is `26.280395 mg mm`. The map is exactly
  `h = 0.55 M / 26.280395`, with the full-precision measured maximum used in
  code, and radius `r = 1+h`. No values are clipped. Ring heights use the same
  map at `5, 10, 15, 20, 25 mg mm`. This is a drawing scale, not a force cap or
  an `L/2` physical bound.

Both balls use the same world axes and camera: azimuth −62° from the original
renderer, elevation −20° so the southern force patch and its labels are visible.
The renderer's light is recomputed for that shared view. No sample is moved to
separate labels.

## Pair provenance

Three complete rows are highlighted with matching colour and number. There are
no connectors between the force sphere and the moment sphere.
**Each is the real winning sample of its own moment bin.** The force partner is
read from that same row; a different sample's bin maximum is never attached to it.
Moment markers use the original sample's axis and its actual height, projected as
annotations over the relief. The numbers below the balls are
`(|F_D|, |tau_D|)`, not replacements for the stored vectors.

| Number | Dense sample row, zero-based | Moment bin | `|F_D| / mg` | `|tau_D| / (mg mm)` |
|---|---:|---:|---:|---:|
| 1 | 193790 | 2812 | 1.324598150 | 2.916676727 |
| 2 | 185836 | 4108 | 1.470938968 | 19.202837561 |
| 3 | 188428 | 3400 | 1.312589624 | 20.909362924 |

This is a finite sampled domain and a per-direction-bin maximum. At fixed `K`,
it does **not** assert that all moments from zero to that maximum are available.
Separate displays also do not test joint force–moment feasibility; the underlying
paired rows and shared contact reactions still matter. No contact-count or
coverage-solver result is shown.

## Reproduce and verify

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/obj_supp/demand/demand.py
```

Only `demand_pairs.png` is written. Setup pose/region selection and its intermediate
meshes, XML and renders run with object copies in a temporary directory. Importing
the module generates nothing. Other figures, source files and area load tables are
unchanged. For a complete external audit table, append
`--audit /tmp/cadgrasp-demand-relief-pairs.npz`.

The script checks paired equations, unit process directions, moment perpendicularity,
agreement with the original invoice centre of mass, bin-winner provenance and
linear height scaling. The optional audit records all `(pt,d,F_D,tau_D)` rows,
bin assignments, maxima, winners and final highlighted IDs. The current centre
of mass is `(0.013601859, -0.066934468, 0.061937985) m`.
