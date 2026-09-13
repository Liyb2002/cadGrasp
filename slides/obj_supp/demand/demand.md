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
(F_D,\tau_D)=\left(mg\hat y-F_{\rm push},\;-(\mathrm{pt}-c)\times F_{\rm push}\right).
\]

`F_push` is the applied process force, `pt` its location in the work region,
`c` the workpiece center of mass, and `y` points upward. The figure shows
`0 <= |F_push| <= 0.5 mg`, as in the reference.

For each covered load, find **one** passive contact reaction field that
simultaneously supplies this R6 demand and satisfies no uplift. The figure
encloses the following conditions in one box to show their shared unknowns:

\[
\left(\sum_{\rm contacts}F_{\rm supp},\;
\sum_{\rm contacts}r_{\rm supp}\times F_{\rm supp}\right)=(F_D,\tau_D),
\]

\[
\sum_{\rm heads}F_{\rm supp}\cdot\hat y\geq0.
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

## Sampling, signs and scales (B / pose 2, 2026-09-13)

Both fields now use the **32,768 paired B/pose_2 samples** already saved in
`baseline_algo/output/B/pose_2/step_1_needs/samples.json`. Positions are sampled
by work-surface area, directions in the reachable 30-degree inward cone, and
magnitudes uniformly between zero and 0.5 mg (`seed=20260907`). The saved
six-dimensional demand is checked by direct substitution in the Y-up equations.
No pose search, temporary object copy or historical tip-1 table is used.

- Force directions are displayed at **-F_D/|F_D|**, preserving the original
  arrival-direction convention. Of 5,120 sphere tiles, 234 contain samples;
  the visual three-round closing paints 236. Closing is a display operation,
  not evidence about additional physical directions.
- Moment directions are **+tau_D/|tau_D|**. Each occupied bin retains the actual
  sample of greatest moment magnitude; 3,836 bins are occupied. The maximum is
  34.967017135 mg mm. Relief height is `0.55 * M / max(M)`, without clipping,
  and rings mark 5, 10, 15, 20, 25 and 30 mg mm.
- Spheres retain their own common directional-space camera so the force cap is
  readable. They represent vector spaces; they do not rotate the physical pose.

## Pair provenance

Matching colour and number refer to the **same load row** on both spheres.
Every highlighted row is the actual winner of its moment bin.

| Number | Sample row (zero-based) | Moment bin | Force / mg | Moment / (mg mm) |
|---|---:|---:|---:|---:|
| 1 | 10069 | 2944 | 1.094621329 | 30.317094330 |
| 2 | 25655 | 2624 | 1.454827592 | 29.992314486 |
| 3 | 17608 | 2152 | 1.371919002 | 14.328746494 |

These finite directional projections do not certify joint contact feasibility
or complete continuous-domain coverage. Empty bins are unsampled, not proved
impossible. The earlier B/tip-1 numerical values no longer describe this figure.

## Reproduce and verify

```sh
python slides/render.py --only demand
# Optional complete audit data:
python slides/obj_supp/demand/demand.py --audit /tmp/cadgrasp-demand-pairs.npz
```

The generator writes `demand_pairs.png` and `demand_pairs.json` in this folder;
baseline and setup inputs remain unchanged. The optional audit contains the
complete paired samples, bin assignments, maxima and highlighted IDs. The
current centre of mass is `(0.025843321, 0.080612977, -0.003417037) m`.
