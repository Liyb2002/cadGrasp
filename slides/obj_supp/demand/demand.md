# Demand and Step 3 contact constraints

[Mechanics, connectivity and insertion](demand_equation.png) ·
[Head total force: B / pose 2](head_total_force.png) ·
[Common head sweep: B / pose 2](head_sweep.png) ·
[Force and moment demand fields](demand_pairs.png)

## Contact-module requirements: mechanics, connectivity, and insertion

2026-09-21: the figure now states four target conditions: (1) the force/moment
pair, (2) no uplift, (3) a connected contact module outside forbidden regions,
and (4) a common insertion direction for that same complete module. Conditions
01–02 share a mechanics block; 03 and 04 are geometric blocks. The original
force/moment and no-uplift equations are unchanged.

This is the intended contact-module formulation, not a claim that the existing
baseline Step3 already constructs the module. Its implementation boundary is
recorded below. The paired demand remains:

\[
\operatorname{demand}(F_{\rm push},\mathrm{pt})=(F_D,\tau_D)\in\mathbb R^6,
\qquad
(F_D,\tau_D)=\left(mg\hat z-F_{\rm push},\;-r_{\rm push}\times F_{\rm push}\right).
\]

`F_push` is the applied process force, `pt` its location in the work region,
`c` the workpiece center of mass, and `r_push = pt - c` the push moment arm.
The `z` axis points upward. The figure shows
`0 <= |F_push| <= 0.5 mg`, as in the reference.

For each task and admissible load, find **one** passive contact reaction field that
simultaneously supplies this R6 demand and satisfies no uplift. The figure
encloses the following conditions in one box to show their shared unknowns:

\[
\left(\sum F_{\rm supp},\;
\sum r_{\rm supp}\times F_{\rm supp}\right)=(F_D,\tau_D),
\]

\[
\sum F_{\rm supp}\cdot\hat z\geq0.
\]

The external demand remains six-dimensional. No uplift restricts the feasible
reaction set; it does not add a seventh independent external demand. A lifted
solver representation is separate from this physical demand definition.

`F_supp` is the passive contact force on the workpiece at each contact, not
one fixed active force applied at every point. The reactions can depend on
the applied load. The complete workpiece balance includes both head contacts
and the workpiece's original contact with the ground. The no-uplift condition sums
**head forces only**, excluding that original workpiece–floor reaction,
because it concerns uplift of the connected support itself. The summation
domains are explained in prose rather than printed below the sum signs.

For one connected, massless, unanchored support, this total vertical force
on the workpiece must be nonnegative. Individual heads may push downward.
The redundant floor-normal equation is omitted from the figure. No uplift
is necessary, not a certificate of full support equilibrium, friction or
resistance to tipping. See [the force illustration](head_total_force.md).
That illustration draws the same forces: the support acting on the
workpiece. Their total vertical component must therefore be nonnegative,
consistent with `F_supp` in the equations above.

## 03: one connected module outside the forbidden regions

Find a solid contact module with a connected material interior and finite-thickness
connections, preserving all selected head solids and their contact regions:

\[
\exists\,V_{\rm support}\ \mathrm{connected},\qquad
A_{\rm obj}\subseteq\partial V_{\rm support},\qquad
V_{\rm support}\cap\mathrm{Forbidden}=\varnothing.
\]

`Forbidden` contains the object interior, all task working regions (and any
specified working clearance volumes), and the below-floor halfspaces of all
fixed task poses. Transform every region into the same object coordinate frame
before taking their union. Surface contact with the object and floor is allowed
where intended; penetration is not. Preserving head solids is an additional
explicit requirement, not implied merely by the surface subset formula.

A zero-width path, or two solids touching only at a point, does not count as a
physical connection. Contact patches each avoiding the forbidden regions is not
sufficient: their heads must admit one shared solid connection in the remaining
space. Surface-only routing is valid only if a near-surface design domain is an
explicit modeling restriction; otherwise the connection may route through 3-D
space. See [the multi-pose floor and connectivity derivation](../../../codes/research_notes/multipose_belt_floor_connectivity.md).

## 04: insert that same complete module

Connectivity and insertion must hold for the **same** geometry. In the chosen
initial installation frame, let d_0 point toward the seated state. A prescribed
finite withdrawal stroke of length L defines the reverse insertion path:

\[
\exists\,d_0:\quad
\mathrm{Sweep}(V_{\rm support},d_0)\cap\operatorname{int}(\mathrm{obj})=\varnothing,
\qquad
\mathrm{Sweep}(V_{\rm support},d_0)\cap\{z<0\}=\varnothing,
\]
\[
\mathrm{Sweep}(V_{\rm support},d_0)
=\{x-t d_0:x\in V_{\rm support},\ 0\leq t\leq L\}.
\]

Here `V_support` includes heads and all their connecting material. Transform it
and the object into the installation frame for these sweep equations. L must
reach the intended pre-insertion state; it cannot be chosen as zero to bypass
installation. The direction d_0 replaces the old symbol a, with the same sign
convention: insertion along d_0, withdrawal along -d_0.

Condition 03 checks every task floor at the final task pose. The floor in 04 is
that of the initial installation scene; it does not require installing the blue
module afresh in every task pose. Initial placement may be chosen to help
installation, as agreed. The orange-base docking directions d_1,...,d_K are
outside this figure's contact-module problem.

## Current baseline implementation boundary

As of 2026-09-21, single-pose Step3 jointly screens common head withdrawal
and a finite-thickness connection witness before candidate scoring and at every
trial size. The witness extends interior roots along a certified head direction,
then joins them with a beam tree beyond the object. Its complete solid stays
above both the initial-rest floor and the target-task floor, expressed in the
same frame. Its withdrawal sweep uses the initial-rest floor, and contact
triangles within 1.5 mm of that floor are excluded. Audit reconstructs the
witness and independently checks its sweep.
Every round still requires pure-gravity feasibility and no uplift, allows partial
working-load coverage, and uses at most three heads.

This is a sufficient construction for one connector family, not a complete
arbitrary-path test. Failure means no witness was found in that family and the
finite direction catalogue. Step3 stores construction parameters. Step5 builds
the blue module and a separate stationary orange base with a rectangular
peg/socket, checks blue installation at rest and whole-object-plus-blue
vertical docking, and verifies object/blue/base equilibrium with shared
unilateral interface reactions. The current socket model has zero nominal
clearance and an end stop; it does not add a withdrawal latch or bolt the base
to the floor. Step6 checks the geometric transport construction and illustrates
installation, combined pickup, and docking in that order. Robot grasping is
assumed feasible; joint trajectories and gripper collisions are not certified. The
multi-pose union of forbidden regions is not yet part of the baseline search:
each pose is solved independently. Working surfaces remain excluded, while
extended process-access volumes remain disabled by the existing policy. The
finite-stroke formula above is the target formulation; the implemented direction
certificates retain their full-withdrawal scope. See the
[implementation](../../baseline_algo/step3_scheculer/README.md).

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
`baseline_algo/output/B/pose2+8/step_1_needs/pose_2/samples.json`. Positions are sampled
by work-surface area, directions in the reachable 30-degree inward cone, and
magnitudes uniformly between zero and 0.5 mg (`seed=20260907`). The saved
six-dimensional demand is checked by direct substitution in the Z-up equations.
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
python slides/tools/render.py --only demand
# Optional complete audit data:
python slides/obj_supp/demand/demand.py --audit /tmp/cadgrasp-demand-pairs.npz
```

The generator writes `demand_pairs.png` and `demand_pairs.json` in this folder;
baseline and setup inputs remain unchanged. The optional audit contains the
complete paired samples, bin assignments, maxima and highlighted IDs. The
current centre of mass is `(0.025843321, 0.080612977, -0.003417037) m`.
