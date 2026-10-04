# Fixed belt + dock: three-pose interface pressure screen

## Paper-ready wording: edge contact and pressure (2026-10-03)

### Mechanism / motivation

> A clearance-fit docking interface can transmit moments through opposing contact forces concentrated near its edges. If the effective contact area is bounded by \(A_{\mathrm{eff}}\leq \ell b\), the peak contact pressure satisfies \(p_{\max}\geq N/(\ell b)\), where \(N\) is the normal reaction, \(\ell\) is the contact-edge length, and \(b\) is the effective contact-band width. This motivates evaluating local contact pressure rather than only the net interface force.

中文参考：带间隙的插接接口可通过相对接触面的反力传递力矩，载荷可能集中在接头边缘。若有效接触面积满足 \(A_{\mathrm{eff}}\leq\ell b\)，则峰值接触压强满足 \(p_{\max}\geq N/(\ell b)\)，其中 \(N\) 为该接触面的法向反力，\(\ell\) 为接触边缘长度，\(b\) 为有效接触带宽。因此，接口评价应考虑局部接触压强，不能仅依据接口净合力。

Here \(N\) is a face's compressive normal-force sum, not the magnitude of the vector-summed force on the whole dock. Opposing face reactions can cancel in net force while transmitting a moment. The bound follows from \(p_{\max}\geq N/A_{\mathrm{eff}}\); it does not assume uniform actual pressure.

### Units and normalization

\(mg\) denotes object weight, a force measured in N. Pressure is force divided by area and should be reported in Pa or MPa. For scale-normalized comparisons, define

\[
A_{\mathrm{ref}}=0.01A_{\mathrm{obj}},\qquad
p_{\mathrm{ref}}=\frac{mg}{A_{\mathrm{ref}}},\qquad
\hat p=\frac{p}{p_{\mathrm{ref}}}=\frac{pA_{\mathrm{ref}}}{mg},
\]

where \(A_{\mathrm{obj}}\) is the total object surface area. \(\hat p\) is dimensionless. Use the figure-axis label **Normalized contact pressure, \(pA_{\mathrm{ref}}/(mg)\)**; never label a pressure as “\(Xmg\).” A force may instead be labeled \(N/(mg)\), also dimensionless, with the quantity explicitly identified as force.

Suggested methods sentence:

> We normalize contact pressure by \(p_{\mathrm{ref}}=mg/A_{\mathrm{ref}}\), where \(A_{\mathrm{ref}}\) is 1% of the total object surface area, and report the dimensionless quantity \(pA_{\mathrm{ref}}/(mg)\).

### Conditional numerical result

> For an assumed effective contact-band width of 0.1 mm, the mean and maximum pressure lower bounds over 100 matched loads are respectively 4.40 and 3.86 times the corresponding nominal pressure demands of the passive support.

中文参考：在假设有效接触带宽为 0.1 mm 时，100 个相同载荷下，dock 的压强下界的平均值与最大值分别为被动支撑对应名义压强需求的 4.40 倍和 3.86 倍。

These are ratios of the respective means and maxima, not the mean and maximum of per-load ratios. “Maximum” refers only to the 100 sampled loads. The comparison uses the existing Pose 3 inputs, object-ground load sharing, mu=64 ground model and shared no-uplift condition. The dock bound uses the unlocked frictionless sleeve, relaxed cavity-face force locations, a full-area end stop and assumed side-contact area caps; the support quantity is its optimized nominal peak-pressure demand. This compares different specified contact-area models, not measured elastic peak pressures.

Suggested limitation sentence:

> These conditional bounds illustrate potential pressure concentration at the interface; they do not establish the actual contact-band width or material failure.

Keep **assumed** and **lower bounds** in any numerical claim. The 0.1 mm band width is a scenario parameter, not a measured value or a value inferred from the 0.5 mm assembly clearance. Actual width and pressure require a specified contact/deformation model or measurement. These results motivate examining the interface and retaining belt + dock as a comparison; they do not establish that every dock is inferior or that the present dock necessarily fails. The ideal locked, full-face model below gives lower nominal pressure demand than the support and must not be conflated with this edge-contact scenario.

Numerical provenance: [conditional-bound implementation](clearance_contact_bound.py), with recorded results at `pressure_data/clearance_pressure_bound.json`; matched nominal comparison: [implementation](compare_contact_pressure.py), with results at `pressure_data/pressure_comparison_100.json`. This section records paper wording from existing results; no calculation was rerun for this documentation edit.

## Final presentation and retained evidence

![Shared-base scene and same-view 3D joint detail](../dock_pressure_zoom.png)

The left panel is Dock 1 from the pre-existing `shared_base.png`. The right panel
is a fresh geometry render from `original_geometry.npz`, using the same pose and
camera direction (`[1.05, -1.25, 0.8]`) with a tighter camera framing. There are no
red markers, arrows, or explanatory text in the image. It is a geometry
illustration, not a peak-pressure map. Reproduce with:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python slides/belt_test/code/draw_dock_crop_zoom.py
```

At the user’s request, all earlier diagnostic PNGs created in this discussion
were deleted. Pre-existing illustrations and videos were preserved. The scripts
and JSON data below remain as numerical evidence; their historical render
functions must not be used to restore removed figures without a new request.

The competing fixed-belt design remains viable in principle. Its practical
challenge is an interface that inserts easily, then provides stiffness,
positioning and moment transmission. The nominal locked/full-face calculation
and the clearance/limited-area lower bound answer different conditional
questions. Neither proves material failure or that every dock is worse than a
passive support. No contact-force capacity limit was added to the baseline.

## Clearance and edge contact: conditional pressure lower bounds

The earlier diagnostic plot was removed at the user’s request; numeric records remain below.
Entry: `clearance_contact_bound.py`; records: `pressure_data/clearance_pressure_bound.json`.
This uses the same 100 Pose 3 loads and ground model as the matched comparison,
but the existing unlocked sleeve and an explicitly assumed effective side-contact
band width b. It does **not** infer b from the 0.5 mm assembly clearance.

For each wall, equilibrium supplies its compressive force N. If its actual loaded
area is at most A = wall edge length × b, then peak pressure must satisfy
`p_peak >= N / A`. The LP minimizes this necessary pressure capacity over generous
cavity-corner force locations, with a full-area end stop. Its result is a lower
bound under those area assumptions, not a solved elastic contact distribution.

Relative to the refined baseline's nominal minimum peak pressure, the average
over these 100 loads and the maximum over these 100 loads respectively are:

| Assumed side-contact band width | Average lower-bound ratio | Worst lower-bound ratio |
| --- | ---: | ---: |
| 0.1 mm | 4.40× | 3.86× |
| 0.25 mm | 1.76× | 1.54× |
| 0.5 mm | 0.88× | 0.77× |
| 1.0 mm | 0.44× | 0.39× |

In 19/100 loads, all four relaxed combinations that permit only one X wall and
one Y wall are infeasible. Opposing wall reactions are needed within this fixed
pose, frictionless sleeve model. Clearance can turn the associated contact into
edge contact; the illustration is schematic, not a solved contact configuration.
The actual contact width requires contact/deformation analysis or measurement.
The conditional results support the pressure-concentration mechanism, but do not
establish the real width, material failure, or superiority over every dock design.

## Matched nominal pressure calculation: the requested claim is not established

The user requests numerical evidence that concentrating load at the small dock
causes high interface pressure. `compare_contact_pressure.py` now compares
**pressure rather than resultant force**, using exactly the same 100 Pose 3
loads, object geometry, COM and ground point, mu=64 and shared no-uplift condition.
It does not add a pressure optimization or capacity limit to the baseline pipeline.

For each load, each baseline surface triangle or dock-face tile carries a
constant nonnegative frictionless pressure. Force equals pressure × tile area ×
normal, and its moment uses the tile centroid. An LP minimizes the common
maximum tile pressure required for equilibrium; a secondary solve minimizes
unnecessary total compression at that pressure cap. Baseline contacts use their
actual saved areas, each approximately 1% of object surface. The dock has its
18×8 mm peg, 21 mm engagement and a hypothetical full-area retaining face, with
ideal snug contact. All values are normalized by **mg / (1% object surface area)**,
and are dimensionless pressure ratios, **not mg forces**.

| Minimum maximum nominal pressure demand | Average of 100 | Worst of 100 |
|---|---:|---:|
| Three-head baseline support | 18.24 × reference pressure | 74.77 × reference pressure |
| Ideal locked dock | 1.86 × reference pressure | 7.05 × reference pressure |


In this model the dock's average is **0.102 times** baseline's average, and its
worst is **0.094 times** baseline's worst. Dock demand is higher in only **3/100**
draws. **These results do not support the requested conclusion that this dock
necessarily has much higher pressure.** A small interface can transmit moment
with opposing face pressures, but pressure also depends on available face areas,
their directions, ground load sharing and how efficiently the contacts balance
the wrench. The baseline's curved unilateral contacts do not necessarily use
their full nominal patch area efficiently under each load.

Convergence checks subdivide every baseline triangle into four and double the
dock grid to 16×8 on side faces / 8×8 on end faces. Baseline coarse average/worst
is 18.53/76.99 reference units versus refined 18.24/74.77; dock coarse is
1.89/7.16 versus refined 1.86/7.05. Every solve replays force/moment equilibrium.
Numeric tile pressures, load IDs, areas and source hashes are in
`pressure_data/pressure_comparison_100.json`.

This is an optimistic rigid traction-capacity calculation, not the actual
contact pressure of the existing **0.5 mm-clearance, unlocked** connector.
Real edge/corner contact, deformation, preload, latch geometry and stiffness
are unspecified. Smaller actual contact area can produce larger local pressure,
but selecting an arbitrary tiny area to force the desired conclusion would not
be evidence. The matched belt placement and complete base equilibrium are also
not geometrically/mechanically certified by this diagnostic. Neither the dock
nor baseline results are actual elastic peak-pressure predictions or material
failure certificates. High pressure cannot presently be used as a demonstrated
reason to reject this competing baseline.

```sh
OPENBLAS_NUM_THREADS=1 python slides/belt_test/code/compare_contact_pressure.py
```

## Interpretation correction: small dock resultant is not low interface pressure

The user questions whether the 100-load chart establishes that a dock is less
loaded than a passive support. It does **not**. Net resultant and local contact
pressure are different quantities, and individual baseline heads were being
shown beside a whole-dock aggregate. The latter cancels opposing mating-face
forces while still transmitting a moment. The table is a force summary under
specified LP allocations, not evidence of lower pressure, lower required strength
or a superior interface. Net force alone omits the wrench's moment block.

For a concrete simultaneous example from the existing 100-draw data, historical
Pose 2 sample 92 has a whole-dock resultant of **0.12mg**, but mating-face normal
forces sum to **4.15mg**. Historical Pose 4 sample 6 has a whole-dock resultant of
**0.56mg** while normal forces sum to **7.55mg**. Nothing was changed in these
calculations: opposing contacts carrying torque explain the difference. The sum
of normal forces is also not pressure; pressure needs each face's area and actual
distribution. The same-100 dock's largest normal-force sum is **2.59mg**, with
a simultaneous net resultant of **0.40mg**.

If comparing aggregate net forces, baseline must also sum all three head forces
as vectors before taking a magnitude. Its whole-support average/worst in these
100 selected LP allocations is **0.45/2.01mg**, versus matched dock **0.16/0.41mg**.
That model result still does not compare local pressure or strength. Different
contact geometries and the ability of a locked interface to transmit a couple
change the admissible reaction allocation. `statistics_100_force_diagnostic.json`
records both aggregate net forces and internal normal-force sums for inspection.
No conclusion that the dock is mechanically more economical is justified by the
published resultant chart alone.

## Current 100-load average / worst statistics

`sample_force_statistics.py` implements the latest user request with **100**
loads, seed 20261002. Every reported quantity is the magnitude of the
**vector-summed force received by the whole head or whole dock**, in mg.
Average is the arithmetic mean of the 100 magnitudes; worst is the largest
magnitude among these draws, not a continuous-domain maximum. A component-wise
average vector is not used, because opposite directions would hide the load.

| Contact group | Average | Worst of 100 |
|---|---:|---:|
| Baseline Pose 3 / C041 | 0.470mg | 2.277mg |
| Baseline Pose 3 / C137 | 0.677mg | 2.332mg |
| Baseline Pose 3 / C049 | 0.436mg | 1.407mg |
| Dock on the **same Pose 3 and exact same 100 loads** | 0.157mg | 0.411mg |
| Historical dock / Pose 2 | 0.237mg | 0.500mg |
| Historical dock / tilted pose | 0.223mg | 0.664mg |
| Historical dock / Pose 4 | 0.305mg | 0.637mg |


All seven contact-group rows come from feasible sets of 100 loads. Baseline
draws 100 original samples without replacement, conditioned on negative vertical
tool-force component; original points, directions and magnitudes are retained.
The **same-100 dock** rigidly registers the pinned belt/interface onto the archived
baseline object pose, with matching original vertices, area, volume and COM;
it receives exactly the same points/forces and exact archived ground-contact point.
The three historical dock rows sample their own work patches independently with
uniform area, solid angle within the inward 30-degree cone, and magnitude in
[0,0.5]mg, conditioned on a negative vertical component. These newly generated
historical illustrative-task loads are not screened for tool-ray self-occlusion.
They therefore do not constitute identical-input comparisons with baseline.

**Object-ground contact is included in both designs**, using the same mu=64
four-ray pyramid and shared no-uplift condition. In each model the same LP
minimizes the largest contact-group normal-force sum, after which group net
forces are vector-summed and their magnitudes measured. Baseline has three contact
groups; the dock is one group. Historical dock ground locations use the mean of
the lowest mesh vertices as a single contact point. Matched baseline/dock share
the actual archived point. Force and moment residuals are checked for every solve.
The allocations are feasible LP optima, not unique elastic-contact predictions.

The main dock rows assume a massless belt rigidly fixed to the object and an ideal
zero-clearance locked interface with a hypothetical opposing retainer. No strength
limit is introduced. They do not certify the actual loose, unlocked connector,
the dock/base's full equilibrium, or manufacturing stiffness. A separate matched
unlocked diagnostic also passed these 100 conditioned downward loads in this
ideal contact model; this is not a general anti-withdrawal guarantee.

These statistics supersede the earlier displayed dock resultants of about 1.5mg,
which deliberately omitted object-ground load sharing. Lower whole-dock net force
does not imply lower internal mating-face pressure or moment. Numeric samples,
simultaneous vector reactions, feasibility status and input hashes are in
`pressure_data/statistics_100.json`. Existing baseline inputs, geometry and
acceptance status remain unchanged. Reproduce with:

```sh
OPENBLAS_NUM_THREADS=1 python slides/belt_test/code/sample_force_statistics.py
```

## Random simultaneous force examples

For the user's force-balance inspection, `random_load_examples.py` draws four
original Pose 3 loads sampled with seed 20261002 from the subset whose tool force
has a negative vertical component. These are not strictly vertical forces:
sideways components remain. Magnitudes and points are used unchanged, with no
0.5mg endpoint rescaling. Each panel contains one load and its simultaneous head,
ground and gravity reactions; six-dimensional equilibrium is checked directly.
All arrows are forces on the object; the forces received by the heads reverse
direction but have the same magnitude. The vectors and moment witnesses are in
`pressure_data/random_loads.json`, allowing the user to redraw/check them.


Sample 3359, with a 0.44mg tool force, has head resultants approximately
0.17/0.17/0.11mg and a 1.39mg ground resultant. It illustrates the small head forces
the user expected. Other sampled pushes have different directions/lever arms
and produce larger forces. The earlier per-head maxima should never be presented
as a simultaneous reaction set for an arbitrary downward push.

## Historical removed force figure: resultant force, not pressure

The user clarifies that the desired label is **the magnitude of the net force
received by each whole head**: vector-sum all contact forces on that head, then
take the norm. The force on the head is the negative of the force on the object;
their magnitudes agree. Do not sum force magnitudes or normalize pressure by an
arbitrary reference area for this label. The historical head-resultant diagnostic used
this quantity exclusively. The previous 39/35/58mg dock labels were equivalent
pressure quantities and must not be interpreted as actual dock resultant forces.

For Pose 3, re-evaluating the same original loads and exact-0.5mg endpoint loads
in the chosen optimized equilibrium allocations gives maximum head resultants
**C041: 3.913mg; C137: 3.887mg; C049: 1.794mg**. These are individual maxima over
different loads, not one simultaneous set, measurements, or unique elastic
reactions. The allocation minimizes the largest head normal-force sum; it does
not minimize the largest resultant. In this example, vector cancellation within
the individual curved heads is modest; substantial opposing forces occur between
different heads and the ground. A smaller resultant such as 0.2mg is not supported
by these calculated worst cases.

The dock resultant is `norm(gravity + tool_force)`, giving about 1.5mg at the
largest of the checked loads in each historical pose. This is the net force on
the **whole dock**, not forces on individual mating faces and not the transmitted
moment. The baseline entries are forces on individual heads. Consequently the
chart is a force summary, not a matched interface-strength or pressure comparison.

The historical pressure computation and normal-force sums below are retained as
numeric history, separate from the current resultant-force labels. New resultant
fields and force-vector witnesses are in `pressure_data/comparison_mg.json`;
`pressure_data/pose3_render.json` records the plotted values and quantity.

## Weight-based comparison (2026-10-02)

The user requests pressures expressed relative to object weight rather than SI
pressure units. Define the common reference area as **1% of the object's total
surface area**, and reference pressure as `mg / reference_area`. A pressure ratio
X then means “the same nominal pressure as X object weights pressing on this
1%-surface patch.” It does **not** mean the dock's net force is Xmg.

The dock resultant satisfies `|gravity + machining force| <= 1.5mg`. Opposing
contact forces inside the small interface can nevertheless be much larger to
transmit the moment. The prior hypothetical-retainer pressure results correspond
to approximately **39mg, 35mg and 58mg acting over the common reference area**,
for Pose 2, its tilted variant and Pose 4 respectively.

Read the requested historical `baseline_algo/output/B/pose1+3` source contacts,
loads and archived setup, preserving all original files. Assume each physical
head has the common 1%-surface area for nominal pressure normalization; do not
resize the saved geometric patches or substitute current same-numbered poses.
Retain the original frictionless head normals, object-floor friction model
(mu=64) and shared no-uplift condition. For each load minimize the largest head's
sum of compressive normal forces. Under equal assumed areas, this minimizes the
largest head-mean pressure. Report each head's largest force in the returned
optimal allocations. Force allocation is not unique and this is not a prediction
of elastic force sharing; maximizing a head's force over unrestricted equilibrium
solutions can be unbounded, so that is not the useful “maximum” being reported.

| Baseline pose | Physical contact ID | Maximum normal-force sum in optimized allocations |
|---|---|---:|
| Pose 1 | C041 | 28.4mg |
| Pose 1 | C163 | 25.1mg |
| Pose 1 | C084 | 4.7mg |
| Pose 3 | C041 (separate physical patch) | 4.0mg |
| Pose 3 | C137 | 4.0mg |
| Pose 3 | C049 | 1.8mg |

Each table entry divided by the common reference area is the corresponding
head-mean pressure. It is not the maximum pressure at a point. All 32,768 original
loads per pose, their 32,768 exact-0.5mg endpoints along the same sampled push
directions, and pure gravity are checked; all are feasible in this model. Original
random loads have magnitudes below 0.5mg, so the endpoint check answers the user's
maximum-magnitude question without claiming a continuous global maximum. The
largest optimized common-head capacity is 28.4mg for Pose 1 and 4.0mg for Pose 3.
The head force/moment residual and no-uplift condition are replayed for every solve.


In the deleted historical pressure chart, the orange bars were the dock's **maximum tile-average nominal pressure demand**,
renormalized to the reference area. The blue bars were baseline **head-mean nominal
pressure demands** under the assumed head area. Ground shares baseline loads,
but is omitted in the dock calculation; the compared poses also differ. This
figure describes these two calculations, not a matched-system performance test
or a proof that one design has lower true peak pressure. Large internal force
amplification occurs in both schemes, and cannot establish material failure by
itself. Code: `load_comparison_mg.py`; numerical results and input hashes:
`pressure_data/comparison_mg.json`. Run with NumPy, SciPy, trimesh, highspy and
matplotlib. All baseline Step3/Step4 artifacts remain untouched.

The proposed workflow rigidly attaches the belt to the object. One robot grasps
the belt, moves the assembly into each dock, releases it for the task, then
retrieves the assembly. This differs from the historical separate-loading video.

The earlier low-load Pose 3 force sheet has been removed. Its calculation records remain available; current presentation is the unannotated dock close-up above.

## Inputs and results

Use the pinned object, blue module, socket and peg in `original_geometry.npz`,
and the three historical transforms/work areas in `geometry_report.json`.
The middle pose is the existing illustrative 25-degree tilt, not a new task.
The object mass is **0.735235 kg**, from the historical mesh at the assumed density
of **1000 kg/m³**, not a physical weighing. With g = 9.81 m/s², mg = **7.21266 N**
and the maximum machining force magnitude 0.5mg is **3.60633 N**.

The existing peg is 18×8×26 mm, with 21 mm engagement and 0.5 mm side clearance.
It has an insertion end stop but no anti-withdrawal lock. Side-face nominal areas
are 8×21 = 168 mm² and 18×21 = 378 mm²; the end face is 18×8 = 144 mm².

| Historical pose | Nominal pressure demand with ideal retainer | Finer grid, same selected load | Interface moment for selected load | Largest outward axial demand | Unlocked infeasible loads |
|---|---:|---:|---:|---:|---:|
| Pose 2 | 0.501 MPa | 0.494 MPa | 1.275 N·m | 1.023 N | 109 / 1414 |
| Pose 2 + 25° tilt | 0.455 MPa | 0.448 MPa | 1.190 N·m | 1.223 N | 69 / 1146 |
| Pose 4 | 0.746 MPa | 0.738 MPa | 1.324 N·m | 0 N | 0 / 1380 |

These are sampled results in the model below, not globally maximum pressures
or observed material failures. Zero infeasible samples is not a complete design
certificate. Failure counts describe this deterministic sample set, not a physical
failure probability.

## What is computed

Assume the object is rigidly fixed to a massless belt, the base/dock is fixed,
and **all object gravity and machining load pass through the dock**. Object-ground
load sharing is omitted. This is a defined dock-only load case, not a prediction
of the displayed floor-contact system; adding ground reactions requires jointly
solving them and can change both forces and moments.

For force F at work point q, with object COM c and interface origin o:

```text
external force  = F + (0, 0, -mg)
external moment = (q-o) × F + (c-o) × (0, 0, -mg)
dock wrench     = negative of the above, expressed in interface axes
```

The gravity lever arm matters: pressure is not simply 0.5mg divided by one face
area. The interface carries the combined gravity and machining wrench.

Every work-face centroid receives a normal inward 0.5mg force. In addition,
1024 seeded area-weighted positions and inward 30-degree cone directions are
sampled per pose, with half the extra directions on the cone boundary. All pushes
have magnitude 0.5mg. Tool-ray occlusion, finite-tool access and robot reachability
are not screened, so these are geometric candidate loads rather than certified
reachable processing loads. Existing baseline loads/results are not changed.

The contact model uses frictionless, **ideal zero-clearance** bearing faces,
subdivided into uniform-pressure tiles. A side face has 8 longitudinal × 4
transverse tiles; axial faces have 4×4 tiles. Each tile's resultant acts at its
centroid. For each load, solve six-dimensional force/moment equilibrium with
nonnegative tile pressures and minimize the largest tile-average pressure.
Then take the largest optimum over sampled loads. The selected load is replayed
with a 16×8 / 8×8 refined grid; the change is below 2% for these cases. A secondary
minimum-total-normal-force solve selects the displayed face-mean distribution
among equally peak-optimal solutions. The face means in the bars are distinct
from the largest tile-average demand printed above each plot.

The **unlocked model** has four side faces and the existing end stop. The
**hypothetical locked model** adds a full-area opposing retainer, which is absent
from the apparatus and not a designed latch. Its pressure values are an optimistic
nominal screen of a modified interface, not the actual existing joint pressure.
The model does not predict clearance closure, preload, elastic compatibility,
edge concentration, wall bending, peg stress, wear or positioning errors. Actual
contact pressures require a specified contact/compliance and locking design.

## What this establishes, and why it is not the main research route

In the frictionless dock-only model, some Pose 2 and tilted-pose loads pull along
the withdrawal axis. Side walls cannot provide axial reaction, and the end stop
pushes in the opposite direction, so those loads have no static equilibrium.
This establishes an anti-withdrawal requirement under the stated assumptions;
friction, object-ground load sharing or a real latch could change that outcome.

**The calculation does not establish “pressure too high, cannot succeed.”**
The nominal demand is about 0.45–0.75 MPa in the hypothetical retained model.
No allowable contact pressure, material, print orientation, safety factor or
actual contact area has been supplied. Comparison to a material limit and a
real deformation/contact analysis are necessary before making a strength claim.

The reason to keep this as a competing baseline rather than the main method is
the research scope: it concentrates the solution in a fixed-on-object clamp,
precision docking/locking interface and multiple task-specific docks. Our current
main question concerns a reusable rigid passive support and geometry shared
across object contacts and ground contacts. A fixed belt may offer better task
switching and repeatable registration; this advantage must be acknowledged and
compared against total tooling, footprint, execution time and precision. The
easy-insertion versus low-play requirement motivates interface design, not a
claim that the alternative is impossible or only a materials problem.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 python slides/belt_test/code/interface_pressure.py
```

Dependencies: NumPy, SciPy, trimesh, matplotlib, shapely. The script reads pinned
historical assets directly, avoiding stale imports of deleted baseline stages.
The shared ground ring/posts in the drawing reproduce `shared_base.py` geometry
from the saved pose transforms, without rerunning a fixture search or Boolean
construction. Numeric report, selected force points/vectors, tile pressures,
model dimensions and source SHA-256 hashes are in `pressure_data/report.json`.
The former Pose 3 force PNG has been deleted. Diagnostic scripts retain historical plotting code; do not run their plotting entries to restore deleted figures. Historical
meshes, images, geometry reports and baseline results remain unchanged.
