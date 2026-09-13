# The four rows, written out

**2026-09-13 rendering:** The current PNG uses world Z-up (`z=0` at the floor), matching B/pose_2 and the shared renderer. The derivation, saved arrays and renderer all use the same native Z-up convention.

**Current declarations (2026-09-08): [problem_statement.md](../../problem_statement.md#当前决定与讨论记录).**
The local push cone has half-angle `30°`, and the magnitude interval is `0 ≤ t ≤ K=0.5`,
including gravity alone. The robot holds the target pose while supports are inserted
one at a time and reach perfect contact before it releases the workpiece.

The equations below retain the agreed aggregate floor model. They certify fixed contact
reactions and aggregate floor balance, not the equilibrium and installation of separate
support solids. [Step 3's verification](../../baseline_algo/step3_scheculer/README.md)
now contains explicit uplift and horizontal insertion counterexamples. Historical figures
and numerical records retain their original parameters unless regenerated.

One notation for all of them, so the coupling is visible. Everything is at the **target
pose** `T*`; forces are in **body weights** of the workpiece, so gravity is exactly `−ẑ`.

| symbol | what it is |
|---|---|
| `ẑ` | up. Gravity is `−ẑ` at the centre of mass, magnitude 1 |
| `c` | the centre of mass at `T*`;  `g = (c_x, c_y, 0)` its plumb point on the floor |
| `a` | the workpiece's own ground contact — **one point**, the pose is a tip |
| `W` | the work region on the surface, `n(pt)` its outward normal |
| `C(pt)` | the admissible push directions at `pt ∈ W` (below) |
| `K` | the process disturbance, in body weights. **`K = 0.5`** |
| `pᵢ, uᵢ` | contact `i` is at `pᵢ` and pushes along `uᵢ` (unit, into the part); a rigid support may carry several contacts |
| `λᵢ ≥ 0` | what that contact delivers. **Unbounded** — a contact pushes and never pulls, and nothing caps it |
| `F` | every point the assembly puts on the FLOOR: the supports' feet, and `a` |

**The push.** The process acts at one point `pt ∈ W`, along `d ∈ C(pt)`, with magnitude
`0 ≤ t ≤ K = 0.5`:

```
C(pt)  =  { d : |d| = 1, d·(−n(pt)) ≥ cos α }  ∩  { d : the ray pt − s d, s > 0, is clear of the part }
α = 30°
```

— a cone about the inward normal, **intersected with line of sight**: the gun has to reach
the point from the air.

---

## Rows (1) and (2) — the workpiece alone

Free body: the **workpiece**. Everything pressing on it is external — every support, and
the ground at `a`.

```
(1)   Σᵢ λᵢ uᵢ            +  λ₀ ẑ              =   ẑ  −  t d
(2)   Σᵢ λᵢ (pᵢ − c)×uᵢ   +  λ₀ (a − c)×ẑ      =  −t (pt − c)×d
```

with `λ₀, λᵢ ≥ 0`. Left: what the contacts supply. Right: what gravity and the push owe.

### They are ONE equation, not two

The same `λ` has to satisfy both rows. Stack them into a wrench about `c`:

```
wᵢ = ( uᵢ , (pᵢ − c)×uᵢ )        a contact's wrench,   w₀ = ( ẑ , (a − c)×ẑ )   the pivot's
W(pt,d,t) = ( ẑ − t d , −t (pt − c)×d )                  the demand
```

and rows (1)+(2) for a design `S = {i₁ … i_k}` become one statement in **R⁶**:

> **`W(pt,d,t) ∈ cone{ w₀, w_{i₁}, …, w_{i_k} }` for every `pt ∈ W`, `d ∈ C(pt)`, `t ∈ [0,K]`**

where `cone{·} = { Σ λⱼ wⱼ : λⱼ ≥ 0 }`. **Because `λ` is unbounded, the reachable set is a
CONE**, feasibility is scale-free, and membership is a question about directions alone —
which is the whole of what dropping the force cap buys, and why METHOD §3.2's violet class
(*owed, direction available, not strong enough*) cannot occur.

### The complete magnitude interval

The demand is `D = {W(pt,d,t) : pt ∈ W, d ∈ C(pt), 0 ≤ t ≤ K}`. For a fixed `pt,d`,
`W(pt,d,t)` is the convex combination of `(ẑ,0)` and `W(pt,d,K)` with weights
`1-t/K` and `t/K`. Therefore a convex supply cone covers the complete magnitude interval
iff it covers gravity alone and every full-load demand. Random sampling of magnitudes
is useful for estimating coverage, but is not a proof of these universal statements.

### `d` does NOT come out — and that is not an oversight

`C(pt)` is a spherical **cap**, and every point of a sphere is an extreme point of its own
convex hull, so the interior of the cap is not implied by its rim. Row (3)'s boundary
argument uses a rational map at `K = 0.5` and requires checking critical directions and
visibility boundaries as well as the rim; it has no counterpart here. The current search samples `pt`, `d` and `t`; Step 3 additionally checks the continuous
cap analytically or encloses it by conservative polytopes. No boundary-only direction
reduction is used for the workpiece equations.

### Deciding it, and counting

**Membership is an LP:** `W ∈ cone(G) ⇔ ∃ λ ≥ 0 : Gᵀλ = W`. This is an exact mathematical
statement; a floating-point LP uses feasibility tolerances and its residuals matter.
Conic Carathéodory needs at most **6 generators in R⁶**, or 3 for either separate R³ row.
`slides/tools/contact_cones.py`'s `in_cone()` works in the generators' span, including ranks 0/1/2,
with numerical tolerances; requiring a full three-dimensional cone would reject valid
lower-dimensional demands. The separate-row figures were removed when the demand page
was rebuilt around paired loads; their old counts have not been remeasured.

**A lower bound on the number of contacts, from dimension alone.** A convex cone's
dimension is the dimension of its span, and `cone(S) ⊇ cone(D)` forces
`rank(S) ≥ rank(D)`. Measured on all 11 tips: **`rank(D) = 6` of 6.** Hence at least 6
generators, one of which is the pivot:

> **at least 5 additional point-contact generators, jointly** — historically measured,
> **5 to 7 (typically 6)** on those 11 tips. This is not a count of independent support solids.

Tested SEPARATELY the two rows need 3 and 3. **That number is not the answer**: two
separate memberships in R³ do not imply one joint membership in R⁶. METHOD §11.9 measures
the gap on exactly this — a design answering 100 % of the force problem scored 0.0 % on
the six rows together.

---

## Row (3) — the workpiece and its supports together

Free body: **the assembly**. Every support force is now internal and cancels. What is left
is gravity, the push, and the floor. Write its normal reactions; horizontal force and yaw
are assigned to sufficient floor friction under the current assumption:

```
(3a)  Σⱼ μⱼ      =  R = 1 − t d_z            μⱼ ≥ 0 at foot fⱼ ∈ F
(3b)  Σⱼ μⱼ fⱼ_h =  R · p_h
```

The vertical part of (3a) gives the assembly's total normal force,

```
R  =  1 − t d_z          and         R  ≥  1 − K  =  0.5     for any direction
```

so at `K < 1` the assembly can never be lifted off the floor. Solving (3b) with the moment
balance gives the **centre of pressure** as a lever rule between two points — the plumb
point of the weight, and the push's own floor intercept:

```
h  =  pt − (q_z / d_z) d                       where the push's line of action meets z = 0
p  =  ( 1·g  +  (−t d_z)·h ) / ( 1 − t d_z )  weighted by their vertical force components
```

`μ ≥ 0` with `Σμ = R` is exactly the statement that `p` is a convex combination of the
feet. So:

> **The assembly's normal-reaction equations admit some floor-force distribution iff
> `p(pt,d,t) ∈ conv(F)` for every `pt ∈ W`, `d ∈ C(pt)`, `t ∈ [0,K]`.** This aggregate footprint is the
> current scope of row (3); individual supports are not checked.

`sys_floor/row3.png` illustrates this balance in a single plane: the gravity and push
lines meet at `X`, and their resultant through `X` meets the floor at `p`. This planar
construction is the figure's intended scope. The three-dimensional workpiece plots use
the moment formula in `support_polygon.cop` to obtain both horizontal coordinates of `p`.

### Magnitude reduction and the direction boundary at `K = 0.5`

- Write `L_K = {p(pt,d,K)}`. With `r = pt − g`,
  `p − g = t (r_z d_h − d_z r_h)/(1 − t d_z)`. Since `t/(1−t d_z)` is monotone,
  the complete landing set `L = {p(pt,d,t) : 0 ≤ t ≤ K}` consists of line segments
  from `g` to `L_K`. A convex floor footprint therefore covers it iff it contains both
  `g` and all of `L_K`. Gravity is a required load.
- Substituting `u = d_h/(1−d_z)` (stereographic, from the north pole) at `t = K` gives
  `p(u) = g + K[(1−|u|²)r_h + 2r_z u]/[(1+K)+(1−K)|u|²]`. At **`K = 0.5`** this is
  **`p(u) = g + [(1−|u|²)r_h + 2r_z u]/[3+|u|²]`**, a rational map. The old quadratic
  expression applies only at `K = 1`. A full cone excluding the projection pole becomes
  a disc; the original formula in `d` remains regular at the pole.
- For `r_z > 0`, critical directions satisfy **`d·r = K r_z = 0.5 r_z`**. The full cone's
  image boundary is contained in the images of its rim and this critical set. Rechecked
  on 2026-09-05: the critical circle intersects **0 of 2 640** sampled full cones over
  the current 12 poses. This diagnostic precedes visibility filtering; occlusion can
  introduce additional boundary curves inside the cone. The page's **96 rim angles are
  samples**, and zero folds does not certify the complete visible boundary.

So with the current `L = { p(pt,d,t) : 0 ≤ t ≤ K }`,

> **row (3) solved  ⇔  `conv(F) ⊇ L`  ⇔  `conv(F) ⊇ conv(L)`**

— and the second form is the useful one, because `conv(F)` is convex anyway. **The
requirement is a convex region.** Historical magnitude-extended measurements gave
`conv(L)/L = 1.04–1.25` and reaches of **0.18–0.32 part widths** from `g`; these have not
been recomputed as areas of `L_K`. A triangle can contain any bounded planar set when its
vertices are unrestricted, so three freely placed feet suffice for this assembly-level
geometric containment problem. That does not prescribe the feet of each separate support.

### What its SOLUTION looks like — the ring

The equation above says when a design works. This says what the working designs are. Write
`Q = conv(L)` and `h_Q(θ) = max_{x∈Q} x·θ`. Containment of convex sets is domination of
support functions:

```
F works   ⇔   h_F(θ) ≥ h_Q(θ)  for every θ ∈ S¹,     h_F(θ) = max_{f∈F} f·θ
          ⇔   F meets every supporting halfplane  H(θ) = { x : x·θ ≥ h_Q(θ) }  of Q
```

So the solution set is **not a region — it is a covering condition on the circle of
directions.** Give each foot the directions it answers:

```
Θ(f) = { θ : f·θ ≥ h_Q(θ) }  =  the normal cone of conv(Q ∪ {f}) at f
F is a solution  ⇔  ⋃ᵢ Θ(fᵢ) = S¹
```

For an actual footprint, use this support-function condition or direct convex-hull
containment. Bearings matter: arcs must cover the circle at their actual positions.

**A useful regular-footprint construction.** Enclose `Q` in a disc `(o, ρ)`. A foot at
radius `R ≥ ρ` covers disc-support directions in an arc of half-width `acos(ρ/R)` about
its actual bearing. The former statement `Σᵢ acos(ρ/Rᵢ) ≥ π` is **not sufficient for
arbitrary bearings**: three feet at radius `2ρ` and bearings 0°, 10°, 20° satisfy it but
do not even enclose `o`. Total arc length does not establish coverage.

For **equal radius, equally spaced** feet only, the regular `n`-gon contains the disc iff

> **`R ≥ ρ / cos(π/n)`, `n ≥ 3`.**

| `n` | 3 | 4 | 5 | 6 |
|---|---|---|---|---|
| `R / ρ` | **2.000** | 1.414 | 1.236 | 1.155 |

Containing the enclosing disc is sufficient for containing `Q`, but may use more ground
than `Q` needs. For a proposed nonregular footprint, return to its actual convex hull.

---

## The whole system: solve (1)(2), then check the floor geometrically

Keep the design order: choose workpiece contacts, solve (1)(2) for a common nonnegative
reaction vector, then check (3) by convex-hull containment. **No new floor-reaction
unknowns or mandatory joint LP are needed.** The current row (3) takes the workpiece
and all supports as one assembly. Forces between the workpiece and supports cancel
inside that free body, leaving gravity, the push and the floor reactions.

Use the assembly's `R = 1 − t d_z` and `p(pt,d,t)`, and include every support foot and
the workpiece pivot `a` in `F`. The required geometric condition is `L ⊆ conv(F)`,
equivalently `L_K ∪ {g} ⊆ conv(F)`.
Per-piece pressure centres and compatibility with a selected per-piece reaction split
are outside this aggregate model, as confirmed on 2026-09-06. This replaces the earlier
requirement to pass each solved support load through a separate floor check. Rows (1)
and (2) still require the same reaction vector for force and moment balance.

### How contact is established

Current procedure, confirmed 2026-09-06:

1. The robot pushes the workpiece to `T*` and holds it there.
2. Supports are inserted one at a time, each following its own path and reaching its
   designed contacts in **perfect fit at the endpoint**. Different pieces need not share
   one insertion direction; the sequence must respect the workpiece and previously
   inserted pieces.
3. Once all supports are in position, the robot releases its force.

There is **no release-settle stage** to establish contact. The former one-piece-only
restriction and claim that every rest-loaded contact is automatically closed by gravity
are superseded. Manufacturing clearance and automatic settling are outside this model.
The existing trajectory claw has a measured 1.79 mm gap and remains a corridor example,
not a realised perfectly fitting load-bearing support.

### Assumptions and recorded results

- Supports are massless; normal contact forces are nonnegative and unbounded. Floor
  friction is assumed sufficient for horizontal force and yaw; yaw is not studied here.
- Current loads include **every `t ∈ [0,0.5]`** and every reachable direction in the local **30°** cap. The current baseline searches with 32,768 random demands and validates the continuous domain before reporting fixed-contact completion.
- Historical demand-page sampling for (1)(2): 220 contact points and 48 directions per pose on the
  demand page (the invoice/area page has its own recorded sample). The floor page samples
  220 contact points and 96 rim angles; it does not trace folds or visibility boundaries.
- The `K=0.5` floor update was verified on 2026-09-05: 0/2640 sampled full cones intersect
  the critical circle over 12 poses; five regression checks passed and PNG bytes did not
  change. This is a sampled diagnostic, not a continuous boundary certificate.
- Historical point-contact counts do not count separate support solids. The separate-row
  demand counts are no longer on the paired-load page and have not been remeasured; the joint
  5–7 contact record covers only the 11 tips available in August.
- METHOD and PIPELINE retain their historical `K=1` and capped-force/pressure models;
  [problem_statement.md](../../problem_statement.md#当前决定与讨论记录) records what currently applies to slides.
