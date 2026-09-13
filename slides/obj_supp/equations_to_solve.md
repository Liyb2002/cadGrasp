# Equations to solve — force balance, moment balance, and their joint solutions

**Current scope (2026-09-06): [problem_statement.md](../problem_statement.md#当前决定与讨论记录).** Only `t=K=0.5` is
checked. Independent rigid supports may be inserted sequentially and fit perfectly at
the endpoint. Indices below count contact generators, not separate support solids.

> **Location update, 2026-09-06.** Moved out of `demand/`: this document describes the
> coupled support-feasibility problem. The demand figures now show paired loads only;
> their former separate-force/torque plots and contact counts have been removed.
> Numerical records below are historical and were not remeasured in this move.

*Written 2026-08-29. The current load-only page is described in `demand/demand.md`.
`slides/setup/equations/equations.md` writes all four rows in one notation; this document is rows (1) and
(2) alone, under the declarations settled on 2026-08-28/29, and it goes further than that
page in one respect: it says what the SOLUTION SET is, not only when a given design works.*

---

## 0. The one thing the notation has to show

Rows (1) and (2) cut the free body around the **workpiece**. In `slides/setup/equations/three_equations.png`'s
own symbols:

```
(1)   ∫_supp_obj  F_supp                  dA   =  −( −mg ẑ  +  F_push )
(2)   ∫_supp_obj  r_supp × F_supp         dA   =  −( r_push × F_push )
```

**The same `F_supp`, over the same `supp_obj`, against the same `dA`.** Two rows, **one**
unknown. That is the whole difficulty and the notation must not let it hide: how hard each
patch of skin presses is a single field, and it has to answer the force row and the torque
row *simultaneously*. Spend a patch's pressure balancing a force here and it is already
spent over there.

Discretely — supports at `pᵢ` pushing along the unit inward normal `uᵢ`, the workpiece's own
ground contact at `a` pushing `ẑ`, the process pushing at `q ∈ W` along `d ∈ C(q)` with
fixed magnitude `t = K = 0.5`:

```
(1)   Σᵢ λᵢ uᵢ           +  λ₀ ẑ            =   ẑ  −  t d
(2)   Σᵢ λᵢ (pᵢ − c)×uᵢ  +  λ₀ (a − c)×ẑ    =  −t (q − c)×d
```

Same `λ`. Forces in body weights, so gravity is exactly `−ẑ` and `mg = 1`; moments about
the centre of mass `c`, which is why **gravity appears in row (1) and not in row (2)** —
acting at `c`, it turns nothing about `c`.

### They are one equation

Stack them into a wrench about `c`:

```
wᵢ = ( uᵢ , (pᵢ − c)×uᵢ )        a contact's wrench
w₀ = ( ẑ  , (a  − c)×ẑ  )        the pivot's
φ(q,d) = ( d , (q − c)×d )       the process push's own wrench
w_g    = ( ẑ , 0 )               gravity's
W(q,d,t) = w_g − t·φ(q,d)        what the contacts must supply
```

> **rows (1)+(2)  ⇔  `W(q,d,t) ∈ cone{ w₀, w₁ … w_k }` ⊂ R⁶, for every `q ∈ W`,
> `d ∈ C(q)`, with `t = K = 0.5`**, where `cone{·} = { Σ λⱼ wⱼ : λⱼ ≥ 0 }`.

**The demand is gravity's wrench minus `K` times the push's wrench.** Writing it that way
rather than expanding it is worth the line: it says the demand set is a shrunken reflection
of the push set about `w_g`, and it makes §3's pointedness a one-line check.

Testing the two rows separately is a **necessary condition and not feasibility**, and the
gap is not academic: METHOD §11.9 records a design answering **100 % of the force problem
and 0.0 % of the six rows together**. The former demand page measured the same thing from the other
side — 3 contacts answer each row alone, **5 to 7** answer them jointly.

---

## 1. The declarations these equations are written under

Three, and two of them are new. Every number in METHOD.md and PIPELINE.md predates all
three and was measured under the old ones.

**`K = 0.5`.** The process pushes with half a body weight. METHOD §1 and PIPELINE run at
`K = 1`. §3 below is what the halving buys, and it is not cosmetic.

**`λ` is unbounded.** `λⱼ ≥ 0` and nothing more: a contact pushes and never pulls, and no
cap. METHOD §1 caps a support at one body weight and PIPELINE at 10 kPa. Dropping the cap
makes the reachable set a **cone**, so feasibility is scale-free and membership is a
question about directions alone. Two consequences worth naming:

- METHOD §3.2's **violet** class — *owed, direction available, not strong enough* — cannot
  occur. Under a cone there is no "not strong enough".
- METHOD §1's rule that *each distinct push direction is used at most once* loses its
  justification entirely. It existed to stop the cap being bought off by repetition; with
  no cap there is nothing to buy. Two contacts on one face at different places are two
  different generators — their torques differ, and can be exactly opposite — and both are
  kept.

It is a modelling choice and not physics: a real support on a curved surface cannot deliver
arbitrary force through a point contact. It belongs on the assumptions slide.

**No minimality objective.** This project has never declared "the fewest supports" or "the
fewest contacts" as what it is solving for, and this document does not smuggle it in. The
reason is structural rather than editorial: under an unbounded `λ`, **adding a support can
only enlarge the cone — it never hurts**, so "fewest" does not fall out of the physics. It
would have to be imposed from outside, and nothing has imposed it. What actually limits a
design is the work-region keep-out, insertion and removal, and printability. Counts below
are **observations**, never targets. METHOD §4.2, §11.7 and §11.9's "proved minimal" results
answer a question that was not asked, under the capped model besides.

---

## 2. What comes out exactly, and what stays sampled

**Four variable parameters describe the current full-load push:** `q` (2) and `d` (2).
The magnitude is fixed at `t=K=0.5`. **No zero-process-force invoice is required.** The
previous `[0,K]` task would, by affine dependence and convexity, need both its endpoints;
that is an optional extension, not a prerequisite for this task or a missing check.

**`d` does not retire, and that is not an oversight.** `C(q)` is a spherical cap, and every
point of a sphere is an extreme point of its own convex hull, so the cap's interior is not
implied by its rim. Row (3)'s boundary argument uses a *different* map — rational at
`K = 0.5`, quadratic only at `K = 1` — and also has to account for critical directions
and visibility boundaries. It has no counterpart here.

**So `q` and `d` are sampled**, 220 contact points × 48 directions a pose. Every claim below
inherits that sampling: "no push defeats this design" means "no push in the sample".
Row (3)'s figure also samples contact points and rim angles; neither page establishes a
continuous certificate from its point cloud. Say so in the paper.

**Position on a face is NOT sampled, and that is exact.** For a fixed `u` the map
`p ↦ (p − c)×u` is an isometry (METHOD §11.2), so a face's achievable torques are a
congruent copy of the face. Summing several pads on one face,
`Σ λᵢ (u, (pᵢ−c)×u) = Λ·(u, (p̄−c)×u)` with `p̄` in the convex hull of their positions, so

> the wrenches a face `F` can supply are `C_F = cone{ (u, (p−c)×u) : p ∈ conv(F_adm) }`,
> and its **extreme rays are the vertices of `conv(F_adm)`** — a finite, exact set.

Two things follow. Choosing a face buys a **three-dimensional cone** (one dimension of
scale, two of position), not a single ray. And when the work region bites a hole in a face,
`conv(F_adm) ⊋ F_adm`, so several pads genuinely reach points one pad cannot — that is a
real physical gain, not a bookkeeping trick. This replaces METHOD §11.9's *menu* (patch-hull
vertices plus one area sample per 2 mm², capped at 96 a direction), whose infeasibility
verdicts could only ever be menu-relative.

---

## 3. The demand set, and why `K = 0.5` matters to these rows

```
D = { w_g − K·φ(q,d) :  q ∈ W,  d ∈ C(q) }
```

**`cone(D)` is pointed.** The force half of every demand is `ẑ − K d` with `d` a unit
vector, so its vertical component is `1 − K d_z ∈ [1−K, 1+K] = [0.5, 1.5]` — strictly
positive, with margin. Every demand therefore lies in the open half-space
`{ w : w·w_g > 0 }`, and `cone(D)` is a pointed cone rather than something that reaches all
round the origin.

> At `K = 1` a push straight up gives force part `ẑ − ẑ = 0` and the demand cone touches the
> boundary. **The halving is what keeps rows (1)(2)'s demand pointed**, and §5 shows that is
> exactly what lets six generators do a job that would otherwise need seven. `slides/sys_floor`
> declared `K = 0.5` for row (3)'s own reason — `R ≥ 1 − K` bounds the landing region — and
> it turns out to pay here too, for an unrelated reason.

**`rank(D) = 6` of 6**, measured on all eleven tips that existed on 2026-08-27.

---

## 4. The solution set

This is what the document is for. Row (3) does not answer with a design either — it answers
with `⋃ Θ(fᵢ) = S¹`, a covering condition on the circle of directions, and each foot's `Θ(f)`
is the set of directions that foot answers. **Rows (1) and (2) have the same shape on a
bigger sphere**, and the derivation is three lines.

Containment of closed convex cones is domination of support functions. Under an unbounded
`λ` the supply's support function takes only two values:

```
h_G(y)  =  0     if  y·wⱼ ≤ 0  for every generator
        =  +∞    as soon as  y·wⱼ > 0  for one of them
```

So `h_D ≤ h_G` collapses to a pure sign condition, and the set a support answers is

```
Θ(p)  =  { y ∈ S⁵ :  y · w(p) > 0 }          an OPEN HEMISPHERE, centred on ŵ(p)
Y*    =  S⁵ ∖ cone(D)°                        the directions that must be answered
```

> **`G` is a solution  ⇔  `Θ(w₀) ∪ ⋃ᵢ Θ(pᵢ)  ⊇  Y*`.**
>
> Dually and more shortly: **`cone(G)° ⊆ cone(D)°`** — the supply's polar cone must fit
> inside the demand's.

Each candidate support claims one open hemisphere of the wrench-direction sphere, centred
on its own wrench; the pivot claims one; a design works exactly when the claims cover `Y*`.
No design has to be solved for to state this, and any proposed `G` can be tested against it
by normalising `k+1` vectors.

**The set is upward-closed.** Claiming another hemisphere can only enlarge the union. That
is the same fact as "adding a support never hurts", and it is why §1's third declaration is
structural rather than a matter of taste.

### Three different questions, and which object answers each

| question | answered by |
|---|---|
| **is the solution set empty?** — can this workpiece, in this pose, with this work region, be held *at all*? | the **full-supply test**: `cone(D) ⊆ cone(G_all)`, over every admissible face at once. A NO here means **no design exists**, not "none was found". |
| **is this particular `G` a solution?** | the **covering condition** above; equivalently one phase-1 LP per demand, with numerical feasibility tolerances. |
| **produce a member of the set** | **prune** from the top: start from `G_all`, which works iff anything works, and delete while the check still passes. |

The full-supply test is cheap however large the candidate set is: `∃λ ≥ 0 : Gᵀλ = w` has a
dual in **six** variables against `N` constraints, so `N` = 70 000 costs little more than
`N` = 700. When it fails it hands back the `y` no support can answer — METHOD §3.2's **red**,
*no non-negative mixture points that way at all*, which under an unbounded `λ` is the only
failure mode there is.

### Pruning, and what decides what you get

For a fixed static candidate set, deleting a generator while the same demand test still
passes preserves that static feasibility. It says nothing by itself about the geometry
of the support solids or their insertion paths. Fully pruning a fixed candidate set
returns an **irredundant, not necessarily minimum** set and depends on deletion order.

**Deletion order is a heuristic.** Removing awkward-to-insert or awkward-to-print
candidates first can guide the design, but cannot guarantee that the survivors have
feasible paths or are printable. Adding contacts enlarges the abstract force cone; adding
physical solids can create collisions and restrict the remaining insertions.

Current procedure: the robot holds `T*`; several independent supports can be inserted
**one by one, along different paths/directions**. Check each piece's path, the previously
inserted pieces, and its final designed contacts. At the endpoint contacts fit perfectly;
no release-settle stage supplies missing contacts. There is no shared-direction condition
across all independent pieces. The floor check treats the workpiece and all supports
as one assembly and checks its pressure centre against the aggregate footprint, as in
[equations.md](../setup/equations/equations.md).

The historical greedy-from-zero trap can still occur for a full-dimensional demand whose
first-step coverage is zero. It does not imply that lower-dimensional demands are
infeasible: the corrected `slides/tools/contact_cones.fewest()` checks zero and one extra contact before the
pair/beam search. That search remains heuristic beyond the exhaustively checked levels.

The historical joint count of 5–7 was obtained this way: the union of the LP bases over the
whole demand (Carathéodory caps each at 6), then pruned.

**The known trap.** Nothing in a rigid frictionless model requires a contact to have area,
so every shrinking procedure converges on zero-area points. METHOD §5 measured it: refining
a 40-patch support down to 4–6 patches gives 4965 mm²; re-cutting the survivors finer and
shrinking again leaves the *count* almost unchanged (5→6) and the area at **139 mm² (3 %)**.

Manufacturing limits and feature-size stopping rules are possible later inputs. They are
not newly imposed by this review: current contacts are ideal and fit perfectly, while
printability is a separate design goal that static pruning alone cannot establish.

---

## 5. What the description says about counts

Counts are observations here (§1), but the covering form makes two of them structural
rather than empirical.

- **Six generators is the rank floor.** `cone(G) ⊇ cone(D)` forces `rank(G) ≥ rank(D) = 6`,
  so at least six generators, one of which is the pivot — **at least five additional point-contact generators**. A rigid support can carry several.
- **Seven would be needed to cover the whole sphere.** Open hemispheres centred at
  `ŵ₁ … ŵₙ` cover *all* of `S⁵` exactly when the centres positively span `R⁶`, which takes
  `n + 1 = 7` (METHOD §8's Davis 1954 bound, one dimension up).
- Measured: **6 to 8 generators**, i.e. 5 to 7 additional contact generators, typically 6, on the old 11-tip sample.

> Six is fewer than seven, so **a five-contact design (plus the pivot) cannot be covering the whole sphere —
> it exists only because `cone(D)` is pointed.** And `cone(D)` is pointed because `K < 1`
> (§3). Those two facts arrive from opposite ends of the problem and are the same fact.

METHOD §11.8's warning survives the translation and is worth repeating in this language,
because it is the half that gets dropped: **rank is not enough, the generators must surround
the demand.** Its measured table — three mutually perpendicular generators reach 10.8 % of
axes, a spread tetrahedron 100 %, four in one half-space 10.2 %, **six in one half-space
15.2 %** — is the statement that count decides nothing. In the covering language it is
obvious: hemispheres whose centres all lean one way leave a whole cap uncovered no matter
how many there are.

---

## 6. Where the parallel with row (3) holds, and where it breaks

**Holds.** Both rows answer with a covering condition on a sphere of directions, and in both
the candidate's claim is computed from the candidate alone: row (3)'s `Θ(f)` is the normal
cone of `conv(Q ∪ {f})` at `f`; here `Θ(p)` is the open hemisphere about `ŵ(p)`.

**The geometric checks differ.** Row (3) checks containment by the actual footprint's
convex hull. The old `Σᵢ arccos(ρ/Rᵢ) ≥ π` statement ignored arc locations and is not a
sufficient test for arbitrary foot bearings. Only the regular construction survives:
`n` equal-radius, equally spaced feet enclose a disc of radius `ρ` when
`R ≥ ρ/cos(π/n)`. The higher-dimensional wrench problem retains its cone/polar covering
condition and numerical feasibility tests; a scalar arc-length sum does not replace either.

---

## 7. What is owed

- **`q` and `d` are sampled**, 220 × 48 a pose (§2). Every "no push defeats this" is
  relative to that sample. `t` is fixed at `K`; the face reduction in §2 is an analytic statement.
- **The joint numbers are stale by one row.** The historical 5–7 was measured over the
  **eleven** rows that existed on 2026-08-27. B's second row, added 2026-08-28 from a
  placement `poses.json` never recorded, is a genuinely different pose and is **not**
  covered. Re-running the R⁶ LP over the twelve is unwritten.
- **The full-supply test is not implemented.** Nothing in the repo yet answers "is the
  solution set empty" for a workpiece and pose. It is the cheapest thing in this document
  and the only one that can return a definitive NO.
- **The former separate balls have been removed.** Their count of 3 was not the joint
  result. The current demand page shows paired loads and reports no contact counts;
  the historical joint 5–7 measurement remains a separate calculation.
- **Printability and a final fitting insertion design remain separate geometric tasks**;
  ideal contact fit does not turn a pruning priority into their certificate.
- **The removed METHOD.md used the old declarations** — `λ ≤ 1`, `K = 1` — so its §4.2, §11.7 and
  §11.9 numbers are not comparable with anything here, and its §1 rule about one use per
  direction is superseded (§1). Current declarations remain here and in [slides/problem_statement.md](../problem_statement.md#当前决定与讨论记录).

**2026-09-06 numerical update.** `slides/tools/contact_cones.in_cone()` handles ranks 0/1/2/3, zero vectors,
and nonnegative coefficients with floating-point tolerances. Carathéodory permits at most
3 generators in R³ and 6 in R⁶; triples do not solve the general R⁶ problem. The separate
12-tip figures were removed; their statistics and the joint 11-tip numbers were not rerun.
This document's analytic descriptions do not upgrade those old samples.
