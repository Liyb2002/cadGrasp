# Continuous R6 demand and projection loss

For a working triangle, keep the existing symbols `a_f`, `edge_u_f`,
`edge_v_f`, `n_f`, `g`, and processing force `P`. The exact demand is

\[
r=a_f+u\,edge_{u,f}+v\,edge_{v,f},\qquad
b=(-g-P,-r\times P).
\]

The original triangle bounds, inward force cone, magnitude interval and
object visibility stay in `r6_domain.json`. Grounded and airborne objects
use this same demand; their available reactions differ.

With \(z=(1,P,uP,vP)\), write \(b=B_fz\). `r6_quadratic.npz` contains
the exact 6 by 10 matrix for each working triangle. Compilation reads
geometry and physical constants only, with zero demand samples or fitting.
`r6_quadratic_index.json` inventories all 21 objects, 630 poses and three
angles (1,890 expressions).

For a fixed valid active reaction basis \(A_I\), let
\(Q_I=I-A_IA_I^+\). Include the original seventh equilibrium equation by
appending a zero coordinate to \(Sb\), where \(S\) is the original wrench
scaling. Then

\[
\ell_f(z)=\tfrac12 z^TH_{f,I}z,\quad
H_{f,I}=\widetilde B_f^TQ_I\widetilde B_f,\quad
\nabla_z\ell_f=H_{f,I}z.
\]

On a region with that same projection basis, the integral is
\(\tfrac12\operatorname{tr}(H_{f,I}M_{f,I})\), where
\(M_{f,I}=\int z z^T\,d\mu\). The measure is working-triangle area,
uniform solid angle, and uniform processing-force magnitude, conditioned on
original visibility. It is **not** six-dimensional Lebesgue volume: the
demand domain is embedded in R6 and has five intrinsic parameters.

Different loads can have different active bases. A formula for the demand
does not eliminate the need to locate these regions. The implementation
uses the existing fixed positive quadrature and covered/uncovered strata;
it evaluates these continuous formulas at their integration nodes.

The demand does not depend on exit direction or rigid translation about
the object's COM. Those operations change contact availability. With the
current hard contact grid, loss is flat between contact events, so a useful
operation derivative remains a secant across those events. Reuse valid
projection bases across that pair of layouts; update only when primal or
all-ray dual conditions fail. This preserves the physical unbounded cone:
a positive soft weight on an unbounded contact ray cannot encode contact
availability.

## Controlled B experiment

The unchanged implementation and this unit change each ran all 42 current B
sets from their saved Step4.1 layouts, with six jobs, three CPUs per worker,
and the same 10-round initial / 8-round own-continuation budgets. Both fresh
searches passed 42/42 sets. Entry points, Juxtapose, step pools, XYZ/floor
physics and the original 32,768-demand-per-pose PASS criterion stayed fixed.

Wall time was 22.33 → 19.04 minutes (14.75% less); sum of search time was
67.81 → 60.90 minutes (10.19% less). Fresh exported material summed to
3,829.50 → 3,757.05 cm³ (1.89% less). Thirty sets differed by less than
0.01% in material; twelve differed more, ranging from −51.85% to +50.73%.
The formula/loss equivalence does not imply identical iterative search
branches or identical final shapes. The measured whole-run time also
reflects those branch changes, rather than only projection speed.

On 39 direction/XYZ perturbations of an eight-pose set, with BOTH geometric
and JIT caches prewarmed, loss evaluation took 0.302 → 0.236 seconds;
maximum loss difference was 6.94e-18. All 1,890 compiled expressions passed
independent geometric forward/Jacobian checks (maximum errors 1.04e-17 and
0). Forty-eight targeted tests passed. All original B object inputs and
Step3/4.1 artifacts were preserved. The final audit reads saved masks and
hashes; it does not run another force solve or geometric acceptance.

[Per-set comparison](../slides/Co-optimize/output/B/r6_unit_comparison_v1_results.md)
· [Mesh image comparison](../slides/Co-optimize/output/B/r6_unit_comparison_v1_index.html)
· [Verification](../slides/Co-optimize/output/B/data/r6_unit_comparison_v1/verification.json)
