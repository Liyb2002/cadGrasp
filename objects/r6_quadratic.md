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
