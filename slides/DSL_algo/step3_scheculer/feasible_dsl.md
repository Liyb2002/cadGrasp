# Feasible-incumbent DSL (V5)

The active entry is `run_feasible_dsl.py`. V4's mandatory all-head sharing is
superseded. A state contains pose-specific head subsets, one independent exit
path per pose, and fixture placements. Physical head IDs identify actual installed
contact patches and root solids. A shared ID must have coincident installed
geometry; the constructor creates its solid once.

Initialization preserves the exact saved contact triangles, placements, and
independent exit directions. It does not recompile every head or replace the
independent state with one common object-attached program. All 32,768 original
loads are checked. Saved pose2 seeds are incomplete in five groups; these inputs
need a separately verified initialization repair, not a lower pass threshold.

The incumbent is a complete feasible state with its accepted connected fixture.
Every proposal must satisfy all original loads, work-face exclusion, contact-floor
clearance, its own continuous exit, and full constructor acceptance against all
installed heads, floors, work surfaces and independent object sweeps. Failed
proposals retain the state and its fixture certificate. Step4 publishes the exact
accepted witness; it performs no new construction search or geometry replay.

Two objectives must be non-increasing at every accepted optimization step:

- Number of distinct physical heads, including optional sharing across poses.
- Mean pairwise `1 - dot(u_i, u_j)` of initial object exit vectors, expressed in
  the original object frame. Different exits, including perpendicular exits, are
  allowed. This angular preference is a proxy for channel reuse, not a guarantee
  of smaller final XY area or volume.

Head deletion and optional pair registration change discrete structure. Numerical
central differences and backtracking move head centers and radii on the actual
admissible mesh. Batched float64 CUDA NNLS evaluates force surrogates; exact CPU
LP checks all original loads before a proposal can commit. Surrogate improvements
alone never replace an incumbent. Exit alignment uses small independently checked
steps. Pair registration is a restricted proposal family and may fail its floor
bound; failure does not invalidate the independently initialized fixture.

This is a hybrid continuous/discrete search with feasible commits, not a smooth
end-to-end differentiation of mesh booleans. The finite search does not prove
minimal head count or global impossibility. Rejected scratch proposals can be
infeasible; every published incumbent and accepted update must be feasible.

Run the ten saved B combinations:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=slides/DSL_algo \
  .venv/bin/python slides/DSL_algo/step3_scheculer/run_feasible_dsl.py --jobs 3
```

Each group stores initialization, accepted/rejected proposal records and final
geometry in `step3_scheculer/dsl_feasible/`; Step4 publishes the identical geometry
in `step4/data/dsl_feasible_support/`. Older `dsl` and `dsl_shared` results retain
their original definitions and are not the current algorithm.

## Verified B batch

All 10 combinations and their 32 task instances passed complete acceptance and
a fresh feasible-commit/identity audit: 111 initialized heads → 81 final physical
heads, 36 accepted updates, and 34 regression tests passed. Actual cross-pose
sharing occurs in pose3+6 (8 → 3 heads; 56.6° → 4.5° exit angle); other reductions
remove independent redundant heads. Both historical pose1+3 cases retain their
feasible seeds. This is not evidence of sharing in every combination.

The pose2 repair also needs translation-only seating to clear foreign roots and
preserve enough landing space. LP separator proposals use the real independent
head/path geometry and saved ground material as an initialization witness.
Complete constructor acceptance, rather than the LP alone, certifies the state.
Optional seed landing centers enrich the finite foot-candidate pool; they are
not mandatory feet. Unresolved numerical LP proposals are rejected and retain
the current feasible witness.

Audit and result summary:
`output/B/pose2+9+13+15+17/step3_scheculer/dsl_feasible/experiment_summary.md`.

## Public Step4 delivery

The batch entry now invokes the complete feasible-state audit followed by
`step4_connect_support/publish_feasible_step4.py`. All ten public stage entries
contain the accepted connected model at `step4/shape.obj`, two English images at
`overview.png` and `construction_steps.png`, and current reports at `report.json`
and `data/report.json`. Initialization inputs are preserved byte-exactly before
public replacement; relocated dependencies retain original generator hashes.

Sharing remains restricted: the accepted pose3+6 result is a warm proposal from
the prior experiment. Live pair registration shares a whole head program, rather
than merging only one independent head. Other groups' passed feasibility audits
must not be interpreted as general sharing success or a proof sharing is impossible.
