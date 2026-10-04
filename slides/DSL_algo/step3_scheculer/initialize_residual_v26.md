# Residual-guided additive fallback

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=slides/DSL_algo \
  .venv/bin/python -m step3_scheculer.initialize_residual_v26 --failed-only --jobs 6
```

The unchanged cached v19 greedy handles ordinary cases (613/630). This new
stage starts from its thirty failed seeds in descending coverage order.
All original heads remain fixed. Cached heads of all three area families
may be added, with distinct centers, common continuous exit rays and common
roadmap ports. Default budget: twelve total heads.

Before expanding a seed, form an optimistic force cone from its current
heads and every individually eligible additional head. This deliberately
relaxes joint exit and distinct-center constraints. Test up to 32 evenly
spaced indices of the currently failed original loads. If even this superset
cannot support one demand, no subset of the eligible finite candidate pool
can complete this seed: record the failing index and skip expansion. This
uses the original LP tolerance, not a continuous-geometry infeasibility proof.
Numerically unresolved checks do not prune seeds.

Otherwise score single additions over every original demand. If no single
improves coverage, solve failed-load probes using candidate groups with a
shared ray and port. Sparse primal witnesses identify the heads contributing
necessary generators. Cover those generators with distinct-center compatible
heads, proposing bundles within the remaining head budget (including three
or more heads). Try up to 128 proposals, accept maximum full-load coverage
gain, and repeat. Probe loads only propose or exclude; acceptance uses all
32,768 original loads and an independent CPU check reconstructed from actual
contact triangles. Existing satisfied loads are retained.

Outputs: `output/<object>/independent_poses_residual_v26/<pose>/step3_scheculer/`.
The original greedy and v24 results are preserved. V25 performance trials
are retained separately and are excluded from the final v26 timing.
Greedy seed coverage arrays are reused for initialization only; their shape,
Boolean dtype and covered count are checked, their files are hashed as inputs,
and final coverage is recomputed independently on CPU from actual triangles.
Review all seventeen cases with
`python -m step3_scheculer.review_initialize_residual_v26` in the same environment.

This stage verifies contact forces and cached local exits. It does not build
or validate the complete support body. Failure can still reflect the seed's
restricted exits, finite head catalogue, head budget or incomplete witness
proposals. No global feasibility or head-count optimality claim is made.
