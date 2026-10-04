# Four sampling additions followed by four greedy additions

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=slides/DSL_algo \
  .venv/bin/python -m step3_scheculer.initialize_sampling4_greedy4_v28 --jobs 6
```

Each trajectory starts at zero heads. Additions 1–4 use the original top-ten
coverage-gain-weighted random draw. Additions 5–8 deterministically choose the
largest complete-load coverage count, with candidate ID as the tie break.
A zero-gain best head can still be added, as in the original algorithm; there
is no positive-gain termination before the eight-head budget. Stop immediately
on full coverage, or when no legal candidate remains. Try up to thirty
trajectories, cycling 1%, 2%, 0.5% head area families as before. Every head
within one trajectory uses that trajectory's area family. There is no pair
or witness-bundle fallback in this experiment.

The random seed, fixed dataset, head catalogue, common continuous withdrawal
rays, roadmap component intersections, work exclusions, floor clearance and
no-uplift force conditions match cached v19. All 32,768 original loads are
scored, and the final result is independently checked on CPU from the actual
triangles.

The unchanged first four draws are reused from existing hash-validated v19
trajectory reports. Saved masks are reused only if that original trajectory
ended within four heads, checking sample count and coverage count, and hashing
the array as an input. Longer original trajectories are truncated after four
heads and their prefix coverage is computed afresh. Trajectories absent from
the original run compute their first four random additions normally, using
the same seed. This is exact policy continuation, not replacement by the
original final solution: every earlier trajectory still gets its own four
greedy additions before moving on.

Outputs use a fresh `independent_poses_sampling4_greedy4_v28` stage. Original
reports and all fallback outputs remain unchanged. Report runtime as **warm
continuation with cached sampling prefixes**, not cold end-to-end runtime.
Per-pose reports also record the original time spent on the reused prefixes.

The comparison against the six-head original changes both the tail selection
policy and the head budget. Increased success alone cannot distinguish these
two effects. Compare rescued and regressed cases as well as head counts for
the poses both methods solve. A failure is bounded-search failure, not a
proof of geometric infeasibility. No complete fixture is constructed.

Audit and summarize with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=slides/DSL_algo \
  .venv/bin/python -m step3_scheculer.review_initialize_sampling4_greedy4_v28
```

## Measured outcome

All 630 poses were attempted and completed. Original six-sampling policy:
613/630 (97.30%). Four-sampling/four-greedy policy: **613/630 (97.30%)**.
No rescued or regressed pose. All seventeen original failures remain failed
bounded searches. Mean heads among passing poses: **3.672 → 3.790**.
31 passing poses use more heads, five fewer and 577 the same. Eleven passing
solutions use seven or eight heads.

Measured experiment wall time: **1698.491 seconds (28.31 minutes)**, six workers
initially. This includes 1485.784 s initial batch, 144.925 s interrupted retry,
and 67.782 s final bounded retry. Identical saved sampling prefixes were reused,
so this is not cold full-search timing. Numerically degenerate candidate
retries dominated the tail; this experiment does not demonstrate a speedup.

A4/pose_19 was completed with `initialize_sampling4_greedy4_v29`, which retains
the same selection policy, reuses completed v28 trajectories and limits each
ordinary or recovery LP to one second. NNLS proposals are capped at 1000
iterations. Unknown/timed-out candidates are excluded as numerically unresolved,
never marked physically infeasible. Final acceptance uses unchanged original
residual tolerances and actual-triangle CPU verification. The aggregate
review hashes the actual v28/v29 reports and records this exception explicitly.

Three regression tests and the 630-case provenance, selection-policy and
full-load coverage audit pass. The comparison shows no rescue-rate gain and
higher average head count; it does not support replacing the original policy
with this configuration. The head budget also changes from six to eight,
so policy and budget effects have not been separated.
