# Feasible operation DSL

Entry: `run_operation_batch.py`. The lower-level runner is `run_operation_dsl.py`. Results: each B group has
`step3_scheculer/dsl_operations/` and public `step4/`.

Start from the complete, qualified V5 final contacts and each pose's own object
withdrawal path. This is an incremental experiment; its initial head total must
be compared with V5's final total, not counted as a new independent solution.
No shared exit direction is imposed. The objectives are physical head count
and mean pairwise `1 - dot` of object-frame initial exit directions. Accepted
updates are Pareto improvements in these two objectives. Mean angles in degrees
are an additional descriptive measurement, not the optimized loss.

## Operators

- **Area edit + deletion:** fit real connected surface patches at 1.25%, 1.5%,
  and 2% of actual object surface area, then try removing a head. The operation
  is committed atomically, after all force, exit, and body checks pass.
- **Merge two heads:** replace two active patches with a single connected
  patch, centered at an endpoint or a projected midpoint, with at most 2% area.
- **Exchange + deletion:** remove two old patches and introduce one patch at a
  sampled surface center. This combines a one-head exchange with a deletion,
  so the feasible incumbent is never replaced by a surrogate-only plateau.
- **Gradient proposals:** retain GPU finite-difference contact parameter descent
  and exact line-search acceptance when it enables a deletion. Oversized patches
  are rejected, including patches proposed by this older descent operator.
- **Exit + area/location edit:** independently move a pose's exit toward the
  other object-frame exits, recompile contact areas on admissible faces, and
  retain the edit only when its own complete path and fixture remain feasible.

The area bound is per physical contact patch, **not footprint**. The actual
triangulated surface area is measured after wrap restriction; some patches may
be smaller than the requested target. This does not certify final compactness.

CUDA ranks candidate real wrench cones on a sample subset. Full original-load
LP checks decide force feasibility, and unresolved solver errors reject only
the proposal. A finite ranked shortlist is not an infeasibility proof or a
minimum-head solution. Unchanged qualified incumbents remain valid results.

These new merge/exchange operators operate within a pose's active set. Existing
cross-pose shared patches are preserved and counted by exact installed geometry;
these operators do not establish new cross-pose sharing. Changing a shared patch
in one pose alone is not allowed. The older whole-program pair-sharing search
is not used in this experiment.

## Latest baseline Step4

The executable baseline Step4 sources and resources are copied into
`step4_connect_support/baseline_current/`. Its `sync_manifest.json` records
source and copy hashes. Only Python import prefixes change to isolate the copy
from V5's recorded source dependencies. The actual constructor is its
`EnvelopeGrow`: greedy growth relative to a fixed usage envelope, independent
ground coverage, and complete 5 mm branch/sole cores.

The DSL adapter supplies its current heads, pose placements and independent
paths. It uses the existing DSL path-aware geometry acceptance because the
historical baseline verifier assumes the support moves along a straight ray;
that floor-direction convention cannot check upward-moving object exits.
There is one in-memory full-body acceptance per construction; there is no
exported-geometry check or independent full-body replay. Step3 owns force checks.

`operation_growth_recovery.py` tries the normal copied constructor first. If a
contact cannot start at the baseline's center positions, it proposes other
actual patch centroids/normals, retaining a complete 5 mm start sphere and
recording the short taper exception. Contact-only transitions may bridge the
search-only 0.4 mm padding; they still obey the actual continuous sweeps, floors
and full-body acceptance. It can broaden routed connectivity
proposals and restore translations with floor and foreign-sweep constraints.
Every resulting body receives the same complete acceptance, without relaxed
conditions. `operation_navigation_recovery.py` broadens the finite routing
margin from 50 to 100 mm when needed; this is not a physical support-size bound
and it does not prescribe feet from an earlier support. A failed proposal
preserves the incumbent.

The fresh audit checks all initial/final/accepted states against their original
32768 loads, paths, contact identities, measured areas and recorded construction
certificates. The publisher copies the exact accepted model to public
`step4/shape.obj`, with two English images and both report entry points. Previous
public V5 files are retained under `dsl_operations/previous_public_step4/`.
