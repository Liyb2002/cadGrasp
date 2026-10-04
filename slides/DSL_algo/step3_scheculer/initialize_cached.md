# Cached single-pose initialization

Current entry: `python -m step3_scheculer.initialize_cached_v19 --jobs 6`.
Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=slides/DSL_algo`
with the project virtual environment. Defaults cover all current active objects
and their thirty saved poses. `--objects B A1-f C5` restricts the objects.

Initialization consumes the exact fixed Step1 samples and strict, hash-validated
Step2 caches. It does not fit circles, generate poses, rebuild roadmaps, or
repeat continuous collision checking. Cached six-dimensional moment generators
are lifted with the original shared no-uplift constraint. CUDA checks original
LP primal/dual certificates in batches; the LP solver remains on CPU.

Each pose has a thirty-trajectory budget. Area fractions cycle through 1%,
2%, and 0.5%, giving ten trajectories per family if all thirty are required.
One trajectory uses one area family. At each of at most six additions, rank
eligible cached heads by coverage of all 32,768 original loads and draw from
the ten best with probability proportional to new coverage. Equal zero gains
use uniform probabilities. Stop at the first accepted trajectory. No head
replacement, merging, direction optimization, or pose displacement occurs.

Individual head validity alone is insufficient: every selected head must share
a cached continuous withdrawal direction and a cached roadmap component.
Rigid translation along the common ray clears their union because each head
has a continuous certificate for that ray. These are local contact and exit
certificates, not a constructed-body or robot-motion certificate.

For the final result, reconstruct force normals and moments from the actual
contact triangles and source mesh faces, compare against cached generators,
and classify all original demands again on CPU. Preserve work exclusions,
the precomputation's 1.5 mm contact clearance, and full-head cached legality.
Numerically unresolved candidate proposals are excluded, never accepted or
reported as geometrically impossible. A failed thirty-trajectory search does
not prove the pose has no solution.

Outputs use `output/<object>/independent_poses_cached_v17/<pose>/step3_scheculer/`.
The stage name is retained across small corrections. V17 records that passed
its stricter accidental 2 mm contact check remain valid; V18 reads the actual
shared clearance constant; V19 additionally excludes singular-matrix candidate
errors as unresolved numerical proposals. Each accepted record hashes the
exact generator version used. Do not modify these generators after acceptance.
The summary auditor is `python -m step3_scheculer.review_initialize_cached_v17`.
