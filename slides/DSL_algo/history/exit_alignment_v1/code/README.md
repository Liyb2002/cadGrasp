# Contact DSL + numerical descent experiment

This tree is a full copy of `slides/baseline_algo`. Copied output is a comparison,
not evidence that the new algorithm has run. `copy_manifest.json` records all
original baseline file hashes. This experiment writes only inside `DSL_algo`.
Another workspace process may update baseline concurrently; snapshot drift is
reported separately from the independent audit of this experiment's designs.

The new Step3 entry is `step3_scheculer/run_dsl.py`. Current experimental results
are ONLY each case's `step3_scheculer/dsl/`, not the copied historical schedules.
Ten B combinations include two historical-revision pairs; their own saved domains,
loads and matching setup snapshots are used. The twenty independent poses can also
be run with `--include-independent`. No loads are regenerated or appended.

## Method

Programs contain variable-count connected surface patches with movable centers,
variable radius and a horizontal withdrawal angle. Delete, add and replace rules
change structure; central-difference gradients and backtracking change continuous
parameters. This is a Python embedded DSL: `Program` and `Patch` are its executable
AST, and `Compiler` gives every program real contact geometry and wrench rays.
It describes Step3 contacts and exits; Step4 still constructs connecting material.
Contact surfaces are clipped against work-face masks and the retained
2 mm group floor margin, connected through real mesh edges, with <=90 degree normal
spread. Centers are projected onto admissible mesh triangles. This is a piecewise
smooth numerical-gradient prototype, not automatic differentiation or a faithful
smooth-CAD optimization system.

Phase 1 optimizes conditioned nonnegative least-squares residuals without an exit
gate. Phase 2 repairs exits and prefers aligned object-frame exit directions;
certified full head rays, not normal tests alone, determine exit acceptance.
Deletion plus parameter repair attempts to reduce head count. All 32768 original
loads are classified by the original seven-row LP with the shared no-uplift condition.
`force_only_contacts.npz` and its report/coverage preserve the first-stage solution.
An exit repair that breaks an already feasible force solution is rejected.
Numerically indeterminate LP proposals are rejected and logged; they are not
declared physically infeasible, and no equilibrium tolerance is relaxed.
Reduced REAL wrench rays and a small original-load subset are used for proposals
only; discovered failed original samples are added to that optimization subset.
Positive soft activation is not used, because unlimited reactions can undo it.

CUDA batches the finite-difference candidates and rewrite candidates using float64
Lawson-Hanson active-set cone residuals (600-iteration budget by default). Real patch clipping,
final full-load LP classification and mesh Boolean construction remain on CPU.
GPU residuals are proposal losses, never feasibility certificates. Small row-space
solves handle nearly parallel rays; KKT failures fall back to CPU NNLS and are logged.
Small-matrix eigensolves are chunked at 256 systems to bound cuSOLVER workspace
when multiple workers share the GPU; allocation failures also fall back to CPU.
`--device auto` uses CUDA when available; `--device cuda` requires it, and
`--device cpu` uses SciPy NNLS. Workers use spawn to isolate CUDA contexts.

Contact area is variable, with numerical radius bounds [0.003D, 0.6D]; no new
pressure limit is imposed. Zero mandatory head thickness is retained. Exit tests
use 0.2 mm constructor-sized finite root probes, rather than retaining the old
1%-of-object-size probe solid. Final Step4 still checks every actual emitted solid.
Step4 constructs the original 1%-of-size root depth, separately from this thin
search probe, and replays its actual withdrawal. A thin search probe is not a
certificate for that thicker root or the final connecting structure.
Large contacts can be preferred when head count is the goal; smallest head count,
global optimality and compactness are not proved. Finite rewrite/iteration/maximum
head budgets are explicitly recorded.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python slides/DSL_algo/step3_scheculer/run_dsl.py \
  --include-independent --jobs 4 --device cuda --steps 8 --samples 24
PYTHONPATH=slides/DSL_algo .venv/bin/python -m unittest step3_scheculer.test_contact_dsl
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python slides/DSL_algo/step4_connect_support/try_dsl_growth.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python slides/DSL_algo/step3_scheculer/review_dsl.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python slides/DSL_algo/step3_scheculer/benchmark_gpu.py
PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python slides/DSL_algo/step3_scheculer/summarize_dsl.py
```

## Downstream construction

`try_dsl_growth.py` feeds NEW certified DSL patches and exits to the copied direct
growth constructor, retaining each comparison's saved placements and foot locations.
New results are `step4/data/dsl_support/`; failed construction is explicitly labeled.
It tests the new contacts but does not search new fixture placements. Geometry
acceptance includes full 500 mm sweeps, all installed floors, actual ground hulls,
working faces, contacts, connected closed material and serialized OBJ replay.
Step3 remains the force authority. A failed fixed-placement construction does not
prove the new contacts cannot form a fixture with another placement.

Actual workstation XY footprint and XYZ occupied box are reported separately
from material volume. Similar object-frame directions are only a compactness proxy;
their benefit must be assessed using actual constructed fixtures.

Batch records live in the last five-pose case's `step3_scheculer/dsl/` and
`step4/data/dsl_support/`. No HTML or videos are generated by this experiment.
