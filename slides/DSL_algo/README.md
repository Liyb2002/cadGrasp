# Active: absolute-direction DSL

Head count is no longer an optimization target. Support reaction normals and object exits are evaluated in the fixed fixture frame; different object poses may contact completely different regions. Actual Step5 occupied XYZ bounding-box volume decides whether a fully qualified proposal replaces the incumbent.

All **10/10 B sets / 32 task instances** have completed Step4 and Step5. Compared with the previous qualified V6 fixtures, **7 sets improve occupied volume by 9.50–83.02%**; the two pose1+3 versions and pose3+6 retain their smaller qualified fixtures. All 32768 original loads per task, local paths and accepted construction witnesses pass the final audit. **24 related tests pass.** The latest copied baseline EnvelopeGrow and Step5 metric code are used. No exported-model geometry replay is performed.

[All-set volume/direction table](output/B/pose1+3/step5_evaluate/absolute_comparison.md) · [Comparison plot](output/B/pose1+3/step5_evaluate/absolute_comparison.png) · [Algorithm](step3_scheculer/absolute_dsl.md) · [Step4 source manifest](step4_connect_support/baseline_current/sync_manifest.json) · [Step5 source manifest](step5_current/sync_manifest.json)

Each group's current model is `step4/shape.obj`, with two English images `overview.png` and `construction_steps.png`; its metrics and `bbox.png` are in `step5_evaluate/`. The primary volume includes the objects and installed support in saved workstation axes; it is not support material volume.

Run `PYTHONPATH=slides/DSL_algo .venv/bin/python slides/DSL_algo/step3_scheculer/run_absolute_batch.py --jobs 2`. Add `--resume` to reuse hash-valid completed rounds. Direction/pose descent is followed by real construction and measurement, local collision-aware placement compaction, and automatic strict floor-candidate recovery when recorded failures need it. Failed or larger candidates preserve the incumbent. These bounded searches do not certify a global optimum.

---

## Historical V6: feasible operation DSL

All **10/10 B pose sets** have been regenerated through public Step4 using the latest copied baseline EnvelopeGrow. Every pose passes all 32768 original loads and its own exit path. Relative to the qualified V5 finals, physical heads decrease **81 → 72** with 19 feasible updates. Contact areas may reach 2%; 19 related tests pass. No additional cross-pose sharing is established: the three shared heads remain in pose3+6.

[Complete comparison](output/B/pose2+9+13+15+17/step3_scheculer/dsl_operations/experiment_summary.md) · [Head/angle plot](output/B/pose2+9+13+15+17/step3_scheculer/dsl_operations/all_cases.png) · [Algorithm](step3_scheculer/operation_dsl.md) · [Copied Step4 source manifest](step4_connect_support/baseline_current/sync_manifest.json)

The current models are each group's **`step4/shape.obj`**, with `overview.png`, `construction_steps.png` and `report.json`. All ten new public model hashes differ from the previous public models. For example: [pose1+3, four heads](output/B/pose1+3/step4/shape.obj), [its construction](output/B/pose1+3/step4/construction_steps.png), [pose2+12+15, six heads](output/B/pose2+12+15/step4/shape.obj).

Run `step3_scheculer/run_operation_batch.py --jobs 3` with this tree on PYTHONPATH. It performs bounded navigation recovery when needed, audits all groups and publishes their identical accepted Step3 witnesses. Failed proposals preserve the qualified incumbent; no full-body export replay is performed. Prior public results are preserved in `step3_scheculer/dsl_operations/previous_public_step4/`.

## Historical V5 results

All **10/10 B combinations** have complete feasible fixtures. The 32 task
instances each pass all 32768 original loads. **111 initialized physical heads
→ 81 final heads**, with 36 fully feasible accepted updates. Every accepted
update preserves both objectives within numerical tolerance. Step4 models are
byte-identical to their Step3 accepted construction witnesses. Regression tests:
34 passed; the complete feasible-commit and actual-sharing audit passed.

Actual sharing currently occurs in **pose3+6**: 8 independent heads → 3 shared
physical heads, with object-relative exit angle 56.6° → 4.5°. Other improvements
remove redundant independent heads; both historical pose1+3 copies retain their
six-head feasible states. All five original pose2 seeds were force-incomplete;
initialization repairs restore 32768/32768 coverage before optimization.

[Complete results](output/B/pose2+9+13+15+17/step3_scheculer/dsl_feasible/experiment_summary.md)
· [Head count comparison](output/B/pose2+9+13+15+17/step3_scheculer/dsl_feasible/all_cases.png)
· [Actual shared-head assignment](output/B/pose3+6/step3_scheculer/dsl_feasible/head_assignments.png)
· [Accepted fixture](output/B/pose3+6/step4/construction_steps.png).

# Joint shared-head DSL with numerical descent

The active experiment is `joint_shared_head_descent_v4`. It replaces the previous
independent-pose DSL: one group has **one physical head program**, compiled once
in the first task's world frame. Every other pose receives a rigid view of the
same contact triangles and the same finite-depth head solids. Their installed
geometry must coincide, not merely their labels. Complete support construction
creates each physical root only once.

The program has variable head count, head centers and head radii. Global delete
and merge rules reduce the number of distinct physical heads. Numerical central
differences and backtracking repair the shared parameters; every pose's conditioned
force residual contributes to every gradient. GPU float64 NNLS batches propose
changes; CPU LP checks all 32768 immutable original loads for every task. A head
is counted once regardless of how many poses can use it. All retained heads are
available in every pose; equilibrium may assign zero force to any of them.

The placement family is explicit and restricted: the fixture keeps the same
relationship to the object, and both are rigidly reoriented to each saved task
pose. All object meshes and shared head views coincide in fixture coordinates.
This is genuine physical reuse but does not search arbitrary registrations in
which the object moves between different contact sites in the fixture. Union
work-face exclusions and all-pose 2 mm contact clearance apply. Finite root depth
is chosen once in the canonical frame and rigidly transformed, never recomputed
from each pose's different axis-aligned box.

The objectives are:

1. Minimize distinct physical heads through global deletions/merges, followed by
   all-pose parameter descent and exact force acceptance.
2. Prefer similar feasible exit directions as a **soft** normal-opening loss.
   A common direction is never required; 90-degree alternatives remain allowed.
3. Jointly select continuously checked exit paths using the exact union of their
   padded object sweeps in the common fixture workspace. Shared channel space
   is preferred to disjoint channels. This is discrete beam search; it is not an
   autodifferentiable full sweep-union objective. Actual accepted fixture XY
   footprint and material volume rank the final construction alternatives.

Every physical head remains in collision checks in every pose, even if an
individual equilibrium uses no force from it. Horizontal, oblique, vertical and
lift-then-slide paths are retained. The actual moving object obeys its floor.
Paths are piecewise-linear translations; rotation during extraction is absent.

A cheap checked-root roadmap is a prefilter. Step3 still performs complete body
construction before accepting a case: all-pose floors, work surfaces, full exit
sweeps, connected material, contact roots, complete rod/foot cores and ground-hull
coverage must pass the constructor. This is the previously disclosed heavy
constructive-witness contract, not a new lightweight geometric theorem. There is
one in-memory construction acceptance and an exact serialization comparison;
no independent exported-model geometry replay is performed. Step4 publishes the
same witness without another connector search. Failure of the finite search is
not proof of physical impossibility. No global minimum is claimed.

## Outputs

Active per-case results:

- `step3_scheculer/dsl_shared/report.json`: authoritative Step3 verdict.
- `step3_scheculer/dsl_shared/sharing.json` and `shared_heads.npz`: unique physical
  heads, their rigid placement, and pose use identities.
- `step3_scheculer/dsl_shared/shared_heads.png`: labeled canonical heads and pose
  availability matrix; rejected candidates are not accepted fixtures.
- `step3_scheculer/dsl_shared/pose_*/`: rigid contact views, exact coverage, local
  exit witnesses and force-only checkpoints.
- `step3_scheculer/dsl_shared/construction_witness/`: full construction evidence.
- `step4/data/dsl_shared_support/shape.obj` and `overview.png`: accepted support.

Old `dsl/` and `dsl_support/` folders are historical **independent-head** results,
not evidence of shared-head optimization. Copied baseline outputs are comparison
inputs. Baseline code and outputs are not modified by this experiment.

The 10 B combinations and 20 independent poses are separate cases. Historical
`pose1+3` and its copy retain their exact saved archived task revisions. A single
pose has no cross-pose sharing claim. Batch tables and the complete comparison
image live under `B/pose2+9+13+15+17/step3_scheculer/dsl_shared/`.

## Run

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python slides/DSL_algo/step3_scheculer/run_dsl.py \
  --include-independent --jobs 4 --device cuda --steps 8 --samples 24 --seeds 32
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python slides/DSL_algo/step4_connect_support/try_dsl_growth.py \
  --include-independent --jobs 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python slides/DSL_algo/step3_scheculer/review_shared_dsl.py
PYTHONPATH=slides/DSL_algo .venv/bin/python -m unittest \
  step3_scheculer.test_shared_dsl step3_scheculer.test_contact_dsl \
  step3_scheculer.test_exit_options step3_scheculer.test_construction_contract
```

`review_shared_dsl.py` checks the actual rigid contact/solid identities, freshly
rebuilds force rays for all original loads, checks provenance and verifies that
published meshes are byte-identical to their accepted construction witnesses.
It does not repeat complete exported-body geometry acceptance. The old
`review_dsl.py`, `summarize_dsl.py` and GPU diagnostic outputs belong to V1–V3.

## Completed V4 B batch

All 30 cases were rerun on CUDA with 8 numerical descent steps, 24 base proposal
loads, 32 surface seeds and an adaptive proposal cap of 128 loads. All 52
force-only task checkpoints pass their full 32768 original loads. One of ten
combinations and seventeen of twenty singles construct successfully. The failed
singles are pose2, pose15 and pose20. These finite-search results do not establish
infeasibility.

`pose3+6` reduces eight independently saved seed heads to **three physical heads
shared by both poses**. Installed solid views agree within 5.56e-17 m. Its
selected exit directions differ by 4.465 degrees in common object/fixture
coordinates; this agreement is a soft optimization result, not a requirement.
The constructed support has 20.011 cm³ material and 253.418 cm² workstation XY
footprint. All eighteen Step3 successes publish byte-identical Step4 witnesses.
Thirty-two grammar, sharing, force, CUDA, exit and construction-contract tests
pass.

The accepted shared placement is restricted to a fixed object–fixture
relationship. This improves head reuse but rejects nine combinations in the
current finite search, through exit restrictions or ground coverage. Arbitrary
regrasp registration and a lighter constructive DSL certificate remain absent.

The complete fresh force/sharing audit and report are
[experiment_summary.md](output/B/pose2+9+13+15+17/step3_scheculer/dsl_shared/experiment_summary.md),
with [all_cases.png](output/B/pose2+9+13+15+17/step3_scheculer/dsl_shared/all_cases.png).
The three shared contact patches and availability matrix are
[shared_heads.png](output/B/pose3+6/step3_scheculer/dsl_shared/shared_heads.png).
