# DSL contact search: complete B trial

Step3: 5/10 combinations; 18/20 independent poses.
Force-only phase: 52/52 tasks pass all original loads, before exit repair.
Accepted combinations: 41 → 32 total physical heads.
Accepted independent poses: 63 → 45 heads.
Step4 fixed-placement trials: 5/10 accepted new fixtures.

Each task retains all 32768 original loads. Passed requires sampled force/moment balance, shared no-uplift, real work/floor contact geometry and a certified continuous three-dimensional translation path. No global head-count optimum or compactness guarantee. Historical pose1+3 and its copy use their own saved pose revision. Independent poses are separate experiments, not a twenty-pose fixture. V2 changes both exit selection and adaptive ground construction; this is not an isolated ablation of angular alignment.

| Case | Step3 | Heads before → after | Per-pose heads | Coverage | Verified exit options per pose | Step4 |
|---|---|---|---|---|---|---|
| pose1+2+3+4+5 | FAIL | 20 → 16 | [4, 4, 3, 2, 3] | [32768, 32768, 32768, 32768, 32768] | [1, 0, 2, 1, 1] | False |
| pose1+2+8+17 | FAIL | 14 → 12 | [4, 4, 2, 2] | [32768, 32768, 32768, 32768] | [1, 0, 1, 3] | False |
| pose1+3 | PASS | 6 → 5 | [3, 2] | [32768, 32768] | [2, 4] | True |
| pose1+3copied | PASS | 6 → 5 | [3, 2] | [32768, 32768] | [2, 4] | True |
| pose2+10+15 | FAIL | 11 → 9 | [4, 3, 2] | [32768, 32768, 32768] | [0, 1, 1] | False |
| pose2+12+15 | FAIL | 12 → 8 | [4, 2, 2] | [32768, 32768, 32768] | [0, 1, 1] | False |
| pose2+9+13+15+17 | FAIL | 18 → 13 | [4, 2, 2, 2, 3] | [32768, 32768, 32768, 32768, 32768] | [0, 1, 1, 1, 1] | False |
| pose3+6 | PASS | 8 → 6 | [3, 3] | [32768, 32768] | [2, 2] | True |
| pose5+7 | PASS | 7 → 5 | [3, 2] | [32768, 32768] | [1, 1] | True |
| pose6+8+10+19 | PASS | 14 → 11 | [3, 2, 3, 3] | [32768, 32768, 32768, 32768] | [2, 1, 1, 1] | True |
| pose_1 | PASS | 4 → 4 | [4] | [32768] | [1] | not run |
| pose_10 | PASS | 3 → 3 | [3] | [32768] | [1] | not run |
| pose_11 | PASS | 4 → 3 | [3] | [32768] | [1] | not run |
| pose_12 | PASS | 4 → 2 | [2] | [32768] | [1] | not run |
| pose_13 | PASS | 3 → 2 | [2] | [32768] | [1] | not run |
| pose_14 | PASS | 3 → 2 | [2] | [32768] | [2] | not run |
| pose_15 | PASS | 4 → 2 | [2] | [32768] | [1] | not run |
| pose_16 | PASS | 3 → 2 | [2] | [32768] | [1] | not run |
| pose_17 | PASS | 3 → 2 | [2] | [32768] | [3] | not run |
| pose_18 | PASS | 4 → 3 | [3] | [32768] | [1] | not run |
| pose_19 | PASS | 4 → 3 | [3] | [32768] | [1] | not run |
| pose_2 | FAIL | 4 → 4 | [4] | [32768] | [0] | not run |
| pose_20 | FAIL | 4 → 5 | [5] | [32768] | [0] | not run |
| pose_3 | PASS | 4 → 3 | [3] | [32768] | [2] | not run |
| pose_4 | PASS | 3 → 2 | [2] | [32768] | [1] | not run |
| pose_5 | PASS | 4 → 3 | [3] | [32768] | [1] | not run |
| pose_6 | PASS | 4 → 3 | [3] | [32768] | [2] | not run |
| pose_7 | PASS | 3 → 2 | [2] | [32768] | [1] | not run |
| pose_8 | PASS | 3 → 2 | [2] | [32768] | [1] | not run |
| pose_9 | PASS | 3 → 2 | [2] | [32768] | [1] | not run |

## Limits

The loss uses a subset of real wrench rays and original sampled loads to propose steps; final acceptance uses the full contact surfaces and all original loads. CUDA batches float64 active-set NNLS residuals; CPU LP decides acceptance. Gradient steps can improve the proposal loss without improving exact coverage, so exact checks decide accepted deletions. Force-only contacts and coverage are saved separately; failed exit repair cannot discard a previously force-feasible checkpoint. Radius bounds are numerical search limits, not pressure or strength certification. Exit probes have 0.2 mm thin search thickness; baseline probes used a larger thickness.

Step4 trials retain old placements and grow floor contacts adaptively while using NEW DSL contacts and exits. Failure can arise from inactive material, floors or connector clearance; it does not prove no new placement exists. Copied baseline models remain comparison artifacts and are not new DSL results.

## V1 versus multiple-path search

| Metric | V1 | V2 |
|---|---|---|
| group_passed_count | 5 | 5 |
| single_passed_count | 18 | 18 |
| constructed_group_count | 1 | 5 |
| accepted_group_heads_after | 31 | 32 |
| accepted_single_heads_after | 44 | 45 |

Failed task diagnostics are recorded in exit_diagnostics.json. Absence of a common initial translation opening for a contact set is not proof that no different contacts or rotational path exist.

## GPU check

CUDA float64 active-set NNLS: 27 real patch perturbations/deletions × 24 original loads, maximum CPU/GPU residual difference 4.16e-16; kernel-only speedup 86.8×. This excludes mesh clipping, final LP and initialization.
CPU fallback batches across all task searches: 3.

Constructed pose1+3: material 135.849 → 24.932 cm³; actual workstation XY footprint 427.879 → 352.164 cm². Material reduction and footprint reduction are distinct measurements.

Constructed pose1+3copied: material 33.645 → 24.932 cm³; actual workstation XY footprint 375.531 → 352.164 cm². Material reduction and footprint reduction are distinct measurements.

Constructed pose3+6: material 38.170 → 31.910 cm³; actual workstation XY footprint 295.150 → 301.402 cm². Material reduction and footprint reduction are distinct measurements.

Constructed pose5+7: material 40.159 → 26.088 cm³; actual workstation XY footprint 440.611 → 448.860 cm². Material reduction and footprint reduction are distinct measurements.

Constructed pose6+8+10+19: material 84.756 → 67.923 cm³; actual workstation XY footprint 1287.792 → 1019.554 cm². Material reduction and footprint reduction are distinct measurements.

## Independent replay and input snapshot

Independent artifact audit passed: True. Current workspace baseline equals copy-time snapshot: False.
This experiment uses its copied inputs. A separate workspace job modified the source baseline during the experiment; source snapshot drift is recorded separately and is not a failed design verdict.
