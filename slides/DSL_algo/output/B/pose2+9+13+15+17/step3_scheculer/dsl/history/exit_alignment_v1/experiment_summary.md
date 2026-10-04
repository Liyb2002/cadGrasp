# DSL contact search: complete B trial

Step3: 5/10 combinations; 18/20 independent poses.
Force-only phase: 52/52 tasks pass all original loads, before exit repair.
Accepted combinations: 41 → 31 total physical heads.
Accepted independent poses: 63 → 44 heads.
Step4 fixed-placement trials: 1/10 accepted new fixtures.

Each task retains all 32768 original loads. Passed requires sampled force/moment balance, shared no-uplift, real work/floor contact geometry and a certified continuous horizontal exit. No global head-count optimum or compactness guarantee. Historical pose1+3 and its copy use their own saved pose revision. Independent poses are separate experiments, not a twenty-pose fixture.

| Case | Step3 | Heads before → after | Per-pose heads | Coverage | Object exit angle before → after | Step4 |
|---|---|---|---|---|---|---|
| pose1+2+3+4+5 | FAIL | 20 → 15 | [4, 4, 2, 2, 3] | [32768, 32768, 32768, 32768, 32768] | 75.1° → uncertified | False |
| pose1+2+8+17 | FAIL | 14 → 13 | [4, 5, 2, 2] | [32768, 32768, 32768, 32768] | 105.4° → uncertified | False |
| pose1+3 | PASS | 6 → 5 | [3, 2] | [32768, 32768] | 18.0° → 1.0° | False |
| pose1+3copied | PASS | 6 → 5 | [3, 2] | [32768, 32768] | 18.0° → 1.0° | False |
| pose2+10+15 | FAIL | 11 → 10 | [5, 3, 2] | [32768, 32768, 32768] | 90.2° → uncertified | False |
| pose2+12+15 | FAIL | 12 → 9 | [5, 2, 2] | [32768, 32768, 32768] | 103.3° → uncertified | False |
| pose2+9+13+15+17 | FAIL | 18 → 13 | [4, 2, 2, 2, 3] | [32768, 32768, 32768, 32768, 32768] | 92.8° → uncertified | False |
| pose3+6 | PASS | 8 → 5 | [2, 3] | [32768, 32768] | 56.6° → 56.6° | False |
| pose5+7 | PASS | 7 → 5 | [3, 2] | [32768, 32768] | 69.7° → 56.6° | True |
| pose6+8+10+19 | PASS | 14 → 11 | [3, 2, 3, 3] | [32768, 32768, 32768, 32768] | 76.2° → 76.2° | False |
| pose_1 | PASS | 4 → 4 | [4] | [32768] | — (single pose) | not run |
| pose_10 | PASS | 3 → 3 | [3] | [32768] | — (single pose) | not run |
| pose_11 | PASS | 4 → 3 | [3] | [32768] | — (single pose) | not run |
| pose_12 | PASS | 4 → 2 | [2] | [32768] | — (single pose) | not run |
| pose_13 | PASS | 3 → 2 | [2] | [32768] | — (single pose) | not run |
| pose_14 | PASS | 3 → 2 | [2] | [32768] | — (single pose) | not run |
| pose_15 | PASS | 4 → 2 | [2] | [32768] | — (single pose) | not run |
| pose_16 | PASS | 3 → 2 | [2] | [32768] | — (single pose) | not run |
| pose_17 | PASS | 3 → 2 | [2] | [32768] | — (single pose) | not run |
| pose_18 | PASS | 4 → 3 | [3] | [32768] | — (single pose) | not run |
| pose_19 | PASS | 4 → 3 | [3] | [32768] | — (single pose) | not run |
| pose_2 | FAIL | 4 → 4 | [4] | [32768] | — (single pose) | not run |
| pose_20 | FAIL | 4 → 5 | [5] | [32768] | — (single pose) | not run |
| pose_3 | PASS | 4 → 2 | [2] | [32768] | — (single pose) | not run |
| pose_4 | PASS | 3 → 2 | [2] | [32768] | — (single pose) | not run |
| pose_5 | PASS | 4 → 3 | [3] | [32768] | — (single pose) | not run |
| pose_6 | PASS | 4 → 3 | [3] | [32768] | — (single pose) | not run |
| pose_7 | PASS | 3 → 2 | [2] | [32768] | — (single pose) | not run |
| pose_8 | PASS | 3 → 2 | [2] | [32768] | — (single pose) | not run |
| pose_9 | PASS | 3 → 2 | [2] | [32768] | — (single pose) | not run |

## Limits

The loss uses a subset of real wrench rays and original sampled loads to propose steps; final acceptance uses the full contact surfaces and all original loads. CUDA batches float64 active-set NNLS residuals; CPU LP decides acceptance. Gradient steps can improve the proposal loss without improving exact coverage, so exact checks decide accepted deletions. Force-only contacts and coverage are saved separately; failed exit repair cannot discard a previously force-feasible checkpoint. Radius bounds are numerical search limits, not pressure or strength certification. Exit probes have 0.2 mm constructor thickness; baseline probes used a larger thickness.

Step4 trials retain old placements and foot targets while using NEW DSL contacts and exits. Failure can arise from inactive material, floors or connector clearance; it does not prove no new placement exists. Copied baseline models remain comparison artifacts and are not new DSL results.

## GPU check

CUDA float64 active-set NNLS: 27 real patch perturbations/deletions × 24 original loads, maximum CPU/GPU residual difference 4.16e-16; kernel-only speedup 236.6×. This excludes mesh clipping, final LP and initialization.
CPU fallback batches across all task searches: 0.

Constructed pose5+7: material 40.159 → 37.854 cm³; actual workstation XY footprint 440.611 → 440.611 cm². Material reduction and footprint reduction are distinct measurements.

## Independent replay and input snapshot

Independent artifact audit passed: True. Current workspace baseline equals copy-time snapshot: False.
This experiment uses its copied inputs. A separate workspace job modified the source baseline during the experiment; source snapshot drift is recorded separately and is not a failed design verdict.
