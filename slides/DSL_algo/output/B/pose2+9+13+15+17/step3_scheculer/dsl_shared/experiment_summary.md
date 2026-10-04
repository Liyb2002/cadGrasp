# Joint shared-head DSL: B experiment

Combinations: 1/10 constructed with genuinely shared physical heads.
Independent poses: 17/20 constructed.
Force-only checkpoints: 52/52 tasks, each 32768 original loads.
Accepted combinations: 8 saved independent heads -> 3 physical heads, of which 3 serve multiple poses.
Accepted joint gradient updates: 710.

The physical head count is measured once in the canonical fixture frame. Per-pose views of the same contact triangles and finite-depth solids must coincide when installed. All heads are available in every pose; equilibrium may assign zero force to any head. Work faces from all tasks are excluded. Step4 publishes the identical Step3 construction witness.

| Case | Constructed | Independent seed heads | Physical heads | Shared heads | Coverage |
|---|---|---|---|---|---|
| pose1+2+3+4+5 | False | 20 | 20 | 20 | [32768, 32768, 32768, 32768, 32768] |
| pose1+2+8+17 | False | 14 | 14 | 14 | [32768, 32768, 32768, 32768] |
| pose1+3 | False | 6 | 5 | 5 | [32768, 32768] |
| pose1+3copied | False | 6 | 5 | 5 | [32768, 32768] |
| pose2+10+15 | False | 11 | 11 | 11 | [32768, 32768, 32768] |
| pose2+12+15 | False | 12 | 12 | 12 | [32768, 32768, 32768] |
| pose2+9+13+15+17 | False | 18 | 18 | 18 | [32768, 32768, 32768, 32768, 32768] |
| pose3+6 | True | 8 | 3 | 3 | [32768, 32768] |
| pose5+7 | False | 7 | 7 | 7 | [32768, 32768] |
| pose6+8+10+19 | False | 14 | 14 | 14 | [32768, 32768, 32768, 32768] |
| pose_1 | True | 4 | 4 | 0 | [32768] |
| pose_10 | True | 3 | 3 | 0 | [32768] |
| pose_11 | True | 4 | 3 | 0 | [32768] |
| pose_12 | True | 4 | 2 | 0 | [32768] |
| pose_13 | True | 3 | 2 | 0 | [32768] |
| pose_14 | True | 3 | 2 | 0 | [32768] |
| pose_15 | False | 4 | 4 | 0 | [32768] |
| pose_16 | True | 3 | 2 | 0 | [32768] |
| pose_17 | True | 3 | 2 | 0 | [32768] |
| pose_18 | True | 4 | 3 | 0 | [32768] |
| pose_19 | True | 4 | 3 | 0 | [32768] |
| pose_2 | False | 4 | 4 | 0 | [32768] |
| pose_20 | False | 4 | 4 | 0 | [32768] |
| pose_3 | True | 4 | 3 | 0 | [32768] |
| pose_4 | True | 3 | 2 | 0 | [32768] |
| pose_5 | True | 4 | 3 | 0 | [32768] |
| pose_6 | True | 4 | 3 | 0 | [32768] |
| pose_7 | True | 3 | 2 | 0 | [32768] |
| pose_8 | True | 3 | 2 | 0 | [32768] |
| pose_9 | True | 3 | 2 | 0 | [32768] |

## Scope and limits

- object-attached shared placement family, not arbitrary regrasp registration
- finite numerical descent and rewrites, not global head minimum
- full construction is still performed in Step3 for its acceptance contract
- direction closeness uses normal-opening proxy; exact exit-space union uses discrete path beam search

Old independent DSL experiments remain in `dsl/` and `dsl_support/`. Active shared results are in `dsl_shared/` and `dsl_shared_support/`.
