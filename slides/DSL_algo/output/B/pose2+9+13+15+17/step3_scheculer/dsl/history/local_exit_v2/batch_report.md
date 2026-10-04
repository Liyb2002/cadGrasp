# Contact DSL descent: saved B cases

Numerical real-mesh parameter descent and discrete rewrites. All 32768 original loads per task are checked. Head-count minimality and compactness are not proved.

| Case | Passed | Seed → final heads | Per-pose coverage | Verified exit options | Seconds |
|---|---|---|---|---|---|
| pose1+2+3+4+5 | False | 20 → 16 | [32768, 32768, 32768, 32768, 32768] | [1, 0, 2, 1, 1] | 145.4 |
| pose1+2+8+17 | False | 14 → 12 | [32768, 32768, 32768, 32768] | [1, 0, 1, 3] | 98.3 |
| pose1+3 | True | 6 → 5 | [32768, 32768] | [2, 4] | 31.0 |
| pose1+3copied | True | 6 → 5 | [32768, 32768] | [2, 4] | 31.2 |
| pose2+10+15 | False | 11 → 9 | [32768, 32768, 32768] | [0, 1, 1] | 84.8 |
| pose2+12+15 | False | 12 → 8 | [32768, 32768, 32768] | [0, 1, 1] | 101.3 |
| pose2+9+13+15+17 | False | 18 → 13 | [32768, 32768, 32768, 32768, 32768] | [0, 1, 1, 1, 1] | 82.2 |
| pose3+6 | True | 8 → 6 | [32768, 32768] | [2, 2] | 97.4 |
| pose5+7 | True | 7 → 5 | [32768, 32768] | [1, 1] | 21.7 |
| pose6+8+10+19 | True | 14 → 11 | [32768, 32768, 32768, 32768] | [2, 1, 1, 1] | 65.7 |
| pose_1 | True | 4 → 4 | [32768] | [1] | 16.8 |
| pose_10 | True | 3 → 3 | [32768] | [1] | 8.1 |
| pose_11 | True | 4 → 3 | [32768] | [1] | 17.2 |
| pose_12 | True | 4 → 2 | [32768] | [1] | 39.6 |
| pose_13 | True | 3 → 2 | [32768] | [1] | 9.3 |
| pose_14 | True | 3 → 2 | [32768] | [2] | 8.9 |
| pose_15 | True | 4 → 2 | [32768] | [1] | 15.3 |
| pose_16 | True | 3 → 2 | [32768] | [1] | 6.5 |
| pose_17 | True | 3 → 2 | [32768] | [3] | 26.9 |
| pose_18 | True | 4 → 3 | [32768] | [1] | 16.5 |
| pose_19 | True | 4 → 3 | [32768] | [1] | 31.4 |
| pose_2 | False | 4 → 4 | [32768] | [0] | 65.9 |
| pose_20 | False | 4 → 5 | [32768] | [0] | 40.3 |
| pose_3 | True | 4 → 3 | [32768] | [2] | 83.5 |
| pose_4 | True | 3 → 2 | [32768] | [1] | 10.5 |
| pose_5 | True | 4 → 3 | [32768] | [1] | 21.6 |
| pose_6 | True | 4 → 3 | [32768] | [2] | 13.4 |
| pose_7 | True | 3 → 2 | [32768] | [1] | 13.7 |
| pose_8 | True | 3 → 2 | [32768] | [1] | 12.0 |
| pose_9 | True | 3 → 2 | [32768] | [1] | 10.0 |
