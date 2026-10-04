# Contact DSL descent: saved B cases

Numerical real-mesh parameter descent and discrete rewrites. All 32768 original loads per task are checked. Head-count minimality and compactness are not proved.

| Case | Passed | Seed → final heads | Per-pose coverage | Verified exit options | Seconds |
|---|---|---|---|---|---|
| pose1+2+3+4+5 | False | 20 → 16 | [32768, 32768, 32768, 32768, 32768] | [1, 0, 2, 1, 1] | 125.6 |
| pose1+2+8+17 | False | 14 → 12 | [32768, 32768, 32768, 32768] | [1, 0, 1, 3] | 75.6 |
| pose1+3 | True | 6 → 5 | [32768, 32768] | [2, 4] | 159.7 |
| pose1+3copied | True | 6 → 5 | [32768, 32768] | [2, 4] | 160.2 |
| pose2+10+15 | False | 11 → 9 | [32768, 32768, 32768] | [0, 1, 1] | 61.2 |
| pose2+12+15 | False | 12 → 8 | [32768, 32768, 32768] | [0, 1, 1] | 61.2 |
| pose2+9+13+15+17 | False | 18 → 13 | [32768, 32768, 32768, 32768, 32768] | [0, 1, 1, 1, 1] | 67.6 |
| pose3+6 | True | 8 → 6 | [32768, 32768] | [2, 2] | 270.6 |
| pose5+7 | True | 7 → 5 | [32768, 32768] | [1, 1] | 56.0 |
| pose6+8+10+19 | True | 14 → 11 | [32768, 32768, 32768, 32768] | [2, 1, 1, 1] | 466.1 |
| pose_1 | True | 4 → 4 | [32768] | [1] | 52.1 |
| pose_10 | True | 3 → 3 | [32768] | [1] | 76.3 |
| pose_11 | True | 4 → 3 | [32768] | [1] | 47.5 |
| pose_12 | True | 4 → 2 | [32768] | [1] | 62.4 |
| pose_13 | True | 3 → 2 | [32768] | [1] | 41.6 |
| pose_14 | True | 3 → 2 | [32768] | [2] | 74.6 |
| pose_15 | True | 4 → 2 | [32768] | [1] | 46.8 |
| pose_16 | True | 3 → 2 | [32768] | [1] | 36.9 |
| pose_17 | True | 3 → 2 | [32768] | [3] | 159.5 |
| pose_18 | True | 4 → 3 | [32768] | [1] | 78.4 |
| pose_19 | True | 4 → 3 | [32768] | [1] | 60.6 |
| pose_2 | False | 4 → 4 | [32768] | [0] | 48.9 |
| pose_20 | False | 4 → 5 | [32768] | [0] | 32.2 |
| pose_3 | True | 4 → 3 | [32768] | [2] | 154.0 |
| pose_4 | True | 3 → 2 | [32768] | [1] | 50.9 |
| pose_5 | True | 4 → 3 | [32768] | [1] | 42.5 |
| pose_6 | True | 4 → 3 | [32768] | [2] | 72.8 |
| pose_7 | True | 3 → 2 | [32768] | [1] | 40.4 |
| pose_8 | True | 3 → 2 | [32768] | [1] | 37.0 |
| pose_9 | True | 3 → 2 | [32768] | [1] | 42.2 |
