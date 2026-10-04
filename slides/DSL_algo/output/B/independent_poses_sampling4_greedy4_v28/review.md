# Four sampling heads then four greedy heads

Single-pose contacts, all original loads, cached individual continuous head exits and common ports. No full support constructed.

| Object | Passed / total | Mean seconds | Failed poses |
|---|---:|---:|---|
| A1-f | 30 / 30 | 0.87 | — |
| A1-s | 30 / 30 | 33.77 | — |
| A2 | 30 / 30 | 17.66 | — |
| A3 | 30 / 30 | 3.20 | — |
| A4 | 24 / 30 | 62.28 | pose_4, pose_8, pose_15, pose_19, pose_20, pose_26 |
| A5 | 29 / 30 | 4.98 | pose_29 |
| B | 30 / 30 | 0.55 | — |
| C1 | 29 / 30 | 2.18 | pose_15 |
| C2 | 30 / 30 | 1.97 | — |
| C3 | 30 / 30 | 1.94 | — |
| C4 | 30 / 30 | 1.85 | — |
| C5 | 29 / 30 | 9.89 | pose_26 |
| C6 | 30 / 30 | 0.80 | — |
| C7 | 24 / 30 | 30.84 | pose_4, pose_12, pose_14, pose_15, pose_24, pose_30 |
| C8 | 30 / 30 | 1.33 | — |
| D1 | 30 / 30 | 1.21 | — |
| D2 | 30 / 30 | 0.59 | — |
| D3 | 30 / 30 | 0.85 | — |
| D4 | 28 / 30 | 17.60 | pose_9, pose_27 |
| D8 | 30 / 30 | 1.77 | — |
| cuboid_baseline | 30 / 30 | 0.25 | — |

Total: 613 / 630.
Thirty finite random trajectories found no accepted solution; this does not prove infeasibility.


## Comparison

{
  "original_passed": 613,
  "hybrid_passed": 613,
  "rescued": [],
  "regressed": [],
  "common_passed_head_changes": {
    "fewer": 5,
    "same": 577,
    "more": 31
  },
  "original_mean_passed_heads": 3.6721044045677,
  "hybrid_mean_passed_heads": 3.789559543230016
}

Total measured experiment wall seconds: 1698.491.
Reuses identical saved first-four sampling prefixes; wall time includes both interrupted runs and final bounded retry, not cold full search.
The final A4/pose_19 uses v29 with a one-second limit per ordinary/recovery LP.
Timed-out candidates remain numerically unresolved. This is recorded in the
per-case numerical evidence, not treated as an infeasibility proof. Three
regression tests and the complete 630-case audit pass.
