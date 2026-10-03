# Twenty-state pilot

Completed: 10 groups, 200 states, 126919 state/action records.

All finite values refer to stored successful contact completions with all 32,768 original loads accepted per pose. The record audit checks state/action retention, common directions/paths, object-frame direction costs and the head-count formula. No support solid was constructed.

Null budget-unresolved values are unknown, not proof of infeasibility. Scores are best-found feasible costs. Direction search is exact over the discovered bank for two poses and approximate for larger groups.

| Group | States | Successful labels | Unknown | No path | Step2 rejected | State compute (s) |
|---|---:|---:|---:|---:|---:|---:|
| [pose1+3](../pose1+3/pilot20/summary.json) | 20 | 1873 | 436 | 1578 | 4040 | 676.6 |
| [pose1+3copied](../pose1+3copied/pilot20/summary.json) | 20 | 1391 | 336 | 2140 | 4040 | 502.3 |
| [pose3+6](../pose3+6/pilot20/summary.json) | 20 | 1852 | 280 | 1599 | 4200 | 696.0 |
| [pose5+7](../pose5+7/pilot20/summary.json) | 20 | 2125 | 188 | 1891 | 3740 | 761.1 |
| [pose2+10+15](../pose2+10+15/pilot20/summary.json) | 20 | 2884 | 392 | 3269 | 5340 | 443.7 |
| [pose2+12+15](../pose2+12+15/pilot20/summary.json) | 20 | 2970 | 664 | 2862 | 5400 | 395.2 |
| [pose1+2+8+17](../pose1+2+8+17/pilot20/summary.json) | 20 | 3293 | 773 | 4397 | 7400 | 455.9 |
| [pose6+8+10+19](../pose6+8+10+19/pilot20/summary.json) | 20 | 4217 | 649 | 3248 | 7780 | 497.0 |
| [pose1+2+3+4+5](../pose1+2+3+4+5/pilot20/summary.json) | 20 | 4728 | 809 | 4706 | 9580 | 421.6 |
| [pose2+9+13+15+17](../pose2+9+13+15+17/pilot20/summary.json) | 20 | 5535 | 557 | 4917 | 8840 | 552.3 |

Each group has `states.json`, twenty `state_NNN.json` files, and `training_records.jsonl` in its `pilot20/` directory.

`pose1+3copied` repeats the current pose identities of `pose1+3`, with different sampled states. All inputs come from current independent_poses; historical fixture placements are not used.
