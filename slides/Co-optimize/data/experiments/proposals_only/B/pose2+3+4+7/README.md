# No-gradient_descent control

B/pose2+3+4+7, one chain, every round evaluates all 32 single-pose direction_choice candidates with incremental contact locks and all original loads. No gradient_descent call. Incumbent competes; rank is minimum pose feasible fraction, then summed feasible fractions. Seed 42, 5–30 degree tangent choices, 3 rounds / 96 candidates, same configuration as the force-descent trial. First round's 32 direction choices are exactly identical; subsequent chain states and directions differ as expected.

| Method | Runtime excluding constructor | Counts, pose 2 / 3 / 4 / 7 | Total passed |
| --- | --- | --- | --- |
| direction_choice only | 8.62 s | [1482, 30792, 1068, 31297] | 64639 / 131072 |
| direction_choice + force gradient_descent | 40.52 s | [3204, 32766, 31489, 32751] | 100210 / 131072 |

The no-descent control is 4.70× faster but worse under both minimum-pose and total feasible counts at this SAME proposal budget. Both remain unresolved. This single-seed comparison does not establish an equal-wall-time advantage or convergence guarantee. These are finite seed-point contact-model checks, not complete support/1% clearance acceptance.

Artifacts: data/report.json, data/chain_trajectory.json, data/contact_locks.npz, data/comparison.json, data/run.log. Main entry: --search proposals-only --candidates 32 --max-proposals 96 --seed 42. Original outputs and upstream inputs are preserved. 80 tests pass, including proof that this mode never calls gradient_descent and performs exactly one contact check per raw proposal.
