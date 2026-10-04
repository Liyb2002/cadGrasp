# B common-exit/head co-descent experiment

Nine existing full feasible supports are initial incumbents. Original native object poses and all 32768 original loads per pose remain fixed. Directions are continuous unit vectors, checked by continuous geometry for each proposal rather than looked up in a ray menu. Actual head centers and radii move on admissible real surfaces; source-face normals and torque generators are recompiled. Fixture seating can move, and the original complete support is retained until a full feasible improving candidate replaces it.

Objective: actual Step5 occupied XYZ box; numerical ties use actual support material volume. This is derivative-free local stochastic pattern search inspired by D4Descent, not automatic differentiation through the greedy constructor.

| Set | Final world object exit XYZ | Heads per pose | Occupied volume reduction | Seconds |
|---|---|---|---:|---:|
| pose1+3 | [0.016384, 0.999479, 0.027817] | [6, 6] | 1.481% | 10.03 |
| pose1+2+3+4+5 | [-0.028323, 0.058003, 0.997915] | [3, 4, 3, 5, 5] | 6.353% | 113.61 |
| pose1+2+8+17 | [0.014823, 0.062383, 0.997942] | [3, 4, 3, 4] | 3.248% | 48.27 |
| pose2+10+15 | [-0.027514, -0.004628, 0.999611] | [4, 4, 3] | 1.082% | 24.43 |
| pose2+12+15 | [-0.016663, 0.103006, 0.994541] | [4, 5, 2] | 2.989% | 51.58 |
| pose2+9+13+15+17 | [-0.022676, 0.051735, 0.998403] | [4, 3, 4, 3, 4] | 6.180% | 77.27 |
| pose3+6 | [0.039048, 0.978969, 0.200238] | [6, 5] | 3.805% | 56.40 |
| pose5+7 | [-0.012087, -0.024785, 0.99962] | [5, 5] | 7.393% | 39.88 |
| pose6+8+10+19 | [-0.0, -0.0, 1.0] | [4, 3, 4, 4] | 0.000% | 5.35 |

9/9 remain fully feasible; 8/9 improve. 103 joint proposals, 46 complete feasible proposal constructions, 42 accepted updates including 8 topology edits followed by joint local steps. Every accepted event changes both the actual contact geometry and common exit. The unchanged group preserves the exact initial OBJ and placement; its finite unsuccessful proposals are not an impossibility proof.

Main experiment program wall time 236.237 s: one pilot group then eight groups with two worker processes. Per-group accumulated optimization/construction/initial-image time 426.823 s. Two matched frozen-initial-direction controls take 75.591 s separately; later presentation refresh is excluded.

## Matched-budget frozen-direction controls

| Set | Joint reduction from initial | Frozen-direction reduction from initial | Joint extra reduction versus frozen |
|---|---:|---:|---:|
| pose5+7 | 7.393% | 7.128% | 0.286% |
| pose3+6 | 3.805% | 2.889% | 0.943% |

These two single-seed controls also optimize heads and seating. They show much of the current gain comes from those variables; allowing the direction to change provides a smaller additional improvement in these runs. They do not establish global superiority or an optimum.

[All Step4/Step5 images and models](gallery.md). Fifteen direction, moment/cache, continuous-path and Step5 regressions pass.
