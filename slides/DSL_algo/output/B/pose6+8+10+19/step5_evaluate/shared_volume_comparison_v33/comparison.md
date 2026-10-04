# B common world exit direction and actual Step5 comparison

Run the nine distinct historical experiment combinations against immutable CURRENT pose data. Exclude pose1+3copied duplicate. Historical fixtures/pose revisions are not comparison inputs. Object poses and fixture registration are fixed in native world XYZ (identity basis, zero offset).

Step5 volume means the aggregate XYZ AABB of all fixed objects plus installed complete support, not the exit-sweep proxy or head-only material.

| Pose set | Selected object exit XYZ | Heads per pose | Search seconds | Full construction |
|---|---|---|---:|---|
| pose1+3 | [0.0, 1.0, -0.0] | [7, 6] | 105.41 | Rejected before body growth |
| pose1+2+3+4+5 | [-0.0, -0.0, 1.0] | [3, 4, 3, 4, 5] | 78.10 | Rejected before body growth |
| pose1+2+8+17 | [-0.0, -0.0, 1.0] | [3, 4, 3, 4] | 53.83 | Rejected before body growth |
| pose2+10+15 | [-0.0, -0.0, 1.0] | [4, 4, 3] | 53.14 | Rejected before body growth |
| pose2+12+15 | [-0.0, -0.0, 1.0] | [4, 4, 3] | 29.46 | Rejected before body growth |
| pose2+9+13+15+17 | [-0.0, -0.0, 1.0] | [4, 3, 4, 3, 4] | 80.43 | Rejected before body growth |
| pose3+6 | [0.0, 1.0, -0.0] | [6, 4] | 100.64 | Rejected before body growth |
| pose5+7 | [-0.0, -0.0, 1.0] | [5, 4] | 27.77 | Rejected before body growth |
| pose6+8+10+19 | [-0.0, -0.0, 1.0] | [4, 3, 4, 4] | 80.80 | Rejected before body growth |

Contact programs: 9/9 pass all 32768 original loads per pose, shared no-uplift equation, and own-head continuous exit. Directions: 2 groups +Y, 7 groups +Z.

Complete fixtures: 0/9 for selected directions and 0/9 for +Z. 64 total full-construction attempts, including all distinct locally feasible direction alternatives, reject before body growth. Foreign heads intersect another pose's working surface or its continuous object-exit sweep. No Step5 volume exists for these rejected cases; do not interpret missing volume as zero, larger volume, or global infeasibility.

The active search minimizes a local exit-sweep AABB proxy, not the final Step5 occupied box. It also filters own-pose heads only. Whole-fixture cross-pose root/work checks must be part of head/direction selection before this can be used as a complete support optimizer.

Timing: search sum 609.577 s; comparison worker sum 151.375 s (overlaps search); wall time through comparison/audit 692.151 s. Twelve direction/path regression tests pass.

This comparison uses the new fixed native placements. Old accepted B Step5 metrics were generated from earlier poses and optimized fixture registrations and cannot be used as a fixed +Z baseline.
