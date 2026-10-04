# Feasible-incumbent DSL: B combinations

Complete feasible fixtures: 10/10.

Each accepted update preserves full force coverage and its complete construction witness. Different exits and independent head subsets remain allowed. Shared heads count actual coincident geometry, not duplicated labels.

| Group | Initialized heads | Final physical heads | Shared heads | Object-frame exit angle (initial → final) | Feasible updates |
|---|---:|---:|---:|---:|---:|
| pose1+2+3+4+5 | 14 | 13 | 0 | 89.6° → 89.0° | 2 |
| pose1+2+8+17 | 10 | 9 | 0 | 86.1° → 83.3° | 3 |
| pose1+3 | 6 | 4 | 0 | 18.0° → 18.0° | 2 |
| pose1+3copied | 6 | 4 | 0 | 18.0° → 18.0° | 2 |
| pose2+10+15 | 7 | 7 | 0 | 114.9° → 101.0° | 2 |
| pose2+12+15 | 8 | 6 | 0 | 119.4° → 114.5° | 3 |
| pose2+9+13+15+17 | 11 | 11 | 0 | 96.3° → 93.4° | 2 |
| pose3+6 | 3 | 3 | 3 | 4.5° → 2.5° | 2 |
| pose5+7 | 5 | 4 | 0 | 67.5° → 67.5° | 1 |
| pose6+8+10+19 | 11 | 11 | 0 | 76.2° → 76.2° | 0 |

Initialization uses the qualified V5 final state; every pose retains the 32768/32768 acceptance threshold. New contact areas are bounded by 2% of the actual object surface area. A finite search does not establish minimum head count; an unchanged feasible incumbent is a valid result. Direction similarity is a proxy and does not certify reduced final footprint.

This completed experiment optimizes the previous object-frame angle objective. It does not measure the newly proposed common fixture-frame withdrawal direction. No additional cross-pose shared heads were created. All ten public Step4 models were regenerated; 19 unit tests passed.
