# Feasible-incumbent DSL: B combinations

Complete feasible fixtures: 10/10.

Each accepted update preserves full force coverage and its complete construction witness. Different exits and independent head subsets remain allowed. Shared heads count actual coincident geometry, not duplicated labels.

| Group | Initialized heads | Final physical heads | Shared heads | Exit angle (initial → final) | Feasible updates |
|---|---:|---:|---:|---:|---:|
| pose1+2+3+4+5 | 19 | 14 | 0 | 92.0° → 89.6° | 8 |
| pose1+2+8+17 | 13 | 10 | 0 | 89.2° → 86.1° | 5 |
| pose1+3 | 6 | 6 | 0 | 18.0° → 18.0° | 0 |
| pose1+3copied | 6 | 6 | 0 | 18.0° → 18.0° | 0 |
| pose2+10+15 | 10 | 7 | 0 | 114.9° → 114.9° | 3 |
| pose2+12+15 | 11 | 8 | 0 | 119.4° → 119.4° | 3 |
| pose2+9+13+15+17 | 17 | 11 | 0 | 97.5° → 96.3° | 7 |
| pose3+6 | 8 | 3 | 3 | 56.6° → 4.5° | 4 |
| pose5+7 | 7 | 5 | 0 | 69.7° → 67.5° | 3 |
| pose6+8+10+19 | 14 | 11 | 0 | 76.2° → 76.2° | 3 |

Original pose2 inputs were incomplete; initialization repairs keep the 32768/32768 acceptance threshold. A finite search does not establish minimum head count; an unchanged feasible incumbent is a valid result. Direction similarity is a proxy and does not certify reduced final footprint.
