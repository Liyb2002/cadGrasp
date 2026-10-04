# Absolute direction: actual Step4 / Step5 results

The search uses common workstation +Z object withdrawal, jointly chooses XY seating/contact compatibility, and assigns no head-count cost. Reported volume is the unchanged Step5 aggregate XYZ bounding-box volume, not support material volume.

| Set | Previous occupied cm3 | New occupied cm3 | Change | Status |
|---|---:|---:|---:|---|
| pose1+3 | 6269.3 | 6330.0 | +1.0% | [passed](output/B/pose1+3/step4/overview.png) |
| pose1+3copied | 5651.0 | 6330.0 | +12.0% | [passed](output/B/pose1+3copied/step4/overview.png) |
| pose3+6 | 7405.8 | 5484.2 | -25.9% | [passed](output/B/pose3+6/step4/overview.png) |
| pose5+7 | 6089.9 | 4651.6 | -23.6% | [passed](output/B/pose5+7/step4/overview.png) |
| pose2+10+15 | 20295.8 | 9283.6 | -54.3% | [passed](output/B/pose2+10+15/step4/overview.png) |
| pose2+12+15 | 19060.2 | 11562.4 | -39.3% | [passed](output/B/pose2+12+15/step4/overview.png) |
| pose1+2+8+17 | 30175.4 | 14699.7 | -51.3% | [passed](output/B/pose1+2+8+17/step4/overview.png) |
| pose6+8+10+19 | 31168.2 | 16325.8 | -47.6% | [passed](output/B/pose6+8+10+19/step4/overview.png) |
| pose1+2+3+4+5 | 49307.1 | 24272.3 | -50.8% | [passed](output/B/pose1+2+3+4+5/step4/overview.png) |
| pose2+9+13+15+17 | 59454.5 | 33701.9 | -43.3% | [passed](output/B/pose2+9+13+15+17/step4/overview.png) |
| pose1+7 | — | 4657.1 | — | [passed](output/B/pose1+7/step4/overview.png) |
| pose2+4+12 | — | 13580.2 | — | [passed](output/B/pose2+4+12/step4/overview.png) |
| pose3+6+9+13 | — | 20671.5 | — | [passed](output/B/pose3+6+9+13/step4/overview.png) |
| pose5+8+10+15+17 | — | 29628.1 | — | [passed](output/B/pose5+8+10+15+17/step4/overview.png) |
| pose1+2+7+12+19 | — | 26328.8 | — | [passed](output/B/pose1+2+7+12+19/step4/overview.png) |
| pose4+6 | — | 4361.0 | — | [passed](output/B/pose4+6/step4/overview.png) |
| pose8+13 | — | 4687.7 | — | [passed](output/B/pose8+13/step4/overview.png) |
| pose1+9+15 | — | 14067.3 | — | [passed](output/B/pose1+9+15/step4/overview.png) |
| pose2+5+10+17 | — | 16818.6 | — | [passed](output/B/pose2+5+10+17/step4/overview.png) |
| pose3+7+8+12+19 | — | 31080.6 | — | [passed](output/B/pose3+7+8+12+19/step4/overview.png) |

Completed supports: 20/20. Improvements among comparable baseline sets: 8/10.

Comparisons use byte-equivalent saved mesh vertices/faces and unchanged workstation metric/frame. Contact choices, seating translations and construction are jointly changed, so the volume change is evidence for the combined method, not an isolated causal estimate for normal alignment. Additional transfer sets have no constructed previous comparator.

Only +Z and a finite XY search are tested; results are neither a global optimum nor generalization evidence for a newly trained network. The old N_remaining/object-frame network has not been retrained for this target. Full mechanics comes from each original 32,768-load Step3 certificate; Step4 performs one full construction acceptance and Step5 only measures the accepted model. No independent export replay is claimed.

[Value definition](../method/method.md) · [Machine-readable results](results.json) · [Comparison figure](comparison.png)
