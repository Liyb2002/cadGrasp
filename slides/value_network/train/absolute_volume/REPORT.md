# Neural absolute-volume Step3: actual Step4 / Step5

A newly trained network predicts witnessed-completion evidence, state/action final-volume cost and XY seating. Runtime does not retrieve contact/layout choices from the successful-completion training records. Historical reference files are opened only by the original task loader; their contact/layout choices are discarded. Every accepted model is constructed afresh and passes the original full-load and full-body acceptance. Step5 measures its actual occupied XYZ bounding-box volume, not material volume.

All 20 groups were included in training. This measures memorization and independent construction, not unseen-group generalization. There is one successful completion per group; actions that belong to that completion share its witnessed cost. Unsearched actions have no fake failed-volume labels. Evidence rejection means not demonstrated, not physically impossible. Within regression uncertainty, a separately learned ordering breaks value ties without a head-count penalty.

| Set | Search cm3 | Neural cm3 | vs search | vs original baseline | Support |
|---|---:|---:|---:|---:|---|
| pose1+3 | 6330.0 | 6330.0 | +0.0% | +1.0% | [accepted](output/B/pose1+3/step4/overview.png) |
| pose1+3copied | 6330.0 | 6330.0 | +0.0% | +12.0% | [accepted](output/B/pose1+3copied/step4/overview.png) |
| pose3+6 | 5484.2 | 5484.2 | +0.0% | -25.9% | [accepted](output/B/pose3+6/step4/overview.png) |
| pose5+7 | 4651.6 | 4651.6 | +0.0% | -23.6% | [accepted](output/B/pose5+7/step4/overview.png) |
| pose2+10+15 | 9283.6 | 9283.6 | +0.0% | -54.3% | [accepted](output/B/pose2+10+15/step4/overview.png) |
| pose2+12+15 | 11562.4 | 11562.4 | +0.0% | -39.3% | [accepted](output/B/pose2+12+15/step4/overview.png) |
| pose1+2+8+17 | 14699.7 | 14699.7 | +0.0% | -51.3% | [accepted](output/B/pose1+2+8+17/step4/overview.png) |
| pose6+8+10+19 | 16325.8 | 16325.8 | +0.0% | -47.6% | [accepted](output/B/pose6+8+10+19/step4/overview.png) |
| pose1+2+3+4+5 | 24272.3 | 24272.3 | +0.0% | -50.8% | [accepted](output/B/pose1+2+3+4+5/step4/overview.png) |
| pose2+9+13+15+17 | 33701.9 | 33701.9 | +0.0% | -43.3% | [accepted](output/B/pose2+9+13+15+17/step4/overview.png) |
| pose1+7 | 4657.1 | 4657.1 | +0.0% | — | [accepted](output/B/pose1+7/step4/overview.png) |
| pose2+4+12 | 13580.2 | 13580.2 | +0.0% | — | [accepted](output/B/pose2+4+12/step4/overview.png) |
| pose3+6+9+13 | 20671.5 | 20671.5 | +0.0% | — | [accepted](output/B/pose3+6+9+13/step4/overview.png) |
| pose5+8+10+15+17 | 29628.1 | 29628.1 | +0.0% | — | [accepted](output/B/pose5+8+10+15+17/step4/overview.png) |
| pose1+2+7+12+19 | 26328.8 | 26328.8 | +0.0% | — | [accepted](output/B/pose1+2+7+12+19/step4/overview.png) |
| pose4+6 | 4361.0 | 4361.0 | +0.0% | — | [accepted](output/B/pose4+6/step4/overview.png) |
| pose8+13 | 4687.7 | 4687.7 | +0.0% | — | [accepted](output/B/pose8+13/step4/overview.png) |
| pose1+9+15 | 14067.3 | 14067.3 | +0.0% | — | [accepted](output/B/pose1+9+15/step4/overview.png) |
| pose2+5+10+17 | 16818.6 | 16818.6 | +0.0% | — | [accepted](output/B/pose2+5+10+17/step4/overview.png) |
| pose3+7+8+12+19 | 31080.6 | 31080.6 | +0.0% | — | [accepted](output/B/pose3+7+8+12+19/step4/overview.png) |

Accepted fresh supports: 20/20.
Original-baseline comparisons: 8/10 improve; aggregate occupied volume changes -43.5%. These gains reproduce the preceding geometric-search solutions, rather than a new improvement from training.

[Training diagnostics](training.json) · [Machine-readable results](results.json) · [Volume comparison](comparison.png)
