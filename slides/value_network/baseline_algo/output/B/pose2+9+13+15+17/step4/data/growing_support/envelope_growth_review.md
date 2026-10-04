# Fixed-envelope greedy growth

All nine active Step4 supports and ten Step5 groups / 32 placements are current. Compare with the previous fast single-acceptance method. Negative changes mean smaller occupied boxes. The greedy construction has no guaranteed improvement. No export recheck or independent replay was performed.

| Group | Previous XYZ volume cm3 | Greedy XYZ volume cm3 | Volume change | XY area change | Material change | Previous construction s | Greedy construction s |
|---|---:|---:|---:|---:|---:|---:|---:|
| pose1+3copied | 5237.14 | 5650.99 | +7.90% | +5.53% | +4.08% | 4.54 | 2.95 |
| pose3+6 | 6652.92 | 7405.82 | +11.32% | +11.32% | +10.89% | 1.92 | 3.40 |
| pose5+7 | 6035.79 | 6089.91 | +0.90% | +0.90% | +20.56% | 1.69 | 4.67 |
| pose2+10+15 | 15884.68 | 20295.84 | +27.77% | +26.19% | +9.26% | 3.99 | 10.36 |
| pose2+12+15 | 18902.20 | 19060.19 | +0.84% | +0.84% | +8.37% | 3.49 | 7.71 |
| pose1+2+8+17 | 31232.86 | 30175.40 | -3.39% | -3.39% | +6.38% | 5.65 | 12.61 |
| pose6+8+10+19 | 40212.79 | 31168.25 | -22.49% | -12.75% | -4.51% | 6.84 | 11.04 |
| pose1+2+3+4+5 | 40121.26 | 49307.13 | +22.90% | +22.90% | +16.93% | 6.38 | 16.89 |
| pose2+9+13+15+17 | 58499.84 | 59454.52 | +1.63% | +1.63% | +5.26% | 8.70 | 28.87 |

Complete contact, floor, full 500 mm sweep and ground coverage conditions are accepted once on constructed material. Complete rod/foot cores are preserved. Step3 and historical pose1+3 Step4 match the protected snapshot. Only the historical Step5 batch summary files have authorized protected-file changes.


## Comparison with joint beam experiment

| Group | Volume change | XY area change | Joint construction s | Greedy construction s |
|---|---:|---:|---:|---:|
| pose1+3copied | +5.78% | +3.45% | 17.20 | 2.95 |
| pose3+6 | +23.53% | +23.53% | 10.29 | 3.40 |
| pose5+7 | -0.45% | -0.45% | 10.38 | 4.67 |
| pose2+10+15 | -10.25% | -11.35% | 16.09 | 10.36 |
| pose2+12+15 | -25.67% | -25.67% | 15.70 | 7.71 |
| pose1+2+8+17 | -20.66% | -20.66% | 20.74 | 12.61 |
| pose6+8+10+19 | -25.53% | -16.18% | 23.40 | 11.04 |
| pose1+2+3+4+5 | -5.18% | -5.18% | 26.77 | 16.89 |
| pose2+9+13+15+17 | -21.42% | -21.42% | 118.79 | 28.87 |
