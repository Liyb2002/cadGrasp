# B: Step5 occupied-space comparison

10 groups / 32 installed poses. Aggregate XYZ box volume and XY area in saved workstation coordinates; motion sweeps are excluded.

[All groups](all_groups.png)

| Group | Object XYZ (mm) | Supported XYZ (mm) | Object volume (cm3) | Supported volume (cm3) | Object area (cm2) | Supported area (cm2) | Extra XY area |
|---|---:|---:|---:|---:|---:|---:|---:|
| [pose1+3](bbox.png) | 155.0 x 184.4 x 146.5 | 198.2 x 215.9 x 146.5 | 4188.57 | 6269.26 | 285.87 | 427.88 | +49.68% |
| [pose1+3copied](../../pose1+3copied/step5_evaluate/bbox.png) | 155.0 x 184.4 x 146.5 | 201.3 x 187.4 x 149.8 | 4188.57 | 5650.99 | 285.87 | 377.19 | +31.94% |
| [pose3+6](../../pose3+6/step5_evaluate/bbox.png) | 151.4 x 134.7 x 190.7 | 174.2 x 217.0 x 195.9 | 3890.42 | 7405.82 | 204.03 | 378.04 | +85.29% |
| [pose5+7](../../pose5+7/step5_evaluate/bbox.png) | 206.2 x 168.2 x 134.2 | 209.2 x 217.0 x 134.2 | 4651.56 | 6089.91 | 346.71 | 453.92 | +30.92% |
| [pose2+10+15](../../pose2+10+15/step5_evaluate/bbox.png) | 184.5 x 155.6 x 162.0 | 306.4 x 273.5 x 242.1 | 4651.07 | 20295.84 | 287.12 | 838.22 | +191.94% |
| [pose2+12+15](../../pose2+12+15/step5_evaluate/bbox.png) | 197.6 x 153.2 x 162.0 | 292.8 x 232.9 x 279.5 | 4903.19 | 19060.19 | 302.69 | 681.88 | +125.28% |
| [pose1+2+8+17](../../pose1+2+8+17/step5_evaluate/bbox.png) | 184.5 x 185.6 x 185.7 | 290.7 x 331.6 x 313.1 | 6360.88 | 30175.40 | 342.46 | 963.91 | +181.47% |
| [pose6+8+10+19](../../pose6+8+10+19/step5_evaluate/bbox.png) | 195.4 x 169.8 x 159.7 | 329.6 x 331.6 x 285.2 | 5299.37 | 31168.25 | 331.77 | 1092.79 | +229.38% |
| [pose1+2+3+4+5](../../pose1+2+3+4+5/step5_evaluate/bbox.png) | 205.0 x 186.9 x 190.7 | 375.8 x 413.3 x 317.5 | 7305.71 | 49307.13 | 383.14 | 1553.06 | +305.35% |
| [pose2+9+13+15+17](../../pose2+9+13+15+17/step5_evaluate/bbox.png) | 191.2 x 184.0 x 185.7 | 389.2 x 410.1 x 372.6 | 6536.63 | 59454.52 | 351.92 | 1595.84 | +353.47% |

pose1+3 remains the historical reference. pose1+3copied uses the same archived object poses with new support. Other groups use current growing models and placements. Measurements preserve all upstream acceptance states; occupied box volume is distinct from support material volume.
