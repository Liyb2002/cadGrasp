# Fast adaptive coverage growth: measured comparison

Old/new run times include construction and two internal validations; neither includes setup, independent replay or rendering. The construction column isolates the new contact starts, head network and ground growth. Each group was run sequentially on the same machine.

| Group | Before run s | New run s | Speedup | New construction s | Before box cm3 | New box cm3 | Box change | XY area change | Material change | Feet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pose1+3copied | 41.82 | 6.54 | 6.4x | 4.54 | 5183.04 | 5237.14 | +1.04% | +1.04% | -19.93% | 13 |
| pose3+6 | 50.52 | 4.58 | 11.0x | 1.92 | 5874.61 | 6652.92 | +13.25% | +13.25% | -3.35% | 12 |
| pose5+7 | 34.34 | 3.86 | 8.9x | 1.63 | 6158.27 | 6035.79 | -1.99% | -1.99% | +0.83% | 11 |
| pose2+10+15 | 89.68 | 7.69 | 11.7x | 3.98 | 15294.46 | 15884.68 | +3.86% | +3.86% | -5.23% | 20 |
| pose2+12+15 | 95.02 | 8.22 | 11.6x | 3.86 | 18773.32 | 18902.20 | +0.69% | +0.69% | -7.22% | 16 |
| pose1+2+8+17 | 155.90 | 10.90 | 14.3x | 5.66 | 30412.72 | 31232.86 | +2.70% | +2.70% | +13.43% | 25 |
| pose6+8+10+19 | 164.73 | 12.22 | 13.5x | 6.84 | 28321.01 | 40212.79 | +41.99% | +22.85% | +6.55% | 27 |
| pose1+2+3+4+5 | 183.01 | 14.52 | 12.6x | 6.39 | 39864.68 | 40121.26 | +0.64% | +0.64% | -7.06% | 32 |
| pose2+9+13+15+17 | 215.31 | 16.31 | 13.2x | 8.69 | 53581.51 | 58499.84 | +9.18% | +9.18% | +4.07% | 32 |

All nine current supports pass original ground coverage, contact-surface checks, full core preservation and fresh continuous 500 mm withdrawal replay. All ten Step5 groups are updated. Original pose1+3 Step3/Step4 and upstream protected files match the snapshot at the start of this speed-refinement turn.
Ground feet are adaptive outputs, not prescribed targets. Rod cores remain at least 5 mm with the recorded original-contact/short-transition exceptions. No global optimum, new force certification or equal-strength material comparison is claimed.

## Full current setup and validation timing

| Group | Construction s | Setup + construction + internal validation s | Independent replay s | Proposal soles | Exactly checked soles | Routing pitch mm |
|---|---:|---:|---:|---:|---:|---:|
| pose1+3copied | 4.54 | 10.32 | 2.37 | 552 | 135 | 4.0 |
| pose3+6 | 1.92 | 9.13 | 2.59 | 526 | 18 | 8.0 |
| pose5+7 | 1.63 | 7.20 | 2.41 | 541 | 18 | 8.0 |
| pose2+10+15 | 3.98 | 13.77 | 3.58 | 828 | 57 | 8.0 |
| pose2+12+15 | 3.86 | 14.03 | 3.59 | 765 | 37 | 8.0 |
| pose1+2+8+17 | 5.66 | 18.87 | 5.07 | 1096 | 53 | 8.0 |
| pose6+8+10+19 | 6.84 | 22.11 | 5.07 | 1182 | 193 | 8.0 |
| pose1+2+3+4+5 | 6.39 | 26.00 | 7.16 | 1278 | 66 | 8.0 |
| pose2+9+13+15+17 | 8.69 | 27.03 | 6.70 | 1269 | 52 | 8.0 |

All nine repeated results match the preceding fast run in measured box volume and material volume.

The +41.99% outlier pose6+8+10+19 has an unchanged head-network-plus-object box of 21213.19 cm3 before ground growth. Its final dimensions grow from 317.14 x 321.49 x 277.78 mm to 359.15 x 348.75 x 321.05 mm; the expansion arises during ground-contact growth.
