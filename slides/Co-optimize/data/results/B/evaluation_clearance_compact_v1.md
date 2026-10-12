# 静态工作空间：无支撑叠放基线／真实支撑布局

全部 52 组，共 263 个 pose。只计最终工作位置，单位 L；1 L=1000 cm³。
无支撑基线保留各原任务朝向，将物体的水平包围盒中心对齐、最低点放在z=0，叠放全部pose后取一个BBX。其每轴长度为该轴各pose长度的最大值，达到自由平移时的最小XYZ盒。
任务依次执行；一个 state 服务多个 object pose 不表示多个物体同时装入。Juxtapose 是最终 pose 是否重新落座，multi-object-pose state 是共享关系，两者不是互斥分类。
有支撑两列使用保存的真实优化位置及全部最终支撑state；不改变物理结果。三列图同等轴测方向、同长度比例，分别居中。

| Pose set | Pose数 | 无支撑叠放BBX (L) | +Step4 BBX (L) | +Step5.2 BBX (L) | Juxtapose pose数 | 原位转动复用pose数 | State分组 |
|---|---:|---:|---:|---:|---:|---:|---|
| [pose1+2+3+4+5+6+7+8+9+10](../../../output/B/pose1+2+3+4+5+6+7+8+9+10/evaluation/clearance_compact_v1/index.html) | 10 | 5.27 | 32.80 | 35.67 | 10 | 0 | 1+2+3+4+5+6+7+8+9+10 |
| [pose1+3+5+7+9+12+16+21+23+27](../../../output/B/pose1+3+5+7+9+12+16+21+23+27/evaluation/clearance_compact_v1/index.html) | 10 | 5.47 | 37.64 | 40.67 | 10 | 0 | 1+3+5+7+9+12+16+21+23+27 |
| [pose1+2+3+4+5+6+7+8+9](../../../output/B/pose1+2+3+4+5+6+7+8+9/evaluation/clearance_compact_v1/index.html) | 9 | 5.27 | 26.38 | 29.59 | 9 | 0 | 1+2+3+4+5+6+7+8+9 |
| [pose1+4+7+9+12+16+21+23+27](../../../output/B/pose1+4+7+9+12+16+21+23+27/evaluation/clearance_compact_v1/index.html) | 9 | 4.79 | 27.23 | 28.24 | 9 | 0 | 1+4+7+9+12+16+21+23+27 |
| [pose1+2+3+4+5+6+7+8](../../../output/B/pose1+2+3+4+5+6+7+8/evaluation/clearance_compact_v1/index.html) | 8 | 5.27 | 27.46 | 29.87 | 8 | 0 | 1+2+3+4+5+6+7+8 |
| [pose1+2+4+5+6+7+10+11](../../../output/B/pose1+2+4+5+6+7+10+11/evaluation/clearance_compact_v1/index.html) | 8 | 5.27 | 26.23 | 28.43 | 8 | 0 | 1+2+4+5+6+7+10+11 |
| [pose1+3+7+10+14+18+23+27](../../../output/B/pose1+3+7+10+14+18+23+27/evaluation/clearance_compact_v1/index.html) | 8 | 6.04 | 15.92 | 17.33 | 8 | 0 | 1+3+7+10+14+18+23+27 |
| [pose1+2+3+4+5+6+7](../../../output/B/pose1+2+3+4+5+6+7/evaluation/clearance_compact_v1/index.html) | 7 | 5.27 | 24.19 | 27.03 | 7 | 0 | 1+2+3+4+5+6+7 |
| [pose1+2+4+5+6+7+11](../../../output/B/pose1+2+4+5+6+7+11/evaluation/clearance_compact_v1/index.html) | 7 | 5.27 | 26.84 | 28.33 | 7 | 0 | 1+2+4+5+6+7+11 |
| [pose1+4+7+12+21+23+27](../../../output/B/pose1+4+7+12+21+23+27/evaluation/clearance_compact_v1/index.html) | 7 | 4.79 | 26.81 | 28.24 | 7 | 0 | 1+4+7+12+21+23+27 |
| [illegal/pose4+7+12+21+23+27](../../../output/B/illegal/pose4+7+12+21+23+27/evaluation/clearance_compact_v1/index.html) | 6 | 4.79 | 19.45 | 19.85 | 5 | 1 | 4+7+12+21+23+27 |
| [illegal/pose5+12+20+24+27+28](../../../output/B/illegal/pose5+12+20+24+27+28/evaluation/clearance_compact_v1/index.html) | 6 | 5.18 | 31.25 | 33.14 | 6 | 0 | 5+12+20+24+27+28 |
| [pose1+2+3+4+5+6](../../../output/B/pose1+2+3+4+5+6/evaluation/clearance_compact_v1/index.html) | 6 | 5.27 | 21.39 | 23.90 | 6 | 0 | 1+2+3+4+5+6 |
| [pose1+2+3+4+6+7](../../../output/B/pose1+2+3+4+6+7/evaluation/clearance_compact_v1/index.html) | 6 | 4.67 | 13.92 | 15.47 | 6 | 0 | 1+2+3+4+6+7 |
| [pose1+2+3+4+7+27](../../../output/B/pose1+2+3+4+7+27/evaluation/clearance_compact_v1/index.html) | 6 | 4.88 | 11.28 | 13.02 | 6 | 0 | 1+2+3+4+7+27 |
| [pose1+2+4+5+6+11](../../../output/B/pose1+2+4+5+6+11/evaluation/clearance_compact_v1/index.html) | 6 | 5.27 | 12.60 | 13.63 | 6 | 0 | 1+2+4+5+6+11 |
| [pose1+2+4+5+6+7](../../../output/B/pose1+2+4+5+6+7/evaluation/clearance_compact_v1/index.html) | 6 | 5.27 | 13.30 | 14.04 | 6 | 0 | 1+2+4+5+6+7 |
| [pose1+4+7+12+21+27](../../../output/B/pose1+4+7+12+21+27/evaluation/clearance_compact_v1/index.html) | 6 | 4.57 | 18.60 | 20.14 | 6 | 0 | 1+4+7+12+21+27 |
| [pose1+6+11+13+14+17](../../../output/B/pose1+6+11+13+14+17/evaluation/clearance_compact_v1/index.html) | 6 | 4.96 | 10.08 | 10.93 | 6 | 0 | 1+6+11+13+14+17 |
| [pose4+5+8+9+19+23](../../../output/B/pose4+5+8+9+19+23/evaluation/clearance_compact_v1/index.html) | 6 | 5.98 | 20.10 | 21.74 | 6 | 0 | 4+5+8+9+19+23 |
| [illegal/pose1+2+6+7+29](../../../output/B/illegal/pose1+2+6+7+29/evaluation/clearance_compact_v1/index.html) | 5 | 4.67 | 10.14 | 10.82 | 5 | 0 | 1+2+6+7+29 |
| [illegal/pose5+6+13+15+27](../../../output/B/illegal/pose5+6+13+15+27/evaluation/clearance_compact_v1/index.html) | 5 | 5.20 | 11.84 | 11.85 | 4 | 1 | 5+6+13+15+27 |
| [pose1+2+3+4+27](../../../output/B/pose1+2+3+4+27/evaluation/clearance_compact_v1/index.html) | 5 | 4.88 | 19.22 | 21.49 | 5 | 0 | 1+2+3+4+27 |
| [pose1+2+3+4+6](../../../output/B/pose1+2+3+4+6/evaluation/clearance_compact_v1/index.html) | 5 | 4.62 | 32.44 | 35.29 | 5 | 0 | 1+2+3+4+6 |
| [pose1+2+3+7+27](../../../output/B/pose1+2+3+7+27/evaluation/clearance_compact_v1/index.html) | 5 | 4.88 | 19.21 | 20.83 | 5 | 0 | 1+2+3+7+27 |
| [pose1+2+4+5+6](../../../output/B/pose1+2+4+5+6/evaluation/clearance_compact_v1/index.html) | 5 | 5.27 | 12.08 | 13.11 | 5 | 0 | 1+2+4+5+6 |
| [pose1+4+7+9+24](../../../output/B/pose1+4+7+9+24/evaluation/clearance_compact_v1/index.html) | 5 | 4.17 | 9.82 | 10.66 | 5 | 0 | 1+4+7+9+24 |
| [pose12+16+19+21+27](../../../output/B/pose12+16+19+21+27/evaluation/clearance_compact_v1/index.html) | 5 | 5.19 | 11.90 | 13.07 | 5 | 0 | 12+16+19+21+27 |
| [pose5+6+11+23+29](../../../output/B/pose5+6+11+23+29/evaluation/clearance_compact_v1/index.html) | 5 | 5.50 | 13.78 | 14.51 | 5 | 0 | 5+6+11+23+29 |
| [pose6+10+13+17+30](../../../output/B/pose6+10+13+17+30/evaluation/clearance_compact_v1/index.html) | 5 | 4.48 | 10.85 | 11.49 | 5 | 0 | 6+10+13+17+30 |
| [illegal/pose1+2+6+15](../../../output/B/illegal/pose1+2+6+15/evaluation/clearance_compact_v1/index.html) | 4 | 4.64 | 12.69 | 14.48 | 4 | 0 | 1+2+6+15 |
| [illegal/pose7+11+13+19](../../../output/B/illegal/pose7+11+13+19/evaluation/clearance_compact_v1/index.html) | 4 | 5.01 | 9.58 | 10.04 | 4 | 0 | 7+11+13+19 |
| [pose1+11+14+27](../../../output/B/pose1+11+14+27/evaluation/clearance_compact_v1/index.html) | 4 | 4.33 | 7.48 | 8.12 | 4 | 0 | 1+11+14+27 |
| [pose1+2+3+27](../../../output/B/pose1+2+3+27/evaluation/clearance_compact_v1/index.html) | 4 | 4.88 | 7.55 | 8.19 | 4 | 0 | 1+2+3+27 |
| [pose1+2+4+19](../../../output/B/pose1+2+4+19/evaluation/clearance_compact_v1/index.html) | 4 | 5.17 | 8.32 | 9.39 | 4 | 0 | 1+2+4+19 |
| [pose1+2+4+6](../../../output/B/pose1+2+4+6/evaluation/clearance_compact_v1/index.html) | 4 | 4.62 | 8.34 | 9.54 | 4 | 0 | 1+2+4+6 |
| [pose1+2+4+7](../../../output/B/pose1+2+4+7/evaluation/clearance_compact_v1/index.html) | 4 | 4.24 | 7.39 | 8.50 | 4 | 0 | 1+2+4+7 |
| [pose2+3+4+7](../../../output/B/pose2+3+4+7/evaluation/clearance_compact_v1/index.html) | 4 | 4.64 | 7.16 | 8.34 | 3 | 1 | 2+3+4+7 |
| [pose5+6+23+29](../../../output/B/pose5+6+23+29/evaluation/clearance_compact_v1/index.html) | 4 | 5.50 | 11.15 | 12.05 | 4 | 0 | 5+6+23+29 |
| [pose8+9+13+30](../../../output/B/pose8+9+13+30/evaluation/clearance_compact_v1/index.html) | 4 | 4.20 | 7.31 | 7.97 | 4 | 0 | 8+9+13+30 |
| [illegal/pose1+2+29](../../../output/B/illegal/pose1+2+29/evaluation/clearance_compact_v1/index.html) | 3 | 4.58 | 7.98 | 7.98 | 2 | 1 | 1+2+29 |
| [illegal/pose5+13+27](../../../output/B/illegal/pose5+13+27/evaluation/clearance_compact_v1/index.html) | 3 | 4.56 | 8.61 | 9.35 | 3 | 0 | 5+13+27 |
| [pose1+12+29](../../../output/B/pose1+12+29/evaluation/clearance_compact_v1/index.html) | 3 | 4.68 | 9.39 | 9.39 | 2 | 1 | 1+12+29 |
| [pose18+23+24](../../../output/B/pose18+23+24/evaluation/clearance_compact_v1/index.html) | 3 | 5.33 | 7.34 | 8.40 | 3 | 0 | 18+23+24 |
| [pose4+5+7](../../../output/B/pose4+5+7/evaluation/clearance_compact_v1/index.html) | 3 | 4.71 | 7.98 | 8.94 | 1 | 2 | 4+7; 5 |
| [pose8+10+19](../../../output/B/pose8+10+19/evaluation/clearance_compact_v1/index.html) | 3 | 5.66 | 9.21 | 10.94 | 3 | 0 | 8+10+19 |
| [illegal/pose11+19](../../../output/B/illegal/pose11+19/evaluation/clearance_compact_v1/index.html) | 2 | 5.01 | 8.41 | 9.30 | 1 | 1 | 11+19 |
| [illegal/pose2+29](../../../output/B/illegal/pose2+29/evaluation/clearance_compact_v1/index.html) | 2 | 4.01 | 6.26 | 6.26 | 1 | 1 | 2+29 |
| [pose19+28](../../../output/B/pose19+28/evaluation/clearance_compact_v1/index.html) | 2 | 4.28 | 5.54 | 5.97 | 1 | 1 | 19+28 |
| [pose2+20](../../../output/B/pose2+20/evaluation/clearance_compact_v1/index.html) | 2 | 4.11 | 4.59 | 4.59 | 0 | 2 | 2; 20 |
| [pose3+15](../../../output/B/pose3+15/evaluation/clearance_compact_v1/index.html) | 2 | 4.41 | 6.29 | 8.07 | 2 | 0 | 3+15 |
| [pose8+21](../../../output/B/pose8+21/evaluation/clearance_compact_v1/index.html) | 2 | 3.49 | 5.92 | 6.59 | 2 | 0 | 8+21 |

各集合BBX统计和：无支撑叠放 254.41 L，Step4 769.27 L，完整支撑 834.76 L；不是同一工作站的总体积。
最终 Juxtapose 251 个 pose，原位转动复用 12 个 pose；51 个 state 服务多个 object pose，3 个 state 只服务一个。

以下为真实支撑落座物体的诊断空间，并非无支撑基线。

| Pose set | 支撑落座物体BBX (L) | +Step4 (L) | +Step5.2 (L) | 地脚BBX增幅 | Step4实体 (cm³) | 全部实体 (cm³) |
|---|---:|---:|---:|---:|---:|---:|
| pose1+2+3+4+5+6+7+8+9+10 | 31.25 | 32.80 | 35.67 | +8.75% | 890.44 | 904.39 |
| pose1+3+5+7+9+12+16+21+23+27 | 35.58 | 37.64 | 40.67 | +8.06% | 801.68 | 814.86 |
| pose1+2+3+4+5+6+7+8+9 | 25.47 | 26.38 | 29.59 | +12.14% | 612.87 | 626.25 |
| pose1+4+7+9+12+16+21+23+27 | 25.50 | 27.23 | 28.24 | +3.70% | 754.51 | 767.98 |
| pose1+2+3+4+5+6+7+8 | 25.81 | 27.46 | 29.87 | +8.78% | 702.96 | 717.91 |
| pose1+2+4+5+6+7+10+11 | 24.54 | 26.23 | 28.43 | +8.35% | 722.22 | 735.06 |
| pose1+3+7+10+14+18+23+27 | 14.82 | 15.92 | 17.33 | +8.88% | 454.99 | 468.93 |
| pose1+2+3+4+5+6+7 | 22.66 | 24.19 | 27.03 | +11.74% | 655.05 | 669.10 |
| pose1+2+4+5+6+7+11 | 25.28 | 26.84 | 28.33 | +5.53% | 706.71 | 716.54 |
| pose1+4+7+12+21+23+27 | 25.15 | 26.81 | 28.24 | +5.32% | 610.55 | 621.49 |
| illegal/pose4+7+12+21+23+27 | 18.84 | 19.45 | 19.85 | +2.07% | 532.72 | 624.35 |
| illegal/pose5+12+20+24+27+28 | 30.00 | 31.25 | 33.14 | +6.04% | 546.61 | 561.52 |
| pose1+2+3+4+5+6 | 19.91 | 21.39 | 23.90 | +11.74% | 547.61 | 561.61 |
| pose1+2+3+4+6+7 | 12.89 | 13.92 | 15.47 | +11.10% | 346.32 | 356.80 |
| pose1+2+3+4+7+27 | 10.31 | 11.28 | 13.02 | +15.39% | 347.83 | 358.84 |
| pose1+2+4+5+6+11 | 11.76 | 12.60 | 13.63 | +8.14% | 338.76 | 347.82 |
| pose1+2+4+5+6+7 | 12.27 | 13.30 | 14.04 | +5.54% | 411.41 | 418.96 |
| pose1+4+7+12+21+27 | 17.09 | 18.60 | 20.14 | +8.26% | 632.37 | 641.16 |
| pose1+6+11+13+14+17 | 9.60 | 10.08 | 10.93 | +8.46% | 289.56 | 300.34 |
| pose4+5+8+9+19+23 | 18.59 | 20.10 | 21.74 | +8.14% | 587.70 | 601.86 |
| illegal/pose1+2+6+7+29 | 9.66 | 10.14 | 10.82 | +6.74% | 278.32 | 292.61 |
| illegal/pose5+6+13+15+27 | 11.02 | 11.84 | 11.85 | +0.09% | 382.51 | 435.58 |
| pose1+2+3+4+27 | 17.67 | 19.22 | 21.49 | +11.79% | 515.22 | 535.34 |
| pose1+2+3+4+6 | 30.47 | 32.44 | 35.29 | +8.79% | 581.09 | 601.73 |
| pose1+2+3+7+27 | 18.01 | 19.21 | 20.83 | +8.42% | 470.38 | 486.08 |
| pose1+2+4+5+6 | 11.10 | 12.08 | 13.11 | +8.54% | 317.92 | 328.97 |
| pose1+4+7+9+24 | 9.01 | 9.82 | 10.66 | +8.58% | 277.60 | 287.10 |
| pose12+16+19+21+27 | 10.77 | 11.90 | 13.07 | +9.82% | 300.77 | 311.14 |
| pose5+6+11+23+29 | 13.18 | 13.78 | 14.51 | +5.27% | 271.20 | 286.69 |
| pose6+10+13+17+30 | 10.30 | 10.85 | 11.49 | +5.89% | 285.13 | 294.69 |
| illegal/pose1+2+6+15 | 11.88 | 12.69 | 14.48 | +14.11% | 375.02 | 389.61 |
| illegal/pose7+11+13+19 | 8.94 | 9.58 | 10.04 | +4.79% | 209.31 | 219.91 |
| pose1+11+14+27 | 7.07 | 7.48 | 8.12 | +8.52% | 218.88 | 230.66 |
| pose1+2+3+27 | 6.97 | 7.55 | 8.19 | +8.42% | 222.30 | 230.76 |
| pose1+2+4+19 | 7.67 | 8.32 | 9.39 | +12.89% | 207.17 | 218.56 |
| pose1+2+4+6 | 7.54 | 8.34 | 9.54 | +14.36% | 265.17 | 277.08 |
| pose1+2+4+7 | 6.67 | 7.39 | 8.50 | +14.94% | 259.59 | 270.44 |
| pose2+3+4+7 | 6.84 | 7.16 | 8.34 | +16.47% | 203.07 | 210.77 |
| pose5+6+23+29 | 10.09 | 11.15 | 12.05 | +8.06% | 288.24 | 301.43 |
| pose8+9+13+30 | 6.93 | 7.31 | 7.97 | +9.10% | 195.46 | 204.82 |
| illegal/pose1+2+29 | 7.51 | 7.98 | 7.98 | -0.00% | 186.02 | 194.74 |
| illegal/pose5+13+27 | 7.69 | 8.61 | 9.35 | +8.52% | 303.51 | 329.59 |
| pose1+12+29 | 8.75 | 9.39 | 9.39 | +0.00% | 199.92 | 209.15 |
| pose18+23+24 | 6.55 | 7.34 | 8.40 | +14.54% | 257.45 | 268.46 |
| pose4+5+7 | 6.21 | 7.98 | 8.94 | +12.03% | 185.19 | 197.08 |
| pose8+10+19 | 8.53 | 9.21 | 10.94 | +18.79% | 251.86 | 263.68 |
| illegal/pose11+19 | 7.62 | 8.41 | 9.30 | +10.58% | 171.64 | 180.57 |
| illegal/pose2+29 | 6.06 | 6.26 | 6.26 | +0.00% | 174.51 | 181.29 |
| pose19+28 | 5.07 | 5.54 | 5.97 | +7.74% | 148.13 | 158.98 |
| pose2+20 | 4.31 | 4.59 | 4.59 | -0.00% | 135.94 | 142.08 |
| pose3+15 | 5.66 | 6.29 | 8.07 | +28.32% | 217.53 | 230.09 |
| pose8+21 | 5.33 | 5.92 | 6.59 | +11.25% | 223.00 | 232.05 |

单独分配不同state时的空间诊断统计和：物体 722.34 L，Step4 774.66 L，Step5.2 841.47 L。
[逐 pose 尺寸、共享分组和材料数据](data/evaluation_clearance_compact_v1/summary.json)。
