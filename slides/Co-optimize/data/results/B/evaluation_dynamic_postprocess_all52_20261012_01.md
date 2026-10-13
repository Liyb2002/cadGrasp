# 当前动态换座布局的Step5与Evaluation

全部52组／263个object pose纳入报告。Step5.1 50/50，Step5.2 49/50，Step5.3 49/50；另2组Step4失败，未继续构造。

最终通过组共238个object pose、129个实际支撑安装pose，原位复用129、Juxtapose 109。每组一件共享支撑，任务依次执行。

材料体积单位cm³；BBX单位L，是包含物体和支撑全部最终安装状态的静态包围盒。净变化统一比较Step5.3与Step4名义support.obj，负数表示减少。各组布局、角色、退出方向和原始需求保持不变。失败行没有成品体积。

最终通过组：材料 6835.005 → 10842.841 cm³，净变化 +4007.836 cm³（+58.64%）；BBX 324.424 → 413.720 L，净变化 +89.296 L（+27.52%）。这是独立集合的指标合计。

相对同一批Step4已经构造的完整参考实体：材料 10172.433 → 10842.841 cm³，净变化 +670.408 cm³；BBX 412.714 → 413.720 L，净变化 +1.006 L。

| Set（图） | 状态 | object | 支撑安装 | 复用 | Juxtapose | Step4材料cm³ | Step5.2材料cm³ | Step5.3材料cm³ | 材料净变化 | 材料变化% | Step4 BBX L | Step5.3 BBX L | BBX净变化L | BBX变化% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [pose1+2+3+4+5+6+7+8+9+10](../../../output/B/pose1+2+3+4+5+6+7+8+9+10/step5.3/status.png) | STEP4_FAIL | 10 | 1 | 1 | 9 | — | — | — | — | — | — | — | — | — |
| [pose1+3+5+7+9+12+16+21+23+27](../../../output/B/pose1+3+5+7+9+12+16+21+23+27/step5.3/all_poses.png) | PASS | 10 | 1 | 1 | 9 | 503.32 | 249.41 | 290.66 | -212.66 | -42.25 | 21.86 | 21.32 | -0.53 | -2.44 |
| [pose1+2+3+4+5+6+7+8+9](../../../output/B/pose1+2+3+4+5+6+7+8+9/step5.3/all_poses.png) | PASS | 9 | 1 | 1 | 8 | 310.50 | 87.86 | 125.11 | -185.39 | -59.71 | 12.34 | 11.91 | -0.42 | -3.42 |
| [pose1+4+7+9+12+16+21+23+27](../../../output/B/pose1+4+7+9+12+16+21+23+27/step5.3/status.png) | STEP4_FAIL | 9 | 2 | 2 | 7 | — | — | — | — | — | — | — | — | — |
| [pose1+2+3+4+5+6+7+8](../../../output/B/pose1+2+3+4+5+6+7+8/step5.3/all_poses.png) | PASS | 8 | 3 | 3 | 5 | 112.81 | 61.89 | 78.85 | -33.96 | -30.10 | 7.52 | 8.99 | +1.48 | +19.67 |
| [pose1+2+4+5+6+7+10+11](../../../output/B/pose1+2+4+5+6+7+10+11/step5.3/all_poses.png) | PASS | 8 | 1 | 1 | 7 | 330.82 | 186.31 | 217.00 | -113.83 | -34.41 | 13.57 | 13.29 | -0.27 | -2.02 |
| [pose1+3+7+10+14+18+23+27](../../../output/B/pose1+3+7+10+14+18+23+27/step5.3/all_poses.png) | PASS | 8 | 2 | 2 | 6 | 150.48 | 70.37 | 98.97 | -51.51 | -34.23 | 7.76 | 8.16 | +0.39 | +5.07 |
| [pose1+2+3+4+5+6+7](../../../output/B/pose1+2+3+4+5+6+7/step5.3/all_poses.png) | PASS | 7 | 5 | 5 | 2 | 80.80 | 30.63 | 161.71 | +80.91 | +100.14 | 5.98 | 9.11 | +3.13 | +52.38 |
| [pose1+2+4+5+6+7+11](../../../output/B/pose1+2+4+5+6+7+11/step5.3/all_poses.png) | PASS | 7 | 2 | 2 | 5 | 161.02 | 54.28 | 131.58 | -29.45 | -18.29 | 8.88 | 11.41 | +2.53 | +28.52 |
| [pose1+4+7+12+21+23+27](../../../output/B/pose1+4+7+12+21+23+27/step5.3/all_poses.png) | PASS | 7 | 1 | 1 | 6 | 203.67 | 45.52 | 70.90 | -132.77 | -65.19 | 6.49 | 6.21 | -0.28 | -4.28 |
| [illegal/pose4+7+12+21+23+27](../../../output/B/illegal/pose4+7+12+21+23+27/step5.3/all_poses.png) | PASS | 6 | 1 | 1 | 5 | 197.04 | 58.34 | 94.66 | -102.39 | -51.96 | 6.49 | 6.54 | +0.05 | +0.73 |
| [illegal/pose5+12+20+24+27+28](../../../output/B/illegal/pose5+12+20+24+27+28/step5.3/all_poses.png) | PASS | 6 | 1 | 1 | 5 | 184.34 | 83.89 | 120.69 | -63.66 | -34.53 | 6.74 | 6.63 | -0.11 | -1.65 |
| [pose1+2+3+4+5+6](../../../output/B/pose1+2+3+4+5+6/step5.3/all_poses.png) | PASS | 6 | 2 | 2 | 4 | 204.05 | 43.87 | 79.24 | -124.80 | -61.16 | 7.31 | 7.29 | -0.01 | -0.19 |
| [pose1+2+3+4+6+7](../../../output/B/pose1+2+3+4+6+7/step5.3/all_poses.png) | PASS | 6 | 1 | 1 | 5 | 217.00 | 220.03 | 220.03 | +3.03 | +1.40 | 6.77 | 6.87 | +0.09 | +1.35 |
| [pose1+2+3+4+7+27](../../../output/B/pose1+2+3+4+7+27/step5.3/all_poses.png) | PASS | 6 | 2 | 2 | 4 | 147.46 | 37.22 | 98.16 | -49.30 | -33.43 | 5.93 | 7.04 | +1.11 | +18.68 |
| [pose1+2+4+5+6+11](../../../output/B/pose1+2+4+5+6+11/step5.3/all_poses.png) | PASS | 6 | 1 | 1 | 5 | 215.27 | 55.69 | 70.68 | -144.60 | -67.17 | 6.72 | 6.40 | -0.33 | -4.84 |
| [pose1+2+4+5+6+7](../../../output/B/pose1+2+4+5+6+7/step5.3/status.png) | PRUNING_FAIL | 6 | 2 | 2 | 4 | 159.45 | — | — | — | — | 6.14 | — | — | — |
| [pose1+4+7+12+21+27](../../../output/B/pose1+4+7+12+21+27/step5.3/all_poses.png) | PASS | 6 | 4 | 4 | 2 | 84.03 | 215.12 | 215.12 | +131.08 | +155.99 | 6.58 | 8.41 | +1.82 | +27.72 |
| [pose1+6+11+13+14+17](../../../output/B/pose1+6+11+13+14+17/step5.3/all_poses.png) | PASS | 6 | 1 | 1 | 5 | 189.45 | 89.64 | 376.35 | +186.90 | +98.66 | 5.83 | 5.72 | -0.11 | -1.89 |
| [pose4+5+8+9+19+23](../../../output/B/pose4+5+8+9+19+23/step5.3/all_poses.png) | PASS | 6 | 1 | 1 | 5 | 205.47 | 71.42 | 111.82 | -93.66 | -45.58 | 7.57 | 7.40 | -0.17 | -2.26 |
| [illegal/pose1+2+6+7+29](../../../output/B/illegal/pose1+2+6+7+29/step5.3/all_poses.png) | PASS | 5 | 4 | 4 | 1 | 52.47 | 14.96 | 80.47 | +27.99 | +53.35 | 5.18 | 7.70 | +2.51 | +48.46 |
| [illegal/pose5+6+13+15+27](../../../output/B/illegal/pose5+6+13+15+27/step5.3/all_poses.png) | PASS | 5 | 2 | 2 | 3 | 219.09 | 82.65 | 142.06 | -77.03 | -35.16 | 6.44 | 6.67 | +0.22 | +3.47 |
| [pose1+2+3+4+27](../../../output/B/pose1+2+3+4+27/step5.3/all_poses.png) | PASS | 5 | 4 | 4 | 1 | 85.27 | 23.44 | 146.34 | +61.07 | +71.62 | 6.20 | 7.86 | +1.66 | +26.79 |
| [pose1+2+3+4+6](../../../output/B/pose1+2+3+4+6/step5.3/all_poses.png) | PASS | 5 | 5 | 5 | 0 | 47.71 | 28.89 | 224.59 | +176.88 | +370.70 | 5.76 | 10.00 | +4.23 | +73.42 |
| [pose1+2+3+7+27](../../../output/B/pose1+2+3+7+27/step5.3/all_poses.png) | PASS | 5 | 5 | 5 | 0 | 50.15 | 14.93 | 101.12 | +50.97 | +101.64 | 5.27 | 9.04 | +3.77 | +71.56 |
| [pose1+2+4+5+6](../../../output/B/pose1+2+4+5+6/step5.3/all_poses.png) | PASS | 5 | 5 | 5 | 0 | 88.38 | 7.04 | 154.38 | +66.00 | +74.68 | 5.97 | 8.15 | +2.19 | +36.66 |
| [pose1+4+7+9+24](../../../output/B/pose1+4+7+9+24/step5.3/all_poses.png) | PASS | 5 | 4 | 4 | 1 | 43.40 | 12.92 | 976.80 | +933.40 | +2150.44 | 5.05 | 9.50 | +4.45 | +88.13 |
| [pose12+16+19+21+27](../../../output/B/pose12+16+19+21+27/step5.3/all_poses.png) | PASS | 5 | 5 | 5 | 0 | 63.35 | 18.06 | 155.22 | +91.88 | +145.05 | 6.36 | 17.91 | +11.55 | +181.63 |
| [pose5+6+11+23+29](../../../output/B/pose5+6+11+23+29/step5.3/all_poses.png) | PASS | 5 | 3 | 3 | 2 | 139.98 | 50.96 | 123.72 | -16.26 | -11.61 | 6.66 | 7.81 | +1.14 | +17.18 |
| [pose6+10+13+17+30](../../../output/B/pose6+10+13+17+30/step5.3/all_poses.png) | PASS | 5 | 4 | 4 | 1 | 44.88 | 24.21 | 333.36 | +288.49 | +642.85 | 5.13 | 9.69 | +4.56 | +89.04 |
| [illegal/pose1+2+6+15](../../../output/B/illegal/pose1+2+6+15/step5.3/all_poses.png) | PASS | 4 | 1 | 1 | 3 | 198.07 | 20.04 | 80.59 | -117.48 | -59.31 | 5.21 | 5.59 | +0.38 | +7.20 |
| [illegal/pose7+11+13+19](../../../output/B/illegal/pose7+11+13+19/step5.3/all_poses.png) | PASS | 4 | 2 | 2 | 2 | 109.64 | 57.86 | 124.83 | +15.19 | +13.85 | 6.30 | 7.18 | +0.88 | +13.98 |
| [pose1+11+14+27](../../../output/B/pose1+11+14+27/step5.3/all_poses.png) | PASS | 4 | 4 | 4 | 0 | 36.66 | 7.85 | 116.18 | +79.52 | +216.93 | 5.18 | 8.15 | +2.97 | +57.29 |
| [pose1+2+3+27](../../../output/B/pose1+2+3+27/step5.3/all_poses.png) | PASS | 4 | 3 | 3 | 1 | 137.39 | 24.60 | 96.36 | -41.04 | -29.87 | 5.40 | 7.03 | +1.63 | +30.12 |
| [pose1+2+4+19](../../../output/B/pose1+2+4+19/step5.3/all_poses.png) | PASS | 4 | 4 | 4 | 0 | 65.89 | 18.97 | 2177.06 | +2111.18 | +3204.22 | 7.11 | 21.85 | +14.74 | +207.44 |
| [pose1+2+4+6](../../../output/B/pose1+2+4+6/step5.3/all_poses.png) | PASS | 4 | 4 | 4 | 0 | 91.71 | 9.44 | 115.38 | +23.67 | +25.80 | 5.29 | 7.92 | +2.63 | +49.78 |
| [pose1+2+4+7](../../../output/B/pose1+2+4+7/step5.3/all_poses.png) | PASS | 4 | 4 | 4 | 0 | 55.99 | 4.92 | 795.06 | +739.07 | +1319.96 | 4.95 | 8.59 | +3.65 | +73.72 |
| [pose2+3+4+7](../../../output/B/pose2+3+4+7/step5.3/all_poses.png) | PASS | 4 | 4 | 4 | 0 | 95.51 | 16.23 | 166.20 | +70.69 | +74.01 | 5.72 | 8.00 | +2.28 | +39.96 |
| [pose5+6+23+29](../../../output/B/pose5+6+23+29/step5.3/all_poses.png) | PASS | 4 | 4 | 4 | 0 | 69.29 | 7.10 | 127.42 | +58.13 | +83.89 | 6.76 | 8.12 | +1.36 | +20.08 |
| [pose8+9+13+30](../../../output/B/pose8+9+13+30/step5.3/all_poses.png) | PASS | 4 | 3 | 3 | 1 | 87.98 | 19.59 | 104.17 | +16.19 | +18.40 | 4.89 | 6.01 | +1.12 | +22.96 |
| [illegal/pose1+2+29](../../../output/B/illegal/pose1+2+29/step5.3/all_poses.png) | PASS | 3 | 2 | 2 | 1 | 111.72 | 17.17 | 63.52 | -48.20 | -43.15 | 5.31 | 6.21 | +0.90 | +16.97 |
| [illegal/pose5+13+27](../../../output/B/illegal/pose5+13+27/step5.3/all_poses.png) | PASS | 3 | 1 | 1 | 2 | 227.62 | 17.96 | 515.83 | +288.20 | +126.61 | 5.76 | 5.76 | +0.00 | +0.03 |
| [pose1+12+29](../../../output/B/pose1+12+29/step5.3/all_poses.png) | PASS | 3 | 3 | 3 | 0 | 101.44 | 5.65 | 75.70 | -25.74 | -25.37 | 5.73 | 6.79 | +1.06 | +18.47 |
| [pose18+23+24](../../../output/B/pose18+23+24/step5.3/all_poses.png) | PASS | 3 | 3 | 3 | 0 | 72.11 | 21.88 | 203.05 | +130.94 | +181.57 | 6.48 | 7.37 | +0.89 | +13.72 |
| [pose4+5+7](../../../output/B/pose4+5+7/step5.3/all_poses.png) | PASS | 3 | 3 | 3 | 0 | 104.21 | 17.31 | 414.93 | +310.72 | +298.16 | 5.19 | 6.45 | +1.25 | +24.11 |
| [pose8+10+19](../../../output/B/pose8+10+19/step5.3/all_poses.png) | PASS | 3 | 3 | 3 | 0 | 80.27 | 27.95 | 115.80 | +35.53 | +44.27 | 6.19 | 11.97 | +5.78 | +93.36 |
| [illegal/pose11+19](../../../output/B/illegal/pose11+19/step5.3/all_poses.png) | PASS | 2 | 1 | 1 | 1 | 177.57 | 30.57 | 75.48 | -102.09 | -57.49 | 5.88 | 5.75 | -0.13 | -2.26 |
| [illegal/pose2+29](../../../output/B/illegal/pose2+29/step5.3/all_poses.png) | PASS | 2 | 1 | 1 | 1 | 136.85 | 11.94 | 94.88 | -41.98 | -30.67 | 4.41 | 4.21 | -0.20 | -4.51 |
| [pose19+28](../../../output/B/pose19+28/step5.3/all_poses.png) | PASS | 2 | 2 | 2 | 0 | 152.26 | 8.65 | 129.21 | -23.05 | -15.14 | 6.84 | 7.97 | +1.13 | +16.51 |
| [pose2+20](../../../output/B/pose2+20/step5.3/all_poses.png) | PASS | 2 | 2 | 2 | 0 | 138.83 | 7.82 | 65.88 | -72.95 | -52.54 | 4.42 | 4.34 | -0.09 | -1.92 |
| [pose3+15](../../../output/B/pose3+15/step5.3/all_poses.png) | PASS | 2 | 2 | 2 | 0 | 145.54 | 13.18 | 124.02 | -21.52 | -14.78 | 5.19 | 7.31 | +2.12 | +40.87 |
| [pose8+21](../../../output/B/pose8+21/step5.3/all_poses.png) | PASS | 2 | 2 | 2 | 0 | 106.21 | 8.17 | 61.71 | -44.50 | -41.90 | 3.86 | 4.14 | +0.28 | +7.33 |

## 每个object pose的角色与支撑成员

| Set | 原位复用成员 | Juxtapose成员 | 支撑安装→object pose |
|---|---|---|---|
| pose1+2+3+4+5+6+7+8+9+10 | pose_4 | pose_1, pose_2, pose_3, pose_5, pose_6, pose_7, pose_8, pose_9, pose_10 | S1: pose_1, pose_2, pose_3, pose_4, pose_5, pose_6, pose_7, pose_8, pose_9, pose_10 |
| pose1+3+5+7+9+12+16+21+23+27 | pose_21 | pose_1, pose_3, pose_5, pose_7, pose_9, pose_12, pose_16, pose_23, pose_27 | S1: pose_1, pose_3, pose_5, pose_7, pose_9, pose_12, pose_16, pose_21, pose_23, pose_27 |
| pose1+2+3+4+5+6+7+8+9 | pose_3 | pose_1, pose_2, pose_4, pose_5, pose_6, pose_7, pose_8, pose_9 | S1: pose_1, pose_2, pose_3, pose_4, pose_5, pose_6, pose_7, pose_8, pose_9 |
| pose1+4+7+9+12+16+21+23+27 | pose_7, pose_9 | pose_1, pose_4, pose_12, pose_16, pose_21, pose_23, pose_27 | S1: pose_1, pose_4, pose_7, pose_12, pose_16; S2: pose_9, pose_21, pose_23, pose_27 |
| pose1+2+3+4+5+6+7+8 | pose_4, pose_6, pose_7 | pose_1, pose_2, pose_3, pose_5, pose_8 | S1: pose_1, pose_2, pose_3, pose_4; S2: pose_5, pose_7, pose_8; S3: pose_6 |
| pose1+2+4+5+6+7+10+11 | pose_11 | pose_1, pose_2, pose_4, pose_5, pose_6, pose_7, pose_10 | S1: pose_1, pose_2, pose_4, pose_5, pose_6, pose_7, pose_10, pose_11 |
| pose1+3+7+10+14+18+23+27 | pose_7, pose_18 | pose_1, pose_3, pose_10, pose_14, pose_23, pose_27 | S1: pose_1, pose_18; S2: pose_3, pose_7, pose_10, pose_14, pose_23, pose_27 |
| pose1+2+3+4+5+6+7 | pose_2, pose_3, pose_4, pose_6, pose_7 | pose_1, pose_5 | S1: pose_1, pose_4, pose_5; S2: pose_2; S3: pose_3; S4: pose_6; S5: pose_7 |
| pose1+2+4+5+6+7+11 | pose_1, pose_6 | pose_2, pose_4, pose_5, pose_7, pose_11 | S1: pose_1, pose_4; S2: pose_2, pose_5, pose_6, pose_7, pose_11 |
| pose1+4+7+12+21+23+27 | pose_7 | pose_1, pose_4, pose_12, pose_21, pose_23, pose_27 | S1: pose_1, pose_4, pose_7, pose_12, pose_21, pose_23, pose_27 |
| illegal/pose4+7+12+21+23+27 | pose_7 | pose_4, pose_12, pose_21, pose_23, pose_27 | S1: pose_4, pose_7, pose_12, pose_21, pose_23, pose_27 |
| illegal/pose5+12+20+24+27+28 | pose_27 | pose_5, pose_12, pose_20, pose_24, pose_28 | S1: pose_5, pose_12, pose_20, pose_24, pose_27, pose_28 |
| pose1+2+3+4+5+6 | pose_1, pose_2 | pose_3, pose_4, pose_5, pose_6 | S1: pose_1; S2: pose_2, pose_3, pose_4, pose_5, pose_6 |
| pose1+2+3+4+6+7 | pose_3 | pose_1, pose_2, pose_4, pose_6, pose_7 | S1: pose_1, pose_2, pose_3, pose_4, pose_6, pose_7 |
| pose1+2+3+4+7+27 | pose_4, pose_7 | pose_1, pose_2, pose_3, pose_27 | S1: pose_1, pose_3, pose_7, pose_27; S2: pose_2, pose_4 |
| pose1+2+4+5+6+11 | pose_1 | pose_2, pose_4, pose_5, pose_6, pose_11 | S1: pose_1, pose_2, pose_4, pose_5, pose_6, pose_11 |
| pose1+2+4+5+6+7 | pose_2, pose_6 | pose_1, pose_4, pose_5, pose_7 | S1: pose_1, pose_2, pose_4, pose_5, pose_7; S2: pose_6 |
| pose1+4+7+12+21+27 | pose_1, pose_4, pose_12, pose_21 | pose_7, pose_27 | S1: pose_1, pose_7; S2: pose_4, pose_27; S3: pose_12; S4: pose_21 |
| pose1+6+11+13+14+17 | pose_11 | pose_1, pose_6, pose_13, pose_14, pose_17 | S1: pose_1, pose_6, pose_11, pose_13, pose_14, pose_17 |
| pose4+5+8+9+19+23 | pose_5 | pose_4, pose_8, pose_9, pose_19, pose_23 | S1: pose_4, pose_5, pose_8, pose_9, pose_19, pose_23 |
| illegal/pose1+2+6+7+29 | pose_1, pose_6, pose_7, pose_29 | pose_2 | S1: pose_1, pose_2; S2: pose_6; S3: pose_7; S4: pose_29 |
| illegal/pose5+6+13+15+27 | pose_6, pose_15 | pose_5, pose_13, pose_27 | S1: pose_5, pose_13, pose_15, pose_27; S2: pose_6 |
| pose1+2+3+4+27 | pose_1, pose_2, pose_4, pose_27 | pose_3 | S1: pose_1; S2: pose_2; S3: pose_3, pose_4; S4: pose_27 |
| pose1+2+3+4+6 | pose_1, pose_2, pose_3, pose_4, pose_6 | — | S1: pose_1; S2: pose_2; S3: pose_3; S4: pose_4; S5: pose_6 |
| pose1+2+3+7+27 | pose_1, pose_2, pose_3, pose_7, pose_27 | — | S1: pose_1; S2: pose_2; S3: pose_3; S4: pose_7; S5: pose_27 |
| pose1+2+4+5+6 | pose_1, pose_2, pose_4, pose_5, pose_6 | — | S1: pose_1; S2: pose_2; S3: pose_4; S4: pose_5; S5: pose_6 |
| pose1+4+7+9+24 | pose_1, pose_4, pose_7, pose_9 | pose_24 | S1: pose_1; S2: pose_4, pose_24; S3: pose_7; S4: pose_9 |
| pose12+16+19+21+27 | pose_12, pose_16, pose_19, pose_21, pose_27 | — | S1: pose_12; S2: pose_16; S3: pose_19; S4: pose_21; S5: pose_27 |
| pose5+6+11+23+29 | pose_6, pose_11, pose_29 | pose_5, pose_23 | S1: pose_5, pose_29; S2: pose_6, pose_23; S3: pose_11 |
| pose6+10+13+17+30 | pose_6, pose_10, pose_13, pose_30 | pose_17 | S1: pose_6; S2: pose_10, pose_17; S3: pose_13; S4: pose_30 |
| illegal/pose1+2+6+15 | pose_2 | pose_1, pose_6, pose_15 | S1: pose_1, pose_2, pose_6, pose_15 |
| illegal/pose7+11+13+19 | pose_7, pose_13 | pose_11, pose_19 | S1: pose_7, pose_11; S2: pose_13, pose_19 |
| pose1+11+14+27 | pose_1, pose_11, pose_14, pose_27 | — | S1: pose_1; S2: pose_11; S3: pose_14; S4: pose_27 |
| pose1+2+3+27 | pose_1, pose_2, pose_27 | pose_3 | S1: pose_1; S2: pose_2; S3: pose_3, pose_27 |
| pose1+2+4+19 | pose_1, pose_2, pose_4, pose_19 | — | S1: pose_1; S2: pose_2; S3: pose_4; S4: pose_19 |
| pose1+2+4+6 | pose_1, pose_2, pose_4, pose_6 | — | S1: pose_1; S2: pose_2; S3: pose_4; S4: pose_6 |
| pose1+2+4+7 | pose_1, pose_2, pose_4, pose_7 | — | S1: pose_1; S2: pose_2; S3: pose_4; S4: pose_7 |
| pose2+3+4+7 | pose_2, pose_3, pose_4, pose_7 | — | S1: pose_2; S2: pose_3; S3: pose_4; S4: pose_7 |
| pose5+6+23+29 | pose_5, pose_6, pose_23, pose_29 | — | S1: pose_5; S2: pose_6; S3: pose_23; S4: pose_29 |
| pose8+9+13+30 | pose_8, pose_9, pose_30 | pose_13 | S1: pose_8; S2: pose_9, pose_13; S3: pose_30 |
| illegal/pose1+2+29 | pose_1, pose_29 | pose_2 | S1: pose_1; S2: pose_2, pose_29 |
| illegal/pose5+13+27 | pose_27 | pose_5, pose_13 | S1: pose_5, pose_13, pose_27 |
| pose1+12+29 | pose_1, pose_12, pose_29 | — | S1: pose_1; S2: pose_12; S3: pose_29 |
| pose18+23+24 | pose_18, pose_23, pose_24 | — | S1: pose_18; S2: pose_23; S3: pose_24 |
| pose4+5+7 | pose_4, pose_5, pose_7 | — | S1: pose_4; S2: pose_5; S3: pose_7 |
| pose8+10+19 | pose_8, pose_10, pose_19 | — | S1: pose_8; S2: pose_10; S3: pose_19 |
| illegal/pose11+19 | pose_19 | pose_11 | S1: pose_11, pose_19 |
| illegal/pose2+29 | pose_29 | pose_2 | S1: pose_2, pose_29 |
| pose19+28 | pose_19, pose_28 | — | S1: pose_19; S2: pose_28 |
| pose2+20 | pose_2, pose_20 | — | S1: pose_2; S2: pose_20 |
| pose3+15 | pose_3, pose_15 | — | S1: pose_3; S2: pose_15 |
| pose8+21 | pose_8, pose_21 | — | S1: pose_8; S2: pose_21 |

## 验收与来源

原需求仍是每pose32768个。Step5.2每次实际接触删减必须通过完整需求，Step5.3再次检查实际序列化实体、地面包围、合法空间及受力；不借用旧暖启动布局。

Step5.2优先读取Step4名义材料；若其真实接触不通过，则尝试同一布局已保存的构造实体。实际删减输入来源与体积、相对已构造参考的变化均列在CSV中。

[逐组完整CSV](evaluation_dynamic_postprocess_all52_20261012_01.csv) · [每个object pose角色CSV](evaluation_dynamic_postprocess_all52_20261012_01_poses.csv) · [结构化结果](data/system_dynamic_postprocess_all52_20261012_01/evaluation_results.json)。

图直接在各自step5.1、step5.2、step5.3、evaluation目录，无HTML。Step4及历史数值受保护。Evaluation从保存模型计算，没有再次求解受力或Boolean。

失败记录：

- `pose1+2+3+4+5+6+7+8+9+10`：STEP4_FAIL；No certified Step4 layout

- `pose1+4+7+9+12+16+21+23+27`：STEP4_FAIL；No certified Step4 layout

- `pose1+2+4+5+6+7`：PRUNING_FAIL；Accepted constructed material fails full contact certification: {'0': 32768, '1': 32768, '2': 32748}

第一轮后处理44/50组完成，有界地脚参数与保留更多材料的重试后49/50组完成。各组实际Step5.2/5.3来源均在CSV中，失败尝试保留。

通过组BBX合计：无支撑239.082L／Step4名义324.424L／Step5.2删减315.041L／Step5.3完整413.720L。

材料体积统一取导出三角网格有向体积的绝对值；CSV另列构造内核体积及差值。最大差值0.040cm³；物理验收条件保持原值。
