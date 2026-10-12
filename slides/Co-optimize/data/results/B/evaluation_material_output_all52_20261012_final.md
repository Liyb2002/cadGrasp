# 全部 B 集合的 Evaluation 与 pose 数量

批次 `material_output_all52_20261012_final`：52/52 组完成，263 个 object pose 实例，112 个支撑安装 pose。每组是一件共享支撑实体，任务依次执行。

每组图片直接放在 `slides/Co-optimize/output/B/[set]/step5.2/`，`illegal/` 集合保留其目录前缀。`overview.png` 是完整支撑，`all_poses.png` 汇总全部任务姿态，`pose_N.png` 是单个物体姿态，`evaluation.png` 从左到右是无支撑、加 Step4、加完整 Step5.2；不使用 HTML。

独立集合的 BBX 统计和：**254.41 / 389.14 / 433.19 L**。完整实体材料统计和 **12527.10 cm³**。BBX 是空间包围盒；材料体积是实体用量。跨集合统计和不代表同一工作站的总体积。

与保存的旧 `clearance_compact_v1` 对比：完整 BBX 统计和从 834.76 L 降到 433.19 L，降低 48.11%，52/52 组减小。这是使用已记录暖启动的保存结果对比，不是冷启动算法性能对照。

无支撑基线保留原任务朝向，将水平包围盒中心对齐、最低点放在 z=0，取全部依次使用的姿态的最小 XYZ 包围盒。有支撑两列使用保存的真实布局及全部最终安装姿态。Evaluation 只计最终静态工作位置，不计退出、搬运和机器人运动。

支撑 pose 按保存的实际世界位置和朝向分组。每个支撑 pose 服务的物体 pose 数量之和等于该集合的 object pose 数量；Juxtapose 与原位复用是物体落座分类，不是支撑 pose 数。

| Set（打开全部姿态图） | object pose | 支撑 pose | 每个支撑 pose 服务数量 | 无支撑 BBX (L) | Step4 BBX (L) | Step5.2 BBX (L) | 材料 (cm³) | Evaluation 图 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| [pose1+2+3+4+5+6+7+8+9+10](../../../output/B/pose1+2+3+4+5+6+7+8+9+10/step5.2/all_poses.png) | 10 | 1 | 10 | 5.27 | 20.89 | 21.05 | 539.55 | [PNG](../../../output/B/pose1+2+3+4+5+6+7+8+9+10/step5.2/evaluation.png) |
| [pose1+3+5+7+9+12+16+21+23+27](../../../output/B/pose1+3+5+7+9+12+16+21+23+27/step5.2/all_poses.png) | 10 | 2 | 7, 3 | 5.47 | 18.46 | 18.46 | 248.42 | [PNG](../../../output/B/pose1+3+5+7+9+12+16+21+23+27/step5.2/evaluation.png) |
| [pose1+2+3+4+5+6+7+8+9](../../../output/B/pose1+2+3+4+5+6+7+8+9/step5.2/all_poses.png) | 9 | 1 | 9 | 5.27 | 19.09 | 19.09 | 472.80 | [PNG](../../../output/B/pose1+2+3+4+5+6+7+8+9/step5.2/evaluation.png) |
| [pose1+4+7+9+12+16+21+23+27](../../../output/B/pose1+4+7+9+12+16+21+23+27/step5.2/all_poses.png) | 9 | 1 | 9 | 4.79 | 21.20 | 21.20 | 530.09 | [PNG](../../../output/B/pose1+4+7+9+12+16+21+23+27/step5.2/evaluation.png) |
| [pose1+2+3+4+5+6+7+8](../../../output/B/pose1+2+3+4+5+6+7+8/step5.2/all_poses.png) | 8 | 4 | 2, 3, 2, 1 | 5.27 | 9.33 | 17.86 | 683.31 | [PNG](../../../output/B/pose1+2+3+4+5+6+7+8/step5.2/evaluation.png) |
| [pose1+2+4+5+6+7+10+11](../../../output/B/pose1+2+4+5+6+7+10+11/step5.2/all_poses.png) | 8 | 4 | 1, 5, 1, 1 | 5.27 | 7.95 | 14.33 | 1593.40 | [PNG](../../../output/B/pose1+2+4+5+6+7+10+11/step5.2/evaluation.png) |
| [pose1+3+7+10+14+18+23+27](../../../output/B/pose1+3+7+10+14+18+23+27/step5.2/all_poses.png) | 8 | 4 | 3, 1, 3, 1 | 6.04 | 9.94 | 13.91 | 350.16 | [PNG](../../../output/B/pose1+3+7+10+14+18+23+27/step5.2/evaluation.png) |
| [pose1+2+3+4+5+6+7](../../../output/B/pose1+2+3+4+5+6+7/step5.2/all_poses.png) | 7 | 3 | 2, 4, 1 | 5.27 | 13.37 | 19.22 | 242.81 | [PNG](../../../output/B/pose1+2+3+4+5+6+7/step5.2/evaluation.png) |
| [pose1+2+4+5+6+7+11](../../../output/B/pose1+2+4+5+6+7+11/step5.2/all_poses.png) | 7 | 3 | 2, 3, 2 | 5.27 | 9.76 | 10.62 | 284.63 | [PNG](../../../output/B/pose1+2+4+5+6+7+11/step5.2/evaluation.png) |
| [pose1+4+7+12+21+23+27](../../../output/B/pose1+4+7+12+21+23+27/step5.2/all_poses.png) | 7 | 4 | 4, 1, 1, 1 | 4.79 | 7.34 | 10.00 | 262.39 | [PNG](../../../output/B/pose1+4+7+12+21+23+27/step5.2/evaluation.png) |
| [illegal/pose4+7+12+21+23+27](../../../output/B/illegal/pose4+7+12+21+23+27/step5.2/all_poses.png) | 6 | 2 | 4, 2 | 4.79 | 6.24 | 6.44 | 168.29 | [PNG](../../../output/B/illegal/pose4+7+12+21+23+27/step5.2/evaluation.png) |
| [illegal/pose5+12+20+24+27+28](../../../output/B/illegal/pose5+12+20+24+27+28/step5.2/all_poses.png) | 6 | 2 | 5, 1 | 5.18 | 9.96 | 10.71 | 276.11 | [PNG](../../../output/B/illegal/pose5+12+20+24+27+28/step5.2/evaluation.png) |
| [pose1+2+3+4+5+6](../../../output/B/pose1+2+3+4+5+6/step5.2/all_poses.png) | 6 | 2 | 3, 3 | 5.27 | 6.59 | 6.87 | 245.41 | [PNG](../../../output/B/pose1+2+3+4+5+6/step5.2/evaluation.png) |
| [pose1+2+3+4+6+7](../../../output/B/pose1+2+3+4+6+7/step5.2/all_poses.png) | 6 | 2 | 4, 2 | 4.67 | 5.61 | 5.61 | 241.21 | [PNG](../../../output/B/pose1+2+3+4+6+7/step5.2/evaluation.png) |
| [pose1+2+3+4+7+27](../../../output/B/pose1+2+3+4+7+27/step5.2/all_poses.png) | 6 | 4 | 2, 2, 1, 1 | 4.88 | 7.08 | 7.78 | 155.32 | [PNG](../../../output/B/pose1+2+3+4+7+27/step5.2/evaluation.png) |
| [pose1+2+4+5+6+11](../../../output/B/pose1+2+4+5+6+11/step5.2/all_poses.png) | 6 | 2 | 4, 2 | 5.27 | 7.28 | 7.47 | 202.58 | [PNG](../../../output/B/pose1+2+4+5+6+11/step5.2/evaluation.png) |
| [pose1+2+4+5+6+7](../../../output/B/pose1+2+4+5+6+7/step5.2/all_poses.png) | 6 | 4 | 2, 1, 1, 2 | 5.27 | 6.31 | 7.03 | 144.33 | [PNG](../../../output/B/pose1+2+4+5+6+7/step5.2/evaluation.png) |
| [pose1+4+7+12+21+27](../../../output/B/pose1+4+7+12+21+27/step5.2/all_poses.png) | 6 | 3 | 1, 3, 2 | 4.57 | 8.04 | 8.88 | 170.94 | [PNG](../../../output/B/pose1+4+7+12+21+27/step5.2/evaluation.png) |
| [pose1+6+11+13+14+17](../../../output/B/pose1+6+11+13+14+17/step5.2/all_poses.png) | 6 | 4 | 1, 3, 1, 1 | 4.96 | 5.98 | 7.99 | 120.88 | [PNG](../../../output/B/pose1+6+11+13+14+17/step5.2/evaluation.png) |
| [pose4+5+8+9+19+23](../../../output/B/pose4+5+8+9+19+23/step5.2/all_poses.png) | 6 | 3 | 3, 2, 1 | 5.98 | 7.52 | 9.57 | 157.16 | [PNG](../../../output/B/pose4+5+8+9+19+23/step5.2/evaluation.png) |
| [illegal/pose1+2+6+7+29](../../../output/B/illegal/pose1+2+6+7+29/step5.2/all_poses.png) | 5 | 2 | 4, 1 | 4.67 | 5.17 | 5.33 | 154.83 | [PNG](../../../output/B/illegal/pose1+2+6+7+29/step5.2/evaluation.png) |
| [illegal/pose5+6+13+15+27](../../../output/B/illegal/pose5+6+13+15+27/step5.2/all_poses.png) | 5 | 2 | 4, 1 | 5.20 | 6.78 | 6.80 | 206.60 | [PNG](../../../output/B/illegal/pose5+6+13+15+27/step5.2/evaluation.png) |
| [pose1+2+3+4+27](../../../output/B/pose1+2+3+4+27/step5.2/all_poses.png) | 5 | 3 | 2, 2, 1 | 4.88 | 6.24 | 7.23 | 161.54 | [PNG](../../../output/B/pose1+2+3+4+27/step5.2/evaluation.png) |
| [pose1+2+3+4+6](../../../output/B/pose1+2+3+4+6/step5.2/all_poses.png) | 5 | 3 | 2, 1, 2 | 4.62 | 6.29 | 6.64 | 107.52 | [PNG](../../../output/B/pose1+2+3+4+6/step5.2/evaluation.png) |
| [pose1+2+3+7+27](../../../output/B/pose1+2+3+7+27/step5.2/all_poses.png) | 5 | 2 | 3, 2 | 4.88 | 6.37 | 6.48 | 164.96 | [PNG](../../../output/B/pose1+2+3+7+27/step5.2/evaluation.png) |
| [pose1+2+4+5+6](../../../output/B/pose1+2+4+5+6/step5.2/all_poses.png) | 5 | 3 | 3, 1, 1 | 5.27 | 5.45 | 5.59 | 154.10 | [PNG](../../../output/B/pose1+2+4+5+6/step5.2/evaluation.png) |
| [pose1+4+7+9+24](../../../output/B/pose1+4+7+9+24/step5.2/all_poses.png) | 5 | 2 | 4, 1 | 4.17 | 5.40 | 5.40 | 108.89 | [PNG](../../../output/B/pose1+4+7+9+24/step5.2/evaluation.png) |
| [pose12+16+19+21+27](../../../output/B/pose12+16+19+21+27/step5.2/all_poses.png) | 5 | 2 | 3, 2 | 5.19 | 7.79 | 8.00 | 239.77 | [PNG](../../../output/B/pose12+16+19+21+27/step5.2/evaluation.png) |
| [pose5+6+11+23+29](../../../output/B/pose5+6+11+23+29/step5.2/all_poses.png) | 5 | 2 | 4, 1 | 5.50 | 6.74 | 7.19 | 179.54 | [PNG](../../../output/B/pose5+6+11+23+29/step5.2/evaluation.png) |
| [pose6+10+13+17+30](../../../output/B/pose6+10+13+17+30/step5.2/all_poses.png) | 5 | 3 | 2, 2, 1 | 4.48 | 6.95 | 8.23 | 236.06 | [PNG](../../../output/B/pose6+10+13+17+30/step5.2/evaluation.png) |
| [illegal/pose1+2+6+15](../../../output/B/illegal/pose1+2+6+15/step5.2/all_poses.png) | 4 | 2 | 3, 1 | 4.64 | 5.53 | 6.23 | 183.42 | [PNG](../../../output/B/illegal/pose1+2+6+15/step5.2/evaluation.png) |
| [illegal/pose7+11+13+19](../../../output/B/illegal/pose7+11+13+19/step5.2/all_poses.png) | 4 | 2 | 1, 3 | 5.01 | 6.70 | 7.06 | 213.84 | [PNG](../../../output/B/illegal/pose7+11+13+19/step5.2/evaluation.png) |
| [pose1+11+14+27](../../../output/B/pose1+11+14+27/step5.2/all_poses.png) | 4 | 2 | 3, 1 | 4.33 | 5.39 | 5.80 | 148.72 | [PNG](../../../output/B/pose1+11+14+27/step5.2/evaluation.png) |
| [pose1+2+3+27](../../../output/B/pose1+2+3+27/step5.2/all_poses.png) | 4 | 2 | 2, 2 | 4.88 | 5.58 | 5.79 | 113.66 | [PNG](../../../output/B/pose1+2+3+27/step5.2/evaluation.png) |
| [pose1+2+4+19](../../../output/B/pose1+2+4+19/step5.2/all_poses.png) | 4 | 2 | 3, 1 | 5.17 | 6.84 | 7.05 | 157.17 | [PNG](../../../output/B/pose1+2+4+19/step5.2/evaluation.png) |
| [pose1+2+4+6](../../../output/B/pose1+2+4+6/step5.2/all_poses.png) | 4 | 2 | 3, 1 | 4.62 | 6.06 | 6.56 | 177.20 | [PNG](../../../output/B/pose1+2+4+6/step5.2/evaluation.png) |
| [pose1+2+4+7](../../../output/B/pose1+2+4+7/step5.2/all_poses.png) | 4 | 2 | 1, 3 | 4.24 | 5.56 | 5.99 | 161.94 | [PNG](../../../output/B/pose1+2+4+7/step5.2/evaluation.png) |
| [pose2+3+4+7](../../../output/B/pose2+3+4+7/step5.2/all_poses.png) | 4 | 2 | 1, 3 | 4.64 | 4.98 | 5.23 | 112.59 | [PNG](../../../output/B/pose2+3+4+7/step5.2/evaluation.png) |
| [pose5+6+23+29](../../../output/B/pose5+6+23+29/step5.2/all_poses.png) | 4 | 2 | 3, 1 | 5.50 | 6.62 | 7.22 | 145.72 | [PNG](../../../output/B/pose5+6+23+29/step5.2/evaluation.png) |
| [pose8+9+13+30](../../../output/B/pose8+9+13+30/step5.2/all_poses.png) | 4 | 2 | 3, 1 | 4.20 | 5.13 | 5.15 | 161.92 | [PNG](../../../output/B/pose8+9+13+30/step5.2/evaluation.png) |
| [illegal/pose1+2+29](../../../output/B/illegal/pose1+2+29/step5.2/all_poses.png) | 3 | 1 | 3 | 4.58 | 4.78 | 4.78 | 148.45 | [PNG](../../../output/B/illegal/pose1+2+29/step5.2/evaluation.png) |
| [illegal/pose5+13+27](../../../output/B/illegal/pose5+13+27/step5.2/all_poses.png) | 3 | 1 | 3 | 4.56 | 5.14 | 5.14 | 253.73 | [PNG](../../../output/B/illegal/pose5+13+27/step5.2/evaluation.png) |
| [pose1+12+29](../../../output/B/pose1+12+29/step5.2/all_poses.png) | 3 | 1 | 3 | 4.68 | 4.92 | 4.92 | 166.83 | [PNG](../../../output/B/pose1+12+29/step5.2/evaluation.png) |
| [pose18+23+24](../../../output/B/pose18+23+24/step5.2/all_poses.png) | 3 | 1 | 3 | 5.33 | 5.75 | 5.75 | 205.50 | [PNG](../../../output/B/pose18+23+24/step5.2/evaluation.png) |
| [pose4+5+7](../../../output/B/pose4+5+7/step5.2/all_poses.png) | 3 | 1 | 3 | 4.71 | 5.10 | 5.10 | 211.03 | [PNG](../../../output/B/pose4+5+7/step5.2/evaluation.png) |
| [pose8+10+19](../../../output/B/pose8+10+19/step5.2/all_poses.png) | 3 | 1 | 3 | 5.66 | 6.24 | 6.29 | 208.24 | [PNG](../../../output/B/pose8+10+19/step5.2/evaluation.png) |
| [illegal/pose11+19](../../../output/B/illegal/pose11+19/step5.2/all_poses.png) | 2 | 1 | 2 | 5.01 | 5.61 | 5.61 | 186.03 | [PNG](../../../output/B/illegal/pose11+19/step5.2/evaluation.png) |
| [illegal/pose2+29](../../../output/B/illegal/pose2+29/step5.2/all_poses.png) | 2 | 1 | 2 | 4.01 | 4.32 | 4.32 | 141.94 | [PNG](../../../output/B/illegal/pose2+29/step5.2/evaluation.png) |
| [pose19+28](../../../output/B/pose19+28/step5.2/all_poses.png) | 2 | 1 | 2 | 4.28 | 4.94 | 5.13 | 146.80 | [PNG](../../../output/B/pose19+28/step5.2/evaluation.png) |
| [pose2+20](../../../output/B/pose2+20/step5.2/all_poses.png) | 2 | 2 | 1, 1 | 4.11 | 4.18 | 4.18 | 93.99 | [PNG](../../../output/B/pose2+20/step5.2/evaluation.png) |
| [pose3+15](../../../output/B/pose3+15/step5.2/all_poses.png) | 2 | 1 | 2 | 4.41 | 4.79 | 5.02 | 201.13 | [PNG](../../../output/B/pose3+15/step5.2/evaluation.png) |
| [pose8+21](../../../output/B/pose8+21/step5.2/all_poses.png) | 2 | 1 | 2 | 3.49 | 3.85 | 3.85 | 183.65 | [PNG](../../../output/B/pose8+21/step5.2/evaluation.png) |

每个支撑安装 pose 服务的 object pose 成员：

| Set | 支撑 pose → object pose 成员 | Juxtapose | 原位复用 |
|---|---|---:|---:|
| pose1+2+3+4+5+6+7+8+9+10 | S1: pose_1, pose_2, pose_3, pose_4, pose_5, pose_6, pose_7, pose_8, pose_9, pose_10 | 9 | 1 |
| pose1+3+5+7+9+12+16+21+23+27 | S1: pose_1, pose_3, pose_7, pose_9, pose_12, pose_21, pose_27; S2: pose_5, pose_16, pose_23 | 8 | 2 |
| pose1+2+3+4+5+6+7+8+9 | S1: pose_1, pose_2, pose_3, pose_4, pose_5, pose_6, pose_7, pose_8, pose_9 | 8 | 1 |
| pose1+4+7+9+12+16+21+23+27 | S1: pose_1, pose_4, pose_7, pose_9, pose_12, pose_16, pose_21, pose_23, pose_27 | 8 | 1 |
| pose1+2+3+4+5+6+7+8 | S1: pose_1, pose_4; S2: pose_2, pose_5, pose_7; S3: pose_3, pose_8; S4: pose_6 | 4 | 4 |
| pose1+2+4+5+6+7+10+11 | S1: pose_1; S2: pose_2, pose_4, pose_7, pose_10, pose_11; S3: pose_5; S4: pose_6 | 4 | 4 |
| pose1+3+7+10+14+18+23+27 | S1: pose_1, pose_3, pose_18; S2: pose_7; S3: pose_10, pose_14, pose_23; S4: pose_27 | 4 | 4 |
| pose1+2+3+4+5+6+7 | S1: pose_1, pose_3; S2: pose_2, pose_4, pose_5, pose_7; S3: pose_6 | 4 | 3 |
| pose1+2+4+5+6+7+11 | S1: pose_1, pose_5; S2: pose_2, pose_4, pose_11; S3: pose_6, pose_7 | 4 | 3 |
| pose1+4+7+12+21+23+27 | S1: pose_1, pose_12, pose_21, pose_23; S2: pose_4; S3: pose_7; S4: pose_27 | 3 | 4 |
| illegal/pose4+7+12+21+23+27 | S1: pose_4, pose_21, pose_23, pose_27; S2: pose_7, pose_12 | 4 | 2 |
| illegal/pose5+12+20+24+27+28 | S1: pose_5, pose_12, pose_20, pose_24, pose_27; S2: pose_28 | 4 | 2 |
| pose1+2+3+4+5+6 | S1: pose_1, pose_2, pose_5; S2: pose_3, pose_4, pose_6 | 4 | 2 |
| pose1+2+3+4+6+7 | S1: pose_1, pose_2, pose_3, pose_7; S2: pose_4, pose_6 | 4 | 2 |
| pose1+2+3+4+7+27 | S1: pose_1, pose_4; S2: pose_2, pose_7; S3: pose_3; S4: pose_27 | 2 | 4 |
| pose1+2+4+5+6+11 | S1: pose_1, pose_2, pose_6, pose_11; S2: pose_4, pose_5 | 4 | 2 |
| pose1+2+4+5+6+7 | S1: pose_1, pose_4; S2: pose_2; S3: pose_5; S4: pose_6, pose_7 | 2 | 4 |
| pose1+4+7+12+21+27 | S1: pose_1; S2: pose_4, pose_12, pose_21; S3: pose_7, pose_27 | 3 | 3 |
| pose1+6+11+13+14+17 | S1: pose_1; S2: pose_6, pose_11, pose_14; S3: pose_13; S4: pose_17 | 2 | 4 |
| pose4+5+8+9+19+23 | S1: pose_4, pose_9, pose_23; S2: pose_5, pose_19; S3: pose_8 | 3 | 3 |
| illegal/pose1+2+6+7+29 | S1: pose_1, pose_6, pose_7, pose_29; S2: pose_2 | 3 | 2 |
| illegal/pose5+6+13+15+27 | S1: pose_5, pose_13, pose_15, pose_27; S2: pose_6 | 3 | 2 |
| pose1+2+3+4+27 | S1: pose_1, pose_2; S2: pose_3, pose_27; S3: pose_4 | 2 | 3 |
| pose1+2+3+4+6 | S1: pose_1, pose_2; S2: pose_3; S3: pose_4, pose_6 | 2 | 3 |
| pose1+2+3+7+27 | S1: pose_1, pose_2, pose_3; S2: pose_7, pose_27 | 3 | 2 |
| pose1+2+4+5+6 | S1: pose_1, pose_2, pose_6; S2: pose_4; S3: pose_5 | 2 | 3 |
| pose1+4+7+9+24 | S1: pose_1, pose_4, pose_9, pose_24; S2: pose_7 | 3 | 2 |
| pose12+16+19+21+27 | S1: pose_12, pose_21, pose_27; S2: pose_16, pose_19 | 3 | 2 |
| pose5+6+11+23+29 | S1: pose_5, pose_6, pose_23, pose_29; S2: pose_11 | 3 | 2 |
| pose6+10+13+17+30 | S1: pose_6, pose_30; S2: pose_10, pose_17; S3: pose_13 | 2 | 3 |
| illegal/pose1+2+6+15 | S1: pose_1, pose_2, pose_15; S2: pose_6 | 2 | 2 |
| illegal/pose7+11+13+19 | S1: pose_7; S2: pose_11, pose_13, pose_19 | 2 | 2 |
| pose1+11+14+27 | S1: pose_1, pose_11, pose_14; S2: pose_27 | 2 | 2 |
| pose1+2+3+27 | S1: pose_1, pose_2; S2: pose_3, pose_27 | 2 | 2 |
| pose1+2+4+19 | S1: pose_1, pose_2, pose_19; S2: pose_4 | 2 | 2 |
| pose1+2+4+6 | S1: pose_1, pose_4, pose_6; S2: pose_2 | 2 | 2 |
| pose1+2+4+7 | S1: pose_1; S2: pose_2, pose_4, pose_7 | 2 | 2 |
| pose2+3+4+7 | S1: pose_2; S2: pose_3, pose_4, pose_7 | 2 | 2 |
| pose5+6+23+29 | S1: pose_5, pose_6, pose_23; S2: pose_29 | 2 | 2 |
| pose8+9+13+30 | S1: pose_8, pose_9, pose_30; S2: pose_13 | 2 | 2 |
| illegal/pose1+2+29 | S1: pose_1, pose_2, pose_29 | 2 | 1 |
| illegal/pose5+13+27 | S1: pose_5, pose_13, pose_27 | 3 | 0 |
| pose1+12+29 | S1: pose_1, pose_12, pose_29 | 2 | 1 |
| pose18+23+24 | S1: pose_18, pose_23, pose_24 | 3 | 0 |
| pose4+5+7 | S1: pose_4, pose_5, pose_7 | 2 | 1 |
| pose8+10+19 | S1: pose_8, pose_10, pose_19 | 3 | 0 |
| illegal/pose11+19 | S1: pose_11, pose_19 | 1 | 1 |
| illegal/pose2+29 | S1: pose_2, pose_29 | 1 | 1 |
| pose19+28 | S1: pose_19, pose_28 | 1 | 1 |
| pose2+20 | S1: pose_2; S2: pose_20 | 1 | 1 |
| pose3+15 | S1: pose_3, pose_15 | 1 | 1 |
| pose8+21 | S1: pose_8, pose_21 | 1 | 1 |

Juxtapose 总数 155，原位复用总数 108。72 个支撑 pose 服务多个物体姿态，40 个只服务一个。

[完整 CSV](evaluation_material_output_all52_20261012_final.csv)；[保存的完整 Evaluation JSON](data/evaluation_material_output_all52_20261012_final/summary.json)。

本次发布只复制保存的结果图并整理报告，没有重跑受力、搜索或实体 Boolean。
