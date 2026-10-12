# 5–8 pose 全部26组：混合布局Evaluation

原位复用和Juxtapose使用原Evaluation定义；支撑pose按实际世界变换分组。体积均为完整静态世界XYZ占用盒，单位L；材料单位cm³。
保留各次实际构造通过的模型，按组合目标、完整BBX和材料量选择。暖启动／中断／失败来源在campaign.json中分别记录。

| Pose集合 | 实体完成 | 原位复用 | Juxtapose | 无支撑BBX | Step4 BBX | 完整BBX | 材料 | 支撑pose数 | 每个支撑服务几个物体pose | 同组旧完整BBX |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| [illegal/pose1+2+6+7+29](../../../output/B/illegal/pose1+2+6+7+29/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 2 | 3 | 4.67 | 5.21 | 6.22 | 169.81 | 2 | 4, 1 | 7.65 |
| [illegal/pose4+7+12+21+23+27](../../../output/B/illegal/pose4+7+12+21+23+27/evaluation/mixed_material_all26_20261011_01/index.html) | 完成且组合达标 | 2 | 4 | 4.79 | 7.77 | 8.55 | 201.64 | 2 | 4, 2 | 11.82 |
| [illegal/pose5+12+20+24+27+28](../../../output/B/illegal/pose5+12+20+24+27+28/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 2 | 4 | 5.18 | 22.88 | 24.48 | 376.19 | 2 | 5, 1 | 23.37 |
| [illegal/pose5+6+13+15+27](../../../output/B/illegal/pose5+6+13+15+27/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 2 | 3 | 5.20 | 7.50 | 11.46 | 217.92 | 2 | 4, 1 | 11.84 |
| [pose1+2+3+4+27](../../../output/B/pose1+2+3+4+27/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 3 | 2 | 4.88 | 6.30 | 8.36 | 172.17 | 3 | 2, 2, 1 | 21.78 |
| [pose1+2+3+4+5+6](../../../output/B/pose1+2+3+4+5+6/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 2 | 4 | 5.27 | 8.72 | 8.72 | 244.95 | 2 | 2, 4 | 未完成 |
| [pose1+2+3+4+5+6+7](../../../output/B/pose1+2+3+4+5+6+7/evaluation/mixed_material_coherent_20261012_06/index.html) | 完成且组合达标 | 3 | 4 | 5.27 | 16.62 | 19.42 | 292.44 | 3 | 2, 4, 1 | 27.04 |
| [pose1+2+3+4+5+6+7+8](../../../output/B/pose1+2+3+4+5+6+7+8/evaluation/mixed_material_closed_rescue_20261012_09/index.html) | 完成且组合达标 | 4 | 4 | 5.27 | 30.42 | 112.72 | 435.08 | 4 | 1, 4, 1, 2 | 29.87 |
| [pose1+2+3+4+6](../../../output/B/pose1+2+3+4+6/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 2 | 3 | 4.62 | 8.23 | 9.43 | 224.55 | 2 | 1, 4 | 23.90 |
| [pose1+2+3+4+6+7](../../../output/B/pose1+2+3+4+6+7/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 2 | 4 | 4.67 | 7.18 | 7.67 | 275.85 | 2 | 3, 3 | 未完成 |
| [pose1+2+3+4+7+27](../../../output/B/pose1+2+3+4+7+27/evaluation/mixed_material_probe_rescue_20261011_03/index.html) | 完成且组合达标 | 4 | 2 | 4.88 | 6.25 | 8.58 | 154.57 | 4 | 2, 2, 1, 1 | 13.04 |
| [pose1+2+3+7+27](../../../output/B/pose1+2+3+7+27/evaluation/mixed_material_target_compact_20261011_05/index.html) | 完成且组合达标 | 3 | 2 | 4.88 | 6.88 | 7.66 | 194.01 | 3 | 1, 3, 1 | 20.83 |
| [pose1+2+4+5+6](../../../output/B/pose1+2+4+5+6/evaluation/mixed_material_target_compact_20261011_05/index.html) | 完成且组合达标 | 3 | 2 | 5.27 | 6.49 | 8.03 | 144.37 | 3 | 2, 2, 1 | 未完成 |
| [pose1+2+4+5+6+11](../../../output/B/pose1+2+4+5+6+11/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 2 | 4 | 5.27 | 16.09 | 19.05 | 267.69 | 2 | 5, 1 | 22.23 |
| [pose1+2+4+5+6+7](../../../output/B/pose1+2+4+5+6+7/evaluation/mixed_material_target_compact_20261011_05/index.html) | 完成且组合达标 | 4 | 2 | 5.27 | 7.70 | 9.79 | 166.20 | 4 | 2, 1, 1, 2 | 14.04 |
| [pose1+2+4+5+6+7+10+11](../../../output/B/pose1+2+4+5+6+7+10+11/evaluation/mixed_material_certified_20261012_07/index.html) | 完成且组合达标 | 4 | 4 | 5.27 | 58.35 | 401.45 | 1309.30 | 4 | 1, 1, 5, 1 | 18.40 |
| [pose1+2+4+5+6+7+11](../../../output/B/pose1+2+4+5+6+7+11/evaluation/mixed_material_target_compact_20261011_05/index.html) | 完成且组合达标 | 3 | 4 | 5.27 | 10.50 | 12.40 | 213.09 | 3 | 3, 3, 1 | 28.56 |
| pose1+3+7+10+14+18+23+27 | 未完成 | — | — | — | — | — | — | — | — | 未完成 |
| [pose1+4+7+12+21+23+27](../../../output/B/pose1+4+7+12+21+23+27/evaluation/mixed_material_target_compact_20261011_05/index.html) | 完成且组合达标 | 4 | 3 | 4.79 | 7.40 | 11.11 | 269.68 | 4 | 4, 1, 1, 1 | 14.91 |
| [pose1+4+7+12+21+27](../../../output/B/pose1+4+7+12+21+27/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 4 | 2 | 4.57 | 6.73 | 11.83 | 166.90 | 4 | 3, 1, 1, 1 | 11.74 |
| [pose1+4+7+9+24](../../../output/B/pose1+4+7+9+24/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 2 | 3 | 4.17 | 5.48 | 5.72 | 159.03 | 2 | 2, 3 | 23.68 |
| [pose1+6+11+13+14+17](../../../output/B/pose1+6+11+13+14+17/evaluation/mixed_material_target_compact_20261011_05/index.html) | 完成且组合达标 | 2 | 4 | 4.96 | 11.21 | 13.22 | 199.48 | 2 | 5, 1 | 18.63 |
| [pose12+16+19+21+27](../../../output/B/pose12+16+19+21+27/evaluation/mixed_material_target_compact_20261011_05/index.html) | 完成且组合达标 | 2 | 3 | 5.19 | 8.04 | 8.24 | 240.57 | 2 | 3, 2 | 22.77 |
| [pose4+5+8+9+19+23](../../../output/B/pose4+5+8+9+19+23/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 完成且组合达标 | 2 | 4 | 5.98 | 11.77 | 13.12 | 361.31 | 2 | 4, 2 | 16.41 |
| [pose5+6+11+23+29](../../../output/B/pose5+6+11+23+29/evaluation/mixed_material_target_compact_20261011_05/index.html) | 完成且组合达标 | 2 | 3 | 5.50 | 10.50 | 11.72 | 210.14 | 2 | 1, 4 | 10.11 |
| [pose6+10+13+17+30](../../../output/B/pose6+10+13+17+30/evaluation/mixed_material_target_compact_20261011_05/index.html) | 完成且组合达标 | 2 | 3 | 4.48 | 7.46 | 9.35 | 234.54 | 2 | 4, 1 | 11.49 |

当前完成：25/26；组合达标：25/26。

各支撑pose具体服务成员：

- illegal/pose1+2+6+7+29：[pose_1, pose_6, pose_7, pose_29]；[pose_2]
- illegal/pose4+7+12+21+23+27：[pose_4, pose_21, pose_23, pose_27]；[pose_7, pose_12]
- illegal/pose5+12+20+24+27+28：[pose_5, pose_12, pose_20, pose_24, pose_28]；[pose_27]
- illegal/pose5+6+13+15+27：[pose_5, pose_13, pose_15, pose_27]；[pose_6]
- pose1+2+3+4+27：[pose_1, pose_2]；[pose_3, pose_27]；[pose_4]
- pose1+2+3+4+5+6：[pose_1, pose_2]；[pose_3, pose_4, pose_5, pose_6]
- pose1+2+3+4+5+6+7：[pose_1, pose_3]；[pose_2, pose_4, pose_5, pose_7]；[pose_6]
- pose1+2+3+4+5+6+7+8：[pose_1]；[pose_2, pose_3, pose_4, pose_7]；[pose_5]；[pose_6, pose_8]
- pose1+2+3+4+6：[pose_1]；[pose_2, pose_3, pose_4, pose_6]
- pose1+2+3+4+6+7：[pose_1, pose_4, pose_6]；[pose_2, pose_3, pose_7]
- pose1+2+3+4+7+27：[pose_1, pose_4]；[pose_2, pose_7]；[pose_3]；[pose_27]
- pose1+2+3+7+27：[pose_1]；[pose_2, pose_3, pose_7]；[pose_27]
- pose1+2+4+5+6：[pose_1, pose_2]；[pose_4, pose_5]；[pose_6]
- pose1+2+4+5+6+11：[pose_1, pose_4, pose_5, pose_6, pose_11]；[pose_2]
- pose1+2+4+5+6+7：[pose_1, pose_4]；[pose_2]；[pose_5]；[pose_6, pose_7]
- pose1+2+4+5+6+7+10+11：[pose_1]；[pose_2]；[pose_4, pose_5, pose_7, pose_10, pose_11]；[pose_6]
- pose1+2+4+5+6+7+11：[pose_1, pose_5, pose_7]；[pose_2, pose_4, pose_11]；[pose_6]
- pose1+4+7+12+21+23+27：[pose_1, pose_12, pose_21, pose_23]；[pose_4]；[pose_7]；[pose_27]
- pose1+4+7+12+21+27：[pose_1, pose_12, pose_21]；[pose_4]；[pose_7]；[pose_27]
- pose1+4+7+9+24：[pose_1, pose_7]；[pose_4, pose_9, pose_24]
- pose1+6+11+13+14+17：[pose_1, pose_11, pose_13, pose_14, pose_17]；[pose_6]
- pose12+16+19+21+27：[pose_12, pose_21, pose_27]；[pose_16, pose_19]
- pose4+5+8+9+19+23：[pose_4, pose_5, pose_9, pose_23]；[pose_8, pose_19]
- pose5+6+11+23+29：[pose_5]；[pose_6, pose_11, pose_23, pose_29]
- pose6+10+13+17+30：[pose_6, pose_13, pose_17, pose_30]；[pose_10]
