# 静态工作空间：无支撑叠放基线／真实支撑布局

全部 16 组，共 90 个 pose。只计最终工作位置，单位 L；1 L=1000 cm³。
无支撑基线保留各原任务朝向，将物体的水平包围盒中心对齐、最低点放在z=0，叠放全部pose后取一个BBX。其每轴长度为该轴各pose长度的最大值，达到自由平移时的最小XYZ盒。
任务依次执行；一个 state 服务多个 object pose 不表示多个物体同时装入。Juxtapose 是最终 pose 是否重新落座，multi-object-pose state 是共享关系，两者不是互斥分类。
支撑pose按保存的实际世界位置与朝向分组；同一件支撑换位置或转动后可以有多个支撑pose。每组列出服务的物体pose数量与成员。
有支撑两列使用保存的真实优化位置及全部最终支撑state；不改变物理结果。三列图同等轴测方向、同长度比例，分别居中。

| Pose set | 物体pose数 | 无支撑叠放BBX (L) | +Step4 BBX (L) | +Step5.2 BBX (L) | 支撑pose数 | 每个支撑pose服务的物体pose数 | 物体pose分组 | Juxtapose pose数 | 原位转动复用pose数 |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| [pose1+2+4+5+6+7+11](../../../output/B/pose1+2+4+5+6+7+11/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 7 | 5.27 | 17.91 | 49.94 | 3 | 3, 2, 2 | 1+4+5; 2+11; 6+7 | 4 | 3 |
| [illegal/pose5+12+20+24+27+28](../../../output/B/illegal/pose5+12+20+24+27+28/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 6 | 5.18 | 22.88 | 24.48 | 2 | 5, 1 | 5+12+20+24+28; 27 | 4 | 2 |
| [pose1+2+3+4+5+6](../../../output/B/pose1+2+3+4+5+6/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 6 | 5.27 | 8.72 | 8.72 | 2 | 2, 4 | 1+2; 3+4+5+6 | 4 | 2 |
| [pose1+2+3+4+6+7](../../../output/B/pose1+2+3+4+6+7/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 6 | 4.67 | 7.18 | 7.67 | 2 | 3, 3 | 1+4+6; 2+3+7 | 4 | 2 |
| [pose1+2+3+4+7+27](../../../output/B/pose1+2+3+4+7+27/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 6 | 4.88 | 6.25 | 8.58 | 4 | 2, 2, 1, 1 | 1+4; 2+7; 3; 27 | 2 | 4 |
| [pose1+2+4+5+6+11](../../../output/B/pose1+2+4+5+6+11/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 6 | 5.27 | 16.09 | 19.05 | 2 | 5, 1 | 1+4+5+6+11; 2 | 4 | 2 |
| [pose1+4+7+12+21+27](../../../output/B/pose1+4+7+12+21+27/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 6 | 4.57 | 6.73 | 11.83 | 4 | 3, 1, 1, 1 | 1+12+21; 4; 7; 27 | 2 | 4 |
| [pose1+6+11+13+14+17](../../../output/B/pose1+6+11+13+14+17/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 6 | 4.96 | 22.40 | 52.58 | 2 | 5, 1 | 1+6+11+14+17; 13 | 4 | 2 |
| [pose4+5+8+9+19+23](../../../output/B/pose4+5+8+9+19+23/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 6 | 5.98 | 11.77 | 13.12 | 2 | 4, 2 | 4+5+9+23; 8+19 | 4 | 2 |
| [illegal/pose1+2+6+7+29](../../../output/B/illegal/pose1+2+6+7+29/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 5 | 4.67 | 5.21 | 6.22 | 2 | 4, 1 | 1+6+7+29; 2 | 3 | 2 |
| [illegal/pose5+6+13+15+27](../../../output/B/illegal/pose5+6+13+15+27/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 5 | 5.20 | 7.50 | 11.46 | 2 | 4, 1 | 5+13+15+27; 6 | 3 | 2 |
| [pose1+2+3+4+27](../../../output/B/pose1+2+3+4+27/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 5 | 4.88 | 6.30 | 8.36 | 3 | 2, 2, 1 | 1+2; 3+27; 4 | 2 | 3 |
| [pose1+2+3+4+6](../../../output/B/pose1+2+3+4+6/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 5 | 4.62 | 8.23 | 9.43 | 2 | 1, 4 | 1; 2+3+4+6 | 3 | 2 |
| [pose1+2+3+7+27](../../../output/B/pose1+2+3+7+27/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 5 | 4.88 | 5.87 | 11.00 | 5 | 1, 1, 1, 1, 1 | 1; 2; 3; 7; 27 | 0 | 5 |
| [pose1+2+4+5+6](../../../output/B/pose1+2+4+5+6/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 5 | 5.27 | 6.36 | 9.65 | 5 | 1, 1, 1, 1, 1 | 1; 2; 4; 5; 6 | 0 | 5 |
| [pose1+4+7+9+24](../../../output/B/pose1+4+7+9+24/evaluation/mixed_material_saved_rescue_20261011_04/index.html) | 5 | 4.17 | 5.48 | 5.72 | 2 | 2, 3 | 1+7; 4+9+24 | 3 | 2 |

各集合BBX统计和：无支撑叠放 79.70 L，Step4 164.89 L，完整支撑 257.81 L；不是同一工作站的总体积。
支撑pose统计和：44，各组成员数量之和为90个物体pose实例。
最终 Juxtapose 46 个 pose，原位转动复用 44 个 pose；22 个 state 服务多个 object pose，22 个 state 只服务一个。

以下为真实支撑落座物体的诊断空间，并非无支撑基线。

| Pose set | 支撑落座物体BBX (L) | +Step4 (L) | +Step5.2 (L) | 地脚BBX增幅 | Step4实体 (cm³) | 全部实体 (cm³) |
|---|---:|---:|---:|---:|---:|---:|
| pose1+2+4+5+6+7+11 | 8.96 | 17.91 | 49.94 | +178.89% | 237.22 | 275.64 |
| illegal/pose5+12+20+24+27+28 | 15.66 | 22.88 | 24.48 | +6.97% | 334.45 | 376.19 |
| pose1+2+3+4+5+6 | 6.20 | 8.72 | 8.72 | +0.00% | 239.73 | 244.95 |
| pose1+2+3+4+6+7 | 5.27 | 7.18 | 7.67 | +6.89% | 167.49 | 275.85 |
| pose1+2+3+4+7+27 | 5.42 | 6.25 | 8.58 | +37.30% | 128.02 | 154.57 |
| pose1+2+4+5+6+11 | 12.04 | 16.09 | 19.05 | +18.38% | 254.80 | 267.69 |
| pose1+4+7+12+21+27 | 5.29 | 6.73 | 11.83 | +75.66% | 103.78 | 166.90 |
| pose1+6+11+13+14+17 | 20.28 | 22.40 | 52.58 | +134.72% | 290.30 | 323.32 |
| pose4+5+8+9+19+23 | 9.99 | 11.77 | 13.12 | +11.52% | 173.40 | 361.31 |
| illegal/pose1+2+6+7+29 | 4.87 | 5.21 | 6.22 | +19.19% | 150.54 | 169.81 |
| illegal/pose5+6+13+15+27 | 5.25 | 7.50 | 11.46 | +52.83% | 197.88 | 217.92 |
| pose1+2+3+4+27 | 5.45 | 6.30 | 8.36 | +32.58% | 149.07 | 172.17 |
| pose1+2+3+4+6 | 6.14 | 8.23 | 9.43 | +14.59% | 216.03 | 224.55 |
| pose1+2+3+7+27 | 5.76 | 5.87 | 11.00 | +87.28% | 50.15 | 125.05 |
| pose1+2+4+5+6 | 6.13 | 6.36 | 9.65 | +51.82% | 88.40 | 256.14 |
| pose1+4+7+9+24 | 5.16 | 5.48 | 5.72 | +4.40% | 142.80 | 159.03 |

单独分配不同state时的空间诊断统计和：物体 212.75 L，Step4 279.91 L，Step5.2 451.25 L。
[逐 pose 尺寸、共享分组和材料数据](data/evaluation_mixed_material_saved_rescue_20261011_04/summary.json)。
