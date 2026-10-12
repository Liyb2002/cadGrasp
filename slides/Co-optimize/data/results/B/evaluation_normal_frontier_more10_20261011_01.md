# 静态工作空间：无支撑叠放基线／真实支撑布局

全部 5 组，共 24 个 pose。只计最终工作位置，单位 L；1 L=1000 cm³。
无支撑基线保留各原任务朝向，将物体的水平包围盒中心对齐、最低点放在z=0，叠放全部pose后取一个BBX。其每轴长度为该轴各pose长度的最大值，达到自由平移时的最小XYZ盒。
任务依次执行；一个 state 服务多个 object pose 不表示多个物体同时装入。Juxtapose 是最终 pose 是否重新落座，multi-object-pose state 是共享关系，两者不是互斥分类。
支撑pose按保存的实际世界位置与朝向分组；同一件支撑换位置或转动后可以有多个支撑pose。每组列出服务的物体pose数量与成员。
有支撑两列使用保存的真实优化位置及全部最终支撑state；不改变物理结果。三列图同等轴测方向、同长度比例，分别居中。

| Pose set | 物体pose数 | 无支撑叠放BBX (L) | +Step4 BBX (L) | +Step5.2 BBX (L) | 支撑pose数 | 每个支撑pose服务的物体pose数 | 物体pose分组 | Juxtapose pose数 | 原位转动复用pose数 |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| [illegal/pose5+12+20+24+27+28](../../../output/B/illegal/pose5+12+20+24+27+28/evaluation/normal_frontier_more10_20261011_01/index.html) | 6 | 5.18 | 21.54 | 1440.35 | 1 | 6 | 5+12+20+24+27+28 | 6 | 0 |
| [illegal/pose5+6+13+15+27](../../../output/B/illegal/pose5+6+13+15+27/evaluation/normal_frontier_more10_20261011_01/index.html) | 5 | 5.20 | 10.91 | 38.80 | 1 | 5 | 5+6+13+15+27 | 4 | 1 |
| [pose12+16+19+21+27](../../../output/B/pose12+16+19+21+27/evaluation/normal_frontier_more10_20261011_01/index.html) | 5 | 5.19 | 6.90 | 22.77 | 5 | 1, 1, 1, 1, 1 | 12; 16; 19; 21; 27 | 0 | 5 |
| [illegal/pose1+2+6+15](../../../output/B/illegal/pose1+2+6+15/evaluation/normal_frontier_more10_20261011_01/index.html) | 4 | 4.64 | 12.69 | 8044.73 | 1 | 4 | 1+2+6+15 | 4 | 0 |
| [illegal/pose7+11+13+19](../../../output/B/illegal/pose7+11+13+19/evaluation/normal_frontier_more10_20261011_01/index.html) | 4 | 5.01 | 9.56 | 31.64 | 1 | 4 | 7+11+13+19 | 4 | 0 |

各集合BBX统计和：无支撑叠放 25.21 L，Step4 61.60 L，完整支撑 9578.28 L；不是同一工作站的总体积。
支撑pose统计和：9，各组成员数量之和为24个物体pose实例。
最终 Juxtapose 18 个 pose，原位转动复用 6 个 pose；4 个 state 服务多个 object pose，5 个 state 只服务一个。

以下为真实支撑落座物体的诊断空间，并非无支撑基线。

| Pose set | 支撑落座物体BBX (L) | +Step4 (L) | +Step5.2 (L) | 地脚BBX增幅 | Step4实体 (cm³) | 全部实体 (cm³) |
|---|---:|---:|---:|---:|---:|---:|
| illegal/pose5+12+20+24+27+28 | 20.13 | 21.54 | 1440.35 | +6587.01% | 605.01 | 734.79 |
| illegal/pose5+6+13+15+27 | 9.85 | 10.91 | 38.80 | +255.47% | 412.10 | 511.39 |
| pose12+16+19+21+27 | 6.82 | 6.90 | 22.77 | +230.24% | 63.39 | 145.59 |
| illegal/pose1+2+6+15 | 11.88 | 12.69 | 8044.73 | +63275.49% | 382.52 | 660.81 |
| illegal/pose7+11+13+19 | 8.91 | 9.56 | 31.64 | +230.95% | 216.15 | 240.93 |

单独分配不同state时的空间诊断统计和：物体 67.94 L，Step4 72.97 L，Step5.2 9623.69 L。
[逐 pose 尺寸、共享分组和材料数据](data/evaluation_normal_frontier_more10_20261011_01/summary.json)。
