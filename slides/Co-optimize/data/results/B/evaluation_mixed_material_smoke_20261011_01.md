# 静态工作空间：无支撑叠放基线／真实支撑布局

全部 1 组，共 5 个 pose。只计最终工作位置，单位 L；1 L=1000 cm³。
无支撑基线保留各原任务朝向，将物体的水平包围盒中心对齐、最低点放在z=0，叠放全部pose后取一个BBX。其每轴长度为该轴各pose长度的最大值，达到自由平移时的最小XYZ盒。
任务依次执行；一个 state 服务多个 object pose 不表示多个物体同时装入。Juxtapose 是最终 pose 是否重新落座，multi-object-pose state 是共享关系，两者不是互斥分类。
支撑pose按保存的实际世界位置与朝向分组；同一件支撑换位置或转动后可以有多个支撑pose。每组列出服务的物体pose数量与成员。
有支撑两列使用保存的真实优化位置及全部最终支撑state；不改变物理结果。三列图同等轴测方向、同长度比例，分别居中。

| Pose set | 物体pose数 | 无支撑叠放BBX (L) | +Step4 BBX (L) | +Step5.2 BBX (L) | 支撑pose数 | 每个支撑pose服务的物体pose数 | 物体pose分组 | Juxtapose pose数 | 原位转动复用pose数 |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| [pose1+4+7+9+24](../../../output/B/pose1+4+7+9+24/evaluation/mixed_material_smoke_20261011_01/index.html) | 5 | 4.17 | 5.92 | 6.30 | 2 | 2, 3 | 1+7; 4+9+24 | 3 | 2 |

各集合BBX统计和：无支撑叠放 4.17 L，Step4 5.92 L，完整支撑 6.30 L；不是同一工作站的总体积。
支撑pose统计和：2，各组成员数量之和为5个物体pose实例。
最终 Juxtapose 3 个 pose，原位转动复用 2 个 pose；2 个 state 服务多个 object pose，0 个 state 只服务一个。

以下为真实支撑落座物体的诊断空间，并非无支撑基线。

| Pose set | 支撑落座物体BBX (L) | +Step4 (L) | +Step5.2 (L) | 地脚BBX增幅 | Step4实体 (cm³) | 全部实体 (cm³) |
|---|---:|---:|---:|---:|---:|---:|
| pose1+4+7+9+24 | 5.71 | 5.92 | 6.30 | +6.36% | 142.80 | 159.35 |

单独分配不同state时的空间诊断统计和：物体 8.82 L，Step4 9.58 L，Step5.2 9.90 L。
[逐 pose 尺寸、共享分组和材料数据](data/evaluation_mixed_material_smoke_20261011_01/summary.json)。
