# 静态工作空间：无支撑叠放基线／真实支撑布局

全部 2 组，共 12 个 pose。只计最终工作位置，单位 L；1 L=1000 cm³。
无支撑基线保留各原任务朝向，将物体的水平包围盒中心对齐、最低点放在z=0，叠放全部pose后取一个BBX。其每轴长度为该轴各pose长度的最大值，达到自由平移时的最小XYZ盒。
任务依次执行；一个 state 服务多个 object pose 不表示多个物体同时装入。Juxtapose 是最终 pose 是否重新落座，multi-object-pose state 是共享关系，两者不是互斥分类。
有支撑两列使用保存的真实优化位置及全部最终支撑state；不改变物理结果。三列图同等轴测方向、同长度比例，分别居中。

| Pose set | Pose数 | 无支撑叠放BBX (L) | +Step4 BBX (L) | +Step5.2 BBX (L) | Juxtapose pose数 | 原位转动复用pose数 | State分组 |
|---|---:|---:|---:|---:|---:|---:|---|
| [pose1+2+3+4+5+6+7+8+9+10](../../../output/B/pose1+2+3+4+5+6+7+8+9+10/evaluation/clearance_compact_foot_pilot/index.html) | 10 | 5.27 | 32.80 | 35.67 | 10 | 0 | 1+2+3+4+5+6+7+8+9+10 |
| [pose2+20](../../../output/B/pose2+20/evaluation/clearance_compact_foot_pilot/index.html) | 2 | 4.11 | 4.59 | 4.59 | 0 | 2 | 2; 20 |

各集合BBX统计和：无支撑叠放 9.38 L，Step4 37.39 L，完整支撑 40.26 L；不是同一工作站的总体积。
最终 Juxtapose 10 个 pose，原位转动复用 2 个 pose；1 个 state 服务多个 object pose，2 个 state 只服务一个。

以下为真实支撑落座物体的诊断空间，并非无支撑基线。

| Pose set | 支撑落座物体BBX (L) | +Step4 (L) | +Step5.2 (L) | 地脚BBX增幅 | Step4实体 (cm³) | 全部实体 (cm³) |
|---|---:|---:|---:|---:|---:|---:|
| pose1+2+3+4+5+6+7+8+9+10 | 31.25 | 32.80 | 35.67 | +8.75% | 884.36 | 899.54 |
| pose2+20 | 4.31 | 4.59 | 4.59 | -0.00% | 135.94 | 142.08 |

单独分配不同state时的空间诊断统计和：物体 36.70 L，Step4 38.82 L，Step5.2 42.32 L。
[逐 pose 尺寸、共享分组和材料数据](data/evaluation_clearance_compact_foot_pilot/summary.json)。
