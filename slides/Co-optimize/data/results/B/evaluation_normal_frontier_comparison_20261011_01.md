# 多布局算法的 Evaluation 对比

只重算保存模型的Evaluation，没有运行Step4、Step5、受力或Oracle。支撑pose按实际世界位置和朝向分组，任务依次执行。

| 物体pose集合 | 无支撑BBX (L) | 新Step4 BBX (L) | 新完整支撑BBX (L) | 新支撑pose数 | 各支撑pose服务的物体pose数 | 物体pose分组 |
|---|---:|---:|---:|---:|---|---|
| [pose2+20](../../../output/B/pose2+20/evaluation/normal_frontier_smoke_20261011_01/index.html) | 4.11 | 4.59 | 4.59 | 2 | 1, 1 | 2; 20 |
| [pose1+2+4+7](../../../output/B/pose1+2+4+7/evaluation/normal_frontier_smoke_20261011_01/index.html) | 4.24 | 4.65 | 10.12 | 4 | 1, 1, 1, 1 | 1; 2; 4; 7 |
| [pose1+2+3+4+6](../../../output/B/pose1+2+3+4+6/evaluation/normal_frontier_search_20261011_02/index.html) | 4.62 | 43.52 | 207.59 | 1 | 5 | 1+2+3+4+6 |
| pose5+6+11+23+29 | 5.50 | — | 未通过，没有完整模型 | — | — | — |

以下旧算法对照使用相同法向Oracle、零余量、相同原始输入与初始化，包含三个实际可对照的集合；pose2+20没有同批旧法向结果。

| 物体pose集合 | 旧完整支撑BBX (L) | 新完整支撑BBX (L) | BBX变化 | 旧→新支撑pose数 | 旧→新每个支撑pose的物体pose数 | 旧→新完整材料 (cm³) |
|---|---:|---:|---:|---|---|---:|
| pose1+2+4+7 | 10.10 | 10.12 | +0.15% | 4 → 4 | 1, 1, 1, 1 → 1, 1, 1, 1 | 216.91 → 210.44 |
| pose1+2+3+4+6 | 23.90 | 207.59 | +768.67% | 1 → 1 | 5 → 5 | 541.01 → 603.47 |
| pose5+6+11+23+29 | 10.11 | 未通过 | — | 4 → — | 2, 1, 1, 1 → — | 175.47 → — |

3个成功新集合共7个支撑pose，服务11个物体pose实例。新失败集合没有完成模型的BBX或支撑pose服务统计。

[原始数据与对照条件](data/evaluation_normal_frontier_comparison_20261011_01/summary.json)。
