# 实验结果

这里保存上一轮实验。用户最新目标以 **全部约束通过后的材料体积** 为主要代价，支撑摆放数不算代价；新路线优先转动复用，见 [新算法](reuse_first_algorithm.md) 和 [新结果／图](../Co-optimize/output/B/step4.2/README.md)。

“通过”表示原始承载、完整装卸路径和工作面验收通过，并从导出的实体重新提取接触、保存及回代全部原始载荷的非负反力。每个 pose 有 32,768 条载荷，包含第七维不上抬约束。

| pose set | 方法 | 验收 | 材料 cm³ | 最大投影占地 cm² | 支撑摆放数 | 精确候选数 |
|---|---|---|---:|---:|---:|---:|
| 1–7 | joint | [通过](output/seven_v10/seven_chain/joint/report.json) | 137.6 | 288.0 | 5 | 17 |
| 1–7 | incremental | [通过](output/prefix_v15/seven_chain/incremental/report.json) | 215.4 | 307.9 | 5 | 13 |
| 1,2,4,5,6,7,11 | joint | [通过](output/hard_v16/seven_hard/joint/report.json) | 194.5 | 291.0 | 4 | 35 |
| 1,2,4,5,6,7,11 | incremental | [通过](output/hard_v16/seven_hard/incremental/report.json) | 168.7 | 260.7 | 4 | 17 |
| 1,4,7,12,21,23,27 | joint | [通过](output/spread_v16/seven_spread/joint/report.json) | 140.7 | 224.0 | 6 | 8 |
| 1,4,7,12,21,23,27 | incremental | [通过](output/spread_v16/seven_spread/incremental/report.json) | 178.5 | 245.8 | 5 | 15 |
| 1–8 | joint | [通过](output/eight_v16/eight_chain/joint/report.json) | 238.8 | 337.2 | 1 | 21 |
| 1–8 | incremental | [通过](output/prefix_v15/eight_chain/incremental/report.json) | 215.4 | 307.9 | 6 | 14 |
| 1–10 | joint | [通过](output/ten_v10/pose1-10/joint/report.json) | 286.7 | 369.3 | 1 | 12 |
| 1–10 | incremental | [通过](output/ten_incremental_v15/pose1-10/incremental/report.json) | 209.6 | 307.9 | 8 | 16 |

1–7 和 1–8 的 incremental 是 1–10 增量运行中真实通过的前缀，分别重新导出第 13、14 次精确评估的实体。它们的种子材料、物体及退出切除只包含当时已加入的任务，没有未来 pose 的材料；不是另外随机重跑的独立样本。

1–10 的历史 joint 用一个支撑摆放容纳十个物体 pose；incremental 用八个摆放。增量材料少约 26.9%、最大投影占地小约 16.6%。按用户后来明确的材料体积目标，这份增量结果优于这份整体结果；摆放次数更多不会因此受到惩罚。这仍不是两种方法的全局最优性比较。

这里保留了实验过程中的有效冻结版本：1–7 / 1–10 joint 为 v10，1–10 incremental 为 v15，其他新集合为 v16。几何及物理验收条件相同；v15 增加了等几何工作面数值重参数化，v16 修正了初筛中用物体实际质心计算分散程度。预算、版本及并行运行时的负载不同，表格不构成受控速度或最优性比较。

支撑摆放数是 host 对应的不同刚体摆放数量；占地是这些摆放下实体支撑的最大 XY 凸包投影面积，不是车间所有工位面积之和。精确候选数包含数值失败／超时候选；两个方法连跑时，incremental 表内已减掉 joint 的累计调用。

搜索不以远距离分离保底。最终导出的每个已通过布局都另存实际物体两两重叠体积；材料重新构造，因此增加新 pose 时材料总量也可能减少。材料体积和占地是搜索结果，没有全局最小保证。

两种 1–10 实体支撑的同尺度四视角对照：[四视角 PNG](vis/ten_pose_supports.png)。图中物体已移走，蓝色直接来自最终导出的实体，没有额外补画连接或底座。

整体优化的 [十个物体 pose 图](vis/pose1-10_joint.png)：每格使用完全相同的蓝色支撑、固定世界摆放与镜头；灰色物体依次采用各个最终配置。最终为以 pose1 为 host 的成组 Juxtapose，pose2–10 保持原任务朝向并水平错开约 18.3–20.5 mm，退出都选择世界 +z。这里统计的是一个支撑摆放，实体连通尚未验收。

当前验收沿用原模型：支撑接触只有非负法向反力、没有反力容量，地面仍是 μ=64 的四射线摩擦模型。整件支撑的连通、安装接地、强度和机器人路径尚未验收，报告的 `full_fixture_accepted` 均为 false。

每个结果目录含 `support.obj`、`layout.npz`、`report.json`、逐 pose 的载荷覆盖及反力证书，`trace.json` 记录候选与原子提交。`runtime_sources/manifest.json` 给出源代码快照：新算法的精确 worker 直接运行冻结版本，借用的 Co-optimize / baseline 几何及物理文件由哈希检查保证与快照一致。复现历史结果时应先确认原输入及这些基线文件仍匹配。历史失败／中断保留作诊断，不混入上表的有效结果。

原始输入、输出文件哈希、冻结源代码和 Co-optimize 源代码保护检查：[results.json](results.json)。重新汇总：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python slides/pose_set_search/code/summarize.py`。
