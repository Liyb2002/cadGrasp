# Co-optimize：已确认结果与讨论起点（2026-10-12）

后续流程已调整为Step5.2材料删减、Step5.3完整构造；本快照的数值及物理来源保持原身份。用户随后明确要求清理旧Step5.2/5.3输出，因此下方历史图片链接可能已移除；新结果图直接位于各组step5.2/和step5.3/。

用户已确认：“行，我满意这个结果了。”本文件记录此次确认的结果，作为后续讨论的共同参照。

最终结果批次：`material_output_all52_20261012_final`。主搜索批次：`material_output_all52_20261012_01`。覆盖 `output/B` 中全部52个集合，包括42个常规集合和10个 `illegal/` 集合，集合大小为2–10个 object pose。

本批使用已记录的暖启动：8个样本、8轮搜索、32个初始化提案。结果比较描述保存模型的空间和材料，不能作为冷启动搜索性能对照。

当前实现采用有限混合布局采样、原接触端口匹配、地面高度提案和搜索内成品成本评价，复用原受力内核、法向 Oracle 与实体构造。通用稀疏共享材料选择规划器仍属于后续设计。实现边界见[算法设计](representation_algorithm.txt)和[仓库根 README](../../README.md)。

**确认的完成状态**

- 全部52/52组完成 Step4 → Step5.1 → Step5.2 → Evaluation。
- 263 个 object pose 实例，112 个支撑安装 pose；每组一件共享支撑实体，共52件。
- 保存的 8,617,984 条原始需求全部通过。最终52件实体连通、封闭。
- 4–8个物体pose的36组均满足2–4个原位转动复用pose、2–4个Juxtapose pose的目标；该数量目标只适用于4–8个pose的集合。
- 原位转动复用 108 个，Juxtapose 155 个。
- 72 个支撑安装pose服务多个object pose，40 个服务单个object pose。

**图片位置**

每组当前图片直接放在 `slides/Co-optimize/output/B/[set]/step5.2/`；`illegal/` 集合保留其前缀。B 目录不使用 HTML。

| 文件 | 内容 |
|---|---|
| `overview.png` | 一件完整共享支撑 |
| `all_poses.png` | 该集合全部物体任务姿态 |
| `pose_N.png` | 单个物体pose和实际支撑安装姿态 |
| `evaluation.png` | 从左到右：无支撑、加Step4、加完整Step5.2 |

52组共有419张直接PNG，全部与最终批次原图的SHA-256一致。全部阶段的1150张图及带批次名的图片存档保留。下方逐组图片链接指向此次确认批次的存档，当前直接图片位置沿用上面的目录。

[52件支撑总览](output/B/all_supports_material_output_all52_20261012_final.png) · [完整Evaluation报告与支撑pose成员](data/results/B/evaluation_material_output_all52_20261012_final.md) · [CSV](data/results/B/evaluation_material_output_all52_20261012_final.csv)。

**Evaluation**

| 指标 | 52个独立集合的统计和 |
|---|---:|
| 无支撑物体BBX | 254.41 L |
| 实际布局加Step4支撑BBX | 389.14 L |
| 实际布局加完整Step5.2支撑BBX | 433.19 L |
| 完整支撑实体材料 | 12527.10 cm³ |

与保存的旧 `clearance_compact_v1` 相比，完整BBX统计和从 834.76 L 降到 433.19 L，降低 48.11%；52组均减小。

BBX采用固定世界XYZ包围盒。无支撑基线保留原任务朝向，将水平盒中心对齐、最低点放在z=0；有支撑两列使用保存的实际优化布局及最终安装姿态。任务依次执行。统计范围为最终静态工作位置；退出、搬运和机器人运动另行讨论。跨集合的统计和描述独立案例，实体材料cm³与空间BBX L分别报告。

object pose指物体任务姿态；支撑pose按同一件支撑的实际世界位置与朝向分组。一个支撑pose可以依次服务多个object pose，其服务数量之和等于该集合的object pose数。Juxtapose和原位复用描述物体落座方式，支撑安装pose单独计数。

| Set（打开确认批次姿态图） | object pose | 支撑 pose | 每个支撑pose服务数量 | 无支撑 L | Step4 L | Step5.2 L | Evaluation |
|---|---:|---:|---|---:|---:|---:|---|
| [pose1+2+3+4+5+6+7+8+9+10](output/B/pose1+2+3+4+5+6+7+8+9+10/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 10 | 1 | 10 | 5.27 | 20.89 | 21.05 | [PNG](output/B/pose1+2+3+4+5+6+7+8+9+10/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+3+5+7+9+12+16+21+23+27](output/B/pose1+3+5+7+9+12+16+21+23+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 10 | 2 | 7, 3 | 5.47 | 18.46 | 18.46 | [PNG](output/B/pose1+3+5+7+9+12+16+21+23+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+3+4+5+6+7+8+9](output/B/pose1+2+3+4+5+6+7+8+9/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 9 | 1 | 9 | 5.27 | 19.09 | 19.09 | [PNG](output/B/pose1+2+3+4+5+6+7+8+9/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+4+7+9+12+16+21+23+27](output/B/pose1+4+7+9+12+16+21+23+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 9 | 1 | 9 | 4.79 | 21.20 | 21.20 | [PNG](output/B/pose1+4+7+9+12+16+21+23+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+3+4+5+6+7+8](output/B/pose1+2+3+4+5+6+7+8/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 8 | 4 | 2, 3, 2, 1 | 5.27 | 9.33 | 17.86 | [PNG](output/B/pose1+2+3+4+5+6+7+8/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+4+5+6+7+10+11](output/B/pose1+2+4+5+6+7+10+11/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 8 | 4 | 1, 5, 1, 1 | 5.27 | 7.95 | 14.33 | [PNG](output/B/pose1+2+4+5+6+7+10+11/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+3+7+10+14+18+23+27](output/B/pose1+3+7+10+14+18+23+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 8 | 4 | 3, 1, 3, 1 | 6.04 | 9.94 | 13.91 | [PNG](output/B/pose1+3+7+10+14+18+23+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+3+4+5+6+7](output/B/pose1+2+3+4+5+6+7/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 7 | 3 | 2, 4, 1 | 5.27 | 13.37 | 19.22 | [PNG](output/B/pose1+2+3+4+5+6+7/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+4+5+6+7+11](output/B/pose1+2+4+5+6+7+11/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 7 | 3 | 2, 3, 2 | 5.27 | 9.76 | 10.62 | [PNG](output/B/pose1+2+4+5+6+7+11/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+4+7+12+21+23+27](output/B/pose1+4+7+12+21+23+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 7 | 4 | 4, 1, 1, 1 | 4.79 | 7.34 | 10.00 | [PNG](output/B/pose1+4+7+12+21+23+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [illegal/pose4+7+12+21+23+27](output/B/illegal/pose4+7+12+21+23+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 6 | 2 | 4, 2 | 4.79 | 6.24 | 6.44 | [PNG](output/B/illegal/pose4+7+12+21+23+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [illegal/pose5+12+20+24+27+28](output/B/illegal/pose5+12+20+24+27+28/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 6 | 2 | 5, 1 | 5.18 | 9.96 | 10.71 | [PNG](output/B/illegal/pose5+12+20+24+27+28/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+3+4+5+6](output/B/pose1+2+3+4+5+6/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 6 | 2 | 3, 3 | 5.27 | 6.59 | 6.87 | [PNG](output/B/pose1+2+3+4+5+6/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+3+4+6+7](output/B/pose1+2+3+4+6+7/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 6 | 2 | 4, 2 | 4.67 | 5.61 | 5.61 | [PNG](output/B/pose1+2+3+4+6+7/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+3+4+7+27](output/B/pose1+2+3+4+7+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 6 | 4 | 2, 2, 1, 1 | 4.88 | 7.08 | 7.78 | [PNG](output/B/pose1+2+3+4+7+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+4+5+6+11](output/B/pose1+2+4+5+6+11/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 6 | 2 | 4, 2 | 5.27 | 7.28 | 7.47 | [PNG](output/B/pose1+2+4+5+6+11/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+4+5+6+7](output/B/pose1+2+4+5+6+7/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 6 | 4 | 2, 1, 1, 2 | 5.27 | 6.31 | 7.03 | [PNG](output/B/pose1+2+4+5+6+7/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+4+7+12+21+27](output/B/pose1+4+7+12+21+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 6 | 3 | 1, 3, 2 | 4.57 | 8.04 | 8.88 | [PNG](output/B/pose1+4+7+12+21+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+6+11+13+14+17](output/B/pose1+6+11+13+14+17/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 6 | 4 | 1, 3, 1, 1 | 4.96 | 5.98 | 7.99 | [PNG](output/B/pose1+6+11+13+14+17/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose4+5+8+9+19+23](output/B/pose4+5+8+9+19+23/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 6 | 3 | 3, 2, 1 | 5.98 | 7.52 | 9.57 | [PNG](output/B/pose4+5+8+9+19+23/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [illegal/pose1+2+6+7+29](output/B/illegal/pose1+2+6+7+29/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 5 | 2 | 4, 1 | 4.67 | 5.17 | 5.33 | [PNG](output/B/illegal/pose1+2+6+7+29/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [illegal/pose5+6+13+15+27](output/B/illegal/pose5+6+13+15+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 5 | 2 | 4, 1 | 5.20 | 6.78 | 6.80 | [PNG](output/B/illegal/pose5+6+13+15+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+3+4+27](output/B/pose1+2+3+4+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 5 | 3 | 2, 2, 1 | 4.88 | 6.24 | 7.23 | [PNG](output/B/pose1+2+3+4+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+3+4+6](output/B/pose1+2+3+4+6/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 5 | 3 | 2, 1, 2 | 4.62 | 6.29 | 6.64 | [PNG](output/B/pose1+2+3+4+6/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+3+7+27](output/B/pose1+2+3+7+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 5 | 2 | 3, 2 | 4.88 | 6.37 | 6.48 | [PNG](output/B/pose1+2+3+7+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+4+5+6](output/B/pose1+2+4+5+6/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 5 | 3 | 3, 1, 1 | 5.27 | 5.45 | 5.59 | [PNG](output/B/pose1+2+4+5+6/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+4+7+9+24](output/B/pose1+4+7+9+24/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 5 | 2 | 4, 1 | 4.17 | 5.40 | 5.40 | [PNG](output/B/pose1+4+7+9+24/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose12+16+19+21+27](output/B/pose12+16+19+21+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 5 | 2 | 3, 2 | 5.19 | 7.79 | 8.00 | [PNG](output/B/pose12+16+19+21+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose5+6+11+23+29](output/B/pose5+6+11+23+29/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 5 | 2 | 4, 1 | 5.50 | 6.74 | 7.19 | [PNG](output/B/pose5+6+11+23+29/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose6+10+13+17+30](output/B/pose6+10+13+17+30/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 5 | 3 | 2, 2, 1 | 4.48 | 6.95 | 8.23 | [PNG](output/B/pose6+10+13+17+30/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [illegal/pose1+2+6+15](output/B/illegal/pose1+2+6+15/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 4 | 2 | 3, 1 | 4.64 | 5.53 | 6.23 | [PNG](output/B/illegal/pose1+2+6+15/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [illegal/pose7+11+13+19](output/B/illegal/pose7+11+13+19/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 4 | 2 | 1, 3 | 5.01 | 6.70 | 7.06 | [PNG](output/B/illegal/pose7+11+13+19/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+11+14+27](output/B/pose1+11+14+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 4 | 2 | 3, 1 | 4.33 | 5.39 | 5.80 | [PNG](output/B/pose1+11+14+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+3+27](output/B/pose1+2+3+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 4 | 2 | 2, 2 | 4.88 | 5.58 | 5.79 | [PNG](output/B/pose1+2+3+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+4+19](output/B/pose1+2+4+19/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 4 | 2 | 3, 1 | 5.17 | 6.84 | 7.05 | [PNG](output/B/pose1+2+4+19/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+4+6](output/B/pose1+2+4+6/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 4 | 2 | 3, 1 | 4.62 | 6.06 | 6.56 | [PNG](output/B/pose1+2+4+6/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+2+4+7](output/B/pose1+2+4+7/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 4 | 2 | 1, 3 | 4.24 | 5.56 | 5.99 | [PNG](output/B/pose1+2+4+7/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose2+3+4+7](output/B/pose2+3+4+7/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 4 | 2 | 1, 3 | 4.64 | 4.98 | 5.23 | [PNG](output/B/pose2+3+4+7/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose5+6+23+29](output/B/pose5+6+23+29/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 4 | 2 | 3, 1 | 5.50 | 6.62 | 7.22 | [PNG](output/B/pose5+6+23+29/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose8+9+13+30](output/B/pose8+9+13+30/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 4 | 2 | 3, 1 | 4.20 | 5.13 | 5.15 | [PNG](output/B/pose8+9+13+30/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [illegal/pose1+2+29](output/B/illegal/pose1+2+29/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 3 | 1 | 3 | 4.58 | 4.78 | 4.78 | [PNG](output/B/illegal/pose1+2+29/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [illegal/pose5+13+27](output/B/illegal/pose5+13+27/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 3 | 1 | 3 | 4.56 | 5.14 | 5.14 | [PNG](output/B/illegal/pose5+13+27/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose1+12+29](output/B/pose1+12+29/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 3 | 1 | 3 | 4.68 | 4.92 | 4.92 | [PNG](output/B/pose1+12+29/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose18+23+24](output/B/pose18+23+24/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 3 | 1 | 3 | 5.33 | 5.75 | 5.75 | [PNG](output/B/pose18+23+24/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose4+5+7](output/B/pose4+5+7/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 3 | 1 | 3 | 4.71 | 5.10 | 5.10 | [PNG](output/B/pose4+5+7/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose8+10+19](output/B/pose8+10+19/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 3 | 1 | 3 | 5.66 | 6.24 | 6.29 | [PNG](output/B/pose8+10+19/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [illegal/pose11+19](output/B/illegal/pose11+19/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 2 | 1 | 2 | 5.01 | 5.61 | 5.61 | [PNG](output/B/illegal/pose11+19/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [illegal/pose2+29](output/B/illegal/pose2+29/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 2 | 1 | 2 | 4.01 | 4.32 | 4.32 | [PNG](output/B/illegal/pose2+29/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose19+28](output/B/pose19+28/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 2 | 1 | 2 | 4.28 | 4.94 | 5.13 | [PNG](output/B/pose19+28/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose2+20](output/B/pose2+20/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 2 | 2 | 1, 1 | 4.11 | 4.18 | 4.18 | [PNG](output/B/pose2+20/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose3+15](output/B/pose3+15/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 2 | 1 | 2 | 4.41 | 4.79 | 5.02 | [PNG](output/B/pose3+15/evaluation/material_output_all52_20261012_final/static_occupancy.png) |
| [pose8+21](output/B/pose8+21/step5/step5.2/material_output_all52_20261012_final/all_poses.png) | 2 | 1 | 2 | 3.49 | 3.85 | 3.85 | [PNG](output/B/pose8+21/evaluation/material_output_all52_20261012_final/static_occupancy.png) |

**来源与核对记录**

最终发布保留51件本批已接受实体；`pose1+2+4+19` 在同一保存的合法连接见证和布局中重新构造，处理外部负向壳体导出问题。该修正沿用保存的受力验收与输入，其来源在最终阶段记录中保留。

1332个原始输入、既有布局和证明文件的哈希保持不变。此次MD记录引用已保存结果。

- [Evaluation原始数值](data/results/B/data/evaluation_material_output_all52_20261012_final/summary.json)
- [实体、需求和阶段图片核对](data/results/B/data/system_material_output_all52_20261012_final/visual_and_saved_result_verification.json)
- [直接图片来源与SHA-256](data/results/B/data/system_material_output_all52_20261012_final/direct_images.json)
- [图片、报告链接及原始文件核对](data/results/B/data/system_material_output_all52_20261012_final/direct_images_verification.json)

当前进入讨论阶段，后续讨论以这份已确认快照和对应最终批次为参照。
