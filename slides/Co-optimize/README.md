# Co-optimize：多配置共享刚性支撑

用户已于2026-10-12确认满意当前全部52组结果。[已确认结果与讨论起点](accepted_results_20261012.md)记录最终批次、Evaluation、逐组object／支撑pose数量及图片存档，作为接下来讨论的参照。

当前完整算法、物理边界、Step4／Step5 公式与 Evaluation 指标统一维护在[仓库根 README](../../README.md)。实现采用 whole 联合优化：Direction、世界 XYZ Translation 与离散 Juxtapose，结合动态 base Oracle；允许 airborne，保持原任务朝向与需求。

最新全部52组已完成，每组图片直接放在 `output/B/[set]/step5.2/`，不使用 HTML。其中 `overview.png` 为完整支撑，`all_poses.png` 为全部任务姿态，`pose_N.png` 为单个物体姿态，`evaluation.png` 从左到右为无支撑／加 Step4／加完整 Step5.2；`illegal/` 保留目录前缀。共419张直接图片；[示例姿态图](output/B/pose1+2+4+19/step5.2/all_poses.png) · [示例Evaluation图](output/B/pose1+2+4+19/step5.2/evaluation.png) · [52件支撑总览](output/B/all_supports_material_output_all52_20261012_final.png)。全部52组共263个 object pose、112个支撑安装pose；每组只有一件共享支撑实体。[完整Evaluation与逐组pose数量](data/results/B/evaluation_material_output_all52_20261012_final.md) · [CSV](data/results/B/evaluation_material_output_all52_20261012_final.csv)。最终批次为`material_output_all52_20261012_final`，保留本轮搜索与单组构造修正的来源；1150张阶段图及旧实验图片保留。

| 入口 | 内容 |
| --- | --- |
| [新算法设计](representation_algorithm.txt) | 共享材料表示、固定sample搜索、完整成本评价，以及已集成部分与后续材料规划器的接口 |
| [新混合布局搜索](step4.2/run_material.py) | 原位复用／Juxtapose组合采样、原接触端口匹配、地面高度提案和搜索内成品构造；原物理与法向Oracle保持 |
| [Step3.1](step3.1/README.md)／[Step3.2](step3.2/README.md) | 整组注册、贴合包裹、扣除工作禁区及单独展示 |
| [Step4.1](step4.1/README.md) | 共同／相近退出初始化 |
| [Step4.2](step4.2/README.md) | 连续梯度、离散跳步、动态 Oracle 与可行减材料 |
| [Step5](step5/README.md) | 全部地面需求凸包、确定性构造成品实体 |
| [Evaluation](evaluation/README.md) | 静态独立pose空间、共享state空间与三列整体BBX |
| [Operations](operation_demo/README.md) | Direction／Translation／Juxtapose 演示 |

历史 `base_oracle_v3`／`connected_base_v2` 批次全部52组通过，263个pose实例；10个≥7-pose集合均通过。该批次成品材料总量25,540.38cm³，相对Step4名义实体净增加8.88%；这些是旧条件下的保存结果，不代表新混合布局实验的结果。

[历史逐组数值表](data/results/B/evaluation_connected_base_v2.md)。各阶段 PNG 保留在对应集合和实验目录，B 目录不再提供 HTML 浏览页。

`output/`保留各步最终可视化，B 不使用 HTML；布局、mask、需求、模型和报告在`data/results/`。候选帧、临时布局、日志及已验证的执行副本用完即删。物理输入在`objects/`，原字节保持不变。文件生命周期实现见[artifact_store.py](helper_func/artifact_store.py)，每组直接图片的发布入口为[publish_output_images.py](helper_func/publish_output_images.py)。
