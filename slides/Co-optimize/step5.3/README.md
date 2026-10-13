# Step5.3：将删减后的支撑构造成完整实体

2026-10-13当前圆整删减批次全部52组已重跑并通过。使用`--fixed-leg-reference`，在新已认证连通主体上[重建保存的合法脚](replay_legs.py)：脚的位置、路径、截面保持原值，实体重新导出并检查合法空间、物体排除、连通、真实地面包围和全部原始需求。每组整体成品BBX与上一版保持一致。没有重新搜索脚；未指定固定参考时仍使用下面的原长脚后端。[算法说明](../aesthetic_pruning_algorithm.md) · [52组BBX及前后图](../aesthetic_pruning_results_20261013.md)。图片直接在每组本阶段，`comparison.png`为同视角旧／新对比。

本目录承接原Step5.2构造算法。入口[run.py](run.py)，原几何后端[deterministic_base.py](deterministic_base.py)保留。输入改为新[Step5.2](../step5.2/README.md)通过原需求检查的删减材料、Step5.1需求凸包与Step4保存的合法空间。

核对删减模型、布局及force certificate指纹，保留删减后的必需材料，生成地脚并连接成一件封闭实体。所有新增材料裁进保存的合法空间；真实地脚与物体自身接地点的凸包须包围每个任务的完整需求。构造后检查实际接触面的全部原始力/力矩需求，避免数值边界裁切损失关键反力端口；原模型容差不变。

默认`--leg-profile flat`为矩形截面的扁平棱锥腿，根部34×9mm、末端10×6mm，实际形状裁进保存的合法空间；末端直接落地，不加独立凸起脚垫。`--leg-profile original`保留原小地脚与确定性连接后端。扁平腿根据保留材料和保存成品的实际地面边界补选必要落点，避免漏掉删减前由主体承担的落地部分。每个安装姿态的尺寸范围沿用保存成品，每侧最多扩展1.5mm，不重新搜索物体或支撑摆放。

必需接触块先连接，无法直接抵达地面的锥腿采用同一合法空间内的局部连接。所有新增材料同时扣除原物体在每个任务中的真实内部。真实内腔保留，只有经过正材料重叠检查的外部逆向边界和数值壳可以清理。[导出检查](../helper_func/export_stability.py)重新读取16位小数OBJ，在原容差下复核接触、完整需求、实际物体排除、地面包围、封闭及连通；有界重裁不会改变布局或原需求。

数值与模型在`data/results/B/[set]/step5/step5.3/[experiment]/`。图片直接放在`output/B/[set]/step5.3/`：`overview.png`、`all_poses.png`及每个`pose_N.png`；无实验子目录、无HTML。原Step5.3历史构造代码已移出当前目录。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python slides/Co-optimize/step5.3/run.py B \
  --sets pose1+2+4+6 --experiment my_pruning
```

[Evaluation](../evaluation/README.md)读取本阶段完整实体。法向承压包围、连通和封闭检查沿用原模型，没有增加地面摩擦、yaw或结构强度求解。
# 历史运行记录（2026-10-12）

最新V8布局后处理最终49组通过实际实体、序列化接触及全部原需求检查；第一轮44组，有界平地脚尺寸与材料保留重试后49组。原2组Step4失败与另1组真实接触失败均没有最终成品。图片直接在每组`step5.3/`；[完整Evaluation及每组增量](../data/results/B/evaluation_dynamic_postprocess_all52_20261012_01.md)。`--reference-stage step4`使用本case同一冻结布局的已构造实体作为参考；各组实际来源见CSV。下文保留历史入口说明。

