# Evaluation：静态工作空间

当前完整实体读取新Step5.3；新Step5.2仅为删减材料，不能作为成品评价。旧批次的Step5.2数值仍作为历史完整实体兼容读取。新图片直接位于每组step5.3/与evaluation/，没有实验子目录。

最新批次 `material_output_all52_20261012_final` 已完成52/52组：263个 object pose、112个支撑安装pose。每组当前结果直接放在 `../output/B/[set]/step5.2/`，`evaluation.png` 从左到右为无支撑／加 Step4／加完整 Step5.2，不使用 HTML。[全部52组的数值、pose数量和成员](../data/results/B/evaluation_material_output_all52_20261012_final.md) · [CSV](../data/results/B/evaluation_material_output_all52_20261012_final.csv)。独立集合BBX统计和为254.41／389.14／433.19 L，完整实体材料统计和12,527.10 cm³。

本目录与 Step4、Step5 同级，入口 [run.py](run.py)，公式实现 [metrics.py](metrics.py)。不进行优化、实体 Boolean 或受力求解。

只评价静态工作空间，任务依次执行。无支撑时各 pose 可以重叠，不需要沿用支撑优化产生的落座位置：

| 列 | 计入的占用空间 |
| --- | --- |
| 无支撑叠放物体 | 保留各任务朝向，水平包围盒中心对齐、最低点放在地面上 |
| 加 Step4 支撑 | 保存的真实物体位置，加名义支撑在所有最终 state 的占用 |
| 加 Step5.2 全部支撑 | 同一真实物体布局，加完整实体支撑在所有最终 state 的占用 |

令 $O_i=U_iO$ 为静态物体，$V(X)$ 为点集 $X$ 的固定世界 XYZ 包围盒体积：

$$V(X)=\prod_{a\in\{x,y,z\}}(\max_{p\in X}p_a-\min_{p\in X}p_a).$$

设原任务旋转为 $R_i$，旋转后的各轴包围盒长度为 $d_{i,a}$。无支撑变换 $\bar U_i$ 保持 $R_i$，将水平盒中心移到零、最低点移到 $z=0$。这样全部 pose 叠放的包围盒为

$$V_{\rm no\ support}=V\!\left(\bigcup_i\bar U_iO\right)=\prod_{a\in\{x,y,z\}}\max_i d_{i,a}.$$

每轴必须容纳最长的那个 pose；上述平移同时达到三个轴的下界，所以它是固定朝向、自由平移下最小的世界 XYZ 盒。pose 不同时装入，叠放仅表示先后使用同一空间。无支撑基线不读优化后的 host、$Q_i$ 或 airborne 高度，也不求每个 pose 盒子的体积之和。

有支撑时使用实际 $O_i=U_iO$，三列指标为无支撑基线、$V(\bigcup_iO_i\cup\bigcup_sT_sN)$、$V(\bigcup_iO_i\cup\bigcup_sT_sS)$。不使用退出方向、任意行程或搬运路径。所有图片保持同一等轴测方向与长度比例，各列分别居中。

同一实际支撑变换归为 state $s$，其成员为 $I_s$，$O_s=\bigcup_{i\in I_s}O_i$。另外保留共享 state 空间诊断：

$$\sum_sV(O_s),\qquad\sum_sV(O_s\cup T_sN),\qquad\sum_sV(O_s\cup T_sS).$$

每个 state 的盒子需要容纳它服务的全部 pose；这里把不同 state 看作独立可分配空间，并不要求实际使用多个工作站。该统计及 `objects_in_final_layout` 仅为实际支撑落座诊断，不是无支撑基线。实体材料 cm³ 与占用 BBX L 分列。

最终 Juxtapose pose 指 host 改变或 $Q_i\ne I$；保持自己 host 且 $Q_i=I$ 的归为原位转动复用。两类合计为 pose 数。一个 state 服务多个 object pose 是另一维度，报告实际成员列表与组大小；Juxtapose 与共享 state 不是互斥分类。

Evaluation显式列出**支撑pose数量**，以及**每个支撑pose分别服务几个物体pose、具体哪些pose**。支撑pose按保存的实际世界位置与朝向分组，同一件实体支撑可以有多个安装pose；相同host或不同物体落座位置本身不决定分组。JSON字段为`support_pose_count`、`object_pose_counts_per_support_pose`，成员及安装变换保留在`fixture_states`，成员数量之和等于物体pose总数。该统计版本为`installed_support_pose_membership_v1`，静态BBX公式不变；旧Evaluation可用`--replace`补齐，复用全部Step4/5模型。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/evaluation/run.py B --experiment my_final_base --batch-name evaluation_my_final_base
```

实验原图保存在`output/B/{pose_set}/evaluation/{experiment}/static_occupancy.png`，同时当前图发布为`output/B/{pose_set}/step5.2/evaluation.png`，从左到右显示无支撑叠放物体、实际布局加Step4、加Step5.2，无图片文字。报告在`data/results/B/`同层级，包含无支撑摆放、逐pose尺寸、state成员与尺寸、整体盒、实体材料与来源。总表和统计在`data/results/B/evaluation_{experiment}.md`及`data/evaluation_{experiment}/`，B 不生成 HTML。临时执行副本和进度文件在校验后删除。`--replace`只替换指定evaluation，不重跑物理阶段。

当前指标版本为`static_occupancy_v2_overlapping_objects`；`object_static`表示无支撑叠放基线。旧`static_occupancy_v1`使用了支撑的落座位置，不能作为无支撑对比。

最新新增10组的4布局试跑完成5件可评价实体，另有4组搜索未通过、1组构造未通过。[10组完整表与同条件旧版对照](../data/results/B/normal_frontier_more10_review_20261011_01.md)包含Juxtapose数、支撑pose数、各自服务的物体pose数量与成员、三列BBX和材料；旧实验图片保留在各集合的 `evaluation/normal_frontier_more10_20261011_01/`。同条件旧版完成7件，共同完成的5件中BBX增加4件、不变1件。成功对照没有Juxtapose减少的案例；两件大幅抬高物体的解出现1440.35L／8044.73L的占用盒。未完成模型不分配成品BBX，旧模型只做只读统计。

历史完成实体版`clearance_compact_v1`，全部52组，包括全部10个≥7-pose集合。[逐组完整表](../data/results/B/evaluation_clearance_compact_v1.md) · [尺寸与来源数据](../data/results/B/data/evaluation_clearance_compact_v1/summary.json)；历史图片在各集合的 `evaluation/clearance_compact_v1/`。Evaluation只计算保存模型的盒子并出图，不重跑Step4/Step5，不改变物理结果。

52组独立集合BBX统计和：无支撑叠放254.41L，实际布局加Step4为769.27L，加完整支撑为834.76L。pose1–10三列为5.27／32.80／35.67L。叠放基线是一个共同盒；跨集合统计和不代表同一工作站。相对旧`connected_base_v2`，完整BBX统计和减少61.46%，50组减少、2组增加；[全部新旧对比](../data/results/B/data/system_clearance_compact_v1/comparison.md)。旧批次早于5mm接触余量，不能作为新条件下的可行性证书。

[verify_outputs.py](verify_outputs.py)保留旧`connected_base_v2`的只读核对入口及`data/results/B/data/system_connected_base_v2/`记录。新链的构造验收、上游指纹与Evaluation记录在`data/results/B/data/system_clearance_compact_v1/`及各阶段清单中；正常完成链不重复原受力验收。
# 最新Evaluation（2026-10-12）

批次`dynamic_postprocess_all52_20261012_01`报告全部52组与263个object pose。最终49组通过：238个object pose、129个支撑安装pose，复用129／Juxtapose109。相同49组的材料6835.005→10842.841cm³（+58.64%），静态BBX324.424→413.720L（+27.52%），均相对Step4名义模型。[每组角色、材料／BBX增量和图](../data/results/B/evaluation_dynamic_postprocess_all52_20261012_01.md) · [全部263个pose的角色CSV](../data/results/B/evaluation_dynamic_postprocess_all52_20261012_01_poses.csv)。失败组的成品体积留空；Stage4已有完整参考是另一比较基准，详见[结果说明](../dynamic_postprocess_results_20261012.md)。下文保存历史指标定义与批次。

