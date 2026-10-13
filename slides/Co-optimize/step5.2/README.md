# Step5.2：删减物体—支撑材料

最终进一步圆整结果：[全部52组表及图片](../sculpted_pruning_results_20261013.md)。两阶段52/52通过，原始需求、实际OBJ及源文件／图片审计通过；主体材料减少13.17%，整体BBX保持。当前结果包含34组连续主体圆整、33组进一步整块删减及明确的保守回退。各组来源见最终选择，不将部分／失败执行混称一次52/52。

当前进一步圆整采用[新的整组入口](../helper_func/run_sculpted_postprocess.py)及其保守恢复／续跑版本，算法为[sculpted_pruning.py](sculpted_pruning.py)与[surface_flow.py](surface_flow.py)：先整块删冗余，再固定必要接触和整体BBX，对真实自由表面圆整。复杂模型使用[分批查询](surface_flow_bounded.py)及[连续表面保守分支](surface_only_pruner.py)，原验收保持。全部52组的两阶段执行已通过；[完整算法](../sculpted_pruning_algorithm.md)，最终来源在`data/results/B/data/sculpted_pruning_all52_20261013_01/resolved_batch.json`。每组图片直接在本阶段及Step5.3，成品的放大图为`step5.3/detail.png`。下面的默认`run.py`及6 mm保留区说明属于上一版对照。

当前入口 [run.py](run.py)，默认算法 [rounded_pruning.py](rounded_pruning.py)。冻结 Step4 的座位、安装姿态与退出方向，读取保存的 Step5.1 需求。先整面裁去扩大BBX的原材料，再以较宽的圆整保留区删除大块冗余材料，避免网格逐格删减留下缺口和细连接。[算法说明](../aesthetic_pruning_algorithm.md) · [全部52组结果和前后图](../aesthetic_pruning_results_20261013.md)。完整长脚构造在 [Step5.3](../step5.3/README.md)。

本轮以每组自己上一版已认证的删减主体作为`--shape-reference`：保留其真实接触材料，并在6 mm圆整范围内保留更多原Step4材料。圆整范围与原材料相交，不向Step4外添加材料，也不移动接触。全部52组Step5.2／5.3通过；51组采用6 mm，1组圆整候选被原验收拒绝而保留整块BBX裁边主体。两阶段每组整体BBX和全部pose角色与上一版一致。各安装state小盒是独立诊断，不能混作整体BBX。

每次候选与导出OBJ都检查实际剩余接触、每pose全部32,768个原始需求、合法空间、物体排除、封闭及单一连通材料。6 mm为候选范围半径，不是全模型最小厚度保证。未提供形状参考时默认只进行BBX裁边、保留整块主体；`--pruning-style legacy`保留[旧Pruner](prune_support.py)用于历史对照。以下网格及独立块处理为旧分支说明，当前连通主体批次不依赖外部桥连接。

## 历史网格分支

先尝试删除多余独立块，再用固定粗、细空间分区删减接触材料。每次重建实际剩余接触三角形的原七维非负反力锥；接受的删减必须通过每个pose全部32,768个原始需求。遇到一个原始反例即可拒绝候选，拒绝记录中的计数标明为下界。相同反力列集合复用检查结果，原基底只在全部对应反力列仍存在时复用。

已经连通的主体不接受留下必需孤岛的切割；多余孤岛随被切区域一起删除。输入本身的必需独立块不能假装为多余块删除，报告明确标记并交给Step5.3连接。有界贪心删减不宣称全局最少材料。

优先使用名义支撑；若保存搜索依赖真实成品接触面才满足全部需求，则从该已认证实体开始删减。不会把名义网格的一条未满足需求直接改成PASS。

必要时从保存的完整成品进行有界重试，记录具体来源。实际物体内部材料按原摆放删除；导出后重新检查真实OBJ的接触、合法空间与封闭边界。数值清理同样必须保留全部原始需求，失败时保留此前已认证的材料。单一实体的真实内腔保留，避免重复重裁相同边界制造碎片。

模型、force certificate和报告在`data/results/B/[set]/step5/step5.2/[experiment]/`。图直接放在`output/B/[set]/step5.2/`：`overview.png`、`removed_material.png`、`all_poses.png`和各`pose_N.png`。本轮另有左旧右新的同视角`comparison.png`。蓝色为保留材料，红色为删除材料；无实验子文件夹、无HTML。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python slides/Co-optimize/step5.2/run.py B \
  --sets pose1+2+4+6 --experiment my_pruning --candidate-budget 64
```

本步骤允许删除接触材料，所以改变接触后重新检查原需求；原物理输入与布局保持不变。
# 历史运行记录（2026-10-12）

最新V8布局的50组中，49组删减材料取得实际全需求证书；`pose1+2+4+5+6+7`的真实接触复查失败，图中明确标记。[每组数量、体积与来源](../data/results/B/evaluation_dynamic_postprocess_all52_20261012_01.md)。新选择可用`--reference-stage step4`，读取该case自己保存的`preconstructed_base.json`与实体；不读取另一布局的暖启动参考。布局、退出和32768个原始需求保持不变。下文保留历史入口及示例。

