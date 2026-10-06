# Step4.2：物理反馈驱动的连续退出方向优化

实现入口为 `physics_guided.py`，算法图为 [algorithm.png](algorithm.png)。各 pose 的退出方向联合优化，pose、物体和原始需求不变。新的实验目录独立于旧算法批次。

## 物理模型与连续目标

现有模型是无上限的非负反力锥，含第七维 shared no-uplift 方程。把射线乘正的面积权重不会改变锥，因此不能通过面积 sigmoid 获得真实的承载梯度。

本实现采用接触获取代价的变分松弛。对原始需求 `b`，求：

```
min  sum(r_plus + r_minus) + eta sum_j c_j(d) x_j
s.t. A_floor f + A_contact x + r_plus - r_minus = b
     f, x, r_plus, r_minus >= 0
```

接触生成元先归一化，消除正射线缩放对获取成本的任意影响。力与力矩使用原 task 的尺度。没有引入面积承载上限、受拉接触或额外自由力矩。获取成本只用于指导优化；它不属于最终接受的物理约束。

LP 的包络敏感度为 `d loss / d c_j = eta x_j`。各困难需求通过平滑最大值汇总。物理求解配置了较严格的平衡容差，并检查平衡残差。LP 解切换时目标可能有折点：同时记录中心有限差分和两侧敏感度，不能将单个 LP 解的包络敏感度无条件称为处处可导的梯度。

## 几何模型

`c_j(d)` 包含原接触法向对所有退出方向的平滑不兼容代价。`--geometry sweep` 进一步查询连续退出轨迹经过的固定物体距离场，考虑非局部遮挡。

固定距离场在原物体周围建立，最大 extent 分成 96 格，采用三线性插值。退出区间采用 129 个固定时间节点及平滑最大值。方向本身是连续变量；时间与距离场离散化是优化近似。它不是精确 sweep 的距离，也不证明没有碰撞。距离成本包含共享规则的最大 clearance kernel 半径。

几何敏感度用方向切平面中的中心差分，步长为 `2e-4`。固定距离场避免布尔 sweep 重三角化引起的最近面跳变。真实几何审查脚本另用 `1e-4` 比较敏感度；合成测试覆盖非局部遮挡、球面更新和实际几何差分。

## 连接模式

`--connectivity` 把最差承载残差改为当前最大正体积材料块上的残差。固定初始支撑的内缩面中心与公共边点构成材料图；边的采样点必须在初始材料内。当前主块提供根节点。接触需要支付到该根节点的路径获取代价，因此 LP 使用的反力为要恢复的材料通路定价。

每轮以当前成本选择路径，再对该路径的距离场成本求方向敏感度。路径切换和根块选择都可能产生折点；这是局部连接指导，不是全局网络优化。图的离散边、不可达标记、中心线和采样内点都不是最终实体连通或厚度证明。不会在原支撑之外添加连接杆。

## 更新与接受

方向用球面切平面坐标更新；SLSQP 联合优化各 pose 的两个自由度，满足自身地面的合法半球及信赖域约束。方向归一化后，用精确、完整的 nominal/padded sweep 从不可变初始支撑重构。

每个候选提取真实接触，检查全部原始 32,768 个需求。困难需求工作集只从保存的需求选择，不重新撒点。候选出现的新失败需求在比较步长前加入工作集，旧支撑和新支撑用相同工作集比较。接受条件现在有两条：真实工作集最差反力锥投影残差下降；或者本轮冻结物理价值的接触遮挡代价在候选方向的非线性距离场上明确下降。允许必要的中间承载退步，记录相对最佳状态的退步量；用有限迭代预算与最佳真实状态保留控制探索，不再要求每一条原先可行载荷都保持可行。几何中间步骤不声明物理可行。本轮方向子问题与几何接受检查使用同一组冻结的物理接触价值，下一轮再重新求物理分配。没有用通过数量作为优化目标。

几何代价来自所有退出通道的平滑共同遮挡，权重来自当前困难需求 LP 的接触敏感度，在整轮线搜索中冻结。候选重新查询完整轨迹距离场；不用线性预测下降作为接受证据。最好的实际构造状态另行保留，结束时如果当前状态缺口更大则返回较好的状态。完整载荷的最终验收仍不放宽。

最终另外检查：完整退出、1% 余量及接触核规则、体积划分、退出终点脱离，是否存在一个能独立满足全部需求的正体积连通块。输出可行块时可删除其余材料。未通过则保留候选结果并明确标未解；数值失败不证明物理无解。只在构造时接受，不做导出模型 replay。

## 运行

从项目根目录运行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  slides/Co-optimize/step4.2/physics_guided.py \
  --set pose1+4+7+12+21+27 --iterations 8

# 加入物理加权的材料连接指导
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  slides/Co-optimize/step4.2/physics_guided.py \
  --set pose1+4+7+12+21+27 --connectivity --iterations 8
```

`--directions <npz>` 固定起点；`--native` 从原生 +Z 开始。`--geometry normal` 是仅法向接触成本的对照，不等于完整几何反馈；连接模式的路径仍使用距离场。`--out` 隔离实验。

`report.json` 保存所有需求通过数、梯度审查、候选接受原因、连续残差、连通块承载、来源及产物哈希。`unresolved.json` 记录数值未决。历史开发结果不作为最终版本的科学接受记录。

## 检查与限制

```sh
.venv/bin/python slides/Co-optimize/step4.2/test_physics_guided.py
.venv/bin/python slides/Co-optimize/helper_func/test_physics_guided_cone.py
.venv/bin/python slides/Co-optimize/helper_func/test_physics_guided_paths.py
```

这是连续、局部、由物理敏感度指导的求解器；没有随机候选或角度枚举。它不保证全局最优或找到每个可行解。距离场与时间分辨率、工作集选择、松弛权重、根块及路径离散化都会影响收敛；当前没有平滑宽度 continuation。也尚未优化接地覆盖、安装地面合法性或强度。

实测结果与未解问题见 [实验记录](physics_guided_results.md)。

## 30 组批次

运行 `run_physics_guided_batch.py`，20 个正常组加 10 个非法组。每组只使用 Step4.1 初始化方向，独立冻结方向与初始化报告；旧版批次为 8 轮；补第7点后最多 24 轮，为中间几何进展留出恢复时间，不启用材料通路优化。并行数由 --workers 指定。新版结果位于 `data/experiments/history/comparisons/physics_guided_step7_batch/B/batch.json`，旧版结果保留于 `data/experiments/history/comparisons/physics_guided_batch/B/batch.json`，每组有原始运行日志和结果。

本轮通过条件为全原始载荷、完整退出及 1% 余量构造通过；连通性另记，不阻止本轮通过。`report.json` 的原有 `passed` 仍包含单连通要求，批次使用另一个字段 `force_exit_passed`，不篡改原报告含义。区分初始可行、优化修复、迭代未解和数值构造未解。

最新独立初始化的 [30 组批量结果](physics_guided_batch_results.md)：3/30 通过，其中 1 组优化修复、2 组初始可行；本轮暂不要求连通。

补第7点后的迭代末状态方向额外保存在 `continuation_directions.npz`。最终 `directions.npz` 与导出支撑对应最佳保留状态；若退回最佳状态，报告记录 `returned_best_constructed_state` 和末状态的 `continuation_counts`，可明确区分中间探索与最终产物。
