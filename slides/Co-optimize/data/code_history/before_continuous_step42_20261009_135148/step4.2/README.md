# Step4.2：whole Direction、Juxtapose、Translation 搜索

当前生产实现是 `helper_func/whole_pipeline.py` 与独立的 `helper_func/whole_search/`。从新的 Step4.1 接续，整组 pose 始终参与，目标是在原始承载、完整工作禁区和合法装卸均满足时减少**支撑实体材料体积**。支撑摆放数量不是代价；优先保留转动空支撑复用的布局。

现在进一步用 `compact.py` 从真实可行结果搜索小体积，而不是在第一项 PASS 后结束。`greedy` 继续单路径 Direction／Translation；`beam` 同时保留多个可行布局和少量修复分支，允许重新 Juxtapose、平移、改方向以及恢复原注册复用。最终在各已检查答案及原基线中取真实实体体积最小者。[论文算法](paper_algorithm.md) 给出目标、最远需求、操作、搜索及验收边界；[对照结果](../output/B/compact_comparison.md) 与 [最小支撑图集](../output/B/compact_index.html) 保留两种方法。

先用最不满足的原始力／力矩需求引导单独、联合和协同 Direction 候选，并比较不同角步幅。停滞时选择失败 pose 或已确认的阻挡者，向退出方向相近的 host 提出 Juxtapose；随后用多个地面内 Translation 步幅和 Direction 局部修复。每次仍检查全组，而不是只看被修改的 pose。

未整体可行时，按全组未满足原载荷数、当前最坏需求的反力锥距离、材料估计依次比较；允许个别 pose 退步以改善整组。整体可行后仅接受保持所有原载荷通过的材料下降。最终实际体积不得大于已可行的 Step4.1，保留原实际布局作为回退候选。

候选使用持久接触锁与材料覆盖计数、缓存反力基、共享 Sobol 体积采样域；不重建 Boolean。初始化和最终检查均在独立进程中执行，单次真实几何检查 180 秒超时；初始化先尝试每个 pose 各自世界向上 0.25° 裕量，再尝试原定 0.25°／1° 协同微扰。最终最多检查三个候选的实际接触核心／1% 净空网格，复核原始全部 32,768 载荷／pose、完整工作锥和退出路径。若末次验证数值未决，生产可行性入口另有至多三次有记录的 0.25° 方向延续，`--numerical-recovery-candidates 0` 可关闭额外预算。没有远距离分离或整组 Juxtapose 保底。原物体世界朝向、高度、原始载荷与反力模型不变。

每组保存在 `output/B/{pose_set}/step4/step4.2/`：

- `process.png`：同一参考 pose、固定等轴测视角逐步展示选中支撑；绿色为材料增加，红色为材料删除。最后一格是实际验收网格。无图内文字。
- `final_result.png`：同一个真实蓝色支撑在全部任务下的使用配置，灰色物体、原始橙色工作面。
- `README.md`、`trace.json`、`process.json`：每步选择、host、角步幅、平移距离及全组原载荷计数。
- `data/operation_metrics.json`：按世界物体中心和物体自身退出方向计算实际变化。Juxtapose 换 host 时，原日志的支撑坐标参数差包含坐标系变更，不能直接理解为物体移动距离。
- `process_states/`、`mesh_states/`：实际选中布局与名义过程三角网格。过程的接触采样判断不冒充最终净空验收。
- `support.obj`、`layout.npz`、`*_force.npz`、`data/report.json`：最终实际几何与可复核结果；最终失败仅保存诊断，不能标 PASS。

在项目根目录运行：

```sh
# 从已有的新 Step4.1 运行 Step4.2
.venv/bin/python slides/Co-optimize/step4.2/run.py B --jobs 3
# 两阶段一并运行；可用 --sets 选择已有组
.venv/bin/python slides/Co-optimize/helper_func/whole_pipeline.py B --jobs 3 --sets pose1+2+4+6
# 从可行支撑继续比较两种体积搜索，原基线保留
.venv/bin/python slides/Co-optimize/step4.2/compact.py B --mode both --jobs 2
```

默认预算：8 轮修复、3 个候选、3 轮 Juxtapose 分支修复、3 轮可行后减材料、96 个候选筛选预算、随机种子 42。查看 [详细算法](algorithm.md) 和 [B 图片索引](../output/B/step4_index.html)。工作禁区只在 Step3.2 单独画；普通过程／结果图片不覆盖禁区。当前尚未进行整件支撑接地、连通与强度验收，也不声称全局最优。

新体积延续默认每种方法 6 轮、每轮合计 96 个材料筛选、总计至多 48 次完整原载荷候选评价、3 个真实几何候选。Beam 留 2 个可行布局和 1 个修复分支；`structural-greedy` 是允许换落座但只延续一项可行状态的消融。结果在本组 `compact/{greedy,beam}/`，`compact/best.json` 指向最小已检查答案。六组对照不冒充全部 42 组的新体积搜索；原全部 42 组可行性基线完整保留。

几何超时的原候选可用 `helper_func/resume_whole_validation.py B {pose_set} --timeout 600` 单独继续检查。默认不改变布局；也可显式指定 `--direction-nudge 0.25 --axis 0` 增加一次协同 Direction 微调，完整复核全部原载荷和真实几何。始终保持载荷与容差；原180秒失败记录、额外操作与预算均保存，不能作为相同预算下的独立成功。`collect_whole_step4.py B` 只汇总指纹核对通过的当前结果；`audit_whole_step4.py B --resume` 独立检查真实接触来源和完整工作角度下的射线。

旧 `solver.py`、`run_placement_sampling.py` 与旧两工具实验保留为历史版本，不作为当前入口。
