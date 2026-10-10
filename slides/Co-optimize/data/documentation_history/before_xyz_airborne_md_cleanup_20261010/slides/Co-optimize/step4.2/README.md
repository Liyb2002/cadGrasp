# Step4.2：全组需求梯度与离散 Juxtapose

当前快速算法见 [公式、操作和实验范围](fast_gradient_algorithm.md)。XYZ统一求导和选择步幅，所有pose原32768需求通过即PASS；不运行几何微扰或末尾重复全需求检查。工作禁区和退出路径继续在搜索里限制可用接触。`stable_gradient_xyz_force_v3` 七组8–10-pose重跑已完成，新搜索7/7通过，其中6组初次、1组自身状态梯度接续；见 [结果](../output/B/stable_gradient_xyz_force_v3_results.md)／[图片](../output/B/stable_gradient_xyz_force_v3_index.html)。[v2体积对照](../output/B/stable_gradient_xyz_v2_results.md) 和 [此前十组结果](../output/B/stable_gradient_results.md) 保留旧验收口径。

七项新方案均导出名义实体mesh，合计721.55cm³，旧逐组最小答案713.01cm³（+1.20%）；三组更小、四组更大。最终择优采用3项新答案、4项旧答案，655.74cm³（−8.03%）。整批含出图21.38min；本轮全部阶段搜索中位403.6s／组、mesh导出中位4.5s。总耗时小于旧v2的42.8min，但搜索中位大于旧271.0s，不能将取消末尾几何验收的收益算成XYZ搜索加速。图集先列新搜索，再列最终择优。

Direction／Translation 使用**所有 pose 的需求到原反力锥的积分平方距离**。数值梯度给出连续更新方向，多尺度线搜索确认进展；少量 sampling 跨过离散接触点的平台，Juxtapose 改变落座结构后继续联合修复。先全组可行，再保持可行减材料、恢复转动支撑复用。候选只更新接触锁和材料覆盖，最后固定布局生成名义mesh作为显示产物。

当前入口 `run.py` → `stable_pipeline.py`：先调用 `run_stable_gradient.py` 冷启动；若仍有未满足需求，仅从本次自己的布局追加最多8轮 `run_fast_state_seat_gradient.py` 修复。采样全部需求通过后不再因为网格导出失败修改布局。所有pose同时参与，连续下降块及所有竞争落座分支保持同一积分点和权重。每轮最多96个廉价落座候选、3个完整分支；失败guest和已通过的blocker一起考虑，必要时包含至多4个两-pose跳步样本。后者是搜索动作，没有“一个state只能fit两个pose”的容量限制。

支撑可以转动到不同 state，每个 state 可支持多个 object pose。实际世界变换相同的支撑摆放归为同一 state，分组保存在每个结果的 `state_groups.json`；容量不设上限，体积仍是可行后的首要代价。

当前Translation用同一个三维梯度和候选池，并在全组可行后做两轮Direction／XYZ材料下降。默认运行名 `stable_gradient_xyz_force_v3`，默认材料上界包含此前v2最小答案。最终只让力／力矩通过且导出的实体材料更小的新结果替换旧答案；mesh导出失败仍是力／力矩PASS，但仅有估计值不能替换旧实体。`material_selection.json` 和批次 `pipeline.json` 分开记录本轮PASS、显示状态和保留结果。

49项相关检查通过，包含旋转host下独立／协同XYZ梯度、地面反力切换、真实两pose保存时不重算需求、导出失败不撤销PASS和流水线不进入几何补救。重跑后的存档核对确认：七组冷启动与原Step4.1数组一致，全部原始mask通过，两个阶段各65份执行源码／快照一致，原输入与历史输出未改，exact调用与末尾重复求解均为0。核对只读取存档、哈希和mesh体积，没有新需求求解或Boolean；见 [记录](../output/B/data/stable_gradient_xyz_force_v3/verification.json)。

所有实验要求显式选择 set，避免重新启动已停止的52组批次。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/run.py B \
  --sets pose1+2+3+4+5+6+7+8+9+10 --jobs 1 --output-name my_gradient_run \
  --incumbent-summary slides/Co-optimize/output/B/data/stable_gradient_xyz_force_v3/pipeline.json \
  slides/Co-optimize/output/B/data/stable_gradient_results.json
```

新搜索输出在 `output/B/{pose_set}/step4/step4.2/{output_name}/`。`layout.npz`、`*_force.npz` 和 `data/report.json` 保存已选布局、原需求mask和供给反力；`force_passed`决定PASS，`full_demand_recheck_run=False`。成功导出时，`support.obj`、`process.png` 和 `final_result.png` 显示固定布局的名义支撑，图片无文字，工作禁区只在Step3.2画。图集分别显示本轮新搜索和 `material_selection.json` 所选来源；导出错误单独记录，不能改变PASS。

历史结果表区分冷启动与自身状态接续。此前十组7–10-pose实验累计10/10通过原载荷与真实网格检查：v17冷启动8/10，另外两组分别经几何微扰和额外梯度修复。这个旧流程的成功率和耗时不能作为当前无几何补救流程的结果。

Step3、Step4.1、原物理检查和历史答案保留。`continuous_legacy_run.py`／`optimizer.py`、`optimizer_v5.py` 为前一套完整接触多边形梯度实现；旧 v5 与52组批次保持停止。[旧连续模型](continuous_algorithm.md) 和 [旧修复](reuse_gradient_repair.md) 保留供比较。整件支撑连通、强度、安装接地及整个连续需求域的证明不在当前实验的接受范围。

以下为历史候选采样与体积延续实现，历史成功率不计入新梯度算法。

## 历史whole候选搜索

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
.venv/bin/python slides/Co-optimize/step4.2/legacy_run.py B --jobs 3
# 两阶段一并运行；可用 --sets 选择已有组
.venv/bin/python slides/Co-optimize/helper_func/whole_pipeline.py B --jobs 3 --sets pose1+2+4+6
# 从可行支撑继续比较两种体积搜索，原基线保留
.venv/bin/python slides/Co-optimize/step4.2/compact.py B --mode both --jobs 2
```

默认预算：8 轮修复、3 个候选、3 轮 Juxtapose 分支修复、3 轮可行后减材料、96 个候选筛选预算、随机种子 42。查看 [详细算法](algorithm.md) 和 [B 图片索引](../output/B/step4_index.html)。工作禁区只在 Step3.2 单独画；普通过程／结果图片不覆盖禁区。当前尚未进行整件支撑接地、连通与强度验收，也不声称全局最优。

新体积延续默认每种方法 6 轮、每轮合计 96 个材料筛选、总计至多 48 次完整原载荷候选评价、3 个真实几何候选。Beam 留 2 个可行布局和 1 个修复分支；`structural-greedy` 是允许换落座但只延续一项可行状态的消融。结果在本组 `compact/{greedy,beam}/`，`compact/best.json` 指向最小已检查答案。六组对照不冒充全部 42 组的新体积搜索；原全部 42 组可行性基线完整保留。

几何超时的原候选可用 `helper_func/resume_whole_validation.py B {pose_set} --timeout 600` 单独继续检查。默认不改变布局；也可显式指定 `--direction-nudge 0.25 --axis 0` 增加一次协同 Direction 微调，完整复核全部原载荷和真实几何。始终保持载荷与容差；原180秒失败记录、额外操作与预算均保存，不能作为相同预算下的独立成功。`collect_whole_step4.py B` 只汇总指纹核对通过的当前结果；`audit_whole_step4.py B --resume` 独立检查真实接触来源和完整工作角度下的射线。

旧 `solver.py`、`run_placement_sampling.py` 与旧两工具实验保留为历史版本，不作为当前入口。
