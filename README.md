# cadGrasp

2026-09-29：当前流程改为 **Step0 随机选择并筛选 n 个 pose → Step1 载荷 → Step2 候选头 → Step3 顺序选头 → Step4 实体构造**。入口 `run_sequential_batch.py B --n 3`；旧 Step4 检查前移到 Step0，旧 Step5 改名 Step4。组合穿地就换一组，全部尝试后仍失败则停止。[当前说明](slides/baseline_algo/step0_pose_selection/README.md)；B 的 n=2 验证在第 7 次选中 [pose3+4](slides/baseline_algo/output/B/pose3+4/step0_pose_selection/floor_point_conflicts.png)，尚未选头或造实体。已有图与模型保留，下面的运行记录及阶段编号为历史。

2026-09-28 当前 baseline 使用[逐 pose 顺序求解](slides/baseline_algo/step3_scheculer/README.md)：每 pose 3–4 个头，后续 pose 从所有已选头中继承一个最佳共享头。每个实体头现在必须避开所有输入 pose 的工作面和地面，包括闲置时。`run_sequential_batch.py B --existing-groups` 已重跑保留的四组：3/4 组全覆盖，但 Step5 仍因固定摆放的落脚条件失败；另一组 pose9 为 32379/32768。0/4 个完整支架。[当前结果](slides/baseline_algo/output/B/pose6+9+10/step4/data/batch_summary.json)与头部 PNG 已更新，旧 pose1+3 做图结果保留。`slides/co_design_algo/` 副本未修改；下方其他构造记录为历史。

可见 Step5 结果：[B / pose1+3 窄地框合并模型](slides/baseline_algo/output/B/pose1+3/step4/overview.png)、[单独结构对比](slides/baseline_algo/output/B/pose1+3/step4/separate.png)、[旋转查看](slides/baseline_algo/output/B/pose1+3/step4/index.html)。每个 pose 分别连接自己的三头和空心地框，再合并；约 60.8 cm³，比此前厚体少约 89.4%。原载荷与完整退出通过，尚未校核强度。详见 [Step5](slides/baseline_algo/step4_connect_support/README.md)。

当前 B 的配对输出位于 [baseline_algo/output/B](slides/baseline_algo/output/B)：`pose1+3/`、`pose1+4/`、`pose1+6/`、`pose2+8/`、`pose6+9/`，每对下面按 Step1–5 组织。全部旧结果及依赖已迁移，载荷和搜索结果保持原样。

2026-09-26 新增顺序式 Step3 实验：十条 particle 先各找 pose1 的三个头，再从各组三头中选对 pose2 贡献最大的合法共享头，为 pose2 补两个新头，形成 `3 + 3 − 1 = 5`。每个 pose 只用自己的三头验收，共享实体和闲置部位接地留到 Step5。入口为 [run_sequential.py](slides/baseline_algo/step3_scheculer/run_sequential.py)，规则与结果见 [顺序式实验](slides/baseline_algo/step3_scheculer/README.md#sequential-3plus2)。下方原五头共同接触版本保留为对照。

2026-09-26 最新 Step3 决定：当前双 pose baseline 在每轮选头时固定每头为工件总表面积的 **1%**（相对拟合容差 `1e-4`），只按原 top5 sampling 选择，不优化覆盖／面积比。选头停止后，若尚未完成且**两姿态各自严格超过 98%**，才在原中心、原头数上尝试终止补全，单头面积依次尝试 1.01%、1.02%、1.05%、1.10%；达到全部样本通过即停，否则失败。已经全覆盖的不扩大。面积最小化仍留到后续 structural geometry 阶段。新结果保存在各 pair 阶段下的 `fixed_area_1pct/terminal_expansion/`，见 [当前 Step3](slides/baseline_algo/step3_scheculer/README.md)。

2026-09-26 用户最终确定：每个 pose 固定使用 Step1 的 32,768 个采样载荷，全部通过即通过；不运行连续载荷域验证，不搜索或追加反例。每个样本仍包含重力，并须满足六维平衡和共享合力不上抬条件。单独的零加工力重力检查只作诊断，不另设通过门槛。插入和基本连通性要求保留。此规则覆盖此前关于连续证明的要求。

给定工件的多个指定任务姿态、工作区域与载荷，研究如何共同设计一件可打印的刚性被动支撑，通过重新摆放，在各姿态下承担不同的工件接触和地面支撑功能，并保持承载、工具可达与简单装卸。当前计算与图像统一为 Z-up，地面为 `z=0`。

2026-09-23 用户确认以**多姿态一体支撑的共同设计**为 SIGGRAPH Technical Papers 研究主线。问题与动机已明确；新方法、制造与实际收益仍待验证。当前四臂概念图不限定新算法拓扑，也不代表已经求得合法的多姿态结构。

- [当前研究主线、路线比较与下一轮推导](codes/research_notes/multipose_rigid_fixture_design.md)
- [最新决定与设计历史](codes/algorithm_design_notes.md#current-research-direction)
- [当前参数与坐标约定](slides/params/README.md)
- [Baseline 代码梳理与新算法衔接](codes/research_notes/baseline_to_shared_fixture.md)
- [一体支撑概念图与机器人流程](slides/reuse/README.md)
- [图目与既有 baseline 模型约定](slides/README.md)
- [当前 baseline 算法：Step1–6、停止规则与现有结果](slides/baseline_algo/baseline_algo.md)
- [value 引导选头方案：保留为候选工具，待实现](codes/research_notes/value_guided_contact_search.md)

新流程由单个机械臂执行：先取出物体并在地面放稳、释放，再单独翻转和放下支撑，最后将物体换姿态并重新插入。初次装载先放支撑，再装物体；任务中物体也接触地面。物体与支撑不锁紧、不粘接、不共同搬运。装入最后一段水平推入，退出先水平拔出；各姿态可使用不同接触区域和接地段。

主收益假设是通过跨姿态共享结构，减少整组任务所需的工装资源与重复材料。“通过几何设计简化机器人装夹”保留为总体动机，但不预设免重装、更快或策略直接迁移。共同求解须与各姿态独立支撑、专用结构拼成的一体件比较，区分一体化与有效材料共享；旧模块＋dock 保留为历史方案和对照。

收益验证比较整套材料与工装准备负担，并完整记录暂存、翻转、重新抓取、对准、插入和重试的执行代价。空支撑稳定、插入中抗滑、任务承载和装卸均待验证；概念图允许的局部穿插不能用于物理结论。目前未开展新方案收益实验。[验证记录](codes/research_notes/reuse_benefit_evidence.md)

## 既有单姿态 baseline

2026-09-26 新增 [双 pose baseline 入口](slides/baseline_algo/step3_scheculer/run_pairs.py)：随机选同一物体两个任务，用同一组工件表面接触头分别计算贡献，10 条独立 top5 sampling，止于 Step4 地面需求。它固定工件—头部关系，尚不求新一体支撑的独立摆放与连接。使用方式及结果见 [Step3 双 pose 说明](slides/baseline_algo/step3_scheculer/README.md#two-pose-baseline)。

历史单 pose baseline 先采样载荷、生成接触头，再运行十条独立 top5 随机搜索，每条最多选择 3 个头并轮流优化全部已选头的尺寸；随后计算地面需求，构造带矩形接口的蓝色模块与独立固定底座，分别验证初始安装、组合对接及工件／蓝块／底座的共享反力平衡。所有头属于同一个蓝色刚体；蓝块和底座通过可拆接口接触。实际工作面必须避开，加工射线禁区目前关闭。当前双 pose 入口以页首固定面积规则为准，止于 Step4。

现有 baseline 对每个 pose 分开求解，允许不同头组，以可靠获得合法接触组和诊断失败为目标，不要求全局最小接触面积。它仍使用初始 rest 安装接触模块、工件与模块整体 docking 的旧流程；组合抓持可行性作为假设，见 [baseline 问题定义](slides/README.md#要解决的问题)。初次安装读取 `objects/<object>/poses.json` 的 `rest` 姿态，接触须避开初始贴地部分。

本次研究路线更新不修改 baseline 实现或结果；其三刚体接口、安装方向和承载证书不能直接用于新一体支撑。当前四组设计对象为逐任务的 A_obj^k、A_floor^k、d_k 和共享实体 V，k 遍历全部输入任务；参数图用两个 pose 举例，不限制任务数量。接触区域在 V 的表面实现，各 pose 均须无体积穿透，并允许物体沿对应方向装入固定的 V。支撑摆放的自由度、实体参数化与求解方法是下一步推导内容。

代码位于 `slides/` 和 `codes/`，物体数据位于 `objects/`。Baseline 输出固定放在
`slides/baseline_algo/output/<object>/pose_<number>/<stage>/`。
Git 保存代码、Markdown、模板和固定测试样例；物体数据及生成的图片、视频、JSON/NPZ 输出保留在本地。新检出环境需要准备对应数据才能重跑案例。

所有运行案例由 `objects/cases.json` 的 `active_objects` 选择，并读取 `objects/<name>/tasks.json` 与 `tasks/<pose>/setup.npz`；网格使用同目录的 `mesh.stl`。当前 21 个物体共 210 个目标。每个物体目录中的 `video.mp4` 是完整连续视频，`trajectories/` 保存从 rest 开始的十段轨迹及时间索引；碰撞模型集中存放在 `objects/_simulation_assets/`。新的 setup 为每个物体选择 `rest → pose_1 → … → pose_10`，把目标、连续搬运轨迹和随机工作面保存在物体目录中。姿态允许按搬运可行性筛选，不作为无偏测试集。各目标接地并由 KUKA 夹爪保持，转移阶段允许短暂抬起。旧 baseline 结果仍属历史输入，重新运行才产生对应新姿态的证书。[案例准备说明](codes/setup/README.md)

换抓版本在每个目标间实际落座、松手，再换抓点和抓取方向；视频只有一个主画面，
没有文字或画中画。各对象记录是否采用闭合后的理想刚性抓持；这种 demo 假设
不代表夹持力或实机实验已经验证。具体抓法、控制参数和连续轨迹随数据保存。

```sh
# 重绘当前 slides；不执行 baseline 搜索
python slides/tools/render.py

# 当前固定 1% 多 pose 联合选头，止于 Step4
python slides/baseline_algo/run_sequential_batch.py B --seed 20260928
```

使用已安装项目依赖的 `cadgrasp` Python 环境。运行完成、审计通过、设计通过是不同状态，具体以算法说明和当前案例报告为准。
