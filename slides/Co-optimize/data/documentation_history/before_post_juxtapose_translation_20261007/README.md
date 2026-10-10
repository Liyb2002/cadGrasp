# cadGrasp

## 2026-10-07：多次装卸、多配置共享支撑

当前主线回到每次抓取物体、装入支撑、执行任务、退出物体的几何与承载共同设计。更换支撑摆放前先取出物体；同一件刚性支撑可以在多个摆放姿态下分别使用。

一个支撑姿态可以承载同一物体的多个姿态，进一步可以分别服务不同物体。每个使用配置有自己的物体相对摆放、装卸方向、工作区域和力／力矩需求；支撑姿态与物体姿态不强制一一对应。同时装多个物体属于后续扩展。

- **Step4.1：装卸初始化。**初始化共同／相近合法方向，构造各配置装卸通道与支撑种子。
- **Step4.2：共享支撑优化。**使用 **Direction、Juxtapose** 两个 operations：前者在当前布局上微调退出方向；后者把停滞 pose 直接重叠放到另一个 pose 的支撑位置，利用已有支撑，决定材料在哪里补、怎样连接、补多少。微小 translation 不再是独立 operation。

代码已恢复至 [step4.1](slides/Co-optimize/step4.1/README.md) 与 [step4.2](slides/Co-optimize/step4.2/README.md)。当前抓取／几何锁 Step4 目录已删除，历史实验数据保留。当前代码仍运行旧 direction／translation 搜索；先实现单物体 Direction＋Juxtapose，再研究一般多对多配置与跨物体共享。Juxtapose 与这些扩展尚未实现。

旧 direction／translation 小步七组重跑2/7通过、5/7未通过，没有接受平移，没有分离保底；这不是 Juxtapose 的结果，完整夹具验收仍未完成。新操作都须考虑新增材料和删掉旧墙的代价，检验全部配置。见 [Co-optimize 当前设计](slides/Co-optimize/algorithm.md)、[两个 operations](slides/Co-optimize/step4.2/algorithm.md) 与 [旧实验报告](slides/Co-optimize/data/experiments/multistep_two_tool_v2_B_20261007/README.md)。

## 固定摆放复用与转动摆放复用

研究目标是一件刚性支撑，同时允许两种复用：支撑保持同一种摆放，承载多个不同的物体 pose；必要时取出物体、转动或重新摆放支撑，再用同一件支撑承载其他 pose。多个物体任务可以共享一种支撑摆放，不强制一个物体 pose 对应一次支撑转动，也不强制全部任务只用一种支撑摆放。支撑在每个任务执行期间保持刚性、静止；设计中的几何变化不是使用中的变形。

支撑不必与物体严丝合缝。底部托盘加少量必要接触就可能承载多个 pose；应共同选择承载所需的接触区域、支撑材料及各任务的物体—支撑相对摆放。完整包裹只是一种初始化，不能预先规定最终形状或把所有贴合接触当作必须保留。物体能取出、换姿态后重新装入；同一材料可以在一种支撑摆放下服务多个物体姿态，也可以在支撑翻转后承担不同的接触或接地功能。

退出方向可提供两种复用方式的初始化线索，必须先统一比较坐标系：

- **相对退出方向一致**：指方向在各 pose 的物体局部坐标中一致，即随物体旋转到共同物体参照后方向相同。更适合优先尝试让支撑随任务转动，以保留相近的物体—支撑接触关系；“一个 pose 一次转动”是这种分支的候选模式，不是硬约束。
- **绝对退出方向一致**：指方向在同一世界／工位坐标中一致，例如都沿世界 `+z` 退出、沿 `-z` 放入。更适合优先尝试一个支撑摆放承载多个物体 pose，寻找共同托盘或不同局部接触区域。
- 两种线索可以混合：一部分 pose 共用支撑摆放，其余 pose 通过翻转支撑完成。方向一致是搜索启发，不是几何或承载可行性的充分条件；全部物体占据、装卸路径、工作区、支撑接地及原始力／力矩需求仍需验收。

与 [baseline 选支撑头](slides/baseline_algo/step3_scheculer/README.md) 的衔接：baseline 先从候选接触面中选少量支撑头，每次加入候选后联合重求接触反力，以新增满足的原始载荷数评分，再构造有限厚度连通实体。当前实现各 pose 独立选头，尚未共同搜索上述两种摆放复用。新设计以 Co-optimize 提供包裹与方向初始化，Direction 协调装卸切除，Juxtapose 提出重叠复用布局，再沿用 baseline 的接触贡献评价来决定哪些材料值得补充；全部任务共同验收。既有候选数、头数和头面积不作为新方法的硬约束。

这次记录确定研究问题与搜索线索，不改变现有求解器、数据和验收结果。当前实现入口仍见下方 Co-optimize；新的支撑摆放与接触共同搜索尚待实现。

2026-10-06：当前算法采用共同／相近方向初始化及从 Step4.1 延续的局部下降，以 [Co-optimize](slides/Co-optimize/README.md) 为准；过时的 DSL_algo、DSL_closest_neighbor、co_design_algo 及其生成结果已删除。下方旧算法记录仅供历史参考。

2026-10-03：对象输入已改为共享预计算数据集。每个物体有 30 个 pose、20 个兼容组合（2–6 个 pose 各 4 组），网格最多 5,000 面；固定载荷位于 `objects/<name>/poses/pose_<i>/`，Step2 候选头、力／力矩生成元、退出方向与图保存在其 `step2/` 下；组合与总览位于 `objects/<name>/pose_sets.json`、`sets.png`。生成与复核入口见 [precompute_objects](codes/precompute_objects/README.md)。旧姿态、轨迹及其算法输出属于历史版本；同编号不代表同姿态，不可复用旧 Step2/3/4 结果。

2026-10-02 当前规则：支撑设计须满足[四项条件](slides/obj_supp/README.md)：联合力与力矩平衡、整体不上抬、有限厚度连通实体、完整实体共同插入。[Step3](slides/baseline_algo/step3_scheculer/README.md) 各 pose 独立选头，不要求继承或共享旧头；Step3 验收原始采样载荷的受力，Step4 构造并验证完整实体几何。下方旧流程记录不作为当前入口。

可见 Step5 结果：[B / pose1+3 窄地框合并模型](slides/baseline_algo/output/B/pose1+3/step4/overview.png)、[单独结构对比](slides/baseline_algo/output/B/pose1+3/step4/separate.png)、[旋转查看](slides/baseline_algo/output/B/pose1+3/step4/index.html)。每个 pose 分别连接自己的三头和空心地框，再合并；约 60.8 cm³，比此前厚体少约 89.4%。原载荷与完整退出通过，尚未校核强度。详见 [Step5](slides/baseline_algo/step4_connect_support/README.md)。

当前 B 的配对输出位于 [baseline_algo/output/B](slides/baseline_algo/output/B)：`pose1+3/`、`pose1+4/`、`pose1+6/`、`pose2+8/`、`pose6+9/`，每对下面按 Step1–5 组织。全部旧结果及依赖已迁移，载荷和搜索结果保持原样。

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

当前主线是多次装卸的共享刚性支撑：每个配置都能装入、执行任务并退出；取出物体后可更换空支撑的摆放。共同优化各配置的接触与共享材料，同时检验原始力／力矩、工作面、装卸与接地要求。

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

# 当前各 pose 独立选头
python slides/baseline_algo/step3_scheculer/run_independent.py B --jobs 2
```

使用已安装项目依赖的 `cadgrasp` Python 环境。运行完成、审计通过、设计通过是不同状态，具体以算法说明和当前案例报告为准。
