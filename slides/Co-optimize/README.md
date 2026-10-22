# Co-optimize：多姿态共享刚性支撑算法

给定同一工件的多个任务姿态、工作面及原始力／力矩需求，共同设计一件刚性支撑。任务依次执行：装入工件、完成任务、取出工件，再搬运空支撑到下一安装状态。同一个安装状态可以服务多个任务姿态。

保持每个任务的原始世界朝向，允许工件座位和支撑安装位置作世界 XYZ 平移，允许工件离地。目标先满足全部原始需求和实际实体条件，再减小整体静态包围盒（BBX），其次减少实体材料。安装状态数、Juxtapose 数量不作为优化代价。

这是本目录唯一的 Markdown 算法说明。数值结果、输入指纹、模型、完整需求证书和运行来源保存在 `data/`，阶段图片保存在 `output/`。

## Pose-set representation in world coordinates

A set contains several task poses. Each tuple is ordered as pose, working surface, exit direction and position:

$$\mathcal S=\{\pi_i\}_{i=1}^{n},\qquad
\pi_i=(\Theta_i,\mathcal A_i^W,\hat{\mathbf d}_i^W,\mathbf p_i^W).$$

The first component is labeled **Pose** and denoted by $\Theta_i\in SO(3)$. It is separate from **Absolute world position**, denoted by $\mathbf p_i^W\in\mathbb R^3$. The position is the world location of the object's mesh-frame origin, with $\mathbf x_i^W=\Theta_i\mathbf x^O+\mathbf p_i^W$. Every task retains its own original transform in the shared world frame $W$. The working surface is $\mathcal A_i^W\subseteq\partial O_i^W$, with scalar area $|\mathcal A_i^W|=\int_{\mathcal A_i^W}dA$. Exit directions $\hat{\mathbf d}_i^W\in\mathbb S^2$ are also expressed in $W$.

The [definition](idea/representation/definition.png) and [initialization](idea/representation/initialization.png) are English PNGs in `idea/representation/`; source and provenance are in its `assets/` directory. No SVG is retained or generated.

The selected set is `B/pose1+2+3+27`; its saved certified final solution contains three Juxtapose poses. The initialization figure superimposes four translucent objects using their own original absolute world transforms, like the overlapping views in `output/B/pose1+2+3+27/evaluation.png`. Green patches use each task's original working-face IDs. A light floor at world z=0 appears beneath the overlay and each individual pose; the initialization image omits coordinate axes and per-pose position labels.

The exit arrows now use the actual saved `step4/step4.1/layout.npz` directions. Step3.1 registers objects into the common material frame $F$; Step4.1 chooses identical or close legal directions **in $F$**, rather than parallel directions in the world. Each task converts its own direction by

$$\hat{\mathbf d}_i^W=\Theta_i\hat{\mathbf d}_i^F,\qquad
\hat{\mathbf d}_i^F=\Theta_i^{\mathsf T}\hat{\mathbf d}_i^W.$$

For this saved initialization, material-frame pairwise angles are 4.17-12.76 degrees. The world-space arrows differ because task poses differ. A separate arrow fan shows the actual relative directions in $F$; all vectors and transforms are recorded in `assets/data/manifest.json`. No illustrative exit vectors are substituted.

The initializer first seeks an exactly common floor-legal direction. If none exists, it reduces angular dispersion $\sum_i(1-\hat{\mathbf d}_i^F\cdot\hat{\mathbf c}^F)$ with unit vectors subject to each pulled-back floor hemisphere. Close material-frame sweeps encourage overlapping cuts in the shared support. The implementation optimizes angular dispersion rather than cut volume directly and does not claim a global minimum cut volume. See [initial_directions.py](helper_func/optimization/initial_directions.py), [Model.initial](helper_func/whole_search/model.py), and [Step4.1 construction](helper_func/whole_pipeline.py). The source layout includes the saved 0.25-degree upward numerical recovery. No numerical initialization, geometry acceptance, search or force check is rerun for this figure.

## 坐标、需求与材料约束

共同材料坐标中的点为 $q$。第 $i$ 个任务的支撑安装变换为 $T_i=(R_i,t_i)$，工件相对支撑的座位变换为 $Q_i$，实际工件世界变换为 $U_i=T_iQ_i$。退出方向 $d_i$ 在共同材料坐标中表示，世界退出方向为 $R_id_i$。工件沿 $-d_i$ 装入、沿 $+d_i$ 退出。

只约束最终安装状态。共同材料须位于每个最终地面之上，并避开各任务的工件内部、工作禁区和完整装卸扫掠。搬运空支撑途中不增加连续旋转姿态约束。

原需求绕工件质心表达，包含重力。整体平移不改变原需求数组；每个姿态使用自己原有角度版本的全部 **32,768 个需求**。接触提供无摩擦、非负、无幅值上限的法向反力，使用原七维平衡模型及不上抬 slack。工件离地后删除工件自身的物理地面反力，合法落地后恢复；不上抬方程始终保留。

工件座位平移改变 $Q_i$。支撑安装平移改变 $T_i$，同时改变该安装状态服务的全部任务的实际质心和地面条件。实际变换保存在 `native_world`，原任务变换在 `native_world_original`，安装偏移在 `state_offsets`。

## 流程与入口

| 阶段 | 作用 | 入口／实现 |
| --- | --- | --- |
| Step3.1 | 所有姿态注册到共同参考，生成贴合包裹，扣除全部工作禁区 | [run.py](step3.1/run.py)、[step31.py](step3.1/step31.py) |
| Step3.2 | 展示每个任务的工作禁区 | [run.py](step3.2/run.py) |
| Step4.1 | 初始化共同或相近的合法退出方向，切除完整装卸路径 | [run.py](step4.1/run.py)、[whole_pipeline.py](helper_func/whole_pipeline.py) |
| Step4.2 | 联合搜索座位、退出方向和安装位置，认证连通主体与完整支撑 | [run_population_soft.py](step4.2/run_population_soft.py) |
| Step5 | 计算地面需求 → 连接主体、包裹凸包 → 对同一实体从外向内删减 | [run.py](step5/run.py)、[wrap_pipeline.py](step5/wrap_pipeline.py)、[wrap_geometry.py](step5/wrap_geometry.py) |
| Evaluation | 从实际保存模型计算静态 BBX、材料和安装状态成员 | [run.py](evaluation/run.py)、[metrics.py](evaluation/metrics.py) |

`step3.3/`、`step5.2/`、`step5.3/` 和 `step5.4/` 保留已有阶段入口或被主流程调用的几何实现。`vis_func/` 集中绘图，`helper_func/` 保存这些阶段依赖的输入、几何、需求投影、布局操作和运行工具。

## Step4：动态角色选择与联合搜索

### 初始化

所有姿态从保存的注册原位共同参与，不预先指定谁是 Juxtapose。Step3.1 使用原 5 mm 包覆范围和原工作角度禁区。Step4.1 寻找共同或相近的合法退出方向，并切掉完整装卸路径。几何初始化完成不代表受力或完整支撑通过。

默认维护 32 个槽位，每个槽位保存整组布局。连通模式从相同注册起点开始，后续操作使槽位分化。默认最多 8 轮，已取得实际完整认证成品时可提前结束。初始化、调整中的临时探针和正式粒子分开记录。

### 每轮操作

1. 根据当前各姿态原始需求失败比例选择操作对象；共同系统尚未可行时，也可使用 Oracle 冲突和地面包围差距。
2. 执行一次离散换座或连续平移。离散操作可以新增／取消 Juxtapose、换 host、交换角色，或联动合并当前安装状态。初始座位按实际世界 XY 质心对齐，再进行合法高度和方向修复。`N/3` 只影响新增／取消的抽样倾向。
3. 做一次有界调整，尝试退出方向、世界 XYZ 平移、共同退出方向和高度修复；全部受影响姿态共同计入收益与损失。
4. 完整检查变异前、变异后和调整后的候选。已认证成品优先；未完成布局保留少量暂时变差的离散探索，历史最好解不能被较差探索覆盖。

普通连续分支使用 PSO：同一个姿态在共同材料坐标中的质心按

$$v \leftarrow 0.25v+0.25r_1(p_{\rm best}-p)+0.25r_2(g_{\rm best}-p)$$

更新后平移原座位。角色、host 和原任务旋转保持；不叠加平移高斯噪声。按原规则以 35% 概率扰动退出方向 12°。已接受的离散换座清零座位改变成员的速度。`--population-continuous-policy differential` 保留原差分对照。

当前 Step4 的规则是：首个完整成品通过全部实体验收后立即停止，不追加候选构造、粒子搜索或优化轮次。保留原来的候选排序与构造时机，不在每次原始候选检查时提前启动昂贵实体构造。初始化认证阶段也可停止；全部载荷、实际完整实体、连通性、地面和禁区检查仍须通过。`first_complete_solid.json` 记录首次通过时的计数，结束时核对之后没有新增搜索。标准入口的 `--stop-when-feasible` 会关闭历史追加轮次，即使旧调用仍传入 `--pso-refinement-rounds`。历史实验的额外轮次仅作为原始记录保留，不读取其优化布局作为新搜索起点。

标准入口使用一个可配置的最多 12 轮找解预算；任何时刻认证通过即停止。这样保留困难组尚未找到成品时的搜索机会，不把旧版第 8 轮后仍在找解的工作误称为成品后的优化。

固定工作禁区的原始半空间只生成一次；[fixed_work_planes.py](step4.2/fixed_work_planes.py) 按实际固定朝向复用法向系数，座位和安装位置改变时批量更新确定性的平移项。出口扫掠按退出方向独立计算，地面截面仍绑定当前裁剪窗口。平面 Oracle 使用半空间裁剪，不采样；跨任务安装地面的相对位置依赖仍完整保留。

实现：[population_soft.py](step4.2/population_soft.py)、[population_pso_refine.py](step4.2/population_pso_refine.py)、[material_search.py](step4.2/material_search.py)。

32 个槽位限制的是外层粒子数。每次移动后，`checked_adjustment → refine_preview → continuation_rows` 还会展开局部搜索：共同／单姿态方向差分、座位和安装位置 XYZ 差分，以及换座和高度修复候选；差分目标使用 Oracle gap 时也会查询 Oracle。因此一个粒子移动可能产生几十次 Oracle 查询。历史 8 pose 组 `pose1+2+3+4+5+6+7+8` 的 5 轮冷搜索和额外 4 轮精修共有 288 个主提案、10,730 次候选预评估、14,596 次实际 Oracle 计算。178,506 次姿态对遮挡判定与 7,572,422 次逐面禁区截面裁剪是不同层级的计数，不能称为完整 Oracle 查询。减少内层探针属于搜索预算调整，应与实现加速分别比较。[查询数量核对](data/results/B/data/oracle_query_count_review_20261015_01/query_counts.json)。

### 全需求损失

连续指导损失是全部需求到当前非负反力锥的积分平方距离：

$$\mathcal L(\theta)=\frac1n\sum_i\frac1{s_i^2}\int\frac12\min_{\lambda\ge0}\|A_i(\theta)\lambda-\widetilde b_i(\xi)\|^2\,d\mu_i(\xi).$$

固定正权求积覆盖原工作面位置、施力角度和强度；同一轮的竞争候选共享节点、权重和尺度，已满足需求也参与评分。需求表达式和有效投影基底可以复用，反力列改变后按原／对偶条件重新检查。硬接触事件之间有平台，操作梯度采用跨接触事件的数值割线。求积指导不能代替最终原始需求证书。

实现：[projection.py](helper_func/continuous_support/projection.py)、[material_force.py](step4.2/material_force.py)。

### 连通接触主体与动态 base Oracle

从实际有限包覆材料中保留一个真实材料连通分量，删除孤块后，用保留主体的实际接触多边形重建反力锥并检查全部原始需求。孤块不提供反力，后续加脚也不能补救主体受力失败。内腔按真实材料处理。

Oracle 仅对 base 可达空间使用原工作三角形沿局部外法向的单向无界扫掠并集：没有角度扩张、全局凸包或几何余量。原需求角度、初始化和接触／材料指导仍沿用原模型。

最终世界质心为 $c_i$，原需求为 $(F_D,\tau_D)$，则

$$\tau_O=\tau_D+c_i\times F_D,\qquad p_i=\left(-\frac{\tau_{O,y}}{F_{D,z}},\frac{\tau_{O,x}}{F_{D,z}},0\right).$$

Oracle 先检查合法地脚点与工件真实接地点的凸包是否包围全部压力点；少数候选再检查共同合法空间中是否存在同一个连通分量，既包含真实接触主体，又在每个安装地面提供足够承载机会。地脚凸包内部可以有空洞，不要求实体围成闭环。有限窗口未找到不能宣称所有布局无解。

最终 Step4 PASS 还要求实际完整带底座实体通过封闭、单一材料连通分量、合法空间、实际工件排除、主体保留、真实地面包围和全部原始需求检查。保存完整模型及 `preconstructed_force_certificate.npz`，作为 Step5 可以保留的认证成品。

实现：[connected_core.py](step4.2/connected_core.py)、[base_oracle.py](step4.2/base_oracle.py)、[material_evaluator.py](step4.2/material_evaluator.py)。

2026-10-15 的首个成品停止实验 `first_feasible_random10_step4_20261015_02` 已实际完成：从全部 52 组按 seed 2026101503 固定抽取十组，3 个 worker、32 槽位、最多 12 轮找解、seed 42，从原 Step4.1 冷启动。**10/10 通过；整批墙钟 72 分 29 秒，未达到 20 分钟。** 前九组在整批约 12 分 50 秒时完成，单组中位数 2 分 20 秒、最慢 8 分 1 秒。最后的 9 pose 组在第 10 轮才首次通过，立即停止，单组用时 72 分 27 秒；该组旧版约 171 分钟后仍搜索失败。本次改变停止规则和找解预算的组织方式，不宣称与旧版追加优化后的最终布局逐字节一致，也不把不同样本和机器负载的历史时间当成受控加速比。

| Set | Pose 数 | 单组 worker 用时 |
|---|---:|---:|
| `pose1+12+29` | 3 | 0:17 |
| `pose1+2+3+4+5+6+7+8+9` | 9 | 72:27 |
| `illegal/pose1+2+6+15` | 4 | 2:20 |
| `illegal/pose5+13+27` | 3 | 1:16 |
| `pose1+2+3+4+6` | 5 | 2:21 |
| `pose1+2+4+5+6+7` | 6 | 4:39 |
| `illegal/pose11+19` | 2 | 0:47 |
| `illegal/pose1+2+6+7+29` | 5 | 8:01 |
| `pose19+28` | 2 | 0:07 |
| `pose1+2+3+4+7+27` | 6 | 4:39 |

每组 `first_complete_solid.json` 的计数均与最终搜索计数相同，首次通过后无新增搜索或构造。实际保存的十个完整实体、1,474,560 条原始需求、675 个冻结源码及 93,565 个受保护文件核对通过；冻结的完整实体构造函数 AST 与原基线相同。9 pose 组仍用了 10,884 次预评估、622 次完整需求筛查和 11,888 次 Oracle 计算；Oracle 计时约 16 分 44 秒。固定系数预计算没有消除当前窗口下逐三角面裁剪和并集，该组仍有 6,855,197 次截面缓存未命中，因此整个找解过程仍存在明显瓶颈。计时不含实现修改和最终独立验收；九组的阶段验收曾短暂与最后一组搜索并行，已单独记录时间。[逐组原始时间及计数](data/results/B/data/first_feasible_random10_step4_20261015_02/results.txt)、[实际实体验收](data/results/B/data/first_feasible_random10_step4_20261015_02/physical_audit.json)。

首次试跑曾把实体构造提前到每个原始候选检查中，改变原构造次序，一组内存涨至约 101 GiB；该试跑在 12 分 36 秒时撤回，另计时间，未混入上述正式计时。修正版保留原候选排序及构造时机，只在首个实际成品通过后立即退出。本次十组正常结束的运行／粒子缓存均已删除；另清理撤回试跑的 34,095 个缓存文件、774,826,163 字节。195 个新成品数值文件、两份冻结源码和原五个未完成实验检查点 SHA 均保持不变。原实验检查点未重启，其他 Step5／Evaluation 工作未改动。[完成记录](data/results/B/data/first_feasible_random10_step4_20261015_02/completion.json)、[清理记录](data/results/B/data/first_feasible_random10_step4_20261015_02/cache_cleanup.json)。

下面记录此前使用独立冻结源码和显式原预算的历史实验，保留其原始阶段及计数口径。

2026-10-15 已将两个局部实现优化应用到常规源码：每个 pose 的工作禁区变换只求逆一次；多边形裁剪按循环索引读取相邻顶点，省去重复的 `np.roll` 和标量 `np.clip`。原顶点顺序、运算表达式和容差保持。实际源码与修改前归档的对照通过 1,520 个裁剪字节一致案例、51 个约束数组字节一致案例和 306 次 Oracle 返回内容一致查询。三个保存的 8 pose 布局及附近探针，共 51 次平面查询的三轮交替计时中位数为 1.75 → 1.17 秒；这是局部查询计时，未重跑整组搜索，也未减少搜索探针或更改最终认证。[实际代码验证](data/results/B/data/oracle_query_count_review_20261015_01/verification.json)。

新入口 [run_population_equivalent.py](step4.2/run_population_equivalent.py) 默认开启 `--equivalent-search`，常规入口需显式传入该参数。它在上述局部优化上增加两种有界复用：禁区平面变换按实际平面、逆变换和脚宽复用，最终截面仍绑定当前裁剪窗口；已经严格证明地面不兼容的查询，只在当前安装变换、压力点凸包、工件接地点和原判定参数逐字节相同时复用。每次仍按原顺序访问约束与压力点缓存，保留原候选、查询次序、随机调用、搜索预算和验收。

曾尝试按损失排序跳过不会获胜的候选 Oracle，但完整运行中发现原压力点凸包缓存使用舍入键且按插入顺序淘汰；省略调用会改变后续命中的数值上下文，约 $10^{-17}$ 的差异已实际改变一次局部胜出候选。因此该剪枝已经撤回，不能把它称为严格等价优化。修正后的版本另外复现了该位置的 29 个候选，输入、Oracle 分数、随机状态和胜出结果均与原版完全一致。[回归证据](data/results/B/data/pso_equivalent_random10_step4_20261015_02/known_divergence_regression.json)。

随机十组实验 `pso_equivalent_random10_step4_20261015_02` 已实际完成并验收。用 seed 20261015 从原加速实验的 47 个正常结束组中抽取 10 组；保留原 32 槽位、最多 8 轮冷搜索、额外 4 轮精修、seed 42 和全部完整实体验收，不使用暖启动。10/10 最终布局数组、支撑主体及完整带底座 OBJ、受力证书和反力数组与保存的原结果逐项一致；实际保存的十个实体、1,638,400 条原始需求及实际源码／输入核对通过。Oracle 见证沿用已有的精确浮点导出适配器；格式差异用原 16 位小数导出的 SHA 核对，几何容差未放宽。[逐组时间及结果](data/results/B/data/pso_equivalent_random10_step4_20261015_02/comparison_results.txt)、[实际实体验收](data/results/B/data/system_pso_equivalent_random10_step4_20261015_02/evaluation.json)。

全部候选选择与最终选解相同，8/10 组完整非计时轨迹也完全相同；另外两组各有一个候选的中间统计不同，严格比较仍标为不通过。`pose12+16+19+21+27` 有 4 个 loss 字段不同；对实际同一候选和每姿态 7,000 个检测点重放 100 次，原有 Trimesh 边界射线的随机重试恰好复现旧、新两个 loss，原评分源码未改。10-pose 组最后一轮 slot 3 有 12 个 loss／失败需求计数字段不同，选解未变；其正常完成缓存已清理，未独立复现本组原因。不能据此宣称所有中间数值逐位确定或所有输入都严格等价。[完整差异记录](data/results/B/data/pso_equivalent_random10_step4_20261015_02/final_result_analysis.json)、[实际随机重试复现](data/results/B/data/pso_equivalent_random10_step4_20261015_02/diagnosis/loss_round2/replayed_losses.json)。

十组 worker 时间之和为历史 706.72 → 本轮 281.39 分钟，单组中位数 37.43 → 21.40 分钟；本轮范围 5.60–89.89 分钟。历史与本轮机器负载不同，这不是受控的纯实现加速倍率。实际 Oracle 计算由 71,636 降至 61,248，差额 10,388 均对应精确地面拒绝复用。另有 9,044,107 次**禁区平面组**变换命中缓存：一个工作三角面棱柱包含 4 个平面，一次计数不是一次完整 Oracle，也不是一个独立平面。逐面片扫描仍有大量工作：十组共 62,218,157 次禁区截面访问，38,467,652 次直接复用截面；剩下 23,750,505 次截面计算中，除上述变换复用外，仍实际做了 14,706,398 次禁区平面组变换。10-pose 组每个非空地面扫描 762 个工作三角面棱柱加 10 个退出禁区，772 × 21,818 次地面扫描正好是 16,843,496 次访问。当前版本加快了内部计算，尚未消除这层逐三角面、逐地面的重复扫描；窗口变化及 8,192 项有界缓存的淘汰也会导致重算。[计数口径和逐组分解](data/results/B/data/pso_equivalent_random10_step4_20261015_02/query_fanout.json)。

本次十组结束后的 `.search_runtime/` 和粒子缓存均已自动删除；另外清理撤回版本及诊断试跑的 105,109 个过期缓存文件（2,167,934,900 字节）。清理后 210 个新成品文件、11,747 个受保护旧文件、674 个冻结源码和原先 5 个未完成组的检查点 SHA 均保持一致。原实验的未完成组没有重启。[完成及清理记录](data/results/B/data/pso_equivalent_random10_step4_20261015_02/completion.json)。

### 缓存、GPU 与断点续算

运行缓存位于各组自己的 `.search_runtime/`：复用相同几何谓词、Oracle 截面、接触和实际反力矩阵证书。内存缓存有界；磁盘按查询键写文件，正常返回后清理，异常时保留同次运行的检查点。因此运行中的组会产生大量临时小文件，同步时应排除 `.search_runtime/`。

`data/results/` 还存放最终布局、完整实体、需求证书、原始输入和源码证据，不能因为本轮尝试结束而整目录删除。只清理已正常结束搜索的临时路径；仍未完成搜索的恢复数据继续保留。

每个已提交轮次保存全部粒子、历史最好解、速度、随机状态、求积测度和投影基底。`--continue-checkpoint` 只恢复同一实验、输入和数值源码的检查点，不能当作历史布局暖启动。

`--gpu-replay auto` 尝试 CUDA float64 批量基底重放；边界或不确定需求仍使用原 CPU 长双精度判断。GPU 不替换原 LP／NNLS 评分，也不改变容差。`off` 关闭 GPU，`required` 将设备错误明确记为运行未解决。

实现：[search_runtime.py](helper_func/search_runtime.py)、[search_checkpoint.py](step4.2/search_checkpoint.py)、[gpu_replay.py](helper_func/gpu_replay.py)。运行进度和结果使用 JSON／TXT，历史数值与冻结执行源码保持独立来源。

本轮加速保留原搜索规则、PSO 参数、随机状态恢复规则、需求和容差，具体复用以下工作：

1. **相同查询复用。** 用数组的原始类型、形状和全部字节生成键，并绑定输入、数值源码及依赖版本。Oracle 的压力点凸包、未改变的工件／禁区地面截面和完整查询结果分别复用；恢复完整查询时一并恢复其连通见证和几何证据。Oracle 包含真实布尔几何与连通检查，首次查询仍有计算成本。
2. **按实际变化重算。** 一个姿态的实际接触、反力矩阵和目标需求完全一致时复用其证书。矩阵改变时重新验证原全部需求，不能只检查此前失败的需求。全局合法材料或安装位置变化仍可能影响多个姿态，须检查所有受影响成员。
3. **控制缓存内存。** 原生布尔实体只作有界内存复用；普通数组、完成后的网格边界和证据可以写磁盘。不能把导出网格重新构造成原生实体来冒充原布尔操作数。可变证据在淘汰时保存最新状态，避免旧证据覆盖新结果。
4. **保存完整搜索状态。** 每个已提交轮次原子保存粒子、速度、历史最好解、随机数、求积和基底状态；未提交轮次从前一个检查点按同一随机状态重放。完成的组立即删除临时目录和粒子缓存，异常组保留自己的恢复数据。
5. **有界 GPU 重放。** 至少 4,096 行的大批量基底验证使用 CUDA float64。小矩阵逆仍按原 CPU 长双精度计算；用保守浮点误差界区分确定通过、确定失败和不确定需求，不确定需求回到原 CPU 判断。LP／NNLS 搜索代价保持原实现。
6. **省掉纯统计布尔运算。** 已有合法原材料为 $N$、当前材料为 $A$，用 $|A\setminus N|=|A|-|N|+|N\setminus A|$ 计算新增材料统计。可能越界时明确保存上下界和统计口径。只替换新增体积的报告表达式；总材料代价、实际几何和全部强制验收仍按原条件计算。实现：[deterministic_base.py](step5.3/deterministic_base.py)、[本批归档的报告适配器](data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/reporting_adapter_sources/closed_volume_reporting_adapter.py)。

Step4 删除物体内部材料时会丢弃清理统计，因此使用 `statistics=False` 省去两次只生成统计值的布尔运算。实际删除、子集检查及全部后续认证保持原样；冻结 worker 的 [统计适配器](helper_func/cleanup_statistics_adapter.py) 在安装前逐句核对原几何与验收代码的 AST，并确认调用者没有读取被省略的统计。实际网格数组一致性、验收调用以及真实 worker 启动路径均有检查。

独立实验 `pso_accelerated_all52_step4_20261014_01` 使用 32 槽位、最多 8 轮冷搜索及额外 4 轮 PSO、seed 42。2026-10-15 已完成 52 组尝试的核对与发布：45 组通过完整支撑认证、2 组正常搜索未通过、5 组仍未完成数值搜索。47 组正常结束并保存最终搜索结果，临时缓存全部清理；5 组的原检查点与恢复所需数据保留。当前没有继续重试的数值进程；这不是“52 组均求解完成”。[52 组实际模型／状态图](data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/all_complete_supports.png)、[逐组结果与耗时](data/results/B/pso_accelerated_all52_step4_20261014_01_acceleration_results.txt)、[最终审计](data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/evaluation.json)。

最终审计验证了全部 45 个实际保存实体、7,012,352 个原始需求 mask、674 个冻结源码文件及 11,541 个受保护旧文件，并另外核对全部 52 组的原上游输入与阶段文件。运行中断进程没有返回完整模块清单或最终 GPU／缓存计数，使用实际日志、启动源码归档及观测到的退出状态记录其来源限制，不把上一次启动失败的遥测归给后一次尝试。旧版同预算实验只有 39 条实际记录；共同成功的 36 组中，32 组布局与完整实体几何一致，另 4 组不同，不能声称全部轨迹或成品相同。共同成功组 worker 时间之和的旧／新比值约 1.40；并发负载、调度、实际完成范围不同，且本轮墙钟包含用户要求的计时调查与暂停时段，不能据此声称受控全链路加速。

本轮 `pose1+4+7+12+21+27`、`pose1+2+3+4+7+27`、`pose1+2+3+4+6+7`、`illegal/pose4+7+12+21+23+27` 和 `pose1+6+11+13+14+17` 的原运行均曾被系统内存不足终止。三组曾在恢复启动时遇到模块导入错误，修正后均实际读入自己的原检查点执行，未新增搜索轮数。第一组、第三组独占续算仍被内核 OOM 终止；第二组、第五组及修正观测问题后的第四组，按用户对长时间重复运行的要求主动终止，保持为搜索未完成，保留全部恢复数据。第三组已经提交四轮 PSO、第四组停在冷搜索第一轮检查点、第五组停在已提交第三轮 PSO 后；不得把这些检查点进度写成最终成品。五个检查点的原 SHA 与载荷校验均通过。[恢复台账](data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/checkpoint_recovery_registry.json)、[最终恢复核对](data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/checkpoint_recovery/after_runtime_review_01/final_recovery_review.json)。主动中断与 OOM 分别记录，不作物理不可行结论。

第四组有一次 SIGSEGV 恰好发生在第 4 次周期栈打印中，输出截断在首个栈帧。环境使用 Python 3.12.2；[CPython 曾修复相应的 trampoline 栈打印崩溃](https://github.com/python/cpython/issues/123967)，但本次根因仍属怀疑。运行包装模板现默认只启用致命信号记录，周期打印需显式设置 `periodic_python_trace`；原数值代码、选项、检查点所有权、随机状态和验收条件不变。观测部分以外 AST 一致；真实子进程线程数验证通过。修正后的同检查点续算与此前 56 次采样计数逐项一致，并越过原崩溃点；后续长时间几何认证由用户耗时要求终止，仍未取得本组最终数值结果。[诊断与实际验证](data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/checkpoint_recovery/after_runtime_review_01/observer_diagnosis/diagnosis.json)。

加缓存前的同预算 PSO 实验实际完成 38 组 PASS，成功组单组耗时中位数 46.97 分钟；两组已完成的 6 pose 分别为 84.51、140.23 分钟，另有一组 6 pose 同样因 OOM 中断，不能把旧版描述成全部完成。更早的 `pso_all52_20261013_01` 没有额外 4 轮精修，49 组成功记录的中位数为 5.37 分钟、6 pose 中位数 12.58 分钟；该版还少了后来加入的导出完整实体再验收代码块，因此不能直接比较两版认证能力。完整逐组记录见 [历史耗时](data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/user_runtime_review_20261015/historical_timings.json)，实际源码差异见 [对比记录](data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/user_runtime_review_20261015/old_to_refine_material_evaluator.diff)。历史并发负载不同，须将旧记录与独占测量分别报告。

本次用原 669 文件冻结源码对 `pose4+5+8+9+19+23` 做独立计时，原输入、32 槽位、最多 8 轮冷搜索及 seed 42 不变。第一份含额外 4 轮的对照在 386.74 秒时已得到完整认证冷搜索解，随后主动结束精修，明确保留为未完成对照。`pso_precache_stop_feasible_single6_20261015_01` 关闭额外精修，保留全部原实体验收，现已完整 PASS：数值求解及保存 397.34 秒，整个 worker 397.92 秒，外层计时 6 分 38.23 秒，峰值 RSS 1,822,920 KiB。六组共 196,608 条原始需求全部通过，669 个冻结文件、2 个外部数值源、17 个实际执行源及 11 个报告成品哈希验证通过；完成后删除粒子缓存，19 个最终文件保持原样。独立输出、不使用历史布局、运行缓存、GPU 或新增适配器。[完整验证与计时](data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/user_runtime_review_20261015/legacy_stop_feasible_01/verification.json)。这是一组代表性 6 pose 的实测，不保证其他难例同样耗时；原 ALL52 加速实验仍未完成。常规入口 `run_population_soft.py` 的额外精修默认就是 0；只有显式增加精修轮数才进入追加阶段。

合法空间见证与支撑均使用保留原始 float64 的 OBJ 格式。固定 16 位小数会改变近共面边界，审计已在一组真实模型中复现其布尔差异；原冻结 Oracle 重建的见证以旧格式导出后，与原文件逐字节相同，改为精确格式后通过原容差。该组的原文件、重建依据及物理模型不变的 SHA 记录保存在 [见证导出修复记录](data/results/B/data/system_pso_accelerated_all52_step4_20261014_01/witness_export_diagnostics/pose1+4+7+12+21+23+27/repair_manifest.json)。恢复 worker 使用单独归档的 [导出适配器](helper_func/exact_witness_export.py)，不改冻结数值源码、模型、需求证书或搜索预算。

批处理入口 [run_pso_refinement_all52.py](helper_func/run_pso_refinement_all52.py) 使用可复用的 [运行包装模板](helper_func/step4_worker_runtime.py)。新任务启动前要求至少 15% 可用内存；同批单个任务常驻内存达到 32 GiB 时，提高到 35%，给正在运行的大内存检查留余量。已开始的计算继续运行，不设置进程内存上限。同实验已保存的结果按原样复用，计算中断后可从自己的完整检查点恢复；调度等待不代表物理失败。

## Step5：先连接、包裹凸包，再逐步删减

默认 Step5 已合并原 Step5.1–5.3：固定已选 Step4 布局、全部姿态、退出方向和原始需求，先构造一件连通、覆盖需求凸包的完整支撑，再对这同一件实体删减，末尾对细连接做小幅局部补强。入口为 [step5/run.py](step5/run.py)。不再先筛选理想接触点，也不在删减途中换成另一件圆杆支撑。

1. **Step5.1：计算需求，长出更宽的地脚。** 从原生角度版本读取每姿态全部 32,768 个配对力／力矩需求，以实际最终世界质心计算压力点：$\tau_O=\tau_D+c\times F_D$、$p=(-\tau_{O,y}/F_{D,z},\tau_{O,x}/F_{D,z},0)$。各需求凸包变换回共同材料坐标。从完整 Step4 主体开始，地脚带宽由 4 mm 加倍到 **8 mm**（向外 4 mm、向内 4 mm），高度仍为 5 mm；原约 5 mm 的贴合主体在此构造步骤不加厚。包裹带扣除各姿态工作禁区、装卸扫掠和工件内部，以短实体连接；必要时在同一合法空间寻找连接路径、补足落地范围。整个已有主体和已接受包裹均保留，完整初始实体通过原验收后才开始删减。
2. **Step5.2：从最外侧开始，优先缩小 BBX。** 对同一个完整支撑尝试外侧平面删减。每次删后检查实际实体，失败则保留上一步；BBX 只按真实剩余几何更新。
3. **Step5.3：保守删除多余材料，保存过程。** 按 3³、5³ 两轮空间网格从外向内删块。若一个删块还会分离无用枝条，可尝试把这些枝条一并删掉，合并后的候选仍须为一件连通实体并通过全部原受力与落地检查。两段删减共用至少保留 55% 初始材料的预算，比上一版 70% 允许更多删减；优先去掉整段多余材料，避免反复细削连接。这是保留连接余量的简单启发式，用户所说宽度是形状目标，**没有 8 mm 截面硬门槛或额外宽度认证**，也不声称强度保证。初始和每次接受删减均精确保存 OBJ、原始需求掩码、材料量和实际 BBX。删减终点单独保存为 `pruned_support.obj`，全部已有检查点保持原样，再执行下面的小幅补强。
4. **末尾局部补强。** 定位完成删减后的细连接，只在附近的小区域把支撑表面向外扩约 **1 mm**，并裁剪到原合法空间、工件外和已经达到的系统 BBX。每处扩张都保留已有实体，通过真实连通与几何检查，最后重新验证全部原始受力需求；不通过的提案保留原形状。原有构造和删减完整保留，最终 `support.obj` 保存补强结果。[实现](step5/wrap_surface_repair.py)、[从已有结果追加](step5/wrap_reinforce.py)。

包裹凸包的环边也可以删除，完整闭环和原接触面都不被锁死。保留条件是：每个姿态的**实际支撑落地点与工件真实接地点的凸包**仍覆盖该姿态的全部需求凸包，同时真实有限接触仍满足原力／力矩需求。几何凸包覆盖本身不是完整受力证明。

每次接受删减前检查：封闭、方向一致、**整个物体只有一个真实材料连通分量，删断就拒绝**；位于原合法见证内；不侵入工件；为前一实体的子集；每个姿态落地凸包覆盖；世界 BBX 不增大；每姿态全部原始 32,768 个需求通过。连通检查计入所有正体积材料块，不能忽略游离小碎片。原体积容差 $10^{-10}\,m^3$、BBX 坐标容差 $10^{-8}\,m$ 及落地包围容差 $10^{-8}\,m$ 不变；固定布局，不重跑 Step4。末尾补强检查已有删减实体被完整保留，并重复原合法性、工件排除、落地、连通、BBX 和全部受力验收；原子集检查继续应用于每次删减。

新运行及检查点单独保存在 `data/results/B/<id>/step5/<experiment>/`，全批台账在 `data/results/B/data/step5_<experiment>/`。实际执行 Python 使用运行前冻结的源码字节，原始数据位置不变。原 `step5.1/`、`step5.2/`、`step5.3/` 仅保留历史入口和共用几何／验收实现，不再是默认执行顺序。旧接触筛选与圆杆重构流程可显式使用 `--legacy-periphery`，原有限预算流程保留在 `--legacy-feasible`。

当前批次 `wrap_surface_reinforced_all52_20261015_03` 数值与独立复核已完成：52 组中 49 组通过，3 组保持原上游 Step4 失败，Step5 新失败为 0；39 项相关回归通过。直接复用 `wrap_balanced_all52_20261015_02` 的完整构造和全部删减结果，49 组初始支撑、删减终点 OBJ 及所有原删减检查点逐字节／逐记录不变，本轮只追加末尾补强。430 张 Step5.3 图片、52 张 Evaluation、三部 Franka 和 02 已同步更新，01 保持原文件字节。[保留核对](data/results/B/data/step5_wrap_surface_reinforced_all52_20261015_03/original_pruning_preservation.json)。

历史对照 `outside_in_all52_20261015_03`：52 组中 49 组完整认证、3 组原 `STEP4_FAIL`；234 个姿态实例，最终实体和旧中间实体分别通过 7,667,712 个原始需求。其材料合计 8,112.08 → 640.69 cm³，BBX 合计 375.08 → 370.60 L。这是旧接触筛选／圆杆重构流程，不是本节新算法；原模型、图片和独立分支继续保留。[历史台账](data/results/B/data/step5_feasible_outside_in_all52_20261015_03/summary.json)、[历史实际实体审计](data/results/B/data/step5_feasible_outside_in_all52_20261015_03/actual_solid_audit.json)。

49 组在删减阶段的材料合计仍为 6226.28 cm³；末尾补强后为 **6282.40 cm³**，只增加 56.12 cm³（**0.90%**），较完整初始包裹 9259.48 cm³ 减少 32.15%。46 组获得局部补强，3 组保持原样。全姿态静态 BBX 合计仍为 **340.444 L**，初始为 366.267 L。234 个姿态实例的 7,667,712 个原始需求全部通过；1259 次原删减与 1308 个原始／删减检查点完整保留，另存 46 个补强检查点，共 1354 个；逐检查点核对 216,629,248 个原需求掩码条目。原删减的贪心顺序和保留预算不变。

| 姿态组／当前 Evaluation | 材料 cm³：初始 → 删减 → 补强 | BBX L：初始 → 最终 | 接受删减次数 |
| --- | ---: | ---: | ---: |
| [pose1+2+4+5+6](output/B/pose1+2+4+5+6/evaluation.png) | 158.58 → 109.14 → 110.58 | 7.952 → 7.061 | 35 |
| [pose5+6+23+29](output/B/pose5+6+23+29/evaluation.png) | 109.15 → 105.07 → 107.53 | 7.213 → 7.010 | 5 |
| [pose1+2+3+4+5+6+7+8](output/B/pose1+2+3+4+5+6+7+8/evaluation.png) | 230.09 → 126.67 → 129.61 | 10.827 → 9.815 | 41 |
| [pose1+2+3+4+6](output/B/pose1+2+3+4+6/evaluation.png) | 219.05 → 160.81 → 161.08 | 6.652 → 6.191 | 11 |
| [pose1+2+3+4+5+6+7](output/B/pose1+2+3+4+5+6+7/evaluation.png) | 188.16 → 170.96 → 172.49 | 12.005 → 10.808 | 7 |

用户指出的七姿态组 pose1 左上角小方块连接已补厚；[前后与局部放大](output/B/pose1+2+3+4+5+6+7/step5/wrap_surface_reinforced_all52_20261015_03/connection_detail.png)左旧、右新。该组增加约 1.54 cm³（0.90%），BBX 不变。

[完整台账](data/results/B/data/step5_wrap_surface_reinforced_all52_20261015_03/summary.json)、[实际实体复核](data/results/B/data/step5_wrap_surface_reinforced_all52_20261015_03/actual_solid_audit.json)、[总览](output/B/step5_wrap_surface_reinforced_all52_20261015_03.png)。上一轮 `wrap_conservative_all52_20261015_01` 的数值、源码和证书保留；严格宽度试跑已撤回。当前宽度仍是造型目标，没有硬性最小截面或强度认证。

各组 `output/B/<id>/step5.3/` 的展示图现已统一更新为当前 `wrap_surface_reinforced_all52_20261015_03` 模型：49 组有效结果共 430 张，原目录的 430 张旧图已删除并替换，不另存旧图副本。`overview.png`、各姿态图和 `all_poses.png` 使用当前最终实体，`bbx_trim.png` 使用实际 Step5.2 最后检查点，`comparison.png` 对比本轮完整初始支撑与最终实体。3 组原上游失败没有最终支撑，不伪造结果图。图片读取补强后的认证实体，原始需求、布局和全部旧删减数值保持原字节。[示例](output/B/pose1+2+4+5+6/step5.3/all_poses.png)、[图片替换核对](data/results/B/data/step5_wrap_surface_reinforced_all52_20261015_03/authorized_step53_updates.json)；可用 [wrap_stage53_images.py](step5/wrap_stage53_images.py) 从当前保存模型重画。

## Evaluation 与展示

BBX 使用世界 XYZ 轴对齐盒，体积单位 L；实体材料单位 cm³，二者分别报告。

每个姿态组只保留一张固定路径的 [evaluation.png](output/B/pose1+2+3+4+6/evaluation.png)：三列依次为 Object only、Object + support、Object + support + feet。图中只保留后两列的整数百分比增幅（例如 +34%）和底部的 Placement 1、Placement 2 等安装位置编号；集合名、列标题、pose 标签及所有说明文字均不打印，也不打印体积倍数或 L。机器数据仍保留精确体积和归一值。所有列使用同一长度比例和等轴测方向。

主图各列只显示同一个代表性物体姿态及对应的一份支撑，不再叠加所有物体或支撑朝向。代表姿态从成员最多的安装状态中选取物体竖向尺寸最大者；完整浅色线框盒子的十二条边表示全部任务的共同静态 BBX，并与物体、支撑进行深度遮挡处理。底部共用一条姿态缩略条，按实际支撑世界变换分组，用括号和 Placement 编号表示共享安装位置。同一组缩略图保持相同镜头中心与比例，保留真实平移关系；所有组共用长度比例。缩略图显示当前完整支撑和各任务物体。未通过的组只显示物体缩略图，不声称已形成复用布局。

中间列读取所选 Step4 的支撑主体，最后一列读取最新版认证 Step5 完整实体，包含包裹地脚和连接的实际删减。因此两列之间的差值包含 Step5 删减，不能单独解释为地脚增加的体积。原 Step4 未通过的组保留物体主图及缩略条，支撑两列留空，未通过状态保留在机器数据中。

本次 52 张现有 Evaluation 图片及对应机器数据已全部更新到新包裹删减结果，49 个认证模型的指标已从实际保存实体独立重算；3 个上游失败组保留真实失败展示。默认读取 `data/results/B/data/step5_current.json` 指向的新全批结果。

固定图路径为 `output/B/<pose-set>/evaluation.png`，固定机器数据为 `data/results/B/<pose-set>/evaluation.json`，全批汇总和核对在 `data/results/B/data/evaluation/`。重复运行直接覆盖，不再按实验名创建 Evaluation 目录，也不生成旧 CSV、HTML 或逐实验报告。历史 Evaluation 目录、重复图和纯派生指标已删除；模型、布局、需求掩码及实际实体审计保留。[当前汇总](data/results/B/data/evaluation/summary.json)、[核对](data/results/B/data/evaluation/verification.json)、[清理记录](data/results/B/data/evaluation/cleanup.json)。

无支撑基线保持每个任务原朝向，对齐各物体水平包围盒中心并把最低点放在地面，允许顺序任务重叠。其盒子各轴尺寸等于各姿态该轴尺寸的最大值。有支撑列保留实际布局，分别计算全部物体与 Step4 主体、Step5 完整成品的共同静态 BBX，不计装卸行程。

安装状态按实际支撑世界变换去重，报告每个状态服务的物体姿态成员。Juxtapose／原位转动复用数量是另一种统计，不能由安装状态数直接推算。

`idea/` 根目录保留以下三个当前视频、视频资源 `assets/`，以及表示图目录 `representation/` 和后处理讲解目录 `post_processing/`。以下三个 Franka 视频已按最新用户要求更新，使用新 `wrap_surface_reinforced_all52_20261015_03` 的局部补强认证实体，每部 20 秒、480 帧、24 fps、1600×900，沿用已有 Franka、镜头及画面设计。三组 BBX 增幅均低于 50%，Juxtapose／转动复用比例分别为 3／2、2／3、4／2。任务顺序打乱，每部均有 4 次任务间空支撑转动。六姿态组的成员分布为 4＋2，最充分交错仍需要一次同朝向连续任务，安排为 1 → 3；其他任务间均切换支撑朝向。原三部影片及其分镜已删除并替换。

| 集合 | 当前视频 | 支撑朝向数 | 任务顺序 | BBX 增幅 | 分镜 |
| --- | --- | --- | --- | --- | --- |
| 5 姿态 | [pose1+2+3+4+6](idea/pose1+2+3+4+6_franka.mp4) | 2 | 6 → 1 → 4 → 3 → 2 | +33.99% | [预览](idea/assets/pose1+2+3+4+6_storyboard.png) |
| 5 姿态 | [pose6+10+13+17+30](idea/pose6+10+13+17+30_franka.mp4) | 3 | 17 → 13 → 6 → 10 → 30 | +41.73% | [预览](idea/assets/pose6+10+13+17+30_storyboard.png) |
| 6 姿态 | [pose1+2+3+4+7+27](idea/pose1+2+3+4+7+27_franka.mp4) | 2 | 27 → 4 → 1 → 3 → 2 → 7 | +29.44% | [预览](idea/assets/pose1+2+3+4+7+27_storyboard.png) |

娱乐版 [pose1+2+3+4+6_work_arm](idea/pose1+2+3+4+6_work_arm.mp4)（[分镜](idea/assets/pose1+2+3+4+6_work_arm_storyboard.png)）：直接复用上表 5 姿态视频保存的 Panda 轨迹、支撑和力箭头，不重新规划，另加 whole_task Step5 的 KUKA LBR Med 14 工作臂，站在 0.12 m 台座上；针轴固定为向下 45°、背离基座指向工位，只允许平移。一个 task 只有同时满足以下条件才加工：全部工作三角形沿针轴都是第一可见（bunny 与支撑都算遮挡物）；把工作面沿针轴投影后，内缩外轮廓加三笔扫线，沿针轴投回曲面，全程可达；连杆和针头座按稠密采样不碰 bunny、支撑、等待中的 Panda、台座和地面。否则工作臂原地乱挥 2 秒，然后进入下一个 task。工作臂位置由搜索确定：基座方位角每 15° 一档、距工位中心 0.60–0.85 m，共 144 个候选；剔除台座碰到保存轨迹中 Panda 或零件的位置（63 个）；远景相机保持距离 2.4 m、俯角 −22° 不变，只从原视角最小程度地转向，剔除装不下整个工作单元或挡住工位的位置（18 个）；其余按可加工 task 数、相机转角、工作间隙、基座远近排序。最多只能做 2 个，而且只有前侧的 11 个位置做得到，都是 pose_6 和 pose_4；后侧台座大多会碰到 Panda，剩下的后侧位置最多只能做 pose_1 或 pose_3 中的一个，因为两组工作面朝向相反；pose_2 在全部 144 个候选里都做不了。选中的位置是方位角 15°、距离 0.85 m，相机从原视角转 60°，全程固定、不推拉。按执行顺序 6 → 1 → 4 → 3 → 2：pose_6、pose_4 描边加工（工作间隙 ≥ 20 mm）；pose_1、pose_3 的第一可见面积只有 0.54、0.43；pose_2 虽然整片可见，路径上针头座会贴近 bunny 到 4 mm 以内，所以这 3 个发疯。视频 36.6 秒、878 帧、24 fps、1600×900；原视频字节不变。代码在 `idea/assets/work_arm.py`，搜索排名、判定、挥舞参数和来源哈希记录在 `idea/assets/data/pose1+2+3+4+6_work_arm/plan.json`。`objects/B/mesh.stl` 已被 whole_task 换成 5000 面细分版，而 Co-optimize 各姿态仍索引原 1000 面网格，因此原网格另存在同一数据目录。该视频是运动学展示，不能作为扫掠碰撞、关节速度或动力学证书。

视频代码、实际执行源码、轨迹、选择清单、模型、分镜及核对记录都在 `idea/assets/`。[视频核对](idea/assets/data/verification.json)检查全部 1,440 帧运动、原始数值来源、顺序、真实转动次数、BBX 门槛及完整解码；另检查全部任务分镜和 82 张实际成片抽帧。[本次替换记录](idea/assets/data/wrap_support_movies_wrap_surface_reinforced_all52_20261015_03/replacement.json)保存用户授权替换的旧、新视频及分镜哈希；新的 82 张实际成片抽帧也已核看。已有表示图保存在 [representation](idea/representation/)。Franka 模型许可证及来源文本在两个机器人资源目录内。视频是运动学展示，不能代替动力学、抓持力或连续碰撞认证。

后处理讲解的[第一个视频](idea/post_processing/01_hulls_and_support.mp4)使用 `pose1+2+4+5+6` 的实际新 Step5 构造。从完整 Step4 主体开始，在同一个固定绝对姿态下显示五个原始需求凸包，按各自 `T_fixed @ inverse(T_i)` 变换；蓝色材料按保存的构造逐步长出。已恢复原版彩色散点、凸包轮廓、360° 环绕主视图和右侧五个安装小视图；原散点索引、姿态与起始镜头保持一致。更宽的地脚长出时仅适量缩小画面以完整容纳材料，不改散点的空间布局。无文字。保持约 5.33 秒、128 帧、24 fps、1600×900，第 1 秒开始长脚。连续长出是实际新增实体的揭示动画；终点是完整认证的 `initial_support.obj`。旧示意模型存在越出合法空间和工件重叠，已由合法构造替换，不再作为认证输入。[分镜](idea/post_processing/assets/01_storyboard.png)、[成片抽帧](idea/post_processing/assets/data/video_wrap_surface_reinforced_all52_20261015_03/01_encoded_review.png)。

[第二个视频](idea/post_processing/02_pruning_and_bbx.mp4)直接接续同一个 `initial_support.obj`，固定相机、绝对姿态和比例。只显示蓝色实体；不画 BBX，无文字、凸包线、点标记、物体叠图或右侧面板。全部 37 个形状来自真实的“初始完整支撑＋35 次接受删减＋1 次局部补强”检查点；不混入另一件支撑，不用最终形状作隐藏保留掩码。环边仍可参与删减，本轮多留连接余量。最终实体与新 Step5 的 `support.obj` 顶点、面及文件字节完全一致：材料 158.58 → 109.14 → 110.58 cm³，全姿态系统 BBX 7.95 → 7.06 L；末尾补强保持 BBX 不变。原删减顺序与形状保持不变，时长缩短至 8 秒、192 帧、24 fps、1600×900；01 文件字节保持不变。两段均通过完整解码并检查成片抽帧。[分镜](idea/post_processing/assets/02_storyboard.png)、[成片抽帧](idea/post_processing/assets/data/video_wrap_surface_reinforced_all52_20261015_03/02_encoded_review.png)、[来源和核对](idea/post_processing/assets/data/video_wrap_surface_reinforced_all52_20261015_03/completion.json)。

## 运行

从仓库根目录启动新 Step4 搜索，使用独立实验名：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/run_population_soft.py B --sets all --jobs 3 --output-name my_layout
```

以明确的 Step4 台账运行合并后的 Step5；入口直接读取原生需求并计算凸包，不要求先运行 Step5.1：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step5/run.py B --experiment my_finished_support \
  --selection-summary slides/Co-optimize/data/results/B/data/my_layout/batch.json \
  --jobs 4
```

核对保留的三个视频和来源，不重新搜索布局或渲染：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python slides/Co-optimize/idea/assets/finalize_movies.py
```

Evaluation 直接覆盖当前图片，不创建实验目录：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python slides/Co-optimize/evaluation/run.py B
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python slides/Co-optimize/evaluation/verify_outputs.py B
```

可用 `--sets pose1+2+3+4+6` 只更新指定组；`--experiment` 只选择输入模型，不改变输出位置。
