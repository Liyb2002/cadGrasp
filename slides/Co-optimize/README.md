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

The selected set is `B/pose1+2+3+27`; its saved certified final solution contains three Juxtapose poses. The initialization figure superimposes four translucent objects using their own original absolute world transforms, like the overlapping views in `output/B/pose1+2+3+27/evaluation/`. Green patches use each task's original working-face IDs. A light floor at world z=0 appears beneath the overlay and each individual pose; the initialization image omits coordinate axes and per-pose position labels.

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
| Step5.1 | 按最终布局计算全部系统地面需求及凸包 | [run.py](step5.1/run.py) |
| Step5 | 删除冗余接触 → 构造有厚度的圆截面连接 → 从外围向内削减和圆整 | [run.py](step5/run.py)、[periphery_pipeline.py](step5/periphery_pipeline.py) |
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

可显式追加 `--pso-refinement-rounds 4`：在同次冷搜索之后增加独立计数的续优化轮次，使用非零初速度和收缩系数，分别保存当前粒子、个人最好解及认证成品。移动粒子可以暂时失败，最终结果仍须通过全部实体和原始需求检查。默认额外轮数为 0，不读取其他实验的历史优化布局。

实现：[population_soft.py](step4.2/population_soft.py)、[population_pso_refine.py](step4.2/population_pso_refine.py)、[material_search.py](step4.2/material_search.py)。

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

## Step5：必要接触、粗壮连接与外围削减

先在真实导出几何上认证本组自己的完整 Step4 支撑，并保存完整检查点。旧实体如存在微小工件边界重叠，只在独立副本上清理并重新验收；未经认证的实体不能当兜底。

1. **删除冗余接触。** 从外围接触面开始，分组尝试删除，并把失败的组继续拆分。保留面内的反力作用位置也继续筛选；每次接受筛选均检查该姿态全部 32,768 个原始需求。数值求解未能严格判定的删除会被拒绝。筛选结果只是构形引导，必须再由实际有限接触实体通过完整认证，不能用理想接触点代替实体结果。
2. **粗壮圆截面连接。** 使用 `pose1+2+3+4+5+6+7+8` 和 `pose1+2+3+4+6` 的原始成品测量连接尺度。在原合法空间及原世界 BBX 内，为必要接触和地面覆盖位置寻找能容纳实心球的内部连接点，再用圆截面实心杆连接。每个内部节点之间都要有通过原生实体运算检查的完整圆杆通道；大接触帽彼此重叠不能代替该检查。地面选择包含原始点接触和边接触。初次构造及最终连接加固可能需要补料修复旧细颈；削减阶段只接受当前实体的子集。
3. **从外围向内削减。** 逐级缩小圆截面保留区域、接触圆帽，并尝试将过大的端部改成圆截面短杆。失败的批次继续拆分，避免一个必要宽接触阻止其余位置削减。全过程保留内部最小连接实体，随后尝试实际表面圆整。使用圆形包络交集削减，不用局部方块挖洞；圆润程度来自真实模型，展示图不替代几何修改。
4. **核对实际连接。** 最终实体及 Step5.2 中间实体都要包含同一连通的圆杆骨架。必要地面端部被截断时，保留有限接触区域并寻找有足够半径的实际连接；布尔重划分丢失必要的微小平面接触法向时，只恢复该局部有限接触，再完整验收。数值壳片只有在每个内部球和圆杆都仍被包含时才能剔除。最终加固后重新检查全部物理条件。

每次接受实体变更前，使用 float64 精确往返导出并重读实际 OBJ，按原容差检查单一材料连通、封闭实体、合法空间、工件排除、全部姿态地面包围、世界 BBX 和每姿态全部原始需求。完整检查点及需求掩码逐次保存。新版没有旧的 12 块删减预算或 180 秒美化截止；旧流程保留在 `--legacy-feasible`。

重复布尔剪裁产生的共面冗余三角形按原生计算精度以 `simplify(0.)` 简化，简化后的导出实体同样重新认证，避免细分网格累积拖慢完整验收。该 API 的零参数会使用实体当前容差，不代表数学上的零误差；原物理验收容差保持不变。

参考尺度使用保留孔腔的有符号实体切片测量，采样间距 1 mm；两组参考共同采用的球侵蚀半径为 1 mm。最终厚度凭证是实际支撑中保留的连通原生圆杆骨架，并逐段核对内部球和圆杆。最终批次记录的内部凭证半径为 1.10–1.45 mm，均高于参考尺度；1 mm 采样轮廓只作为诊断，因为它不能可靠分辨侵蚀后仅余 0.1 mm 的骨架。这是连接内部的几何尺度检查，不是全表面统一壁厚或结构强度证明，也不宣称全局最小材料量。

材料总体积直接由实际封闭实体计算，接触面积直接由实际接触三角形计算。近重合实体间仅为比较统计而做的差集可能造成巨大网格，因此单独新增／删去材料量报告保守上下界，未精确计算的值明确为 `null`；原来的合法空间、碰撞、地面及全部受力验收仍完整执行。

默认入口依次运行必要接触筛选、圆杆构形、外围削减、必要端部恢复和最终连通骨架认证。实现：[contact_subset_batch.py](step5/contact_subset_batch.py)、[reaction_ports.py](step5/reaction_ports.py)、[anchor_network.py](step5/anchor_network.py)、[tube_worker.py](step5/tube_worker.py)、[recover_interfaces.py](step5/recover_interfaces.py)、[junction_finish.py](step5/junction_finish.py)。完整验收保持在 [feasible_geometry.py](step5/feasible_geometry.py)；[periphery_audit.py](step5/periphery_audit.py) 独立重算最终保存模型和中间实体的物理、接触面积与连接尺度，并在替换 Step5.2／Step5.3 图片前归档旧图。

2026-10-15 重跑批次为 `outside_in_all52_20261015_03`：固定原 Step4 布局、退出方向及原物理模型，52 组中 49 组生成新的完整认证实体；另外 3 组保留原 `STEP4_FAIL` 状态。49 组共 234 个姿态实例，最终实体和 Step5.2 中间实体各通过 7,667,712 个原始需求，实际保存实体的独立复算也全部通过。总体材料由 8,112.08 降至 640.69 cm³，减少 92.10%；各姿态接触面积合计由 17,882.92 降至 386.14 cm²，减少 97.84%；共同静态 BBX 合计由 375.08 降至 370.60 L。材料量和 BBX 分别计算，不混用。

Step5.2／Step5.3 已发布 811 张实际模型 PNG，原 898 张图片先归档再替换。发布图与保存模型渲染图的哈希逐一一致，旧数值模型、执行源码快照和原输入哈希均已核对。最终汇总：[completion.json](data/results/B/data/step5_feasible_outside_in_all52_20261015_03/completion.json)。

重点案例的实际封闭材料体积和各姿态接触面积合计如下。对比图左旧、右新，所有图片均来自实际保存模型。

| 姿态组／对比图 | 材料 cm³：旧 → 新 | 接触面积 cm²：旧 → 新 |
| --- | ---: | ---: |
| [pose1+2+4+5+6](output/B/pose1+2+4+5+6/step5.3/comparison.png) | 155.46 → 14.42 | 283.82 → 28.75 |
| [pose5+6+23+29](output/B/pose5+6+23+29/step5.3/comparison.png) | 97.71 → 14.58 | 530.22 → 11.66 |
| [pose1+2+3+4+5+6+7+8](output/B/pose1+2+3+4+5+6+7+8/step5.3/comparison.png) | 185.69 → 12.73 | 363.67 → 7.57 |
| [pose1+2+3+4+6](output/B/pose1+2+3+4+6/step5.3/comparison.png) | 185.28 → 9.11 | 396.62 → 2.85 |

结果与依据：[全批台账](data/results/B/data/step5_feasible_outside_in_all52_20261015_03/summary.json)、[实际保存实体独立复算](data/results/B/data/step5_feasible_outside_in_all52_20261015_03/actual_solid_audit.json)、[静态空间评估](data/results/B/data/step5_feasible_outside_in_all52_20261015_03/evaluation.json)、[代码版本与默认入口验证](data/results/B/data/step5_feasible_outside_in_all52_20261015_03/source_finalization.json)。实际数值批次保留原执行源码及逐案例修复记录；随后将已验证的修复并入默认入口，另跑完整入口案例和五姿态有限接触恢复案例。12 项几何回归通过。旧数值模型和旧图归档均保留。

## Evaluation 与展示

BBX 使用世界 XYZ 轴对齐盒，体积单位 L；实体材料单位 cm³，二者分别报告。

无支撑基线保持每个任务原朝向，对齐各物体水平包围盒中心并把最低点放在地面，允许顺序任务重叠。其盒子各轴尺寸等于各姿态该轴尺寸的最大值。有支撑列保留实际布局，分别计算全部物体与 Step4、删减主体、完整成品的共同静态 BBX，不计装卸行程。

安装状态按实际支撑世界变换去重，报告每个状态服务的物体姿态成员。Juxtapose／原位转动复用数量是另一种统计，不能由安装状态数直接推算。

`idea/` 根目录保留以下三个当前视频、视频资源 `assets/`，以及指定的表示图目录 `representation/`。视频使用最新 `outside_in_all52_20261015_03` 的认证实体，每部 20 秒、480 帧、24 fps、1600×900，沿用已有 Franka、镜头及画面设计。三组 BBX 增幅均低于 50%，Juxtapose／转动复用比例分别为 3／2、2／3、4／2。任务顺序打乱，每部均有 4 次任务间空支撑转动。六姿态组的成员分布为 4＋2，最充分交错仍需要一次同朝向连续任务，安排为 1 → 3；其他任务间均切换支撑朝向。原三部影片及其分镜已删除并替换。

| 集合 | 当前视频 | 支撑朝向数 | 任务顺序 | BBX 增幅 | 分镜 |
| --- | --- | --- | --- | --- | --- |
| 5 姿态 | [pose1+2+3+4+6](idea/pose1+2+3+4+6_franka.mp4) | 2 | 6 → 1 → 4 → 3 → 2 | +34.28% | [预览](idea/assets/pose1+2+3+4+6_storyboard.png) |
| 5 姿态 | [pose6+10+13+17+30](idea/pose6+10+13+17+30_franka.mp4) | 3 | 17 → 13 → 6 → 10 → 30 | +45.42% | [预览](idea/assets/pose6+10+13+17+30_storyboard.png) |
| 6 姿态 | [pose1+2+3+4+7+27](idea/pose1+2+3+4+7+27_franka.mp4) | 2 | 27 → 4 → 1 → 3 → 2 → 7 | +30.31% | [预览](idea/assets/pose1+2+3+4+7+27_storyboard.png) |

视频代码、实际执行源码、轨迹、选择清单、模型、分镜及核对记录都在 `idea/assets/`。[视频核对](idea/assets/data/verification.json)检查全部 1,440 帧运动、原始数值来源、顺序、真实转动次数、BBX 门槛及完整解码；另检查全部任务分镜和 82 张实际成片抽帧。[本次替换记录](idea/assets/data/round_support_movies_20261015_01/replacement.json)保存旧视频删除哈希和旧元数据。已有表示图保存在 [representation](idea/representation/)。Franka 模型许可证及来源文本在两个机器人资源目录内。视频是运动学展示，不能代替动力学、抓持力或连续碰撞认证。

## 运行

从仓库根目录启动新 Step4 搜索，使用独立实验名：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/run_population_soft.py B --sets all --jobs 3 --output-name my_layout
```

保存 Step5.1 后，以明确的 Step4 台账运行默认 Step5：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step5/run.py B --experiment my_finished_support \
  --selection-summary slides/Co-optimize/data/results/B/data/my_layout/batch.json \
  --step51-experiment my_floor --jobs 3
```

核对保留的三个视频和来源，不重新搜索布局或渲染：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python slides/Co-optimize/idea/assets/finalize_movies.py
```
