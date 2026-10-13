# cadGrasp

2026-10-13进一步修改Step5.3：**保留已认证的连通主体，重新生成连续实心圆截面脚；直路受阻时绕开禁区，不使用带孔叶片或大盒桥接。** Step4布局、Step5.1、Step5.2与pose角色冻结，每组整体BBX保持，成品材料不增加。

本轮结果为`continuous_legs_all52_20261013_01`：全部52组Step5.3、Evaluation及实际导出实体审计通过；首轮48组通过，四组在独立恢复批次通过，实际来源逐组记录。263个object pose，123个支撑安装pose，原位复用123／Juxtapose140。每组是一件共享支撑，任务依次执行。

| 指标：全部52组统计和 | 上一版自由表面圆整 | 本轮连续实心脚 |
| --- | ---: | ---: |
| 无支撑整体BBX（L） | 254.408145 | 254.408145 |
| Step4主体整体BBX（L） | 371.057878 | 371.057878 |
| Step5.2删减后整体BBX（L） | 351.890791 | 351.890791 |
| Step5.3成品整体BBX（L） | 432.466717 | 432.466717 |
| Step5.2实际实体材料（cm³） | 3169.428 | 3169.428 |
| Step5.3实际实体材料（cm³） | 9616.340 | 4490.510（−53.30%） |

BBX逐组保持至原数值精度，成品最大体积差为1.192e-9 L；材料在每组均不增加。`pose1+2+4+19`原约1991 cm³盒桥接已替换，成品2164.621→100.886 cm³；`pose5+6+23+29`改为无贯穿孔的实心脚，成品120.825→60.802 cm³。必要工件容纳开口、接触和禁区边界仍保留。

[全部52组数量、BBX、材料和图片](slides/Co-optimize/continuous_legs_results_20261013.md) · [实心脚算法与实际入口](slides/Co-optimize/continuous_legs_algorithm.md) · [BBX Evaluation](slides/Co-optimize/data/results/B/evaluation_continuous_legs_all52_20261013_01_bbx.md)。成品看各组`step5.3/overview.png`，前后对比看`comparison.png`，放大看`detail.png`；所有图直接在各组自己的目录，无HTML。此前已确认的[自由表面圆整结果](slides/Co-optimize/sculpted_pruning_results_20261013.md)作为本轮数值对照保留，公共Step5.3图片现显示本轮实心脚。

另一个PSO对照活动正在跑全部52组Step4→Step5→Evaluation，结果与上述已接受设计分开保存。仅连续位置提案及其历史记忆改变；[代码与配置核对](slides/Co-optimize/data/results/B/data/system_pso_comparison_all52_20261013_01/proposal_scope_audit.json)记录30个原函数/方法保持。[运行状态](slides/Co-optimize/data/results/B/data/system_pso_comparison_all52_20261013_01/work_state.json)与[提案统计](slides/Co-optimize/data/results/B/data/system_pso_comparison_all52_20261013_01/proposal_diagnostics.json)区分已完成组和中途记录。纯PSO不再包含旧位置噪声；历史最优吸引不能保证每组更早成功，最终以同配置冷启动差分和原始验收结果比较。

共同设计一件刚性支撑，在多个最终摆放 state 下支持多个 object pose，满足任务的力／力矩需求，并尽量减少实体材料。每次装入一个物体、执行任务、沿退出方向取出，再由机器人抓起空支撑换到下一 state。一个 state 可以服务多个 pose；空支撑搬运、转动途中的姿态不参与设计约束。

当前主线采用 **whole 联合优化**：所有 pose 从初始化开始共同参与，每个操作同时计算对其他 pose 的收益和损失。工件原任务朝向不变，允许世界 XYZ Translation 和 airborne。airborne 使用同一套绕质心的需求，只移除工件自身地面反力；最低点仍不得穿地。

## 流程与坐标

布局搜索沿用[连通主体与BBX优先删减](slides/Co-optimize/connected_core_algorithm.md)：Step4在原有限包覆范围内保留一个连通材料主体，删除的孤块不参与受力；真实剩余接触必须通过全部原始需求及Oracle。Step5.2只删并圆整自由表面；本轮复用其已认证主体及各组Step5.1，Step5.3在原合法空间内重新生成实心脚，主体自身不靠后续桥接补救。

共同支撑坐标中的材料点记为 $q$，第 $i$ 个任务的最终支撑变换为 $T_i=(R_i,t_i)$，物体相对支撑变换为 $Q_i$，物体世界变换为 $U_i=T_iQ_i$。退出方向 $d_i$ 在支撑坐标中表达，满足 $\|d_i\|=1$；世界退出方向是 $R_id_i$。

| 阶段 | 作用 |
| --- | --- |
| Step3.1 | 所有 pose 注册到共同参考，生成贴合包裹，扣除全部完整工作禁区 |
| Step3.2 | 单独展示工作禁区的环绕等轴测图 |
| Step4.1 | 初始化共同／相近的合法退出方向，扣除完整装卸路径 |
| Step4.2 | 固定槽位中离散换座或PSO平移，再做小调整；实际主体连通、全部原始需求和动态base Oracle共同验收 |
| Step5.1 | 冻结最终布局，计算每个 pose 全部系统—地面需求点与凸包 |
| Step5.2 | 只删材料：优先BBX裁边，再整块删减、圆整自由表面；保持主体连通及全部原始需求 |
| Step5.3 | 保留主体，生成无贯穿孔的连续实心脚；短路径或合法绕行，禁止大盒桥接，保持整体BBX及材料预算 |
| Evaluation | 分别报告无支撑、Step4、Step5.2、Step5.3整体静态BBX；另列材料和state分组诊断 |

## Step4：联合布局优化与动态 base Oracle

当前搜索以[动态换座的32槽位算法](slides/Co-optimize/population_soft_algorithm.md)为基础，加入[真实连通主体验收](slides/Co-optimize/connected_core_algorithm.md)。初始化不指定谁Juxtapose，身份在优化中决定；`N/3`只影响离散操作的抽样倾向，不是数量评分或PASS目标。入口为[run_population_soft.py](slides/Co-optimize/step4.2/run_population_soft.py)。已接受的连通布局覆盖全部52组，包含独立32／48槽位重试和保存候选续算，不能称为相同预算下第一轮52/52。[执行来源及全部组结果](slides/Co-optimize/connected_core_results_20261013.md)。本次圆整没有重新搜索Step4。

### Step4.1：退出初始化

从Step3.1的整组注册布局出发，寻找共同或相近的合法退出方向，切除各pose的完整退出sweep。默认32个sample均为注册原位，所有support pose = object pose，不预设Juxtapose身份。当前连通模式复制相同起点，后续随机操作使槽位分化；原任务朝向与载荷保持。初始化构造成功不代表全部任务受力通过。

### Step4.2：离散换座／PSO平移，再做小调整

连续损失衡量**全部需求到可用反力锥的距离**，不只针对最远的一个需求。工作面上的施力位置为 $r=a_f+u\,edge_{u,f}+v\,edge_{v,f}$，加工力为 $P$，重力为 $g$，绕物体质心的需求为

$$b=(-g-P,-r\times P)=B_fz,\qquad z=(1,P,uP,vP).$$

$B_f$ 直接由物体几何与物理参数构造，不由 32,768 个样本拟合。每个 pose／角度的表达式保存在 [objects/](objects/r6_quadratic.md)。设布局变量为 $\theta$，$A_i(\theta)$ 为原七维平衡模型中的可用反力列，$\widetilde b_i$ 为按原尺度归一化、附加第七维零的需求，则

$$\mathcal L(\theta)=\frac1n\sum_i\frac1{s_i^2}\int\frac12\min_{\lambda\ge0}
\|A_i(\theta)\lambda-\widetilde b_i(\xi)\|^2\,d\mu_i(\xi).$$

反力系数非负、没有上界；$s_i$ 为优化前固定的需求尺度。$\mu_i$ 使用原工作面面积、施力角度／强度与可见性定义，不是 R6 的六维体积。实现用固定正权重求积；局部下降及其竞争分支共用节点、权重和尺度。需求公式不能消除不同接触基底的切换，因此不是每个 pose 只算一次公式就完成全部投影。

有效投影基底 $A_I$ 内，令 $\Pi_I=I-A_IA_I^+$，则

$$\ell_{f,I}(z)=\tfrac12z^T\widetilde B_f^T\Pi_I\widetilde B_fz.$$

基底满足原／对偶条件时复用；删除基底接触或新增反力列违反条件时才重新求投影。退出方向和平移不改变绕质心需求，改变的是接触是否可用。当前硬接触网格在事件间有平台，因此**操作梯度仍采用跨接触事件的数值割线**，加速的是割线两端的全组损失评估。

每个槽位保存一个完整布局，包含全组退出方向、座位、host和支撑安装XYZ。每轮各槽位执行一次变异，再一次小调整，活动sample数不增加；调整中的临时探测不是新的种群成员。

1. **选谁操作。** 主要按当前pose的原始需求失败比例抽样；Oracle尚未通过时加入真实冲突与包围差距。没有可归属pose的共同连接失败可随机探索，已经完整通过的成品不再无目的扰动。
2. **离散换座。** 注册原位pose先尝试Juxtapose；已有guest可新增／取消／换host／交换角色，或联动合并当前安装状态。参考成员在操作时选择，从世界XY质心对齐开始，不用预定名单。`N/3`只控制新增与取消的概率。
3. **PSO平移。** 已有guest可随机走连续分支。每个槽位保留按原物理评分选出的历史最好完整布局，每轮冻结这些历史布局，并按偶数受力／奇数Oracle偏好选择群体最好示范。同一个pose的共同材料坐标质心按`v = 0.25*v + 0.25*r1*(pbest-current) + 0.25*r2*(gbest-current)`更新速度，再平移原座位；角色、host与任务旋转保持。初速度为零，已接受的离散换座清零座位改变成员的速度；不增加平移高斯噪声，保留原35%概率的12°退出方向扰动。显式`--population-continuous-policy differential`保留旧差分对照。
4. **小调整与接受。** 离散或PSO之后，尝试Direction、XYZ Translation、共同退出和高度修复。所有受影响pose共同计入收益与损失；真实验收后比较变异、调整和父代。未完成布局保留少量允许暂时变差的离散探索，完整成品优先。历史最好解也在同轮固定需求测度下比较，较差探索不能覆盖它。

连通模式在Oracle和最终受力认证前检查实际有限包覆实体，删除孤块并只用保留主体的真实接触重建原反力锥。原七维模型要求物体对支撑的竖直合力向下或为零；每pose全部原始需求仍须通过，孤块不贡献反力。Oracle的可连接空间不能替代接触主体自身连通，也不能由后续脚材料补救主体受力失败。

默认32槽位、最多8轮，难组可独立使用48槽位或继续精修保存候选，额外计算与来源分别记录。[旧4布局frontier入口](slides/Co-optimize/step4.2/layout_frontier.py)仅作历史对照。Oracle使用`normal`模式；历史角度模式为`angular`。

每个pose的**32,768个原始需求全部通过**即工件受力PASS。未改变的布局与实体复用保存证书；Step5删减或圆整改变真实接触后，必须重新验收全部原始需求，不能沿用旧mask冒充新实体通过。

物体支撑接触区只须在**全部最终摆放中不穿地**，不设置几何安全余量。物体表面点 $p$ 在共同坐标中为 $q=Q_ip$，其反力列只有在

$$\min_{j\in\text{最终支撑 states}}(T_jQ_ip)_z\ge 0$$

时才可用，并仍须满足原有材料覆盖、工作锥与退出路径条件。物体与 base 地脚可接触 $z=0$；数值容差只处理浮点误差，不预留物理间隙。XYZ 使用同一规则，不额外奖励 +Z；抬高获得的接触、其他 pose 丢失的接触及工件自身地面反力的丢失，共同进入原全组损失。Step4 名义实体导出同时裁去所有最终地面下的材料。

### 动态 base Oracle：地脚是否能包围，并能连上

合法材料空间为

$$\mathcal A(\theta)=\bigcap_i\{q:(R_iq+t_i)_z\ge0\}
\setminus\bigcup_i\bigl(\text{物体内部}_i\cup\text{完整工作锥}_i\cup\text{退出 sweep}_i\bigr).$$

地面半空间只来自最终安装姿态。当前 **Oracle 的工作禁区**改为工作面沿自身局部外法向的单向扫掠体：$W_i^O=\bigcup_{p\in A_i}\{p+\lambda n(p):\lambda\ge0\}$。各原网格工作三角形生成向外无界棱柱，再取并集，不额外扩张角度、不把不同工作面整体取凸包，也不增加余量。此改动仅用于base Oracle及其连接空间；初始化、接触／材料指导与原工件力需求仍使用原30°半角（总开角60°）。每个pose输入的15°／30°／60°需求版本保持不变；普通结果图不叠加工作禁区。

所有pose的最终地面都约束同一件材料。物体坐标里的地面为$(U_ip)_z\ge0$，包含最终XYZ；转换到共同材料坐标后为$(U_iQ_i^{-1}q)_z=(T_iq)_z\ge0$。物体座位 Translation 改 $Q_i$；新增的安装 Translation 改 $T_j=(R_j,t_j^0+\delta t_j)$，其中 $\delta t_j\in\mathbb R^3$，$R_j$ 保持原任务旋转。只改 $Q_i$ 不移动共同地面；改 $T_j$ 会改变该 state 的地面限制和全部所属任务的实际质心、需求凸包与工件地面反力。两套地面方程都保存，不将物体Z重复加入共同偏置。布局文件的 `native_world` 保存实际 $T_j$，`native_world_original` 保留原任务变换，`state_offsets` 保存安装位移；旧布局默认位移为零。

地面本身是世界$z=0$的零厚度平面，材料需满足$z\ge0$；Oracle视频只画薄片边界。即使不向−Z平移，换旋转姿态也可能使原来地面上方的材料穿地。[pose2→pose20的视频](slides/Co-optimize/seperate_base_support_proof/videos/05_pose2_to_pose20_oracle.mp4)将pose2的粉色地面约束随支撑变换带到pose20；最终显示两个地面约束与pose20自己的工作锥。实际地面始终水平，搬运过程只作示意；原布局与受力结果不重新求解。

Oracle 的法向禁区使用原三角形外法向，只有$\lambda\ge0$，没有反向延长。旧版带角度并经可证明自身遮挡裁剪的禁区保留为`work_mode='angular'`，用于读取历史诊断或对照查询。原工件反力需求与接触模型仍用原完整角度锥；缩小base可达禁区不改旧受力数据和已保存的批次。

工作锥按pose独立定义。做任务$i$时只要求$T_iS\cap W_i=\varnothing$；同一件刚性支撑依次通过全部任务，才在共同材料坐标中形成$S\cap\bigcup_iT_i^{-1}W_i=\varnothing$。当前Oracle分别保存各pose的工作锥与坐标变换，区分当前任务的世界工作禁区和跨任务的材料约束，不直接叠加不同任务的世界工作锥。[pose2+20的Oracle工作锥文件](slides/Co-optimize/output/B/pose2+20/step4/step4.2/oracle/work_cones)。

最终世界质心为 $c_i$，原绕质心需求为 $(F_D,\tau_D)$，则绕地面原点的力矩与压力中心为

$$\tau_O=\tau_D+c_i\times F_D,\qquad
p_i=\left(-\frac{\tau_{O,y}}{F_{D,z}},\frac{\tau_{O,x}}{F_{D,z}},0\right).$$

取全部压力点的凸包 $H_i$。位移引起

$$\Delta p_{xy}=\Delta c_{xy}-\Delta c_z\frac{F_{D,xy}}{F_{D,z}},$$

所以 XYZ 位置改变时，需求凸包和查询缓存必须随之更新。

快速查询在每个最终地面上找合法地脚点，按原障碍边界裁剪，不为脚块厚度膨胀禁区，也不设置最小通道宽度。考虑其他安装地面及完整障碍，检查

$$H_i\subseteq\operatorname{conv}(G_i\cup O_i),$$

其中 $G_i$ 是合法地脚候选点，$O_i$ 是该任务中工件自身真实接地点。这个凸包是虚拟承载范围，内部可以有洞，地脚不必沿边界形成闭环。

只有少数受力及包围通过的候选进一步查询共同合法连通分量：同一分量必须包含保留的真实接触主体及其承载锚点，并在每个安装地面提供足够接点来包围 $H_i$；已删孤块的接触不再参与。保存`oracle_connected_witness.obj`作为长脚可用空间见证；它不是成品base，也不计作支撑材料。

当前Step4整体PASS要求**实际接触主体连通 + 原模型向下合力与全部需求通过 + 合法地脚包围通过 + 共同连接空间通过**。未通过的中间布局可以继续搜索；已通过布局的后续减材料不能丢失Oracle可行性。有限查询未找到不等于所有布局无解。详见[Oracle公式](slides/Co-optimize/step4.2/base_oracle.md)及[连通主体验收](slides/Co-optimize/connected_core_algorithm.md)。

## Step5：固定布局下的整块删减与自由表面圆整

### Step5.1：全部地面需求与凸包

冻结 Step4 的 $T_i,Q_i,d_i$，读取原始需求，按最终质心计算上面的 $\tau_O,p_i$，得到每个 pose 全部 32,768 个压力点与凸包。XYZ 平移特别是 Z 的力臂已进入计算；airborne 不需要另一套工件需求。

计算只有数组叉乘、除法和凸包，不做方向／平移搜索、需求重采样或工件受力求解。保存逐 pose 点云、实际变换，以及共同支撑视图。默认支撑自重为零，与原需求模型一致。

### Step5.2：BBX裁边、整块删减与主体圆整

输入必须是Step4已经认证的连通主体。BBX扩展部分优先尝试整面裁边；只有通过原验收才接受，不能为强行缩盒删除必要接触。原连通批次已经取得的BBX缩减作为本次固定目标，进一步美观／省材料删减保持该组整体BBX。

1. **整块删减。** 从上一版已认证主体开始，旧网格删减形状只指导候选范围，尝试3 mm／4.5 mm圆整保留区并与当前材料求交。保留较宽的脚根部重叠区和必要地面端部，避免逐格啃出缺口及细连接。
2. **固定功能边界。** 一致细分实际闭合网格，通常目标边长3 mm，过大网格采用5 mm；固定距实际物体表面0.3 mm以内的顶点及整体BBX的支撑极值。
3. **圆整自由表面。** 其余顶点做8次邻域平均调整，步长0.4，累计位移上限先试0.9 mm、失败再试0.55 mm；用真实曲面切向量插值并细分。固定边用线性切向量，候选以20 μm阈值简化后裁回输入材料。这些是候选参数，不是统一圆角半径或最小厚度。
4. **真实验收与回退。** 每次接受只删除当前材料，保持一个封闭连通主体。导出16位小数OBJ并读回，检查实际接触、每pose全部32,768个原始需求、合法空间、物体排除和固定整体BBX；失败时保留上一次认证结果。

输出`pruned_support.obj`及新实体的原需求证书。当前主体是上一版主体的子集，也在原Step4材料之内；不能添加桥来补救删断的主体。工作／退出禁区、布局、载荷及真实内腔保持。

初期版本在连续圆整失败后尝试小范围凸棱圆角；复杂模型的最终恢复分支关闭这一布尔备用操作，使用有界查询和连续表面圆整，失败即保守回退。最终52组选用的实际版本由来源清单分别记录。[算法与参数](slides/Co-optimize/sculpted_pruning_algorithm.md)。

### Step5.3：连续实心脚与合法绕行

保留完整已认证Step5.2主体。在原Oracle空间中扣除各实际工件内部，并按全部最终安装状态裁进上一版整体BBX；不移动物体或支撑安装姿态。本版重新生成脚，替换上一版固定宽扁脚截面的处理：

1. 从原成品真实地面边界中删去冗余目标，保留包围全部地面需求所需的端点；对支撑贡献的整体BBX极值，保留一个真实边界端点。恢复分支可另选覆盖仍成立的合法地面端点。
2. 优先向当前连通材料生成短圆截面实心脚，通常候选半径3–5 mm，根部适当加宽；实际裁剪后逐条检查单一封闭连通、正材料重叠，以及无贯穿孔／内腔，拒绝带孔叶片。
3. 直路受阻时，按保留真实内腔的合法实体切片构造有界路径指导，优先较宽通道；路径压缩后生成连续胶囊体，并裁进真实合法空间。恢复分支使用轴向邻格防止穿角与捷径不推进，增加根部／合法端点选择；不扩大连接盒。
4. 导出16位小数`support_with_base.obj`并读回，检查完整主体保留、原合法空间、工件排除、封闭和单一材料连通，实际接地包围全部地面需求；每pose全部32,768个原始力／力矩需求重新验收。

新脚可在原合法空间中重新选路，成品总材料必须逐组不增加，整体BBX保持原数值精度。几何包含／排除限值1e-10 m³，BBX边界1e-8 m、体积1e-8 L。导出守卫关闭旧盒桥接修复，失效候选不能重新长出大块连接。[完整算法与恢复入口](slides/Co-optimize/continuous_legs_algorithm.md)。

无孔检查针对新脚，工件容纳区和主体必要开口保留。图中的彩色环是虚拟需求凸包，不填成实体底环；当前验收沿用原力学与几何模型，不代表结构强度或统一最小厚度证明。

## Evaluation：静态工作空间

主指标为`static_occupancy_v2_overlapping_objects`：每组固定布局下整体静态占用并集的BBX体积，单位L，不加入装卸行程。材料体积另列cm³，不能替代BBX。记有支撑时的实际物体为 $O_i=U_iO$，Step4主体为 $N$，完整支撑为 $S$，固定世界XYZ包围盒体积为

$$V(X)=\prod_{a\in\{x,y,z\}}\left(\max_{p\in X}p_a-\min_{p\in X}p_a\right).$$

**无支撑叠放基线**：任务依次执行，各 pose 可使用同一空间。保留原任务朝向，把每个物体的水平包围盒中心对齐、最低点放在 $z=0$，得到 $\bar U_iO$。不沿用支撑优化后的平移位置与高度。若第 $i$ 个朝向的物体盒长为 $d_{i,a}$，则

$$V_{\rm no\ support}=V\!\left(\bigcup_i\bar U_iO\right)=\prod_{a\in\{x,y,z\}}\max_i d_{i,a}.$$

这达到固定朝向、自由平移的 XYZ 盒下界，不把各 pose 的盒子体积相加。重叠表示先后执行任务，不是把多个物体同时装入。

**有支撑的整体盒**：保留保存的真实物体布局，对 $\bigcup_iO_i\cup\bigcup_sT_sN$、$\bigcup_iO_i\cup\bigcup_sT_sS$ 分别取一个共同世界XYZ盒。Step5.2删减主体也单独计算该指标。阶段Evaluation图从左到右为无支撑叠放、加Step4、加该阶段实体；成品在Step5.3。同等轴测方向与长度比例，各列分别居中，不改支撑摆放，也不重跑布局优化。

**共享 state 空间诊断**：按相同实际支撑变换 $T_s$ 分组，组内 pose 集合为 $I_s$，$O_s=\bigcup_{i\in I_s}O_i$。若给不同 state 单独分配空间，可另列

$$V_{\rm state,object}=\sum_sV(O_s),\qquad
V_{\rm state,4}=\sum_sV(O_s\cup T_sN),\qquad
V_{\rm state,5}=\sum_sV(O_s\cup T_sS).$$

任务依次执行。一个支撑 state 服务多个 object pose 不表示多个物体同时装入；盒子表示这一空间需要容纳该 state 的全部任务。

该统计及最终落座物体整体盒只作诊断，不作为无支撑基线。整体盒与多个独立空间的体积之和是不同指标；旋转状态可由机器人搬运，也可使用同一工作站，state 空间统计不表示必须有多个工位。

逐 pose 保存静态尺寸，逐 state 保存成员与盒子，另列实体材料 cm³。Juxtapose pose 指最终 host 已改变或相对位置 $Q_i\ne I$；保持注册关系的转动复用指仍保持自己的 host 且 $Q_i=I$，允许该 state 的安装位置平移。这两类合计为 pose 数。一个 state 支持几个 object pose 是另一种共享统计，可同时包含 Juxtapose pose。

Evaluation表显式列出支撑pose数、每个支撑pose服务的物体pose数与成员。支撑pose由保存的实际世界位置和朝向去重；同一件支撑可有多个安装pose。数量之和等于物体pose总数，JSON中为`support_pose_count`、`object_pose_counts_per_support_pose`与`fixture_states`。

## 验证与保存结果

本轮连续实心脚全部52组Step5.3及实际导出实体审计通过，8,617,984条原始需求mask全部通过；11,524个原输入／历史数值文件保持原字节。初始批次48/52，两次选用的独立恢复各2/2，最终来源见[resolved_batch.json](slides/Co-optimize/data/results/B/data/continuous_legs_all52_20261013_01/resolved_batch.json)。六项孔洞、内腔与路径回归通过；[实际实体审计](slides/Co-optimize/data/results/B/data/system_continuous_legs_all52_20261013_01/actual_solid_geometry_audit.json)及[执行依赖核对](slides/Co-optimize/data/results/B/data/system_continuous_legs_all52_20261013_01/executed_source_scope_audit.json)保留完整证据。

以下记录此前自由表面圆整及更早的历史数值身份。

当前已接受版覆盖B的全部52个2–10姿态集合，包含10个`illegal/`集合。Step5.2／5.3各52/52通过，两阶段分别保存263个pose共8,617,984条原始需求的完整通过证据；52件实际导出主体和成品的几何审计通过。11212个原输入／历史数值文件、6915张旧PNG存档及执行源码副本均已核对。

当前1570张PNG包含104张同视角前后对比和52张脚部放大图；每组阶段图均更新，Step4图片保持。最终选择为[resolved_batch.json](slides/Co-optimize/data/results/B/data/sculpted_pruning_all52_20261013_01/resolved_batch.json)，按组解析六个实际执行来源；初始批次和中断／失败尝试保留各自身份，不称为一次统一执行52/52。[全部组结果](slides/Co-optimize/sculpted_pruning_results_20261013.md) · [实际实体审计](slides/Co-optimize/data/results/B/data/system_sculpted_pruning_all52_20261013_01/actual_solid_geometry_audit.json) · [全需求、来源和图片审计](slides/Co-optimize/data/results/B/data/system_sculpted_pruning_all52_20261013_01/sculpted_audit.json)。

### 历史实验与对照

上一版6 mm整块保留区圆整已完成52/52，但保留了较多棱面；本次在相同整体BBX下进一步删除并圆整真实自由表面。[上一版算法](slides/Co-optimize/aesthetic_pruning_algorithm.md) · [上一版52组结果](slides/Co-optimize/aesthetic_pruning_results_20261013.md)。原V8冷启动Step4为50/52通过，其实体后处理为49/52；这些统计与当前连通52/52版分开。[V8搜索](slides/Co-optimize/population_soft_results_20261012.md) · [V8后处理](slides/Co-optimize/dynamic_postprocess_results_20261012.md)。

以下为此前暖启动和更早的独立实验，保留其原统计与阶段编号。暖启动运行覆盖 `Co-optimize/output/B` 的全部52个2–10姿态集合（含10个`illegal/`集合）。Step4采用8个sample、8轮搜索和初始化预算32，已有布局的暖启动来源明确保存；52/52完成完整实体和Evaluation。全部36个4–8姿态集合满足2–4原位复用＋2–4 Juxtapose；这是该批结果，不是当前算法的数量目标。最终发布为`material_output_all52_20261012_final`，复用本轮`material_output_all52_20261012_01`的布局、需求和51件成品，只在同一保存合法空间内重构`pose1+2+4+19`，修正构造后残留的外部反向边界，不改变布局、物理门槛或原需求。

更新1150张阶段图及一张52件支撑总览；263个姿态实例的8,617,984条保存需求全部通过，52件导出模型连通、封闭，1332个原始输入／既有布局／证明视频的指纹未变。每组当前成品与Evaluation图片直接放在 `slides/Co-optimize/output/B/[set]/step5.2/`，不使用 HTML，共419张 PNG；`illegal/` 集合保留前缀。263个 object pose 对应112个支撑安装pose，每组是一件共享支撑，任务依次执行。独立集合三列BBX统计和为254.41／389.14／433.19L，完整材料统计和12,527.10cm³；这是暖启动批次，不是冷启动性能对照。[直接看单组姿态图](slides/Co-optimize/output/B/pose1+2+4+19/step5.2/all_poses.png) · [52件支撑总览](slides/Co-optimize/output/B/all_supports_material_output_all52_20261012_final.png) · [全部52组Evaluation与pose数量](slides/Co-optimize/data/results/B/evaluation_material_output_all52_20261012_final.md) · [CSV](slides/Co-optimize/data/results/B/evaluation_material_output_all52_20261012_final.csv) · [直接图片来源记录](slides/Co-optimize/data/results/B/data/system_material_output_all52_20261012_final/direct_images.json)。

4布局搜索的新增10组实验为`normal_frontier_more10_20261011_01`：Step4通过6/10，实体构造完成5/6，最终Evaluation完成5/10；其余4组搜索未通过、1组构造未通过。同条件旧单布局算法在这10组中完成7件；共同完成的5组中，完整BBX增加4组、不变1组，没有减少的组。Juxtapose数3组不变、2组增加，没有完成的减少案例，因此这批尚未检验到减少Juxtapose后的体积收益。两个抬升高度约0.93m／1.86m的解分别占用1440.35L／8044.73L，体积是BBX，材料分别为734.79cm³／660.81cm³。[新增10组Evaluation与旧版对照](slides/Co-optimize/data/results/B/normal_frontier_more10_review_20261011_01.md)。此次搜索参数不变，旧模型只作只读Evaluation，原始输入和历史结果486个文件的指纹保持。

安装XYZ／零余量版通过76项相关测试，并以新实验 `movable_fixture_zero_margin_smoke_v2` 跑通 `pose2+20`、`pose1+2+4+7`、`pose1+2+3+4+5+6+7+8+9+10`：3/3从Step4通过全部原始需求与Oracle，完成Step5.2实体及Evaluation。十pose布局含非零安装位移，Step5使用实际安装变换计算需求凸包。完整静态BBX依次为4.59L、8.72L、35.67L。退出凸包筛选与全部可用接触同分量要求保持；未重跑全部52组。[试运行静态评估](slides/Co-optimize/data/results/B/evaluation_movable_fixture_zero_margin_smoke_v2.md)。

上一批完成链为 `base_oracle_compact_clearance_v1` → Step5.1／Step5.2／Evaluation `clearance_compact_v1`，**52/52** 集合全部完成，包括原 `illegal/` 中10组及全部10个≥7-pose集合。263个pose实例，8,617,984个原始需求均通过；该批仍使用5mm接触余量和固定支撑安装位置，早于本次安装XYZ／零余量改动，保留其原始来源。物体允许airborne，地脚仍落在z=0。50组一个支撑state，2组两个state。成品均为连通、封闭实体，保存布局不变，没有重复工件受力验收。

| 统计 | 历史`clearance_compact_v1`结果 |
| --- | ---: |
| Step4 名义实体材料总量 | 20,071.94 cm³ |
| Step5.2 完整实体材料总量 | 20,817.49 cm³ |
| 材料净增幅 | 3.71% |
| 每组新增材料中位数 | 11.63 cm³ |
| 无支撑叠放 BBX 的独立集合统计和 | 254.41 L |
| 最终布局加 Step4 支撑整体 BBX 的统计和 | 769.27 L |
| 最终布局加完整支撑整体 BBX 的统计和 | 834.76 L |
| 最终 Juxtapose／原位转动复用 pose 数 | 251／12 |
| 多 object pose／单 object pose 的 state 数 | 51／3 |

跨集合统计和不是一个共同工作站的大小。pose1–10：无支撑叠放5.27L，实际Step4布局32.80L，完整支撑35.67L；pose2+20三列为4.11／4.59／4.59L。Step5增加BBX的中位数为8.52%，全部组均使用少量小地脚，没有退回整块薄层。

保留旧 `base_oracle_v3` → `connected_base_v2` 作同指标对照；旧批次早于5mm接触余量，不能作为新条件下的通过结果。新完整BBX统计和相对旧2166.02L下降61.46%，50组减少、2组增加；全部10个≥7-pose集合中8组减少、2组增加。pose1–10从旧118.51L降至35.67L，减少69.90%；仍为可行构造，没有全局最小占用证明。[全部新旧对比](slides/Co-optimize/data/results/B/data/system_clearance_compact_v1/comparison.md)。

Step4三进程含图720.35秒，Step5.1含图87.44秒，Step5.2首次批次41.57秒、单组续算1.17秒，Evaluation含图11.81秒。第一次实体构造完成51件；修正已合法材料重复裁切导致的数值碎片后，只续算pose3+15，其余51件保持原字节。两次执行来源及合并完成记录均保留。

[历史静态空间、共享分组与逐组数值表](slides/Co-optimize/data/results/B/evaluation_clearance_compact_v1.md)；历史阶段 PNG 保留在各集合的对应实验目录。

## 输出与文件生命周期

`slides/Co-optimize/output/B/{pose_set}/`保存各步骤最终可视化：Step3.1／3.2、Step4.1／4.2、Step5.1／5.2／5.3与Evaluation。每组图在自己的阶段目录，不使用HTML；原`illegal/`集合保留路径。

当前成品看`step5.3/overview.png`；`all_poses.png`及`pose_N.png`展示实际任务姿态。两个阶段的`comparison.png`均为左旧右新、同视角同比例；`step5.3/detail.png`放大脚部，`step5.2/removed_material.png`展示删除材料。[大块桥接替换前后](slides/Co-optimize/output/B/pose1+2+4+19/step5.3/comparison.png) · [带孔叶片替换前后](slides/Co-optimize/output/B/pose5+6+23+29/step5.3/comparison.png) · [全部52组图片索引](slides/Co-optimize/continuous_legs_results_20261013.md)。

对应的最终布局、force mask、需求点云、实体模型、Oracle 空间见证、计算报告与批次统计保存在 `slides/Co-optimize/data/results/B/{pose_set}/` 或其 `data/` 子目录。这些是后续步骤与复用需要的最终数据。旧报告中的原输出路径由兼容读取器解析，不重新计算输入。

按用户要求，`pose2+20/step4/step4.2/oracle/`例外保留视频及其工作锥OBJ／无界方程NPZ／说明JSON，方便直接查看每个任务的约束；它们是已有布局的诊断输出，不覆盖原受力与实体结果。

已接受的历史数值、旧图存档、冻结执行源码及失败／中断记录保留，不因发布当前图片而改写旧结果。新实验使用独立名字和来源清单；当前公开PNG可更新，原物理输入与既有证书保持原字节。

## 使用与相关目录

当前Step4入口为动态角色搜索，默认32槽位并启用连通主体验收；从仓库根目录启动独立新实验：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/run_population_soft.py B --sets all --jobs 3 --output-name my_connected_layout
```

当前推荐连续实心脚的[安全路径入口](slides/Co-optimize/helper_func/run_continuous_legs_safe_recovery.py)，包括轴向邻格、扩展根部及合法地面端点选择。以下示例冻结上一版已接受的Step4／Step5.1／Step5.2，使用新名字启动全部52组独立Step5.3；[初始入口](slides/Co-optimize/helper_func/run_continuous_legs.py)保留本轮首批运行来源：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/helper_func/run_continuous_legs_safe_recovery.py \
  --experiment my_continuous_legs --jobs 3
```

当前全部52组实际最终来源见`slides/Co-optimize/data/results/B/data/continuous_legs_all52_20261013_01/resolved_batch.json`，输入选择和上一版参考在对应`system_continuous_legs_all52_20261013_01/`内。上述命令只执行独立Step5.3并暂存图片；[finish_continuous_legs.py](slides/Co-optimize/helper_func/finish_continuous_legs.py)负责解析实际通过来源、Evaluation、实体审计及发布。

主体的自由表面圆整仍使用[run_sculpted_surface_only.py](slides/Co-optimize/helper_func/run_sculpted_surface_only.py)等已记录入口。原`step5.2/run.py`默认是历史6 mm保留区算法，原`step5.3/run.py`默认是旧扁脚后端；其默认行为不代表当前连续实心脚。[本轮结果说明](slides/Co-optimize/continuous_legs_results_20261013.md)。

- [Co-optimize 代码导航](slides/Co-optimize/README.md)
- [三个 operations 与演示](slides/Co-optimize/operation_demo/README.md)
- [需求表达式](objects/r6_quadratic.md)与[XYZ／airborne 物理公式](slides/obj_supp/airborne_equations.md)
- [物体数据预处理](codes/precompute_objects/README.md)
- [pose_set_search 对照测试](slides/pose_set_search/README.md)
- [参数与坐标约定](slides/params/README.md)
