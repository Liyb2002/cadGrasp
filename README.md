# cadGrasp

2026-10-13最新Step5.2整块圆整删减已执行全部52组的Step5.2／5.3，均通过原始全需求及实际实体检查。保留更多原Step4材料，改善逐格删减留下的细连接和缺口；整体BBX逐组保持上一版结果：删减351.890791L、成品432.466717L合计。Step4、Step5.1、263个object pose及复用123／Juxtapose140／支撑安装123保持原样。51组采用6 mm圆整范围，1组保守保留整块裁边主体。[算法](slides/Co-optimize/aesthetic_pruning_algorithm.md) · [全部52组BBX、pose数量及同视角前后图](slides/Co-optimize/aesthetic_pruning_results_20261013.md)。每组`step5.2`／`step5.3`图已更新，无HTML。下方保留原连通布局搜索及历史实验说明。

共同设计一件刚性支撑，在多个最终摆放 state 下支持多个 object pose，满足任务的力／力矩需求，并尽量减少实体材料。每次装入一个物体、执行任务、沿退出方向取出，再由机器人抓起空支撑换到下一 state。一个 state 可以服务多个 pose；空支撑搬运、转动途中的姿态不参与设计约束。

当前主线采用 **whole 联合优化**：所有 pose 从初始化开始共同参与，每个操作同时计算对其他 pose 的收益和损失。工件原任务朝向不变，允许世界 XYZ Translation 和 airborne。airborne 使用同一套绕质心的需求，只移除工件自身地面反力；最低点仍不得穿地。

## 流程与坐标

最新实现按[连通主体与 BBX 优先删减](slides/Co-optimize/connected_core_algorithm.md)运行：Step4 在原有限包覆范围内保留一个连通材料主体，删除的孤块不参与受力；真实剩余接触必须通过全部原始需求及 Oracle。Step5.2 只删，先尝试缩小 BBX，再删其它冗余材料；Step5.3 保留长脚。新流程全部52组／263个object pose已通过Step4与完整Step5，复用123／Juxtapose140，123个支撑安装pose。最终Evaluation的BBX合计为无支撑254.408 → Step4主体371.058 → 删减351.891 → 成品432.467L；成品较Step4净增加16.55%。[全部组BBX、pose数量与图片](slides/Co-optimize/data/results/B/evaluation_connected_core_postprocess_all52_20261013_final_bbx.md) · [执行与审计](slides/Co-optimize/connected_core_results_20261013.md)。以下V8及其后处理数值属于此前独立对照。

共同支撑坐标中的材料点记为 $q$，第 $i$ 个任务的最终支撑变换为 $T_i=(R_i,t_i)$，物体相对支撑变换为 $Q_i$，物体世界变换为 $U_i=T_iQ_i$。退出方向 $d_i$ 在支撑坐标中表达，满足 $\|d_i\|=1$；世界退出方向是 $R_id_i$。

| 阶段 | 作用 |
| --- | --- |
| Step3.1 | 所有 pose 注册到共同参考，生成贴合包裹，扣除全部完整工作禁区 |
| Step3.2 | 单独展示工作禁区的环绕等轴测图 |
| Step4.1 | 初始化共同／相近的合法退出方向，扣除完整装卸路径 |
| Step4.2 | Direction、XYZ Translation、离散 Juxtapose 联合优化，动态 base Oracle 审查，保持可行后减材料 |
| Step5.1 | 冻结最终布局，计算每个 pose 全部系统—地面需求点与凸包 |
| Step5.2 | 从连通主体开始，只删材料，优先缩小 BBX，并保持连通及全部原始需求 |
| Step5.3 | 保留已连通主体，在合法空间内长出支撑脚，完成实际触地与受力验收 |
| Evaluation | 比较无支撑 pose 叠放、真实 Step4／Step5.3 布局的静态 BBX，另列 state 分组诊断 |

## Step4：联合布局优化与动态 base Oracle

最新V8布局的后处理已完成Step5.1／5.2／5.3与Evaluation：最终49/52组实际实体通过（50组进入后处理，1组真实接触复查失败，2组原Step4失败）。最终238个object pose、129个支撑安装pose，复用129／Juxtapose109。材料相对Step4名义模型+58.64%，静态BBX+27.52%；相对搜索内已构造参考，材料+6.59%，BBX+0.24%。[逐组数量、材料／BBX增量与PNG](slides/Co-optimize/data/results/B/evaluation_dynamic_postprocess_all52_20261012_01.md) · [执行与失败说明](slides/Co-optimize/dynamic_postprocess_results_20261012.md)。原Step4搜索统计及历史成品保留其各自证据身份。

2026-10-12最新实验采用[动态换座的32槽位算法V8](slides/Co-optimize/population_soft_algorithm.md)：32个注册原位候选，初始化不指定谁Juxtapose；每轮各自做一次离散或差分变异，再一次调整。换座从世界XY质心对齐开始，优化中恢复旧方向、translation及高度修复能力；N/3只控制抽样倾向。入口为[run_population_soft.py](slides/Co-optimize/step4.2/run_population_soft.py)。全部52组／263个object pose已完成最多8轮冷启动，**50组完整通过、2组原受力失败**，无运行错误。通过组为244个object pose、131个实际支撑安装pose。[逐组Evaluation与469张PNG](slides/Co-optimize/data/results/B/step4_population_soft_all52_20261012_final.md) · [调整与结果说明](slides/Co-optimize/population_soft_results_20261012.md)。上一轮固定32候选冷启动33/52与此前已接受的暖启动52/52分别保留。以下物理公式与验收沿用；frontier搜索描述为保留的旧入口对照，最新入口使用固定32槽位。

### Step4.1：退出初始化

从 Step3.1 的整组注册布局出发，寻找共同或相近的合法退出方向，切除各 pose 的完整退出 sweep。保留原任务朝向与原始载荷。此阶段是搜索起点，不能把初始化构造成功理解为全部任务受力通过。

### Step4.2：连续下降、离散跳步与减材料

连续损失衡量**全部需求到可用反力锥的距离**，不只针对最远的一个需求。工作面上的施力位置为 $r=a_f+u\,edge_{u,f}+v\,edge_{v,f}$，加工力为 $P$，重力为 $g$，绕物体质心的需求为

$$b=(-g-P,-r\times P)=B_fz,\qquad z=(1,P,uP,vP).$$

$B_f$ 直接由物体几何与物理参数构造，不由 32,768 个样本拟合。每个 pose／角度的表达式保存在 [objects/](objects/r6_quadratic.md)。设布局变量为 $\theta$，$A_i(\theta)$ 为原七维平衡模型中的可用反力列，$\widetilde b_i$ 为按原尺度归一化、附加第七维零的需求，则

$$\mathcal L(\theta)=\frac1n\sum_i\frac1{s_i^2}\int\frac12\min_{\lambda\ge0}
\|A_i(\theta)\lambda-\widetilde b_i(\xi)\|^2\,d\mu_i(\xi).$$

反力系数非负、没有上界；$s_i$ 为优化前固定的需求尺度。$\mu_i$ 使用原工作面面积、施力角度／强度与可见性定义，不是 R6 的六维体积。实现用固定正权重求积；局部下降及其竞争分支共用节点、权重和尺度。需求公式不能消除不同接触基底的切换，因此不是每个 pose 只算一次公式就完成全部投影。

有效投影基底 $A_I$ 内，令 $\Pi_I=I-A_IA_I^+$，则

$$\ell_{f,I}(z)=\tfrac12z^T\widetilde B_f^T\Pi_I\widetilde B_fz.$$

基底满足原／对偶条件时复用；删除基底接触或新增反力列违反条件时才重新求投影。退出方向和平移不改变绕质心需求，改变的是接触是否可用。当前硬接触网格在事件间有平台，因此**操作梯度仍采用跨接触事件的数值割线**，加速的是割线两端的全组损失评估。

搜索默认持续保留 **4 个完整布局**，每个布局包含全组退出方向、物体座位、host 和支撑安装 XYZ。每轮把父布局和新候选一起比较，同时保留受力误差较小的、合法地脚包围较好的，以及地脚极点／布局几何不同的分支。方向改善受力却切掉关键 base 空间时，另一个可造分支可以继续搜索；受力暂时稍差不等于立即删除。已经取得全部原始需求与共同连接通过的布局优先保留。

同一轮的全部父布局、梯度探针、sampling 和 Juxtapose 候选共用固定求积点、权重和尺度。96个正式候选的快筛预算由各布局分摊，原导数探针另计；每轮最多8个短名单候选执行全部原始需求分类，最多3个新受力／包围通过的候选查共同连接。快筛同时看受力损失和Oracle包围差距，地脚位置的差异用于保留多样性，不能拿合法区域面积代替承载能力。有限窗口未找到的布局仍可探索，不能发布为整体PASS。

每轮对保留布局生成以下操作：

1. **Direction**：在单位球面的两个切方向估计全组损失梯度，比较多个角步幅；同时尝试共同转向，计入新释放与新锁定接触。
2. **Translation**：同时允许物体座位 $Q_i$ 和支撑最终摆放 $T_j$ 的世界 XYZ 平移，比较多个位移步幅；三轴共用差分规则、范数和步幅池。移动 $T_j$ 时，该 state 的所有任务一起移动，原任务旋转保持不变。地面边界用可行单边探针并投影，保持物体最低点高度非负；Oracle 按实际新摆放更新地面与需求凸包。不同 host 的物体座位共同平移先分别转换为支撑坐标。
3. **Juxtapose**：连续停滞时及周期性的分支扩展中，离散尝试新的 host／重叠关系、间距与共同位置，保留后在下一轮继续 Direction／Translation 精修。允许少量明确记录的事件采样，不限制一个 state 只能容纳两个 pose。
4. 候选使用接触／材料增删缓存，避免每个小探针都重建实体 Boolean。释放接触只有在不再被任何其他 pose 锁住时才计入收益；新增阻挡同时计入损失。
5. 受力可行后，继续恢复转动复用和减少实体材料；接受的可行改动必须保持全部原始需求与 base Oracle 通过。比较实体材料体积，不把 state 数或 BBX 当作材料成本。

Juxtapose 的全需求覆盖损失小于等于`1e-28`时，候选排序将它作为数值零，避免微小浮点残差优先选择更大的布局；等价候选先试静态物体／地面需求凸包范围较小的布局。该便宜排序量不用实体Boolean，也不替代最终全需求与Oracle门槛；最终减材料目标仍是实体材料体积。

实现为[layout_frontier.py](slides/Co-optimize/step4.2/layout_frontier.py)。`--frontier-width`默认4；设为1可运行原单布局搜索对照。Oracle默认`--oracle-work-mode normal`，历史角度模式可显式设为`angular`。旧保存批次的结果和来源不因新搜索策略改变。

每个 pose 的 **32,768 个原始需求全部通过**即工件受力 PASS。最终保存复用已经得到的 mask／反力，不重新求解全部需求，也不进行末尾几何微扰补救。

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

只有少数受力及包围通过的候选进一步查询共同合法连通分量：同一分量必须容纳全部承载接触的外侧锚点，并在每个安装地面提供足够接点来包围 $H_i$。保存 `oracle_connected_witness.obj` 作为可连接空间见证，供 Step5 裁剪；它不是成品 base，也不计作支撑材料。

整体 PASS = **工件全部需求通过 + 合法地脚包围通过 + 共同连接空间通过**。未通过的中间布局可以继续搜索；已通过布局的后续减材料不能丢失 Oracle 可行性。有限查询未找到不等于所有布局无解。详见 [Oracle 公式](slides/Co-optimize/step4.2/base_oracle.md)。

## Step5：固定布局的确定性实体构造

### Step5.1：全部地面需求与凸包

冻结 Step4 的 $T_i,Q_i,d_i$，读取原始需求，按最终质心计算上面的 $\tau_O,p_i$，得到每个 pose 全部 32,768 个压力点与凸包。XYZ 平移特别是 Z 的力臂已进入计算；airborne 不需要另一套工件需求。

计算只有数组叉乘、除法和凸包，不做方向／平移搜索、需求重采样或工件受力求解。保存逐 pose 点云、实际变换，以及共同支撑视图。默认支撑自重为零，与原需求模型一致。

### Step5.2：删减接触材料与多余独立块

冻结布局，先尝试删除多余独立块，再按固定空间分区删减接触材料。每次从剩余真实接触三角形重建原七维反力锥，检查每pose全部32,768个原始需求。已经连通的主体不切成受力必需的孤岛；多余孤岛与切除区域一起删除。必要输入独立块如无法删除则明确记录，由下一步连接。当前为有预算的贪心删减，不宣称全局最少材料。

输出`pruned_support.obj`与新的原需求检查记录；原布局、载荷与工作/退出禁区保持不变。历史数值批次继续保留原Step5.2构造身份。

### Step5.3：选地脚并连成实体

使用Step4已保存的合法连通分量$W$与新Step5.2通过的删减支撑$N$，按固定顺序构造：

1. 保留 $N\cap W$ 的全部合法原材料；裁去原名义模型中已经处于禁区／地下的部分，并单独记录删除量。
2. 在局部范围内取各地面上方的 5 mm 合法薄层，并按该state在任务地面XY轴上的需求凸包限定选脚范围，初始向外余量5mm；不足时才按固定倍数扩大。由合法接地顶点选择少量地脚，逆向删除冗余点，只保留仍能包围所有需求凸包的点位。
3. 在保留点位生成小地脚并裁进 $W$；5 mm 是构造模板尺寸，裁剪后允许更细的部位，没有额外安全余量或最小宽度要求。不填充虚拟凸包内部。若小地脚构造不足，使用合法薄层兜底。
4. 依组件最近点尝试短连接体，受阻时按固定序列扩大局部连接盒；所有连接都裁进 $W$，只加入真正减少组件数的连接，优先选择当前宽度层中新增材料较少的结果。
5. 导出一件连通、封闭的实体 `support_with_base.obj`。仅删除体积低于 $10^{-13}\,\mathrm{m}^3$ 的数值碎片，OBJ 使用 16 位小数，不移动物体或顶点。

新地脚范围随各最终地面的实际需求凸包定义，不直接取共同坐标大盒的外侧角点；只在不能包围时扩大。连接仍按新增材料比较，没有宣称全局最小占地。图中的彩色环是需求凸包轮廓，不是实体底环；法向压力模型要求接地点的凸包包围需求，允许敞空。

若名义材料已处于合法空间，直接保留，避免沿重合边界重复Boolean裁切产生数值碎片；保留检查使用缓存的实际起始合法实体，原有空间包含与材料保留阈值不变。

该阶段不重新构造退出／工作障碍，不调用 Direction／Translation，也不重跑工件需求。完成的是当前模型的**法向地面压力包围与实体连通**；完整地面摩擦、yaw 平衡和结构强度不是当前算法已经求解的指标。

## Evaluation：静态工作空间

评价静态工作空间，不加入装卸行程。记有支撑时的实际物体为 $O_i=U_iO$，名义支撑为 $N$，完整支撑为 $S$，固定世界 XYZ 包围盒体积为

$$V(X)=\prod_{a\in\{x,y,z\}}\left(\max_{p\in X}p_a-\min_{p\in X}p_a\right).$$

**无支撑叠放基线**：任务依次执行，各 pose 可使用同一空间。保留原任务朝向，把每个物体的水平包围盒中心对齐、最低点放在 $z=0$，得到 $\bar U_iO$。不沿用支撑优化后的平移位置与高度。若第 $i$ 个朝向的物体盒长为 $d_{i,a}$，则

$$V_{\rm no\ support}=V\!\left(\bigcup_i\bar U_iO\right)=\prod_{a\in\{x,y,z\}}\max_i d_{i,a}.$$

这达到固定朝向、自由平移的 XYZ 盒下界，不把各 pose 的盒子体积相加。重叠表示先后执行任务，不是把多个物体同时装入。

**有支撑的整体盒**：保留保存的真实物体布局，对 $\bigcup_iO_i\cup\bigcup_sT_sN$、$\bigcup_iO_i\cup\bigcup_sT_sS$ 分别取一个共同世界 XYZ 盒。图从左到右为无支撑叠放、加 Step4、加 Step5.2；同等轴测方向与长度比例，各列分别居中。不改支撑摆放，也不重跑物理优化。

**共享 state 空间诊断**：按相同实际支撑变换 $T_s$ 分组，组内 pose 集合为 $I_s$，$O_s=\bigcup_{i\in I_s}O_i$。若给不同 state 单独分配空间，可另列

$$V_{\rm state,object}=\sum_sV(O_s),\qquad
V_{\rm state,4}=\sum_sV(O_s\cup T_sN),\qquad
V_{\rm state,5}=\sum_sV(O_s\cup T_sS).$$

任务依次执行。一个支撑 state 服务多个 object pose 不表示多个物体同时装入；盒子表示这一空间需要容纳该 state 的全部任务。

该统计及最终落座物体整体盒只作诊断，不作为无支撑基线。整体盒与多个独立空间的体积之和是不同指标；旋转状态可由机器人搬运，也可使用同一工作站，state 空间统计不表示必须有多个工位。

逐 pose 保存静态尺寸，逐 state 保存成员与盒子，另列实体材料 cm³。Juxtapose pose 指最终 host 已改变或相对位置 $Q_i\ne I$；保持注册关系的转动复用指仍保持自己的 host 且 $Q_i=I$，允许该 state 的安装位置平移。这两类合计为 pose 数。一个 state 支持几个 object pose 是另一种共享统计，可同时包含 Juxtapose pose。

Evaluation表显式列出支撑pose数、每个支撑pose服务的物体pose数与成员。支撑pose由保存的实际世界位置和朝向去重；同一件支撑可有多个安装pose。数量之和等于物体pose总数，JSON中为`support_pose_count`、`object_pose_counts_per_support_pose`与`fixture_states`。

## 验证与保存结果

最新全目录运行覆盖 `Co-optimize/output/B` 的全部52个2–10姿态集合（含10个`illegal/`集合）。Step4采用8个sample、8轮搜索和初始化预算32，已有布局的暖启动来源明确保存；52/52完成完整实体和Evaluation。全部36个4–8姿态集合满足2–4原位复用＋2–4 Juxtapose；其他规模不设置该数量目标。最终发布为`material_output_all52_20261012_final`，复用本轮`material_output_all52_20261012_01`的布局、需求和51件成品，只在同一保存合法空间内重构`pose1+2+4+19`，修正构造后残留的外部反向边界，不改变布局、物理门槛或原需求。

更新1150张阶段图及一张52件支撑总览；263个姿态实例的8,617,984条保存需求全部通过，52件导出模型连通、封闭，1332个原始输入／既有布局／证明视频的指纹未变。每组当前成品与Evaluation图片直接放在 `slides/Co-optimize/output/B/[set]/step5.2/`，不使用 HTML，共419张 PNG；`illegal/` 集合保留前缀。263个 object pose 对应112个支撑安装pose，每组是一件共享支撑，任务依次执行。独立集合三列BBX统计和为254.41／389.14／433.19L，完整材料统计和12,527.10cm³；这是暖启动批次，不是冷启动性能对照。[直接看单组姿态图](slides/Co-optimize/output/B/pose1+2+4+19/step5.2/all_poses.png) · [52件支撑总览](slides/Co-optimize/output/B/all_supports_material_output_all52_20261012_final.png) · [全部52组Evaluation与pose数量](slides/Co-optimize/data/results/B/evaluation_material_output_all52_20261012_final.md) · [CSV](slides/Co-optimize/data/results/B/evaluation_material_output_all52_20261012_final.csv) · [直接图片来源记录](slides/Co-optimize/data/results/B/data/system_material_output_all52_20261012_final/direct_images.json)。

4布局搜索的新增10组实验为`normal_frontier_more10_20261011_01`：Step4通过6/10，实体构造完成5/6，最终Evaluation完成5/10；其余4组搜索未通过、1组构造未通过。同条件旧单布局算法在这10组中完成7件；共同完成的5组中，完整BBX增加4组、不变1组，没有减少的组。Juxtapose数3组不变、2组增加，没有完成的减少案例，因此这批尚未检验到减少Juxtapose后的体积收益。两个抬升高度约0.93m／1.86m的解分别占用1440.35L／8044.73L，体积是BBX，材料分别为734.79cm³／660.81cm³。[新增10组Evaluation与旧版对照](slides/Co-optimize/data/results/B/normal_frontier_more10_review_20261011_01.md)。此次搜索参数不变，旧模型只作只读Evaluation，原始输入和历史结果486个文件的指纹保持。

安装XYZ／零余量版通过76项相关测试，并以新实验 `movable_fixture_zero_margin_smoke_v2` 跑通 `pose2+20`、`pose1+2+4+7`、`pose1+2+3+4+5+6+7+8+9+10`：3/3从Step4通过全部原始需求与Oracle，完成Step5.2实体及Evaluation。十pose布局含非零安装位移，Step5使用实际安装变换计算需求凸包。完整静态BBX依次为4.59L、8.72L、35.67L。退出凸包筛选与全部可用接触同分量要求保持；未重跑全部52组。[试运行静态评估](slides/Co-optimize/data/results/B/evaluation_movable_fixture_zero_margin_smoke_v2.md)。

上一批完成链为 `base_oracle_compact_clearance_v1` → Step5.1／Step5.2／Evaluation `clearance_compact_v1`，**52/52** 集合全部完成，包括原 `illegal/` 中10组及全部10个≥7-pose集合。263个pose实例，8,617,984个原始需求均通过；该批仍使用5mm接触余量和固定支撑安装位置，早于本次安装XYZ／零余量改动，保留其原始来源。物体允许airborne，地脚仍落在z=0。50组一个支撑state，2组两个state。成品均为连通、封闭实体，保存布局不变，没有重复工件受力验收。

| 统计 | 当前结果 |
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

`slides/Co-optimize/output/B/{pose_set}/` **只保留各步骤最终可视化**及 HTML 浏览页：Step3.1／3.2、Step4.1／4.2、Step5.1／5.2 与 Evaluation。每步仍使用原目录结构，原 `illegal/` 集合也保留路径。

对应的最终布局、force mask、需求点云、实体模型、Oracle 空间见证、计算报告与批次统计保存在 `slides/Co-optimize/data/results/B/{pose_set}/` 或其 `data/` 子目录。这些是后续步骤与复用需要的最终数据。旧报告中的原输出路径由兼容读取器解析，不重新计算输入。

按用户要求，`pose2+20/step4/step4.2/oracle/`例外保留视频及其工作锥OBJ／无界方程NPZ／说明JSON，方便直接查看每个任务的约束；它们是已有布局的诊断输出，不覆盖原受力与实体结果。

候选布局、process／mesh 帧、工作日志、进度文件、临时执行源码副本在最后一个消费者完成后删除；保留最终结果及哈希记录。历史批次、失败中间产物和重复图已清理，物理模型、优化参数和操作调用顺序不变。初始化需要的旧源码仅保留按哈希去重的原始版本，避免 I/O 调整破坏原初始化的字节校验。

## 使用与相关目录

在仓库根目录启动一个新 Step4 实验：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/run.py B --sets all --jobs 3 --output-name my_base_oracle
```

从当前保存的 Step4 选择，运行新的 Step5 与评估批次：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step5/run.py B --experiment my_final_base --jobs 3
```

当前保存选择在`data/results/B/data/base_oracle_compact_clearance_v1_results/summary.json`，完成链记录在`data/results/B/data/system_clearance_compact_v1/pipeline.json`。入口支持`--selection-summary`；已完成的续算构造清单可用`--step52-batch step52_clearance_compact_v1_completed`复用。运行新批次使用新名字；数值存入`data/results/`，最终图自动发布到`output/`。

- [Co-optimize 代码导航](slides/Co-optimize/README.md)
- [三个 operations 与演示](slides/Co-optimize/operation_demo/README.md)
- [需求表达式](objects/r6_quadratic.md)与[XYZ／airborne 物理公式](slides/obj_supp/airborne_equations.md)
- [物体数据预处理](codes/precompute_objects/README.md)
- [pose_set_search 对照测试](slides/pose_set_search/README.md)
- [参数与坐标约定](slides/params/README.md)
