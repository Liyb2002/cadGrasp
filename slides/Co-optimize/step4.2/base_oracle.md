# Step4 的动态 base Oracle

入口 `run.py` → `run_oracle.py`。Direction / XYZ Translation 的全需求梯度和原始需求保持不变；Juxtapose 仍为离散跳步。新的候选审查同时查询地脚包围与共同连通空间。旧的仅受力通过结果不能绕过 Oracle 成为新实验的通过结果。

最新工作禁区仅修改Oracle：默认`work_mode='normal'`，以[normal_work_access.py](normal_work_access.py)的$W_i^O=\bigcup_{p\in A_i}\{p+\lambda n(p):\lambda\ge0\}$替换角度锥。每个原工作三角形沿自己的外法向形成无界棱柱；取并集，不整体取凸包、不加余量。快速二维地脚查询与共同连接查询使用同一法向几何。原初始化、接触／材料指导、30°加工力需求、退出路径及地面约束全部不变。旧角度模式可显式选择`work_mode='angular'`；下文自身遮挡角度区间说明仅属于该旧模式。

## 合法空间

当前实现为 [pose_oracle.py](pose_oracle.py) 的 `PoseOracleConstraints`，由 `BaseOracle` 的快速包围和共同连接查询使用。各工作锥保留任务来源，不把不同任务的世界工作锥直接叠加到当前任务。

在共同支撑坐标系中，安装姿态为 $T_i=(R_i,t_i^0+\delta t_i)$，材料点为 $q$。安装位移 $\delta t_i$ 在世界 XYZ 优化，$R_i$ 保持原任务旋转；同一 state 的全部任务共享安装位移。每个最终工作姿态要求

$$h_i(q)=(R_iq+t_i)_z\ge0.$$

所有这些半空间取交，形成可能有斜面、也可能无界的多面体，并不一定是封闭盒子。机器人取出物体、抓起空支撑后换姿态；中间转动不生成额外半空间或几何锁。

**物理地面是世界 $z=0$ 的零厚度平面**，合法材料必须在其上方 $z\ge0$。地面边界和被它排除的半空间是两件事：视频只画长方形薄片，不再把 $z<0$ 的空间画成有厚度的粉色实体。只禁止支撑的平移向−Z仍不够，旋转也可能把材料点转到地面下面，必须检查所有最终安装姿态。

Step4 的**物体支撑接触区**没有几何安全余量：表面点 $p$ 只有在所有最终 state 都满足 $(T_jQ_ip)_z\ge0$ 时才提供反力列。物体与地脚均可接地。筛选在 `OracleModel.contact_allowed` 内接入，接触增删缓存、全组投影损失与原始需求检查使用同一筛选。名义实体导出裁去所有最终地面下的材料，浮点容差不代表预留间隙。

注册 pose 允许直接 XYZ Translation；支撑安装位置也有世界 XYZ 梯度与同一轴向、多个步幅的有限候选。跨 state 地面导致 Oracle 不通过时，先尝试移动现有安装位置，再进入原 Juxtapose 跳步搜索。平移支撑会带动该 state 的所有物体任务；抬高丢失工件自身地面反力的代价仍进入原全组损失，不额外设置 +Z 奖励。

每个物体坐标系里的地面方程由最终世界变换 $U_i=T_iQ_i$ 重新计算：$(U_ip)_z\ge0$，包含该物体的最终 Z。转换到共同材料点 $q=Q_ip$ 后，它成为 $(U_iQ_i^{-1}q)_z=(T_iq)_z\ge0$。物体座位 Translation 改 $Q_i$，安装 Translation 改 $T_j$；前者不移动共同地面，后者更新对所有材料点的地面限制。所有任务的最终地面都检查，不只检查当前任务。最终文件保存实际 `native_world`、原任务 `native_world_original` 和世界安装位移 `state_offsets`，Step5 使用相同实际变换；旧文件默认安装位移为零。

`PoseOracleConstraints`同时保存`floor_planes_object`和`floor_planes_common`。不能把物体座位的 Z 又加到共同地面偏置上，那会把同一次位移算两遍。当前pose2+20例子有两个最终支撑state，共同材料坐标中的地面边界不重合。

工作锥是 **pose specific**：第 $i$ 个任务只检查自己的世界工作锥 $W_i$，要求 $T_iS\cap W_i=\varnothing$。同一件刚性支撑必须依次通过全部任务，所以共同材料坐标里的条件为

$$S\cap\bigcup_iT_i^{-1}W_i=\varnothing.$$

物体相对支撑变换为 $Q_i$，原物体坐标工作锥为 $W_i^O$，因此 $W_i=U_iW_i^O$，$T_i^{-1}W_i=Q_iW_i^O$。每个任务使用自己的工作面和角度；不是把 $W_j$ 不经变换放在第 $i$ 个任务的世界里。物体内部与退出路径也按各自的任务变换约束同一件材料。

### 历史角度模式：朝外半射线与自身遮挡

原物体上的工作面为 $A_i$，局部外法向为 $n(p)$，半角为 $\alpha_i$。理想的可达工作禁区为

$$W_i^O=\{p+\lambda v:p\in A_i,\ \lambda\ge0,\ \|v\|=1,\ v\cdot n(p)\ge\cos\alpha_i,\ \operatorname{reachable}(p,v)\}.$$

`reachable`沿用`objects/`：从$p+10^{-5}n(p)$发出朝外射线，不与物体自身相交。$\lambda$不能取负；这不是双向无限直线。工作面倾斜时，合法的朝外30°锥可以向物体侧面延伸，不能仅凭世界Z低或位于侧边判错。

原`WorkAccess`已经是朝外半射线，但保留所有自身遮挡的方向。Oracle现在使用[work_access_shadow.py](../helper_func/work_access_shadow.py)，保留原32边角度外包与连续工作三角形，仅删掉**能证明对整个源三角形都被自身遮挡的角度区间**：三角形三个顶点沿同一方向的正射线都撞上同一个凸的平面物体面片，则交点随源点线性插值，整个三角形都不可达。平面凸面片合并不改变物体表面；不确定的遮挡、数值划分失败仍保留原禁区。因此这是理想可达禁区的保守外包，不宣称删尽全部自身遮挡。预计算与布局位置无关，按物体／工作面／角度缓存，查询时仍为半空间和多边形运算。

原初始化和受力/contact kernel继续使用原完整锥作为保守约束，冻结的52组物理结果没有重新求解；自身遮挡裁剪用于base Oracle和诊断输出。

`classify_world(i, points)` 分别返回：`global_floor`（这一材料点在任意最终安装姿态下穿地）、`current_pose_work`（仅任务i的世界工作锥）、`shared_material_work`（同一材料点在任一任务中挡住该任务自己的工作锥）。各pose转圈视频显示地面约束的零厚度边界及当前工作锥；第三项仍用于固定共享材料的设计。其他任务的地面拉回当前材料摆放后可能是斜片，这表示另一最终安装姿态的约束边界，并非现场还存在一个斜地面。

第 $i$ 个姿态的潜在地脚位置，是合法材料空间与 $h_i(q)=0$ 的交。点必须在当前姿态接地，在其他最终姿态中允许悬空。

## 快速包围查询

1. 原始32768 COM需求只读加载；按当前物体世界质心映射到地面。$p=(-\tau_{O,y}/F_z,\tau_{O,x}/F_z)$，$\tau_O=\tau_C+c\times F$。平移时 $\Delta p_{xy}=\Delta c_{xy}-\Delta c_zF_{xy}/F_z$，所以需求凸包必须随布局更新。
2. 先检查不受尺寸限制的跨姿态地面半空间分离证书；加入物体自身实际地面接触，不能把这些接触错误地套用到别的支撑姿态。
3. 在有限候选范围内剪出二维合法地脚区域；当前工作禁区使用每个三角形法向棱柱的4个精确半空间，历史角度模式使用上述经可证明自身遮挡裁剪的32边外包模型。快速exit查询仍使用包含真实 sweep 的凸包，并将退出长度延伸到整个候选范围之外，因此保守。相同host的地面只计算一次。
4. 使用未膨胀的地面与障碍边界；`foot_width=0`，不为5mm脚块预留空间，也不要求最小通道宽度。取合法区域边界极点与实际物体接地点的凸包 $C_i$，检查需求凸包 $H_i\subseteq C_i$。只检查已有需求凸包顶点；不在需求边界上限定脚的位置，也不要求实体闭合包围。Step5 的有限尺寸模板会裁进合法空间，允许更细的实体部位；Oracle 不把正宽度通道因厚度余量而排除。

缓存包含物体座位、方向、host、支撑安装位移和查询层次；需求凸包另外按真实世界位置缓存。安装位置变化不会复用旧地面或旧需求凸包，接触与名义材料也重新应用实际地面裁剪。

## 共同连接查询

仅对少数受力通过且包围通过的finalist，构造共同候选空间：剪所有最终地面，减完整工作锥与原始union sweep（退出长度覆盖整个候选范围）。逐个连通分量检查所有姿态的实际地面接点凸包，并检查当前搜索承载接触的外侧锚点是否都属于同一个分量。

通过返回共同连通空间的几何见证；它说明有空间可连接，**不是优化后的base，也不是强度或完整地面摩擦／扭转受力验收**。其体积不作为最终base材料体积。Step4已有工件反力锥的全部原始需求检查不重复执行。

## 候选搜索

默认搜索保留4个完整布局。每个候选在全组受力快筛时同时查合法地脚包围，父布局参与下一轮筛选；保留受力／包围之间的取舍，以及不同地脚极点和布局几何。正式候选快筛的96个额度由全部父布局共享，原数值导数探针另计。同一轮全部竞争分支固定需求测度，不能让某个分支自行选择对它有利的求积点。未找到的中间布局可以继续探索，不能发布为整体PASS。

局部sampling、周期性Juxtapose与Oracle修复共同扩展分支。Oracle修复仍尝试多pose共同host、不同重叠间距、排列和共同XYZ位置，实际候选逐个查询，不假设高度可行性全局单调。最多8个短名单候选执行全部原始受力需求分类；只有受力与包围均通过的少量候选进一步查询共同连接。已取得共同连接的可行布局和对应空间见证必须保留，不能被受力误差更低但无法造base的改动挤掉。

分支仍用原Direction/Translation helper与全组需求积分；未可行时同时保留Oracle指标与几何多样性。已通过Oracle后的复用恢复和材料下降均受Oracle约束；若新布局不能保持共同连接，保留上一个通过布局。成功条件是：全部原始工件需求通过、合法地脚包围通过、共同连接查询通过。`--frontier-width 1`保留旧单布局算法，包括原高度细分，用于对照。

覆盖损失不超过`1e-28`时只把它当作候选排序的数值零，避免浮点残差让已经覆盖的候选优先选择更大布局。等价候选按实际物体与地面需求凸包的静态布局包围范围排序，然后仍执行全部原始需求和共同连接门槛；排序阈值不能代替受力通过。该范围只是不用Boolean的便宜排序量，最终报告仍区分实体材料体积与完整支撑的真实BBX。

`certified_floor_incompatible` 是当前固定布局的无界分离证书。`not_found_in_window` / `no_common_component_in_window` 是有限范围或保守几何下未找到，不宣称所有布局物理无解。

全部B集合（包括原`illegal/`下保存的10组）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python slides/Co-optimize/step4.2/run.py B --sets all --jobs 3 --output-name my_base_oracle
```

最终数值与空间见证保存在`data/results/B/{pose_set}/step4/step4.2/{experiment}/`，最终图片在同层级`output/B/`。临时候选与帧在出图后删除，执行副本在校验后删除。

当前按任务拆分的两个视频与工作锥文件在 [pose2+20/step4/step4.2/oracle](../output/B/pose2+20/step4/step4.2/oracle/index.html)。灰色是当前物体，淡粉色是零厚度地面约束边界，淡黄色是当前任务自己的工作锥，金黄色工作面属于该任务。镜头以35.264°等轴测仰角绕一圈，视频无文字、无其他任务工作锥、无退出路径。`work_cones/{pose}.npz`保存物体／共同支撑／任务世界坐标中的无界半空间方程、工作面、变换和全部最终地面；OBJ仅是工作锥的球形截取展示片段，地面另用零厚度长方形边界片。展示截取不改变Oracle判定。

诊断schema为`pose_specific_oracle_v2`：JSON保存单向射线、自身遮挡裁剪策略及逐工作面比例；NPZ额外保存所有物体坐标里的最终地面方程，供核对Z和坐标变换。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/render_pose_oracle.py --set pose2+20 --seconds 8 --replace
```

未来Step4搜索保存最终结果时，同时导出该最终布局的按任务分开的工作锥文件，不生成候选帧、不重跑工件需求。已有最终布局、force mask和Step5实体保持原样。

### 把另一最终pose的地面约束带到当前pose

[pose2→pose20的地面视频](../seperate_base_support_proof/videos/05_pose2_to_pose20_oracle.mp4)使用已完成的两state布局、尚未加base的Step4.2支撑与保存的工作锥。灰色物体与蓝色支撑先在pose2工作，再一起抬起、转动并放到pose20。pose2的粉色薄片是地面约束，按$T(t)T_2^{-1}$随支撑搬运；到pose20时，其边界方程为$\bigl(T_2T_{20}^{-1}x\bigr)_z=0$，同时显示pose20自己的水平$z=0$地面。因此最后有两个地面约束与一个当前pose的工作锥。

视频的蓝色部分取Step4.2名义壳与已经保存的Oracle合法分量的交集，和Step5加base之前保留的支撑一致。原始名义壳会因约5mm外扩而进入地面下方，不能直接作为合法支撑展示；本例去除5.010964cm³非法名义材料，不新增base，也不修改原布局、支撑文件或受力结果。

[视频06：pose20地面包围机会](../seperate_base_support_proof/videos/06_pose20_ground_enclosure.mp4)接续视频05的最终pose20场景：紫色为全部32,768个原始地面压力需求的凸包，随后镜头转为地面俯视，绿色为显示范围内的合法地脚区域，粉色为非法区域。三个绿色标记均来自已保存的共同合法连通空间，它们的虚拟凸包以绿色虚线显示并包围需求凸包；标记和虚线不是实际脚或实体底环。地面切片同时计入全部任务拉回当前材料摆放后的法向工作禁区、完整退出路径及最终地面限制。复用视频05同一布局、原需求与连接见证，只派生展示几何，不运行受力、布局搜索或Oracle查询。[渲染与几何记录](../seperate_base_support_proof/videos/06_pose20_ground_enclosure.json)。

被搬运的是约束的坐标表示，物理地面始终水平。旧工作锥不会被带到新任务中；只显示当前pose的工作锥。搬运过程仅作示意，不增加中间姿态的合法性要求。`oracle/`保留约束NPZ/OBJ及图例记录，视频统一放在`seperate_base_support_proof/videos/`。

出图读取源实验保存的Oracle判定，不重新求解受力或重建连通见证。诊断报告明确标记`query_origin=saved_final_experiment`和`base_oracle_query_run=false`；若旧的独立有限窗口重建未找到分量，保留简短诊断记录，不把它当成源实体的新验收或改写源结果。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  slides/Co-optimize/step4.2/render_floor_transfer.py --seconds 12
```

保存的`base_oracle_v3`在新增5mm支撑接触地面余量之前完成：52/52通过，包含全部10个≥7-pose集合。263个pose实例的8,617,984原始需求全部通过，名义材料23,456.73cm³；见证不计入材料。这不是新余量条件的验收记录。实际base由Step5.2构造，完整系统结果与指标见[根README](../../../README.md)。

[逐组结果](../data/results/B/base_oracle_results.md) · [图集](../output/B/base_oracle_results_index.html)
