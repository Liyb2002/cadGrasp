# Slides

最新可见实体：[B / pose1+3 窄地框合并模型](baseline_algo/output/B/pose1+3/step5/overview.png)、[单独结构对比](baseline_algo/output/B/pose1+3/step5/separate.png)、[交互查看](baseline_algo/output/B/pose1+3/step5/index.html)。每个 pose 分别设计地框与头部连接，再合并，约 60.8 cm³；此前 575.2 cm³ 鞍形厚体已因材料浪费而替换。五组标识、六块接触面保持，原载荷和整件退出通过，尚未校核强度。详见 [当前 Step5](baseline_algo/step5_connect_support/README.md)。

配对输出直接按任务对组织：`baseline_algo/output/B/pose1+3/` 等目录下分别是 Step1–5；不再把双 pose 结果放在 `B/pose_1/.../pair_pose_3/`。每对 Step1 下保留两个任务的原始输入，详见 [输出入口](baseline_algo/step3_scheculer/README.md)。

此前的 Step5 固定配准检查：[共享结构设计与检查](baseline_algo/step5_connect_support/shared_design.md) 复核五对、11 组接触成功，固定当时的共享曲面配准时均违反地脚兼容的必要条件。此结论保留；当前双接触面表示下的实体见上方链接。

2026-09-26 新增顺序式 Step3 实验：每条 particle 先求 pose1 三头，选一个对 pose2 贡献最大的合法共享头，再为 pose2 补两个；分别验收两套三头，完整共享实体留给 Step5。入口及结果见 [3+2 实验](baseline_algo/step3_scheculer/README.md#sequential-3plus2)。以下共同五头版本保留为对照。

2026-09-26 最新实施：双 pose Step3 每轮固定每头为工件总面积 1%，只按覆盖增量做 top5 sampling。选头停止后，仅对两姿态各自严格超过 98% 且未完成的链尝试终止补全：中心和头数不变，单头面积最多到 1.10%，重新检查受力、插入和基本连通性；仍未全覆盖则失败。已经全覆盖的不扩大，面积最小化留到后续结构实体阶段。载荷验收仍为每 pose 原始 32,768 个样本全部通过，纯重力只作诊断。当前流程止于 Step4；下方单 pose、连续验证与 dock 说明为历史模型。见 [当前 Step3](baseline_algo/step3_scheculer/README.md)。

## Current slide figures

The current [design-parameter slide](params/README.md) uses general task indices:
`A_obj^k`, `A_floor^k`, one shared `V`, and `d_k`. Its two storyboards illustrate
tasks 1 and 2; they do not limit the number of input poses. In each task the
fixture stays still while the object slides into it. Contact regions and the
terminal/swept nonpenetration constraints use the common fixture frame.

[Combined equations](combined_equations.png) puts workpiece equilibrium
(`obj_supp`), the reference floor torque integral (`sys_floor`), and common
insertion (`trajectory`) on one pure-white Z-up slide. Regenerate with
`python slides/tools/combined_equations.py` or the `equations` render group.
The floor row follows the user-confirmed reference integral, with scalar normal pressure and
`F_push d_push` for the process-force vector. It uses the reference vertical-pressure
model; full frictional equilibrium is checked separately.

The setup and floor pages show five saved object/pose pairs: B/pose_2, B/pose_3,
A1-f/pose_2, A1-f/pose_3 and C5/pose_2. Other baseline illustrations use B/pose_2;
the separate multi-pose research schematics are described below.
`tools/slide_scene.py` centralizes the white canvas, faceted grey body, muted green
working surface, Z-up camera `[0.8, -1, 0.12]`, and finite ground plane. The ground
margin is 13% of the object's maximum extent; depicted floor data can enlarge it.
Formula figures explicitly export with an opaque pure-white (`#FFFFFF`)
canvas and axes background, including the floor and insertion equation panels.

The [working-area schematic](setup/working_area.png) explains the green patch:
12 red arrows show possible process forces at different positions and directions.
Their tips lie on the actual working surface, using saved reachable load samples.
They represent separate possible loads; arrow lengths are illustrative.

Use the existing cadgrasp environment and local object, setup and baseline data:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/tools/render.py
# Redraw setup and floor, and remove the known retired slide exports:
PYTHONDONTWRITEBYTECODE=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python slides/tools/render.py --only setup floor --clean-old
```

| Case | Target pose |
|---|---|
| B / pose 2 | [Setup](setup/poses/B/pose_2/pose.png) |
| B / pose 3 | [Setup](setup/poses/B/pose_3/pose.png) |
| A1-f / pose 2 | [Setup](setup/poses/A1-f/pose_2/pose.png) |
| A1-f / pose 3 | [Setup](setup/poses/A1-f/pose_3/pose.png) |
| C5 / pose 2 | [Setup](setup/poses/C5/pose_2/pose.png) |
| All five | [Setup gallery](setup/poses/target_poses.png) |

Floor loads are shown together in one [five-case overview](sys_floor/on_the_floor.png).

| Group | Other current figures |
|---|---|
| `setup` | [Working-area forces](setup/working_area.png); B/pose_2 aliases: [target pose](setup/poses/target_pose.png), `setup/poses/tip_B.png` |
| `floor` | [B/pose_2：三个 3D 载荷与地面点](sys_floor/row3.png) |
| `area` | [Contact area and combinations](obj_supp/area/area_B.png) |
| `demand` | [Paired force–moment demand](obj_supp/demand/demand_pairs.png) |
| `heads` | [Head forces](obj_supp/demand/head_total_force.png), [common insertion direction](obj_supp/demand/head_sweep.png) |
| `equations` | [Setup equations](setup/equations/three_equations.png), [workpiece equations](obj_supp/two_equations.png), [Step3 conditions](obj_supp/demand/demand_equation.png), [matrix expansion](obj_supp/solution/solution.png) |
| `trajectory` | [Insertion](trajectory/sweep_demo.png), [sweep equations](trajectory/sweep_eq.png) |

The current multi-pose concept is in [reuse](reuse/README.md): one rigid four-arm
fixture changes orientation and uses different arm tips as object contacts while
other arm bodies support it on the floor. Curved arms follow the latest visual
direction, with a larger footprint and side entry for horizontal insertion and
withdrawal. A single KUKA parks the object on the floor, reorients the fixture
separately, then reorients and reinserts the object. It does not preserve their
relative pose or promise one-time loading. The object is opaque; the requested
video uses one full-width robot scene with no split screen or text. These are
illustrative shapes and movements, not validated multi-pose contact or load results.
See the [current design decision](../codes/algorithm_design_notes.md).

The earlier `belt_test` multi-pose research figures share one blue contact module and one fixed orange
ground frame with three docking sockets. The three tasks use saved B/pose_2,
an explicitly illustrative 25° tilt of pose_2, and saved B/pose_4. The existing static
figures use the same geometry, station placement and working regions:
[three poses](belt_test/three_poses.png), [why interesting](belt_test/whyinteresting/overview.png),
[why interesting without force arrows](belt_test/whyinteresting/overview_no_labels.png),
and [shared-base overview](belt_test/shared_base.png). The tilted task has a distinct
upper-head patch checked for fixture occlusion in the chosen view; its task loads
and finite-tool clearance have not been validated. See the [geometry notes](belt_test/code/README.md).

The earlier dock-based [workflow video](belt_test/shared_workflow.mp4) loads the blue module
first and the object second at each dock; after work it removes the object first
and the module second. It reuses the object–module assembly relationship while
transporting the two parts separately. Its illustrative geometry was rebuilt for
separate loading; the older static figures above were not regenerated. Its layout is one
full-width scene with no steps, detail inset or title/footer. The gripper has a
common palm attached to the wrist and two sliding jaws with fixed-length fingers.
The [combined storyboard](belt_test/shared_workflow_storyboard.png) remains;
individual `shared_workflow_step_*.png` images have been removed and are no longer generated.
This is a KUKA kinematic concept reusing `codes/simulation`, not a physical grasp or
load validation. See [workflow notes](belt_test/code/shared_workflow.md).

Each floor page independently recomputes its 32,768 paired loads and checks the
results against its own Step4 data. Loads range from zero to 0.5 mg in a reachable
30-degree inward cone. The area percentages are sampled joint contact coverage,
not continuous-domain integrals or final support certificates. The insertion
illustration uses the actual B/pose_2 Step5 support and checks its continuous path.
The opposed-head illustration retains its exact local-cone proof.

Retired images were deleted, including old pose galleries, unselected pose
previews, old A1/C5 exports and the independent insertion-study images. Historical
numerical records and setup data remain. `render.py` verifies that baseline files
and setup inputs have not changed; it never invokes a baseline stage or the
separate `setup/poses/target_poses.py` dataset builder. The explicit `--clean-old`
option only removes known retired exports, excluding all baseline outputs.

Git stores rendering code and Markdown. Generated PNG/JSON/NPZ/media remain local
under the repository's existing ignore rules.

---

## 当前决定与讨论记录

2026-09-23 用户确认：以 [一件刚性支撑的多姿态共同设计](../codes/research_notes/multipose_rigid_fixture_design.md) 为 SIGGRAPH 研究主线。共享实体通过重新摆放改变接触／接地角色，物体与支撑分开搬运并重复装载；旧模块＋dock 保留为对照。主要待证收益为工装资源与重复用料减少，方法、实际装卸及收益尚待验证。当前四臂 [reuse 图示](reuse/README.md) 不限定算法拓扑，也不构成物理证据。

以下“要解决的问题”、坐标载荷约定与 Step1–6 流程描述的是**既有单姿态 baseline**，保留 2026-09-21 的初始躺姿安装／固定底座定义；不是新一体支撑的模型。完整操作定义、停止规则和结果统一见 [baseline_algo.md](baseline_algo/baseline_algo.md)。本次文档更新不修改搜索和验收规则。

用户已确认保留：每轮纯重力平衡和整体不上抬硬约束；尺寸优化允许以部分覆盖换面积效率；蓝块采用有厚度连接证据，固定底座和矩形接口在 Step5 的有限候选中联合验证。具体边界见 [已确认规则](baseline_algo/baseline_algo.md#已确认的-baseline-规则)。

### 要解决的问题

输入工件完整网格、初始 rest 躺姿、已保存的目标姿态、质心、原始地面接触点和绿色工作区域。先在初始躺姿安装蓝色接触模块，再抓起工件＋蓝块，整体插入预先放好的静止底座；撤去抓持后承受允许的加工力与重力。载荷仍在目标姿态计算，接触面不得落在初始贴地部分。

当前 baseline 为整体搬运构造几何净空路径，但不求机器人关节轨迹，也不搜索跨姿态复用。Setup 提供的物体姿态不随坐标记号或公式排版改变。

初次装蓝块的方向与对接底座的方向分别处理：前者检查蓝块在初始地面上的完整扫掠；后者检查完整工件＋蓝块相对固定底座的扫掠。当前对接构造采用竖直插入，蓝块终态同时避开初始与目标地面。初始躺姿可作水平平移安排工位。底座保留不锚固、无自重的承载模型，矩形插座侧壁与止挡参与三刚体平衡。

机器人实验须检查准备与安装路径无碰撞、抓持与保持可执行；保持用的夹爪和机械臂不能阻挡支撑及其搬运工具。支撑接管后，夹爪须能张开并撤离，载荷验证不计机器人扶持力。当前 baseline 的支撑扫掠证书不包含机器人与夹爪。

2026-09-18 范围确认：本版优先把**单个固定 pose 下的找头问题**弄清楚，区分候选区域、组合搜索、尺寸调整与验证未决造成的失败；先可靠获得符合当前姿态的接触组，不要求全局最小面积。本版不实施多姿态联合优化和共享腰带。当前头部成功不等于底座与完整支撑成功。

2026-09-22 搜索方向更新：已确认以多 shape 的昂贵搜索结果训练 value network，预测部分头组的后续完成价值。运行时外层选中心，内层轮流优化全部已选头的尺寸，并保留多个候选组合；拟复用小型集合网络。当前仅完成 [方案记录](../codes/research_notes/value_guided_contact_search.md)，尚未实现、训练或运行对照；baseline 规则保持不变。接触密度／有限压强模型仍未采用。

**历史后续方案（以下三段）：** 记录旧共享头／腰带与组合搬运路线，已被 2026-09-23 的一体支撑主线取代；不作为新方案的约束或收益前提。

2026-09-20 后续多姿态方向更新：共享蓝色的同一组头、紧凑连接腰带和同一模块接口，各姿态分别设计橙色地脚与到腰带的连接结构。蓝橙配合几何和局部直线插入动作保持相同，其世界位置、朝向和推入方向可以不同；不要求整个橙色底座复用，也不再采用三条蓝色长分支去适配一个固定插座。主要动机是把反复对复杂工件表面装夹，转为机器人可执行的接口对接（single straight-line push），无需另行主动夹紧；接口导向可降低精细对准需求，不声称消除对准。共享的是物体坐标中的接触几何，每个 pose 都须根据旋转后的接触力／力矩、重力、任务载荷和地面条件重新求自己的合法反力。完整说明及比较边界见 [研究动机与接口对接流程](../codes/algorithm_design_notes.md#2026-09-20从多步装夹到单次接口对接)。

两个阶段的关系是：**先把各 pose 独立找头的 baseline 做好，再研究共享头的共同求解。** 当前允许各姿态使用不同头组；后续才增加“所有姿态使用同一组接触几何”的约束。各姿态独立求解是正式对照，不因未来共享目标提前缩小其设计空间。

未来完整实验与 MuJoCo 验证仍须覆盖初次安装、机器人重新接管、姿态切换、模块装卸、重新落座与释放，不能只把各姿态重置后分别测试。期望让蓝色腰带留在工件上，由机器人携带这个组合对接预先放好的不同橙色底座，大底座留在工位；腰带的保持机制及可能的共同抓持仍须在实现前明确，不能从“共享”或“连接”推断为与工件刚性绑定。搬运可以采用已有规划器或示教；静态终点通过不能认证转换过程。用户选择以**操作时间**和**全套材料消耗**两个实验验证复用收益，与各 pose 独立完整支撑比较；定位精度和任务可执行性作为公平验收条件，不把省料直接等同于总成本下降。实验尚未运行，详见 [两个主实验](../codes/research_notes/reuse_benefit_evidence.md#用户确认的两个主实验)。

### 坐标、载荷与接触

- 右手世界坐标 Z-up，地面 `z=0`，重力为 `-mg zhat`；几何计算用米。
- 加工力作用点 `q`（数据文件中为 `pt`）位于绿色工作面，方向在局部内法向的 30° 半角锥内，大小为 `0 ≤ |F_push| ≤ 0.5mg`。每条载荷是一个独立工况。
- Step1 拒绝被工件自身遮挡的工具射线；当前支撑设计关闭加工射线体积禁区。这两件事不同，不能把“禁区关闭”理解成 Step1 不查可达性。支撑接触仍避开实际工作面。
- `r_push=q-c`、`r_supp=p_contact-c`，力矩均关于工件质心 `c`。
- 接触头只有无摩擦的单边法向压力，没有拉力和压强上限；允许不同头的力分别向上、水平或向下。所有头最后连接为同一个刚体。
- 工件原地面支点与支撑底座使用单边摩擦模型。Baseline 的有限摩擦参数和最终验收见算法说明，公式示意图中的竖直地面压力是简化展示。
- 假设准静态、刚体、支撑无自重；暂不计算打印强度、变形或定位误差；接触模块优化不求机器人关节运动，KUKA 演示使用逐帧 IK，不等于完整机器人执行验证。

### 三组公式的角色

[合并公式图](combined_equations.png) 分别表达工件平衡、参考地面力矩式、完整支撑的插入条件。
[需求公式图](obj_supp/demand/demand_equation.png) 表达力／力矩、整体不上抬、完整蓝块的连接与初次安装要求；Step3 用充分连接构造筛选，Step5 再构造包含接口的实际蓝块与底座。

工件的配对需求为

\[
(F_D,\tau_D)=\left(mg\hat z-F_{\rm push},\;-r_{\rm push}\times F_{\rm push}\right).
\]

同一组接触反力须同时平衡力和力矩。反力可以随工况改变；接触几何不随工况改变。
Step3 还要求所有头作用于工件的竖直力之和非负，即工件对整个无自重支撑的合力不能向上拔起它。这里不计工件自身的地面反力，也不要求每个头单独满足该不等式。

地面部分保留两种用途不同的图：

- [合并公式图](combined_equations.png) 的 `sys_floor` 行沿用已确认的参考力矩积分式。这一行采用标量地面压力，图中另用粗体区分完整力向量。
- [row3.png](sys_floor/row3.png) 读取当前 `objects/B/tasks/pose_2`，展示三个不同载荷的 3D 图。每幅只有一个加工力与重力，两条作用线交于 X，合力 W 从 X 延伸至地面 p；工作面使用当前 6–10% 设置。这三个例子专门选取实际相交的作用线并检查完整力矩等价，不把任意三维工况的作用线视为必然相交。单独重画：`python slides/sys_floor/row3.py`。

压力中心落入实际接地凸包只是必要条件，完整支撑仍须通过共享反力的平衡检查。Step4 只产生需求点，不产生可直接认证的底座。

下式保留旧整件支撑图的记号，`a` 表示装入方向；当前模块初次安装须换到初始躺姿坐标，对接底座另作整体扫掠：

\[
\mathrm{Sweep}(\mathrm{supp},a)=\{x-ta:x\in\mathrm{supp},\ t\geq0\},\qquad
\mathrm{Sweep}(\mathrm{supp},a)\cap
\bigl(\operatorname{int}(\mathrm{obj})\cup\{z<0\}\bigr)=\varnothing.
\]

代码方向表保存的是退出方向，初次装蓝块取其反向。Step3 检查接触头和连接证据；Step5 检查实际蓝块初次安装、工件＋蓝块对接静止底座及接口传力；Step6 重放这些检查并构造搬运演示。接触面允许贴合，实体内部不得穿透。

### 当前 baseline 的流程

1. **Step1：载荷需求。** 按工作面积、方向立体角和力度采样，保留 32,768 个可达工况，生成配对六维需求。
2. **Step2：候选接触头。** 在非工作曲面按真实面积分配 200 个中心；拟合目标面积为物体总面积 1% 的连通接触圆，检查整块法向夹角、实体净空和各头的退出方向。
3. **Step3：十条 top5 搜索链。** 每条最多 3 个头，先筛几何、面积、共同初次安装方向、有厚度连接和纯重力平衡；在联合覆盖评分前五中抽样，轮流调整全部已选头的尺寸。优化后样本全覆盖才做连续域验证。
4. **Step4：地面需求。** 映射重力、加工样本、已发现反例和连续载荷外包，输出地面压力中心及外包点，不设计底脚。
5. **Step5：模块与固定底座。** 构造实际蓝块、矩形插头、静态地面环与插座；验证蓝块初始安装与组合竖直对接，检查工件／蓝块／底座三刚体共享反力的连续承载。
6. **Step6：搬运与视频。** 重放几何和承载证据，构造避开固定底座的高处搬运路径，生成初次装蓝块、一起抓起、整体对接的 15 秒示意视频；抓持及机器人运动学未认证。

完整设计通过需同时满足 Step3 连续覆盖、Step5 底座检查、Step6 实体/间隙/接触/实际接地检查、整件连续轨迹和共享反力的连续承载。视频能播放、样本全覆盖或审计通过都不能单独替代这一结论。

历史记录（旧整件模型）：2026-09-17 将共享承载加入 Step5 底座筛选后，B/pose_1、B/pose_2 均完整通过，底座占地较原流程分别减少约 48% 和 55%。其余十例尚未按新流程重跑；历史结果与当前停止点见 [当前结果](baseline_algo/baseline_algo.md#当前保存结果)。
