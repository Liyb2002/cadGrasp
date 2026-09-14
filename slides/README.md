# Slides

## Current slide figures

[Combined equations](combined_equations.png) puts workpiece equilibrium
(`obj_supp`), the reference floor torque integral (`sys_floor`), and common
insertion (`trajectory`) on one pure-white Z-up slide. Regenerate with
`python slides/tools/combined_equations.py` or the `equations` render group.
The floor row follows the user-confirmed reference integral, with scalar normal pressure and
`F_push d_push` for the process-force vector. It uses the reference vertical-pressure
model; full frictional equilibrium is checked separately.

The setup and floor pages show five saved object/pose pairs: B/pose_2, B/pose_3,
A1-f/pose_2, A1-f/pose_3 and C5/pose_2. The other current illustrations use B/pose_2.
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
| `floor` | [Reference resultant schematic](sys_floor/row3.png), [B/pose_2 resultant](sys_floor/resultant_B_pose_2.png) |
| `area` | [Contact area and combinations](obj_supp/area/area_B.png) |
| `demand` | [Paired force–moment demand](obj_supp/demand/demand_pairs.png) |
| `heads` | [Head forces](obj_supp/demand/head_total_force.png), [common insertion direction](obj_supp/demand/head_sweep.png) |
| `equations` | [Setup equations](setup/equations/three_equations.png), [workpiece equations](obj_supp/two_equations.png), [Step3 conditions](obj_supp/demand/demand_equation.png), [matrix expansion](obj_supp/solution/solution.png) |
| `trajectory` | [Insertion](trajectory/sweep_demo.png), [sweep equations](trajectory/sweep_eq.png) |

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

以下为 2026-09-13 按当前代码核对的模型和算法。此前逐次追加的旧流程说明已整理移除；当前 baseline 的完整操作定义、停止规则和结果统一见 [baseline_algo.md](baseline_algo/baseline_algo.md)。代码清理不改变下述搜索和验收规则。

用户已确认保留：每轮纯重力平衡和整体不上抬硬约束；尺寸优化允许以部分覆盖换面积效率；Step5 首套几何/轨迹通过后的承载失败不触发重新搜索。具体边界见 [已确认规则](baseline_algo/baseline_algo.md#已确认的-baseline-规则)。

### 要解决的问题

输入工件完整网格、已保存的目标姿态、质心、原始地面接触点和绿色工作区域。目标姿态在搜索中固定；安装时工件由外部保持在该姿态。搜索一件连接所有接触头的刚性支撑，使它能装入，并在撤去外部保持后承受允许的加工力与重力。

当前 baseline 不搜索工件翻转到目标姿态的过程，也不搜索跨姿态复用。Setup 提供的物体姿态不随坐标记号或公式排版改变。

### 坐标、载荷与接触

- 右手世界坐标 Z-up，地面 `z=0`，重力为 `-mg zhat`；几何计算用米。
- 加工力作用点 `q`（数据文件中为 `pt`）位于绿色工作面，方向在局部内法向的 30° 半角锥内，大小为 `0 ≤ |F_push| ≤ 0.5mg`。每条载荷是一个独立工况。
- Step1 拒绝被工件自身遮挡的工具射线；当前支撑设计关闭加工射线体积禁区。这两件事不同，不能把“禁区关闭”理解成 Step1 不查可达性。支撑接触仍避开实际工作面。
- `r_push=q-c`、`r_supp=p_contact-c`，力矩均关于工件质心 `c`。
- 接触头只有无摩擦的单边法向压力，没有拉力和压强上限；允许不同头的力分别向上、水平或向下。所有头最后连接为同一个刚体。
- 工件原地面支点与支撑底座使用单边摩擦模型。Baseline 的有限摩擦参数和最终验收见算法说明，公式示意图中的竖直地面压力是简化展示。
- 假设准静态、刚体、支撑无自重；暂不计算打印强度、变形、机器人运动学或定位误差。

### 三组公式的角色

[合并公式图](combined_equations.png) 分别表达工件平衡、参考地面力矩式、完整支撑的插入条件。
[需求公式图](obj_supp/demand/demand_equation.png) 表达的是 Step3 的接触模型与头部共同方向，尚未涉及最终框架和底座。

工件的配对需求为

\[
(F_D,\tau_D)=\left(mg\hat z-F_{\rm push},\;-r_{\rm push}\times F_{\rm push}\right).
\]

同一组接触反力须同时平衡力和力矩。反力可以随工况改变；接触几何不随工况改变。
Step3 还要求所有头作用于工件的竖直力之和非负，即工件对整个无自重支撑的合力不能向上拔起它。这里不计工件自身的地面反力，也不要求每个头单独满足该不等式。

地面部分保留两种用途不同的图：

- [合并公式图](combined_equations.png) 的 `sys_floor` 行沿用已确认的参考力矩积分式。这一行采用标量地面压力，图中另用粗体区分完整力向量。
- [row3.png](sys_floor/row3.png) 沿用 [平面作用线参考图](sys_floor/ref.png)：两个共面力的作用线交于 X，合力 W 从 X 延伸至地面 p；这是平面示意。一般三维工况按完整合力和合力矩计算压力中心，不能假设两条作用线总能相交。

压力中心落入实际接地凸包只是必要条件，完整支撑仍须通过共享反力的平衡检查。Step4 只产生需求点，不产生可直接认证的底座。

轨迹中 `a` 表示装入方向：

\[
\mathrm{Sweep}(\mathrm{supp},a)=\{x-ta:x\in\mathrm{supp},\ t\geq0\},\qquad
\mathrm{Sweep}(\mathrm{supp},a)\cap
\bigl(\operatorname{int}(\mathrm{obj})\cup\{z<0\}\bigr)=\varnothing.
\]

代码方向表保存的是退出方向 `d=-a`。Step3 检查接触头，Step5 检查连接完成后的整件实体；工件固定，支撑沿同一直线平移，接触面允许贴合，实体内部不得穿透。

### 当前 baseline 的流程

1. **Step1：载荷需求。** 按工作面积、方向立体角和力度采样，保留 32,768 个可达工况，生成配对六维需求。
2. **Step2：候选接触头。** 在非工作曲面按真实面积分配 200 个中心；拟合目标面积为物体总面积 1% 的连通接触圆，检查整块法向夹角、实体净空和各头的退出方向。
3. **Step3：最多 3 轮贪心。** 先筛几何、面积、共同方向和纯重力平衡；按“旧头加候选”的联合载荷覆盖数选一个头，再只优化新头半径，以联合覆盖率／总接触面积为目标。旧头不回溯修改。优化后样本全覆盖才做连续域验证。
4. **Step4：地面需求。** 映射重力、加工样本、已发现反例和连续载荷外包，输出地面压力中心及外包点，不设计底脚。
5. **Step5：整件构造与验证。** 先试共同退出方向，再造颈部、宽松框架、开口底座和粗连接。第一套几何及整件轨迹通过后检查承载；承载失败当前不返回继续换结构或重选头。

完整设计通过需同时满足 Step3 连续覆盖、Step5 实体/间隙/接触/实际接地检查、整件连续轨迹和共享反力的连续承载。视频能播放、样本全覆盖或审计通过都不能单独替代这一结论。

当前已保存的 12 个案例中，B/pose_1、B/pose_2 完整通过；其余案例的具体停止点和局限见 [当前结果](baseline_algo/baseline_algo.md#当前保存结果)。
