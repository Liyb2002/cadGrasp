# Whole task：工作区、姿态与共享支撑

## 当前 task：每个表面三角面最多加工一次（2026-10-21）

Step3 和 Step4 现在实际使用 Step1 的**不重复分区 `face_task`**。同一个源三角面只属于一个 task；每个姿态约 40% 的 `task_reach` 仅作为候选可达区域保留，不再整体传给下游。姿态、工具、区域清理和允许部分覆盖的设定沿用原值。

| 实际 task | 面数 | 占目标表面积 |
|---|---:|---:|
| +X | 672 | 12.38% |
| +Y | 1,215 | 25.56% |
| +Z | 800 | 15.85% |
| −X | 552 | 10.37% |
| −Y | 1,467 | 29.85% |

合计 **4,706 面、94.02% 表面积，重复面数和重复表面积均为 0**。−Z 没有独有面，保留输入摆放记录，但不生成下游任务或支撑。剩余 294 面仍未分配。未认证的包围补入面继续单独标记，不因此获得工具认证。Step1.3 的粉色同步显示这份实际分区。

Step3 的 5-task 初始化及其独立磁盘复验已更新；Step4 从同一组实际 task 重建禁区。旧的 Step4 座位搜索／合并和 Step5 成品尚未重跑；输入版本检查会拒绝旧座位缓存。32 项相关测试通过。

## Step1：固定工位，兔子按六个轴对齐姿态落桌（2026-10-21）

机械臂底座和工位名义针方向固定，名义针方向为 `(0, −0.7071, −0.7071)`，针可相对它偏转最多 25°。加工时机械臂仍带着头平移、调整针轴。兔子采用 `+X、+Y、+Z、−X、−Y、−Z` 六种候选轴对齐摆放；分区后非空的摆放成为实际 task，允许单个 task 的表面区域不连通。

方向名表示物体自身的哪一侧**水平朝向机器人**。四个侧视摆放保持原模型 +Y 朝上；+Y／−Y 两个顶底视摆放分别以 −Z／+Z 朝上。旋转只改变轴向，不按斜向针轴倾斜兔子；每个姿态再按最低顶点落到同一桌面。当前桌面高 222.607 mm，相对机械臂底座安装面计；桌板为 500×500×20 mm。落桌是几何定位，承载和稳定性由后续支撑验收处理。

每个姿态的实际针方向先转换到物体坐标系：`axis_object = rotation_world_object.T @ working_direction_world`，再在这个名义轴的 25° 范围内搜索。+X 等输入姿态标签与实际斜向针轴分别保存为 `view_object` 和 `axis_object`。

**区域清理规则（2026-10-21）：** 小块阈值为目标总表面积的 1%（`small_piece_fraction = 0.01`）。先把边界完全属于同一 task 的小包围区域归给该 task，即使它未通过该姿态的可达认证；再删除该 task 主体之外、面积小于阈值的不连续小块。每个 task 的最大连通块保留，较大的不连通区域也保留。对每个姿态的独立区域和最终不重复分区都执行这些规则。

其余面可以保持未分配；算法不借通道连接碎片，也不将剩余面强制归最近 task。目前默认允许部分覆盖，99.5% 只保留为可选的参考门槛。

`reach` 保留原始认证可达集合，`task_reach` 保存清理后的候选区域，`task_enclosed` 标记候选补入面，`task_removed` 标记删除的原始可达小块。实际任务取 `face_task == direction_index`，与报告中各 task 的 `face_ids` 完全一致。补入区域不会因此变成已认证可达。

| 输入姿态 | 原始认证面积 | 清理后候选面积 | 实际 task 面积 | 删除面数 | 补入面数 |
|---|---:|---:|---:|---:|---:|
| +X | 43.29% | 42.76% | 12.38% | 39 | 11 |
| +Y | 41.88% | 42.67% | 25.56% | 31 | 63 |
| +Z | 43.07% | 43.40% | 15.85% | 30 | 48 |
| -X | 38.48% | 39.03% | 10.37% | 33 | 47 |
| -Y | 40.35% | 42.19% | 29.85% | 16 | 117 |
| -Z | 41.88% | 42.67% | 0.00% | 31 | 63 |

各姿态的独立区域允许重叠，表中的面数按姿态分别统计。六姿态原始认证并集为 **94.44%**；清理后的区域并集为 **95.99%**（含未认证补入面）；不重复分区最终分配 **94.02%**，剩余 **294 个面、5.98%** 保持 `face_task = −1`。

报告状态为 `analyzed`。`requirement_met = false` 只说明原始认证覆盖低于参考值 99.5%，不阻止当前分析。当前 −Z 姿态候选区域为 42.67%，但面已归其他姿态，因此分配为 0；六个输入姿态都保留，下游只有五个非空 task。106 个已分配面未通过所属姿态认证，单列在 `uncertified_face_ids`。

认证逐面检查三个顶点和质心：针轴无遮挡、2 mm 半径针身避碰、100 mm 针长之后的 20 mm 半径工具头避碰；机械臂逆运动学满足限位、离地，以及连杆对兔子和有限桌板的避碰。针身和头也必须位于桌面以上。针轴按约 3° 间隔取样，深度图用于初筛，最终用三角形裁切和距离计算认证。

32 项相关测试通过。原始认证数据及清理规则会从磁盘复验，包括删除面、包围补入面、分区中没有遗留小孤岛或小包围孔，以及固定工位和落桌姿态。实现见 [区域清理](step1/areas.py)、[摆放规则](step1/workspace.py)、[规划与复验](step1/plan_reach.py)、[机械臂和桌面检查](step1/arm.py)。

- [Step1.1：固定工位与桌面](output/B/step1/step1.1/station.png) · [输入](output/B/step1/step1.1/setup.json)
- [Step1.2：清理后不重复分区的六个视角](output/B/step1/step1.2/areas.png) · [各姿态分到的面](output/B/step1/step1.2/overview.png)
- [Step1.3：六种落桌姿态的实际 task 区域](output/B/step1/step1.3/working_poses.png)：粉色取不重复分区 `face_task`；−Z 为空，全灰显示，不放置加工工具。五个有效 task 的工作近景保持原取景，图中无文字。
- [实际 task 并集：粉色](output/B/step1/step1.3/covered.png)（94.02%） · [未分配：红色](output/B/step1/step1.3/uncovered.png)（5.98%）：两图各含相同的六个视角；[面积、面编号、逐 task 面积与成对重复面数](output/B/step1/step1.3/coverage.json)。
- [规划](data/B/step1_reach/plan.json) · [复验](data/B/step1_reach/validation.json)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step1.plan_reach
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step1.plan_reach --validate-only
```

默认不要求覆盖全部目标；只有显式传入 `--require-coverage` 才检查配置中的认证覆盖门槛。`--allow-partial` 保留为兼容选项。

[工具与金锥示意图](idea/find_tasks.png) 和 [10 秒移动示意](idea/find_tasks.mp4) 保留去重前的候选可达区域示意，尚未按这次实际 task 分区重制；其专属 JSON 仍绑定旧 plan，不作为当前 task 图。当前实际工作面以 Step1.3 和 Step3 新初始化为准。

此前的 [head_reach 视频](idea/head_reach.mp4) 保留原摆放演示，已去掉前 6 秒，裁后 53.125 秒、1,275 帧；它尚未按新的落桌姿态重制。Step4–5 的已发布结果尚未按当前桌面姿态和区域规则重建。

## Step2 与 Step3 初始化：5 个不重复 task 共用一块 base（2026-10-21）

Step3 直接读取 `face_task`，并逐 task 核对报告中的 `face_ids`，只初始化五个非空 task。每个 task 的针轴和 KUKA 关节角来自所属姿态自己的 `reach` 认证，不借用其他姿态的证据；包围补入面参与受力，但不因此获得工具可达认证。Step3 的构造报告和独立复验都从实际 task 面列表重算重复面数与表面积，要求均为 0。

每个 task 生成一个 5 mm 薄壳座位，五个座位用脚和同一块水平底板连成**一个完整 base**。主图上排展示五只兔子及互不重复的粉色 task，下排展示同一块空 base。图中没有文字。相邻座位间隔 8 cm；五件薄壳合计 **797.506 cm³**，完整 base 为 **1,790.986 cm³**。连接和序列化后，成对受力、工具、KUKA 与连续退出均通过独立磁盘复验，验收容差保持原值。

Step2 已重新导出这五个 task 在 Step3 中实际使用的成对力／力矩，每个 task 保存 32,768 个样本，总计 **163,840 对**，另存 **13,253 个连续力锥外包目标**。所需反力和力矩为 `[-g-P, -(q-COM)×P]`，关于当前世界坐标系中的质心计算；完整关联关系保存在数组中。

- [Step3 新初始化：一个 5-task base](output/B/step3/initialization.png) · [base.obj](output/B/step3/base.obj)
- [Step3 报告](data/B/step3_tool_membranes/report.json) · [独立磁盘复验](data/B/step3_tool_membranes/validation.json)
- [Step2：力与力矩图](output/B/step2/force_moment.png) · [受力域与数组索引](data/B/step2/report.json) · [成对映射复验](data/B/step2/validation.json)

旧版 Step3 图已删除，发布目录只保留这张新的完整 base 图。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step3.run --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step3.tool_membranes --validate-only
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python whole_task_2/data/B/step2/export_loads.py
```

## Step4 准备：五个实际 task 的针 ∪ 头 ∪ KUKA 禁区（2026-10-21）

Step4 的 `keepout.py` 和座位模型从 Step3 读取同一组五个不重复 task。禁区只沿该 task 自己的认证工作面和关节角构造；不再使用每个姿态约 40% 的候选区域。实际面列表的独立重复检查写入禁区报告及复验。

| task | 实际工作面 | 其中认证面 | 保存姿态 | KUKA 凸体 | 合计禁区 cm³ |
|---|---:|---:|---:|---:|---:|
| +X | 672 | 672 | 2,688 | 329 | 90,693 |
| +Y | 1,215 | 1,165 | 4,660 | 441 | 91,058 |
| +Z | 800 | 800 | 3,200 | 304 | 86,030 |
| -X | 552 | 528 | 2,112 | 330 | 90,540 |
| -Y | 1,467 | 1,435 | 5,740 | 498 | 97,026 |

共 18,400 组保存关节角。KUKA 按连杆中心所在网格分组取凸包，边长为物体对角线的 12%；构造前在 1 µm 网格上合并近重复点。本次最大包含残差为 0.837 µm，小于配置容差 2 µm。工具、物体、桌面与实际连杆的精确避碰验收沿用原值；这里的禁区只覆盖保存的加工姿态，不认证姿态间路径。

[keepout 图](idea/keepout.png) 展示五个实际 task，粉色与新 Step3 一致；针和头为橙色，KUKA 为紫色，禁区以 45% 不透明度叠加，无文字。头部延伸显示为完整数据的 70%，只影响出图。复验从磁盘输入独立重建针、头和 KUKA 实体，并比较保存几何的对称差，体积容差为 `1e-10 m³`。[报告](data/B/step4_keepout/report.json) · [复验](data/B/step4_keepout/validation.json) · [图像记录](data/B/step4_keepout/figure.json)。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step4.keepout --workers 5
```

Step4 的旧座位搜索／合并结果未按新 task 重跑。`seats.py` 将当前输入签名写入转角及座位搜索结果，后续读取时拒绝旧任务版本，避免把旧六任务缓存用于新五任务。

以下 Step4 座位搜索、合并及 Step5 的已发布结果仍来自此前版本，尚未按最新输入重建。

## 新 Step4：哪些 task 共用一个座位（2026-10-21）

`step4/seats.py`。支撑坐标系就用兔子坐标系，一个"座位"就是兔子在支撑里的一个固定位置。几个 task 共用一个座位时，它们共用同一块 5 mm 薄壳：Step3 的外层减去每个成员的"针 ∪ 头 ∪ KUKA"禁区和退出扫掠。所以共用座位的体积一定不超过成员里最小的那一块，合并门槛 V(AB) ≤ ρ(V(A)+V(B)) 只要 ρ ≥ 0.5 就自动满足。真正决定能不能合并的，是薄壳还能不能托住每个成员的全部载荷、让 KUKA 和退出都通过，并且能立在桌上。

- **转角**：每个 task 可以绕针轴转 ψ（底座固定，KUKA 在该转角下重新解 IK，禁区按 `turned_keepout` 重新生成）。转动会让"重力在兔子身上的方向"沿一个锥面移动。两个 task 的锥面相交时，在交点处两者重力方向相同，共用座位时就是支撑同一面朝下，只需在桌上绕竖直轴转一下。所以每个 task 的候选转角是 0 和"与别的 task 锥面相交"的那些角度。兔子上相邻的两个面都恰好相交于一点（立方体棱的方向），相对的面不相交。20 个候选转角里，只有 −X 转 135° 时 KUKA 够不到所有面。
- **逐层搜索**：从单个 task 开始，一组 task 只有在它的所有子集都可行时才去试（成员越多，薄壳被切得越多，所以可行性是单调的）。每个成员在每个可行转角下，对这组的薄壳做完整验收（32,768 对载荷 + 连续外包、逐面连续退出、禁区、KUKA 精确检查）。然后按"朝下面最少、转角最少"的顺序挑转角组合，把这个座位做成**能自己站在桌上的一件**：薄壳、最低处一根立柱、一块盖住立柱和每个成员压力凸包的底板（外扩 15 mm），对所有成员再完整验收一次，并要求压力凸包落在接地面内。
- 之所以把底板也放进座位一起验收：manifold 做任何布尔运算，都会在整张网格上合并短边，第一版只验收了光薄壳，连起来以后 −Y 丢了一个 0.26 mm² 的接触碎片，第 859 号载荷就过不去了。并过一次以后接触不再改变，所以 Step4 验收的就是最终会用到的网格。

结果（46 秒，转角认证有缓存）：6 个单独座位都可行；12 对相邻面都能共用座位，3 对相对面不能（只靠重力就托不住）；8 个"三个面共用"的组合都不可行——三个面的禁区和退出把薄壳切得只剩一个角，无摩擦接触托不住。12 对里 10 对能做成单独站立的一件（+X/−Y 体积最小，0.12，但站不住）。合并比为 0.26–0.33，远低于 0.6。

## 新 Step4–5：三个座位连成一整件，1 个 pose（2026-10-21）

`step4/join.py` 先选组合：用可行的座位把 6 个 task 各覆盖一次。每个座位转到让朝下方向尽量重合，算出整件需要几个 pose（翻面才算换 pose；在桌上绕竖直轴转、平移都不算）。排序依次看 pose ≤ 3、座位最少、薄壳最少。选中的是 **{+Z, −Y}、{+X, +Y}、{−X, −Z}**，每个座位的两个 task 转角分别是 (0°, 180°)、(−135°, 90°)、(−135°, −90°)，两个 task 之间支撑在桌上转 180°。三个座位朝下方向相同，所以**整件只有 1 个 pose**，永远同一面朝下。

排座位：三个座位沿 x 排成一排，兔子都在 y = 0 上。对每一种先后顺序、每个座位转不转半圈，找相邻两座的最小间距，条件有三条：各自的站立件都不碰对方的禁区和退出扫掠；薄壳之间至少隔 2 mm（各座位只靠立柱和底板相连，验收过的接触面不受影响）；不相邻的座位也要两两满足。同一时间只有一个座位上有兔子，所以座位可以靠得很近，三个座位中心相距一共 148 mm。

`step5/joined.py` 把三块底板合成一块（取凸包，厚 8 mm），然后对 6 个 task 在整件上逐个完整验收：

| task | 座位 | 转角 | 载荷 | 连续外包 | KUKA 碰撞面 |
|---|---|---:|---:|---:|---:|
| +Z | +Z/−Y | 0° | 32,768/32,768 | 3,217/3,217 | 0 |
| −Y | +Z/−Y | 180° | 32,768/32,768 | 1,777/1,777 | 0 |
| +X | +X/+Y | −135° | 32,768/32,768 | 2,257/2,257 | 0 |
| +Y | +X/+Y | 90° | 32,768/32,768 | 1,633/1,633 | 0 |
| −X | −X/−Z | −135° | 32,768/32,768 | 2,017/2,017 | 0 |
| −Z | −X/−Z | −90° | 32,768/32,768 | 2,833/2,833 | 0 |

逐面连续退出、和兔子及禁区不重叠、放在桌上时压力凸包落在接地面内，全部通过。从磁盘复验时由保存的座位重建整件，和存盘实体的对称差为 0，再完整验收一遍也通过。**整件 803.7 cm³**，其中薄壳 295.8 cm³（Step3 六个单独薄壳是 1,035.5 cm³，连在一起是 1,450.1 cm³），立柱和底板 507.8 cm³。外形 44 × 19 cm，接地面积 634 cm²。底板要盖住每个 task 在推力下的压力中心范围，所以大，也是体积的大头；改成稀疏的脚可以省掉大部分底板材料，但接地范围（凸包）不会变。

门槛 ρ 的影响：ρ 从 0.31 到 1，结果完全相同；ρ = 0.3 时变成 4 个座位（523.8 cm³ 薄壳），ρ = 0.25 时变成 6 个座位。这个模型里收紧门槛不会让搜索更快，只会把好的合并刷掉。搜索本身是穷举，不是随机。

- [整件](output/B/step5/joined.png)：上排是每个座位放着兔子（粉色是该座位两个 task 的区域），下排只有支撑。· [使用](output/B/step5/use.png)：6 个 task 各一幅，支撑转好、滑到工位，KUKA 用认证过的关节角夹着针，停在该 task 中间的认证面上。按"机械臂跟着兔子升降"的约定，在认证高度画。
- [座位搜索](data/B/step4_seats/sets.json) · [排座位](data/B/step4_join/report.json) · [整件报告](data/B/step5_joined/report.json) · [复验](data/B/step5_joined/validation.json) · [support.obj](data/B/step5_joined/support.obj)

**使用视频** `step5/joined_video.py` → [joined.mp4](output/B/step5/joined.mp4)（无字，2000×1000，24 fps，约 93 s）。左边是近景，从能看到针所涂各面的方向拍工位；右边是全景，能看到桌子、支撑、立柱和 KUKA。场景约定：桌子固定，KUKA 站在立柱上，只沿竖直方向升降。按座位顺序做 6 个 task：
- **做一个 task**：支撑在该 task 的安装位（绕竖直轴转好，座位对准工位），KUKA 降到这个 task 的高度。针依次走过该 task 的认证面：每个停点的针尖落在面心，沿该面的认证轴；停点之间沿针轴退一点再过去。浅粉是还没涂的，深粉是涂过的，涂过的一直留着。
- **同一座位换 task**：针沿针轴退开，KUKA 升高，支撑在桌上转 180°，KUKA 再降下来。
- **换座位**：支撑滑离工位到上料位。兔子沿认证过的退出方向（逆针方向）离开座位，飞到下一个座位上方，沿那个座位的退出方向反向放进去，然后支撑滑回工位。

关节角由逐帧的最小变化跟踪（阻尼最小二乘）得到；每段动作分配的帧数，保证回放时每个关节都不超过 KUKA 限速的 90%。每一帧都做检查，不只检查停点：
- KUKA 连杆凸包对支撑、兔子、桌子，用 step1.arm 的精确三角形测试；
- 针和头（外接棱柱）对支撑和桌子；
- 针对兔子，按 Step1 reach 的规则：针线不进入兔子，针身在针尖后 r(1+tan φ) 之外不碰兔子；停点本身是 Step1 认证过的接触；
- 兔子移动时对支撑，用精确布尔运算。

检查没通过的动作会依次加大让位量（抬针高度、后退距离、升高、飞行弧高）再试。最终 2238 帧全部通过：后退 6 cm、升高 5 cm 就够了，全程没有 IK 重启。见 [video.json](data/B/step5_joined/video.json)。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step4.keepout          # 禁区
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step4.seats --workers 16  # 转角 + 座位
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step4.join --workers 20   # 选组合 + 排座位（--rho）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step5.joined             # 连成一整件 + 验收 + 复验 + 出图
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step5.joined_video       # 使用视频（逐帧检查）
```

下面"当前结果（Step4 v3）"及以后是旧的 13 个锥形工作区的结果，保留为历史；`output/B/step4/overview.png` 和 `output/B/step5/` 里的 `hulls_and_support.png`、`result.mp4`、`robot_use.mp4` 也是旧的。

## 当前结果（2026-10-20，Step4 v3）

Step1–3 不变：13 个工作区域，13 个独立的 5 mm 薄壳初始化，共 **2062.312 cm³**。Step4 v3 只有一种高阶操作——**把一件已验收的支撑刚体地放到另一件上，保留合法且省料的并集**：Rotational reuse 在 SO(3) 中采样整件旋转，Juxtaposition 只平移；用 LP 让所有安装地面相容，在真实构造上精确评分并局部细化，再完整验收；合并门槛仍为 **V(AB) ≤ 0.75×(V(A)+V(B))**。ground oracle 与 Step5 使用同一个落地模型（薄壳切平面落地、保持座位的绕针轴转角、同一个地脚可用空间函数），所以 Step4 接受的支撑都能被 Step5 落地。步长、重叠和网格尺度都相对物体尺寸给出。详见 [PROCESS.md](PROCESS.md)。

从 Step3 冷启动约 77 分钟，结果为 **一件支撑支持全部 13 个 task**：Step4 薄壳 **241.732 cm³**（v1 为两组 437.847 cm³），12 次合并的最大父体积比 0.7487；Step5 真实稀疏地脚 279.803 → **253.380 cm³**。全部 13 个 task 的 425,984 对原始载荷、35,437 个连续目标、原始面棱柱连续退出、实际接地面和薄壳直接贴地通过磁盘复验。作为对照，同一次运行合并前的最优两组（9+4）Step4 为 322.862 cm³、Step5 为 322.700 cm³，一组小 21.5%；两组方案同样完整跑过 Step5 与双臂视频，见 `data/B/step5_v3_two_set/`（`robot_use.mp4`）。

[Step4 图](output/B/step4/overview.png) · [Step5 图](output/B/step5/hulls_and_support.png) · [转台视频](output/B/step5/result.mp4) · [双臂使用视频](output/B/step5/robot_use.mp4) · [Step4 冻结状态](data/B/step4_v3_one_set/report.json) · [Step5 最终状态](data/B/step5_v3_one_set/report.json) · [Step5 独立复验](data/B/step5_v3_one_set/validation.json) · [新旧算法对照](data/B/step4_algorithm_v2_comparison.json)

双臂视频：**地面固定不动**，每个安装的切地平面就是工作台面。Franka Panda 负责装卸——把支撑从台面停放位装到工位、按每个 task 的保存安装转动空支撑（转角过大时放回台面换手再转）、沿保存的退出方向反向插入兔子、加工后沿退出方向取出；KUKA LBR Med 14 负责加工——针轴方向全程固定、只平移，每个安装先把针头平移到该安装的高度（一组方案比参考位置低 130–241 mm，两组方案 136–261 mm），沿工作区描边并填涂，装卸期间摆开。Step4/Step5 的证书只依赖相对位置（每个安装以自己的切地平面为局部 z=0，工具点随之平移），所以“地面不动、工作臂升降”与此前“针头不动、升降台”是同一组约束，支撑与体积不变。视频是保存几何上的运动学演示，不是整臂扫掠碰撞或动力学证书；工作臂与支撑、台面的采样干涉统计写在报告里。

运行入口不变：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step4.run \
  --workdir whole_task_2/data/B/step4_v3_families --rounds 10 --samples 32 --workers 20 --target-groups 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task_2.step5.grounded_feet \
  --workdir whole_task_2/data/B/step5_v3_one_set --workers 2
```

以下为 2026-10-19 及更早的说明，保留为历史。

当前 Step3 使用 **13 个独立的 5 mm 薄壳**初始化，总材料 **2062.312492 cm³**；连接 baseline 仅作对照。Step4 从这份初始化启动 32 个 sample，Juxtaposition 与 Rotational reuse 各半，基础梯度修复后验算整组。新的合并门槛为 **V(AB) ≤ 75% × (V(A)+V(B))**；合法子组跨 sample 保存，通过完整覆盖重新探索成员组合，避免早期分组锁死。**ground oracle** 保证整件支撑在每个安装中不穿地，并有同一个连通、避开工作区和连续退出的合法空间，可构造包住各 task 压力需求凸包的地脚。Step5 构造真实稀疏地脚、减材并复验。详见 [PROCESS.md](PROCESS.md)。

先前 **540.331 cm³** 的无地面解缺少这些约束，不能直接视为本轮合法成品；本轮冷启动状态位于 `data/B/step4_ground_membrane_families/`，发布以 `data/B/step4_active.json` 为准。可见结果保持 Step3 两张图、Step4 一张图，数据和证书放在 data 下。以下共用平板、垫高和旧 Step3.1 / Step3.2 的描述保留为历史。

当前结果为 **2 组，分别支持 9 个／4 个 task**。Step4 薄壳为 **437.847356 cm³**，冻结文件和代码未改。新的 Step5 按每个姿态的薄壳最低点定义接地平面，保持物理针头位置固定，在保留物体座位的前提下修复安装转角，再长出包住压力需求凸包的真实稀疏地脚。含脚材料 **514.200165 → 449.655425 cm³（−12.55%）**，两件分别 **353.281262／96.374163 cm³**；220 次删除提案中接受 56 次。全部 13 个 task 的 **425984 对原始载荷、35437 个连续目标**、工作区、原非凸连续退出、实际接地面和薄壳直接贴地检查通过磁盘复验。脚垫以非零面积共面三角片验收，单个贴地点不能代替脚垫。当前数据位于 `data/B/step5_ground_contact_final/`，旧固定地面和长脚结果仅作历史对照。

[Step4 图](output/B/step4/overview.png) · [Step5 唯一结果图](output/B/step5/hulls_and_support.png) · [16 秒转一圈视频](output/B/step5/result.mp4) · [最终状态](data/B/step5_ground_contact_final/report.json) · [独立复验](data/B/step5_ground_contact_final/validation.json)。图和视频显示保存的最终实体，周围为全部原始压力点与凸包；每组右侧只展示 4 个使用姿态，无文字和坐标轴。Step5 的可见 output 仅保留这张图和视频，没有旧结果子目录。

（历史，旧 Step1，输出已删除）旧 Step1 按用户确认的最简定义选面：**头部为球心、针长 100 mm 为半径、方向锥半角 25°，范围内的物体表面全部计入**。不再使用 270° 补洞、80–95 mm 壳层、法向角、首次可见性、头部间隙或连通分量裁剪。新的 Step1.2 自动选择 **20 个可重叠工作区域**，覆盖原始 **5,000 个面、100% 表面积**；前 **13 区覆盖 96.7878%**。当前任务采用前 **13 区（96.7878%）**；完整 20 区计划保留为 baseline。独立 Step2 已删除。

- [Step1.1：工具与球扇形](output/B/step1/step1.1/result.png)
- [Step1.3：四个区域与头部位置](output/B/step1/step1.3/working_areas_overview.png)：四组无文字配对图，统一粉色，金线显示 100 mm／25° 球扇形。
- [Step1.2：新划分总览](output/B/step1/step1.2/working_areas_overview.png)
- [Step1.2：累计覆盖曲线](output/B/step1/step1.2/coverage_progress.png)
- [覆盖计划](output/B/step1/step1.2/coverage_plan.json) · [独立复核](output/B/step1/step1.2/validation.json)

## Step1：只取球扇形与物体表面的交集

统一入口是 [step1/run.py](step1/run.py)，实现见 [run_needle.py](step1/run_needle.py) 与 [needle_geometry.py](step1/needle_geometry.py)。机器人仍取 [04 KUKA LBR Med 配置](../robots_config/04_kuka_lbr_med14_r820.json)，工具参数保存在 [needle_workspace.json](step1/needle_workspace.json)，并写入 Step1.1 有效配置快照 `working_tool`。原始机器人文件与物体网格保持原样。

令 `O` 为头部前端／针根位置，`a` 为名义针方向，`r=100 mm`、`alpha=25°`。选中曲面点的条件只有：

```text
||p - O|| <= r
(p - O) · a >= ||p - O|| * cos(alpha)
```

金线与判定共用同一球心、半径和锥角。这里的 25° 是从头部看向曲面点的**位置方向角**，不是表面法向与针的夹角。没有内球、没有 80 mm 最小距离，也不再把可用半径或锥角缩成 95 mm／23.75°。图中的头部正是球心，删除上一版额外引入的 20 mm 进给模型。图中 100 mm 针表示球扇形半径示意，不冒充已经规划的接触轨迹。

范围内的正面、背面、被其它表面遮住的面，都按同一几何条件计入；不按局部法向或可见性扣掉中间面片。一个球扇形碰到多个不连通曲面时，也保留为同一个工作区域。270° 补洞从当前算法和配置中移除，当前不调用 `fill_working_holes.py` 或 `radial_visibility.py`。

边界按现有 5,000 个源三角面保守离散：三个顶点都满足上述条件则保留整个面；横跨外边界的面暂不切碎。球和正向锥都是凸集，因此整个保留面都在范围内。这一离散只近似选区边缘，不会因法向或遮挡在内部挖洞。报告和独立检查明确记录该边界精度。

头部球与针的粗略尺寸仅用于绘图，不参与选面。候选姿态只整体排除“头部球心落入物体实体内部”的放置，不会据此删除某些面片。`--inset` 的默认 5% 仅用于候选瞄准深度，不改变实际 100 mm／25° 的选区范围；区域之间允许重叠。

## Step1.2：全表面覆盖 baseline

当前共有 **6,721 个自动候选**；另外追加的四个 Step1.3 耳朵／嘴部等展示候选不参与覆盖选择。候选按全部选中表面积计分，不再按入射余弦扣分。算法按真实三角面面积做贪心集合覆盖：每次选择新增覆盖面积最大的完整球扇形交集，直到每个源面都被覆盖，不预设区域数量。

结果为 **20 区、559.72 cm² 全覆盖**；10 区达到 90%，12 区达到 95%，16 区达到 99%，前 13 区为 **96.7878%**。完整区域面积之和约为总表面积的 **1.94 倍**。每个区域保留全部面，允许重叠；`first_covering_area_index` 只记录首次归属，不代表裁剪工作区。这是几何覆盖 baseline，尚未优化支撑、执行次序或最少工作次数。

`coverage_plan.json/.npz` 保存面集合、每次新增面积、候选编号、头部位置和参考物体 pose；`pose_coverage.json/.npz` 保存完整候选池及四个展示区域。[validate_needle.py](step1/validate_needle.py) 从磁盘重放覆盖选择，并独立验证**选中集合恰好等于球扇形内的完整源面集合**，同时检查没有漏选内部面或选入外部面。13 项几何与集合覆盖测试通过，覆盖包括旧内壳层、旧角度余量、背向面、被遮挡后层及多个不连通区域。

这一步只定义几何工作面，报告中的原生机械臂、针体／夹具碰撞与连续轨迹认证均为 false；原有下游证书没有重新绑定成新结果。

```bash
# 完整运行 Step1；输入或算法变化时自动重建候选缓存。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m whole_task_2.step1.run

# 独立复核磁盘上的球扇形选面、全覆盖及图片。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m whole_task_2.step1.validate_needle

.venv/bin/python -B -m unittest whole_task_2.step1.test_needle \
  whole_task_2.step1.test_coverage_plan
```

## Step3.1：同一块平板上的一排倒模

当前输入是前 13 个 task（W01–W13），不是 13 个预先固定的最终 pose。独立 Step2 已删除。Step3.1 从每个 task 在 **Step1.2 的参考工作姿态**出发，只允许绕原金锥轴旋转，并把兔子、对应头部及倒模一起平移。所有针头统一到 **220.677 mm** 高度，工作方向一致，沿 X 排开，物体之间互不接触。较低的工位在倒模下垫高 **0–134.069 mm**；不再强行让每只兔子的最低点都落在板面。粉色区域必须始终在原 100 mm／25° 金锥内，不能为了放稳而任意翻转物体。姿态后续仍可在合法范围内调整。

平板厚 **8 mm**。每个座位沿兔子的外形向外包覆 **5 mm**，再用对应高度的垫台接到同一块平板上，扣除兔子内部和这个 task 自己的工作禁区，形成负形倒模。13 个座位共用一块连续平板；只有高度差需要垫高。工作禁区仍是 Step1 的 100 mm／25° 球扇形，加现有针和头部代理的留空；保留原有 13 个工作面集合。

[Step3.1 总览图](data/B/step3_connected_output_legacy/step3.1/overview.png) 从上到下依次显示：

1. 13 只兔子按合法工作姿态放在垫高后的座位上；所有灰色头部等高，粉色为工作区，金线为各自的 Step1 球扇形。垫台采用实际支撑的一部分，已扣掉工作禁区。
2. 兔子就位后的仿形倒模，同时显示相同的金锥和头部位置。
3. 同一份实际支撑去掉兔子后的空模。

[逐 task 工作姿态图](data/B/step3_connected_output_legacy/step3.1/working_poses.png) 单独放大 13 个物体与金锥的关系，保留已确认的局部工作姿态作为参考，名义针方向全部一致；这张近景保留垫高前的参考板面，实际安装高度与垫台以 overview 为准。近景和总览的物体／头部相对姿态完全相同。这里仅画板面局部及物体，不画倒模，以便看清粉色选区。金锥表示 100 mm／25° 几何范围，不是已经规划的接触轨迹。

三排使用相同姿态、排布与相机；第三排只是隐藏兔子以展示内腔，**不表示已经存在退出路径**。Step3.1 保存开槽前的初始化；退出开槽和检查见下面的 Step3.2。

[报告](data/B/step3_connected_output_legacy/step3.1/report.json)、[独立复核](data/B/step3_connected_output_legacy/step3.1/validation.json)、[整体支撑模型](data/B/step3_connected_output_legacy/step3.1/support.obj) 与 `initialization.npz` 保存完整实体、13 个当前姿态、工作区及接触证据。工作机器人不用横跨这一排：这是初始化材料布局，执行某个 task 时将其座位移到工作工位；执行时针头高度保持统一，通过各座位底下的垫高补齐高度差。

入口 [step3/run.py](step3/run.py)，实现 [run_independent.py](step3/run_independent.py)、[independent_geometry.py](step3/independent_geometry.py)。力／力矩在当前 pose 下即时计算：作用力方向在机器人名义世界方向的 25° 锥内，幅值为 0–0.5 mg；关于质心的需求为 `(-g-P, -(q-COM)×P)`。从实际导出的整体支撑重新提取接触，检查每个 task 的 32,768 个原始载荷及连续力锥保守外包，不使用预先固定的 Step2 载荷表。

[validate_independent.py](step3/validate_independent.py) 独立重建全部接触、受力和工作禁区，逐面核对粉色集合与 Step1 完全相同，并检查每个粉色三角面的全部顶点在金锥内，同时检查共用平板、针头等高、垫台高度、互不相交、单一连通支撑及实际底面压力包围。布尔容差保持 `1e-10 m³`，受力容差保持 `2e-9`。接触模型沿用无摩擦、单侧、无幅值上限的法向反力，平板按刚性安装处理；未认证整臂轨迹或材料强度。

当前 **13 / 13 task 通过**，实际整体支撑通过 **425,984 / 425,984 原始载荷**与 **35,677 / 35,677 连续外包目标**。平板和全部倒模是同一个封闭连通实体，针头高度均为 220.677 mm；垫高后重新检查实际底面压力包围，不能沿用垫高前的地面力矩。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m whole_task_2.step3.run --workers 4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m whole_task_2.step3.validate_independent
```

## Step3.2：每个 task 独立退出，最终整体验收

入口 [run_exit.py](step3/run_exit.py)，参数见 [exit_policy.json](step3/exit_policy.json)，流程说明见 [PROCESS.md](PROCESS.md)。每个 task 独立选择上半球退出方向 `d_i` 与长度 `L_i`，不再强制所有 task 共用方向或位移。先尝试反工作方向，失败则优先做 2°、5°、10° 的小幅调整，再扩展到其它上半球方向。结果以第一个满足全部条件的初始化为准，暂不优化体积、占地或退出距离。

所有 Step3.1 的物体姿态、金锥、垫台和工作面保持。对各个 task，用整只原始非凸物体沿 `p_i(t)=p_i(0)+t*L_i*d_i, 0≤t≤1` 连续平移的扫掠从支撑中减材。扫掠由初始实体及全部正向源三角面的平移棱柱构成，保留凹处可用接触，不使用整个兔子的凸包或离散姿态替代连续路径。工作头先撤开，再退出物体；终点整只物体高于最终支撑至少 15 mm。

每个候选挖槽后，从实际局部支撑重新提取接触，检查全部配对载荷。局部通过后，把全部退出槽合并扣除，再对**最终导出的实际整体支撑重新检查所有 task**：连续退出无碰撞、工作面和工作空间留空、物体内部留空、支撑封闭且接地连通、没有增加 Step3.1 外的材料、全部力／力矩要求以及实际底面压力中心均通过。某个 task 失败就调整该 task 的方向、重建整体并复验所有 task；局部通过不能代替整体验收。实现见 [exit_search.py](step3/exit_search.py)。

浮点布尔残差仅通过重复扣除原来的禁区和原始面棱柱处理，不放宽容差；保留接地外边界并扣除其内部空腔，避免丢弃浮块后遗留孤立负边界。连续碰撞检查逐个重建原始面棱柱，其交叠体积之和为整个连续扫掠交叠的保守上界，每个 task 不超过 `1e-10 m³`。最终载荷全部从当前实际接触重算，不沿用开槽前证书。

[Step3.2 总览图](data/B/step3_connected_output_legacy/step3.2/overview.png) 从上到下显示开槽后就位、空支撑与各 task 自己的退出箭头、全部物体完全退出。实际执行逐个 task 进行。各个方向可以恰好相同，但不存在强制相同的约束。

[独立复验](data/B/step3_connected_output_legacy/step3.2/validation.json) 从磁盘重建几何与接触，只有全部通过才发布 [报告](data/B/step3_connected_output_legacy/step3.2/report.json)、[完整支撑](data/B/step3_connected_output_legacy/step3.2/support.obj)、`initialization.npz` 与 [搜索记录](data/B/step3_connected_output_legacy/step3.2/search_trace.json)。审查包含每个 task 的 32,768 对原始载荷及连续力锥保守外包目标。力仍在机器人名义工作方向的 25° 锥内，关于质心的力／力矩配对不变。合法性范围沿用 Step3.1 的几何与准静态接触模型。

本轮 13 / 13 task 的最终整体审查与磁盘独立复验全部通过：425,984 / 425,984 对原始载荷、35,677 / 35,677 个连续力锥外包目标。13 个 task 均独立接受初始反工作方向，无需微调；各自退出长度为 247.453–437.054 mm。37 项相关数学、几何与失败后重试测试通过。当前是合法、尚未优化的初始化。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m whole_task_2.step3.run_exit

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m whole_task_2.step3.validate_exit
```

## 后续阶段

Step4 当前入口是 [step4/run.py](step4/run.py)，从 Step3.2 的 initialization 开始，每个 task 初始单独成组。随机选择两个组，用 Juxtaposition 或 Rotational reuse 提议合并，再用三个基本操作的梯度修复；只重新验收新组，其他组在几何保持条件下沿用已有证书。失败保留此前合法结果。操作定义、视频和运行方式见 [PROCESS.md](PROCESS.md)。可见的 `output/B/step4/` 只保留一张 `overview.png`，新算法的完整状态与磁盘复验存入 `data/B/step4_group_merges/`；`data/B/step4_active.json` 指定当前已发布的合法结果。

旧 Step4 的其它实验入口仍是历史模型，不是当前默认流程。这些历史优化入口与当前 Step5 分开；当前成品见文首链接。旧视频曾发现工作臂与支撑碰撞，不作为整臂避碰证书。

此前已发布的 Step4 结果：实际体积 **48,517.432 → 4,437.642 cm³（−90.85%）**，比上一版水平结果 **6,284.111 cm³** 再小 **29.38%**。最终 **W04 平放、W09 将同一整件支撑侧立 90°**，两者在支撑内的物体座位完全重合，并共用材料坐标中的退出方向。18 次接受的修改均严格减小体积；最终 13 / 13 task 的原始载荷、连续力锥、实际底面、工作范围和完整退出通过独立磁盘复验。28 项操作与验收测试通过。[唯一总览图](output/B/step4/overview.png) 展示空支撑、整件侧立安装及全部实际用法；[Rotational reuse 演示](operations/rotational_reuse.mp4) 改用 W06 与 W08，展示同一件从头到尾形状不变的双窗口薄壳依次支持两个工作区域：针头固定，取出物体后转动空壳，再按第二个 task 的 pose 放回；最终设计的完整证据在 `data/B/step4_standing/`。

`output/B/step3/continuous13` 与 `output/B/step4/continuous13` 是此前单朝向方案的历史输出，已按要求删除，当前不再保留这些历史对照目录。

当前 Step4 默认算法使用 32 条从 Step3 initialization 开始的持续分支，每轮 16 次 Juxtaposition、16 次 Rotational reuse。均匀随机选两个组及组内代表 task，移除距离评分与距离分组。一次成功合并减少一个独立组：`ABC + DEF → ABCDEF`，完整验收 ABCDEF，未参与组的证书在几何不受影响时复用。随后继续随机选组；不使用 PSO。每条分支相对自己的父状态下降，另存全局最优。实现见 [group_acceptance.py](step4/group_acceptance.py)、[population_search.py](step4/population_search.py) 和 [gradient_repair.py](step4/gradient_repair.py)，运行 `python -m whole_task_2.step4.run --rounds 32 --samples 32 --repair-rounds 1 --workers 4`。

此前从旧 Step4 热启动的实验发布了 **3,920.704 cm³** 的合法解，比 **4,437.642 cm³** 减少 **11.65%**。冻结证据在 `data/B/step4_pose_neighbours_best/`，当时的 13 个 task 全部通过磁盘重放。当前从 Step3 重启的分组算法已完成首个合法合并：W02/W09，13 → 12 组，**48,517.432 → 44,463.462 cm³**，这条分支的 13 个 task 通过独立磁盘复验，65 项相关测试通过；首步冻结证据在 `data/B/step4_group_merges_first_verified/`。32 分支继续搜索，当前可见图仍保留此前较小的合法结果；发布状态以 `data/B/step4_active.json` 为准。
