# Whole task：正方体工作窗口、姿态与支撑

## 当前版本：固定六 task 预览

先用 ±X、±Y、±Z 六种落桌姿态，最终 task 数量另行求解。报告保持 `task_count_mode = fixed_six_pose_preview_not_optimized`，不把六个输入姿态当作已优化的最终数量。

## 工作空间：边长取物体最长边的一半

工作空间为正方体。计算 +X 参考物体的世界 AABB，取最长边的一半作为三轴共同边长：

```text
object_size = object_AABB.max - object_AABB.min
cube_edge = 0.5 * max(object_size.x, object_size.y, object_size.z)
work_size = (cube_edge, cube_edge, cube_edge)
c = center_of_robot_facing_object_face
n = outward_normal_of_that_face
work_center = c - 0.5 * cube_edge * n
bounds = (work_center - 0.5 * work_size, work_center + 0.5 * work_size)
```

参考物体世界 AABB 尺寸为 119.086 × 154.786 × 153.007 mm，因此当前工作空间为 **77.393 × 77.393 × 77.393 mm 的正方体**。靠机械臂的面仍与物体框 +Y 面共中心，正方体由此朝物体内部延伸。三轴采用同一边长；对于窄物体，正方体允许超出物体 AABB 的较窄轴范围。六个姿态共用固定世界工作窗口。

| 轴 | 最小值 mm | 最大值 mm |
|---|---:|---:|
| X | -38.697 | 38.697 |
| Y | -710.000 | -632.607 |
| Z | 260.414 | 337.807 |

**可加工面 = 给定角度可达的物体表面 ∩ 正方体内的物体表面。** 位置限制作用在表面加工点／针尖 TCP，头部和针根可在框外。针长 100 mm、针半径 2 mm、头半径 20 mm，名义方向 `(0, −0.7071, −0.7071)`，方向锥半角 25°。射线、针、头、IK、关节、连杆和桌面避碰均重新检查。

当前仍按原始三角面保守离散：三个顶点与质心都在正方体内，且该面认证通过才计入可加工区域。跨框三角面暂不裁切；不补入未认证面，不丢弃可加工小块。工作框以三维线段绘制，物体遮住后方棱线；参考物体大框不画。

参数见 [reach_policy.json](step1/reach_policy.json)，工位与正方体数据见 [setup.json](output/B/step1/step1.1/setup.json)。

## 重算结果

兔子总表面积 **559.72 cm²**。本轮六个实际 task 共 **2,456 面、288.57 cm²，占 51.56%**；此前为 41.49%。重复分配面数和面积均为 0，全部已分配面均属于所属姿态的认证交集。

| task | 完整可加工面积 | 去重后实际面积 | 实际面数 |
|---|---:|---:|---:|
| +X | 75.54 cm²（13.50%） | 46.34 cm²（8.28%） | 384 |
| +Y | 45.12 cm²（8.06%） | 28.77 cm²（5.14%） | 277 |
| +Z | 66.25 cm²（11.84%） | 55.96 cm²（10.00%） | 490 |
| -X | 67.90 cm²（12.13%） | 44.36 cm²（7.92%） | 354 |
| -Y | 85.15 cm²（15.21%） | 73.45 cm²（13.12%） | 631 |
| -Z | 48.40 cm²（8.65%） | 39.70 cm²（7.09%） | 320 |

`reach`／`task_reach` 是各姿态完整候选区 `R_i`，候选间允许重叠；`face_task` 和报告 `face_ids` 是下游实际使用的互斥任务 `A_i`。所有当前可加工交集的并集均已分配。

## 图、视频与验收

- [find_tasks：完整可加工候选区与正方体](idea/find_tasks.png) · [针沿三维边界描一圈](idea/find_tasks.mp4)
- [Step1.1：工位](output/B/step1/step1.1/station.png)
- [Step1.2：互斥分区](output/B/step1/step1.2/areas.png) · [逐 task](output/B/step1/step1.2/overview.png)
- [Step1.3：放大的六个 working poses](output/B/step1/step1.3/working_poses.png)
- [已分配](output/B/step1/step1.3/covered.png) · [未分配](output/B/step1/step1.3/uncovered.png) · [面积数据](output/B/step1/step1.3/coverage.json)
- [计划](data/B/step1_reach/plan.json) · [磁盘复验](data/B/step1_reach/validation.json) · [三维框独立核对](data/B/step1_reach/box_3d_audit.json)

**3,293 组面片／姿态通过独立复验，54 项相关测试通过。** 验收检查三轴等长、边长为参考物体最长边的 1/2、共面中心与向内位置、逐姿态 XYZ 掩码、认证轴、实际 FK 加工点和互斥分配。

find_tasks 展示 +X 完整候选区：**626 面、13.50%**。针在 10 秒、240 帧内沿完整三维边界闭合描边，覆盖全部 82 条边、433.01 mm；顶点相接的边界分支也全部保留并描过。每帧重新检查表面首次命中、针／头、方向锥、正方体内加工点、IK 和 KUKA。working poses 按完整物体与正方体取景，显示去重后的实际任务。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m whole_task.step1.run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m whole_task.step1.find_tasks_figure
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m whole_task.step1.find_tasks_video
```

本轮按要求重跑 Step1。旧 Step1 图和结果已替换，旧 keepout 图已清理；Step2–5 的数值结果仍基于此前长方体，尚未按正方体重建。`whole_task_2/` 保留另一份实验状态。旧版 115 张图的清理记录见 [清单](data/figure_cleanup_20261021.json)。

---

# 以下为旧输入下的 Step2–5 历史记录

## Step2 与 Step3 初始化：6 个 task 共用一块 base（2026-10-21）

Step3 直接读取六个输入姿态 `+X、+Y、+Z、−X、−Y、−Z` 各自的 `task_reach`，与 Step1.3 的粉色区域一致。区域允许重叠，六个 task 全部保留；不使用 Step1.2 的不重复面分区来删除姿态。每个 task 的针轴和 KUKA 关节角来自该姿态自己的 `reach` 认证，包围补入面参与受力，但不因此获得工具可达认证。

每个 task 生成一个 5 mm 薄壳座位，六个座位用脚和同一块水平底板连成**一个完整 base**。主图上排展示六只兔子分别放在六个座位上，粉色为对应 task 区域；下排展示同一块空 base。图中没有文字。相邻座位间隔 8 cm，以避开相邻座位对腕部的干涉；六件薄壳合计 861.925 cm³，完整 base 为 2,162.715 cm³。整块 base 在连接、序列化及必要的同禁区重切后，重新检查成对受力、工具、KUKA 和连续退出；不放宽验收容差。

Step2 导出这六个姿态在 Step3 中实际使用的成对力／力矩，每个 task 保存 32,768 个样本，总计 196,608 对，另存 25,542 个连续力锥外包目标。所需反力和力矩为 `[-g-P, -(q-COM)×P]`，关于当前世界坐标系中的质心计算；完整关联关系保存在数组中。

- Step3 新初始化：一个 6-task base（旧图已删除） · [base.obj](output/B/step3/base.obj)
- [Step3 报告](data/B/step3_tool_membranes/report.json) · [独立磁盘复验](data/B/step3_tool_membranes/validation.json)
- Step2：力与力矩图（旧图已删除） · [受力域与数组索引](data/B/step2/report.json) · [成对映射复验](data/B/step2/validation.json)

旧版 Step3 图已删除，发布目录只保留这张新的完整 base 图。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task.step3.run --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task.step3.tool_membranes --validate-only
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python whole_task/data/B/step2/export_loads.py
```

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

- [整件](output/B/step5/joined.png)：上排是每个座位放着兔子（粉色是该座位两个 task 的区域），下排只有支撑。· 使用（旧图已删除）：6 个 task 各一幅，支撑转好、滑到工位，KUKA 用认证过的关节角夹着针，停在该 task 中间的认证面上。按"机械臂跟着兔子升降"的约定，在认证高度画。
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
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task.step4.keepout          # 禁区
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task.step4.seats --workers 16  # 转角 + 座位
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task.step4.join --workers 20   # 选组合 + 排座位（--rho）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task.step5.joined             # 连成一整件 + 验收 + 复验 + 出图
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task.step5.joined_video       # 使用视频（逐帧检查）
```

下面"当前结果（Step4 v3）"及以后是旧的 13 个锥形工作区的结果，保留为历史；`output/B/step4/overview.png` 和 `output/B/step5/` 里的 `hulls_and_support.png`、`result.mp4`、`robot_use.mp4` 也是旧的。

## 当前结果（2026-10-20，Step4 v3）

Step1–3 不变：13 个工作区域，13 个独立的 5 mm 薄壳初始化，共 **2062.312 cm³**。Step4 v3 只有一种高阶操作——**把一件已验收的支撑刚体地放到另一件上，保留合法且省料的并集**：Rotational reuse 在 SO(3) 中采样整件旋转，Juxtaposition 只平移；用 LP 让所有安装地面相容，在真实构造上精确评分并局部细化，再完整验收；合并门槛仍为 **V(AB) ≤ 0.75×(V(A)+V(B))**。ground oracle 与 Step5 使用同一个落地模型（薄壳切平面落地、保持座位的绕针轴转角、同一个地脚可用空间函数），所以 Step4 接受的支撑都能被 Step5 落地。步长、重叠和网格尺度都相对物体尺寸给出。详见 [PROCESS.md](PROCESS.md)。

从 Step3 冷启动约 77 分钟，结果为 **一件支撑支持全部 13 个 task**：Step4 薄壳 **241.732 cm³**（v1 为两组 437.847 cm³），12 次合并的最大父体积比 0.7487；Step5 真实稀疏地脚 279.803 → **253.380 cm³**。全部 13 个 task 的 425,984 对原始载荷、35,437 个连续目标、原始面棱柱连续退出、实际接地面和薄壳直接贴地通过磁盘复验。作为对照，同一次运行合并前的最优两组（9+4）Step4 为 322.862 cm³、Step5 为 322.700 cm³，一组小 21.5%；两组方案同样完整跑过 Step5 与双臂视频，见 `data/B/step5_v3_two_set/`（`robot_use.mp4`）。

[Step4 图](output/B/step4/overview.png) · [Step5 图](output/B/step5/hulls_and_support.png) · [转台视频](output/B/step5/result.mp4) · [双臂使用视频](output/B/step5/robot_use.mp4) · [Step4 冻结状态](data/B/step4_v3_one_set/report.json) · [Step5 最终状态](data/B/step5_v3_one_set/report.json) · [Step5 独立复验](data/B/step5_v3_one_set/validation.json) · [新旧算法对照](data/B/step4_algorithm_v2_comparison.json)

双臂视频：**地面固定不动**，每个安装的切地平面就是工作台面。Franka Panda 负责装卸——把支撑从台面停放位装到工位、按每个 task 的保存安装转动空支撑（转角过大时放回台面换手再转）、沿保存的退出方向反向插入兔子、加工后沿退出方向取出；KUKA LBR Med 14 负责加工——针轴方向全程固定、只平移，每个安装先把针头平移到该安装的高度（一组方案比参考位置低 130–241 mm，两组方案 136–261 mm），沿工作区描边并填涂，装卸期间摆开。Step4/Step5 的证书只依赖相对位置（每个安装以自己的切地平面为局部 z=0，工具点随之平移），所以“地面不动、工作臂升降”与此前“针头不动、升降台”是同一组约束，支撑与体积不变。视频是保存几何上的运动学演示，不是整臂扫掠碰撞或动力学证书；工作臂与支撑、台面的采样干涉统计写在报告里。

运行入口不变：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task.step4.run \
  --workdir whole_task/data/B/step4_v3_families --rounds 10 --samples 32 --workers 20 --target-groups 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m whole_task.step5.grounded_feet \
  --workdir whole_task/data/B/step5_v3_one_set --workers 2
```

以下为 2026-10-19 及更早的说明，保留为历史。

当前 Step3 使用 **13 个独立的 5 mm 薄壳**初始化，总材料 **2062.312492 cm³**；连接 baseline 仅作对照。Step4 从这份初始化启动 32 个 sample，Juxtaposition 与 Rotational reuse 各半，基础梯度修复后验算整组。新的合并门槛为 **V(AB) ≤ 75% × (V(A)+V(B))**；合法子组跨 sample 保存，通过完整覆盖重新探索成员组合，避免早期分组锁死。**ground oracle** 保证整件支撑在每个安装中不穿地，并有同一个连通、避开工作区和连续退出的合法空间，可构造包住各 task 压力需求凸包的地脚。Step5 构造真实稀疏地脚、减材并复验。详见 [PROCESS.md](PROCESS.md)。

先前 **540.331 cm³** 的无地面解缺少这些约束，不能直接视为本轮合法成品；本轮冷启动状态位于 `data/B/step4_ground_membrane_families/`，发布以 `data/B/step4_active.json` 为准。可见结果保持 Step3 两张图、Step4 一张图，数据和证书放在 data 下。以下共用平板、垫高和旧 Step3.1 / Step3.2 的描述保留为历史。

当前结果为 **2 组，分别支持 9 个／4 个 task**。Step4 薄壳为 **437.847356 cm³**，冻结文件和代码未改。新的 Step5 按每个姿态的薄壳最低点定义接地平面，保持物理针头位置固定，在保留物体座位的前提下修复安装转角，再长出包住压力需求凸包的真实稀疏地脚。含脚材料 **514.200165 → 449.655425 cm³（−12.55%）**，两件分别 **353.281262／96.374163 cm³**；220 次删除提案中接受 56 次。全部 13 个 task 的 **425984 对原始载荷、35437 个连续目标**、工作区、原非凸连续退出、实际接地面和薄壳直接贴地检查通过磁盘复验。脚垫以非零面积共面三角片验收，单个贴地点不能代替脚垫。当前数据位于 `data/B/step5_ground_contact_final/`，旧固定地面和长脚结果仅作历史对照。

[Step4 图](output/B/step4/overview.png) · [Step5 唯一结果图](output/B/step5/hulls_and_support.png) · [16 秒转一圈视频](output/B/step5/result.mp4) · [最终状态](data/B/step5_ground_contact_final/report.json) · [独立复验](data/B/step5_ground_contact_final/validation.json)。图和视频显示保存的最终实体，周围为全部原始压力点与凸包；每组右侧只展示 4 个使用姿态，无文字和坐标轴。Step5 的可见 output 仅保留这张图和视频，没有旧结果子目录。

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
  .venv/bin/python -B -m whole_task.step3.run --workers 4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m whole_task.step3.validate_independent
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
  .venv/bin/python -B -m whole_task.step3.run_exit

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m whole_task.step3.validate_exit
```

## 后续阶段

Step4 当前入口是 [step4/run.py](step4/run.py)，从 Step3.2 的 initialization 开始，每个 task 初始单独成组。随机选择两个组，用 Juxtaposition 或 Rotational reuse 提议合并，再用三个基本操作的梯度修复；只重新验收新组，其他组在几何保持条件下沿用已有证书。失败保留此前合法结果。操作定义、视频和运行方式见 [PROCESS.md](PROCESS.md)。可见的 `output/B/step4/` 只保留一张 `overview.png`，新算法的完整状态与磁盘复验存入 `data/B/step4_group_merges/`；`data/B/step4_active.json` 指定当前已发布的合法结果。

旧 Step4 的其它实验入口仍是历史模型，不是当前默认流程。这些历史优化入口与当前 Step5 分开；当前成品见文首链接。旧视频曾发现工作臂与支撑碰撞，不作为整臂避碰证书。

此前已发布的 Step4 结果：实际体积 **48,517.432 → 4,437.642 cm³（−90.85%）**，比上一版水平结果 **6,284.111 cm³** 再小 **29.38%**。最终 **W04 平放、W09 将同一整件支撑侧立 90°**，两者在支撑内的物体座位完全重合，并共用材料坐标中的退出方向。18 次接受的修改均严格减小体积；最终 13 / 13 task 的原始载荷、连续力锥、实际底面、工作范围和完整退出通过独立磁盘复验。28 项操作与验收测试通过。[唯一总览图](output/B/step4/overview.png) 展示空支撑、整件侧立安装及全部实际用法；[Rotational reuse 演示](operations/rotational_reuse.mp4) 改用 W06 与 W08，展示同一件从头到尾形状不变的双窗口薄壳依次支持两个工作区域：针头固定，取出物体后转动空壳，再按第二个 task 的 pose 放回；最终设计的完整证据在 `data/B/step4_standing/`。

`output/B/step3/continuous13` 与 `output/B/step4/continuous13` 是此前单朝向方案的历史输出，已按要求删除，当前不再保留这些历史对照目录。

当前 Step4 默认算法使用 32 条从 Step3 initialization 开始的持续分支，每轮 16 次 Juxtaposition、16 次 Rotational reuse。均匀随机选两个组及组内代表 task，移除距离评分与距离分组。一次成功合并减少一个独立组：`ABC + DEF → ABCDEF`，完整验收 ABCDEF，未参与组的证书在几何不受影响时复用。随后继续随机选组；不使用 PSO。每条分支相对自己的父状态下降，另存全局最优。实现见 [group_acceptance.py](step4/group_acceptance.py)、[population_search.py](step4/population_search.py) 和 [gradient_repair.py](step4/gradient_repair.py)，运行 `python -m whole_task.step4.run --rounds 32 --samples 32 --repair-rounds 1 --workers 4`。

此前从旧 Step4 热启动的实验发布了 **3,920.704 cm³** 的合法解，比 **4,437.642 cm³** 减少 **11.65%**。冻结证据在 `data/B/step4_pose_neighbours_best/`，当时的 13 个 task 全部通过磁盘重放。当前从 Step3 重启的分组算法已完成首个合法合并：W02/W09，13 → 12 组，**48,517.432 → 44,463.462 cm³**，这条分支的 13 个 task 通过独立磁盘复验，65 项相关测试通过；首步冻结证据在 `data/B/step4_group_merges_first_verified/`。32 分支继续搜索，当前可见图仍保留此前较小的合法结果；发布状态以 `data/B/step4_active.json` 为准。


## 此前长方体工作框下的 keepout（历史）

旧 keepout 图已清理。以下数值来自此前长方体工作框，尚未按本次正方体重算。当时按 **1,970 个互斥认证面、7,880 组保存关节角**重算。输入直接来自已复验的 Step1 `face_task`、每个 task 自己的 pose、针轴与关节角；不读取旧 Step3 初始化。表面点在向内工作框内，工具、针根和连杆禁区可延伸到框外。

针与头沿每个面的认证轴构造；KUKA 逐连杆按连杆中心网格分组取碰撞凸包。网格边长仍为物体对角线的 12%，近重复点合并网格为 1 µm，包含性容差为 2 µm。本次最大包含性残差 **0.833 µm**；六个 task 的 KUKA 禁区与兔子及外侧 5 mm 薄层交叠均为 0。

| task | 认证面 | 保存姿态 | KUKA 凸体 | 针 + 头 cm³ | KUKA cm³ | 合计 cm³ |
|---|---:|---:|---:|---:|---:|---:|
| +X | 352 | 1,408 | 146 | 5,183.5 | 58,767.3 | 59,517.8 |
| +Y | 264 | 1,056 | 187 | 6,880.2 | 67,227.6 | 67,841.2 |
| +Z | 375 | 1,500 | 117 | 4,748.6 | 56,072.6 | 56,865.6 |
| -X | 297 | 1,188 | 333 | 10,510.3 | 88,837.7 | 89,690.3 |
| -Y | 476 | 1,904 | 272 | 9,923.0 | 77,356.9 | 78,347.8 |
| -Z | 206 | 824 | 123 | 4,773.1 | 63,983.4 | 64,582.5 |

独立磁盘复验从当前 Step1 关节角重建针／头、KUKA 和组合实体，与存盘几何逐项比较对称差，通过 `1e-10 m³` 体积容差。图中蓝色为固定工作框，橙色为针／头禁区，紫色为 KUKA 禁区，禁区以 45% 不透明度叠加。头部延伸仅在显示时缩至 70%，完整数据与验收使用原尺寸。这里只覆盖保存的加工姿态。

[数据报告](data/B/step4_keepout/report.json) · [独立复验](data/B/step4_keepout/validation.json) · [图像输入记录](data/B/step4_keepout/figure.json)。当时 52 项相关测试通过。Step2、Step3 支撑初始化、Step4 座位搜索／合并和 Step5 的数值结果尚未按此版本重建，下面仅为历史记录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m whole_task.step1.reach_render
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m whole_task.step4.keepout --workers 6
```

