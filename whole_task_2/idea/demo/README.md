# 实机 demo 交接：工位与两个臂

实测 SO101 固定桌面、人工换位的版本见 [REAL_STATION_SO101.md](REAL_STATION_SO101.md)。该版本使用独立实机 planner，不修改 Step1–Step5 或现有 fixture；下文的升降立柱与 Franka 属于原视频工位。

2026-10-25。本文件只写 setup、工作臂和搬运臂，以当前代码为准，交给写实机 demo 的 agent。`whole_task_2/README.md`、`PROCESS.md` 和 `robots_config/README.md` 的其余内容停在 10-16 至 10-24，没有更新。

分工：工作臂（KUKA 或 SO101）在固定工位上用针加工；搬运臂（Franka Panda）在任务之间搬支撑和兔子。支撑是被动的：自由放在桌上，兔子放进座位，没有销钉、夹具、真空或磁铁。搬运臂不是方法的贡献，但实机需要一个通用的 pick-and-place planner。两臂轮流动，搬运臂绝不碰工作臂。

## 1. 工位（`robots_config/`）

一个工位是一份 `robots_config/random/NN_<model>.json`（schema 1，由 `robots_config.constraints.load_config` 读取）。`robots_config/draw.py` 抽取工位，型号常量在 `robots_config/models/`。

| 字段 | 含义 |
|---|---|
| `task_constraint.working_direction_world` | 名义针方向 a（工具→表面），在北半球随机抽取，针从上方来 |
| `task_constraint.cone_half_angle_deg` | 针相对 a 的倾斜锥半角 |
| `task_constraint.surface_workspace.edge_m` | 工作正方体边长，加工点（针尖）必须在内；`null` 表示取物体最长边的一半 |
| `task_constraint.world_position_constraint.center_world_m` | 工作窗口中心的初值，Step1 从这里开始找工作点 |
| `feasibility.base_pose_world` | 工作臂底座在世界原点，+X 朝工作窗口 |
| `feasibility.arm_module` | 工作臂模块（第 2 节） |
| `feasibility.tcp_transform` | 法兰→针尖 0.17 m |

| 工位 | 工作臂 | a | 锥 | 正方体 | 用到它的 run |
|---|---|---|---:|---:|---|
| 01_kuka | KUKA | [−0.604, −0.754, −0.260] | 15° | 不限 | – |
| 02_kuka | KUKA | [0.644, −0.171, −0.746] | 15° | 75 mm | – |
| 03_kuka | KUKA | [−0.195, 0.916, −0.350] | 35° | 75 mm | B+03_kuka |
| 04_kuka | KUKA | [0.047, 0.960, −0.274] | 25° | 75 mm | B+04_kuka（只有 logs） |
| 05_so101 | SO101 | [0.601, −0.495, −0.627] | 15° | 150 mm | – |
| 06_so101 | SO101 | [0.282, −0.648, −0.707] | 15° | 100 mm | B+8pose、B+9pose |
| 07_kuka | KUKA | [0.683, −0.182, −0.707] | 20° | 75 mm | whole_shape（`step3/big_init.py` 的默认工位） |
| 08_kuka | KUKA | [0.966, −0.257, 0] | 20° | 75 mm | – |
| 09_kuka | KUKA | [0.908, −0.242, −0.342] | 20° | 75 mm | – |

有 Step5 合格支撑、可以上实机的 run（Step5 均于 2026-10-25 19:37 重算）：

| run | 工位 | task | Step5 passed | 入口 |
|---|---|---|---|---|
| B+03_kuka | 03_kuka | 6（±X、±Y、±Z） | r00 | `python -m whole_task_2.run_station B+03_kuka` |
| B+8pose | 06_so101 | 8（S1–S8） | r00、r01、r02 | `python -m whole_task_2.run_poses --poses 8 --config robots_config/random/06_so101.json` |
| B+9pose | 06_so101 | 9（S1–S9） | r00 | 同上，`--poses 9` |

**工作点与桌面**（Step1，`step1/plan_reach.py`、`step1/reach_video.py:201`）：

- 工作点：从窗口中心沿底座→窗口的水平方向每 1 cm 往前移，取第一个满足条件的点：工作臂在点周围半个正方体内的 60 组随机针尖、锥内方向都能求出 IK。B+9pose 的工作点是 [0.152, −0.348, 0.230] m。
- 桌面：高度 = 工作点 z − 0.5 × 物体最长边，范围是工作点周围 0.5 × 0.5 m。这是物体直接落桌的认证姿态。加上支撑以后，每个任务的支撑落地高度不同，见 2.4。
- `robots_config/place.py`（舒适工作点：先比能完成的工作比例，再比余量）已经写好，但从未 `--write`。九个配置都没有 `feasibility.placement` 记录，现有 run 用的都是上面的旧规则。旧规则选在可达边界上，B+03 的 KUKA 肘关节 J4 停在 −106° 至 −120° 附近，针沿轴只能退 2 至 6 cm。`--write` 会改动已完成 run 的输入，执行前要先问用户。

## 2. 工作臂

### 2.1 两个模块

| | KUKA LBR Med 14 R820 | SO101 |
|---|---|---|
| 模块 | `whole_task_2/step5/video/kuka.py` | `whole_task_2/step5/video/so101.py` |
| 轴数 | 7 | 5（shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll；夹爪固定闭合） |
| 模型 | `codes/simulation/kuka.py` 与 `assets/lbr_med14_r820`（LBR-Stack 模型限位） | `codes/simulation/assets/so101/so101_new_calib.urdf`（Apache-2.0） |
| 针的安装 | 沿法兰 +Z，法兰在针根后 70 mm | 像握笔一样横夹在夹爪里：沿 gripper 坐标 y 轴、垂直于 wrist roll，针根距 gripper 原点 70 mm |
| 关节速度 | 模块自带 | 模块没有；视频按 180°/s 计（STS3215 空载约 270°/s 的 2/3） |

接口 `Arm(base, rotation, tool)`，其中 tool = 针长 + 0.07 = 0.17 m：

- `forward(q, geometry=False)`
- `solve(point, outward, previous)`：outward 是针方向取反，结果保持在关节限位内
- `lower` / `upper`：关节限位
- `collisions`：每个连杆碰撞网格的顶点
- `visuals()`

Step1、Step3、Step4 用 7 关节接口，SO101 由 `run_poses.Padded` 在后面补零；视频里 SO101 不补零。

### 2.2 工具

`step1/needle_workspace.json`：

- 针长 100 mm、半径 2 mm（Walther PILOT GA 1030 Extrusion 的 100 mm 延长）；
- 头部代理半径 20 mm（视频画成长 70 mm、半径 20 mm 的圆柱）；
- 接触加工，离面距离 0。

### 2.3 认证了什么

**Step1**（`step1/arm.py`、`step1/reach.py`）：每个面取 4 个测试点（3 个顶点和质心），针尖沿该面的认证轴落在测试点上，并满足：

- IK 有解，在关节限位内，且高于桌面；
- 每个连杆碰撞网格的凸包都不碰物体三角形（精确测试）；
- 针线不穿物体，针身和头部无碰撞。

关节角存盘，验收只用正运动学重放。

**Step3 至 Step5**：再审工作臂连杆与支撑的碰撞，真机包里是 `arm_link_faces_hit`。

这些认证都是静止的触点，不是连续路径。连续路径只在视频里逐帧检查过（2.5）。

### 2.4 底座高度：视频用了升降立柱，真机没有

Step4 让多个任务共用座位，支撑装好后兔子的高度和 Step1 的认证姿态不同。为了保持认证时的相对位姿，视频（`step5/joined_video.py` 的 `Plan`）让桌子不动、工作臂底座在立柱上竖直移动 `off[k] = Z0 − lift[k]`：

- 作业时的底座偏移，B+9pose 是 −38.4 至 +22.7 mm（跨度 61 mm），B+03 是 −18.4 至 +6.1 mm。这两组数取自 10-24 的视频缓存；Step5 已于 10-25 重算，现值以重新导出的真机包为准。
- 换任务时的"升高"（B+9pose 是 0.050 m）也是立柱抬升，不是臂自己的动作。
- 真机包里，同一件事表现为每个任务的 `tasks[k].support_lowest_z_mm` 不同；这里工作臂底座固定在世界原点。
- `info.json` 的 `station.table_top_z_mm` 是 Step1 物体直接落桌的高度，不是支撑底下的桌面。

真机底座固定，有两种做法，需要和用户确定：

- 每个任务按 `support_lowest_z_mm` 垫块；
- 固定桌高，为每个任务重新做 IK 和连杆碰撞认证（可能丢覆盖）。

### 2.5 视频里的工作臂运动

代码是 `step5/joined_video.py` 的 `Plan` 和 `idea/demo/code/video.py` 的 `Demo`：

- **停点：** 每个认证面的质心，沿该面的认证轴。关节逐帧连续跟踪：先用 `J.track`，不行再用 `Arm.solve`，单步最多变化 10°。
- **每帧检查：**
  - 连杆凸包对支撑、兔子、桌面，用精确三角形测试；
  - 针和头（外接棱柱）对支撑和桌面；
  - 针对兔子，按 Step1 的规则。
- **离开停点：** 先沿针直退 12、6 或 0 mm，再偏转 45° 至 75°。选第一个能走 8 cm 的方向。
- **换任务：**
  1. 退针，距离从 0.3 倍针长试到 2.4 倍。
  2. 升高。
  3. 绕 J1 转开 25° 至 105°，直到连杆和工具离 Panda 的工作区至少 0.1 m。工作区是支撑各落脚、各出口末端的兔子、休息点和 Panda 底座的水平凸包。
  4. 等 Panda 搬完。
  5. 转回、下降到下一个任务的第一个停点。
- **速度：** 关节限速的 0.9 倍。

## 3. 搬运臂：Franka Panda

代码是 `idea/demo/code/franka.py`（planner）和 `idea/demo/code/video.py` 的 `Handler`。`step5/video/franka.py` 是 10-21 的旧运动学，不要用。

### 3.1 模型

- **模型：** MuJoCo Menagerie 的 franka_emika_panda（Apache-2.0），从 `slides/Co-optimize/idea/assets/franka_emika_panda/panda.xml` 读取。它放在已过时的 `slides/` 下，清理 `slides/` 会把它删掉。
- **指尖：** 原装 54 mm 手指加长 50 mm，截面是 15 × 8 mm 的细条。指垫中心距指尖 9.5 mm，行程 80 mm。实机要另做这对加长指尖。
- **碰撞体：** 每个连杆、手和手指的凸包，取 26-DOP。

### 3.2 Planner：`Mover.transfer(body, A, B, out, into, grasps, …)`

把一个物体从位姿 A 搬到 B。物体可以是兔子 `B`、支撑 `S`，或支撑连兔子 `SB`。先算便宜的部分（IK 和直线），最后才跑 RRT。

1. **抓取候选：** 物体坐标下的中心、接近方向、闭合方向和宽度。从好到差最多试 24 个，闭合方向两种符号都试。
2. **抓取 IK：** 阻尼最小二乘，零空间拉向 ready 姿态。种子依次是上一个关节、ready 和 16 个随机值。
3. **接近直线：** 手指张开，从后方 6 cm 沿接近方向进来，然后夹紧（0.35 s）。
4. **离开直线 `out`：**
   - 兔子沿认证出口离开，走的长度是兔子沿出口方向的尺寸加 2 cm，是认证出口的前缀；
   - 支撑抬离桌面 2 cm，兔子 3 cm。
5. **放置：** 放置 IK，然后沿进入直线 `into` 到位，松开，后退 6 cm。
6. **自由运动（到抓取点、搬运、回停靠位）：**
   - 关节空间 RRT-Connect：步长 0.35 rad，每条边每 1.5° 检查一次，最多 4000 次迭代或 20 s；
   - 之后做 120 次随机 shortcut；
   - 按关节限速的 50% 计时。

直线段速度 0.12 m/s，每 5 mm 跟踪一次 IK，单步关节变化不超过 0.15 rad。

### 3.3 碰撞（`World.clear`）

- **Panda 运动部件的 26-DOP，对：**
  - 支撑和兔子的 2 mm 稠密表面采样（夹持时手指对所夹物体除外）；
  - 桌面；
  - 等待中的工作臂（连杆、针和头）；
  - 自身相隔至少 3 级的连杆。
- **被夹物体的表面采样：**
  - 离场景和桌面至少 5 mm（出入直线上对座位和桌面例外）；
  - 不进入工作臂；
  - 不进入 Panda 的连杆。

### 3.4 布置（`Demo.franka_cell`、`Handler.parking`、`Handler.rest_pose`）

- **Panda 底座：** 立在桌面上，离兔子各任务的平均工作位置 0.6 m，面向它。方位每 30° 取一个候选，与工作臂方向至少差 90°，选兔子出口最指向的那个。
- **兔子休息点：** 和 Panda 等距，绕 Panda 底座往远离工作臂的一侧转 45° 至 90°，离所有支撑落脚至少 4 cm。兔子在休息点的姿态，从最可能的 8 个稳定姿态 × 8 个朝向里选，取两边可执行抓取最多的那个。
- **停靠位：** ready 姿态绕 J1 转到手离工位最远、且无碰撞的位置。每次换场景都从停靠位开始，回停靠位结束。
- **B+9pose 视频（10-24，旧 Step5）：** 底座 [−0.364, −0.647, 0.110] m，yaw 30°；休息点 [0.216, −0.803, 0.110] m。坐标是视频坐标：第一个任务时工作臂底座在原点，z 是桌面高度。

### 3.5 一次换场景（`Handler.change`）

- **换座：**
  1. 兔子沿 task k 的出口取出，放到休息点；
  2. 支撑搬到下一个安装位；
  3. 兔子从休息点沿 task k2 的出口反向放入新座。
- **同座转向：** 支撑连兔子一起搬。
- **抓取：**
  - 支撑上用薄壁边缘夹持（距边缘 5 mm，倾斜 0°、±25°）和对径抓取（两侧法向反向误差在 20° 以内，24 个接近方向）；
  - 手不碰物体和周边，手指每侧留 6 mm；
  - 兔子上的抓取，在座内以及沿出口每 1 cm 都要无碰撞。

各次换场景并行规划。

### 3.6 命令与现有结果

```bash
PYTHONPATH=. .venv/bin/python -B whole_task_2/idea/demo/code/video.py --run B+9pose [--stills]
```

- `B+9pose.mp4`（10-24 22:37，用的是 10-25 重算前的 Step5 r00）：30 s，8 次换场景全部规划成功，每次 7 至 40 s。720 帧里有 1 帧（第 176 帧）Panda 擦到兔子（`panda-B`）。
- 记录在 `code/B+9pose.json`。缓存 `code/*_plan.pkl`、`code/*_film.pkl` 按输入和代码做 key，Step5 变了会重新规划。
- `B+03.mp4` 是 10-24 18:40 的旧版（倒挂 KUKA 做搬运），现在的代码复现不了。10-24 的记录是：B+03 task 4 的兔子在座内低于桌面 14.3 mm，`Handler` 会直接报错。这一点没在 10-25 的 r00 上复查。

## 4. 上实机前要知道的

1. **底座高度**：见 2.4。
2. **真机包缺失。** `whole_task_2/output/` 现在只有 `whole_shape`，三个 run 的 `output/<run>/step5/rNN/` 都不在了，`data/<run>/step5_combined/rNN.npz` 还在。`step5/combined.py` 的 `figures()` 只从磁盘读，会重画图并调用 `package()` 写包，不重跑算法。下面两条命令没试跑过：

   ```bash
   PYTHONPATH=. .venv/bin/python -B -m whole_task_2.step5.combined --object B+03_kuka --figures-only
   PYTHONPATH=. .venv/bin/python -B -c "import whole_task_2.run_poses; from whole_task_2.step5.combined import figures; figures('B+9pose')"
   ```

   包的内容：

   - `support.stl`：mm，支撑坐标，放大 1.01 倍留间隙；认证按原尺寸做。
   - `object.stl`：mm，物体坐标。
   - `info.json`：
     - 工位：关节顺序和限位、底座、工具、正方体；
     - 每个座位：`support_from_object`；
     - 每个任务：`world_from_object`、`world_from_support`、`support_lowest_z_mm`、`exit_direction_world`、`exit_length_mm`、地脚、检查结果、中心面的关节角。
   - `tasks.npz`：每个认证面的 `<task>_certified_faces`、`_needle_direction_world`、`_arm_joints_deg`、`_world_from_object_mm`、`_world_from_support_mm`。任务名里 `+` 写作 `P`，`−` 写作 `N`。SO101 只存 5 个关节，补的零已去掉。

   长度单位 mm，角度单位 deg。
3. **B+8pose、B+9pose 的脚本要先 `import whole_task_2.run_poses`。** 它在加载时把工作臂换成 SO101，不 import 的话臂的检查用的是默认 KUKA，结果是错的。
4. **07、08、09 号配置的 `arm_module` 指向已放弃的 `subdivide_task/step5/video/kuka.py`。** 它和 `whole_task_2/step5/video/kuka.py` 内容相同，但删掉 `subdivide_task/` 就会断。
5. **载荷。** 当前算法认证的是：
   - 针的加工力，方向在世界针方向锥内（半角等于工位的倾斜锥），大小 0 至 0.5 mg（`step3/independent_policy.json`；按 B 的 meta 质量 0.735 kg 约 3.6 N）；
   - 力矩只有这个力对质心的力臂，再加重力；
   - 接触按无摩擦算，桌面摩擦没有认证。

   08、09 号配置里写的 `maximum_processing_force_mg = 1.0` 只有 `subdivide_task` 会读。
   **其他加载方式：** 我们已经讨论过，除了针的推压，还可能有别的力、力矩加载方式，比如拉拽、扭转等；这些目前都没有进入算法。
