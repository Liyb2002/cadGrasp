# SO101 实机工位与 planning 交接

2026-10-25。实机侧来源：`~/Desktop/hiwonder-SoArm-101`。下面的工位尺寸来自实测。本文件同时记录需求、实现和验证；旧 B+9pose 包不能当作本工位的可执行结果。

## 已确认的实物参数

- 白色 SO-101 follower，底座刚性固定在桌面，无升降柱。世界坐标采用 `so101_new_calib.urdf` 的 `base_link`；桌面恒为 **z = −0.0024 m**，底座原点高于桌面 2.4 mm。
- 兔子：`objects/B/mesh.stl`，实机侧提供的 SHA256 前缀 `91691a4e136ea6b3`。
- 允许摆放区域：底座坐标 x = 0.05–0.35 m、y = −0.30–0.30 m。工作点由可达性选择，物体／支撑实际落在固定桌面上。
- 固定 NexiGo 1080p 相机，内参已标定。桌面标记 ID 1 在底座坐标 [2, 111, −2] mm，底座到标记已配准，不确定度约 ±5 mm／±3°；相机可将兔子摆放核对到几毫米。
- 初始工具：针横夹在夹爪中，针尖沿 gripper frame 的 +y 距原点 170 mm（100 mm 针 + 70 mm 安装距离）。实机随后提供 TCP 的 4×4 修正，由 `feasibility.tcp_transform` 输入。
- 先用 URDF 关节限位。初估零点偏置 shoulder_lift +12°、elbow_flex −3°，尚未进入 planning；实机标定后在执行侧重新解 IK。每个停点必须导出针尖位置与方向。
- 暂无搬运臂，人手换姿态，相机核对。保留出口方向，不需要 Franka 搬运轨迹。
- 五个工作关节速度均不超过 **60°/s**。

## 最终范围：只改独立 planning，不改 Step1–Step5

用户随后明确：**不修改 Step1–Step5，不重做 fixture。** 此要求取代最初交接中“修改 Step1 推导、重算 Step3–5、修改 06 号配置”的实施建议。

1. 在 `idea/demo/` 新增实机配置和 planner。原 `robots_config/random/06_so101.json`、`run_poses.py`、Step1–Step5 源码与已有设计数据均保持原样。
2. 桌高为固定输入。仅通过实机规划层把兔子或整个 fixture 平移到桌面；工作臂世界位姿固定，底座偏移恒为 0。
3. 首个 demo 为单 task、无支撑，使用概率最大的自然稳定姿态。沿用包结构：`info.json`、`tasks.npz`、`object.stl`，`support_from_object` 为空。
4. 后续读取已有 B+9pose r00，保留九个物体朝向、座位、支撑几何和原认证加工方向，重新搜索平面位置并解 IK。原任务中未被新工位认证的面明确列出；不能把旧包全部标成实机可达，也不通过改 fixture 补救。
5. 保留 `exit_direction_world`，没有 Franka 段，人工换位。打印尺寸与体积写入实机包。
6. 实机侧已在 Mac 上不渲染导出旧包，中心面关节均在旧模型限位内；但旧 `support_lowest_z_mm` 为 87–148 mm，与实际桌高不符。原设计验收与新工位的机器人可达性是两个检查。

## 实机侧负责

读包 → 底座／桌面标记坐标换算 → 相机叠加目标位姿供人摆放并核对 → 用标定后的臂模型对停点重解 IK → 按退出／过渡规则生成逐帧 `program.json` → 相机画面干跑 → 低速执行、回读关节与针尖轨迹 → 在网格上计算虚拟覆盖。以后切换 KUKA／Franka 由执行侧更换执行器。

## 实现与验证状态

实现文件：

- [real_station_so101.json](real_station_so101.json)：实测桌面、摆放范围、恒定底座、60°/s 上限及完整 TCP 变换。没有修改原 06 号随机工位。
- [code/real_arm.py](code/real_arm.py)：独立 SO101 适配器，读取完整的 gripper frame→加工 TCP 变换；解析 Jacobian、带限位 IK、实测桌面检查。原 `step5/video/so101.py` 未修改。
- [code/real_plan.py](code/real_plan.py)：无渲染离线入口，搜索平面摆放、认证加工触点、导出并从磁盘重放。实际导入检查确认不加载 MuJoCo 或 OpenGL；没有搬运臂或渲染运行依赖。
- [code/test_real_station.py](code/test_real_station.py)：6 项测试通过，包括原运动学等价性、标定 TCP 的有限差分 Jacobian、固定桌面／座位关系和包内停点单位。

搜索先用确定性随机 FK 点云和已有工具 screen 排候选；点云本身不作为认证。单 task 只使用同一个最高概率稳定姿态，允许绕竖直轴转动和平面平移，并选取 100 mm 工作立方体。现有 fixture 版本只保留原任务工作面与加工方向，不扩大任务。最终逐面检查三个顶点和质心：针身／头部对兔子的精确几何检查、针身／头部离桌、IK 和限位、连杆凸包对兔子及实际打印 fixture 的三角形检查、连杆离固定桌面；有 fixture 时另检查工具对 fixture。验收使用存盘关节作 FK 重放，不再解 IK。

### 首次单 task：已完成

目录：[real_so101/single/](real_so101/single/)。包含 `info.json`、`tasks.npz`、`object.stl`、`validation.json`，没有 `support.stl`。

- 稳定姿态概率 **0.2109504418374809**，与实机侧约 0.21 的自然坐姿一致；水平转角 225°。
- **549 个认证面／质心停点**，工作面积 **64.8733 cm²**，约整只兔子表面的 **11.5904%**。这是一个 task 的可工作区域，不代表整只兔子完成覆盖。
- 每面四组触点／关节，共 **2,196 组**；五个工作关节，未加实际零点偏置。
- 兔子实际世界 AABB（mm）：x **212.568–347.432**，y **−6.421–186.421**，z **−2.4–106.800**。全部在实测允许摆放区域内。
- 桌面 **−2.4 mm**，底座位置 `[0,0,0]`、旋转单位阵、所有底座偏移 **0**。
- 磁盘 FK 重放通过；最大位置误差约 **0.000106 mm**，最大工具轴向量误差小于 **8×10⁻⁷**。这是模型数值误差，不是实物定位精度。
- 对 Step1–Step5 的 **233 个 Python 源文件**记录 SHA256，运行前后完全相同。

重算命令（仓库根目录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -B whole_task_2/idea/demo/code/real_plan.py --workers 8
```

### 现有九 task fixture：已完成静止触点重规划与复验

读取 `data/B+9pose/step5_combined/r00.npz` 及其原始座位记录，输出到 `idea/demo/real_so101/B+9pose_r00/`，不覆盖原设计数据。

实际打印沿用原包已有的 1.01 放大约定，输出形状与原包的打印几何一致；没有重新设计。打印 AABB 尺寸 **183.321×193.439×185.177 mm**，实体体积 **301.877 cm³**。逐一把支撑落到固定桌面后，九个姿态的兔子最低点都在桌面之上，净高 **5.717–66.780 mm**；未发现必须重做 fixture 的桌面穿透问题。可达性检查另外进行。

九个 task 均已生成新的认证触点，独立磁盘重放通过。本次搜索保留 **368／2,974 个原认证面**，合计 **41.0698 cm²**，约原任务面积 **342.3895 cm² 的 11.995%**。这不是原九 task 全部完成的执行方案，也没有证明剩余面绝对不可达。`unprocessed_face_ids` 明确列出每个任务未认证的原面。搜索只在实测桌面范围内平移既有安装，不改变原朝向或 fixture。

| task | 原认证面 | 新工位认证面 |
|---|---:|---:|
| S1 | 553 | 48 |
| S2 | 141 | 20 |
| S3 | 509 | 94 |
| S4 | 325 | 30 |
| S5 | 576 | 25 |
| S6 | 513 | 72 |
| S7 | 80 | 34 |
| S8 | 216 | 18 |
| S9 | 61 | 27 |

所有支撑最低点均为 **−2.4 mm**（最大数值偏差小于 2×10⁻¹⁴ mm），所有工作臂底座偏移均为 0。源 Step5 fixture 及 Step1／3／4 输入文件 SHA256 均保持不变。

磁盘重放 **1,472 组**关节配置全部通过，错误列表为空。最大 FK 位置误差 **0.004504 mm**，最大轴向量误差 **2.85×10⁻⁵**。导出 `object.stl` 和 `support.stl` 的三角形面序及 mm 单位检查通过，STL 浮点量化的最大顶点偏差小于 **0.000008 mm**。Step1–Step5 的 233 个源文件仍与运行前一致。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -B whole_task_2/idea/demo/code/real_plan.py \
    --fixture-run B+9pose --result r00 --workers 8
```

只复验已有包，在对应命令末尾加 `--validate-only`，不会搜索、不解 IK、不渲染。验证分别检查导出 STL 的 mm 单位和面序、存盘关节 FK、固定桌面、工具与连杆碰撞；结果写入同目录 `validation.json`。

### 执行侧读取方式与验收边界

`info.json` 沿用 `tasks[k].world_from_object`、`world_from_support`、`support_lowest_z_mm`、`central_face`、`exit_direction_world` 和 `station`。无支撑时相关 support 变换为 `null`，`seats=[]`。矩阵平移为 mm，关节角为 deg。

`tasks.npz` 保留 `<task>_certified_faces`、`_needle_direction_world`、`_arm_joints_deg`、`_world_from_object_mm`、`_world_from_support_mm`，新增显式停点，执行侧不用从 STL 再推导：

| 数组后缀 | 形状 | 单位与含义 |
|---|---|---|
| `_test_points_world_mm` | F×4×3 | 面的三个顶点和质心，底座坐标，mm |
| `_arm_joints_deg` | F×4×5 | 对应四个测试位置的五关节角，deg |
| `_stop_tip_world_mm` | F×3 | 已排序的质心停点，mm |
| `_stop_needle_direction_world` | F×3 | 单位向量，工具→表面 |
| `_stop_arm_joints_deg` | F×5 | 对应质心的五关节角，deg |

无支撑时 `_world_from_support_mm` 为空数组；JSON 用 `null`。`object.stl` 保持原网格面序，单位 mm。面号是原始 `objects/B/mesh.stl` 的面号。

`real_station_so101.json` 的 `feasibility.tcp_transform` 是 **gripper_frame_from_process_tcp 的完整 4×4 齐次变换，平移单位 m，TCP 的 +z 是针指向表面的方向**。默认矩阵的平移为 gripper 的 `[0,0.17,0]`，不是旧配置的 flange +Z 表示。如果实机提供的是相对修正矩阵，先按其声明的坐标系与名义矩阵合成，再替换此完整变换；替换后重跑 planner。

这些包只认证静止触点，`continuous_motion_verified=false`；没有生成或执行 `program.json`。停点按相邻行蛇形排序，`info.json.execution.transition_rules` 记录 12／6／0 mm 直退、45／60／75° 退让方向和 80 mm 搜索长度，**底座升降禁止**。执行侧仍须用标定模型重解 IK，并逐帧验证退针、过渡与 60°/s 限速。无支撑单 task 未认证加工载荷下的倾覆／滑动，原 fixture 的载荷证书也没有在本 planning 层重算。

测试命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -B -m unittest whole_task_2.idea.demo.code.test_real_station -v
```
