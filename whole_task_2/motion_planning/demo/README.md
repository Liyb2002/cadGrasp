# bunny_movement：SO101 抓取并搬运 B

这是一个可独立下载运行的离线 motion planner。输入 B 的几何、起终点位姿、搬运 SO101 的底座／当前关节／夹爪状态以及桌面和障碍物，输出抓取与放置的完整关节轨迹。它不连接机器人，也不修改 Step1–Step5、fixture 或已有加工 plan。

下载本目录中的 `bunny_movement.zip`，解压后进入 `bunny_movement/` 即可使用，不需要原仓库、MuJoCo、OpenGL 或工作臂的针具代码。实机侧交接见 [HANDOFF.md](HANDOFF.md)。

当前 planner 版本为 **2.0**，新版独立下载名为 `bunny_movement_v2.zip`，与 `bunny_movement.zip` 内容相同。`program.json`／`trajectory.npz` 保持 v1 格式，兼容已有实机播放器。2026-10-26 的实机反馈与本次修复说明见 [FEEDBACK_20261026.md](FEEDBACK_20261026.md)。

## 运行

Python 3.10 或更高版本。依赖 `numpy`、`scipy`、`trimesh`、`rtree`、`python-fcl`；`matplotlib` 用于画预览。

```bash
python -m pip install -r requirements.txt
python bunny_movement.py --request example_request.json --output example_output
python bunny_movement.py --request example_request.json --output example_output --validate-only
python preview.py --request example_request.json --output example_output
python -m unittest -v test_bunny_movement
```

退出码 0 表示本次模型重放通过；退出码 2 表示规划或重放失败，原因在 `result.json`。失败搜索只表示在当前模型、抓取池与计算预算内没有找到解。新一轮正常规划会清除指定输出目录内的旧轨迹文件，避免留下旧的 `program.json`；请为需要保留的结果使用不同输出目录。

兼容示例仍保留原来的法线取样配置，用于回归比较；v2 已重新生成并验证 775 帧、38.7 秒的计划。20 项测试覆盖五轴 IK、原生 jaw、单位换算、碰撞、绕障搜索、计时、20° 斜夹、多初值精修、旋转指垫边界、重放失败后的继续搜索，以及悬空放置／物体瞬移的拒绝。运行环境与测试记录在 `tested_environment.json` 和 `test_report.json`。

白臂站立搬运请先按实机侧流程调整摆放并重新观测，用新的相机／关节请求运行：

```bash
python prepare_station_request.py --request so101_station_request_S1.json --output station_upright_v2.json --target-yaw-deg -25 --pad-radius-mm 15
python bunny_movement.py --request station_upright_v2.json --output station_upright_v2
python bunny_movement.py --request station_upright_v2.json --output station_upright_v2 --validate-only
```

`--target-yaw-deg` 是显式换目标：保持观测到的站立姿态，绕真实 pan 轴的位置作世界竖直轴旋转。不是绕底座原点旋转；源位姿和机器人标定不改。省略该选项会保留原来的 S1 目标。`--pad-radius-mm 15` 仅用于实机已经确认的指垫，省略则保留输入半径，不自动增大。新请求保留原输入，并重新计算相对路径。搜索配置来自 `station_search_profile.json`（8,000 点、36 个 roll、12 个随机精修初值、20° 法线锥）。

## 与系统输出的关系

示例的源位姿直接读取已有系统输出 `idea/demo/real_so101/single/info.json` 的 S1；副本在 `inputs/system_single/info.json`。B 网格保持原尺寸和 SHA256。目标是在桌面上绕示例搬运臂底座转动后的另一个位置。

示例搬运臂的底座位置和起始关节是用于验证算法的配置，并非实测工位。搬运臂与工作臂是两台独立机器人；不能把工作臂的底座、针尖 TCP 或关节计划当作搬运臂配置。示例只包含搬运臂、B 和桌面。真实工位的 fixture、停靠工作臂及针头需要添加到 `obstacles` 后重新规划。

`source`、`target` 均可直接读取系统包：

```json
{
  "package": "path/to/system_package",
  "task": "S1",
  "include_support": true,
  "travel_m": 0.08
}
```

包内应有 `info.json`；使用 `include_support` 时还应有 `support.stl`。读取 `world_from_object`、`exit_direction_world`、`exit_length_mm`，并把系统的 mm 明确换成 m。Step5 包若有打印引起的 `object_lift_mm`，也会加到实际物体高度上。`include_support` 将该安装的打印 fixture 加为固定障碍物，沿退出和插入段照常检查，不忽略整件 fixture。若起终点用了两个不同安装，则两件支撑必须已经摆在那里；本规划器只搬 B，不搬 fixture。

也可以直接指定 `world_from_object`（4×4，平移 m）、`exit_direction_world`（世界方向，指向座外）和 `travel_m`。取出时沿出口正向运动，放置时沿目标出口反向进入。未提供 fixture 时，默认向上抬离、从上方下降。

## 算法

1. 在 B 表面采样，在内法线的 20° 锥中发射多条射线。两端法线分别须在闭合轴的 20° 容差内，允许斜夹；检查跨度是否在原生 SO101 夹爪范围内。射线分批计算，接触对去重后受 `contact_pair_limit` 限制，避免大采样产生无限量 roll 候选；统计会记录截断前后数量。
2. 夹爪是一个固定指加一个旋转指。直接使用 URDF 的真实 jaw joint，依据两个标定指垫点求夹爪角度；绕接触连线采样抓取方向，并利用这一自由度微调五轴 IK。没有套用 Panda 的对称平行夹爪模型。
3. 为同一刚性抓取求起点和终点的完整位置／朝向 IK。精修从停靠／朝向初值开始，失败后使用默认 12 个随机初值，并保留不同关节分支；路径失败时继续尝试其他分支。SO101 只有五轴，不满足六维残差容差的候选会被拒绝。一次搬运全程保持同一抓取，不在空中换抓或丢弃姿态约束。
4. 先检查便宜的装卸段：抓取前直线接近、旋转夹爪闭合、沿源出口取出、沿目标出口插入、松开和后退。直线段逐步跟踪 IK，同时检查关节空间中间点。接近／后退先试工具轴方向，再试该端点的退出通道；还可指定 `approach_directions_world` 和 `pregrasp_distances_m`。闭合和松开使用与其他关节边一致的细分规则。
5. 到抓取前位置、携物搬运、返回停靠位采用双向 RRT-Connect。每条边先避碰检查，连通后做随机 shortcut。若直接连线可行则直接使用。
6. 对路径分段用五次 smoothstep 计时，同时满足配置的速度和加速度上限。每个候选都要重放最终计时帧与帧间细分点；失败则返回抓取搜索。选中后再用独立新建的碰撞场景验收，只有通过的结果才导出。这样不会因第一个候选在 close／release 末端失败而直接结束整次搜索。

碰撞使用 FCL BVH 三角形网格，包含原生夹爪开合、各连杆、B、桌面、输入障碍物及非相邻连杆的自碰撞。相邻机械装配链的碰撞不检查。允许的抓取接触仅限命名指尖附近的小区域；其余手部仍检查。携物姿态由 `world_from_gripper @ gripper_from_object` 计算，不能独立插值 B 来掩盖跟踪误差。

轨迹阶段：`prepare_gripper → approach_free → approach → close → extract → carry → insert → release → retreat → return → reset_gripper`。

## 配置与输出

- `robot.world_from_base`：搬运臂底座在公共世界坐标中的变换，平移 m。桌面始终使用同一世界坐标。
- `initial_joints_deg`、`park_joints_deg`：五个 URDF 模型角度；`initial_jaw_deg` 是第六个旋转夹爪关节的当前角度。
- `hardware_signs`、`hardware_zero_offsets_deg`：六关节映射，定义为 `model_deg = signs × hardware_deg + offsets_deg`。这不是 LeRobot 的归一化 motor position。
- `gripper`：两个指垫标定点、抓取跨度限制和允许接触区域。默认点取自名义 CAD，实机换指尖后必须更新。`contact_patch_radius_m` 保留物体接触点附近的球形限制；可再加 `contact_patch_half_extents_m: {"fixed": [x,y,z], "moving": [x,y,z]}`，按真实指垫的半尺寸限制接触。坐标在 jaw=0 时与 gripper frame 对齐，x 是名义垫面法线、y 横向、z 沿指尖；活动指垫坐标随 jaw 旋转。两种边界同时检查，不会因半径增大而忽略手掌碰撞。
- `obstacles`：固定障碍物数组。可用 `mesh`、`mesh_units`、`world_from_mesh`，或 `box_extents_m`、`world_from_mesh`。停靠工作臂也应以完整网格／保守包络加入。承接 B 的支撑须显式设 `supports_bunny: true`；系统包的 `include_support` 会自动设置。若端点不接近桌面（1 mm 内）或声明的支撑（2 mm 内），拒绝悬空放置；这个邻近检查不代表稳定性／承载证明。
- `planner`：随机种子、抓取／RRT 预算、IK 容差、避碰分辨率、关节速度／加速度、输出频率。默认 30°/s、90°/s²、20 Hz。

通过后生成：

| 文件 | 内容 |
|---|---|
| `program.json` | 有时间戳的六关节 model／hardware 角度、阶段、抓取和刚性连接、输入与代码哈希、执行交接约定 |
| `trajectory.npz` | 时间、六关节角、B 与 gripper 的世界位姿、阶段、是否携物 |
| `validation.json` | FK／碰撞重放、最大速度与加速度、文件哈希、认证范围 |
| `result.json` | 成功结果或失败原因 |
| `search_report.json` | 采样／接触对统计、精修初值数量、分阶段失败计数和有详细碰撞信息的失败候选 |
| `preview.png` | 运行 `preview.py` 后生成的位置／关节曲线预览 |

`--validate-only` 不搜索、不求 IK，先核对输入资产、代码和轨迹哈希，再重放；移动整个独立目录不会影响相对路径哈希。它检查磁盘上的结果，而非复用搜索时的“通过”标记。

若出现 `outside_pad_contact`，看 `pad_contact_failures` 或 `search_report.json`：其中记录帧／阶段、jaw 角、最大接触半径、允许半径、穿透量及实际指垫局部坐标。只依据实测指垫修改参数，再重规划；不能通过改动已有 validation 或删除失败帧来继续执行。

## 验证范围

这是名义模型下的离散运动规划与重放，不是数学意义上的连续扫掠证明。细分同时限制关节步长和用连杆长度界估计的空间步长。实机执行侧仍需使用实测底座、关节零点、夹爪接触点、实际障碍物和控制周期重新生成轨迹，并确认抓住／放稳后再继续。

夹持力、摩擦、SO101 的实际负载能力和动态挠曲尚未认证。原 B 元数据中的 0.735 kg 来自设计密度，不代表打印实物重量；实机侧应输入实际重量并验证能够保持抓取。`validation.json` 对这些项明确保留 `false`，没有把几何可行解释成实机承载通过。

SO101 URDF／STL 来源为 TheRobotStudio 的 SO-ARM100 仓库中的 SO101（Apache-2.0）；详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
