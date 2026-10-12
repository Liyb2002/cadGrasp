2026-10-03：目标姿态与载荷预计算已迁移到 [../precompute_objects](../precompute_objects/README.md)。当前对象仅保留网格、物理元信息、30 个 pose 和 20 个组合；旧轨迹、视频、任务目录和缓存已清理。本目录的机器人流程为历史／后续运动研究代码，不再是数据集生成入口。

# 姿态生成与机器人轨迹

## 兼容组合优先的姿态生成（2026-09-29）

当前 B 已重新生成 **20 个接地目标姿态**，其中有 206 组三姿态、100 组四姿态、5 组五姿态通过完整原始载荷的地面兼容检查。`pose_1` 至 `pose_5` 就是一组合法五姿态；前三、前四个也是合法子集。20 个姿态的最小两两重力方向夹角为 19.783°，保留至少 18° 的差异要求。这里认证的是 Step0 条件。后续同一批 n=2、3、4、5 各两组 baseline 已加入 Step3 整组退出方向继承并重跑：1/8 组选头全覆盖且构造连接实体（后续 Step4 完整退出通过，但联合载荷验收失败），其余七组在十条链内未全覆盖，详见[八组运行结果](../../slides/baseline_algo/output/B/pose2+9+13+15+17/step4/data/batch_report.md)。

```sh
# 生成并发布 20 个目标，要求存在至少一个五姿态兼容组
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python codes/setup/demo_regrasp.py B --poses-only --pose-count 20 --compatible-size 5 --pose-seed 20260929

# 只产生候选计划，不替换 objects/B；--publish 才发布
python codes/setup/compatible_pose_search.py B --output /tmp/B-pose-plan --pose-count 20 --compatible-size 5

# 独立重算全部载荷与组合；以及通过 baseline 正式输入再次复验
python codes/setup/verify_sequences.py B
python codes/setup/verify_baseline_pose_set.py B
```

`compatible_pose_search.py` 随机生成完整三维旋转，按最低点接地，仍要求唯一原始顶点接地、质心投影与支点相差至少 1 mm。每个候选先固定一块可见、朝上、离地的连通工作面（总面积 6–10%），再计算兼容性；载荷规则仍为 `K=0.5`、30° 半角、固定种子。没有缩小加工力来获得通过。

候选图中一条边表示双方所有地面需求都不穿入对方地面，并且姿态差异足够。先用同一固定样本流的前 1024 个载荷筛选；搜索五顶点两两相连的组合，使用完整 32768 个载荷复验，删除不通过的边后继续搜索。面内凸包顶点只是仿射高度检查的精确加速，最终独立验证仍遍历全部样本。找到合法组合后补齐其他有足够差异的姿态；有有限候选预算，失败不会发布半成品，也不宣称全局无解。

`compatible_pose_export.py` 在临时目录完成完整复核后才替换输入；保存固定工作面，后续不重新抽工作面。正式 baseline 从这些 setup 重建载荷，已核对 20×20 个方向的冲突计数完全一致，并输出三、四、五姿态示例的 Step0 结果。

当前数据使用 `cadgrasp_pose_set_v1`，**只有目标姿态，没有新机器人轨迹或视频**。完整运动流程另做；不得把历史轨迹当作这 20 个 pose 的机器人验证。`poses.json`、每个 `setup.json` 都显式标记 `placement_trajectory_verified=false`。旧十姿态输入、视频在 `objects/B/history/before_compatible_poses_860a4233e74b/`；旧 baseline 结果在 `slides/baseline_algo/output/B/history/before_compatible_poses_860a4233e74b/`。新旧 pose 编号代表不同姿态。

结果入口：[20 姿态总览](../../objects/B/overview.png)、[全部兼容计数与见证组合](../../objects/B/floor_compatibility.json)、[baseline 复验](../../slides/baseline_algo/output/B/step0_pose_selection/pose_set_validation.json)。

## 连续机器人轨迹生成（历史十姿态流程）

每个物体保存一条 `rest → pose_1 → … → pose_10` 轨迹。姿态按搬运可行性筛选，
允许 cherry-pick；这是共享被动支撑研究的准备流程，不作为抓取或运动规划贡献。

此节适用于 `cadgrasp_sequence_v1` 数据。目标数量现可由 `demo_regrasp.py` / `regrasp_sequence.py` 的 `--pose-count` 指定（默认仍为 10）；导出、分段、总览和验证读取实际数量。直接运行机器人搜索会寻找自己的目标，不代表已复现上面的兼容目标集。带 `--compatible-size` 的组合保证目前仅用于 `--poses-only` 分支。

新一轮 demo 入口是 `demo_regrasp.py`，搜索核心是 `regrasp_sequence.py`：每次先回到可自稳定的中间落座姿态，张开
夹爪并退开，再换接触位置和接近／夹紧方向，搬到下一个目标。中间落座是实际
连续轨迹的一部分，不是剪接或物体重置。只有全部十个目标成功后才替换对象数据；
`poses.json` 中 `trajectory_revision: diverse_regrasp_v1` 标记新结果，未带该标记
的对象仍是旧的单次夹持序列，不能当作已完成换抓。

使用 KUKA LBR Med 和通用平行直指夹爪。十个目标姿态均接地，在目标处由夹爪保持；
状态间允许抬起后转动、落下；抬升高度按物体及夹爪的转动间隙计算。各目标只有一个原始网格顶点接地，
质心投影偏离该点，因此不能把夹爪保持误当作物体自稳定。全过程不重置物体；
是否使用闭合抓持的理想约束，按下文的模式及每个对象记录区分。
部分难例从实际连续运动中选择目标，具体规则记录在各自
`poses.json` 的 `pose_selection` 中。

```sh
# 搜索每次换抓的新序列；临时搜索检查点保存在系统临时目录，可自动续跑
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python codes/setup/demo_regrasp.py B

# 搜索、渲染、验证并同步多个物体；已完成对象直接验证复用
PYTHONDONTWRITEBYTECODE=1 python codes/setup/batch_regrasp.py A1-f A2 A3 --workers 2

# 从保存的物理轨迹制作无文字视频，保存为 objects/B/video.mp4
PYTHONDONTWRITEBYTECODE=1 python codes/setup/sequence.py B --replay

# 检查所有物体的数据一致性
PYTHONDONTWRITEBYTECODE=1 python codes/setup/verify_sequences.py

# 生成每个物体的十姿态总览；也可在命令后指定 B A5 等物体名
PYTHONDONTWRITEBYTECODE=1 python codes/setup/overview.py

# 在原工作面内部缩小至 6–10%，保留原种子、姿态和轨迹，并更新总览
PYTHONDONTWRITEBYTECODE=1 python codes/setup/resize_work_areas.py

# 独立复核夹爪/地面间隙和撤夹后的运动；结果并入原有 poses.json
PYTHONDONTWRITEBYTECODE=1 python codes/setup/audit_sequences.py
```

`sequence.py --replay` 是通用视频入口，其搜索模式是旧的单次夹持流程。
个别物体使用同目录的专属脚本，最终输入中记录
生成器和筛选规则。视频回放工件与夹爪的原始积分状态，机械臂以完整末端姿态
逆解跟随；工件用深蓝灰色，地面用浅色。新视频只有一个主画面，无文字和画中画。

换抓序列的 `grasps` 给出物体坐标系中的接触对、夹紧轴和接近方向；
`regrasp_events` 给出实际松手、无夹持落稳和重新抓取的时间。接触对交换左右
标签不算新抓点。新目标两两的重力方向至少相差 18°、相邻至少 32°，因而仅
绕世界竖直轴旋转不算新目标；相邻夹法方向至少改变 30°，同时改变接触位置。
阈值和实际接触／机器人检查随结果记录，不能把名义夹持候选当作实机验证。

`demo_regrasp.py` 默认采用核心入口的 `--ideal-grasp` 假设：两指闭合并检测到持续双侧接触后，启用
刚性抓持约束；落回稳定中间姿态后先解除约束，再张开并退开。此模式不验证
真实夹持力是否足够，`grasp_contact_dynamics_simulated=false` 和
`verification_scope` 明确记录这个限制。`trajectory.npz` 及逐段文件保存
`ideal_grasp_active` / `ideal_grasp_eq_data`，可查看约束在哪些帧启用。
整个路径仍检查物体落地、松手后的中间稳定性和 KUKA 的可达性、限位、速度、
碰撞；不会在松手期间保留抓持约束。
无摩擦地面上，中间落座可能仍有绕竖直轴的残余转动。Demo 模式检查其不会
倾倒，并在重新靠近及闭合到首次手指接触前跟踪物体的实际位姿；首次接触后
保持夹爪位置完成闭合，避免跟随夹爪自己造成的物体偏移。不会把速度清零，也不声称
中间状态完全静止。全部三维转动和倾斜漂移分别记录在 `regrasp_events`。
理想抓持启用后，当前控制器将继续闭合的力上限降为 2 N，避免夹指与理想约束
相互挤压；初次双侧接触的闭合过程不变。这不是抓持力验证。实际值逐次记录于
`post_attachment_closing_force_N`；旧检查点前缀没有该字段时使用原先记录的
`closing_force_scalar_N`，不能把旧段也标成 2 N。

核心入口 `regrasp_sequence.py` 不加该选项时运行真实自由物体与夹爪的接触动力学；`--soft-pads` 使用同尺寸
通用平直软垫（MuJoCo `condim=4`，扭转摩擦有效长度 3 mm）的仿真假设。
三种模式使用独立临时检查点，不能把不同模型的半条轨迹拼起来。
接触维度含义见 [MuJoCo 官方文档](https://mujoco.readthedocs.io/en/3.7.0/computation.html)。

`--tool compact` 选择另一种固定尺寸的通用平行夹爪：最大开口 100 mm，
两根 20 × 8 × 60 mm 平直手指，掌部高度 60 mm，法兰至指尖仍为 120 mm。
手指接触区位于工具坐标 z=60–120 mm，现有 80–118 mm 抓取深度仍在该区间内。
它不是按物体表面设计的仿形夹爪，也没有对应已采购型号。默认 `--tool standard`
保留原工具。Compact 的三种模式分别使用带 `-compact` 的独立检查点，必须从
该工具自己的初始状态运行，不能接续 standard 的轨迹。实际模型文件写入
`grasp.model`，回放和碰撞检查均加载同一工具。

`--min-grasp-width` 设定候选接触对的最小间距，默认 0.015 m。
薄壁物体可使用更小值搜索局部边缘夹持；静态预检的闭合位置限制在真实关节范围内，
仍需仿真中实际双侧接触后才能启用理想抓持，不允许用负开口制造接触。

`--com-weight` 控制候选排序中偏离质心的惩罚，默认 2.5；较小值让搜索更多考虑
靠外侧的抓位。A5 的后续搜索使用 0.25，仍保留实际相邻接触位置至少改变 6 mm、
方向至少改变 29.5° 的检查。D4 的薄边搜索使用 `--min-grasp-width 0.001`。
这些选项改变候选搜索，不会跳过碰撞、实际双侧接触或机器人路径检查。

## 数据

```text
objects/cases.json                    # 当前物体列表
objects/index.json                    # 几何来源与尺寸索引
objects/<name>/mesh.stl               # 原始网格，单位 m
objects/<name>/meta.json              # 几何及模型质量
objects/<name>/video.mp4              # rest → pose_1 → … → pose_10 的完整视频
objects/<name>/overview.png           # 2×5 十姿态总览，橙色为已有工作面
objects/<name>/poses.json             # rest、顺序十姿态、夹持和检查结果
objects/<name>/trajectory.npz         # 连续物理轨迹与采样机械臂关节轨迹
objects/<name>/trajectories/index.json # 逐段文件、起止姿态、完整视频时间、字段说明
objects/<name>/trajectories/rest_to_pose_1.npz
objects/<name>/trajectories/pose_1_to_pose_2.npz
...                                  # 一直到 pose_9_to_pose_10.npz，共十段
objects/<name>/tasks.json             # pose_1 至 pose_10
objects/<name>/tasks/<pose>/setup.npz  # baseline 位姿、工作面、地面支点、载荷输入
objects/<name>/tasks/<pose>/setup.json # 输入检查与来源
simulation/videos/<name>_grounded_sequence.mp4
objects/_simulation_assets/<name>/scene.xml
objects/_simulation_assets/<name>/collision/ # MuJoCo 接触计算必需的后台碰撞几何
```

逐段轨迹直接切取原始 `trajectory.npz`，没有重新规划或插值。第一段包含从 rest
开始的靠近、夹取和第一次落座；相邻两段共享一个边界帧，拼接去掉重复边界后
恢复完整原轨迹。`time_s` 是段内时间，`sequence_time_s` 是完整仿真时钟；索引的
`video_start_time_s` / `video_end_time_s` 按实际视频帧率给出播放器中的时间位置。
`T_world_object` 保存物体实际位姿，`robot_q` 保存 KUKA 七轴关节角，其采样时间
单独保存在 `robot_sequence_time_s`。目标事件可能在 33 ms 存储帧之间，因此保留
名义起止时刻和实际采样时刻，未伪造精确端点。

每个 `poses.json` 的 transition 和每个任务的 `setup.json` 都指向对应进入轨迹。
`python codes/setup/organize.py` 可从现有完整轨迹/视频重建这个交付布局。
新导出自动生成逐段文件并清除过期的本地视频；重新回放即可生成匹配的新视频。

`overview.py` 直接读取 `poses.json` 的十个目标变换和各任务保存的工作面，
按 pose 1–10 生成一张 3200×1472 PNG。同一物体的十格使用统一视角和比例。
图中省略机械臂与夹爪以便比较形状；这些是需要保持的目标姿态，不表示能自行站稳。
姿态或工作面更新后应重新运行该命令。

每个目标另用固定随机种子选择连通工作面：面积占总表面积的 6–10%，外法向朝上，
法向射线不穿物体，离地至少 1.5 mm。这里没有用支撑搜索结果筛选工作面。
Baseline 按 `tasks.json` 读取输入，保留 `K=0.5` 和 30° 载荷锥。
目标更新后，旧 baseline 支撑结果不对应这些新输入，必须重新求解。

## 验证范围

- MuJoCo 接触动力学模式检查双指接触、正地面力、原网格离地误差以及目标
  跟踪误差；理想抓持模式明确假设闭合夹持成立，记录实际接触数和约束状态。
- KUKA 七轴完整末端姿态逆解、关节限位和采样速度检查；用 URDF 碰撞网格的
  凸包检查机械臂与地面、物体、夹爪及非相邻连杆的采样碰撞。
- `verify_sequences.py` 检查十姿态顺序、轨迹连续性、变换、输入哈希和数值记录。

机械臂是运动学回放，未模拟驱动力矩。控制器使用理想的物体位姿反馈；没有认证
真实硬件误差、鲁棒抓持、支架安装或夹爪释放。Standard 夹爪最大开口 160 mm、两根
40 × 12 × 80 mm 直指，闭合力标量限值 70 N、滑动摩擦系数 0.8；这些是通用
仿真假设，尚未对应采购型号和实测参数。搬运夹爪的摩擦不改变工件与被动支撑
无摩擦的约定。地面碰撞采用原网格凸包，对平面的支撑高度与原网格一致；
手指接触使用保存的凸分解，B 使用保边界四面体。

D4 的薄壳外缘约 161 mm，使用 `parallel_jaw_180.xml` 的 180 mm 通用开口规格，
直指尺寸相同，掌宽相应为 210 mm。每个物体在 `grasp.model` 记录实际工具，
视频和场景检查按该配置加载。夹持力及专属动作参数以各对象记录为准。

独立审查结果保存在 `poses.json → verification.independent_audit`，包含全部
存储帧的夹爪/地面碰撞，以及每个目标撤销夹爪接触后 1 秒的自由运动。
该独立审查只在实际执行 `audit_sequences.py` 后写入，不能把旧数据的审查值
套用于重新生成的轨迹。轨迹以 33 ms 取样，终点接触力来自生成过程记录；
碰撞检查允许的数值穿透为 0.2 mm，具体最大值随每条轨迹记录。
生成器哈希记录导出时磁盘中的源码；续跑的检查点前缀可能由更早版本生成。
精确回放以保存的逐帧轨迹为准，不能只凭当前源码哈希宣称整条搜索可完全复现。

`grasp.py` 的独立三转角演示、`imprint.py` 的仿形夹爪以及
`prepare_baseline_tasks.py` 的两个独立倾斜目标是历史实验入口；当前十姿态
数据使用 `sequence.py` 或记录的专属生成器，不运行这些旧入口。
