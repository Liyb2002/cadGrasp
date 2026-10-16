# Single-use fixture / Juxtaposed fixture

2026-10-11 的 representation 研究记录：[讨论与论文主线](support_representation_idea.txt) · [具体表示与算法设计](support_representation_algorithm.txt)。设计以共享材料单元、各 pose 的作用端口及合法连接统一组织几何与力学。[六个代数小例子](code/representation_demo.py)及[输出](data/representation_demo.json)检查共享计费、几何冲突、力矩、反力密度松弛、地面/连通与坐标一致性；[只读占用下界检查](code/representation_bbox_probe.py)及[输出](data/representation_bbox_probe.json)核对已有五组完整模型。当前是设计与表示验证，尚未接入生产求解器；下方原视频与数据保持。

用同一个 B 物体的当前 `pose_1`、`pose_2` 展示同一件刚性支撑的两种复用方式，名称采用用户选定的 **Single-use fixture / Juxtaposed fixture**。前者每种支撑摆放对应一个物体姿态，后者同一种支撑摆放分别容纳多个物体姿态。蓝色是支撑，灰色是物体；每次只放一个物体。两段视频均为 **8 秒、48 fps、1600×900**，使用原有全部 384 帧以 2 倍速度播放。固定相机、纯白背景，无地面网格和文字覆盖。取景聚焦完整支撑和两个落座姿态，物体装卸时从画面上方或斜上方进出。

相对上一版，主体在画面中的线性尺寸分别放大约 1.8 倍、2.1 倍；两个落座姿态均完整留在画面内。数值对比见 [presentation_changes.json](data/presentation_changes.json)，白底与取景检查见 [display_checks.json](data/display_checks.json)。

| 名称 | 支撑与物体姿态的关系 | 装卸路径 |
|---|---|---|
| **Single-use fixture** | 支撑有两种摆放，每个 support pose 对应一个 object pose；物体相对支撑的落座关系相同 | 在物体／支撑共同局部坐标中找到同一条直线退出路径；世界方向随支撑旋转 |
| **Juxtaposed fixture** | 一个 support pose 分别服务物体的 pose1、pose2 | 两个 pose 均从世界 +z 一侧沿 −z 装入，沿 +z 退出 |

## 视频

- [Single-use fixture](vis/single_use_fixture.mp4)：pose1 装入、落座、退出 → 物体移出画面 → 空支撑抬起、转动、放回 → pose2 装入、落座、退出。支撑的实体形状全程相同。
- [Juxtaposed fixture](vis/juxtaposed_fixture.mp4)：pose1 从上方落座、退出 → 物体在支撑上方换姿态 → pose2 从上方落座、退出。支撑的位置、朝向和实体形状全程相同。

![两种支撑在 pose1 和 pose2 下的落座形状](vis/comparison.png)

## 构造与检查

Single-use fixture 采用 5 mm 外偏移的共同非工作面包裹，裁去两种支撑摆放下的地面禁区，并扣除共同退出方向的完整、连续、非凸物体扫掠。找到的局部退出方向为 `(-1, 1, -1) / sqrt(3)`；变换到 pose1、pose2 的世界坐标后都向上，但不是同一世界绝对方向。

Juxtaposed fixture 在两种物体世界朝向下合并局部包裹，扣除两条世界 +z 退出扫掠和工作面禁区，加共同底托。为清楚展示姿态变化，删掉 60 mm 以上的耳部包裹、高墙和立柱，只保留低矮共同托座；相对上一版实体，支撑体积减少约 32.8%。物体整体抬高 8 mm，在底托上落座；两个姿态使用同一中心区域，分别装入，不同时占据，也不远距离分成两座。两种姿态都保留了实际非工作面接触，原始全部载荷未重新验收，因此这是展示简化，不是力学最小材料结论。

两种最终支撑都是一个连通实体。已检查各 500 mm 连续装卸扫掠与最终支撑的体积碰撞、落座／退出终点，以及保存动画的路径与地面关系；详细记录在 [geometry.json](data/geometry.json) 和 [motion_checks.json](data/motion_checks.json)。仅清理了体积小于 `1e-15 m³` 的浮点退化孤片，并记录其实际体积。

这是几何复用概念展示，未运行原始载荷受力优化或强度验收，也不是 Direction／Juxtapose 新搜索的实验结果。视频播放使用 240 mm 装卸距离；500 mm 用于完整扫掠检查。既有 Co-optimize 结果不变。

## 复现

在仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python slides/idea/code/prepare.py
.venv/bin/python slides/idea/code/motion.py
blender -b --factory-startup --python-exit-code 1 --python slides/idea/code/render_blender.py -- --mode pose_following_wrap
blender -b --factory-startup --python-exit-code 1 --python slides/idea/code/render_blender.py -- --mode fixed_seat_reuse
.venv/bin/python slides/idea/code/finalize.py
```

`code/` 保存构造、动画、渲染与编码脚本；`data/` 保存实体、刚体变换、检查和原始帧；`vis/` 保存成片与落座预览。[video_report.json](data/video_report.json) 记录成片参数、帧数和 SHA256。

上一版画面、实体与代码保存在 `data/history/before_white_closeup/`，不作为当前展示。

此次改名与提速之前的 16 秒白底版本保存在 `data/history/before_selected_names_2x/`。原始几何和动画轨迹保持相同；`*_motion.json` 中的时间是 16 秒源动画时间，最终视频的对应时间为其一半。

## 机器人单侧加工与背面遮挡

选取已有结果 [`pose1+6+11+13+14+17`](../Co-optimize/output/B/pose1+6+11+13+14+17/step5.3/pose_11.png)。pose17 和 pose13 的原工作区中心相距 96.02 mm，平均外法向夹角 164.08°。把这两块真实工作区同时标在保存的 pose11 安装上：绿色为机器人当前对准的 pose17 工作区，橙色为背面的 pose13 工作区。蓝色使用当前认证实体 `wrap_surface_reinforced_all52_20261015_03/support.obj`，工件与支撑的相对变换保持原样，整套安装只绕重力方向转动并抬到工作台上。

四张图均为 2400×1500、白底、无文字覆盖：

- [机器人加工全景](vis/robot_access/01_robot_working.png)
- [正面工作区近景](vis/robot_access/02_front_work_area.png)
- [背面工作区近景](vis/robot_access/03_back_work_area.png)
- [透明遮挡示意](vis/robot_access/04_blocked_access.png)：红色虚线是从当前工具安装点朝背面目标的直线，叉号标出首次被工件截住的位置。

机器人使用原生尺寸的 Franka 网格、关节范围和正运动学，末端增加示意加工主轴。工具尖端与选中前表面三角形中心的偏差为 0.0016 mm；朝背面目标的直线确实先穿过工件。此处表现固定摆放下的单侧工具进入限制，未证明机械臂所有构型均无法绕到背面，也未把 pose11 下的示意加工当作 pose17 的原受力任务。原始求解结果、模型与既有视频保持不变。

[场景与来源](data/robot_access/scene.json)、[检查](data/robot_access/verification.json)和可继续制作视频的 [Blender 场景](data/robot_access/robot_access.blend)保存在 `data/robot_access/`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python slides/idea/code/prepare_robot_access.py
blender -b -t 12 --factory-startup --python-exit-code 1 --python slides/idea/code/render_robot_access.py
```
