# 一对一复用 / 一对多复用

用同一个 B 物体的当前 `pose_1`、`pose_2` 展示同一件刚性支撑的两种复用方式。名称中的“一对一／一对多”指 **support pose → object pose** 的使用关系。蓝色是支撑，灰色是物体；每次只放一个物体。两段视频均为 16 秒、24 fps、1600×900，固定相机、纯白背景，无地面网格和文字覆盖。取景聚焦完整支撑和两个落座姿态，物体装卸时从画面上方或斜上方进出。

相对上一版，主体在画面中的线性尺寸分别放大约 1.8 倍、2.1 倍；两个落座姿态均完整留在画面内。数值对比见 [presentation_changes.json](data/presentation_changes.json)，白底与取景检查见 [display_checks.json](data/display_checks.json)。

| 名称 | 支撑与物体姿态的关系 | 装卸路径 |
|---|---|---|
| **一对一复用（One-to-one Reuse）** | 支撑有两种摆放，每个 support pose 对应一个 object pose；物体相对支撑的落座关系相同 | 在物体／支撑共同局部坐标中找到同一条直线退出路径；世界方向随支撑旋转 |
| **一对多复用（One-to-many Reuse）** | 一个 support pose 分别服务物体的 pose1、pose2 | 两个 pose 均从世界 +z 一侧沿 −z 装入，沿 +z 退出 |

## 视频

- [一对一复用](vis/one_to_one_reuse.mp4)：pose1 装入、落座、退出 → 物体移出画面 → 空支撑抬起、转动、放回 → pose2 装入、落座、退出。支撑的实体形状全程相同。
- [一对多复用](vis/one_to_many_reuse.mp4)：pose1 从上方落座、退出 → 物体在支撑上方换姿态 → pose2 从上方落座、退出。支撑的位置、朝向和实体形状全程相同。

![两种支撑在 pose1 和 pose2 下的落座形状](vis/comparison.png)

## 构造与检查

一对一支撑采用 5 mm 外偏移的共同非工作面包裹，裁去两种支撑摆放下的地面禁区，并扣除共同退出方向的完整、连续、非凸物体扫掠。找到的局部退出方向为 `(-1, 1, -1) / sqrt(3)`；变换到 pose1、pose2 的世界坐标后都向上，但不是同一世界绝对方向。

一对多支撑在两种物体世界朝向下合并局部包裹，扣除两条世界 +z 退出扫掠和工作面禁区，加共同底托。为清楚展示姿态变化，删掉 60 mm 以上的耳部包裹、高墙和立柱，只保留低矮共同托座；相对上一版实体，支撑体积减少约 32.8%。物体整体抬高 8 mm，在底托上落座；两个姿态使用同一中心区域，分别装入，不同时占据，也不远距离分成两座。两种姿态都保留了实际非工作面接触，原始全部载荷未重新验收，因此这是展示简化，不是力学最小材料结论。

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
