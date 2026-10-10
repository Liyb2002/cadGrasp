# Step4.1：整组退出方向初始化

从新的 Step3.1 读取真实贴合支撑与注册元数据，整组 pose 同时参与。先在物体坐标中找共同合法退出方向；没有共同方向时找相近的逐 pose 合法方向。保持原始世界朝向、高度、工作面与载荷。保存方向 d 后，沿 −d 装入、沿 +d 退出。

切除全部配置的完整退出空间，保留接触核心外原定 1% 退出净空。受力从真实最终接触重新计算，沿用每个 pose 的全部原始 32,768 载荷、非负反力和第七维不上抬条件。初始化承载失败也保存，交给 Step4.2 修复；初始化完成不能当作承载通过。

图片直接复用 `vis_func/step41_render.py` 的 `draw_pose`：灰色物体、蓝色不透明实体支撑、黑色实际退出方向、红色当前 pose 自己的名义 sweep 切除。透明 sweep 只显示前 100 mm，实际路径完整切除并检查末端分离。无图内文字。

每组保存在 `output/B/{pose_set}/step4/step4.1/`：

- `exit_directions.png`：各原生 pose 的实际世界退出方向。
- `exit_sweeps.png`：当前 pose 的切除及路径。
- `final_result.png`：全部路径切除后的同一个实体支撑，按各任务摆放。
- `support.obj`、`layout.npz`、`*_force.npz`：真实网格、方向与原始载荷结果。
- `data/report.json`、`data/render.json`：输入、执行代码和产物指纹。

在项目根目录运行：

```sh
# 仅初始化全部 B 组
.venv/bin/python slides/Co-optimize/step4.1/run.py B --jobs 3
# 从已有初始化运行当前 Step4.2，显式选组和新输出名
.venv/bin/python slides/Co-optimize/step4.2/run.py B \
  --sets pose1+2+3+4+5+6+7+8+9+10 --jobs 1 --output-name my_xyz_gradient \
  --incumbent-summary slides/Co-optimize/output/B/data/stable_gradient_xyz_force_v3/pipeline.json
```

真实初始化使用独立进程，单次构造 180 秒超时；在初始方向后，先尝试每个 pose 相对于各自地面法向的世界向上 0.25° 裕量，再按原定 0.25°／1° 协同微扰恢复。逐 pose 裕量可处理相反地面法向下共同水平 sweep 的近共面构造；不改变原注册布局。超时是数值未解决，不是无解证明。

Step4.1保存原始位置的初始化；后续Step4.2允许已落座配置进行世界XYZ Translation和airborne，并复用已通过的原需求结果。Step5按最终位置处理系统—地面与base，整件连通及强度留到后续。实现顺序见 [pipeline.md](pipeline.md)。
