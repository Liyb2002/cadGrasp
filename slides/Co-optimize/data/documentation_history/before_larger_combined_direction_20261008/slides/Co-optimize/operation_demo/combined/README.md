# Combined：Juxtapose → Translation → Direction

[8 秒视频](vis/juxtapose_translation_direction.mp4) · [单独 6 秒 Translation](../translation/README.md)

使用 Direction、Juxtapose 同一个 B shape，操作阶段固定相机，蓝色实体支撑、灰色物体，绿色新增材料、红色切除材料；没有屏幕文字或图例。结尾取出物体，再用等轴测相机绕空支撑转一圈。

1. **0–1.75 秒：Juxtapose。** 支撑固定在 pose1，pose2 沿真实装入 sweep 进入。扫到旧支撑后才删除对应交集；到位后补所有不妨碍当前退出的贴合包裹。
2. **1.75–3.25 秒：Translation。** 在上述布局上，将 pose2 横向挪 1/3 身位。每个显示位置重新计算所有配置的退出切除与贴合增补，展示原支撑损失及新材料。
3. **3.25–4.75 秒：Direction。** 位置保持不变，将 pose2 的退出方向微调 8°，同步重算支撑。直接复用 `direction/` 原视频的黄色箭头和透明蓝色完整 sweep，展示方向、扫掠和切除如何一起变化。
4. **4.75–5.25 秒：取出物体。** 沿最终方向完整退出，新增材料显示为蓝色。
5. **5.25–8 秒：空支撑的 360° 观察。** 转入正交等轴测高度，固定支撑几何，绕它转完整一圈；物体、箭头与 sweep 全部隐藏。

后两步是设计参数的改变：横移轨迹不作为滑动通道保留，改方向时也只要求当前完整装卸路径。8° 和 1/3 身位均是展示设定，不是新的力学梯度求解轨迹。

每个布局检查两个物体、四个完整 500 mm sweep 与工作禁区；关键终点检查原始工作面。Juxtapose 切除逐帧检查真实已扫过区域，最后完整退出逐位置检查物体交集，并确认完全离开后才隐藏。数据见 [geometry.json](vis/data/geometry.json) 和 [animation.json](vis/animation.json)。未运行全部原始载荷或完整夹具验收。

Direction 叠加层调用原视频的 [Renderer](../../vis_func/animate_exit_directions.py)，沿用 `#6ebad8`／`0.084` 透明 sweep 和填充黄色箭头；正交投影与本视频实体渲染对齐。显示完整 500 mm 连续实体 sweep 在当前镜头内的部分，箭头和切除使用同一当前方向。

共享代码：[geometry.py](code/geometry.py)、[render.py](code/render.py)。以下命令生成两份视频：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cadgrasp_juxtapose_mpl .venv/bin/python slides/Co-optimize/operation_demo/combined/code/render.py --rebuild
```
