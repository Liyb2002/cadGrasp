# Combined：Juxtapose → Translation → Direction

[8 秒视频](vis/juxtapose_translation_direction.mp4) · [单独 6 秒 Translation](../translation/README.md)

使用 Direction、Juxtapose 同一个 B shape，单个固定相机，蓝色实体支撑、灰色物体，绿色新增材料、红色切除材料；没有屏幕文字或图例。

1. **0–2.5 秒：Juxtapose。** 支撑固定在 pose1，pose2 沿真实装入 sweep 进入。扫到旧支撑后才删除对应交集；到位后补所有不妨碍当前退出的贴合包裹。
2. **2.5–5.5 秒：Translation。** 在上述布局上，将 pose2 横向挪 1/3 身位。每个显示位置重新计算所有配置的退出切除与贴合增补，展示原支撑损失及新材料。
3. **5.5–8 秒：Direction。** 位置保持不变，将 pose2 的退出方向微调 8°，同步重算支撑。随后物体沿新方向短距离退出、返回，最终新增材料显示为蓝色。

后两步是设计参数的改变：横移轨迹不作为滑动通道保留，改方向时也只要求当前完整装卸路径。8° 和 1/3 身位均是展示设定，不是新的力学梯度求解轨迹。

每个布局检查两个物体、四个完整 500 mm sweep 与工作禁区；关键终点检查原始工作面。Juxtapose 切除逐帧检查真实已扫过区域，最后的短退出逐位置检查物体交集。数据见 [geometry.json](vis/data/geometry.json) 和 [animation.json](vis/animation.json)。未运行全部原始载荷或完整夹具验收。

共享代码：[geometry.py](code/geometry.py)、[render.py](code/render.py)。以下命令生成两份视频：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cadgrasp_juxtapose_mpl .venv/bin/python slides/Co-optimize/operation_demo/combined/code/render.py --rebuild
```
