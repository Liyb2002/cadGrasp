---
status: active
---

# Translation：Juxtapose 之后微调重叠位置

[6 秒视频](vis/pose1+2+4+6_translation.mp4) · [8 秒完整操作](../combined/README.md) · [原数学示意](vis/translation_math.png)

Translation 恢复为当前工具。Juxtapose 为难以仅靠方向微调解决的 pose 选择另一个支撑摆放，并补贴合材料；放在哪里仍然需要优化。Translation 在这个重叠布局上继续调整相对位置，Direction 再调整装卸切除。三者都要比较新增材料、删掉原接触及占地的代价。

视频从隔壁 [Juxtapose](../juxtapose/README.md) 的 B/pose2 结果开始。pose1 的支撑摆放固定，pose2 沿地面、朝画面右侧平移 **1/3 身位**。身位取实际 pose2 网格沿该水平轴的投影宽度；实际距离、单位与每帧位置保存在 [几何记录](../combined/vis/data/geometry.json)。当前位移是演示设定，尚未由受力梯度选出。

每个位置都使用当前物体、工作区和完整退出 sweep 重新构造支撑。从已经挖好的 Juxtapose 支撑开始，增加新位置的同样 5 mm 贴合包裹，扣除所有四个配置的完整退出和工作禁区。旧位置不再作为额外任务要求；新包裹可以补回不受当前配置占据的区域。这份演示保守保留其他历史空缺，不从未切除的整壳强制恢复全部旧材料。

灰色物体、蓝色原有支撑、绿色当前新材料、红色刚发生的原支撑损失。只有与当前几何禁区相交的旧蓝色材料才会删去；新的绿色候选材料随当前贴合位置更新。没有文字、框、退出箭头或 sweep 叠加。平移完成后，绿色并入蓝色，物体沿原退出方向完整取出；最后只显示实体支撑，以等轴测高度绕它转 360°。

总长仍为 6 秒：0–2.5 秒展示平移和材料变化，2.5–3 秒取出物体，3–6 秒转入等轴测视角并绕空支撑观察完整一圈。镜头转动，支撑形状保持不变。

这表示设计布局变化，不是物体在已制造好的支撑内横向滑动；不切除横移轨迹、不沿横移轨迹生成支撑。其他 pose 的退出和工作区域仍作为约束。实际反向装入、1% 净空、全部载荷与连接／强度验收不由这份视频证明。

检查各显示位置的两个物体占据、四条连续 500 mm 退出及工作禁区；起点和终点另检查四个原始工作面。新增结尾逐位置检查实际取出过程，并以投影间隔确认物体完全离开后才隐藏；等轴测部分不显示物体。结果与视频时间表见 [animation.json](vis/animation.json)。几何量不代表承载容量，未运行新的七组力学搜索。

代码：[render.py](code/render.py) 使用 [combined/code/geometry.py](../combined/code/geometry.py) 的同一套布局，不另造一套视频几何。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cadgrasp_juxtapose_mpl .venv/bin/python slides/Co-optimize/operation_demo/translation/code/render.py
```

原 pose1 四格平移视频、最终四视角图及原渲染入口已归档至 `Co-optimize/data/documentation_history/before_post_juxtapose_translation_20261007/`。数学图保留；当前视频从 Juxtapose 的 pose2 开始。
