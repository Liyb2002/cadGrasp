# pose1+11+14+27：whole 初始化

全部 pose 注册到参考 pose_1。内部使用参考物体 mesh 坐标保存，世界注册变换逐 pose 留档。

[Step3.1：真实包裹支撑](step3.1/overview.png) · [Step3.2：整组禁区环绕图](step3.2/overview.png)

Step3.1 合并注册和贴合包裹，扣除所有完整工作角度禁区；Step3.2 只展示禁区并集。

支撑材料 94.664 cm³，实际接触面积 208.514 cm²。

| Pose | 原始载荷通过数 | 当前接触受力诊断 |
|---|---:|---|
| pose_1 | 32768/32768 | pass |
| pose_11 | 32768/32768 | pass |
| pose_14 | 32768/32768 | pass |
| pose_27 | 32768/32768 | pass |

INITIALIZED 表示注册和完整禁区切除完成，不表示最终受力或可装卸支撑已求解。受力失败仅针对当前重合布局，继续保留给后续 whole 的 Direction/Juxtapose 调整。本轮没有生成接地环、没有挖退出路径、没有运行 Step4。

旧阶段结果保存在 B 根目录的 `_history/before_whole_initialize_20261008/`。原 Step4 文件仍是旧初始化下的历史结果，其数值与图片未重算。
