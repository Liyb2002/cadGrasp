# C5 的目标姿态

当前所有保存的坐标已迁移为 Y-up，地面为 y=0。下方早期实验结论按原日期保留；最新 baseline 状态见 [当前说明](../../../baseline_algo/baseline_algo.md)。

[姿态与工作面总览](poses.png)。灰色为物体，绿色为工作面，橙圈为实际地面接触点；每行有整体、工作面和接地点放大三个视角。

| Case | 姿态 | 工作面面积 / 总面积 |
|---|---|---:|
| [pose_1](pose_1/pose.png) | 原始椅脚着地姿态 | 9.25% |
| [pose_2](pose_2/pose.png) | 倒立倾斜，椅背顶角着地 | 11.24% |
| [pose_3](pose_3/pose.png) | 向后翻转，背部角着地 | 9.02% |
| [pose_4](pose_4/pose.png) | 横向侧倾，侧边角着地 | 10.35% |

两两重力方向夹角为 80.54°–158.81°，工作面面积交并比最高为 0.140。差异由物体坐标系中的重力方向衡量，不受相机或绕竖直轴旋转影响。

`pose_1/setup.npz` 保留原始输入。每个新姿态有自己的 `setup.npz`、可读的 `setup.json` 和 `pose.png`；设定包含刚体变换、质心、单点着地位置、工作面索引和源文件哈希。

所有姿态不穿地面；工作面为一块连通区域，面积占 8%–15%，工作面法向朝上的分量大于 0.35，面中心沿外法向射线可达，新工作面全部顶点离地超过 1.5 mm。工作面与 30° 力锥的完整可达性、加工禁区及支撑可行性由后续步骤检查。

这些是需要外部支撑保持的目标姿态，没有要求被动稳定，也没有验证物体翻转到目标姿态的轨迹。

生成：`python slides/setup/poses/target_poses.py C5`。
运行：`python slides/baseline_algo/step3_scheculer/run_all.py C5 --pose pose_3`。

四个姿态从 Step1 运行到 Step5；进度、结果和预览见 [output/C5](../../../baseline_algo/output/C5/pose_1/step5_connect_support/results.md)。
