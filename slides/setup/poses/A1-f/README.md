# A1-f 的目标姿态

当前所有保存的坐标已迁移为 Z-up，地面为 z=0。下方早期实验结论按原日期保留；最新 baseline 状态见 [当前说明](../../../baseline_algo/baseline_algo.md)。

[新版五组姿态与工作面总览](../target_poses.png)。灰色是物体，绿色是工作面；只保留五个选定 case 的新版预览，其余 case 的姿态数据仍保留。

| Case | 姿态 | 工作面面积 / 总面积 |
|---|---|---:|
| pose_1（旧图已删除） | 原始侧倾，保留工作面 | 13.84% |
| [pose_2](pose_2/pose.png) | 近直立倾斜，下端角着地 | 12.84% |
| [pose_3](pose_3/pose.png) | 倒立倾斜，上端角着地 | 13.13% |
| pose_4（旧图已删除） | 横向侧放，侧边角着地 | 9.61% |

两两重力方向夹角为 88.75°–143.94°，工作面面积交并比最高为 0.202。差异由物体坐标系中的重力方向衡量，不受相机或绕竖直轴旋转影响。

`pose_1/setup.npz` 保留原始输入。每个新姿态有自己的 `setup.npz`、可读的 `setup.json` 和 `pose.png`；设定包含刚体变换、质心、单点着地位置、工作面索引和源文件哈希。

所有姿态不穿地面；工作面为一块连通区域，面积占 8%–15%，工作面法向朝上的分量大于 0.35，面中心沿外法向射线可达，新工作面全部顶点离地超过 1.5 mm。工作面与 30° 力锥的完整可达性、加工禁区及支撑可行性由后续步骤检查。

这些是需要外部支撑保持的目标姿态，没有要求被动稳定，也没有验证物体翻转到目标姿态的轨迹。

生成：`python slides/setup/poses/target_poses.py A1-f`。
运行：`python slides/baseline_algo/step3_scheculer/run_all.py A1-f --pose pose_3`。

四个姿态从 Step1 运行到 Step5；进度、结果和预览见 [output/A1-f](../../../baseline_algo/output/A1-f/pose_1/step5_connect_support/results.md)。
