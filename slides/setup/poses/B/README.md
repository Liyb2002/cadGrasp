# B 的目标姿态

当前所有保存的坐标已迁移为 Z-up，地面为 z=0。下方早期实验结论按原日期保留；最新 baseline 状态见 [当前说明](../../../baseline_algo/baseline_algo.md)。

[新版五组姿态与工作面总览](../target_poses.png)。灰色是物体，绿色是工作面；只保留五个选定 case 的新版预览，其余 case 的姿态数据仍保留。

| Case | 目标姿态 | 工作面 / 物体总表面积 |
|---|---|---:|
| pose_1（旧图已删除） | 原始倾斜姿态，保留原工作面 | 10.18% |
| [pose_2](pose_2/pose.png) | 另一侧倾斜，改用另一片工作面 | 9.17% |
| [pose_3](pose_3/pose.png) | 倒立，耳 A 的耳尖着地 | 10.45% |
| pose_4（旧图已删除） | 另一倾角倒立，耳 B 的耳尖着地 | 10.96% |

每个文件夹的 `setup.npz` 保存 `T_world_mesh`、质心、地面接触点、工作面 mask 和源文件哈希；`setup.json` 记录工作面编号与检查结果。新姿态在统一细分一次的 4000 个三角面上定义，Step1 重放相同细分。原始 `objects/B/poses.json` 保留旧的稳定放置采样结果，坐标已同步换轴。

各姿态都不穿地面、只有一个顶点落在 z=0；工作面连通，占总面积的 8%–15%，所有工作面朝上的法向分量大于 0.35，面中心沿外法向的射线不与物体相交。新姿态的工作面全部顶点离地大于 1.5 mm。完整 30° 力锥内的逐载荷可达性由后续 Step1 检查。

两个倒立姿态分别接触原始 mesh 顶点 8 与 70，重力在物体坐标系内的方向不同，不只是绕竖直轴换了朝向。它们是需要外部支撑保持的目标姿态；这里没有验证从稳定放置到目标姿态的翻转轨迹。

生成：`python slides/setup/poses/target_poses.py`。
后续运行某个 case：`python slides/baseline_algo/step3_scheculer/run_all.py B --pose pose_3`。
结果写入 `slides/baseline_algo/output/B/pose_3/`。四个 case 已执行到 Step5，并按新的工作体积禁入约束重跑；禁区现由 Step4 根据加工射线可达性计算和出图，Step5 读取它进行连接避碰，见 [运行结果](../../../baseline_algo/output/B/pose_1/step5_connect_support/results.md)：Step5 已增加绕行连接，pose_1 在原始地面需求圈上通过连接几何及顺序插入；pose_2–pose_4 的实际接触界面与当前保守工作禁区冲突，包含工作面边界贴合。独立支撑受力平衡仍未证明。
