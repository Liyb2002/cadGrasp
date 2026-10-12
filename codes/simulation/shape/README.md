# 测试物体输入

每个案例放在 `<object>/<pose>/`，目前为 `B/pose_2/`。通过 `python simulation/prepare_shape.py` 从已经通过审计的最新 baseline Step 5 / Step 6 快照同步。

| 文件 | 内容 |
|---|---|
| `manifest.json` | 坐标、单位、质量、载荷域、来源与 SHA-256 |
| `object.stl` / `object_geometry.npz` | 已放到 target pose 的完整工件；NPZ 保留工作面索引、COM、原始地面接触点 |
| `object_source.stl` | 原始局部坐标网格；用于保持原边界的四面体分解，须应用目标变换 |
| `target_pose.json` / `.npz` | 目标姿态快照；`T_world_mesh` 将原网格变换到世界坐标 |
| `working_area.stl` / `.npz` | 工作面、源面编号、内法向及真实面积；是表面而非实体 |
| `support.stl` | 最新完整支撑，世界坐标，单位 m |
| `support_geometry.npz` | 支撑合并网格、227 个凸构件及实际接地三角面 |
| `contacts.npz` | 三个接触头的接触三角面、中心、半径与候选 ID |
| `directions.json` | 整件退出/装入方向与距离；本轮动力学不执行这段运动 |
| `load_domain.json` | 连续加工载荷域及工件几何，力按 mg 归一化 |
| `load_samples.json` | baseline 保存的可达物理载荷样本；只能用于追溯/挑选，不能充当独立留出集 |
| `object_meta.json` | 工件质量及原网格坐标中的惯性参数 |
| `baseline_base.json` / `baseline_base_geometry.npz` | 最新 Step 5 底座报告与几何 |
| `baseline_connection.json` / `baseline_audit.json` / `baseline_trajectory.json` | 最新 Step 6 报告及可追溯证据 |

右手世界坐标，Z-up，地面 `z=0`，长度单位 m。除 `object_source.stl` 外，导出的实体和工作面均已位于目标世界坐标，**不要再次变换**。STL 不保留稳定面编号，关联工作面时用 NPZ。读取支撑 union 网格应保留拓扑，即 `trimesh.Trimesh(..., process=False)`，自动焊接非常靠近的顶点会破坏原拓扑。

工件–支架接触无摩擦；地面充分摩擦，允许脱离，不将支架锚定。baseline 使用无自重支撑；本轮动力学按暂定支架密度积分质量和惯性。这一差异需通过质量敏感性检查处理。

输入文件的 baseline 通过状态不等于动力学通过；新的结论看 `simulation/output/` 下的对应运行记录。`manifest.json` 的 `simulation_status` 只说明快照准备状态，不随实验结果改写，以保持输入哈希稳定。
