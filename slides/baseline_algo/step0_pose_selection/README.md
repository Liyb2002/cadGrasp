# Step0：读取数据集预计算组合

2026-10-03：姿态生成、载荷采样和组合枚举已迁入 [codes/precompute_objects](../../../codes/precompute_objects/README.md)。每个物体固定保存 30 个 pose 与 20 个组合，大小 2、3、4、5、6 各 4 组。运行算法时选择已保存组合，不重新生成 pose 或撒点。

- `objects/<name>/poses.json`：30 个 pose 的清单与生成参数。
- `objects/<name>/pose_sets.json`：20 个组合、完整两两兼容矩阵和输入哈希。
- `objects/<name>/poses/pose_<i>/`：`setup.npz/json`、`needs.json`、`samples.json`、`floor_contact.npz`。

`select_poses.py` 是算法适配层；随机种子只决定已保存组合的选择顺序。`TaskCache` 从数据集复用完整固定载荷及地面需求。需要阶段图与旧下游接口时，由 `run_floor_points.py` 发布阶段诊断；这不重新搜索姿态或采样。

兼容性仍按全部原始 32,768 个载荷计算：地面需求点随工件从来源 pose 映射到目标 pose，所有目标高度须 ≥ −1e−9 m。地面横向无限，不构造实体脚位。只证明固定配准下的采样地面必要条件；接触、完整实体、装卸与强度须由后续阶段处理。

旧算法结果使用旧 pose 清单。输入哈希变化后读者拒绝这些结果；须在新的输出中重新求解，不把同编号旧结果当作新数据的证书。
