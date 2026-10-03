> 本副本当前 Step3 使用 value network。入口、权重与结果见 [README](../README.md)。以下为复制时保留的历史 baseline 说明。

# Step3：各 pose 独立选头

目标为[工件—支撑的四项条件](../../obj_supp/README.md)：联合力与力矩平衡、整体不上抬、有限厚度连通实体、完整实体共同插入。Step3 搜索接触方案并验收前两项，后两项由 Step4 构造最终实体后验证。

各 pose 从自己的候选池独立选头，不要求继承旧头或固定共享数量。当前入口为 [run_independent.py](run_independent.py)；整组重跑入口为 [run_surface_batch.py](../run_surface_batch.py)。

每任务 200 个候选，每头固定为工件总面积 1%；每条链优先选 3 个头，未全覆盖则补第 4 个，最多十条独立 top5 链，首条通过即停止。每轮把候选与当前已选接触合并，对整组重新求反力，以原始载荷联合覆盖增量评分。从合法候选的 top5 按增量归一化抽样，全为零时均匀抽样；不因零即时增益直接删掉候选。

每个计入覆盖的载荷必须由同一组非负反力同时满足六维平衡和头部整体不上抬。每 pose 原始 32,768 个载荷全部通过才算成功；不追加反例、不运行连续载荷域验证，零加工力检查只作诊断。不逐轮优化面积、不作末尾扩展。预算内失败保留最好部分结果，不表示不存在其他接触方案。

同一 pose 的已选头须有共同水平退出方向和基本通路；这不认证连接实体。各 pose 的受力、工作面与退出检查独立，组模式额外施加下述完整接触面的地面余量要求。

2026-09-29 当前修改：**各 pose 仍独立选头、独立受力和退出，但增加整组地面余量筛选。** 头的构造输入按零厚度接触面表示；其全部有限三角面顶点变换到组内每个 pose 后，最低地面高度必须至少为 **2 mm**（数值容差 `1e-10 m`，恰好 2 mm 通过）。平面高度是仿射函数，因此检查全部三角形顶点等价于检查完整面，不使用中心点替代。只筛选原来的 200 个候选，不重定位、裁小或扩大头，不添加跨 pose 的受力、退出或工作面限制。

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  python \
  slides/baseline_algo/step3_scheculer/run_independent.py B \
  --poses pose_3 pose_6 --floor-poses pose_3 pose_6 --floor-clearance-mm 2 \
  --output-root slides/baseline_algo/output/B/pose3+6/step3_scheculer/independent_poses_floor2mm \
  --jobs 1
```

每组独立结果根目录必须显式指定，避免同一个 pose 在不同组错误复用未筛选的旧结果。新 schema 为 `independent_single_pose_floor_margin_v1`；主报告及候选表都保存整组 pose、原始输入哈希、2 mm 条件和每个头的逐 pose 最低高度。最终验收重新计算选中面。保留全部 32,768 原始载荷、1% 接触面积、200 候选、3–4 个头及最多十条链。

本地退出和细路径暂时继续用原有限厚度头作**保守探测**；零厚度直接送入旧实体凸包会退化。`normal_depth_m` / `local_geometry_probe_depth_m` 记录这个探测体厚度，**不再要求 Step4 保留这个旧实体体积**；`head_model=zero_thickness_contact_surface` / `construction_thickness_m=0` 明确新构造输入。Step4 从这些完整接触面造实际有体积的连接支架，并重新检查全部真实材料。2 mm 面筛选只修复头部地面余量，不能保证任意连接体、退出或受力通过。组模式不重建用户已删除的单 pose `heads.png`。

## 输出与后续阶段

每 pose 保存候选、逐轮贡献与抽样记录、最终接触面、覆盖结果和 `schedule.json`；组模式位于 `step3_scheculer/independent_poses_floor2mm/pose_<i>/`。报告明确 `independent=true`、`shared_heads=false`，并分别记录受力状态和几何筛选。

Step4 为所有 pose 的接触面构造同一件有限厚度刚性支撑，确定各任务摆放并检查全部活动与闲置材料。Step3 通过不能替代最终实体连通、工作面／地面净空、接地需求和完整退出验收；Step4 几何通过也不能把未通过的 Step3 受力改成通过。
