# B 九组：统一物理几何上的退出空间优化，global_release_v7

最终结果：在本轮有限方向检查中，九组均没有新增共同退出方向，头级平均集中角均未改善。不能把这个结果解释成连续球面上不存在其他共同方向。

所有组最终头级原始载荷检查通过（每 pose 原始 32768 个力／力矩需求与 shared no-uplift），各 pose 对实际头并集仍有经过连续扫掠检查的退出路径。完整夹具只有 pose1+3 与 pose3+6 通过 Step4 接受并产生有效 Step5 测量；其余七组候选被构造器或完整几何检查拒绝，不是完整成功。

本轮只隔离测试退出空间削减，执行参数为 --global-release --merge-trials 0；初始化中已有相连实体的规范化不新增连接材料。各 pose 的安装关系在初始化后固定。九组均未得到跨 pose 物理模块。

| Pose set | 初始化接触区域 | 物理头：初始→最终 | 接受削减步数 | 头级集中角：初始→最终 | 完整 Step4/5 | 耗时 |
|---|---:|---:|---:|---:|---|---:|
| pose1+3 | 8 | 7 → 7 | 0 | 1.88° → 1.88° | 通过 | 25.92 s |
| pose1+2+3+4+5 | 21 | 19 → 19 | 23 | 41.69° → 41.69° | 未通过 | 272.79 s |
| pose1+2+8+17 | 16 | 13 → 13 | 5 | 46.57° → 46.57° | 未通过 | 78.53 s |
| pose2+10+15 | 13 | 9 → 9 | 1 | 46.57° → 46.57° | 未通过 | 58.51 s |
| pose2+12+15 | 14 | 11 → 11 | 3 | 35.39° → 35.39° | 未通过 | 60.29 s |
| pose2+9+13+15+17 | 18 | 16 → 16 | 4 | 54.69° → 54.69° | 未通过 | 110.37 s |
| pose3+6 | 8 | 8 → 8 | 1 | 32.95° → 32.95° | 通过 | 32.91 s |
| pose5+7 | 8 | 8 → 8 | 2 | 53.39° → 53.39° | 未通过 | 31.80 s |
| pose6+8+10+19 | 15 | 12 → 12 | 3 | 40.39° → 40.39° | 未通过 | 88.86 s |

最终批次两进程运行的墙钟耗时 **420.85 秒，约 7 分 1 秒**，包含本批次初始化、头编辑、CPU 载荷检查、完整构造和 Step4/5 图片；不包含前期开发试跑或之后的视频渲染。

集中角是在相同的最终方向菜单上，计算每个 pose 距共同参考方向最近的可行方向，再取平均。参考方向不等于所有 pose 已能使用的方向。完整夹具的方向集合另行记录，不能用头级结果替代。

两个通过组的占用 XYZ 盒体积均没有减少。pose1+3 没有接受削减；pose3+6 接受一次削减，材料体积从 27.886293 cm³ 降至 27.850270 cm³，约 0.13%，但退出集中角和占用盒体积未改善。

失败组中三个被连接端口搜索拒绝，四个被完整几何接受拒绝。pose1+2+8+17 和 pose2+9+13+15+17 在本轮初始完整构造也未通过，不能把其最终失败全部归因于削减；原 DSL 的已有成功方案没有被这些候选替换。

这一版本只验证有限切除体、有限参考方向以及有预算的连接构造。结果说明当前体积梯度没有实现退出方向趋近，且头自身连通／保留局部连接见证不足以保证最后完整构造通过。不能据此证明两操作本身无解，也不能把阻挡损失下降或接触面积下降当作目标完成。

接触区域可能重叠，现有面积汇总是各区域面积之和，不是唯一接触并集面积；面积不作为奖励。初始化中的计数降低来自已有相连头的身份规范化，本轮编辑没有进一步减少物理头数量。

逐组报告、每次切除、梯度和拒绝原因位于 output/B/<pose_set>/step3_scheculer/global_release_v7/。完整通过的模型与两张英文图片在对应 step4/global_release_v7/；Step5 数据在 step5_evaluate/global_release_v7/。视频显示各 pose 的接触、实际头实体和检查过的退出方向点集，标注头级过程与最终完整构造状态。

[汇总 JSON](output/B/pose6+8+10+19/step5_evaluate/global_release_v7/all_nine_results.json)

| Pose set | 逐组报告 | 头优化视频 |
|---|---|---|
| pose1+3 | [report](output/B/pose1+3/step3_scheculer/global_release_v7/report.json) | [video](vis/vis_result/B_pose1+3_global_release_v7.mp4) |
| pose1+2+3+4+5 | [report](output/B/pose1+2+3+4+5/step3_scheculer/global_release_v7/report.json) | [video](vis/vis_result/B_pose1+2+3+4+5_global_release_v7.mp4) |
| pose1+2+8+17 | [report](output/B/pose1+2+8+17/step3_scheculer/global_release_v7/report.json) | [video](vis/vis_result/B_pose1+2+8+17_global_release_v7.mp4) |
| pose2+10+15 | [report](output/B/pose2+10+15/step3_scheculer/global_release_v7/report.json) | [video](vis/vis_result/B_pose2+10+15_global_release_v7.mp4) |
| pose2+12+15 | [report](output/B/pose2+12+15/step3_scheculer/global_release_v7/report.json) | [video](vis/vis_result/B_pose2+12+15_global_release_v7.mp4) |
| pose2+9+13+15+17 | [report](output/B/pose2+9+13+15+17/step3_scheculer/global_release_v7/report.json) | [video](vis/vis_result/B_pose2+9+13+15+17_global_release_v7.mp4) |
| pose3+6 | [report](output/B/pose3+6/step3_scheculer/global_release_v7/report.json) | [video](vis/vis_result/B_pose3+6_global_release_v7.mp4) |
| pose5+7 | [report](output/B/pose5+7/step3_scheculer/global_release_v7/report.json) | [video](vis/vis_result/B_pose5+7_global_release_v7.mp4) |
| pose6+8+10+19 | [report](output/B/pose6+8+10+19/step3_scheculer/global_release_v7/report.json) | [video](vis/vis_result/B_pose6+8+10+19_global_release_v7.mp4) |
