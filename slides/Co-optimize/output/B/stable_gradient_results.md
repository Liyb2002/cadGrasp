# 快速全组梯度：7–10 pose 结果

已完成 10/10 组，原完整载荷与真实几何通过 10/10。

所有组的原始起点是保存的 Step4.1，完整起点数组已核对。标注“接续”的实验随后加载自己的前一轮状态，额外预算独立保存；不能称为一次统一预算冷启动。方向／平移以全组锥距离为目标；有限接触分辨率的数值梯度加少量候选采样，Juxtapose 为离散跳步。

同一个支撑可以有多个摆放 state，每个 state 可支持多个 object pose，没有两个 pose 的容量限制。state 按实际支撑世界变换分组；Juxtapose 的 guest/host 是操作参数，不是容量。

| Pose set | 状态 | 来源链总搜索 s | 本轮检查 s | 冷启动至本轮累计 s | 实体 cm³ | 转动复用／J | state数／最大fit数 | 来源链提交梯度步下界 | 图 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pose1+3+5+7+9+12+16+21+23+27 | 通过（冷启动） | 226.6 | 66.0 | 347.8 | 58.47 | 7/3 | 7/3 | 3 | [过程](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_cold_v17/process.png) · [最终](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_cold_v17/final_result.png) · [states](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_cold_v17/state_groups.json) |
| pose1+2+3+4+5+6+7 | 通过（冷启动） | 121.6 | 44.2 | 189.4 | 94.31 | 4/3 | 4/3 | 3 | [过程](pose1+2+3+4+5+6+7/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+3+4+5+6+7/step4/step4.2/stable_gradient_remaining_v17/final_result.png) · [states](pose1+2+3+4+5+6+7/step4/step4.2/stable_gradient_remaining_v17/state_groups.json) |
| pose1+2+4+5+6+7+11 | 通过（冷启动） | 311.3 | 47.2 | 401.9 | 86.59 | 3/4 | 3/4 | 4 | [过程](pose1+2+4+5+6+7+11/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+4+5+6+7+11/step4/step4.2/stable_gradient_remaining_v17/final_result.png) · [states](pose1+2+4+5+6+7+11/step4/step4.2/stable_gradient_remaining_v17/state_groups.json) |
| pose1+4+7+12+21+23+27 | 通过（自身状态接续） | 229.5 | 20.2 | 661.8 | 141.26 | 3/4 | 3/4 | 11 | [过程](pose1+4+7+12+21+23+27/step4/step4.2/stable_joint_conditioning_v17/process.png) · [最终](pose1+4+7+12+21+23+27/step4/step4.2/stable_joint_conditioning_v17/final_result.png) · [states](pose1+4+7+12+21+23+27/step4/step4.2/stable_joint_conditioning_v17/state_groups.json) |
| pose1+2+3+4+5+6+7+8 | 通过（冷启动） | 232.4 | 70.7 | 336.2 | 100.56 | 5/3 | 5/3 | 4 | [过程](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_remaining_v17/final_result.png) · [states](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_remaining_v17/state_groups.json) |
| pose1+2+4+5+6+7+10+11 | 通过（冷启动） | 252.6 | 45.1 | 331.7 | 146.65 | 3/5 | 4/5 | 8 | [过程](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_remaining_v17/final_result.png) · [states](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_remaining_v17/state_groups.json) |
| pose1+3+7+10+14+18+23+27 | 通过（冷启动） | 85.0 | 25.2 | 133.3 | 68.23 | 6/2 | 6/2 | 4 | [过程](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_remaining_v17/final_result.png) · [states](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_remaining_v17/state_groups.json) |
| pose1+2+3+4+5+6+7+8+9 | 通过（冷启动） | 477.1 | 258.5 | 779.1 | 90.19 | 5/4 | 5/3 | 6 | [过程](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_remaining_v17/final_result.png) · [states](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_remaining_v17/state_groups.json) |
| pose1+4+7+9+12+16+21+23+27 | 通过（自身状态接续） | 667.9 | 228.5 | 1035.0 | 111.30 | 3/6 | 4/5 | 9 | [过程](pose1+4+7+9+12+16+21+23+27/step4/step4.2/fast_state_seat_gradient_repair_v21/process.png) · [最终](pose1+4+7+9+12+16+21+23+27/step4/step4.2/fast_state_seat_gradient_repair_v21/final_result.png) · [states](pose1+4+7+9+12+16+21+23+27/step4/step4.2/fast_state_seat_gradient_repair_v21/state_groups.json) |
| pose1+2+3+4+5+6+7+8+9+10 | 通过（冷启动） | 624.3 | 281.0 | 982.4 | 139.09 | 5/5 | 5/5 | 11 | [过程](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_remaining_v17/final_result.png) · [states](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_remaining_v17/state_groups.json) |

所选方案的完整来源链搜索（含额外conditioning筛选、不含真实检查／出图）：中位数 242.5 s，范围 85.0–667.9 s。
所选方案从冷启动至本轮的累计运行（含该来源链的失败检查及出图）：中位数 374.8 s，范围 133.3–1035.0 s。这不含其他弃用实验的研发成本。

梯度步列只计过程图中明确已提交的梯度步，不把被丢弃分支的局部接受步算进成功路径。分支完整统计、逐步预算和原始耗时在 JSON 中保留。

“来源链总搜索”和“冷启动至本轮累计”均包含接续之前的本算法搜索；“本轮检查”只列最终这一轮的检查成本。其他弃用试跑不混入该来源链。候选几何超时与失败都记入相应检查耗时，没有计为通过。

实体体积来自实际接受网格，不是包围盒。每个 pose 原 32768 个需求以及工作／退出／核心／净空检查保持原定义；整件安装、连通与强度检查范围与原实验相同。

[算法与公式](../../step4.2/fast_gradient_algorithm.md) · [完整数据](data/stable_gradient_results.json) · [图片浏览](stable_gradient_index.html)。
