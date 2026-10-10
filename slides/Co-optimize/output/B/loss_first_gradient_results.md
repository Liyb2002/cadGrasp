# 快速全组梯度：7–10 pose 结果

已完成 10/10 组，原完整载荷与真实几何通过 9/10。

所有组的原始起点是保存的 Step4.1，完整起点数组已核对。标注“接续”的实验随后加载自己的前一轮状态，额外预算独立保存；不能称为一次统一预算冷启动。方向／平移以全组锥距离为目标；有限接触分辨率的数值梯度加少量候选采样，Juxtapose 为离散跳步。

| Pose set | 状态 | 本轮搜索 s | 本轮检查 s | 冷启动至本轮累计 s | 实体 cm³ | 转动复用／J | 已提交梯度步下界 | 图 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pose1+2+3+4+5+6+7 | 通过（冷启动） | 351.3 | 62.9 | 452.5 | 98.47 | 2/5 | 6 | [过程](pose1+2+3+4+5+6+7/step4/step4.2/loss_first_gradient_v14/process.png) · [最终](pose1+2+3+4+5+6+7/step4/step4.2/loss_first_gradient_v14/final_result.png) |
| pose1+2+4+5+6+7+11 | 通过（冷启动） | 219.4 | 270.2 | 519.0 | 65.02 | 5/2 | 2 | [过程](pose1+2+4+5+6+7+11/step4/step4.2/loss_first_gradient_v14/process.png) · [最终](pose1+2+4+5+6+7+11/step4/step4.2/loss_first_gradient_v14/final_result.png) |
| pose1+4+7+12+21+23+27 | 通过（冷启动） | 249.4 | 44.8 | 329.7 | 93.01 | 4/3 | 8 | [过程](pose1+4+7+12+21+23+27/step4/step4.2/loss_first_gradient_v14/process.png) · [最终](pose1+4+7+12+21+23+27/step4/step4.2/loss_first_gradient_v14/final_result.png) |
| pose1+2+3+4+5+6+7+8 | 通过（冷启动） | 627.9 | 211.4 | 878.3 | 84.12 | 2/6 | 8 | [过程](pose1+2+3+4+5+6+7+8/step4/step4.2/loss_first_gradient_v14/process.png) · [最终](pose1+2+3+4+5+6+7+8/step4/step4.2/loss_first_gradient_v14/final_result.png) |
| pose1+2+4+5+6+7+10+11 | 通过（冷启动） | 371.2 | 345.5 | 774.8 | 171.00 | 3/5 | 10 | [过程](pose1+2+4+5+6+7+10+11/step4/step4.2/loss_first_gradient_v14/process.png) · [最终](pose1+2+4+5+6+7+10+11/step4/step4.2/loss_first_gradient_v14/final_result.png) |
| pose1+3+7+10+14+18+23+27 | 通过（冷启动） | 137.8 | 34.7 | 201.4 | 35.85 | 7/1 | 2 | [过程](pose1+3+7+10+14+18+23+27/step4/step4.2/loss_first_gradient_v14/process.png) · [最终](pose1+3+7+10+14+18+23+27/step4/step4.2/loss_first_gradient_v14/final_result.png) |
| pose1+2+3+4+5+6+7+8+9 | 通过（冷启动） | 485.7 | 96.4 | 642.5 | 119.49 | 4/5 | 5 | [过程](pose1+2+3+4+5+6+7+8+9/step4/step4.2/loss_first_gradient_v14/process.png) · [最终](pose1+2+3+4+5+6+7+8+9/step4/step4.2/loss_first_gradient_v14/final_result.png) |
| pose1+4+7+9+12+16+21+23+27 | 通过（冷启动） | 683.4 | 83.6 | 822.7 | 90.05 | 4/5 | 15 | [过程](pose1+4+7+9+12+16+21+23+27/step4/step4.2/loss_first_gradient_v14/process.png) · [最终](pose1+4+7+9+12+16+21+23+27/step4/step4.2/loss_first_gradient_v14/final_result.png) |
| pose1+2+3+4+5+6+7+8+9+10 | 通过（冷启动） | 586.3 | 179.8 | 822.0 | 90.60 | 4/6 | 9 | [过程](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/loss_first_gradient_v14/process.png) · [最终](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/loss_first_gradient_v14/final_result.png) |
| pose1+3+5+7+9+12+16+21+23+27 | 未解决 | — | — | — | — | — | — | [记录](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/loss_first_gradient_v14/run_result.json) |

所选本轮搜索耗时：中位数 371.2 s，范围 137.8–683.4 s；有接续时不代表从头搜索。
所选方案从冷启动至本轮的累计运行（含该来源链的失败检查及出图）：中位数 642.5 s，范围 201.4–878.3 s。这不含其他弃用实验的研发成本。

梯度步列只计过程图中明确已提交的梯度步，不把被丢弃分支的局部接受步算进成功路径。分支完整统计、逐步预算和原始耗时在 JSON 中保留。

接续行的耗时只对应该次额外修复，不包含前一轮搜索；完整冷启动结论应单独查看同一批次的十组结果。候选几何超时与失败都记入末次检查耗时，没有计为通过。

实体体积来自实际接受网格，不是包围盒。每个 pose 原 32768 个需求以及工作／退出／核心／净空检查保持原定义；整件安装、连通与强度检查范围与原实验相同。

[算法与公式](../../step4.2/fast_gradient_algorithm.md) · [完整数据](data/loss_first_gradient_results.json) · [图片浏览](loss_first_gradient_index.html)。
