# 快速全组梯度：7–10 pose 结果

已完成 10/10 组，原完整载荷与真实几何通过 10/10。

所有组的原始起点是保存的 Step4.1，完整起点数组已核对。标注“接续”的实验随后加载自己的前一轮状态，额外预算独立保存；不能称为一次统一预算冷启动。方向／平移以全组锥距离为目标；有限接触分辨率的数值梯度加少量候选采样，Juxtapose 为离散跳步。

| Pose set | 状态 | 本轮搜索 s | 本轮检查 s | 冷启动至本轮累计 s | 实体 cm³ | 转动复用／J | 已提交梯度步下界 | 图 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pose1+2+3+4+5+6+7 | 通过（冷启动） | 198.4 | 30.1 | 240.2 | 92.69 | 2/5 | 1 | [过程](pose1+2+3+4+5+6+7/step4/step4.2/fast_gradient_v3/process.png) · [最终](pose1+2+3+4+5+6+7/step4/step4.2/fast_gradient_v3/final_result.png) |
| pose1+2+4+5+6+7+11 | 通过（冷启动） | 364.7 | 54.1 | 448.7 | 115.28 | 4/3 | 3 | [过程](pose1+2+4+5+6+7+11/step4/step4.2/active_gradient_v10/process.png) · [最终](pose1+2+4+5+6+7+11/step4/step4.2/active_gradient_v10/final_result.png) |
| pose1+4+7+12+21+23+27 | 通过（冷启动） | 82.2 | 16.9 | 110.2 | 33.21 | 5/2 | 4 | [过程](pose1+4+7+12+21+23+27/step4/step4.2/fast_gradient_v3/process.png) · [最终](pose1+4+7+12+21+23+27/step4/step4.2/fast_gradient_v3/final_result.png) |
| pose1+2+3+4+5+6+7+8 | 通过（冷启动） | 522.2 | 38.7 | 590.0 | 97.81 | 3/5 | 5 | [过程](pose1+2+3+4+5+6+7+8/step4/step4.2/active_gradient_v10/process.png) · [最终](pose1+2+3+4+5+6+7+8/step4/step4.2/active_gradient_v10/final_result.png) |
| pose1+2+4+5+6+7+10+11 | 通过（冷启动） | 367.2 | 224.5 | 606.9 | 90.58 | 3/5 | 1 | [过程](pose1+2+4+5+6+7+10+11/step4/step4.2/fast_gradient_v3/process.png) · [最终](pose1+2+4+5+6+7+10+11/step4/step4.2/fast_gradient_v3/final_result.png) |
| pose1+3+7+10+14+18+23+27 | 通过（冷启动） | 49.1 | 213.8 | 282.3 | 30.19 | 7/1 | 1 | [过程](pose1+3+7+10+14+18+23+27/step4/step4.2/fast_gradient_v3/process.png) · [最终](pose1+3+7+10+14+18+23+27/step4/step4.2/fast_gradient_v3/final_result.png) |
| pose1+2+3+4+5+6+7+8+9 | 通过（自身状态接续） | 47.7 | 122.0 | 1135.0 | 116.22 | 4/5 | 1 | [过程](pose1+2+3+4+5+6+7+8+9/step4/step4.2/descent_gradient_probe_v5/process.png) · [最终](pose1+2+3+4+5+6+7+8+9/step4/step4.2/descent_gradient_probe_v5/final_result.png) |
| pose1+4+7+9+12+16+21+23+27 | 通过（自身状态接续） | 351.8 | 143.2 | 1364.9 | 138.89 | 3/6 | 0 | [过程](pose1+4+7+9+12+16+21+23+27/step4/step4.2/descent_gradient_repairs_v6/process.png) · [最终](pose1+4+7+9+12+16+21+23+27/step4/step4.2/descent_gradient_repairs_v6/final_result.png) |
| pose1+2+3+4+5+6+7+8+9+10 | 通过（冷启动） | 893.3 | 102.6 | 1035.9 | 177.46 | 3/7 | 1 | [过程](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/fast_gradient_v3/process.png) · [最终](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/fast_gradient_v3/final_result.png) |
| pose1+3+5+7+9+12+16+21+23+27 | 通过（冷启动） | 864.6 | 105.2 | 1077.8 | 38.53 | 8/2 | 6 | [过程](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/fast_gradient_v3/process.png) · [最终](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/fast_gradient_v3/final_result.png) |

所选本轮搜索耗时：中位数 358.2 s，范围 47.7–893.3 s；有接续时不代表从头搜索。
所选方案从冷启动至本轮的累计运行（含该来源链的失败检查及出图）：中位数 598.4 s，范围 110.2–1364.9 s。这不含其他弃用实验的研发成本。

梯度步列只计过程图中明确已提交的梯度步，不把被丢弃分支的局部接受步算进成功路径。分支完整统计、逐步预算和原始耗时在 JSON 中保留。

接续行的耗时只对应该次额外修复，不包含前一轮搜索；完整冷启动结论应单独查看同一批次的十组结果。候选几何超时与失败都记入末次检查耗时，没有计为通过。

实体体积来自实际接受网格，不是包围盒。每个 pose 原 32768 个需求以及工作／退出／核心／净空检查保持原定义；整件安装、连通与强度检查范围与原实验相同。

[算法与公式](../../step4.2/fast_gradient_algorithm.md) · [完整数据](data/fast_gradient_results.json) · [图片浏览](fast_gradient_index.html)。
