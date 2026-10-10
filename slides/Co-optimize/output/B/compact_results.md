# Whole 实体体积优化对照

所有方法从各自同一个真实可行Step4.2结果继续；体积为支撑实体，而非包围盒。新布局保留全组原载荷及完整工作/退出约束；实际候选不通过时保留原结果。方法启发式，不能证明全局最优。

| Pose set | 方法 | 基线 cm³ | 最终 cm³ | 减少 | 图 |
|---|---|---:|---:|---:|---|
| pose1+2+3+4+7+27 | greedy | 85.193 | 75.983 | 10.81% | [过程](pose1+2+3+4+7+27/step4/step4.2/compact/greedy/process.png) · [最终](pose1+2+3+4+7+27/step4/step4.2/compact/greedy/final_result.png) |
| pose1+2+3+4+7+27 | beam | 85.193 | 76.171 | 10.59% | [过程](pose1+2+3+4+7+27/step4/step4.2/compact/beam/process.png) · [最终](pose1+2+3+4+7+27/step4/step4.2/compact/beam/final_result.png) |
| pose1+2+4+5+6+11 | greedy | 99.636 | 87.392 | 12.29% | [过程](pose1+2+4+5+6+11/step4/step4.2/compact/greedy/process.png) · [最终](pose1+2+4+5+6+11/step4/step4.2/compact/greedy/final_result.png) |
| pose1+2+4+5+6+11 | structural-greedy | 99.636 | 83.175 | 16.52% | [过程](pose1+2+4+5+6+11/step4/step4.2/compact/structural-greedy/process.png) · [最终](pose1+2+4+5+6+11/step4/step4.2/compact/structural-greedy/final_result.png) |
| pose1+2+4+5+6+11 | beam | 99.636 | 76.971 | 22.75% | [过程](pose1+2+4+5+6+11/step4/step4.2/compact/beam/process.png) · [最终](pose1+2+4+5+6+11/step4/step4.2/compact/beam/final_result.png) |
| pose19+28 | greedy | 139.831 | 116.646 | 16.58% | [过程](pose19+28/step4/step4.2/compact/greedy/process.png) · [最终](pose19+28/step4/step4.2/compact/greedy/final_result.png) |
| pose19+28 | structural-greedy | 139.831 | 102.883 | 26.42% | [过程](pose19+28/step4/step4.2/compact/structural-greedy/process.png) · [最终](pose19+28/step4/step4.2/compact/structural-greedy/final_result.png) |
| pose19+28 | beam | 139.831 | 55.604 | 60.24% | [过程](pose19+28/step4/step4.2/compact/beam/process.png) · [最终](pose19+28/step4/step4.2/compact/beam/final_result.png) |
| pose4+5+7 | greedy | 70.287 | 45.230 | 35.65% | [过程](pose4+5+7/step4/step4.2/compact/greedy/process.png) · [最终](pose4+5+7/step4/step4.2/compact/greedy/final_result.png) |
| pose4+5+7 | beam | 70.287 | 50.454 | 28.22% | [过程](pose4+5+7/step4/step4.2/compact/beam/process.png) · [最终](pose4+5+7/step4/step4.2/compact/beam/final_result.png) |
| pose4+5+8+9+19+23 | greedy | 92.802 | 82.607 | 10.99% | [过程](pose4+5+8+9+19+23/step4/step4.2/compact/greedy/process.png) · [最终](pose4+5+8+9+19+23/step4/step4.2/compact/greedy/final_result.png) |
| pose4+5+8+9+19+23 | beam | 92.802 | 66.207 | 28.66% | [过程](pose4+5+8+9+19+23/step4/step4.2/compact/beam/process.png) · [最终](pose4+5+8+9+19+23/step4/step4.2/compact/beam/final_result.png) |
| pose5+6+11+23+29 | greedy | 19.410 | 16.792 | 13.49% | [过程](pose5+6+11+23+29/step4/step4.2/compact/greedy/process.png) · [最终](pose5+6+11+23+29/step4/step4.2/compact/greedy/final_result.png) |
| pose5+6+11+23+29 | beam | 19.410 | 19.410 | 0.00% | [过程](pose5+6+11+23+29/step4/step4.2/compact/beam/process.png) · [最终](pose5+6+11+23+29/step4/step4.2/compact/beam/final_result.png) |
