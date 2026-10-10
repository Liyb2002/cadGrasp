# B：重新初始化与连续 Step4.2

全部旧 Step4.1 从活动目录删除后重建；所有集合重新运行 Direction、Translation 与 Juxtapose。原 Step3、工作面、物体、32768载荷/pose和物理容差保持不变。

已完成 34/52；实际通过 11；未解决 23。

[图片浏览](continuous_step4_index.html) · [批次记录](data/continuous_step4_batch.json) · [旧结果归档](_history/before_continuous_B_rerun_20261009_143409/)

通过指最终真实网格满足全部原载荷与原工作／退出检查；初始化承载通过但优化未获更小有效结果时，保留本次重建的初始化。连续需求求积与移动接触边界属于近似，未证明整个连续需求域全覆盖。

| Pose set | 状态 | 初始化承载 | 最终材料 cm³ | 结果 |
|---|---|---|---:|---|
| illegal/pose1+2+29 | 运行中／待运行 | — | — | — |
| illegal/pose1+2+6+15 | pass | True | 90.228 | [初始化](illegal/pose1+2+6+15/step4/step4.1/final_result.png) · [过程](illegal/pose1+2+6+15/step4/step4.2/continuous/process.png) · [最终](illegal/pose1+2+6+15/step4/step4.2/continuous/final_result.png) |
| illegal/pose1+2+6+7+29 | pass | True | 63.667 | [初始化](illegal/pose1+2+6+7+29/step4/step4.1/final_result.png) · [过程](illegal/pose1+2+6+7+29/step4/step4.2/continuous/process.png) · [最终](illegal/pose1+2+6+7+29/step4/step4.2/continuous/final_result.png) |
| illegal/pose11+19 | 运行中／待运行 | — | — | — |
| illegal/pose2+29 | 运行中／待运行 | — | — | — |
| illegal/pose4+7+12+21+23+27 | optimization_unresolved | False | 0.000 | [初始化](illegal/pose4+7+12+21+23+27/step4/step4.1/final_result.png) · [日志](illegal/pose4+7+12+21+23+27/step4/step4.2/continuous/data/run.log) |
| illegal/pose5+12+20+24+27+28 | pass | True | 114.427 | [初始化](illegal/pose5+12+20+24+27+28/step4/step4.1/final_result.png) · [过程](illegal/pose5+12+20+24+27+28/step4/step4.2/continuous/process.png) · [最终](illegal/pose5+12+20+24+27+28/step4/step4.2/continuous/final_result.png) |
| illegal/pose5+13+27 | 运行中／待运行 | — | — | — |
| illegal/pose5+6+13+15+27 | pass | True | 128.058 | [初始化](illegal/pose5+6+13+15+27/step4/step4.1/final_result.png) · [过程](illegal/pose5+6+13+15+27/step4/step4.2/continuous/process.png) · [最终](illegal/pose5+6+13+15+27/step4/step4.2/continuous/final_result.png) |
| illegal/pose7+11+13+19 | 运行中／待运行 | — | — | — |
| pose1+11+14+27 | pass | True | 31.512 | [初始化](pose1+11+14+27/step4/step4.1/final_result.png) · [过程](pose1+11+14+27/step4/step4.2/continuous/process.png) · [最终](pose1+11+14+27/step4/step4.2/continuous/final_result.png) |
| pose1+12+29 | 运行中／待运行 | — | — | — |
| pose1+2+3+27 | 运行中／待运行 | — | — | — |
| pose1+2+3+4+27 | pass | True | 58.700 | [初始化](pose1+2+3+4+27/step4/step4.1/final_result.png) · [过程](pose1+2+3+4+27/step4/step4.2/continuous/process.png) · [最终](pose1+2+3+4+27/step4/step4.2/continuous/final_result.png) |
| pose1+2+3+4+5+6 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+3+4+5+6/step4/step4.1/final_result.png) · [日志](pose1+2+3+4+5+6/step4/step4.2/continuous/data/run.log) |
| pose1+2+3+4+6 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+3+4+6/step4/step4.1/final_result.png) · [日志](pose1+2+3+4+6/step4/step4.2/continuous/data/run.log) |
| pose1+2+3+4+6+7 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+3+4+6+7/step4/step4.1/final_result.png) · [日志](pose1+2+3+4+6+7/step4/step4.2/continuous/data/run.log) |
| pose1+2+3+4+7+27 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+3+4+7+27/step4/step4.1/final_result.png) · [日志](pose1+2+3+4+7+27/step4/step4.2/continuous/data/run.log) |
| pose1+2+3+7+27 | pass | True | 44.077 | [初始化](pose1+2+3+7+27/step4/step4.1/final_result.png) · [过程](pose1+2+3+7+27/step4/step4.2/continuous/process.png) · [最终](pose1+2+3+7+27/step4/step4.2/continuous/final_result.png) |
| pose1+2+4+19 | pass | True | 53.006 | [初始化](pose1+2+4+19/step4/step4.1/final_result.png) · [过程](pose1+2+4+19/step4/step4.2/continuous/process.png) · [最终](pose1+2+4+19/step4/step4.2/continuous/final_result.png) |
| pose1+2+4+5+6 | pass | True | 82.123 | [初始化](pose1+2+4+5+6/step4/step4.1/final_result.png) · [过程](pose1+2+4+5+6/step4/step4.2/continuous/process.png) · [最终](pose1+2+4+5+6/step4/step4.2/continuous/final_result.png) |
| pose1+2+4+5+6+11 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+4+5+6+11/step4/step4.1/final_result.png) · [日志](pose1+2+4+5+6+11/step4/step4.2/continuous/data/run.log) |
| pose1+2+4+5+6+7 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+4+5+6+7/step4/step4.1/final_result.png) · [日志](pose1+2+4+5+6+7/step4/step4.2/continuous/data/run.log) |
| pose1+2+4+6 | pass | True | 82.325 | [初始化](pose1+2+4+6/step4/step4.1/final_result.png) · [过程](pose1+2+4+6/step4/step4.2/continuous/process.png) · [最终](pose1+2+4+6/step4/step4.2/continuous/final_result.png) |
| pose1+2+4+7 | 运行中／待运行 | — | — | — |
| pose1+4+7+12+21+27 | optimization_unresolved | False | 0.000 | [初始化](pose1+4+7+12+21+27/step4/step4.1/final_result.png) · [日志](pose1+4+7+12+21+27/step4/step4.2/continuous/data/run.log) |
| pose1+4+7+9+24 | optimization_unresolved | False | 0.000 | [初始化](pose1+4+7+9+24/step4/step4.1/final_result.png) · [日志](pose1+4+7+9+24/step4/step4.2/continuous/data/run.log) |
| pose1+6+11+13+14+17 | optimization_unresolved | False | 0.000 | [初始化](pose1+6+11+13+14+17/step4/step4.1/final_result.png) · [日志](pose1+6+11+13+14+17/step4/step4.2/continuous/data/run.log) |
| pose12+16+19+21+27 | pass | True | 40.995 | [初始化](pose12+16+19+21+27/step4/step4.1/final_result.png) · [过程](pose12+16+19+21+27/step4/step4.2/continuous/process.png) · [最终](pose12+16+19+21+27/step4/step4.2/continuous/final_result.png) |
| pose18+23+24 | 运行中／待运行 | — | — | — |
| pose19+28 | 运行中／待运行 | — | — | — |
| pose2+20 | 运行中／待运行 | — | — | — |
| pose2+3+4+7 | 运行中／待运行 | — | — | — |
| pose3+15 | 运行中／待运行 | — | — | — |
| pose4+5+7 | 运行中／待运行 | — | — | — |
| pose4+5+8+9+19+23 | optimization_unresolved | False | 0.000 | [初始化](pose4+5+8+9+19+23/step4/step4.1/final_result.png) · [日志](pose4+5+8+9+19+23/step4/step4.2/continuous/data/run.log) |
| pose5+6+11+23+29 | optimization_unresolved | False | 0.000 | [初始化](pose5+6+11+23+29/step4/step4.1/final_result.png) · [日志](pose5+6+11+23+29/step4/step4.2/continuous/data/run.log) |
| pose5+6+23+29 | 运行中／待运行 | — | — | — |
| pose6+10+13+17+30 | optimization_unresolved | False | 0.000 | [初始化](pose6+10+13+17+30/step4/step4.1/final_result.png) · [日志](pose6+10+13+17+30/step4/step4.2/continuous/data/run.log) |
| pose8+10+19 | 运行中／待运行 | — | — | — |
| pose8+21 | 运行中／待运行 | — | — | — |
| pose8+9+13+30 | 运行中／待运行 | — | — | — |
| pose1+2+3+4+5+6+7+8+9 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+3+4+5+6+7+8+9/step4/step4.1/final_result.png) · [日志](pose1+2+3+4+5+6+7+8+9/step4/step4.2/continuous/data/run.log) |
| pose1+2+3+4+5+6+7+8+9+10 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+3+4+5+6+7+8+9+10/step4/step4.1/final_result.png) · [日志](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/continuous/data/run.log) |
| pose1+2+4+5+6+7+10+11 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+4+5+6+7+10+11/step4/step4.1/final_result.png) · [日志](pose1+2+4+5+6+7+10+11/step4/step4.2/continuous/data/run.log) |
| pose1+3+7+10+14+18+23+27 | optimization_unresolved | False | 0.000 | [初始化](pose1+3+7+10+14+18+23+27/step4/step4.1/final_result.png) · [日志](pose1+3+7+10+14+18+23+27/step4/step4.2/continuous/data/run.log) |
| pose1+4+7+9+12+16+21+23+27 | optimization_unresolved | False | 0.000 | [初始化](pose1+4+7+9+12+16+21+23+27/step4/step4.1/final_result.png) · [日志](pose1+4+7+9+12+16+21+23+27/step4/step4.2/continuous/data/run.log) |
| pose1+3+5+7+9+12+16+21+23+27 | optimization_unresolved | False | 0.000 | [初始化](pose1+3+5+7+9+12+16+21+23+27/step4/step4.1/final_result.png) · [日志](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/continuous/data/run.log) |
| pose1+2+4+5+6+7+11 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+4+5+6+7+11/step4/step4.1/final_result.png) · [日志](pose1+2+4+5+6+7+11/step4/step4.2/continuous/data/run.log) |
| pose1+2+3+4+5+6+7 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+3+4+5+6+7/step4/step4.1/final_result.png) · [日志](pose1+2+3+4+5+6+7/step4/step4.2/continuous/data/run.log) |
| pose1+2+3+4+5+6+7+8 | optimization_unresolved | False | 0.000 | [初始化](pose1+2+3+4+5+6+7+8/step4/step4.1/final_result.png) · [日志](pose1+2+3+4+5+6+7+8/step4/step4.2/continuous/data/run.log) |
| pose1+4+7+12+21+23+27 | optimization_unresolved | False | 0.000 | [初始化](pose1+4+7+12+21+23+27/step4/step4.1/final_result.png) · [日志](pose1+4+7+12+21+23+27/step4/step4.2/continuous/data/run.log) |

预算、失败候选和回溯保存在每组新目录；未解决不冒充无解证明。完整夹具接地、连通与强度尚未验收。
