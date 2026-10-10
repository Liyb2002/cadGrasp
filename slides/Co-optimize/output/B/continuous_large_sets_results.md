# 已停止的v5十组试跑

从各组保存的 Step4.1 开始，全部 pose 始终参与；历史优化答案未作为起点。


通过要求全部原始 32768 载荷/pose 与真实工作禁区、退出及接触几何检查通过。

| Pose set | 状态 | 实体材料 cm³ | 结果 |
|---|---|---:|---|
| pose1+2+3+4+5+6+7+8+9 | 待完成 | — | — |
| pose1+2+3+4+5+6+7+8+9+10 | unresolved | — | [日志](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/continuous_gradient_v5/data/run.log) |
| pose1+2+4+5+6+7+10+11 | 待完成 | — | — |
| pose1+3+7+10+14+18+23+27 | 待完成 | — | — |
| pose1+4+7+9+12+16+21+23+27 | unresolved | — | [日志](pose1+4+7+9+12+16+21+23+27/step4/step4.2/continuous_gradient_v5/data/run.log) |
| pose1+3+5+7+9+12+16+21+23+27 | unresolved | — | [日志](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/continuous_gradient_v5/data/run.log) |
| pose1+2+4+5+6+7+11 | unresolved | — | [日志](pose1+2+4+5+6+7+11/step4/step4.2/continuous_gradient_v5/data/run.log) |
| pose1+2+3+4+5+6+7 | 待完成 | — | — |
| pose1+2+3+4+5+6+7+8 | 待完成 | — | — |
| pose1+4+7+12+21+23+27 | 待完成 | — | — |

[批次与完整预算](data/continuous_large_sets_v5/batch.json)。

连续求积属于搜索指导，未证明整个连续需求域。未解决不表示数学无解。

用户已停止：3组搜索完成、0组通过；7组未完成。额外取消诊断不计为完成的搜索。

[恢复旧成功搜索结构的新代码和单组真实检查](../../step4.2/reuse_gradient_repair.md)。
