# 连通接触主体：全52组Step4结果

批次 `connected_core_all52_20261013_final`：已结束 52/52 组，完整通过 52 组，失败 0 组。批次状态：完成。各次独立重试及最终选择来源保留在批次文件中。

全部集合共 263 个object pose。当前通过组共 263 个object pose、123 个支撑安装pose。失败组的安装数量仅描述保存的诊断布局。

Juxtapose采样数量中心为每组object pose数/3；只影响新增/取消概率，不是优化目标、通过条件或数量惩罚。最终数量由物理结果产生。

冷启动注册初始化，种群数量按组为 {32: 50, 48: 2}，最多8轮；取得实体认证后提前返回。每轮先离散换座或种群差分，再一次短调整；Juxtapose身份由优化选择。没有历史优化布局暖启动，也没有数量目标。

[算法说明](../../../connected_core_algorithm.md)。每组 `final_result.png` 显示实际已认证的连通接触主体；`population.png` 显示该次执行的全部槽位。

PASS要求保留主体为一个真实材料连通分量，原向下合力模型、全部原始力/力矩需求、Oracle及长脚可行性参考通过。名义BBX对应接触主体；成品BBX对应搜索内长脚参考，最终Step5.3另见Evaluation报告。

| 每组object pose | 组数 | 已结束 | PASS | 失败 |
|---:|---:|---:|---:|---:|
| 2 | 6 | 6 | 6 | 0 |
| 3 | 6 | 6 | 6 | 0 |
| 4 | 10 | 10 | 10 | 0 |
| 5 | 10 | 10 | 10 | 0 |
| 6 | 10 | 10 | 10 | 0 |
| 7 | 3 | 3 | 3 | 0 |
| 8 | 3 | 3 | 3 | 0 |
| 9 | 2 | 2 | 2 | 0 |
| 10 | 2 | 2 | 2 | 0 |

通过组中完整BBX最大的集合为 `pose1+3+5+7+9+12+16+21+23+27`：22.73L。完整通过也可能有较大的成品占用盒，须结合下方尺寸列判断搜索效果。

| Set | 状态 | object pose | 支撑pose | 原位复用 | Juxtapose | 采样中心N/3 | 未通过原需求数 | Oracle | 无支撑BBX L | 名义BBX L | 成品BBX L | 成品材料cm³ | 图 |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| pose1+2+3+4+5+6+7+8+9+10 | PASS | 10 | 1 | 1 | 9 | 3.33 | 0 | common_connected_enclosure_witness | 5.27 | 20.92 | 20.92 | 422.68 | [布局](../../../output/B/pose1+2+3+4+5+6+7+8+9+10/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+4+5+6+7+8+9+10/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+4+5+6+7+8+9+10/step4.2/evaluation.png) |
| pose1+3+5+7+9+12+16+21+23+27 | PASS | 10 | 1 | 1 | 9 | 3.33 | 0 | common_connected_enclosure_witness | 5.47 | 22.73 | 22.73 | 507.35 | [布局](../../../output/B/pose1+3+5+7+9+12+16+21+23+27/step4.2/final_result.png) · [48候选](../../../output/B/pose1+3+5+7+9+12+16+21+23+27/step4.2/population.png) · [Evaluation](../../../output/B/pose1+3+5+7+9+12+16+21+23+27/step4.2/evaluation.png) |
| pose1+2+3+4+5+6+7+8+9 | PASS | 9 | 1 | 1 | 8 | 3.00 | 0 | common_connected_enclosure_witness | 5.27 | 15.11 | 15.11 | 416.80 | [布局](../../../output/B/pose1+2+3+4+5+6+7+8+9/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+4+5+6+7+8+9/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+4+5+6+7+8+9/step4.2/evaluation.png) |
| pose1+4+7+9+12+16+21+23+27 | PASS | 9 | 1 | 1 | 8 | 3.00 | 0 | common_connected_enclosure_witness | 4.79 | 15.57 | 15.57 | 350.72 | [布局](../../../output/B/pose1+4+7+9+12+16+21+23+27/step4.2/final_result.png) · [48候选](../../../output/B/pose1+4+7+9+12+16+21+23+27/step4.2/population.png) · [Evaluation](../../../output/B/pose1+4+7+9+12+16+21+23+27/step4.2/evaluation.png) |
| pose1+2+3+4+5+6+7+8 | PASS | 8 | 4 | 4 | 4 | 2.67 | 0 | common_connected_enclosure_witness | 5.27 | 9.05 | 10.35 | 316.20 | [布局](../../../output/B/pose1+2+3+4+5+6+7+8/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+4+5+6+7+8/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+4+5+6+7+8/step4.2/evaluation.png) |
| pose1+2+4+5+6+7+10+11 | PASS | 8 | 1 | 1 | 7 | 2.67 | 0 | common_connected_enclosure_witness | 5.27 | 7.27 | 7.27 | 232.47 | [布局](../../../output/B/pose1+2+4+5+6+7+10+11/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+4+5+6+7+10+11/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+4+5+6+7+10+11/step4.2/evaluation.png) |
| pose1+3+7+10+14+18+23+27 | PASS | 8 | 1 | 1 | 7 | 2.67 | 0 | common_connected_enclosure_witness | 6.04 | 13.82 | 13.82 | 317.87 | [布局](../../../output/B/pose1+3+7+10+14+18+23+27/step4.2/final_result.png) · [32候选](../../../output/B/pose1+3+7+10+14+18+23+27/step4.2/population.png) · [Evaluation](../../../output/B/pose1+3+7+10+14+18+23+27/step4.2/evaluation.png) |
| pose1+2+3+4+5+6+7 | PASS | 7 | 5 | 5 | 2 | 2.33 | 0 | common_connected_enclosure_witness | 5.27 | 7.15 | 9.56 | 216.27 | [布局](../../../output/B/pose1+2+3+4+5+6+7/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+4+5+6+7/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+4+5+6+7/step4.2/evaluation.png) |
| pose1+2+4+5+6+7+11 | PASS | 7 | 1 | 1 | 6 | 2.33 | 0 | common_connected_enclosure_witness | 5.27 | 6.76 | 6.76 | 227.71 | [布局](../../../output/B/pose1+2+4+5+6+7+11/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+4+5+6+7+11/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+4+5+6+7+11/step4.2/evaluation.png) |
| pose1+4+7+12+21+23+27 | PASS | 7 | 1 | 1 | 6 | 2.33 | 0 | common_connected_enclosure_witness | 4.79 | 6.63 | 6.63 | 212.18 | [布局](../../../output/B/pose1+4+7+12+21+23+27/step4.2/final_result.png) · [32候选](../../../output/B/pose1+4+7+12+21+23+27/step4.2/population.png) · [Evaluation](../../../output/B/pose1+4+7+12+21+23+27/step4.2/evaluation.png) |
| illegal/pose4+7+12+21+23+27 | PASS | 6 | 2 | 2 | 4 | 2.00 | 0 | common_connected_enclosure_witness | 4.79 | 6.94 | 6.94 | 189.03 | [布局](../../../output/B/illegal/pose4+7+12+21+23+27/step4.2/final_result.png) · [32候选](../../../output/B/illegal/pose4+7+12+21+23+27/step4.2/population.png) · [Evaluation](../../../output/B/illegal/pose4+7+12+21+23+27/step4.2/evaluation.png) |
| illegal/pose5+12+20+24+27+28 | PASS | 6 | 1 | 1 | 5 | 2.00 | 0 | common_connected_enclosure_witness | 5.18 | 6.74 | 6.74 | 189.19 | [布局](../../../output/B/illegal/pose5+12+20+24+27+28/step4.2/final_result.png) · [32候选](../../../output/B/illegal/pose5+12+20+24+27+28/step4.2/population.png) · [Evaluation](../../../output/B/illegal/pose5+12+20+24+27+28/step4.2/evaluation.png) |
| pose1+2+3+4+5+6 | PASS | 6 | 2 | 2 | 4 | 2.00 | 0 | common_connected_enclosure_witness | 5.27 | 7.31 | 7.34 | 208.93 | [布局](../../../output/B/pose1+2+3+4+5+6/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+4+5+6/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+4+5+6/step4.2/evaluation.png) |
| pose1+2+3+4+6+7 | PASS | 6 | 5 | 5 | 1 | 2.00 | 0 | common_connected_enclosure_witness | 4.67 | 6.18 | 8.72 | 154.93 | [布局](../../../output/B/pose1+2+3+4+6+7/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+4+6+7/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+4+6+7/step4.2/evaluation.png) |
| pose1+2+3+4+7+27 | PASS | 6 | 2 | 2 | 4 | 2.00 | 0 | common_connected_enclosure_witness | 4.88 | 5.93 | 6.93 | 161.97 | [布局](../../../output/B/pose1+2+3+4+7+27/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+4+7+27/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+4+7+27/step4.2/evaluation.png) |
| pose1+2+4+5+6+11 | PASS | 6 | 1 | 1 | 5 | 2.00 | 0 | common_connected_enclosure_witness | 5.27 | 6.79 | 6.79 | 218.12 | [布局](../../../output/B/pose1+2+4+5+6+11/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+4+5+6+11/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+4+5+6+11/step4.2/evaluation.png) |
| pose1+2+4+5+6+7 | PASS | 6 | 5 | 5 | 1 | 2.00 | 0 | common_connected_enclosure_witness | 5.27 | 6.30 | 9.31 | 773.51 | [布局](../../../output/B/pose1+2+4+5+6+7/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+4+5+6+7/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+4+5+6+7/step4.2/evaluation.png) |
| pose1+4+7+12+21+27 | PASS | 6 | 1 | 1 | 5 | 2.00 | 0 | common_connected_enclosure_witness | 4.57 | 6.09 | 6.09 | 218.57 | [布局](../../../output/B/pose1+4+7+12+21+27/step4.2/final_result.png) · [32候选](../../../output/B/pose1+4+7+12+21+27/step4.2/population.png) · [Evaluation](../../../output/B/pose1+4+7+12+21+27/step4.2/evaluation.png) |
| pose1+6+11+13+14+17 | PASS | 6 | 2 | 2 | 4 | 2.00 | 0 | common_connected_enclosure_witness | 4.96 | 5.75 | 6.51 | 175.21 | [布局](../../../output/B/pose1+6+11+13+14+17/step4.2/final_result.png) · [32候选](../../../output/B/pose1+6+11+13+14+17/step4.2/population.png) · [Evaluation](../../../output/B/pose1+6+11+13+14+17/step4.2/evaluation.png) |
| pose4+5+8+9+19+23 | PASS | 6 | 1 | 1 | 5 | 2.00 | 0 | common_connected_enclosure_witness | 5.98 | 7.57 | 7.57 | 210.60 | [布局](../../../output/B/pose4+5+8+9+19+23/step4.2/final_result.png) · [32候选](../../../output/B/pose4+5+8+9+19+23/step4.2/population.png) · [Evaluation](../../../output/B/pose4+5+8+9+19+23/step4.2/evaluation.png) |
| illegal/pose1+2+6+7+29 | PASS | 5 | 4 | 4 | 1 | 1.67 | 0 | common_connected_enclosure_witness | 4.67 | 5.47 | 6.25 | 110.35 | [布局](../../../output/B/illegal/pose1+2+6+7+29/step4.2/final_result.png) · [32候选](../../../output/B/illegal/pose1+2+6+7+29/step4.2/population.png) · [Evaluation](../../../output/B/illegal/pose1+2+6+7+29/step4.2/evaluation.png) |
| illegal/pose5+6+13+15+27 | PASS | 5 | 1 | 1 | 4 | 1.67 | 0 | common_connected_enclosure_witness | 5.20 | 6.68 | 6.68 | 240.04 | [布局](../../../output/B/illegal/pose5+6+13+15+27/step4.2/final_result.png) · [32候选](../../../output/B/illegal/pose5+6+13+15+27/step4.2/population.png) · [Evaluation](../../../output/B/illegal/pose5+6+13+15+27/step4.2/evaluation.png) |
| pose1+2+3+4+27 | PASS | 5 | 4 | 4 | 1 | 1.67 | 0 | common_connected_enclosure_witness | 4.88 | 5.80 | 7.42 | 125.17 | [布局](../../../output/B/pose1+2+3+4+27/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+4+27/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+4+27/step4.2/evaluation.png) |
| pose1+2+3+4+6 | PASS | 5 | 2 | 2 | 3 | 1.67 | 0 | common_connected_enclosure_witness | 4.62 | 6.65 | 6.76 | 194.66 | [布局](../../../output/B/pose1+2+3+4+6/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+4+6/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+4+6/step4.2/evaluation.png) |
| pose1+2+3+7+27 | PASS | 5 | 5 | 5 | 0 | 1.67 | 0 | common_connected_enclosure_witness | 4.88 | 5.27 | 8.89 | 123.84 | [布局](../../../output/B/pose1+2+3+7+27/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+7+27/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+7+27/step4.2/evaluation.png) |
| pose1+2+4+5+6 | PASS | 5 | 5 | 5 | 0 | 1.67 | 0 | common_connected_enclosure_witness | 5.27 | 5.97 | 8.36 | 275.94 | [布局](../../../output/B/pose1+2+4+5+6/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+4+5+6/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+4+5+6/step4.2/evaluation.png) |
| pose1+4+7+9+24 | PASS | 5 | 1 | 1 | 4 | 1.67 | 0 | common_connected_enclosure_witness | 4.17 | 4.81 | 4.81 | 190.97 | [布局](../../../output/B/pose1+4+7+9+24/step4.2/final_result.png) · [32候选](../../../output/B/pose1+4+7+9+24/step4.2/population.png) · [Evaluation](../../../output/B/pose1+4+7+9+24/step4.2/evaluation.png) |
| pose12+16+19+21+27 | PASS | 5 | 4 | 4 | 1 | 1.67 | 0 | common_connected_enclosure_witness | 5.19 | 6.31 | 16.07 | 332.61 | [布局](../../../output/B/pose12+16+19+21+27/step4.2/final_result.png) · [32候选](../../../output/B/pose12+16+19+21+27/step4.2/population.png) · [Evaluation](../../../output/B/pose12+16+19+21+27/step4.2/evaluation.png) |
| pose5+6+11+23+29 | PASS | 5 | 4 | 4 | 1 | 1.67 | 0 | common_connected_enclosure_witness | 5.50 | 7.44 | 10.22 | 398.30 | [布局](../../../output/B/pose5+6+11+23+29/step4.2/final_result.png) · [32候选](../../../output/B/pose5+6+11+23+29/step4.2/population.png) · [Evaluation](../../../output/B/pose5+6+11+23+29/step4.2/evaluation.png) |
| pose6+10+13+17+30 | PASS | 5 | 1 | 1 | 4 | 1.67 | 0 | common_connected_enclosure_witness | 4.48 | 5.93 | 5.93 | 222.49 | [布局](../../../output/B/pose6+10+13+17+30/step4.2/final_result.png) · [32候选](../../../output/B/pose6+10+13+17+30/step4.2/population.png) · [Evaluation](../../../output/B/pose6+10+13+17+30/step4.2/evaluation.png) |
| illegal/pose1+2+6+15 | PASS | 4 | 1 | 1 | 3 | 1.33 | 0 | common_connected_enclosure_witness | 4.64 | 5.21 | 5.44 | 204.54 | [布局](../../../output/B/illegal/pose1+2+6+15/step4.2/final_result.png) · [32候选](../../../output/B/illegal/pose1+2+6+15/step4.2/population.png) · [Evaluation](../../../output/B/illegal/pose1+2+6+15/step4.2/evaluation.png) |
| illegal/pose7+11+13+19 | PASS | 4 | 2 | 2 | 2 | 1.33 | 0 | common_connected_enclosure_witness | 5.01 | 5.72 | 6.02 | 156.92 | [布局](../../../output/B/illegal/pose7+11+13+19/step4.2/final_result.png) · [32候选](../../../output/B/illegal/pose7+11+13+19/step4.2/population.png) · [Evaluation](../../../output/B/illegal/pose7+11+13+19/step4.2/evaluation.png) |
| pose1+11+14+27 | PASS | 4 | 2 | 2 | 2 | 1.33 | 0 | common_connected_enclosure_witness | 4.33 | 5.35 | 5.48 | 151.77 | [布局](../../../output/B/pose1+11+14+27/step4.2/final_result.png) · [32候选](../../../output/B/pose1+11+14+27/step4.2/population.png) · [Evaluation](../../../output/B/pose1+11+14+27/step4.2/evaluation.png) |
| pose1+2+3+27 | PASS | 4 | 2 | 2 | 2 | 1.33 | 0 | common_connected_enclosure_witness | 4.88 | 5.96 | 6.03 | 187.73 | [布局](../../../output/B/pose1+2+3+27/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+3+27/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+3+27/step4.2/evaluation.png) |
| pose1+2+4+19 | PASS | 4 | 4 | 4 | 0 | 1.33 | 0 | common_connected_enclosure_witness | 5.17 | 7.11 | 21.34 | 1018.29 | [布局](../../../output/B/pose1+2+4+19/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+4+19/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+4+19/step4.2/evaluation.png) |
| pose1+2+4+6 | PASS | 4 | 4 | 4 | 0 | 1.33 | 0 | common_connected_enclosure_witness | 4.62 | 5.29 | 7.76 | 145.68 | [布局](../../../output/B/pose1+2+4+6/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+4+6/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+4+6/step4.2/evaluation.png) |
| pose1+2+4+7 | PASS | 4 | 4 | 4 | 0 | 1.33 | 0 | common_connected_enclosure_witness | 4.24 | 4.95 | 8.00 | 216.95 | [布局](../../../output/B/pose1+2+4+7/step4.2/final_result.png) · [32候选](../../../output/B/pose1+2+4+7/step4.2/population.png) · [Evaluation](../../../output/B/pose1+2+4+7/step4.2/evaluation.png) |
| pose2+3+4+7 | PASS | 4 | 4 | 4 | 0 | 1.33 | 0 | common_connected_enclosure_witness | 4.64 | 5.72 | 8.22 | 148.99 | [布局](../../../output/B/pose2+3+4+7/step4.2/final_result.png) · [32候选](../../../output/B/pose2+3+4+7/step4.2/population.png) · [Evaluation](../../../output/B/pose2+3+4+7/step4.2/evaluation.png) |
| pose5+6+23+29 | PASS | 4 | 4 | 4 | 0 | 1.33 | 0 | common_connected_enclosure_witness | 5.50 | 6.68 | 7.96 | 102.48 | [布局](../../../output/B/pose5+6+23+29/step4.2/final_result.png) · [32候选](../../../output/B/pose5+6+23+29/step4.2/population.png) · [Evaluation](../../../output/B/pose5+6+23+29/step4.2/evaluation.png) |
| pose8+9+13+30 | PASS | 4 | 3 | 3 | 1 | 1.33 | 0 | common_connected_enclosure_witness | 4.20 | 4.73 | 5.94 | 124.70 | [布局](../../../output/B/pose8+9+13+30/step4.2/final_result.png) · [32候选](../../../output/B/pose8+9+13+30/step4.2/population.png) · [Evaluation](../../../output/B/pose8+9+13+30/step4.2/evaluation.png) |
| illegal/pose1+2+29 | PASS | 3 | 2 | 2 | 1 | 1.00 | 0 | common_connected_enclosure_witness | 4.58 | 5.31 | 6.11 | 128.99 | [布局](../../../output/B/illegal/pose1+2+29/step4.2/final_result.png) · [32候选](../../../output/B/illegal/pose1+2+29/step4.2/population.png) · [Evaluation](../../../output/B/illegal/pose1+2+29/step4.2/evaluation.png) |
| illegal/pose5+13+27 | PASS | 3 | 1 | 1 | 2 | 1.00 | 0 | common_connected_enclosure_witness | 4.56 | 5.36 | 5.36 | 231.04 | [布局](../../../output/B/illegal/pose5+13+27/step4.2/final_result.png) · [32候选](../../../output/B/illegal/pose5+13+27/step4.2/population.png) · [Evaluation](../../../output/B/illegal/pose5+13+27/step4.2/evaluation.png) |
| pose1+12+29 | PASS | 3 | 3 | 3 | 0 | 1.00 | 0 | common_connected_enclosure_witness | 4.68 | 5.69 | 6.69 | 121.73 | [布局](../../../output/B/pose1+12+29/step4.2/final_result.png) · [32候选](../../../output/B/pose1+12+29/step4.2/population.png) · [Evaluation](../../../output/B/pose1+12+29/step4.2/evaluation.png) |
| pose18+23+24 | PASS | 3 | 2 | 2 | 1 | 1.00 | 0 | common_connected_enclosure_witness | 5.33 | 5.79 | 7.03 | 142.12 | [布局](../../../output/B/pose18+23+24/step4.2/final_result.png) · [32候选](../../../output/B/pose18+23+24/step4.2/population.png) · [Evaluation](../../../output/B/pose18+23+24/step4.2/evaluation.png) |
| pose4+5+7 | PASS | 3 | 3 | 3 | 0 | 1.00 | 0 | common_connected_enclosure_witness | 4.71 | 5.22 | 6.14 | 143.21 | [布局](../../../output/B/pose4+5+7/step4.2/final_result.png) · [32候选](../../../output/B/pose4+5+7/step4.2/population.png) · [Evaluation](../../../output/B/pose4+5+7/step4.2/evaluation.png) |
| pose8+10+19 | PASS | 3 | 3 | 3 | 0 | 1.00 | 0 | common_connected_enclosure_witness | 5.66 | 6.15 | 11.50 | 106.47 | [布局](../../../output/B/pose8+10+19/step4.2/final_result.png) · [32候选](../../../output/B/pose8+10+19/step4.2/population.png) · [Evaluation](../../../output/B/pose8+10+19/step4.2/evaluation.png) |
| illegal/pose11+19 | PASS | 2 | 1 | 1 | 1 | 0.67 | 0 | common_connected_enclosure_witness | 5.01 | 5.81 | 5.98 | 182.22 | [布局](../../../output/B/illegal/pose11+19/step4.2/final_result.png) · [32候选](../../../output/B/illegal/pose11+19/step4.2/population.png) · [Evaluation](../../../output/B/illegal/pose11+19/step4.2/evaluation.png) |
| illegal/pose2+29 | PASS | 2 | 1 | 1 | 1 | 0.67 | 0 | common_connected_enclosure_witness | 4.01 | 4.41 | 4.41 | 138.35 | [布局](../../../output/B/illegal/pose2+29/step4.2/final_result.png) · [32候选](../../../output/B/illegal/pose2+29/step4.2/population.png) · [Evaluation](../../../output/B/illegal/pose2+29/step4.2/evaluation.png) |
| pose19+28 | PASS | 2 | 2 | 2 | 0 | 0.67 | 0 | common_connected_enclosure_witness | 4.28 | 6.84 | 8.19 | 165.89 | [布局](../../../output/B/pose19+28/step4.2/final_result.png) · [32候选](../../../output/B/pose19+28/step4.2/population.png) · [Evaluation](../../../output/B/pose19+28/step4.2/evaluation.png) |
| pose2+20 | PASS | 2 | 2 | 2 | 0 | 0.67 | 0 | common_connected_enclosure_witness | 4.11 | 4.42 | 4.42 | 145.09 | [布局](../../../output/B/pose2+20/step4.2/final_result.png) · [32候选](../../../output/B/pose2+20/step4.2/population.png) · [Evaluation](../../../output/B/pose2+20/step4.2/evaluation.png) |
| pose3+15 | PASS | 2 | 1 | 1 | 1 | 0.67 | 0 | common_connected_enclosure_witness | 4.41 | 4.52 | 4.90 | 171.90 | [布局](../../../output/B/pose3+15/step4.2/final_result.png) · [32候选](../../../output/B/pose3+15/step4.2/population.png) · [Evaluation](../../../output/B/pose3+15/step4.2/evaluation.png) |
| pose8+21 | PASS | 2 | 2 | 2 | 0 | 0.67 | 0 | common_connected_enclosure_witness | 3.49 | 3.86 | 4.33 | 126.49 | [布局](../../../output/B/pose8+21/step4.2/final_result.png) · [32候选](../../../output/B/pose8+21/step4.2/population.png) · [Evaluation](../../../output/B/pose8+21/step4.2/evaluation.png) |

[结构化结果与图片来源](data/system_connected_core_all52_20261013_final/step4_results.json)。
[组合执行批次](data/connected_core_all52_20261013_final/batch.json)。
[冻结源码与摘要](data/system_connected_core_all52_20261013_10/frozen_sources.json)。
[保存产物核对](data/system_connected_core_all52_20261013_final/saved_artifact_audit.json)。

每组 `evaluation.png` 比较无支撑／Step4名义模型／完整支撑的静态BBX，并列出每个支撑安装服务的object pose。失败组不提供完整成品体积。

通过组的逐组BBX之和：无支撑254.408L／Step4名义模型371.058L／完整支撑440.343L；完整支撑材料之和12196.256cm³。这是不同集合的指标合计，不是一件共同支撑的尺寸。

共同通过50组，完整BBX之和：旧418.872L → 新403.851L；材料之和：旧10334.420cm³ → 新11422.861cm³。该比较只含两次均通过的相同集合。

新增通过：`pose1+2+3+4+5+6+7+8+9+10`、`pose1+4+7+9+12+16+21+23+27`。

原通过转失败：无。
