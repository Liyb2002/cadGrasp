# 固定32候选的新Step4结果

批次 `population32_all52_20261012_01`：已结束 19/52 组，完整通过 3 组，失败 16 组。批次状态：运行中。

32个固定槽位、8轮、注册原位初始化，无历史优化布局暖启动。每槽位每轮随机选择一次内部离散换座或跨槽位差分，再做一轮短调整。谁换座由优化过程选择；换座起点为世界XY质心对齐，保持原任务朝向。

[算法说明](../../../population32_algorithm.md)。当前每组图在 `output/B/[set]/step4.2/`：`final_result.png` 为有导出实体的所选布局；未过受力、没有实体导出的集合提供物体姿态诊断 `layout.png`。`process.png` 为已有的几何搜索过程，`population.png` 为32个槽位的换座数及每轮分支分配。

PASS要求原始需求、Oracle共同连接和搜索内实际成品构造通过。失败行的布局和名义BBX仅用于诊断，完整成品列为空。组合目标单独报告，原受力／几何验收不放宽。本轮只运行Step4。

| Set | 状态 | object pose | 支撑pose | 原位复用 | Juxtapose | 未通过原需求数 | Oracle | 名义BBX L | 成品BBX L | 图 |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| pose1+2+3+4+5+6+7+8+9+10 | FORCE_FAIL | 10 | 9 | 9 | 1 | 57065 | certified_floor_incompatible | — | — | [PNG](../../../output/B/pose1+2+3+4+5+6+7+8+9+10/step4.2/layout.png) |
| pose1+3+5+7+9+12+16+21+23+27 | FORCE_FAIL | 10 | 8 | 8 | 2 | 59393 | certified_floor_incompatible | — | — | [PNG](../../../output/B/pose1+3+5+7+9+12+16+21+23+27/step4.2/layout.png) |
| pose1+2+3+4+5+6+7+8+9 | FORCE_FAIL | 9 | 8 | 8 | 1 | 30801 | certified_floor_incompatible | — | — | [PNG](../../../output/B/pose1+2+3+4+5+6+7+8+9/step4.2/layout.png) |
| pose1+4+7+9+12+16+21+23+27 | FORCE_FAIL | 9 | 7 | 7 | 2 | 294298 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+4+7+9+12+16+21+23+27/step4.2/layout.png) |
| pose1+2+3+4+5+6+7+8 | FORCE_FAIL | 8 | 4 | 4 | 4 | 115621 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+2+3+4+5+6+7+8/step4.2/layout.png) |
| pose1+2+4+5+6+7+10+11 | FORCE_FAIL | 8 | 4 | 4 | 4 | 251311 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+2+4+5+6+7+10+11/step4.2/layout.png) |
| pose1+3+7+10+14+18+23+27 | FORCE_FAIL | 8 | 5 | 4 | 4 | 186054 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+3+7+10+14+18+23+27/step4.2/layout.png) |
| pose1+2+3+4+5+6+7 | FORCE_FAIL | 7 | 3 | 3 | 4 | 35968 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+2+3+4+5+6+7/step4.2/layout.png) |
| pose1+2+4+5+6+7+11 | FORCE_FAIL | 7 | 4 | 4 | 3 | 72459 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+2+4+5+6+7+11/step4.2/layout.png) |
| pose1+4+7+12+21+23+27 | FORCE_FAIL | 7 | 4 | 3 | 4 | 119617 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+4+7+12+21+23+27/step4.2/layout.png) |
| illegal/pose4+7+12+21+23+27 | FORCE_FAIL | 6 | 4 | 4 | 2 | 61796 | enclosure_witness | — | — | [PNG](../../../output/B/illegal/pose4+7+12+21+23+27/step4.2/layout.png) |
| illegal/pose5+12+20+24+27+28 | PASS | 6 | 2 | 2 | 4 | 0 | common_connected_enclosure_witness | 7.58 | 11.14 | [PNG](../../../output/B/illegal/pose5+12+20+24+27+28/step4.2/final_result.png) |
| pose1+2+3+4+5+6 | PASS | 6 | 4 | 4 | 2 | 0 | common_connected_enclosure_witness | 7.52 | 9.75 | [PNG](../../../output/B/pose1+2+3+4+5+6/step4.2/final_result.png) |
| pose1+2+3+4+6+7 | FORCE_FAIL | 6 | 4 | 4 | 2 | 19355 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+2+3+4+6+7/step4.2/layout.png) |
| pose1+2+3+4+7+27 | FORCE_FAIL | 6 | 2 | 2 | 4 | 2985 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+2+3+4+7+27/step4.2/layout.png) |
| pose1+2+4+5+6+11 | FORCE_FAIL | 6 | 2 | 2 | 4 | 2 | no_common_component_in_window | — | — | [PNG](../../../output/B/pose1+2+4+5+6+11/step4.2/layout.png) |
| pose1+2+4+5+6+7 | FORCE_FAIL | 6 | 4 | 4 | 2 | 4598 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+2+4+5+6+7/step4.2/layout.png) |
| pose1+4+7+12+21+27 | FORCE_FAIL | 6 | 2 | 2 | 4 | 29635 | enclosure_witness | — | — | [PNG](../../../output/B/pose1+4+7+12+21+27/step4.2/layout.png) |
| pose1+6+11+13+14+17 | RUNNING | 6 | — | — | — | — | — | — | — | — |
| pose4+5+8+9+19+23 | RUNNING | 6 | — | — | — | — | — | — | — | — |
| illegal/pose1+2+6+7+29 | RUNNING | 5 | — | — | — | — | — | — | — | — |
| illegal/pose5+6+13+15+27 | PASS | 5 | 2 | 2 | 3 | 0 | common_connected_enclosure_witness | 9.76 | 11.15 | [PNG](../../../output/B/illegal/pose5+6+13+15+27/step4.2/final_result.png) |
| pose1+2+3+4+27 | RUNNING | 5 | — | — | — | — | — | — | — | — |
| pose1+2+3+4+6 | RUNNING | 5 | — | — | — | — | — | — | — | — |
| pose1+2+3+7+27 | RUNNING | 5 | — | — | — | — | — | — | — | — |
| pose1+2+4+5+6 | RUNNING | 5 | — | — | — | — | — | — | — | — |
| pose1+4+7+9+24 | RUNNING | 5 | — | — | — | — | — | — | — | — |
| pose12+16+19+21+27 | RUNNING | 5 | — | — | — | — | — | — | — | — |
| pose5+6+11+23+29 | RUNNING | 5 | — | — | — | — | — | — | — | — |
| pose6+10+13+17+30 | RUNNING | 5 | — | — | — | — | — | — | — | — |
| illegal/pose1+2+6+15 | RUNNING | 4 | — | — | — | — | — | — | — | — |
| illegal/pose7+11+13+19 | RUNNING | 4 | — | — | — | — | — | — | — | — |
| pose1+11+14+27 | RUNNING | 4 | — | — | — | — | — | — | — | — |
| pose1+2+3+27 | RUNNING | 4 | — | — | — | — | — | — | — | — |
| pose1+2+4+19 | RUNNING | 4 | — | — | — | — | — | — | — | — |
| pose1+2+4+6 | RUNNING | 4 | — | — | — | — | — | — | — | — |
| pose1+2+4+7 | RUNNING | 4 | — | — | — | — | — | — | — | — |
| pose2+3+4+7 | RUNNING | 4 | — | — | — | — | — | — | — | — |
| pose5+6+23+29 | RUNNING | 4 | — | — | — | — | — | — | — | — |
| pose8+9+13+30 | RUNNING | 4 | — | — | — | — | — | — | — | — |
| illegal/pose1+2+29 | RUNNING | 3 | — | — | — | — | — | — | — | — |
| illegal/pose5+13+27 | RUNNING | 3 | — | — | — | — | — | — | — | — |
| pose1+12+29 | RUNNING | 3 | — | — | — | — | — | — | — | — |
| pose18+23+24 | RUNNING | 3 | — | — | — | — | — | — | — | — |
| pose4+5+7 | RUNNING | 3 | — | — | — | — | — | — | — | — |
| pose8+10+19 | RUNNING | 3 | — | — | — | — | — | — | — | — |
| illegal/pose11+19 | RUNNING | 2 | — | — | — | — | — | — | — | — |
| illegal/pose2+29 | RUNNING | 2 | — | — | — | — | — | — | — | — |
| pose19+28 | RUNNING | 2 | — | — | — | — | — | — | — | — |
| pose2+20 | RUNNING | 2 | — | — | — | — | — | — | — | — |
| pose3+15 | RUNNING | 2 | — | — | — | — | — | — | — | — |
| pose8+21 | RUNNING | 2 | — | — | — | — | — | — | — | — |

[结构化结果与图片来源](data/system_population32_all52_20261012_01/step4_results.json)。
[冻结源码与摘要](data/system_population32_all52_20261012_01/frozen_sources.json)。
