# 统一 XYZ：七组重跑

本轮新搜索力／力矩通过 **7/7**，初次预算通过6组，额外自身状态梯度接续通过1组。

连续 Translation 使用同一世界 XYZ 梯度、范数和步幅池；地面约束保留，抬高即去掉物体地面反力。每个 pose 沿用全部 32768 个原始需求。选定结果后复用已算出的通过掩码，不运行几何微扰补救或末尾重复求解。

本轮搜索中位数 **403.6s/组**；固定布局 mesh 导出中位数4.5s。3个并行进程，所有阶段含出图总计 **21.38min**。计入失败与接续预算，保留旧答案的历史耗时未算入。

## 本轮新搜索

通过判据是搜索接触模型上的全部原始力／力矩需求。体积取固定布局导出的名义实体 mesh；未通过或未导出 mesh 的估计体积不计入实体总体积。对照为本轮开始前保存的逐组最小答案，合计713.007866cm³。

| pose set | 力／力矩 | 原体积 cm³ | 新实体体积 cm³ | 变化 | 离地 pose／最高 mm | 搜索 s | 图 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| pose1+3+5+7+9+12+16+21+23+27 | 通过 | 58.47 | 54.55 | -6.70% | 1／0.050 | 115.9 | [过程](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_xyz_force_v3/process.png) · [最终](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_xyz_force_v3/final_result.png) |
| pose1+2+3+4+5+6+7+8 | 通过 | 100.45 | 122.69 | +22.13% | 0／0.000 | 403.6 | [过程](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_xyz_force_v3/process.png) · [最终](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_xyz_force_v3/final_result.png) |
| pose1+2+4+5+6+7+10+11 | 通过 | 146.65 | 156.07 | +6.42% | 1／4.800 | 307.4 | [过程](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_xyz_force_v3/process.png) · [最终](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_xyz_force_v3/final_result.png) |
| pose1+3+7+10+14+18+23+27 | 通过 | 66.85 | 40.69 | -39.13% | 0／0.000 | 71.4 | [过程](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_xyz_force_v3/process.png) · [最终](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_xyz_force_v3/final_result.png) |
| pose1+2+3+4+5+6+7+8+9 | 通过 | 90.19 | 113.22 | +25.53% | 0／0.000 | 515.4 | [过程](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_xyz_force_v3_repair/process.png) · [最终](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_xyz_force_v3_repair/final_result.png) |
| pose1+4+7+9+12+16+21+23+27 | 通过 | 111.30 | 122.43 | +10.00% | 1／12.361 | 523.9 | [过程](pose1+4+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_xyz_force_v3/process.png) · [最终](pose1+4+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_xyz_force_v3/final_result.png) |
| pose1+2+3+4+5+6+7+8+9+10 | 通过 | 139.09 | 111.90 | -19.55% | 0／0.000 | 515.0 | [过程](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_xyz_force_v3/process.png) · [最终](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_xyz_force_v3/final_result.png) |

已导出新实体的7组：合计721.55cm³，同组旧答案713.01cm³，变化+1.20%。新结果中3个 pose 离地。

## 最终材料择优

最终保留 **7/7** 个通过方案；本轮改善3组，沿用旧答案4组。总体积655.74cm³，旧答案713.01cm³，变化-8.03%。这些保留答案未用作新搜索的初始化。

| pose set | 最终体积 cm³ | 来源 | 图 |
| --- | ---: | --- | --- |
| pose1+3+5+7+9+12+16+21+23+27 | 54.55 | 本轮新搜索 | [过程](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_xyz_force_v3/process.png) · [最终](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_xyz_force_v3/final_result.png) |
| pose1+2+3+4+5+6+7+8 | 100.45 | 沿用旧答案 | [过程](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_xyz_v1/process.png) · [最终](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_xyz_v1/final_result.png) |
| pose1+2+4+5+6+7+10+11 | 146.65 | 沿用旧答案 | [过程](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_remaining_v17/final_result.png) |
| pose1+3+7+10+14+18+23+27 | 40.69 | 本轮新搜索 | [过程](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_xyz_force_v3/process.png) · [最终](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_xyz_force_v3/final_result.png) |
| pose1+2+3+4+5+6+7+8+9 | 90.19 | 沿用旧答案 | [过程](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_remaining_v17/final_result.png) |
| pose1+4+7+9+12+16+21+23+27 | 111.30 | 沿用旧答案 | [过程](pose1+4+7+9+12+16+21+23+27/step4/step4.2/fast_state_seat_gradient_repair_v21/process.png) · [最终](pose1+4+7+9+12+16+21+23+27/step4/step4.2/fast_state_seat_gradient_repair_v21/final_result.png) |
| pose1+2+3+4+5+6+7+8+9+10 | 111.90 | 本轮新搜索 | [过程](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_xyz_force_v3/process.png) · [最终](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_xyz_force_v3/final_result.png) |

[图片浏览](stable_gradient_xyz_force_v3_index.html) · [完整记录](data/stable_gradient_xyz_force_v3/summary.json)
