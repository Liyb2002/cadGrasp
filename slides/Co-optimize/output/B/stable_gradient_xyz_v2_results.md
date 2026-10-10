# XYZ Translation：8–10 pose 结果

实际通过 **7/7**；初次搜索通过5组，额外梯度修复通过1组，几何conditioning通过1组。

同组实体材料体积合计713.01cm³，旧结果714.49cm³，变化-0.21%。本轮各阶段搜索中位数271.0s，实际候选检查中位数240.3s；批次运行合计42.8min。计入失败、超时和接续预算，保留答案的历史耗时不算作新运行。

逐组真实材料体积不退步检查7组；本轮新搜索通过7组，新改善1组，保留已验证答案6组。新搜索与保留结果明确分开，旧答案未用作冷启动。

在已Juxtapose位置允许世界XYZ移动，始终保持工件不穿地；抬高后移除物体地面反力，保留原重力、全部32768需求及第七条约束。原Step3／4.1与历史答案保留。

体积是实际支撑实体体积；对照是此前stable-gradient最终所选答案，包含各自明确记录的接续，并非相同冷启动预算对照。

| pose set | 实际结果 | 原体积 cm³ | 最终体积 cm³ | 变化 | 来源 | 悬空pose数／最高 mm | 图 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| pose1+3+5+7+9+12+16+21+23+27 | 通过 | 58.47 | 58.47 | +0.0% | 保留已验证结果 | 0／0.000 | [过程](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_cold_v17/process.png) · [最终](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_cold_v17/final_result.png) |
| pose1+2+3+4+5+6+7+8 | 通过 | 100.56 | 100.45 | -0.1% | 保留已验证结果 | 0／0.000 | [过程](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_xyz_v1/process.png) · [最终](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_xyz_v1/final_result.png) |
| pose1+2+4+5+6+7+10+11 | 通过 | 146.65 | 146.65 | +0.0% | 保留已验证结果 | 0／0.000 | [过程](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_remaining_v17/final_result.png) |
| pose1+3+7+10+14+18+23+27 | 通过 | 68.23 | 66.85 | -2.0% | 本轮新搜索 | 0／0.000 | [过程](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_xyz_v2/process.png) · [最终](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_xyz_v2/final_result.png) |
| pose1+2+3+4+5+6+7+8+9 | 通过 | 90.19 | 90.19 | +0.0% | 保留已验证结果 | 0／0.000 | [过程](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_remaining_v17/final_result.png) |
| pose1+4+7+9+12+16+21+23+27 | 通过 | 111.30 | 111.30 | +0.0% | 保留已验证结果 | 0／0.000 | [过程](pose1+4+7+9+12+16+21+23+27/step4/step4.2/fast_state_seat_gradient_repair_v21/process.png) · [最终](pose1+4+7+9+12+16+21+23+27/step4/step4.2/fast_state_seat_gradient_repair_v21/final_result.png) |
| pose1+2+3+4+5+6+7+8+9+10 | 通过 | 139.09 | 139.09 | +0.0% | 保留已验证结果 | 0／0.000 | [过程](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_remaining_v17/process.png) · [最终](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_remaining_v17/final_result.png) |

最终通过方案中，共有0个pose实际离地。允许+z并不强制抬高；搜索只接受计入所有pose收益损失后的候选。

Step5仍需按最终位置设计与检验系统—地面的基座；本实验未增加安装、连通或强度验收。

[图片浏览](stable_gradient_xyz_v2_index.html) · [完整记录](data/stable_gradient_xyz_v2/summary.json)
