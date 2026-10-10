# XYZ Translation：8–10 pose 结果

实际通过 **7/7**；初次搜索通过7组，额外梯度修复通过0组，几何conditioning通过0组。

同组实体材料体积合计721.94cm³，旧结果714.49cm³，变化+1.04%。搜索中位数259.5s，实际检查中位数224.6s；本轮3个进程，总运行28.4min。实际检查时间包含失败与超时，额外出图计入批次总时间。

在已Juxtapose位置允许世界XYZ移动，始终保持工件不穿地；抬高后移除物体地面反力，保留原重力、全部32768需求及第七条约束。原Step3／4.1与历史答案保留。

体积是实际支撑实体体积；对照是此前stable-gradient最终所选答案，包含各自明确记录的接续，并非相同冷启动预算对照。

| pose set | 实际结果 | 原体积 cm³ | 新体积 cm³ | 变化 | 悬空pose数／最高 mm | 图 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| pose1+3+5+7+9+12+16+21+23+27 | 通过 | 58.47 | 58.70 | +0.4% | 1／1.209 | [过程](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_xyz_v1/process.png) · [最终](pose1+3+5+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_xyz_v1/final_result.png) |
| pose1+2+3+4+5+6+7+8 | 通过 | 100.56 | 100.45 | -0.1% | 0／0.000 | [过程](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_xyz_v1/process.png) · [最终](pose1+2+3+4+5+6+7+8/step4/step4.2/stable_gradient_xyz_v1/final_result.png) |
| pose1+2+4+5+6+7+10+11 | 通过 | 146.65 | 147.11 | +0.3% | 0／0.000 | [过程](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_xyz_v1/process.png) · [最终](pose1+2+4+5+6+7+10+11/step4/step4.2/stable_gradient_xyz_v1/final_result.png) |
| pose1+3+7+10+14+18+23+27 | 通过 | 68.23 | 71.69 | +5.1% | 0／0.000 | [过程](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_xyz_v1/process.png) · [最终](pose1+3+7+10+14+18+23+27/step4/step4.2/stable_gradient_xyz_v1/final_result.png) |
| pose1+2+3+4+5+6+7+8+9 | 通过 | 90.19 | 92.97 | +3.1% | 0／0.000 | [过程](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_xyz_v1/process.png) · [最终](pose1+2+3+4+5+6+7+8+9/step4/step4.2/stable_gradient_xyz_v1/final_result.png) |
| pose1+4+7+9+12+16+21+23+27 | 通过 | 111.30 | 111.55 | +0.2% | 1／9.144 | [过程](pose1+4+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_xyz_v1/process.png) · [最终](pose1+4+7+9+12+16+21+23+27/step4/step4.2/stable_gradient_xyz_v1/final_result.png) |
| pose1+2+3+4+5+6+7+8+9+10 | 通过 | 139.09 | 139.47 | +0.3% | 0／0.000 | [过程](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_xyz_v1/process.png) · [最终](pose1+2+3+4+5+6+7+8+9+10/step4/step4.2/stable_gradient_xyz_v1/final_result.png) |

最终通过方案中，共有2个pose实际离地。允许+z并不强制抬高；搜索只接受计入所有pose收益损失后的候选。

Step5仍需按最终位置设计与检验系统—地面的基座；本实验未增加安装、连通或强度验收。

[图片浏览](stable_gradient_xyz_v1_index.html) · [完整记录](data/stable_gradient_xyz_v1/summary.json)
