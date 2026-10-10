# B：Step4 whole优化

沿用新Step3初始化和原B集合。所有pose始终参与约束；候选增删更新，保存真实三角支撑。先恢复原载荷承载，再保持全组可行减少实体材料。没有远距离分离保底。

[图片浏览](step4_index.html) · [新初始化](README.md)

| 集合 | 状态 | Step4.1承载 | 初始材料cm³ | 最终材料cm³ | 转动/Juxtapose | 图片 |
|---|---|---|---:|---:|---|---|
| illegal/pose1+2+29 | pass | True | 91.233 | 86.055 | 3/0 | [初始化](illegal/pose1+2+29/step4/step4.1/final_result.png) · [过程](illegal/pose1+2+29/step4/step4.2/process.png) · [最终](illegal/pose1+2+29/step4/step4.2/final_result.png) |
| illegal/pose1+2+6+15 | pass | True | 92.236 | 85.497 | 4/0 | [初始化](illegal/pose1+2+6+15/step4/step4.1/final_result.png) · [过程](illegal/pose1+2+6+15/step4/step4.2/process.png) · [最终](illegal/pose1+2+6+15/step4/step4.2/final_result.png) |
| illegal/pose1+2+6+7+29 | pass | True | 64.435 | 56.982 | 5/0 | [初始化](illegal/pose1+2+6+7+29/step4/step4.1/final_result.png) · [过程](illegal/pose1+2+6+7+29/step4/step4.2/process.png) · [最终](illegal/pose1+2+6+7+29/step4/step4.2/final_result.png) |
| illegal/pose11+19 | pass | True | 117.535 | 104.818 | 2/0 | [初始化](illegal/pose11+19/step4/step4.1/final_result.png) · [过程](illegal/pose11+19/step4/step4.2/process.png) · [最终](illegal/pose11+19/step4/step4.2/final_result.png) |
| illegal/pose2+29 | pass | True | 98.513 | 90.085 | 2/0 | [初始化](illegal/pose2+29/step4/step4.1/final_result.png) · [过程](illegal/pose2+29/step4/step4.2/process.png) · [最终](illegal/pose2+29/step4/step4.2/final_result.png) |
| illegal/pose4+7+12+21+23+27 | pass | False | 34.737 | 43.763 | 5/1 | [初始化](illegal/pose4+7+12+21+23+27/step4/step4.1/final_result.png) · [过程](illegal/pose4+7+12+21+23+27/step4/step4.2/process.png) · [最终](illegal/pose4+7+12+21+23+27/step4/step4.2/final_result.png) |
| illegal/pose5+12+20+24+27+28 | pass | True | 114.500 | 110.768 | 6/0 | [初始化](illegal/pose5+12+20+24+27+28/step4/step4.1/final_result.png) · [过程](illegal/pose5+12+20+24+27+28/step4/step4.2/process.png) · [最终](illegal/pose5+12+20+24+27+28/step4/step4.2/final_result.png) |
| illegal/pose5+13+27 | pass | True | 144.116 | 124.197 | 3/0 | [初始化](illegal/pose5+13+27/step4/step4.1/final_result.png) · [过程](illegal/pose5+13+27/step4/step4.2/process.png) · [最终](illegal/pose5+13+27/step4/step4.2/final_result.png) |
| illegal/pose5+6+13+15+27 | pass | True | 128.263 | 111.766 | 5/0 | [初始化](illegal/pose5+6+13+15+27/step4/step4.1/final_result.png) · [过程](illegal/pose5+6+13+15+27/step4/step4.2/process.png) · [最终](illegal/pose5+6+13+15+27/step4/step4.2/final_result.png) |
| illegal/pose7+11+13+19 | pass | True | 101.960 | 89.390 | 4/0 | [初始化](illegal/pose7+11+13+19/step4/step4.1/final_result.png) · [过程](illegal/pose7+11+13+19/step4/step4.2/process.png) · [最终](illegal/pose7+11+13+19/step4/step4.2/final_result.png) |
| pose1+11+14+27 | pass | True | 33.922 | 27.668 | 4/0 | [初始化](pose1+11+14+27/step4/step4.1/final_result.png) · [过程](pose1+11+14+27/step4/step4.2/process.png) · [最终](pose1+11+14+27/step4/step4.2/final_result.png) |
| pose1+12+29 | pass | True | 96.377 | 91.264 | 3/0 | [初始化](pose1+12+29/step4/step4.1/final_result.png) · [过程](pose1+12+29/step4/step4.2/process.png) · [最终](pose1+12+29/step4/step4.2/final_result.png) |
| pose1+2+3+27 | pass | True | 69.409 | 64.088 | 4/0 | [初始化](pose1+2+3+27/step4/step4.1/final_result.png) · [过程](pose1+2+3+27/step4/step4.2/process.png) · [最终](pose1+2+3+27/step4/step4.2/final_result.png) |
| pose1+2+3+4+27 | pass | True | 59.441 | 59.441 | 5/0 | [初始化](pose1+2+3+4+27/step4/step4.1/final_result.png) · [过程](pose1+2+3+4+27/step4/step4.2/process.png) · [最终](pose1+2+3+4+27/step4/step4.2/final_result.png) |
| pose1+2+3+4+5+6 | pass | False | 41.137 | 38.282 | 5/1 | [初始化](pose1+2+3+4+5+6/step4/step4.1/final_result.png) · [过程](pose1+2+3+4+5+6/step4/step4.2/process.png) · [最终](pose1+2+3+4+5+6/step4/step4.2/final_result.png) |
| pose1+2+3+4+6 | pass | False | 45.761 | 40.652 | 4/1 | [初始化](pose1+2+3+4+6/step4/step4.1/final_result.png) · [过程](pose1+2+3+4+6/step4/step4.2/process.png) · [最终](pose1+2+3+4+6/step4/step4.2/final_result.png) |
| pose1+2+3+4+6+7 | pass | False | 40.100 | 72.487 | 4/2 | [初始化](pose1+2+3+4+6+7/step4/step4.1/final_result.png) · [过程](pose1+2+3+4+6+7/step4/step4.2/process.png) · [最终](pose1+2+3+4+6+7/step4/step4.2/final_result.png) |
| pose1+2+3+4+7+27 | pass | False | 28.398 | 85.193 | 4/2 | [初始化](pose1+2+3+4+7+27/step4/step4.1/final_result.png) · [过程](pose1+2+3+4+7+27/step4/step4.2/process.png) · [最终](pose1+2+3+4+7+27/step4/step4.2/final_result.png) |
| pose1+2+3+7+27 | pass | True | 48.016 | 43.883 | 5/0 | [初始化](pose1+2+3+7+27/step4/step4.1/final_result.png) · [过程](pose1+2+3+7+27/step4/step4.2/process.png) · [最终](pose1+2+3+7+27/step4/step4.2/final_result.png) |
| pose1+2+4+19 | pass | True | 59.148 | 52.261 | 4/0 | [初始化](pose1+2+4+19/step4/step4.1/final_result.png) · [过程](pose1+2+4+19/step4/step4.2/process.png) · [最终](pose1+2+4+19/step4/step4.2/final_result.png) |
| pose1+2+4+5+6 | pass | True | 84.201 | 84.201 | 5/0 | [初始化](pose1+2+4+5+6/step4/step4.1/final_result.png) · [过程](pose1+2+4+5+6/step4/step4.2/process.png) · [最终](pose1+2+4+5+6/step4/step4.2/final_result.png) |
| pose1+2+4+5+6+11 | pass | False | 35.879 | 99.636 | 4/2 | [初始化](pose1+2+4+5+6+11/step4/step4.1/final_result.png) · [过程](pose1+2+4+5+6+11/step4/step4.2/process.png) · [最终](pose1+2+4+5+6+11/step4/step4.2/final_result.png) |
| pose1+2+4+5+6+7 | pass | False | 31.497 | 81.660 | 4/2 | [初始化](pose1+2+4+5+6+7/step4/step4.1/final_result.png) · [过程](pose1+2+4+5+6+7/step4/step4.2/process.png) · [最终](pose1+2+4+5+6+7/step4/step4.2/final_result.png) |
| pose1+2+4+6 | pass | True | 86.496 | 80.375 | 4/0 | [初始化](pose1+2+4+6/step4/step4.1/final_result.png) · [过程](pose1+2+4+6/step4/step4.2/process.png) · [最终](pose1+2+4+6/step4/step4.2/final_result.png) |
| pose1+2+4+7 | pass | True | 50.181 | 42.350 | 4/0 | [初始化](pose1+2+4+7/step4/step4.1/final_result.png) · [过程](pose1+2+4+7/step4/step4.2/process.png) · [最终](pose1+2+4+7/step4/step4.2/final_result.png) |
| pose1+4+7+12+21+27 | pass | False | 26.580 | 63.251 | 3/3 | [初始化](pose1+4+7+12+21+27/step4/step4.1/final_result.png) · [过程](pose1+4+7+12+21+27/step4/step4.2/process.png) · [最终](pose1+4+7+12+21+27/step4/step4.2/final_result.png) |
| pose1+4+7+9+24 | pass | False | 18.609 | 56.216 | 3/2 | [初始化](pose1+4+7+9+24/step4/step4.1/final_result.png) · [过程](pose1+4+7+9+24/step4/step4.2/process.png) · [最终](pose1+4+7+9+24/step4/step4.2/final_result.png) |
| pose1+6+11+13+14+17 | pass | False | 28.118 | 69.611 | 3/3 | [初始化](pose1+6+11+13+14+17/step4/step4.1/final_result.png) · [过程](pose1+6+11+13+14+17/step4/step4.2/process.png) · [最终](pose1+6+11+13+14+17/step4/step4.2/final_result.png) |
| pose12+16+19+21+27 | pass | True | 49.887 | 49.887 | 5/0 | [初始化](pose12+16+19+21+27/step4/step4.1/final_result.png) · [过程](pose12+16+19+21+27/step4/step4.2/process.png) · [最终](pose12+16+19+21+27/step4/step4.2/final_result.png) |
| pose18+23+24 | pass | True | 64.174 | 64.174 | 3/0 | [初始化](pose18+23+24/step4/step4.1/final_result.png) · [过程](pose18+23+24/step4/step4.2/process.png) · [最终](pose18+23+24/step4/step4.2/final_result.png) |
| pose19+28 | pass | True | 139.831 | 139.831 | 2/0 | [初始化](pose19+28/step4/step4.1/final_result.png) · [过程](pose19+28/step4/step4.2/process.png) · [最终](pose19+28/step4/step4.2/final_result.png) |
| pose2+20 | pass | True | 132.026 | 132.026 | 2/0 | [初始化](pose2+20/step4/step4.1/final_result.png) · [过程](pose2+20/step4/step4.2/process.png) · [最终](pose2+20/step4/step4.2/final_result.png) |
| pose2+3+4+7 | pass | True | 91.244 | 85.528 | 4/0 | [初始化](pose2+3+4+7/step4/step4.1/final_result.png) · [过程](pose2+3+4+7/step4/step4.2/process.png) · [最终](pose2+3+4+7/step4/step4.2/final_result.png) |
| pose3+15 | pass | True | 130.603 | 118.241 | 2/0 | [初始化](pose3+15/step4/step4.1/final_result.png) · [过程](pose3+15/step4/step4.2/process.png) · [最终](pose3+15/step4/step4.2/final_result.png) |
| pose4+5+7 | pass | True | 90.338 | 70.287 | 3/0 | [初始化](pose4+5+7/step4/step4.1/final_result.png) · [过程](pose4+5+7/step4/step4.2/process.png) · [最终](pose4+5+7/step4/step4.2/final_result.png) |
| pose4+5+8+9+19+23 | pass | False | 37.548 | 92.802 | 4/2 | [初始化](pose4+5+8+9+19+23/step4/step4.1/final_result.png) · [过程](pose4+5+8+9+19+23/step4/step4.2/process.png) · [最终](pose4+5+8+9+19+23/step4/step4.2/final_result.png) |
| pose5+6+11+23+29 | pass | False | 21.025 | 19.410 | 4/1 | [初始化](pose5+6+11+23+29/step4/step4.1/final_result.png) · [过程](pose5+6+11+23+29/step4/step4.2/process.png) · [最终](pose5+6+11+23+29/step4/step4.2/final_result.png) |
| pose5+6+23+29 | pass | True | 64.784 | 64.784 | 4/0 | [初始化](pose5+6+23+29/step4/step4.1/final_result.png) · [过程](pose5+6+23+29/step4/step4.2/process.png) · [最终](pose5+6+23+29/step4/step4.2/final_result.png) |
| pose6+10+13+17+30 | pass | False | 44.380 | 76.536 | 3/2 | [初始化](pose6+10+13+17+30/step4/step4.1/final_result.png) · [过程](pose6+10+13+17+30/step4/step4.2/process.png) · [最终](pose6+10+13+17+30/step4/step4.2/final_result.png) |
| pose8+10+19 | pass | True | 74.173 | 70.007 | 3/0 | [初始化](pose8+10+19/step4/step4.1/final_result.png) · [过程](pose8+10+19/step4/step4.2/process.png) · [最终](pose8+10+19/step4/step4.2/final_result.png) |
| pose8+21 | pass | True | 95.685 | 87.183 | 2/0 | [初始化](pose8+21/step4/step4.1/final_result.png) · [过程](pose8+21/step4/step4.2/process.png) · [最终](pose8+21/step4/step4.2/final_result.png) |
| pose8+9+13+30 | pass | True | 58.212 | 58.212 | 4/0 | [初始化](pose8+9+13+30/step4/step4.1/final_result.png) · [过程](pose8+9+13+30/step4/step4.2/process.png) · [最终](pose8+9+13+30/step4/step4.2/final_result.png) |
