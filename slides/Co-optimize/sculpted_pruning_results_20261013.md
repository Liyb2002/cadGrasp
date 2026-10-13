# 更圆润的支撑：全部52组结果（2026-10-13）

最终选择`sculpted_pruning_all52_20261013_01`。全部52组／263个object pose的Step5.2和Step5.3已完成并通过，Evaluation、实际OBJ审计及全需求／源文件／图片审计均通过。123个支撑安装pose，复用123／Juxtapose140，逐pose角色与上一版一致。Step4和Step5.1未重跑。

[算法与入口](sculpted_pruning_algorithm.md)。本版先整块删除冗余材料，再圆整真实非接触外表面和脚的自由部分。固定必要接触、整体BBX极值、地面及已认证主体；成品重新导出、重新检查全部原始需求。

## BBX与材料

BBX为每组整体静态占用并集盒的合计，单位L；材料为实际导出实体的体积，单位cm³。两种指标分别报告。

| BBX阶段 | 上一版（L） | 本次（L） |
| --- | ---: | ---: |
| 无支撑 | 254.408145232 | 254.408145232 |
| Step4主体 | 371.057877703 | 371.057877703 |
| Step5.2删减后 | 351.890791316 | 351.890791316 |
| Step5.3成品 | 432.466716637 | 432.466716631 |

全部52组的新旧BBX边界、尺寸和体积逐组核对。最大体积差2.512e-09 L，为浮点误差；每组保持原整体BBX。各安装state的小盒另作诊断。

| 材料阶段 | 上一版（cm³） | 本次（cm³） | 变化 |
| --- | ---: | ---: | ---: |
| Step5.2 | 3650.240 | 3169.428 | -13.17% |
| Step5.3 | 10145.041 | 9616.340 | -5.21% |

每组两阶段的材料均未增加；39组主体材料减少、44组成品材料减少。圆整候选与已有材料相交，不向原Step4外补肉。脚沿用已有合法材料及位置，本版调整自由侧面，没有重新搜索安装布局。

主体连续表面圆整通过34/52组，脚自由表面圆整通过29/52组；33组还通过了进一步整块删减。接受的局部圆角共主体71条、脚58条。其余操作在原验收下拒绝或回退，不冒充统一圆角成功。

物体接触面、工作／退出禁区、地面和固定BBX仍限制形状，部分功能棱面必须保留。宽根部和连续圆整改善形状；当前验证为原静力模型与几何验证，不是抗弯或破坏强度证书。

## 全部52组及图片

O＝object pose数，S＝实际支撑安装pose数，R＝原位复用，J＝Juxtapose。BBX单位L；净增量／百分比比较成品与Step4。材料列比较上一版与本次成品，单位cm³。每组两阶段均PASS。

| Set | O | S | R | J | 无支撑 | Step4 | Step5.2 | Step5.3 | 净增量 | 增幅 | 成品材料旧→新 | 材料变化 | 图 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pose2+20 | 2 | 2 | 2 | 0 | 4.110 | 4.423 | 4.110 | 4.349 | -0.074 | -1.66% | 63.224→58.422 | -7.60% | [5.2](output/B/pose2+20/step5.2/comparison.png) · [成品](output/B/pose2+20/step5.3/comparison.png) · [放大](output/B/pose2+20/step5.3/detail.png) |
| illegal/pose2+29 | 2 | 1 | 1 | 1 | 4.009 | 4.412 | 4.009 | 4.009 | -0.403 | -9.14% | 53.907→45.092 | -16.35% | [5.2](output/B/illegal/pose2+29/step5.2/comparison.png) · [成品](output/B/illegal/pose2+29/step5.3/comparison.png) · [放大](output/B/illegal/pose2+29/step5.3/detail.png) |
| pose3+15 | 2 | 1 | 1 | 1 | 4.414 | 4.524 | 4.414 | 4.916 | +0.392 | +8.66% | 67.770→58.956 | -13.01% | [5.2](output/B/pose3+15/step5.2/comparison.png) · [成品](output/B/pose3+15/step5.3/comparison.png) · [放大](output/B/pose3+15/step5.3/detail.png) |
| pose8+21 | 2 | 2 | 2 | 0 | 3.489 | 3.856 | 3.602 | 4.138 | +0.283 | +7.33% | 75.158→70.068 | -6.77% | [5.2](output/B/pose8+21/step5.2/comparison.png) · [成品](output/B/pose8+21/step5.3/comparison.png) · [放大](output/B/pose8+21/step5.3/detail.png) |
| illegal/pose11+19 | 2 | 1 | 1 | 1 | 5.005 | 5.812 | 5.188 | 5.867 | +0.055 | +0.95% | 90.518→77.603 | -14.27% | [5.2](output/B/illegal/pose11+19/step5.2/comparison.png) · [成品](output/B/illegal/pose11+19/step5.3/comparison.png) · [放大](output/B/illegal/pose11+19/step5.3/detail.png) |
| pose19+28 | 2 | 2 | 2 | 0 | 4.278 | 6.839 | 6.173 | 7.956 | +1.117 | +16.33% | 112.805→104.796 | -7.10% | [5.2](output/B/pose19+28/step5.2/comparison.png) · [成品](output/B/pose19+28/step5.3/comparison.png) · [放大](output/B/pose19+28/step5.3/detail.png) |
| illegal/pose1+2+29 | 3 | 2 | 2 | 1 | 4.584 | 5.309 | 5.027 | 6.307 | +0.998 | +18.81% | 71.123→66.338 | -6.73% | [5.2](output/B/illegal/pose1+2+29/step5.2/comparison.png) · [成品](output/B/illegal/pose1+2+29/step5.3/comparison.png) · [放大](output/B/illegal/pose1+2+29/step5.3/detail.png) |
| pose1+12+29 | 3 | 3 | 3 | 0 | 4.676 | 5.690 | 5.389 | 6.679 | +0.988 | +17.37% | 89.957→86.298 | -4.07% | [5.2](output/B/pose1+12+29/step5.2/comparison.png) · [成品](output/B/pose1+12+29/step5.3/comparison.png) · [放大](output/B/pose1+12+29/step5.3/detail.png) |
| pose4+5+7 | 3 | 3 | 3 | 0 | 4.709 | 5.216 | 4.827 | 6.246 | +1.030 | +19.74% | 134.055→126.438 | -5.68% | [5.2](output/B/pose4+5+7/step5.2/comparison.png) · [成品](output/B/pose4+5+7/step5.3/comparison.png) · [放大](output/B/pose4+5+7/step5.3/detail.png) |
| illegal/pose5+13+27 | 3 | 1 | 1 | 2 | 4.563 | 5.363 | 4.978 | 4.978 | -0.386 | -7.19% | 69.086→68.072 | -1.47% | [5.2](output/B/illegal/pose5+13+27/step5.2/comparison.png) · [成品](output/B/illegal/pose5+13+27/step5.3/comparison.png) · [放大](output/B/illegal/pose5+13+27/step5.3/detail.png) |
| pose8+10+19 | 3 | 3 | 3 | 0 | 5.658 | 6.148 | 5.818 | 11.643 | +5.495 | +89.39% | 146.134→137.739 | -5.74% | [5.2](output/B/pose8+10+19/step5.2/comparison.png) · [成品](output/B/pose8+10+19/step5.3/comparison.png) · [放大](output/B/pose8+10+19/step5.3/detail.png) |
| pose18+23+24 | 3 | 2 | 2 | 1 | 5.328 | 5.791 | 5.715 | 7.136 | +1.345 | +23.22% | 122.132→107.432 | -12.04% | [5.2](output/B/pose18+23+24/step5.2/comparison.png) · [成品](output/B/pose18+23+24/step5.3/comparison.png) · [放大](output/B/pose18+23+24/step5.3/detail.png) |
| pose1+2+3+27 | 4 | 2 | 2 | 2 | 4.876 | 5.959 | 5.255 | 5.814 | -0.145 | -2.43% | 101.683→94.567 | -7.00% | [5.2](output/B/pose1+2+3+27/step5.2/comparison.png) · [成品](output/B/pose1+2+3+27/step5.3/comparison.png) · [放大](output/B/pose1+2+3+27/step5.3/detail.png) |
| pose1+2+4+6 | 4 | 4 | 4 | 0 | 4.621 | 5.287 | 5.098 | 7.758 | +2.471 | +46.75% | 145.802→139.250 | -4.49% | [5.2](output/B/pose1+2+4+6/step5.2/comparison.png) · [成品](output/B/pose1+2+4+6/step5.3/comparison.png) · [放大](output/B/pose1+2+4+6/step5.3/detail.png) |
| pose1+2+4+7 | 4 | 4 | 4 | 0 | 4.239 | 4.948 | 4.948 | 7.532 | +2.584 | +52.24% | 311.920→311.920 | +0.00% | [5.2](output/B/pose1+2+4+7/step5.2/comparison.png) · [成品](output/B/pose1+2+4+7/step5.3/comparison.png) · [放大](output/B/pose1+2+4+7/step5.3/detail.png) |
| pose1+2+4+19 | 4 | 4 | 4 | 0 | 5.165 | 7.107 | 6.534 | 21.847 | +14.741 | +207.42% | 2164.621→2164.621 | +0.00% | [5.2](output/B/pose1+2+4+19/step5.2/comparison.png) · [成品](output/B/pose1+2+4+19/step5.3/comparison.png) · [放大](output/B/pose1+2+4+19/step5.3/detail.png) |
| illegal/pose1+2+6+15 | 4 | 1 | 1 | 3 | 4.638 | 5.211 | 4.987 | 5.586 | +0.375 | +7.20% | 98.273→92.160 | -6.22% | [5.2](output/B/illegal/pose1+2+6+15/step5.2/comparison.png) · [成品](output/B/illegal/pose1+2+6+15/step5.3/comparison.png) · [放大](output/B/illegal/pose1+2+6+15/step5.3/detail.png) |
| pose1+11+14+27 | 4 | 2 | 2 | 2 | 4.326 | 5.355 | 4.987 | 5.535 | +0.181 | +3.37% | 93.599→85.161 | -9.02% | [5.2](output/B/pose1+11+14+27/step5.2/comparison.png) · [成品](output/B/pose1+11+14+27/step5.3/comparison.png) · [放大](output/B/pose1+11+14+27/step5.3/detail.png) |
| pose2+3+4+7 | 4 | 4 | 4 | 0 | 4.637 | 5.716 | 5.404 | 7.997 | +2.281 | +39.91% | 175.639→175.639 | +0.00% | [5.2](output/B/pose2+3+4+7/step5.2/comparison.png) · [成品](output/B/pose2+3+4+7/step5.3/comparison.png) · [放大](output/B/pose2+3+4+7/step5.3/detail.png) |
| pose5+6+23+29 | 4 | 4 | 4 | 0 | 5.504 | 6.680 | 6.607 | 8.081 | +1.401 | +20.97% | 130.070→120.825 | -7.11% | [5.2](output/B/pose5+6+23+29/step5.2/comparison.png) · [成品](output/B/pose5+6+23+29/step5.3/comparison.png) · [放大](output/B/pose5+6+23+29/step5.3/detail.png) |
| illegal/pose7+11+13+19 | 4 | 2 | 2 | 2 | 5.005 | 5.720 | 5.557 | 6.141 | +0.421 | +7.36% | 112.085→98.004 | -12.56% | [5.2](output/B/illegal/pose7+11+13+19/step5.2/comparison.png) · [成品](output/B/illegal/pose7+11+13+19/step5.3/comparison.png) · [放大](output/B/illegal/pose7+11+13+19/step5.3/detail.png) |
| pose8+9+13+30 | 4 | 3 | 3 | 1 | 4.203 | 4.730 | 4.328 | 5.957 | +1.227 | +25.94% | 151.452→151.452 | +0.00% | [5.2](output/B/pose8+9+13+30/step5.2/comparison.png) · [成品](output/B/pose8+9+13+30/step5.3/comparison.png) · [放大](output/B/pose8+9+13+30/step5.3/detail.png) |
| pose1+2+3+4+6 | 5 | 2 | 2 | 3 | 4.621 | 6.652 | 6.191 | 6.408 | -0.244 | -3.66% | 120.065→115.299 | -3.97% | [5.2](output/B/pose1+2+3+4+6/step5.2/comparison.png) · [成品](output/B/pose1+2+3+4+6/step5.3/comparison.png) · [放大](output/B/pose1+2+3+4+6/step5.3/detail.png) |
| pose1+2+3+4+27 | 5 | 4 | 4 | 1 | 4.876 | 5.805 | 5.416 | 7.679 | +1.874 | +32.29% | 157.146→148.090 | -5.76% | [5.2](output/B/pose1+2+3+4+27/step5.2/comparison.png) · [成品](output/B/pose1+2+3+4+27/step5.3/comparison.png) · [放大](output/B/pose1+2+3+4+27/step5.3/detail.png) |
| pose1+2+3+7+27 | 5 | 5 | 5 | 0 | 4.876 | 5.270 | 5.032 | 9.044 | +3.774 | +71.62% | 110.740→101.368 | -8.46% | [5.2](output/B/pose1+2+3+7+27/step5.2/comparison.png) · [成品](output/B/pose1+2+3+7+27/step5.3/comparison.png) · [放大](output/B/pose1+2+3+7+27/step5.3/detail.png) |
| pose1+2+4+5+6 | 5 | 5 | 5 | 0 | 5.268 | 5.966 | 5.626 | 8.155 | +2.189 | +36.69% | 156.252→155.434 | -0.52% | [5.2](output/B/pose1+2+4+5+6/step5.2/comparison.png) · [成品](output/B/pose1+2+4+5+6/step5.3/comparison.png) · [放大](output/B/pose1+2+4+5+6/step5.3/detail.png) |
| illegal/pose1+2+6+7+29 | 5 | 4 | 4 | 1 | 4.667 | 5.471 | 5.276 | 6.310 | +0.839 | +15.34% | 298.853→298.853 | +0.00% | [5.2](output/B/illegal/pose1+2+6+7+29/step5.2/comparison.png) · [成品](output/B/illegal/pose1+2+6+7+29/step5.3/comparison.png) · [放大](output/B/illegal/pose1+2+6+7+29/step5.3/detail.png) |
| pose1+4+7+9+24 | 5 | 1 | 1 | 4 | 4.171 | 4.809 | 4.461 | 4.461 | -0.348 | -7.23% | 100.626→95.749 | -4.85% | [5.2](output/B/pose1+4+7+9+24/step5.2/comparison.png) · [成品](output/B/pose1+4+7+9+24/step5.3/comparison.png) · [放大](output/B/pose1+4+7+9+24/step5.3/detail.png) |
| pose5+6+11+23+29 | 5 | 4 | 4 | 1 | 5.504 | 7.441 | 7.277 | 9.001 | +1.560 | +20.97% | 103.526→103.526 | +0.00% | [5.2](output/B/pose5+6+11+23+29/step5.2/comparison.png) · [成品](output/B/pose5+6+11+23+29/step5.3/comparison.png) · [放大](output/B/pose5+6+11+23+29/step5.3/detail.png) |
| illegal/pose5+6+13+15+27 | 5 | 1 | 1 | 4 | 5.196 | 6.684 | 6.047 | 6.371 | -0.313 | -4.69% | 104.358→87.153 | -16.49% | [5.2](output/B/illegal/pose5+6+13+15+27/step5.2/comparison.png) · [成品](output/B/illegal/pose5+6+13+15+27/step5.3/comparison.png) · [放大](output/B/illegal/pose5+6+13+15+27/step5.3/detail.png) |
| pose6+10+13+17+30 | 5 | 1 | 1 | 4 | 4.479 | 5.933 | 5.529 | 5.606 | -0.328 | -5.52% | 101.636→83.596 | -17.75% | [5.2](output/B/pose6+10+13+17+30/step5.2/comparison.png) · [成品](output/B/pose6+10+13+17+30/step5.3/comparison.png) · [放大](output/B/pose6+10+13+17+30/step5.3/detail.png) |
| pose12+16+19+21+27 | 5 | 4 | 4 | 1 | 5.192 | 6.310 | 6.157 | 15.026 | +8.716 | +138.14% | 183.299→183.151 | -0.08% | [5.2](output/B/pose12+16+19+21+27/step5.2/comparison.png) · [成品](output/B/pose12+16+19+21+27/step5.3/comparison.png) · [放大](output/B/pose12+16+19+21+27/step5.3/detail.png) |
| pose1+2+3+4+5+6 | 6 | 2 | 2 | 4 | 5.268 | 7.306 | 6.819 | 7.266 | -0.039 | -0.54% | 106.707→89.216 | -16.39% | [5.2](output/B/pose1+2+3+4+5+6/step5.2/comparison.png) · [成品](output/B/pose1+2+3+4+5+6/step5.3/comparison.png) · [放大](output/B/pose1+2+3+4+5+6/step5.3/detail.png) |
| pose1+2+3+4+6+7 | 6 | 5 | 5 | 1 | 4.667 | 6.181 | 5.952 | 8.872 | +2.691 | +43.54% | 151.727→151.598 | -0.08% | [5.2](output/B/pose1+2+3+4+6+7/step5.2/comparison.png) · [成品](output/B/pose1+2+3+4+6+7/step5.3/comparison.png) · [放大](output/B/pose1+2+3+4+6+7/step5.3/detail.png) |
| pose1+2+3+4+7+27 | 6 | 2 | 2 | 4 | 4.876 | 5.933 | 5.758 | 7.041 | +1.108 | +18.68% | 201.019→198.654 | -1.18% | [5.2](output/B/pose1+2+3+4+7+27/step5.2/comparison.png) · [成品](output/B/pose1+2+3+4+7+27/step5.3/comparison.png) · [放大](output/B/pose1+2+3+4+7+27/step5.3/detail.png) |
| pose1+2+4+5+6+7 | 6 | 5 | 5 | 1 | 5.268 | 6.304 | 5.917 | 9.467 | +3.163 | +50.17% | 487.022→487.022 | +0.00% | [5.2](output/B/pose1+2+4+5+6+7/step5.2/comparison.png) · [成品](output/B/pose1+2+4+5+6+7/step5.3/comparison.png) · [放大](output/B/pose1+2+4+5+6+7/step5.3/detail.png) |
| pose1+2+4+5+6+11 | 6 | 1 | 1 | 5 | 5.268 | 6.786 | 6.274 | 6.274 | -0.512 | -7.54% | 89.528→77.503 | -13.43% | [5.2](output/B/pose1+2+4+5+6+11/step5.2/comparison.png) · [成品](output/B/pose1+2+4+5+6+11/step5.3/comparison.png) · [放大](output/B/pose1+2+4+5+6+11/step5.3/detail.png) |
| pose1+4+7+12+21+27 | 6 | 1 | 1 | 5 | 4.571 | 6.089 | 5.572 | 5.572 | -0.516 | -8.48% | 120.701→104.758 | -13.21% | [5.2](output/B/pose1+4+7+12+21+27/step5.2/comparison.png) · [成品](output/B/pose1+4+7+12+21+27/step5.3/comparison.png) · [放大](output/B/pose1+4+7+12+21+27/step5.3/detail.png) |
| pose1+6+11+13+14+17 | 6 | 2 | 2 | 4 | 4.956 | 5.747 | 5.588 | 6.200 | +0.454 | +7.89% | 111.895→111.851 | -0.04% | [5.2](output/B/pose1+6+11+13+14+17/step5.2/comparison.png) · [成品](output/B/pose1+6+11+13+14+17/step5.3/comparison.png) · [放大](output/B/pose1+6+11+13+14+17/step5.3/detail.png) |
| pose4+5+8+9+19+23 | 6 | 1 | 1 | 5 | 5.981 | 7.571 | 7.203 | 7.203 | -0.368 | -4.86% | 126.061→108.978 | -13.55% | [5.2](output/B/pose4+5+8+9+19+23/step5.2/comparison.png) · [成品](output/B/pose4+5+8+9+19+23/step5.3/comparison.png) · [放大](output/B/pose4+5+8+9+19+23/step5.3/detail.png) |
| illegal/pose4+7+12+21+23+27 | 6 | 2 | 2 | 4 | 4.791 | 6.942 | 5.851 | 6.536 | -0.405 | -5.84% | 148.061→123.782 | -16.40% | [5.2](output/B/illegal/pose4+7+12+21+23+27/step5.2/comparison.png) · [成品](output/B/illegal/pose4+7+12+21+23+27/step5.3/comparison.png) · [放大](output/B/illegal/pose4+7+12+21+23+27/step5.3/detail.png) |
| illegal/pose5+12+20+24+27+28 | 6 | 1 | 1 | 5 | 5.176 | 6.742 | 6.462 | 6.462 | -0.279 | -4.14% | 94.451→81.953 | -13.23% | [5.2](output/B/illegal/pose5+12+20+24+27+28/step5.2/comparison.png) · [成品](output/B/illegal/pose5+12+20+24+27+28/step5.3/comparison.png) · [放大](output/B/illegal/pose5+12+20+24+27+28/step5.3/detail.png) |
| pose1+2+3+4+5+6+7 | 7 | 5 | 5 | 2 | 5.268 | 7.152 | 6.513 | 9.473 | +2.321 | +32.45% | 194.525→194.286 | -0.12% | [5.2](output/B/pose1+2+3+4+5+6+7/step5.2/comparison.png) · [成品](output/B/pose1+2+3+4+5+6+7/step5.3/comparison.png) · [放大](output/B/pose1+2+3+4+5+6+7/step5.3/detail.png) |
| pose1+2+4+5+6+7+11 | 7 | 1 | 1 | 6 | 5.268 | 6.761 | 6.274 | 6.452 | -0.309 | -4.58% | 112.222→92.439 | -17.63% | [5.2](output/B/pose1+2+4+5+6+7+11/step5.2/comparison.png) · [成品](output/B/pose1+2+4+5+6+7+11/step5.3/comparison.png) · [放大](output/B/pose1+2+4+5+6+7+11/step5.3/detail.png) |
| pose1+4+7+12+21+23+27 | 7 | 1 | 1 | 6 | 4.791 | 6.632 | 6.211 | 6.211 | -0.421 | -6.34% | 112.546→91.626 | -18.59% | [5.2](output/B/pose1+4+7+12+21+23+27/step5.2/comparison.png) · [成品](output/B/pose1+4+7+12+21+23+27/step5.3/comparison.png) · [放大](output/B/pose1+4+7+12+21+23+27/step5.3/detail.png) |
| pose1+2+3+4+5+6+7+8 | 8 | 4 | 4 | 4 | 5.268 | 9.053 | 8.011 | 10.679 | +1.626 | +17.96% | 632.754→632.754 | +0.00% | [5.2](output/B/pose1+2+3+4+5+6+7+8/step5.2/comparison.png) · [成品](output/B/pose1+2+3+4+5+6+7+8/step5.3/comparison.png) · [放大](output/B/pose1+2+3+4+5+6+7+8/step5.3/detail.png) |
| pose1+2+4+5+6+7+10+11 | 8 | 1 | 1 | 7 | 5.268 | 7.275 | 7.004 | 7.249 | -0.026 | -0.36% | 122.713→98.707 | -19.56% | [5.2](output/B/pose1+2+4+5+6+7+10+11/step5.2/comparison.png) · [成品](output/B/pose1+2+4+5+6+7+10+11/step5.3/comparison.png) · [放大](output/B/pose1+2+4+5+6+7+10+11/step5.3/detail.png) |
| pose1+3+7+10+14+18+23+27 | 8 | 1 | 1 | 7 | 6.044 | 13.823 | 13.609 | 13.609 | -0.214 | -1.55% | 202.934→170.463 | -16.00% | [5.2](output/B/pose1+3+7+10+14+18+23+27/step5.2/comparison.png) · [成品](output/B/pose1+3+7+10+14+18+23+27/step5.3/comparison.png) · [放大](output/B/pose1+3+7+10+14+18+23+27/step5.3/detail.png) |
| pose1+2+3+4+5+6+7+8+9 | 9 | 1 | 1 | 8 | 5.268 | 15.109 | 14.807 | 15.201 | +0.092 | +0.61% | 270.342→241.482 | -10.68% | [5.2](output/B/pose1+2+3+4+5+6+7+8+9/step5.2/comparison.png) · [成品](output/B/pose1+2+3+4+5+6+7+8+9/step5.3/comparison.png) · [放大](output/B/pose1+2+3+4+5+6+7+8+9/step5.3/detail.png) |
| pose1+4+7+9+12+16+21+23+27 | 9 | 1 | 1 | 8 | 4.791 | 15.574 | 15.360 | 15.431 | -0.143 | -0.92% | 207.713→173.808 | -16.32% | [5.2](output/B/pose1+4+7+9+12+16+21+23+27/step5.2/comparison.png) · [成品](output/B/pose1+4+7+9+12+16+21+23+27/step5.3/comparison.png) · [放大](output/B/pose1+4+7+9+12+16+21+23+27/step5.3/detail.png) |
| pose1+2+3+4+5+6+7+8+9+10 | 10 | 1 | 1 | 9 | 5.268 | 20.918 | 20.628 | 20.628 | -0.290 | -1.38% | 313.162→284.948 | -9.01% | [5.2](output/B/pose1+2+3+4+5+6+7+8+9+10/step5.2/comparison.png) · [成品](output/B/pose1+2+3+4+5+6+7+8+9+10/step5.3/comparison.png) · [放大](output/B/pose1+2+3+4+5+6+7+8+9+10/step5.3/detail.png) |
| pose1+3+5+7+9+12+16+21+23+27 | 10 | 1 | 1 | 9 | 5.469 | 22.726 | 22.106 | 22.106 | -0.620 | -2.73% | 321.446→287.440 | -10.58% | [5.2](output/B/pose1+3+5+7+9+12+16+21+23+27/step5.2/comparison.png) · [成品](output/B/pose1+3+5+7+9+12+16+21+23+27/step5.3/comparison.png) · [放大](output/B/pose1+3+5+7+9+12+16+21+23+27/step5.3/detail.png) |

每组`step5.3/overview.png`为成品，`comparison.png`左旧右新，`detail.png`为同视角脚部放大；`all_poses.png`、全部`pose_N.png`及Evaluation图已更新。Step5.2同时更新删减主体、删除材料图和前后对比。所有图直接在各自set的阶段目录，无HTML。

![成品前后对比](output/B/pose1+4+7+9+24/step5.3/comparison.png)

![脚部前后放大](output/B/pose1+4+7+9+24/step5.3/detail.png)

## 验收与执行记录

- 每pose全部32,768个原始需求不变；两阶段均检查实际导出接触，分别核对8,617,984个通过mask。原反力模型及容差保持。
- 独立实际OBJ审计核对连通、原材料包含、合法空间、物体内部排除及成品主体保留；原几何体积限值1e-10 m³。实际地面包围和整体BBX另行严格检查。
- 11212个旧数值／输入文件、6915张旧PNG副本及1570张当前PNG验证；原Step4图片保持原样。每组选用自己的执行版本与源码副本。
- 初始批次有拒绝、资源问题和几何库崩溃，全部保留记录；已有通过结果保留，剩余组在新数值身份下执行分批查询、导出后守卫或连续表面保守分支。不能将部分／中断批次描述成一次52/52。最终选择只包含真实通过的52组。
- 5个新几何回归测试和5个原删减测试通过；分批最近点查询与一次查询在700个点上的结果一致。

[最终52组来源](data/results/B/data/sculpted_pruning_all52_20261013_01/resolved_batch.json) · [Evaluation](data/results/B/evaluation_sculpted_pruning_all52_20261013_01_bbx.md) · [逐pose角色](data/results/B/evaluation_sculpted_pruning_all52_20261013_01_poses.csv) · [新旧BBX／材料CSV](data/results/B/comparison_sculpted_pruning_all52_20261013_01.csv) · [实际实体审计](data/results/B/data/system_sculpted_pruning_all52_20261013_01/actual_solid_geometry_audit.json) · [完整审计](data/results/B/data/system_sculpted_pruning_all52_20261013_01/sculpted_audit.json)。
