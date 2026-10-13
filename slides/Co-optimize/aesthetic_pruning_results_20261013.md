# Step5.2整块圆整删减：全部52组结果（2026-10-13）

批次`aesthetic_pruning_all52_20261013_01`。全部52组、263个object pose的Step5.2和Step5.3均通过，Evaluation及两个独立审计完成。Step4和Step5.1没有重跑；123个支撑安装pose、复用123／Juxtapose140及逐pose成员保持原样。

新算法先裁去扩大BBX的部分，再保留较宽的原材料带、删去大块冗余区域，替换逐格啃材料。图中用同一相机、比例和需求凸包比较旧模型与新模型。[算法与实际实现](aesthetic_pruning_algorithm.md)。

## BBX结果

以下为全部52组各自整体静态占用BBX的合计，单位L；不是支撑材料体积，也不是各安装state小盒的体积之和。

| 阶段 | 上一版 | 本次 | 本次−上一版 |
| --- | ---: | ---: | ---: |
| 无支撑 | 254.408145232 | 254.408145232 | 0.000e+00 |
| Step4连通主体 | 371.057877703 | 371.057877703 | 0.000e+00 |
| Step5.2删减后 | 351.890791316 | 351.890791316 | 1.705e-13 |
| Step5.3长脚成品 | 432.466716637 | 432.466716637 | 0.000e+00 |

每组两阶段的新旧整体BBX边界和尺寸均核对；最大体积差为7.105e-15 L，属于浮点误差。相对Step4，删减共缩小19.167086 L，长脚增加80.575925 L，成品净增加61.408839 L（16.5497%），保持上一版结果。

51组采用6 mm圆整保留范围。`pose1+2+3+4+7+27`的6／8／12 mm候选都被原材料包含验收拒绝，保守保留整块BBX裁边主体；该组两阶段仍通过，BBX不变。必要的接触／工作／退出边界保留，6 mm半径不代表所有局部最小厚度或结构强度证书。

整块保留的代价是保留更多原有材料：Step5.2由2216.873变为3650.240 cm³，成品由8831.761变为10145.041 cm³。两者的整体BBX均不增加，Step5.2仍只删原Step4主体。本次主体删除约4954.629 cm³。Step5.3保留上一版脚的位置、路径和截面，在新主体上重新构造及全量验收，没有重新搜索脚。

每个安装state内部的小盒可能因保留更多材料而扩大；这是与整体静态并集BBX分开的诊断，审计CSV保留其变化。整体盒及物体／支撑布局没有变化。

## 全部52组及图片

每组Step5.2／Step5.3均PASS。O为object pose数，S为实际支撑安装pose数，R为原位复用，J为Juxtapose。BBX单位L；净增量及百分比比较成品与Step4。两阶段与上一版比较均不变，精确差值见[审计CSV](data/results/B/comparison_aesthetic_pruning_all52_20261013_01.csv)。

| Set | O | S | R | J | 无支撑 | Step4 | Step5.2 | Step5.3 | 净增量 | 增幅 | 前后图 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pose2+20 | 2 | 2 | 2 | 0 | 4.110 | 4.423 | 4.110 | 4.349 | -0.074 | -1.66% | [5.2](output/B/pose2+20/step5.2/comparison.png) · [5.3](output/B/pose2+20/step5.3/comparison.png) |
| illegal/pose2+29 | 2 | 1 | 1 | 1 | 4.009 | 4.412 | 4.009 | 4.009 | -0.403 | -9.14% | [5.2](output/B/illegal/pose2+29/step5.2/comparison.png) · [5.3](output/B/illegal/pose2+29/step5.3/comparison.png) |
| pose3+15 | 2 | 1 | 1 | 1 | 4.414 | 4.524 | 4.414 | 4.916 | +0.392 | +8.66% | [5.2](output/B/pose3+15/step5.2/comparison.png) · [5.3](output/B/pose3+15/step5.3/comparison.png) |
| pose8+21 | 2 | 2 | 2 | 0 | 3.489 | 3.856 | 3.602 | 4.138 | +0.283 | +7.33% | [5.2](output/B/pose8+21/step5.2/comparison.png) · [5.3](output/B/pose8+21/step5.3/comparison.png) |
| illegal/pose11+19 | 2 | 1 | 1 | 1 | 5.005 | 5.812 | 5.188 | 5.867 | +0.055 | +0.95% | [5.2](output/B/illegal/pose11+19/step5.2/comparison.png) · [5.3](output/B/illegal/pose11+19/step5.3/comparison.png) |
| pose19+28 | 2 | 2 | 2 | 0 | 4.278 | 6.839 | 6.173 | 7.956 | +1.117 | +16.33% | [5.2](output/B/pose19+28/step5.2/comparison.png) · [5.3](output/B/pose19+28/step5.3/comparison.png) |
| illegal/pose1+2+29 | 3 | 2 | 2 | 1 | 4.584 | 5.309 | 5.027 | 6.307 | +0.998 | +18.81% | [5.2](output/B/illegal/pose1+2+29/step5.2/comparison.png) · [5.3](output/B/illegal/pose1+2+29/step5.3/comparison.png) |
| pose1+12+29 | 3 | 3 | 3 | 0 | 4.676 | 5.690 | 5.389 | 6.679 | +0.988 | +17.37% | [5.2](output/B/pose1+12+29/step5.2/comparison.png) · [5.3](output/B/pose1+12+29/step5.3/comparison.png) |
| pose4+5+7 | 3 | 3 | 3 | 0 | 4.709 | 5.216 | 4.827 | 6.246 | +1.030 | +19.74% | [5.2](output/B/pose4+5+7/step5.2/comparison.png) · [5.3](output/B/pose4+5+7/step5.3/comparison.png) |
| illegal/pose5+13+27 | 3 | 1 | 1 | 2 | 4.563 | 5.363 | 4.978 | 4.978 | -0.386 | -7.19% | [5.2](output/B/illegal/pose5+13+27/step5.2/comparison.png) · [5.3](output/B/illegal/pose5+13+27/step5.3/comparison.png) |
| pose8+10+19 | 3 | 3 | 3 | 0 | 5.658 | 6.148 | 5.818 | 11.643 | +5.495 | +89.39% | [5.2](output/B/pose8+10+19/step5.2/comparison.png) · [5.3](output/B/pose8+10+19/step5.3/comparison.png) |
| pose18+23+24 | 3 | 2 | 2 | 1 | 5.328 | 5.791 | 5.715 | 7.136 | +1.345 | +23.22% | [5.2](output/B/pose18+23+24/step5.2/comparison.png) · [5.3](output/B/pose18+23+24/step5.3/comparison.png) |
| pose1+2+3+27 | 4 | 2 | 2 | 2 | 4.876 | 5.959 | 5.255 | 5.814 | -0.145 | -2.43% | [5.2](output/B/pose1+2+3+27/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+27/step5.3/comparison.png) |
| pose1+2+4+6 | 4 | 4 | 4 | 0 | 4.621 | 5.287 | 5.098 | 7.758 | +2.471 | +46.75% | [5.2](output/B/pose1+2+4+6/step5.2/comparison.png) · [5.3](output/B/pose1+2+4+6/step5.3/comparison.png) |
| pose1+2+4+7 | 4 | 4 | 4 | 0 | 4.239 | 4.948 | 4.948 | 7.532 | +2.584 | +52.24% | [5.2](output/B/pose1+2+4+7/step5.2/comparison.png) · [5.3](output/B/pose1+2+4+7/step5.3/comparison.png) |
| pose1+2+4+19 | 4 | 4 | 4 | 0 | 5.165 | 7.107 | 6.534 | 21.847 | +14.741 | +207.42% | [5.2](output/B/pose1+2+4+19/step5.2/comparison.png) · [5.3](output/B/pose1+2+4+19/step5.3/comparison.png) |
| illegal/pose1+2+6+15 | 4 | 1 | 1 | 3 | 4.638 | 5.211 | 4.987 | 5.586 | +0.375 | +7.20% | [5.2](output/B/illegal/pose1+2+6+15/step5.2/comparison.png) · [5.3](output/B/illegal/pose1+2+6+15/step5.3/comparison.png) |
| pose1+11+14+27 | 4 | 2 | 2 | 2 | 4.326 | 5.355 | 4.987 | 5.535 | +0.181 | +3.37% | [5.2](output/B/pose1+11+14+27/step5.2/comparison.png) · [5.3](output/B/pose1+11+14+27/step5.3/comparison.png) |
| pose2+3+4+7 | 4 | 4 | 4 | 0 | 4.637 | 5.716 | 5.404 | 7.997 | +2.281 | +39.91% | [5.2](output/B/pose2+3+4+7/step5.2/comparison.png) · [5.3](output/B/pose2+3+4+7/step5.3/comparison.png) |
| pose5+6+23+29 | 4 | 4 | 4 | 0 | 5.504 | 6.680 | 6.607 | 8.081 | +1.401 | +20.97% | [5.2](output/B/pose5+6+23+29/step5.2/comparison.png) · [5.3](output/B/pose5+6+23+29/step5.3/comparison.png) |
| illegal/pose7+11+13+19 | 4 | 2 | 2 | 2 | 5.005 | 5.720 | 5.557 | 6.141 | +0.421 | +7.36% | [5.2](output/B/illegal/pose7+11+13+19/step5.2/comparison.png) · [5.3](output/B/illegal/pose7+11+13+19/step5.3/comparison.png) |
| pose8+9+13+30 | 4 | 3 | 3 | 1 | 4.203 | 4.730 | 4.328 | 5.957 | +1.227 | +25.94% | [5.2](output/B/pose8+9+13+30/step5.2/comparison.png) · [5.3](output/B/pose8+9+13+30/step5.3/comparison.png) |
| pose1+2+3+4+6 | 5 | 2 | 2 | 3 | 4.621 | 6.652 | 6.191 | 6.408 | -0.244 | -3.66% | [5.2](output/B/pose1+2+3+4+6/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+4+6/step5.3/comparison.png) |
| pose1+2+3+4+27 | 5 | 4 | 4 | 1 | 4.876 | 5.805 | 5.416 | 7.679 | +1.874 | +32.29% | [5.2](output/B/pose1+2+3+4+27/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+4+27/step5.3/comparison.png) |
| pose1+2+3+7+27 | 5 | 5 | 5 | 0 | 4.876 | 5.270 | 5.032 | 9.044 | +3.774 | +71.62% | [5.2](output/B/pose1+2+3+7+27/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+7+27/step5.3/comparison.png) |
| pose1+2+4+5+6 | 5 | 5 | 5 | 0 | 5.268 | 5.966 | 5.626 | 8.155 | +2.189 | +36.69% | [5.2](output/B/pose1+2+4+5+6/step5.2/comparison.png) · [5.3](output/B/pose1+2+4+5+6/step5.3/comparison.png) |
| illegal/pose1+2+6+7+29 | 5 | 4 | 4 | 1 | 4.667 | 5.471 | 5.276 | 6.310 | +0.839 | +15.34% | [5.2](output/B/illegal/pose1+2+6+7+29/step5.2/comparison.png) · [5.3](output/B/illegal/pose1+2+6+7+29/step5.3/comparison.png) |
| pose1+4+7+9+24 | 5 | 1 | 1 | 4 | 4.171 | 4.809 | 4.461 | 4.461 | -0.348 | -7.23% | [5.2](output/B/pose1+4+7+9+24/step5.2/comparison.png) · [5.3](output/B/pose1+4+7+9+24/step5.3/comparison.png) |
| pose5+6+11+23+29 | 5 | 4 | 4 | 1 | 5.504 | 7.441 | 7.277 | 9.001 | +1.560 | +20.97% | [5.2](output/B/pose5+6+11+23+29/step5.2/comparison.png) · [5.3](output/B/pose5+6+11+23+29/step5.3/comparison.png) |
| illegal/pose5+6+13+15+27 | 5 | 1 | 1 | 4 | 5.196 | 6.684 | 6.047 | 6.371 | -0.313 | -4.69% | [5.2](output/B/illegal/pose5+6+13+15+27/step5.2/comparison.png) · [5.3](output/B/illegal/pose5+6+13+15+27/step5.3/comparison.png) |
| pose6+10+13+17+30 | 5 | 1 | 1 | 4 | 4.479 | 5.933 | 5.529 | 5.606 | -0.328 | -5.52% | [5.2](output/B/pose6+10+13+17+30/step5.2/comparison.png) · [5.3](output/B/pose6+10+13+17+30/step5.3/comparison.png) |
| pose12+16+19+21+27 | 5 | 4 | 4 | 1 | 5.192 | 6.310 | 6.157 | 15.026 | +8.716 | +138.14% | [5.2](output/B/pose12+16+19+21+27/step5.2/comparison.png) · [5.3](output/B/pose12+16+19+21+27/step5.3/comparison.png) |
| pose1+2+3+4+5+6 | 6 | 2 | 2 | 4 | 5.268 | 7.306 | 6.819 | 7.266 | -0.039 | -0.54% | [5.2](output/B/pose1+2+3+4+5+6/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+4+5+6/step5.3/comparison.png) |
| pose1+2+3+4+6+7 | 6 | 5 | 5 | 1 | 4.667 | 6.181 | 5.952 | 8.872 | +2.691 | +43.54% | [5.2](output/B/pose1+2+3+4+6+7/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+4+6+7/step5.3/comparison.png) |
| pose1+2+3+4+7+27 | 6 | 2 | 2 | 4 | 4.876 | 5.933 | 5.758 | 7.041 | +1.108 | +18.68% | [5.2](output/B/pose1+2+3+4+7+27/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+4+7+27/step5.3/comparison.png) |
| pose1+2+4+5+6+7 | 6 | 5 | 5 | 1 | 5.268 | 6.304 | 5.917 | 9.467 | +3.163 | +50.17% | [5.2](output/B/pose1+2+4+5+6+7/step5.2/comparison.png) · [5.3](output/B/pose1+2+4+5+6+7/step5.3/comparison.png) |
| pose1+2+4+5+6+11 | 6 | 1 | 1 | 5 | 5.268 | 6.786 | 6.274 | 6.274 | -0.512 | -7.54% | [5.2](output/B/pose1+2+4+5+6+11/step5.2/comparison.png) · [5.3](output/B/pose1+2+4+5+6+11/step5.3/comparison.png) |
| pose1+4+7+12+21+27 | 6 | 1 | 1 | 5 | 4.571 | 6.089 | 5.572 | 5.572 | -0.516 | -8.48% | [5.2](output/B/pose1+4+7+12+21+27/step5.2/comparison.png) · [5.3](output/B/pose1+4+7+12+21+27/step5.3/comparison.png) |
| pose1+6+11+13+14+17 | 6 | 2 | 2 | 4 | 4.956 | 5.747 | 5.588 | 6.200 | +0.454 | +7.89% | [5.2](output/B/pose1+6+11+13+14+17/step5.2/comparison.png) · [5.3](output/B/pose1+6+11+13+14+17/step5.3/comparison.png) |
| pose4+5+8+9+19+23 | 6 | 1 | 1 | 5 | 5.981 | 7.571 | 7.203 | 7.203 | -0.368 | -4.86% | [5.2](output/B/pose4+5+8+9+19+23/step5.2/comparison.png) · [5.3](output/B/pose4+5+8+9+19+23/step5.3/comparison.png) |
| illegal/pose4+7+12+21+23+27 | 6 | 2 | 2 | 4 | 4.791 | 6.942 | 5.851 | 6.536 | -0.405 | -5.84% | [5.2](output/B/illegal/pose4+7+12+21+23+27/step5.2/comparison.png) · [5.3](output/B/illegal/pose4+7+12+21+23+27/step5.3/comparison.png) |
| illegal/pose5+12+20+24+27+28 | 6 | 1 | 1 | 5 | 5.176 | 6.742 | 6.462 | 6.462 | -0.279 | -4.14% | [5.2](output/B/illegal/pose5+12+20+24+27+28/step5.2/comparison.png) · [5.3](output/B/illegal/pose5+12+20+24+27+28/step5.3/comparison.png) |
| pose1+2+3+4+5+6+7 | 7 | 5 | 5 | 2 | 5.268 | 7.152 | 6.513 | 9.473 | +2.321 | +32.45% | [5.2](output/B/pose1+2+3+4+5+6+7/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+4+5+6+7/step5.3/comparison.png) |
| pose1+2+4+5+6+7+11 | 7 | 1 | 1 | 6 | 5.268 | 6.761 | 6.274 | 6.452 | -0.309 | -4.58% | [5.2](output/B/pose1+2+4+5+6+7+11/step5.2/comparison.png) · [5.3](output/B/pose1+2+4+5+6+7+11/step5.3/comparison.png) |
| pose1+4+7+12+21+23+27 | 7 | 1 | 1 | 6 | 4.791 | 6.632 | 6.211 | 6.211 | -0.421 | -6.34% | [5.2](output/B/pose1+4+7+12+21+23+27/step5.2/comparison.png) · [5.3](output/B/pose1+4+7+12+21+23+27/step5.3/comparison.png) |
| pose1+2+3+4+5+6+7+8 | 8 | 4 | 4 | 4 | 5.268 | 9.053 | 8.011 | 10.679 | +1.626 | +17.96% | [5.2](output/B/pose1+2+3+4+5+6+7+8/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+4+5+6+7+8/step5.3/comparison.png) |
| pose1+2+4+5+6+7+10+11 | 8 | 1 | 1 | 7 | 5.268 | 7.275 | 7.004 | 7.249 | -0.026 | -0.36% | [5.2](output/B/pose1+2+4+5+6+7+10+11/step5.2/comparison.png) · [5.3](output/B/pose1+2+4+5+6+7+10+11/step5.3/comparison.png) |
| pose1+3+7+10+14+18+23+27 | 8 | 1 | 1 | 7 | 6.044 | 13.823 | 13.609 | 13.609 | -0.214 | -1.55% | [5.2](output/B/pose1+3+7+10+14+18+23+27/step5.2/comparison.png) · [5.3](output/B/pose1+3+7+10+14+18+23+27/step5.3/comparison.png) |
| pose1+2+3+4+5+6+7+8+9 | 9 | 1 | 1 | 8 | 5.268 | 15.109 | 14.807 | 15.201 | +0.092 | +0.61% | [5.2](output/B/pose1+2+3+4+5+6+7+8+9/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+4+5+6+7+8+9/step5.3/comparison.png) |
| pose1+4+7+9+12+16+21+23+27 | 9 | 1 | 1 | 8 | 4.791 | 15.574 | 15.360 | 15.431 | -0.143 | -0.92% | [5.2](output/B/pose1+4+7+9+12+16+21+23+27/step5.2/comparison.png) · [5.3](output/B/pose1+4+7+9+12+16+21+23+27/step5.3/comparison.png) |
| pose1+2+3+4+5+6+7+8+9+10 | 10 | 1 | 1 | 9 | 5.268 | 20.918 | 20.628 | 20.628 | -0.290 | -1.38% | [5.2](output/B/pose1+2+3+4+5+6+7+8+9+10/step5.2/comparison.png) · [5.3](output/B/pose1+2+3+4+5+6+7+8+9+10/step5.3/comparison.png) |
| pose1+3+5+7+9+12+16+21+23+27 | 10 | 1 | 1 | 9 | 5.469 | 22.726 | 22.106 | 22.106 | -0.620 | -2.73% | [5.2](output/B/pose1+3+5+7+9+12+16+21+23+27/step5.2/comparison.png) · [5.3](output/B/pose1+3+5+7+9+12+16+21+23+27/step5.3/comparison.png) |

图片直接放在`output/B/[set]/step5.2/`和`step5.3/`。每阶段的`comparison.png`左旧右新；`overview.png`、`all_poses.png`、各`pose_N.png`及`evaluation.png`已全部更新。Step5.2另有删除材料图。104张同视角前后对比，加标准阶段／Evaluation图共1518张，逐一核对PNG及哈希；原Step4图片保持原样，无HTML。

![Step5.2前后对比](output/B/pose1+4+7+9+24/step5.2/comparison.png)

[全部组BBX图](output/B/all_bbx_aesthetic_pruning_all52_20261013_01.png) · [BBX Evaluation](data/results/B/evaluation_aesthetic_pruning_all52_20261013_01_bbx.md) · [逐pose角色CSV](data/results/B/evaluation_aesthetic_pruning_all52_20261013_01_poses.csv)。

## 验收与来源

- 新Step5.2与Step5.3分别实际检查每个pose全部32,768个原始需求。Step5.2保存8,617,984个通过mask；Step5.3重新检查实际新成品，全部通过。原七维反力及物理容差保持。
- 两阶段导出OBJ后重新读取，按原限值检查真实接触和完整需求；独立实体审计核对实际连通、封闭、原材料包含、合法空间、物体内部排除及主体保留。几何体积限值仍为1e-10 m³。
- 每组使用自己的冻结Step4布局、Step5.1需求和旧认证参考，未混用其它布局。51组6 mm候选和1组整块裁边回退记录真实保存。
- 先3组试跑，再同一实现运行全部52组；这次52组两阶段均通过。原连通Step4包含先前冷启动重试，其计算记录不改变。
- 10952个原数值／输入文件、6810张旧图片副本保持哈希；385份执行源码副本核对。旧结果没有覆盖。

[新批次](data/results/B/data/aesthetic_pruning_all52_20261013_01/batch.json) · [Evaluation数值](data/results/B/data/system_aesthetic_pruning_all52_20261013_01/evaluation_results.json) · [实际实体审计](data/results/B/data/system_aesthetic_pruning_all52_20261013_01/actual_solid_geometry_audit.json) · [BBX／原始证书／输入／图片审计](data/results/B/data/system_aesthetic_pruning_all52_20261013_01/aesthetic_audit.json) · [同视角图片记录](data/results/B/data/system_aesthetic_pruning_all52_20261013_01/visual_comparisons.json) · [执行源码记录](data/results/B/data/system_aesthetic_pruning_all52_20261013_01/execution.json)。

有针对性的20项测试已通过：包括人工细连接的加宽、整面BBX裁边、固定脚重建，以及现有删减和连通主体检查。
