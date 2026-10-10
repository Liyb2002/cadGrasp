# B：whole 注册与完整工作禁区初始化

[新 Step4 whole 结果](step4_results.md) · [方向、sweep、过程及最终图片](step4_index.html)

[继续减实体材料：六组对照](compact_comparison.md) · [最小支撑图片](compact_index.html)。全部42组已有可行性基线保留；新体积搜索分别运行 greedy 和多个布局／修复分支 beam，再选择实际通过且体积最小者。它保持全部原载荷与工作／退出约束，不把找到一个答案当作体积优化结束。

Step3.1 合并注册、贴合包裹和整组工作禁区切除；Step3.2 展示整组禁区并集的8个环绕等轴测视角。参考 pose 为每组保存顺序的第一个。沿用已有集合；原始物体、工作面和载荷不变。

[图片浏览](index.html) · [初始化记录](data/batch.json) · [旧结果](_history/before_whole_initialize_20261008/)

以下 INITIALIZED 表为 Step3 几何初始化和重合布局受力诊断。当前 Step4 已从这个初始化重新运行，原载荷、工作禁区和退出结果请读上面的 Step4 结果与图片索引；旧 Step3.3/Step4 保存在历史目录。整件支撑接地、连通和强度仍未验收。

| Pose set | 初始化 | 当前受力诊断 | 材料 cm³ | 图片 |
|---|---|---|---:|---|
| illegal/pose1+2+29 | initialized | pass | 161.099 | [支撑](illegal/pose1+2+29/step3/step3.1/overview.png) · [禁区](illegal/pose1+2+29/step3/step3.2/overview.png) |
| illegal/pose1+2+6+15 | initialized | pass | 160.087 | [支撑](illegal/pose1+2+6+15/step3/step3.1/overview.png) · [禁区](illegal/pose1+2+6+15/step3/step3.2/overview.png) |
| illegal/pose1+2+6+7+29 | initialized | pass | 101.723 | [支撑](illegal/pose1+2+6+7+29/step3/step3.1/overview.png) · [禁区](illegal/pose1+2+6+7+29/step3/step3.2/overview.png) |
| illegal/pose11+19 | initialized | pass | 173.403 | [支撑](illegal/pose11+19/step3/step3.1/overview.png) · [禁区](illegal/pose11+19/step3/step3.2/overview.png) |
| illegal/pose2+29 | initialized | pass | 193.262 | [支撑](illegal/pose2+29/step3/step3.1/overview.png) · [禁区](illegal/pose2+29/step3/step3.2/overview.png) |
| illegal/pose4+7+12+21+23+27 | initialized | pass | 64.198 | [支撑](illegal/pose4+7+12+21+23+27/step3/step3.1/overview.png) · [禁区](illegal/pose4+7+12+21+23+27/step3/step3.2/overview.png) |
| illegal/pose5+12+20+24+27+28 | initialized | pass | 155.560 | [支撑](illegal/pose5+12+20+24+27+28/step3/step3.1/overview.png) · [禁区](illegal/pose5+12+20+24+27+28/step3/step3.2/overview.png) |
| illegal/pose5+13+27 | initialized | pass | 238.978 | [支撑](illegal/pose5+13+27/step3/step3.1/overview.png) · [禁区](illegal/pose5+13+27/step3/step3.2/overview.png) |
| illegal/pose5+6+13+15+27 | initialized | pass | 204.585 | [支撑](illegal/pose5+6+13+15+27/step3/step3.1/overview.png) · [禁区](illegal/pose5+6+13+15+27/step3/step3.2/overview.png) |
| illegal/pose7+11+13+19 | initialized | pass | 161.781 | [支撑](illegal/pose7+11+13+19/step3/step3.1/overview.png) · [禁区](illegal/pose7+11+13+19/step3/step3.2/overview.png) |
| pose1+11+14+27 | initialized | pass | 94.664 | [支撑](pose1+11+14+27/step3/step3.1/overview.png) · [禁区](pose1+11+14+27/step3/step3.2/overview.png) |
| pose1+12+29 | initialized | pass | 167.726 | [支撑](pose1+12+29/step3/step3.1/overview.png) · [禁区](pose1+12+29/step3/step3.2/overview.png) |
| pose1+2+3+27 | initialized | pass | 142.023 | [支撑](pose1+2+3+27/step3/step3.1/overview.png) · [禁区](pose1+2+3+27/step3/step3.2/overview.png) |
| pose1+2+3+4+27 | initialized | pass | 110.089 | [支撑](pose1+2+3+4+27/step3/step3.1/overview.png) · [禁区](pose1+2+3+4+27/step3/step3.2/overview.png) |
| pose1+2+3+4+5+6 | initialized | pass | 106.795 | [支撑](pose1+2+3+4+5+6/step3/step3.1/overview.png) · [禁区](pose1+2+3+4+5+6/step3/step3.2/overview.png) |
| pose1+2+3+4+6 | initialized | pass | 113.678 | [支撑](pose1+2+3+4+6/step3/step3.1/overview.png) · [禁区](pose1+2+3+4+6/step3/step3.2/overview.png) |
| pose1+2+3+4+6+7 | initialized | pass | 76.546 | [支撑](pose1+2+3+4+6+7/step3/step3.1/overview.png) · [禁区](pose1+2+3+4+6+7/step3/step3.2/overview.png) |
| pose1+2+3+4+7+27 | initialized | pass | 69.864 | [支撑](pose1+2+3+4+7+27/step3/step3.1/overview.png) · [禁区](pose1+2+3+4+7+27/step3/step3.2/overview.png) |
| pose1+2+3+7+27 | initialized | pass | 84.743 | [支撑](pose1+2+3+7+27/step3/step3.1/overview.png) · [禁区](pose1+2+3+7+27/step3/step3.2/overview.png) |
| pose1+2+4+19 | initialized | pass | 139.424 | [支撑](pose1+2+4+19/step3/step3.1/overview.png) · [禁区](pose1+2+4+19/step3/step3.2/overview.png) |
| pose1+2+4+5+6 | initialized | pass | 145.737 | [支撑](pose1+2+4+5+6/step3/step3.1/overview.png) · [禁区](pose1+2+4+5+6/step3/step3.2/overview.png) |
| pose1+2+4+5+6+11 | initialized | pass | 81.744 | [支撑](pose1+2+4+5+6+11/step3/step3.1/overview.png) · [禁区](pose1+2+4+5+6+11/step3/step3.2/overview.png) |
| pose1+2+4+5+6+7 | initialized | pass | 83.964 | [支撑](pose1+2+4+5+6+7/step3/step3.1/overview.png) · [禁区](pose1+2+4+5+6+7/step3/step3.2/overview.png) |
| pose1+2+4+6 | initialized | pass | 152.621 | [支撑](pose1+2+4+6/step3/step3.1/overview.png) · [禁区](pose1+2+4+6/step3/step3.2/overview.png) |
| pose1+2+4+7 | initialized | pass | 97.048 | [支撑](pose1+2+4+7/step3/step3.1/overview.png) · [禁区](pose1+2+4+7/step3/step3.2/overview.png) |
| pose1+4+7+12+21+27 | initialized | pass | 69.865 | [支撑](pose1+4+7+12+21+27/step3/step3.1/overview.png) · [禁区](pose1+4+7+12+21+27/step3/step3.2/overview.png) |
| pose1+4+7+9+24 | initialized | pass | 76.586 | [支撑](pose1+4+7+9+24/step3/step3.1/overview.png) · [禁区](pose1+4+7+9+24/step3/step3.2/overview.png) |
| pose1+6+11+13+14+17 | initialized | pass | 87.974 | [支撑](pose1+6+11+13+14+17/step3/step3.1/overview.png) · [禁区](pose1+6+11+13+14+17/step3/step3.2/overview.png) |
| pose12+16+19+21+27 | initialized | pass | 110.449 | [支撑](pose12+16+19+21+27/step3/step3.1/overview.png) · [禁区](pose12+16+19+21+27/step3/step3.2/overview.png) |
| pose18+23+24 | initialized | pass | 138.981 | [支撑](pose18+23+24/step3/step3.1/overview.png) · [禁区](pose18+23+24/step3/step3.2/overview.png) |
| pose19+28 | initialized | pass | 207.807 | [支撑](pose19+28/step3/step3.1/overview.png) · [禁区](pose19+28/step3/step3.2/overview.png) |
| pose2+20 | initialized | pass | 200.067 | [支撑](pose2+20/step3/step3.1/overview.png) · [禁区](pose2+20/step3/step3.2/overview.png) |
| pose2+3+4+7 | initialized | pass | 115.720 | [支撑](pose2+3+4+7/step3/step3.1/overview.png) · [禁区](pose2+3+4+7/step3/step3.2/overview.png) |
| pose3+15 | initialized | pass | 213.840 | [支撑](pose3+15/step3/step3.1/overview.png) · [禁区](pose3+15/step3/step3.2/overview.png) |
| pose4+5+7 | initialized | pass | 153.038 | [支撑](pose4+5+7/step3/step3.1/overview.png) · [禁区](pose4+5+7/step3/step3.2/overview.png) |
| pose4+5+8+9+19+23 | initialized | pass | 115.278 | [支撑](pose4+5+8+9+19+23/step3/step3.1/overview.png) · [禁区](pose4+5+8+9+19+23/step3/step3.2/overview.png) |
| pose5+6+11+23+29 | initialized | pass | 89.372 | [支撑](pose5+6+11+23+29/step3/step3.1/overview.png) · [禁区](pose5+6+11+23+29/step3/step3.2/overview.png) |
| pose5+6+23+29 | initialized | pass | 137.643 | [支撑](pose5+6+23+29/step3/step3.1/overview.png) · [禁区](pose5+6+23+29/step3/step3.2/overview.png) |
| pose6+10+13+17+30 | initialized | pass | 78.899 | [支撑](pose6+10+13+17+30/step3/step3.1/overview.png) · [禁区](pose6+10+13+17+30/step3/step3.2/overview.png) |
| pose8+10+19 | initialized | pass | 145.773 | [支撑](pose8+10+19/step3/step3.1/overview.png) · [禁区](pose8+10+19/step3/step3.2/overview.png) |
| pose8+21 | initialized | pass | 176.254 | [支撑](pose8+21/step3/step3.1/overview.png) · [禁区](pose8+21/step3/step3.2/overview.png) |
| pose8+9+13+30 | initialized | pass | 126.922 | [支撑](pose8+9+13+30/step3/step3.1/overview.png) · [禁区](pose8+9+13+30/step3/step3.2/overview.png) |

共 42 个已有集合，42 个初始化完成，0 个几何未决。
圆弧为球形显示截断边界；算法切除完整半无限区域。蓝色为实际三角支撑；未使用voxel平滑。
