# B：重合物体与全表面包裹

当前输出已按新流程重新生成；旧单 pose greedy 支撑和旧 Step4 诊断已删除。

| Pose set | Step3 结果 | 实际接触面积 cm² | Step4 |
|---|---|---:|---|
| [pose3+15](pose3+15/step3/README.md) | PASS | 473.00 | 可进入通道雕刻 |
| [pose19+28](pose19+28/step3/README.md) | PASS | 479.78 | 可进入通道雕刻 |
| [pose8+21](pose8+21/step3/README.md) | PASS | 469.40 | 可进入通道雕刻 |
| [pose2+20](pose2+20/step3/README.md) | PASS | 464.56 | 可进入通道雕刻 |
| [pose1+12+29](pose1+12+29/step3/README.md) | PASS | 440.29 | 可进入通道雕刻 |
| [pose4+5+7](pose4+5+7/step3/README.md) | PASS | 460.60 | 可进入通道雕刻 |
| [pose8+10+19](pose8+10+19/step3/README.md) | PASS | 420.71 | 可进入通道雕刻 |
| [pose18+23+24](pose18+23+24/step3/README.md) | PASS | 427.43 | 可进入通道雕刻 |
| [pose8+9+13+30](pose8+9+13+30/step3/README.md) | PASS | 404.71 | 可进入通道雕刻 |
| [pose2+3+4+7](pose2+3+4+7/step3/README.md) | PASS | 407.98 | 可进入通道雕刻 |
| [pose1+11+14+27](pose1+11+14+27/step3/README.md) | PASS | 392.37 | 可进入通道雕刻 |
| [pose5+6+23+29](pose5+6+23+29/step3/README.md) | PASS | 385.44 | 可进入通道雕刻 |
| [pose1+4+7+9+24](pose1+4+7+9+24/step3/README.md) | PASS | 343.13 | 可进入通道雕刻 |
| [pose5+6+11+23+29](pose5+6+11+23+29/step3/README.md) | PASS | 348.03 | 可进入通道雕刻 |
| [pose12+16+19+21+27](pose12+16+19+21+27/step3/README.md) | PASS | 342.13 | 可进入通道雕刻 |
| [pose6+10+13+17+30](pose6+10+13+17+30/step3/README.md) | PASS | 372.24 | 可进入通道雕刻 |
| [pose1+2+4+5+6+7](pose1+2+4+5+6+7/step3/README.md) | PASS | 361.58 | 可进入通道雕刻 |
| [pose4+5+8+9+19+23](pose4+5+8+9+19+23/step3/README.md) | PASS | 361.50 | 可进入通道雕刻 |
| [pose1+6+11+13+14+17](pose1+6+11+13+14+17/step3/README.md) | PASS | 343.48 | 可进入通道雕刻 |
| [pose1+4+7+12+21+27](pose1+4+7+12+21+27/step3/README.md) | PASS | 353.45 | 可进入通道雕刻 |

共 20 组、80 个 pose 实例：20 PASS，0 FAIL，0 UNRESOLVED。

PASS 只表示包裹接触的力／力矩阶段通过，尚未得到可装卸的完整共享夹具。内部数据在 `data/`，公开结果查看各组图片、模型和说明。

## Step3.3：各 pose 下的支撑

每组只保留一张 `step3/step3.3/overview.png`，按各 pose 分格显示壳子、全部接地圈、当前地面与原始撒点。不生成连接杆，不要求实体连通。

| Pose set | 图 |
|---|---|
| pose3+15 | [各 pose 视图](pose3+15/step3/step3.3/overview.png) |
| pose19+28 | [各 pose 视图](pose19+28/step3/step3.3/overview.png) |
| pose8+21 | [各 pose 视图](pose8+21/step3/step3.3/overview.png) |
| pose2+20 | [各 pose 视图](pose2+20/step3/step3.3/overview.png) |
| pose1+12+29 | [各 pose 视图](pose1+12+29/step3/step3.3/overview.png) |
| pose4+5+7 | [各 pose 视图](pose4+5+7/step3/step3.3/overview.png) |
| pose8+10+19 | [各 pose 视图](pose8+10+19/step3/step3.3/overview.png) |
| pose18+23+24 | [各 pose 视图](pose18+23+24/step3/step3.3/overview.png) |
| pose8+9+13+30 | [各 pose 视图](pose8+9+13+30/step3/step3.3/overview.png) |
| pose2+3+4+7 | [各 pose 视图](pose2+3+4+7/step3/step3.3/overview.png) |
| pose1+11+14+27 | [各 pose 视图](pose1+11+14+27/step3/step3.3/overview.png) |
| pose5+6+23+29 | [各 pose 视图](pose5+6+23+29/step3/step3.3/overview.png) |
| pose1+4+7+9+24 | [各 pose 视图](pose1+4+7+9+24/step3/step3.3/overview.png) |
| pose5+6+11+23+29 | [各 pose 视图](pose5+6+11+23+29/step3/step3.3/overview.png) |
| pose12+16+19+21+27 | [各 pose 视图](pose12+16+19+21+27/step3/step3.3/overview.png) |
| pose6+10+13+17+30 | [各 pose 视图](pose6+10+13+17+30/step3/step3.3/overview.png) |
| pose1+2+4+5+6+7 | [各 pose 视图](pose1+2+4+5+6+7/step3/step3.3/overview.png) |
| pose4+5+8+9+19+23 | [各 pose 视图](pose4+5+8+9+19+23/step3/step3.3/overview.png) |
| pose1+6+11+13+14+17 | [各 pose 视图](pose1+6+11+13+14+17/step3/step3.3/overview.png) |
| pose1+4+7+12+21+27 | [各 pose 视图](pose1+4+7+12+21+27/step3/step3.3/overview.png) |

## Step4.1：退出初始化

各 pose 的退出路径都初始化为自己的原生世界 +z 向上方向；未执行路径优化。青色为扫掠，红色为必须切除材料，灰色为剩余支撑。每组仅一张图。

| Pose set | 图 | 原始力／力矩需求 | 原环剩余覆盖（诊断） |
|---|---|---|---|
| pose3+15 | [初始化图](pose3+15/step4/step4.1/overview.png) | 通过 | 后续重建接地材料 |
| pose19+28 | [初始化图](pose19+28/step4/step4.1/overview.png) | 通过 | 后续重建接地材料 |
| pose8+21 | [初始化图](pose8+21/step4/step4.1/overview.png) | 通过 | 通过 |
| pose2+20 | [初始化图](pose2+20/step4/step4.1/overview.png) | 通过 | 通过 |
| pose1+12+29 | [初始化图](pose1+12+29/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose4+5+7 | [初始化图](pose4+5+7/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose8+10+19 | [初始化图](pose8+10+19/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose18+23+24 | [初始化图](pose18+23+24/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose8+9+13+30 | [初始化图](pose8+9+13+30/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose2+3+4+7 | [初始化图](pose2+3+4+7/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose1+11+14+27 | [初始化图](pose1+11+14+27/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose5+6+23+29 | [初始化图](pose5+6+23+29/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose1+4+7+9+24 | [初始化图](pose1+4+7+9+24/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose5+6+11+23+29 | [初始化图](pose5+6+11+23+29/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose12+16+19+21+27 | [初始化图](pose12+16+19+21+27/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose6+10+13+17+30 | [初始化图](pose6+10+13+17+30/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose1+2+4+5+6+7 | [初始化图](pose1+2+4+5+6+7/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose4+5+8+9+19+23 | [初始化图](pose4+5+8+9+19+23/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose1+6+11+13+14+17 | [初始化图](pose1+6+11+13+14+17/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |
| pose1+4+7+12+21+27 | [初始化图](pose1+4+7+12+21+27/step4/step4.1/overview.png) | 未通过 | 后续重建接地材料 |

## Step4.2：退出路径共同优化、恢复承载

**20/20 组、80 个 pose 实例的全部原始力／力矩需求通过。** 路径连续退出且不进入自身地面；本阶段不要求材料连通，也未重建接地材料。绿色表示相对 Step4.1 恢复的材料。

| Pose set | 图 | 恢复材料 cm³ | 搜索组合数 |
|---|---|---:|---:|
| pose3+15 | [恢复后的支撑](pose3+15/step4/step4.2/overview.png) | 0.00 | 1 |
| pose19+28 | [恢复后的支撑](pose19+28/step4/step4.2/overview.png) | 124.74 | 2 |
| pose8+21 | [恢复后的支撑](pose8+21/step4/step4.2/overview.png) | 0.16 | 1 |
| pose2+20 | [恢复后的支撑](pose2+20/step4/step4.2/overview.png) | 0.00 | 1 |
| pose1+12+29 | [恢复后的支撑](pose1+12+29/step4/step4.2/overview.png) | 76.98 | 2 |
| pose4+5+7 | [恢复后的支撑](pose4+5+7/step4/step4.2/overview.png) | 118.00 | 2 |
| pose8+10+19 | [恢复后的支撑](pose8+10+19/step4/step4.2/overview.png) | 124.77 | 2 |
| pose18+23+24 | [恢复后的支撑](pose18+23+24/step4/step4.2/overview.png) | 120.13 | 2 |
| pose8+9+13+30 | [恢复后的支撑](pose8+9+13+30/step4/step4.2/overview.png) | 108.81 | 7 |
| pose2+3+4+7 | [恢复后的支撑](pose2+3+4+7/step4/step4.2/overview.png) | 116.89 | 2 |
| pose1+11+14+27 | [恢复后的支撑](pose1+11+14+27/step4/step4.2/overview.png) | 109.29 | 2 |
| pose5+6+23+29 | [恢复后的支撑](pose5+6+23+29/step4/step4.2/overview.png) | 102.40 | 2 |
| pose1+4+7+9+24 | [恢复后的支撑](pose1+4+7+9+24/step4/step4.2/overview.png) | 84.78 | 17 |
| pose5+6+11+23+29 | [恢复后的支撑](pose5+6+11+23+29/step4/step4.2/overview.png) | 81.39 | 9 |
| pose12+16+19+21+27 | [恢复后的支撑](pose12+16+19+21+27/step4/step4.2/overview.png) | 89.74 | 5 |
| pose6+10+13+17+30 | [恢复后的支撑](pose6+10+13+17+30/step4/step4.2/overview.png) | 99.40 | 4 |
| pose1+2+4+5+6+7 | [恢复后的支撑](pose1+2+4+5+6+7/step4/step4.2/overview.png) | 135.80 | 130 |
| pose4+5+8+9+19+23 | [恢复后的支撑](pose4+5+8+9+19+23/step4/step4.2/overview.png) | 96.69 | 324 |
| pose1+6+11+13+14+17 | [恢复后的支撑](pose1+6+11+13+14+17/step4/step4.2/overview.png) | 75.42 | 174 |
| pose1+4+7+12+21+27 | [恢复后的支撑](pose1+4+7+12+21+27/step4/step4.2/overview.png) | 85.66 | 1 |

## 无文字退出示意

灰色实体是支撑；蓝色半透明物体依次显示安装、中途和完全退出。每组一张图，按该组的 pose 顺序排列。

- [pose3+15](pose3+15/step4/step4.2/exit_motion.png)
- [pose19+28](pose19+28/step4/step4.2/exit_motion.png)
- [pose8+21](pose8+21/step4/step4.2/exit_motion.png)
- [pose2+20](pose2+20/step4/step4.2/exit_motion.png)
- [pose1+12+29](pose1+12+29/step4/step4.2/exit_motion.png)
- [pose4+5+7](pose4+5+7/step4/step4.2/exit_motion.png)
- [pose8+10+19](pose8+10+19/step4/step4.2/exit_motion.png)
- [pose18+23+24](pose18+23+24/step4/step4.2/exit_motion.png)
- [pose8+9+13+30](pose8+9+13+30/step4/step4.2/exit_motion.png)
- [pose2+3+4+7](pose2+3+4+7/step4/step4.2/exit_motion.png)
- [pose1+11+14+27](pose1+11+14+27/step4/step4.2/exit_motion.png)
- [pose5+6+23+29](pose5+6+23+29/step4/step4.2/exit_motion.png)
- [pose1+4+7+9+24](pose1+4+7+9+24/step4/step4.2/exit_motion.png)
- [pose5+6+11+23+29](pose5+6+11+23+29/step4/step4.2/exit_motion.png)
- [pose12+16+19+21+27](pose12+16+19+21+27/step4/step4.2/exit_motion.png)
- [pose6+10+13+17+30](pose6+10+13+17+30/step4/step4.2/exit_motion.png)
- [pose1+2+4+5+6+7](pose1+2+4+5+6+7/step4/step4.2/exit_motion.png)
- [pose4+5+8+9+19+23](pose4+5+8+9+19+23/step4/step4.2/exit_motion.png)
- [pose1+6+11+13+14+17](pose1+6+11+13+14+17/step4/step4.2/exit_motion.png)
- [pose1+4+7+12+21+27](pose1+4+7+12+21+27/step4/step4.2/exit_motion.png)
