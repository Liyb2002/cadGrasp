# seven_spread / incremental

**本结果违反工作射线约束。** 已在 5/7 个 pose 找到从原始工作面向外射线撞到支撑的真实反例。原有采样载荷通过不能视为该支撑可行。[遮挡诊断](../../work_ray_audit.md)。

快速采样搜索；这次按要求不做最终实体、退出和工作面验收。

[每步选择与新支撑](process.png) · [最终每个 pose](final_result.png) · [全部候选记录](trace.json) · [布局与过程](process.json) · [搜索统计](search_report.json)

当前加入 7/7 个 pose；采样力／力矩 通过；体积估计 128.21 cm³；搜索 27.3 s。

旧网格图已经存档。图里蓝色是从保存布局直接 Boolean 构造的三角网格边界：当前贴合壳并集，扣除所有当前物体、工作带和完整退出路径。没有挖平移轨迹，也没有造一个供物体横向滑动的通道。过程图不含文字；按行从左到右，每步只有一个等轴测视角，显示支撑、相关物体和橙色工作面积；最终图用同一蓝色形状随支撑摆放做刚体变换。

V~ 是搜索时的 Sobol 体积估计。搜索候选共同扩张采样框时 epoch 会变化，不同 epoch 的原始 V~ 不宜直接比较。下表增加／删除使用统一显示网格，可比较相邻形状。

| 步 | 选择 | 未满足载荷数 | V~ cm³ / epoch | 旧显示网格 + / − cm³ |
|---|---|---:|---:|---:|
| 0 | Initialize similar exits; rotate fixture reuse | 0 | 156.61 / 0 | +154.46 / −0.00 |
| 1 | Insert pose_4 at native seating | 0 | 133.22 / 0 | +0.00 / −24.05 |
| 2 | Insert pose_7 at native seating | 2076 | 72.80 / 0 | +0.00 / −58.19 |
| 3 | Direction-gradient-joint, step 8 deg | 1905 | 75.45 / 0 | +3.31 / −0.89 |
| 4 | Direction-coherent, step -5 deg | 1196 | 76.82 / 0 | +2.59 / −2.23 |
| 5 | Direction-gradient-joint, step 8 deg | 661 | 78.52 / 0 | +2.85 / −0.79 |
| 6 | Juxtapose pose_7 -> pose_4, offset 0.0 mm | 0 | 160.52 / 1 | +102.52 / −20.63 |
| 7 | Insert pose_12 at native seating | 0 | 160.52 / 1 | +0.00 / −0.03 |
| 8 | Insert pose_21 at native seating | 0 | 160.52 / 1 | +0.00 / −0.06 |
| 9 | Insert pose_23 at native seating | 0 | 160.52 / 1 | +0.00 / −0.00 |
| 10 | Insert pose_27 at native seating | 0 | 148.01 / 1 | +0.00 / −14.27 |
| 11 | Direction-coherent, step 5 deg | 0 | 139.67 / 1 | +0.90 / −8.64 |
| 12 | Direction-coherent, step -5 deg | 0 | 128.21 / 1 | +0.36 / −8.11 |

下面包括未更新形状的拒绝回合。Juxtapose 内部的临时调整在分支被选中后才成为过程步骤。

## 候选回合 0: incremental_registered_initial

选择结果：初始化 / 插入; 筛选数：—。


## 候选回合 1: insert_pose_4_registered

选择结果：初始化 / 插入; 筛选数：—。


## 候选回合 2: insert_pose_7_registered

选择结果：初始化 / 插入; 筛选数：—。


## 候选回合 3: rotate_reuse_direction

选择结果：True; 筛选数：42。

最难原始载荷：`{"pose_index": 0, "load_index": 16507, "loss": 0.020483326748531677, "residual": [0.07958027075043372, 0.04276998633807092, 0.01133346979626304, -0.06562255552591514, 0.14976226955153654, 0.07707695868669531, 1.6653345369377348e-16]}`。

- 选中：Direction-gradient-joint, step 8 deg；结果 {"counts": {"0": 31092, "1": 32539, "2": 32768}, "lost_protected_loads": 1905, "rank": [1905, 1905, 75.44766024785011, 0.02368330965935167, 0]}。
- 未选：Direction, step 5 deg；结果 {"counts": {"0": 31084, "1": 32506, "2": 32768}, "lost_protected_loads": 1946, "rank": [1946, 1946, 75.44766024785011, 0.02368330965935167, 0]}。
- 未选：Direction-gradient-joint, step 0.5 deg；结果 {"counts": {"0": 30995, "1": 32514, "2": 32768}, "lost_protected_loads": 2027, "rank": [2027, 2027, 73.22549914658104, 0.02368330965935167, 0]}。

## 候选回合 4: rotate_reuse_direction

选择结果：True; 筛选数：36。

最难原始载荷：`{"pose_index": 0, "load_index": 16507, "loss": 0.01925269280771404, "residual": [0.0756988715151437, 0.04291484306073665, 0.01135643988695656, -0.06801283281806539, 0.1446727728610504, 0.07244623765030818, 0.0]}`。

- 选中：Direction-coherent, step -5 deg；结果 {"counts": {"0": 31764, "1": 32576, "2": 32768}, "lost_protected_loads": 1196, "rank": [1196, 1196, 76.82328378673097, 0.02368330965935167, 0]}。
- 未选：Direction, step 5 deg；结果 {"counts": {"0": 31343, "1": 32543, "2": 32768}, "lost_protected_loads": 1650, "rank": [1650, 1650, 76.1883806149398, 0.02368330965935167, 0]}。
- 未选：Direction-coherent, step 2 deg；结果 {"counts": {"0": 31092, "1": 32539, "2": 32768}, "lost_protected_loads": 1905, "rank": [1905, 1905, 74.60112268546189, 0.02368330965935167, 0]}。

## 候选回合 5: rotate_reuse_direction

选择结果：True; 筛选数：42。

最难原始载荷：`{"pose_index": 0, "load_index": 27794, "loss": 0.014171655710546498, "residual": [0.06039214203170379, 0.04304811773617355, 0.011506114816887125, -0.07794300518451472, 0.11857470737123194, 0.050749346771580134, -5.551115123125783e-17]}`。

- 选中：Direction-gradient-joint, step 8 deg；结果 {"counts": {"0": 32249, "1": 32626, "2": 32768}, "lost_protected_loads": 661, "rank": [661, 661, 78.51635891150741, 0.02368330965935167, 0]}。
- 未选：Direction-coherent, step -5 deg；结果 {"counts": {"0": 32212, "1": 32576, "2": 32768}, "lost_protected_loads": 748, "rank": [748, 748, 76.29419781023833, 0.02368330965935167, 0]}。
- 未选：Direction-gradient-joint, step 4 deg；结果 {"counts": {"0": 32157, "1": 32635, "2": 32768}, "lost_protected_loads": 744, "rank": [744, 744, 77.45818695852213, 0.02368330965935167, 0]}。

## 候选回合 6: selective_juxtapose

选择结果：True; 筛选数：93。

最难原始载荷：`{"pose_index": 0, "load_index": 27794, "loss": 0.007957530842202402, "residual": [0.04009855332891138, 0.04014677574945717, 0.00999858781898122, -0.06771632063617869, 0.06685614924131006, 0.059499475392679635, 2.220446049250313e-16]}`。

- 选中：Juxtapose pose_7 -> pose_4, offset 0.0 mm；结果 {"counts": {"0": 32768, "1": 32768, "2": 32768}, "lost_protected_loads": 0, "rank": [0, 0, 160.51796379219587, 0.026446415931241748, 1]}。

## 候选回合 7: insert_pose_12_registered

选择结果：初始化 / 插入; 筛选数：—。


## 候选回合 8: insert_pose_21_registered

选择结果：初始化 / 插入; 筛选数：—。


## 候选回合 9: insert_pose_23_registered

选择结果：初始化 / 插入; 筛选数：—。


## 候选回合 10: insert_pose_27_registered

选择结果：初始化 / 插入; 筛选数：—。


## 候选回合 11: restore_rotating_reuse

选择结果：False; 筛选数：—。


## 候选回合 12: feasible_volume_descent

选择结果：True; 筛选数：108。

- 选中：Direction-coherent, step 5 deg；结果 {"counts": {"0": 32768, "1": 32768, "2": 32768, "3": 32768, "4": 32768, "5": 32768, "6": 32768}}。
- 未选：Translation pose_7, step 1.21 mm；结果 {"counts": {"0": 32768, "1": 32768, "2": 32768, "3": 32768, "4": 32768, "5": 32768, "6": 32768}}。
- 未选：Direction, step 5 deg；结果 {"counts": {"0": 32768, "1": 32768, "2": 32768, "3": 32768, "4": 32768, "5": 32768, "6": 32768}}。

## 候选回合 13: feasible_volume_descent

选择结果：True; 筛选数：108。

- 选中：Direction-coherent, step -5 deg；结果 {"counts": {"0": 32768, "1": 32768, "2": 32768, "3": 32768, "4": 32768, "5": 32768, "6": 32768}}。
- 未选：Translation pose_7, step 4.84 mm；结果 {"counts": {"0": 32768, "1": 32768, "2": 32768, "3": 32768, "4": 32768, "5": 32768, "6": 32768}}。
- 未选：Direction-coherent, step 5 deg；结果 {"counts": {"0": 32768, "1": 32768, "2": 32768, "3": 32768, "4": 32768, "5": 32768, "6": 32768}}。



当前 mesh：[support.obj](support.obj)。本集合的两个视频位于上一级：[搜索过程](../process.mp4)、[逐 pose 使用](../result.mp4)。固定单个等轴测视角，先 whole、后 incremental，中间短暂白场分隔。橙色面片为各 pose 原始工作面积；放稳后停顿 1 秒，工作面上的橙红色小箭头朝内，表示可能施加的力。该方法最终三角网格材料体积 129.006 cm³；仅重建展示几何，没有重新搜索或运行最终力学验收。
