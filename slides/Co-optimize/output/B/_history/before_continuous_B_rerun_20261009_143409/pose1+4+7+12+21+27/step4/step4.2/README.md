# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 26.580 → 63.251 cm³。
转动支撑复用 3 个pose；Juxtapose 3 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 130879 | 33.332 |  |
| 1 | rotate_reuse_direction | 126880 | 32.909 | pose_1: host pose_1→pose_1, 方向5.000°, 平移0.000mm |
| 2 | rotate_reuse_direction | 126868 | 33.015 | pose_4: host pose_4→pose_4, 方向0.320°, 平移0.000mm |
| 3 | selective_juxtapose | 84618 | 24.204 | pose_1: host pose_1→pose_1, 方向2.641°, 平移0.000mm；pose_4: host pose_4→pose_12, 方向76.294°, 平移182.002mm |
| 4 | selective_juxtapose | 52700 | 47.363 | pose_27: host pose_27→pose_7, 方向0.000°, 平移151.463mm |
| 5 | direction_after_commit | 52700 | 45.107 | pose_12: host pose_12→pose_12, 方向5.000°, 平移0.000mm |
| 6 | direction_after_commit | 52041 | 45.107 | pose_7: host pose_7→pose_7, 方向2.000°, 平移0.000mm |
| 7 | selective_juxtapose | 15697 | 64.968 | pose_1: host pose_1→pose_1, 方向3.329°, 平移0.000mm；pose_4: host pose_12→pose_12, 方向0.925°, 平移0.000mm；pose_7: host pose_7→pose_21, 方向79.072°, 平移172.844mm；pose_12: host pose_12→pose_12, 方向11.842°, 平移0.000mm；pose_21: host pose_21→pose_21, 方向4.733°, 平移0.000mm；pose_27: host pose_7→pose_7, 方向4.208°, 平移0.000mm |
| 8 | direction_after_commit | 5485 | 68.528 | pose_1: host pose_1→pose_1, 方向3.716°, 平移0.000mm；pose_4: host pose_12→pose_12, 方向4.901°, 平移0.000mm；pose_7: host pose_21→pose_21, 方向4.987°, 平移0.000mm；pose_12: host pose_12→pose_12, 方向2.617°, 平移0.000mm；pose_21: host pose_21→pose_21, 方向2.498°, 平移0.000mm；pose_27: host pose_7→pose_7, 方向2.770°, 平移0.000mm |
| 9 | direction_after_commit | 728 | 68.528 | pose_7: host pose_21→pose_21, 方向5.000°, 平移0.000mm |
| 10 | selective_juxtapose | 0 | 78.692 | pose_4: host pose_12→pose_12, 方向0.000°, 平移9.674mm；pose_7: host pose_21→pose_12, 方向74.135°, 平移139.252mm |
| 11 | feasible_volume_descent | 0 | 73.265 | pose_7: host pose_12→pose_12, 方向0.000°, 平移4.837mm |
| 12 | feasible_volume_descent | 0 | 69.647 | pose_1: host pose_1→pose_1, 方向3.722°, 平移0.000mm；pose_4: host pose_12→pose_12, 方向4.902°, 平移0.000mm；pose_7: host pose_12→pose_12, 方向2.624°, 平移0.000mm；pose_12: host pose_12→pose_12, 方向2.624°, 平移0.000mm；pose_21: host pose_21→pose_21, 方向2.505°, 平移0.000mm；pose_27: host pose_7→pose_7, 方向2.778°, 平移0.000mm |
| 13 | feasible_volume_descent | 0 | 63.315 | pose_27: host pose_7→pose_7, 方向0.000°, 平移4.837mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。

<!-- physical-operation-metrics -->

换 host 时，上表的方向／平移是支撑坐标参数差，包含坐标系变更。物体实际变化应读下面的世界中心距离和物体自身退出角；原日志和布局保持不变。

| 步 | pose | 世界物体中心移动mm | 物体自身退出角变化° |
|---|---|---:|---:|
| 1 | pose_1 | 0.000 | 5.000 |
| 2 | pose_4 | 0.000 | 0.320 |
| 3 | pose_1 | 0.000 | 2.641 |
| 3 | pose_4 | 9.749 | 68.031 |
| 4 | pose_27 | 0.000 | 52.807 |
| 5 | pose_12 | 0.000 | 5.000 |
| 6 | pose_7 | 0.000 | 2.000 |
| 7 | pose_1 | 0.000 | 3.329 |
| 7 | pose_4 | 0.000 | 0.925 |
| 7 | pose_7 | 55.104 | 4.024 |
| 7 | pose_12 | 0.000 | 11.842 |
| 7 | pose_21 | 0.000 | 4.733 |
| 7 | pose_27 | 0.000 | 4.208 |
| 8 | pose_1 | 0.000 | 3.716 |
| 8 | pose_4 | 0.000 | 4.901 |
| 8 | pose_7 | 0.000 | 4.987 |
| 8 | pose_12 | 0.000 | 2.617 |
| 8 | pose_21 | 0.000 | 2.498 |
| 8 | pose_27 | 0.000 | 2.770 |
| 9 | pose_7 | 0.000 | 5.000 |
| 10 | pose_4 | 9.674 | 0.000 |
| 10 | pose_7 | 55.104 | 45.315 |
| 11 | pose_7 | 4.837 | 0.000 |
| 12 | pose_1 | 0.000 | 3.722 |
| 12 | pose_4 | 0.000 | 4.902 |
| 12 | pose_7 | 0.000 | 2.624 |
| 12 | pose_12 | 0.000 | 2.624 |
| 12 | pose_21 | 0.000 | 2.505 |
| 12 | pose_27 | 0.000 | 2.778 |
| 13 | pose_27 | 4.837 | 0.000 |
