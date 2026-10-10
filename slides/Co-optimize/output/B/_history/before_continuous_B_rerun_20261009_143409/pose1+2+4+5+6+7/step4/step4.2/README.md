# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 31.497 → 81.660 cm³。
转动支撑复用 4 个pose；Juxtapose 2 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 95244 | 38.412 |  |
| 1 | rotate_reuse_direction | 95092 | 39.152 | pose_1: host pose_1→pose_1, 方向4.745°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向4.876°, 平移0.000mm；pose_4: host pose_4→pose_4, 方向3.616°, 平移0.000mm；pose_5: host pose_5→pose_5, 方向4.745°, 平移0.000mm；pose_6: host pose_6→pose_6, 方向4.745°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向4.731°, 平移0.000mm |
| 2 | rotate_reuse_direction | 94287 | 40.211 | pose_1: host pose_1→pose_1, 方向4.747°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向4.876°, 平移0.000mm；pose_4: host pose_4→pose_4, 方向3.616°, 平移0.000mm；pose_5: host pose_5→pose_5, 方向4.747°, 平移0.000mm；pose_6: host pose_6→pose_6, 方向4.747°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向4.733°, 平移0.000mm |
| 3 | rotate_reuse_direction | 94245 | 41.480 | pose_4: host pose_4→pose_4, 方向4.137°, 平移0.000mm |
| 4 | selective_juxtapose | 3666 | 47.232 | pose_4: host pose_4→pose_6, 方向47.768°, 平移192.890mm；pose_7: host pose_7→pose_7, 方向5.000°, 平移0.000mm |
| 5 | direction_after_commit | 2828 | 47.232 | pose_2: host pose_2→pose_2, 方向4.898°, 平移0.000mm |
| 6 | direction_after_commit | 1588 | 49.654 | pose_2: host pose_2→pose_2, 方向4.904°, 平移0.000mm |
| 7 | selective_juxtapose | 44 | 83.733 | pose_7: host pose_7→pose_6, 方向90.810°, 平移181.952mm |
| 8 | direction_after_commit | 31 | 83.733 | pose_4: host pose_6→pose_6, 方向2.000°, 平移0.000mm |
| 9 | direction_after_commit | 31 | 85.593 | pose_1: host pose_1→pose_1, 方向2.907°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向2.976°, 平移0.000mm；pose_4: host pose_6→pose_6, 方向0.082°, 平移0.000mm；pose_5: host pose_5→pose_5, 方向2.907°, 平移0.000mm；pose_6: host pose_6→pose_6, 方向2.907°, 平移0.000mm；pose_7: host pose_6→pose_6, 方向3.474°, 平移0.000mm |
| 10 | selective_juxtapose | 0 | 96.789 | pose_4: host pose_6→pose_2, 方向36.818°, 平移101.763mm |
| 11 | feasible_volume_descent | 0 | 87.644 | pose_4: host pose_2→pose_2, 方向0.000°, 平移4.837mm |
| 12 | feasible_volume_descent | 0 | 85.358 | pose_2: host pose_2→pose_2, 方向5.000°, 平移0.000mm |
| 13 | feasible_volume_descent | 0 | 84.595 | pose_2: host pose_2→pose_2, 方向2.000°, 平移0.000mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。

<!-- physical-operation-metrics -->

换 host 时，上表的方向／平移是支撑坐标参数差，包含坐标系变更。物体实际变化应读下面的世界中心距离和物体自身退出角；原日志和布局保持不变。

| 步 | pose | 世界物体中心移动mm | 物体自身退出角变化° |
|---|---|---:|---:|
| 1 | pose_1 | 0.000 | 4.745 |
| 1 | pose_2 | 0.000 | 4.876 |
| 1 | pose_4 | 0.000 | 3.616 |
| 1 | pose_5 | 0.000 | 4.745 |
| 1 | pose_6 | 0.000 | 4.745 |
| 1 | pose_7 | 0.000 | 4.731 |
| 2 | pose_1 | 0.000 | 4.747 |
| 2 | pose_2 | 0.000 | 4.876 |
| 2 | pose_4 | 0.000 | 3.616 |
| 2 | pose_5 | 0.000 | 4.747 |
| 2 | pose_6 | 0.000 | 4.747 |
| 2 | pose_7 | 0.000 | 4.733 |
| 3 | pose_4 | 0.000 | 4.137 |
| 4 | pose_4 | 35.946 | 36.134 |
| 4 | pose_7 | 0.000 | 5.000 |
| 5 | pose_2 | 0.000 | 4.898 |
| 6 | pose_2 | 0.000 | 4.904 |
| 7 | pose_7 | 19.348 | 0.000 |
| 8 | pose_4 | 0.000 | 2.000 |
| 9 | pose_1 | 0.000 | 2.907 |
| 9 | pose_2 | 0.000 | 2.976 |
| 9 | pose_4 | 0.000 | 0.082 |
| 9 | pose_5 | 0.000 | 2.907 |
| 9 | pose_6 | 0.000 | 2.907 |
| 9 | pose_7 | 0.000 | 3.474 |
| 10 | pose_4 | 31.389 | 94.278 |
| 11 | pose_4 | 4.837 | 0.000 |
| 12 | pose_2 | 0.000 | 5.000 |
| 13 | pose_2 | 0.000 | 2.000 |
