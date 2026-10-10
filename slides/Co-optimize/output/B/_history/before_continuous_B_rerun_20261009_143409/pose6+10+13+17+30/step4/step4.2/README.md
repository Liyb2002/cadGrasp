# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 44.380 → 76.536 cm³。
转动支撑复用 3 个pose；Juxtapose 2 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 96994 | 47.194 |  |
| 1 | rotate_reuse_direction | 96831 | 47.194 | pose_13: host pose_13→pose_13, 方向5.000°, 平移0.000mm |
| 2 | rotate_reuse_direction | 96608 | 46.665 | pose_6: host pose_6→pose_6, 方向2.000°, 平移0.000mm |
| 3 | selective_juxtapose | 32757 | 70.343 | pose_17: host pose_17→pose_30, 方向8.607°, 平移144.039mm |
| 4 | direction_after_commit | 11069 | 68.211 | pose_6: host pose_6→pose_6, 方向3.084°, 平移0.000mm；pose_10: host pose_10→pose_10, 方向0.898°, 平移0.000mm；pose_13: host pose_13→pose_13, 方向2.490°, 平移0.000mm；pose_17: host pose_30→pose_30, 方向2.140°, 平移0.000mm；pose_30: host pose_30→pose_30, 方向2.140°, 平移0.000mm |
| 5 | direction_after_commit | 5661 | 68.211 | pose_6: host pose_6→pose_6, 方向2.000°, 平移0.000mm |
| 6 | selective_juxtapose | 0 | 100.986 | pose_6: host pose_6→pose_30, 方向51.851°, 平移156.059mm |
| 7 | feasible_volume_descent | 0 | 91.369 | pose_17: host pose_30→pose_30, 方向0.000°, 平移19.348mm |
| 8 | feasible_volume_descent | 0 | 88.163 | pose_6: host pose_30→pose_30, 方向2.820°, 平移0.000mm；pose_10: host pose_10→pose_10, 方向0.891°, 平移0.000mm；pose_13: host pose_13→pose_13, 方向2.404°, 平移0.000mm；pose_17: host pose_30→pose_30, 方向2.147°, 平移0.000mm；pose_30: host pose_30→pose_30, 方向2.147°, 平移0.000mm |
| 9 | feasible_volume_descent | 0 | 84.957 | pose_6: host pose_30→pose_30, 方向5.000°, 平移0.000mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。

<!-- physical-operation-metrics -->

换 host 时，上表的方向／平移是支撑坐标参数差，包含坐标系变更。物体实际变化应读下面的世界中心距离和物体自身退出角；原日志和布局保持不变。

| 步 | pose | 世界物体中心移动mm | 物体自身退出角变化° |
|---|---|---:|---:|
| 1 | pose_13 | 0.000 | 5.000 |
| 2 | pose_6 | 0.000 | 2.000 |
| 3 | pose_17 | 4.837 | 107.058 |
| 4 | pose_6 | 0.000 | 3.084 |
| 4 | pose_10 | 0.000 | 0.898 |
| 4 | pose_13 | 0.000 | 2.490 |
| 4 | pose_17 | 0.000 | 2.140 |
| 4 | pose_30 | 0.000 | 2.140 |
| 5 | pose_6 | 0.000 | 2.000 |
| 6 | pose_6 | 0.000 | 51.058 |
| 7 | pose_17 | 19.348 | 0.000 |
| 8 | pose_6 | 0.000 | 2.820 |
| 8 | pose_10 | 0.000 | 0.891 |
| 8 | pose_13 | 0.000 | 2.404 |
| 8 | pose_17 | 0.000 | 2.147 |
| 8 | pose_30 | 0.000 | 2.147 |
| 9 | pose_6 | 0.000 | 5.000 |
