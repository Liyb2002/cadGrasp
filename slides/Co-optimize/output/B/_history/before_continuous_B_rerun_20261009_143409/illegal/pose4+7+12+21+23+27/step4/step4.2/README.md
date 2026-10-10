# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 34.737 → 43.763 cm³。
转动支撑复用 5 个pose；Juxtapose 1 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 126861 | 42.433 |  |
| 1 | rotate_reuse_direction | 124193 | 40.211 | pose_4: host pose_4→pose_4, 方向4.750°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向4.750°, 平移0.000mm；pose_12: host pose_12→pose_12, 方向4.027°, 平移0.000mm；pose_21: host pose_21→pose_21, 方向4.750°, 平移0.000mm；pose_23: host pose_23→pose_23, 方向4.453°, 平移0.000mm；pose_27: host pose_27→pose_27, 方向3.744°, 平移0.000mm |
| 2 | rotate_reuse_direction | 124007 | 42.433 | pose_23: host pose_23→pose_23, 方向4.713°, 平移0.000mm |
| 3 | rotate_reuse_direction | 122419 | 41.798 | pose_23: host pose_23→pose_23, 方向5.000°, 平移0.000mm |
| 4 | selective_juxtapose | 0 | 49.618 | pose_23: host pose_23→pose_12, 方向24.783°, 平移68.251mm |
| 5 | feasible_volume_descent | 0 | 47.429 | pose_23: host pose_12→pose_12, 方向0.000°, 平移19.348mm |
| 6 | feasible_volume_descent | 0 | 43.781 | pose_23: host pose_12→pose_12, 方向0.000°, 平移4.837mm |
| 7 | feasible_volume_descent | 0 | 43.051 | pose_23: host pose_12→pose_12, 方向1.907°, 平移0.000mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。

<!-- physical-operation-metrics -->

换 host 时，上表的方向／平移是支撑坐标参数差，包含坐标系变更。物体实际变化应读下面的世界中心距离和物体自身退出角；原日志和布局保持不变。

| 步 | pose | 世界物体中心移动mm | 物体自身退出角变化° |
|---|---|---:|---:|
| 1 | pose_4 | 0.000 | 4.750 |
| 1 | pose_7 | 0.000 | 4.750 |
| 1 | pose_12 | 0.000 | 4.027 |
| 1 | pose_21 | 0.000 | 4.750 |
| 1 | pose_23 | 0.000 | 4.453 |
| 1 | pose_27 | 0.000 | 3.744 |
| 2 | pose_23 | 0.000 | 4.713 |
| 3 | pose_23 | 0.000 | 5.000 |
| 4 | pose_23 | 19.386 | 58.677 |
| 5 | pose_23 | 19.348 | 0.000 |
| 6 | pose_23 | 4.837 | 0.000 |
| 7 | pose_23 | 0.000 | 1.907 |
