# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 23.777 → 29.203 cm³。
转动支撑复用 7 个pose；Juxtapose 1 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 103358 | 24.021 |  |
| 1 | selective_juxtapose | 0 | 37.783 | pose_1: host pose_1→pose_10, 方向68.226°, 平移174.376mm |
| 2 | feasible_volume_descent | 0 | 34.634 | pose_1: host pose_10→pose_10, 方向4.937°, 平移0.000mm；pose_3: host pose_3→pose_3, 方向3.835°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向2.782°, 平移0.000mm；pose_10: host pose_10→pose_10, 方向3.835°, 平移0.000mm；pose_14: host pose_14→pose_14, 方向3.835°, 平移0.000mm；pose_18: host pose_18→pose_18, 方向3.500°, 平移0.000mm；pose_23: host pose_23→pose_23, 方向3.835°, 平移0.000mm；pose_27: host pose_27→pose_27, 方向1.839°, 平移0.000mm |
| 3 | feasible_volume_descent | 0 | 33.375 | pose_1: host pose_10→pose_10, 方向2.000°, 平移0.000mm |
| 4 | feasible_volume_descent | 0 | 32.745 | pose_1: host pose_10→pose_10, 方向5.000°, 平移0.000mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。
