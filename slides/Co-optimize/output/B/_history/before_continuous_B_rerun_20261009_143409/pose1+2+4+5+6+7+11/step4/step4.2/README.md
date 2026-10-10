# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 35.591 → 73.054 cm³。
转动支撑复用 4 个pose；Juxtapose 3 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 127019 | 41.269 |  |
| 1 | rotate_reuse_direction | 126472 | 41.904 | pose_1: host pose_1→pose_1, 方向0.343°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向1.969°, 平移0.000mm；pose_4: host pose_4→pose_4, 方向1.464°, 平移0.000mm；pose_5: host pose_5→pose_5, 方向1.888°, 平移0.000mm；pose_6: host pose_6→pose_6, 方向1.890°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向1.889°, 平移0.000mm；pose_11: host pose_11→pose_11, 方向1.880°, 平移0.000mm |
| 2 | rotate_reuse_direction | 123816 | 42.539 | pose_2: host pose_2→pose_2, 方向4.914°, 平移0.000mm |
| 3 | rotate_reuse_direction | 123094 | 42.327 | pose_4: host pose_4→pose_4, 方向1.217°, 平移0.000mm |
| 4 | selective_juxtapose | 35625 | 58.212 | pose_2: host pose_2→pose_2, 方向2.000°, 平移0.000mm；pose_4: host pose_4→pose_6, 方向24.221°, 平移183.283mm；pose_11: host pose_11→pose_11, 方向5.000°, 平移0.000mm |
| 5 | selective_juxtapose | 4690 | 61.259 | pose_11: host pose_11→pose_6, 方向10.415°, 平移41.928mm |
| 6 | selective_juxtapose | 0 | 83.130 | pose_7: host pose_7→pose_6, 方向92.614°, 平移167.372mm |
| 7 | feasible_volume_descent | 0 | 79.015 | pose_11: host pose_6→pose_6, 方向0.000°, 平移4.837mm |
| 8 | feasible_volume_descent | 0 | 76.545 | pose_7: host pose_6→pose_6, 方向5.000°, 平移0.000mm |
| 9 | feasible_volume_descent | 0 | 67.492 | pose_7: host pose_6→pose_6, 方向4.145°, 平移0.000mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。
