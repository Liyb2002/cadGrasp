# Step4.2：whole优化

[过程图](process.png) · [最终各pose](final_result.png)

状态：pass；材料 8.150 → 54.765 cm³。
转动支撑复用 4 个pose；Juxtapose 3 个pose。

全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。

process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。

没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。

| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |
|---|---|---:|---:|---|
| 0 | step4.1_all_registered | 139135 | 11.746 |  |
| 1 | rotate_reuse_direction | 138681 | 11.322 | pose_1: host pose_1→pose_1, 方向0.995°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向0.986°, 平移0.000mm；pose_3: host pose_3→pose_3, 方向0.986°, 平移0.000mm；pose_4: host pose_4→pose_4, 方向1.474°, 平移0.000mm；pose_5: host pose_5→pose_5, 方向1.431°, 平移0.000mm；pose_6: host pose_6→pose_6, 方向0.922°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向0.053°, 平移0.000mm |
| 2 | selective_juxtapose | 95531 | 19.247 | pose_3: host pose_3→pose_7, 方向30.369°, 平移147.969mm |
| 3 | direction_after_commit | 95502 | 19.889 | pose_4: host pose_4→pose_4, 方向3.070°, 平移0.000mm |
| 4 | direction_after_commit | 95496 | 19.889 | pose_5: host pose_5→pose_5, 方向0.500°, 平移0.000mm |
| 5 | selective_juxtapose | 25799 | 61.291 | pose_1: host pose_1→pose_1, 方向9.496°, 平移0.000mm；pose_2: host pose_2→pose_2, 方向9.506°, 平移0.000mm；pose_3: host pose_7→pose_7, 方向9.974°, 平移0.000mm；pose_4: host pose_4→pose_6, 方向130.877°, 平移169.531mm；pose_5: host pose_5→pose_5, 方向7.787°, 平移0.000mm；pose_6: host pose_6→pose_6, 方向9.284°, 平移0.000mm；pose_7: host pose_7→pose_7, 方向9.974°, 平移0.000mm |
| 6 | direction_after_commit | 25776 | 61.291 | pose_4: host pose_6→pose_6, 方向4.346°, 平移0.000mm |
| 7 | direction_after_commit | 25579 | 60.389 | pose_5: host pose_5→pose_5, 方向1.965°, 平移0.000mm |
| 8 | selective_juxtapose | 0 | 66.055 | pose_1: host pose_1→pose_6, 方向7.360°, 平移148.696mm |
| 9 | feasible_volume_descent | 0 | 59.633 | pose_3: host pose_7→pose_7, 方向0.000°, 平移4.837mm |
| 10 | feasible_volume_descent | 0 | 55.046 | pose_4: host pose_6→pose_6, 方向0.000°, 平移1.209mm |
| 11 | feasible_volume_descent | 0 | 51.376 | pose_3: host pose_7→pose_7, 方向0.000°, 平移4.837mm |

完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。
